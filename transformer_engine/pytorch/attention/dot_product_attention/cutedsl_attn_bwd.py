# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Opt-in CuTe DSL backend for the FP8 fused-attention backward.

``NVTE_CUTEDSL_FUSED_ATTN_BWD=1`` (default 0) routes the FP8 current-scaling attention backward to cuDNN-frontend's
``cudnn.sdpa.bwd.sdpa_bwd_fp8_gqa_d64`` when the call is one that kernel implements: e4m3 q/k/v, bf16 O (``NVTE_DPA_FP8CS_O_in_F16=1``),
e4m3 or e5m2 dO, sbhd layouts, head_dim 64, GQA, causal mask (window ``(-1, 0)`` or ``(W, 0)``), vanilla or learnable softmax,
no bias, no dropout, no padded cu_seqlens. Every other call, and any environment without the kernel, keeps cuDNN's backward.

The kernel is not bitwise equal to cuDNN's (fp32 atomic dQ accumulation; folded scale arithmetic), and the kernel must have been
compiled before CUDA graph capture (a warm-up iteration does that).
"""

import os
import warnings
from typing import Optional

import torch

_ENV = "NVTE_CUTEDSL_FUSED_ATTN_BWD"
_E4M3, _E5M2 = 7, 8  # tex.DType codes
_STATE = {"fn": None, "failed": False, "routed": 0}


def enabled() -> bool:
    """True when the environment switch is on."""
    return os.getenv(_ENV, "0") == "1"


def _kernel():
    """The cudnn-frontend entry point, or None (missing / too old package, wrong device)."""
    if _STATE["fn"] is not None or _STATE["failed"]:
        return _STATE["fn"]
    try:
        from cudnn.sdpa.bwd import sdpa_bwd_fp8_gqa_d64  # pylint: disable=import-outside-toplevel

        _STATE["fn"] = sdpa_bwd_fp8_gqa_d64
    except Exception as e:  # pylint: disable=broad-except
        _STATE["failed"] = True
        warnings.warn(f"{_ENV}=1 but cudnn.sdpa.bwd.sdpa_bwd_fp8_gqa_d64 is unavailable ({type(e).__name__}: {e}); using cuDNN")
    return _STATE["fn"]


def _is_float8(t) -> bool:
    return hasattr(t, "_fp8_dtype") and hasattr(t, "_scale_inv") and hasattr(t, "_data")


def _dtype_code(quantizer, default=_E5M2) -> int:
    try:
        return int(quantizer.dtype)
    except Exception:  # pylint: disable=broad-except
        try:
            return int(quantizer.dtype.value)
        except Exception:  # pylint: disable=broad-except
            return default


def _eligible(q, k, v, o, d_o, backend, qkv_layout, o_format, do_format, dqkv_layout, attn_bias_type, attn_mask_type,
              softmax_type, window_size, dropout, cu_seqlens_q_padded, cu_seqlens_kv_padded) -> bool:
    # pylint: disable=too-many-return-statements
    from .utils import FusedAttnBackend  # pylint: disable=import-outside-toplevel

    try:
        if FusedAttnBackend.cast(backend) != FusedAttnBackend["FP8"]:
            return False
    except Exception:  # pylint: disable=broad-except
        return False
    if not (_is_float8(q) and _is_float8(k) and _is_float8(v) and _is_float8(d_o)):
        return False
    if any(int(t._fp8_dtype) != _E4M3 for t in (q, k, v)) or int(d_o._fp8_dtype) not in (_E4M3, _E5M2):
        return False
    if _is_float8(o) or o.dtype != torch.bfloat16:
        return False  # only the O-in-F16 variant
    if qkv_layout != "sbhd_sbhd_sbhd" or dqkv_layout != "sbhd_sbhd_sbhd" or o_format != "sbhd" or do_format != "sbhd":
        return False
    if attn_bias_type != "no_bias" or dropout != 0.0 or attn_mask_type != "causal":
        return False
    if tuple(window_size) not in ((-1, 0), (-1, -1), (127, 0)) or softmax_type not in ("vanilla", "learnable"):
        return False
    if cu_seqlens_q_padded is not None or cu_seqlens_kv_padded is not None:
        return False
    s, b, h, d = q.shape
    sk, bk, hg, dk = k.shape
    return d == 64 and dk == 64 and s == sk and b == bk and h % hg == 0


def try_fused_attn_bwd(max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv, q, k, v, o, d_o, fake_dtype,
                       aux_ctx_tensors, fused_attention_backend, cu_seqlens_q_padded, cu_seqlens_kv_padded,
                       s_quantizer, dp_quantizer, dqkv_quantizer, attn_scale, dropout, fast_zero_fill, qkv_layout,
                       o_format, do_format, dqkv_layout, qkv_scale_inv_format, do_scale_inv_format, attn_bias_type,
                       attn_mask_type, softmax_type, window_size, bottom_right_diagonal, deterministic,
                       cuda_graph) -> Optional[tuple]:
    """Same arguments as ``fused_attn_bwd``. Returns ``(dq, dk, dv, d_bias, d_softmax_offset)`` or None to use cuDNN."""
    # pylint: disable=unused-argument
    if not enabled():
        return None
    if not _eligible(q, k, v, o, d_o, fused_attention_backend, qkv_layout, o_format, do_format, dqkv_layout,
                     attn_bias_type, attn_mask_type, softmax_type, window_size, dropout, cu_seqlens_q_padded,
                     cu_seqlens_kv_padded):
        return None
    fn = _kernel()
    if fn is None:
        return None
    d = q.shape[-1]
    scale = attn_scale if attn_scale is not None else 1.0 / (d**0.5)
    softmax_offset = aux_ctx_tensors[-1] if softmax_type != "vanilla" else None
    dq, dk, dv, d_softmax_offset, amax_dp = fn(
        q._data, k._data, v._data, o, d_o._data,
        q._scale_inv.reshape(1), k._scale_inv.reshape(1), v._scale_inv.reshape(1), d_o._scale_inv.reshape(1),
        s_quantizer.scale.reshape(1), dp_quantizer.scale.reshape(1), aux_ctx_tensors[0], softmax_offset,
        float(scale), int(window_size[0]), int(d_o._fp8_dtype), _dtype_code(dp_quantizer),
    )
    # cuDNN writes amax(dP) into the dP quantizer's amax buffer (the delayed-scaling update); do the same
    if getattr(dp_quantizer, "amax", None) is not None:
        dp_quantizer.amax.view(-1)[0:1].copy_(amax_dp.view(-1)[0:1])
    _STATE["routed"] += 1
    return dq, dk, dv, torch.empty(0, device=q.device), d_softmax_offset
