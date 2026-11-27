# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 (E4M3) backward GEMMs for a language-model head trained with cross-entropy.

The forward logits GEMM stays in high precision. :func:`cross_entropy_fp8_grad` computes the
cross-entropy loss of the logits and writes the gradient with respect to the logits (dlogits) as
E4M3 with one fp32 dequantization scale per token row, ``(1 - p_y) * scale / 448``, using
unbiased stochastic rounding (Philox from the default CUDA generator, CUDA-graph safe).
:func:`lm_head_fp8_backward` then runs the dgrad and wgrad GEMMs in FP8 with per-tensor fp32
GEMM scales:

  dgrad  dX = diag(s_row) . (Q_D @ Q_W) * s_W * dgrad_scale    (row scale applied on M)
  wgrad  dW = Q_D^T @ Q_X' * s_X',  X' = diag(s_row) . X        (row scale folded into X')

W is quantized per tensor with round-to-nearest; X' per tensor with stochastic rounding (its rows
span the (1 - p_y) range, so round-to-nearest would flush the rows of confident tokens).
Rounding the gradient stochastically per row keeps it unbiased, unlike a per-tensor
round-to-nearest FP8 gradient, whose consistent-sign bias can make training drift.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import triton

import transformer_engine_torch as tex
from transformer_engine.common.triton.lm_head_fp8_backward import (
    amax_kernel,
    cross_entropy_fp8_grad_kernel,
    quantize_fp8_kernel,
    row_scale_kernel,
    row_scaled_amax_kernel,
    row_scaled_quantize_fp8_kernel,
)
from ..cpp_extensions import general_gemm
from ..tensor.float8_tensor import Float8Tensor

__all__ = ["cross_entropy_fp8_grad", "lm_head_fp8_backward"]

# Philox offsets reserved per kernel launch (one Philox call per counter subsequence)
_RNG_ELTS_PER_THREAD = 4
_ROWS_PER_PROGRAM = 8

_unit_scales = {}


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _unit_scale(device: torch.device) -> torch.Tensor:
    """fp32 tensor holding 1, cached per device outside of CUDA graph capture."""
    scale = _unit_scales.get(device)
    if scale is None:
        scale = torch.ones(1, dtype=torch.float32, device=device)
        if not torch.cuda.is_current_stream_capturing():
            _unit_scales[device] = scale
    return scale


def _as_float8(data: torch.Tensor, scale_inv: torch.Tensor) -> Float8Tensor:
    return Float8Tensor(
        shape=data.shape,
        dtype=torch.bfloat16,
        data=data.view(torch.uint8),
        fp8_scale_inv=scale_inv,
        fp8_dtype=tex.DType.kFloat8E4M3,
        requires_grad=False,
    )


def cross_entropy_fp8_grad(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss: torch.Tensor,
    grad_scale: torch.Tensor,
    batch_size: int,
    ignore_idx: int = -100,
    block_size: int = 32768,
    grad_block_size: int = 32768,
    num_warps: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy loss and its E4M3 logits gradient with per-row dequantization scales.

    Parameters
    ----------
    logits : torch.Tensor
        [T, V] logits with unit stride in V. Row ``r`` is token ``(s, b) = (r // batch_size,
        r % batch_size)`` of a sequence-first [s, b] layout.
    target : torch.Tensor
        int64 target indices, [batch_size, T // batch_size] (any strides) or [T] when
        ``batch_size`` is 1.
    loss : torch.Tensor
        fp32 output for the per-token loss, shaped like ``target``.
    grad_scale : torch.Tensor
        fp32 tensor whose first element multiplies the gradient (e.g. 1 / number of tokens).
    batch_size : int
        Number of sequences ``b``.
    ignore_idx : int
        Target value of tokens without loss and gradient.
    block_size, grad_block_size, num_warps : int
        Kernel tiling of the statistics and gradient passes.

    Returns
    -------
    (dlogits, row_scale): [T, V] ``torch.float8_e4m3fn`` gradient and its [T] fp32 dequantization
    scales; ``row_scale[r] * dlogits[r]`` is the gradient of ``grad_scale * sum(loss)``.
    """
    T, V = logits.shape
    if logits.stride(1) != 1:
        raise ValueError("logits must have unit stride in the vocabulary dimension")
    if target.dtype != torch.int64:
        raise ValueError("target must be int64")
    if target.dim() == 2:
        if target.shape[0] != batch_size or target.numel() != T:
            raise ValueError("target must be [batch_size, T // batch_size]")
        target_strides = (target.stride(0), target.stride(1))
    else:
        if batch_size != 1 or target.numel() != T:
            raise ValueError("a 1D target needs batch_size 1 and T elements")
        target_strides = (0, target.stride(0))
    loss_strides = (loss.stride(0), loss.stride(1)) if loss.dim() == 2 else (0, loss.stride(0))
    block_size = min(block_size, triton.next_power_of_2(V))
    grad_block_size = min(grad_block_size, triton.next_power_of_2(V))
    dlogits = torch.empty((T, V), dtype=torch.float8_e4m3fn, device=logits.device)
    row_scale = torch.empty((T,), dtype=torch.float32, device=logits.device)
    rng_state = tex.philox_rng_state(_RNG_ELTS_PER_THREAD)
    cross_entropy_fp8_grad_kernel[(T,)](
        logits,
        logits.stride(0),
        target,
        *target_strides,
        loss,
        *loss_strides,
        grad_scale,
        dlogits,
        dlogits.stride(0),
        row_scale,
        rng_state,
        _rank(),
        batch_size,
        V,
        ignore_idx,
        BLOCK_SIZE=block_size,
        GRAD_BLOCK=grad_block_size,
        num_warps=num_warps,
    )
    return dlogits, row_scale


def _quantize_per_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """E4M3 with round-to-nearest and the current per-tensor scale; returns (data, scale_inv)."""
    n = x.numel()
    buf = torch.zeros(2, dtype=torch.float32, device=x.device)
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    grid = (min(triton.cdiv(n, 8192), 4096),)
    amax_kernel[grid](x, n, buf[0:1], BLOCK=8192, num_warps=8)
    quantize_fp8_kernel[grid](x, n, buf[0:1], q, buf[1:2], BLOCK=8192, num_warps=8)
    return q, buf[1:2]


def _quantize_row_scaled(
    x: torch.Tensor, row_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """E4M3 of diag(row_scale) . x with stochastic rounding and the current per-tensor scale."""
    T, H = x.shape
    buf = torch.zeros(2, dtype=torch.float32, device=x.device)
    q = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=x.device)
    grid = (triton.cdiv(T, _ROWS_PER_PROGRAM),)
    block_h = min(4096, triton.next_power_of_2(H))
    row_scaled_amax_kernel[grid](
        x, row_scale, T, H, buf[0:1], ROWS=_ROWS_PER_PROGRAM, BLOCK_H=block_h, num_warps=8
    )
    rng_state = tex.philox_rng_state(_RNG_ELTS_PER_THREAD)
    row_scaled_quantize_fp8_kernel[grid](
        x,
        row_scale,
        T,
        H,
        buf[0:1],
        q,
        buf[1:2],
        rng_state,
        _rank(),
        ROWS=_ROWS_PER_PROGRAM,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return q, buf[1:2]


def _scale_rows(
    y: torch.Tensor, row_scale: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    T, H = y.shape
    out = torch.empty((T, H), dtype=out_dtype, device=y.device)
    block_h = min(4096, triton.next_power_of_2(H))
    row_scale_kernel[(triton.cdiv(T, _ROWS_PER_PROGRAM),)](
        y, row_scale, scale, out, T, H, ROWS=_ROWS_PER_PROGRAM, BLOCK_H=block_h, num_warps=8
    )
    return out


def lm_head_fp8_backward(
    inp: torch.Tensor,
    weight: torch.Tensor,
    dlogits: torch.Tensor,
    row_scale: torch.Tensor,
    dgrad_scale: Optional[torch.Tensor] = None,
    main_grad: Optional[torch.Tensor] = None,
    accumulate: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """FP8 dgrad and wgrad of ``logits = inp @ weight^T`` from the output of
    :func:`cross_entropy_fp8_grad`.

    Parameters
    ----------
    inp : torch.Tensor
        [T, H] contiguous forward input of the LM head. Any factor of the loss gradient that
        should reach the weight gradient must already be applied to it.
    weight : torch.Tensor
        [V, H] contiguous LM-head weight.
    dlogits, row_scale : torch.Tensor
        [T, V] E4M3 gradient and its [T] fp32 dequantization scales.
    dgrad_scale : torch.Tensor, optional
        fp32 tensor whose first element multiplies the input gradient (default 1).
    main_grad : torch.Tensor, optional
        [V, H] buffer that receives the weight gradient (accumulated into it unless
        ``accumulate`` is False). Without it the weight gradient is returned in weight's dtype.

    Returns
    -------
    (grad_input, grad_weight): [T, H] input gradient in inp's dtype, and the weight gradient, or
    None when it was written into ``main_grad``.
    """
    if not (inp.is_contiguous() and weight.is_contiguous()):
        raise ValueError("inp and weight must be contiguous")
    if dgrad_scale is None:
        dgrad_scale = _unit_scale(inp.device)
    dlogits_fp8 = _as_float8(dlogits, _unit_scale(dlogits.device))

    weight_data, weight_scale_inv = _quantize_per_tensor(weight)
    dgrad = general_gemm(
        _as_float8(weight_data, weight_scale_inv),
        dlogits_fp8,
        out_dtype=torch.float32,
        layout="NN",
        use_split_accumulator=True,
    )[0]
    del weight_data
    grad_input = _scale_rows(dgrad, row_scale, dgrad_scale, inp.dtype)
    del dgrad

    inp_data, inp_scale_inv = _quantize_row_scaled(inp, row_scale)
    inp_fp8 = _as_float8(inp_data, inp_scale_inv)
    if main_grad is not None:
        general_gemm(
            inp_fp8,
            dlogits_fp8,
            out_dtype=main_grad.dtype,
            layout="NT",
            out=main_grad,
            accumulate=accumulate,
            use_split_accumulator=True,
        )
        return grad_input, None
    grad_weight = general_gemm(
        inp_fp8, dlogits_fp8, out_dtype=weight.dtype, layout="NT", use_split_accumulator=True
    )[0]
    return grad_input, grad_weight
