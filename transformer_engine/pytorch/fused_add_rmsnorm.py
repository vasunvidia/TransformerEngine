# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Residual add fused with the RMSNorm and NVFP4 amax computation of the following layer.

In a pre-norm Transformer, the residual add ``s = residual + x`` that ends a sublayer feeds the
RMSNorm of the next linear layer, whose NVFP4 input quantizer (with random Hadamard transform and
post-RHT amax) then needs two amax passes over the RMSNorm output before it can cast it.
:func:`fused_add_rmsnorm_nvfp4` computes ``s``, the RMSNorm output, ``rsigma`` and both amaxes in
one kernel and keeps them; when the next RMSNorm of :class:`LayerNormLinear`,
:class:`LayerNormMLP` or :class:`ops.RMSNorm` runs on ``s`` with the same weight and epsilon and an
eligible NVFP4 quantizer, it takes the precomputed result and only runs the cast with the
precomputed amaxes (``nvfp4_quantize_with_amax``). Every output is bitwise equal to the unfused
computation. Any other consumer of ``s`` runs the unfused RMSNorm.

Enabled with ``NVTE_FUSED_ADD_RMSNORM_NVFP4=1`` (default off).

With ``NVTE_FUSED_ADD_RMSNORM_NVFP4_GRAD_RHT_AMAX=1`` (default off) the gradient of ``s`` also
comes with its NVFP4 RHT amaxes. ``s`` has two consumers, its RMSNorm and the next residual add,
whose gradients autograd adds before they reach the linear layer that produced ``x`` (which casts
that sum with RHT). The add then returns, besides ``s``, a second output sharing ``s``'s storage
that the next :func:`fused_add_rmsnorm_nvfp4` takes as its residual; the backward receives the two
gradients apart and adds them with :func:`.precomputed_rht_amax.add_with_rht_amax`, which also
registers the amaxes of the sum for its cast. The sum is bitwise autograd's. Supports BF16 inputs with hidden
size 4096 and a number of tokens divisible by 16 on devices of compute capability 10.0 or newer
that allow about 193 KiB of shared memory per block, and only when the RMSNorm forward runs on
Transformer Engine's kernels (``NVTE_NORM_FWD_USE_CUDNN`` unset or 0).
The precomputed result is held in process-global state between the add and its RMSNorm, so the
two calls must run on the same thread with no other fused add in between.
"""

import os
import struct
from typing import List, Optional, Tuple

import torch

import transformer_engine_torch as tex
from .constants import TE_DType
from .quantized_tensor import QuantizedTensorStorage
from .tensor.nvfp4_tensor import NVFP4Quantizer, get_random_sign_mask_for_rht

__all__ = ["fused_add_rmsnorm_nvfp4", "is_fused_add_rmsnorm_nvfp4_enabled"]

_HIDDEN_SIZE = 4096
_ROWS_MULTIPLE = 16
# Dynamic shared memory of the fused kernel: Layout::kBytes in rmsnorm_fwd_add_rht_amax.cu
_KERNEL_SMEM_BYTES = 197320


class _Precomputed:
    """RMSNorm results of a fused add, kept until the RMSNorm on its sum runs."""

    __slots__ = (
        "sum",
        "ptr",
        "numel",
        "version",
        "weight",
        "eps",
        "sign_mask",
        "z",
        "rsigma",
        "amax",
    )

    def __init__(self, sum_, weight, eps, sign_mask, z, rsigma, amax):
        # Holding the sum keeps its storage alive, so no other tensor can take its address
        self.sum = sum_
        self.ptr = sum_.data_ptr()
        self.numel = sum_.numel()
        self.version = sum_._version
        self.weight = weight
        self.eps = eps
        self.sign_mask = sign_mask
        self.z = z
        self.rsigma = rsigma
        self.amax = amax


_pending: Optional[_Precomputed] = None
# (sum, residual output) of the last add with a separate output for the residual consumer
_last_split: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
_sign_masks = {}
_supported_devices = {}


def is_fused_add_rmsnorm_nvfp4_enabled() -> bool:
    """Whether the fused residual add + RMSNorm + NVFP4 amax path is enabled."""
    return os.getenv("NVTE_FUSED_ADD_RMSNORM_NVFP4", "0") == "1"


def is_fused_add_rmsnorm_nvfp4_grad_rht_amax_enabled() -> bool:
    """Whether the fused residual add also emits the NVFP4 RHT amaxes of its gradient."""
    return os.getenv("NVTE_FUSED_ADD_RMSNORM_NVFP4_GRAD_RHT_AMAX", "0") == "1"


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _device_supported(device: torch.device) -> bool:
    idx = _device_index(device)
    supported = _supported_devices.get(idx)
    if supported is None:
        props = torch.cuda.get_device_properties(idx)
        supported = (props.major, props.minor) >= (10, 0) and (
            props.shared_memory_per_block_optin >= _KERNEL_SMEM_BYTES
        )
        _supported_devices[idx] = supported
    return supported


def _norm_fwd_uses_cudnn() -> bool:
    """Whether the unfused RMSNorm forward may run on cuDNN (``NVTE_NORM_FWD_USE_CUDNN``).

    Transformer Engine reads the variable as a boolean that is off when unset, empty or 0. Any
    other value selects cuDNN (or is rejected by the unfused path), whose results the fused kernel
    does not reproduce.
    """
    return os.getenv("NVTE_NORM_FWD_USE_CUDNN", "") not in ("", "0")


def _rht_sign_mask(device: torch.device) -> int:
    """Column-wise RHT sign mask that the default NVFP4 quantizer uses on this device."""
    idx = _device_index(device)
    mask = _sign_masks.get(idx)
    if mask is None:
        mask = int(get_random_sign_mask_for_rht(True, idx))
        _sign_masks[idx] = mask
    return mask


def _is_plain_tensor(t) -> bool:
    if type(t) is torch.Tensor:
        return True
    return (
        isinstance(t, torch.Tensor)
        and not isinstance(t, QuantizedTensorStorage)
        and type(t).__torch_dispatch__ is torch.Tensor.__torch_dispatch__
    )


def _is_supported_bf16(t: torch.Tensor) -> bool:
    return (
        _is_plain_tensor(t)
        and t.dtype == torch.bfloat16
        and t.is_cuda
        and t.is_contiguous()
        and t.data_ptr() % 16 == 0
    )


def _supports(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> bool:
    if not (_is_supported_bf16(x) and _is_supported_bf16(residual) and _is_supported_bf16(weight)):
        return False
    if x.device != residual.device or weight.device != residual.device:
        return False
    if x.shape != residual.shape or x.dim() < 2 or x.shape[-1] != _HIDDEN_SIZE:
        return False
    if weight.numel() != _HIDDEN_SIZE:
        return False
    rows = x.numel() // _HIDDEN_SIZE
    if rows == 0 or rows % _ROWS_MULTIPLE != 0:
        return False
    return _device_supported(x.device)


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _is_eligible_quantizer(quantizer, sign_mask: int) -> bool:
    """NVFP4 quantizers whose amaxes the fused kernel computes: RHT with post-RHT amax, 1D
    blocks, E4M3 scales with the default bound, no amax reduction."""
    return (
        type(quantizer) is NVFP4Quantizer  # pylint: disable=unidiomatic-typecheck
        and quantizer.with_rht
        and quantizer.with_post_rht_amax
        and not quantizer.with_2d_quantization
        and not quantizer.row_scaled_nvfp4
        and not quantizer.nvfp4_use_4over6
        and not quantizer.with_amax_reduction
        and not quantizer.disable_second_level_scale
        and quantizer.nvfp4_e4m3_max in (0, 448)
        and quantizer.scale_dtype == tex.DType.kFloat8E4M3
        and int(quantizer.rht_matrix_random_sign_mask_t) == sign_mask
        and os.getenv("NVTE_USE_FAST_MATH", "0") in ("", "0")
    )


class _FusedAdd(torch.autograd.Function):
    """``residual + x`` with the RMSNorm results of the sum as non-differentiable outputs."""

    @staticmethod
    def forward(ctx, x, residual, weight, eps, sign_mask):
        # pylint: disable=missing-function-docstring
        sum_, z, rsigma, amax = tex.rmsnorm_fwd_add_rht_amax(x, residual, weight, eps, sign_mask, 0)
        ctx.mark_non_differentiable(z, rsigma, amax)
        # Only the sum has a gradient: do not zero-fill gradients of the other outputs
        ctx.set_materialize_grads(False)
        return sum_, z, rsigma, amax

    @staticmethod
    def backward(ctx, grad_sum, *unused_grads):
        # pylint: disable=missing-function-docstring
        return grad_sum, grad_sum, None, None, None


class _FusedAddSplitResidual(torch.autograd.Function):
    """:class:`_FusedAdd` with a second output, sharing the sum's storage, for the residual
    consumer, so that the backward gets the two gradients of the sum apart and adds them with the
    NVFP4 RHT amaxes of the result."""

    @staticmethod
    def forward(ctx, x, residual, weight, eps, sign_mask):
        # pylint: disable=missing-function-docstring
        sum_, z, rsigma, amax = tex.rmsnorm_fwd_add_rht_amax(x, residual, weight, eps, sign_mask, 0)
        residual_out = torch.empty(0, dtype=sum_.dtype, device=sum_.device).set_(
            sum_.untyped_storage(), sum_.storage_offset(), sum_.size(), sum_.stride()
        )
        ctx.mark_non_differentiable(z, rsigma, amax)
        # A consumer that is not used gets None, not zeros
        ctx.set_materialize_grads(False)
        return sum_, residual_out, z, rsigma, amax

    @staticmethod
    def backward(ctx, grad_sum, grad_residual_out, *unused_grads):
        # pylint: disable=missing-function-docstring
        if grad_sum is None or grad_residual_out is None:
            grad = grad_sum if grad_residual_out is None else grad_residual_out
        else:
            # pylint: disable=import-outside-toplevel
            from .precomputed_rht_amax import add_with_rht_amax

            grad = add_with_rht_amax(grad_sum, grad_residual_out)
        return grad, grad, None, None, None


def fused_add_rmsnorm_nvfp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Optional[torch.Tensor]:
    """Return ``residual + x`` and precompute the RMSNorm and NVFP4 amaxes of the sum.

    The next RMSNorm that runs on the returned tensor with ``weight`` and ``eps`` inside
    :class:`LayerNormLinear`, :class:`LayerNormMLP` or :class:`ops.RMSNorm`, with an NVFP4
    quantizer that uses the random Hadamard transform and post-RHT amax, uses the precomputed
    result and only casts.

    Returns ``None`` without doing anything when the path is disabled
    (``NVTE_FUSED_ADD_RMSNORM_NVFP4``), the RMSNorm forward uses cuDNN
    (``NVTE_NORM_FWD_USE_CUDNN``) or the inputs or device are not supported; the caller then
    computes the add itself.

    Parameters
    ----------
    x : torch.Tensor
        Sublayer output, BF16 of shape ``[..., 4096]``.
    residual : torch.Tensor
        Residual input of the same shape and dtype.
    weight : torch.Tensor
        RMSNorm weight (gamma) of the consumer, BF16 of shape ``[4096]``.
    eps : float
        RMSNorm epsilon of the consumer.
    """
    global _pending, _last_split
    if (
        not is_fused_add_rmsnorm_nvfp4_enabled()
        or _norm_fwd_uses_cudnn()
        or not _supports(x, residual, weight)
    ):
        return None
    weight = weight.detach()
    sign_mask = _rht_sign_mask(x.device)
    # The residual is the sum of the previous add: take its output for the residual consumer
    last_split, _last_split = _last_split, None
    if last_split is not None and residual is last_split[0]:
        residual = last_split[1]
    if is_fused_add_rmsnorm_nvfp4_grad_rht_amax_enabled() and torch.is_grad_enabled():
        sum_, residual_out, z, rsigma, amax = _FusedAddSplitResidual.apply(
            x, residual, weight, float(eps), sign_mask
        )
        _last_split = (sum_, residual_out)
    else:
        sum_, z, rsigma, amax = _FusedAdd.apply(x, residual, weight, float(eps), sign_mask)
    _pending = _Precomputed(sum_, weight, float(eps), sign_mask, z, rsigma, amax)
    return sum_


def take_precomputed_rmsnorm(
    inp: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    ln_out: Optional[torch.Tensor],
    quantizer,
    otype,
    zero_centered_gamma: bool,
) -> Optional[List]:
    """RMSNorm forward output ``[out, None, rsigma]`` from a preceding fused add, or ``None``.

    Arguments are those of ``tex.rmsnorm_fwd``. Returns ``None`` unless ``inp`` is the sum of the
    last :func:`fused_add_rmsnorm_nvfp4` call and the call matches the precomputed RMSNorm; the
    caller then runs the RMSNorm itself.
    """
    global _pending
    p = _pending
    if p is None or not _is_plain_tensor(inp) or inp.data_ptr() != p.ptr:
        return None
    _pending = None  # this sum reaches its RMSNorm now, served or not
    if (
        inp.numel() != p.numel
        or inp._version != p.version
        or inp.dtype != torch.bfloat16
        or not inp.is_contiguous()
        or ln_out is not None
        or zero_centered_gamma
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.numel() != _HIDDEN_SIZE
        or weight.data_ptr() != p.weight.data_ptr()
        or _float32(eps) != _float32(p.eps)
        or TE_DType.get(otype, otype) != tex.DType.kBFloat16
        or not _is_eligible_quantizer(quantizer, p.sign_mask)
    ):
        return None
    # The cast of the unfused path, with the amaxes it would compute
    out = tex.nvfp4_quantize_with_amax(p.z.view(inp.shape), quantizer, p.amax[0:1], p.amax[1:2])
    return [out, None, p.rsigma]
