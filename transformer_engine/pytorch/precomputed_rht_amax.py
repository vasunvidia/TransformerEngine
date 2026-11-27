# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVFP4 RHT amaxes emitted by the producer of a tensor, for the cast that consumes it.

An NVFP4 cast with random Hadamard transform (RHT) and post-RHT amax first runs
``nvte_hadamard_transform_amax`` over its input: one more read of the tensor. The producers here
write a BF16 tensor and its two amaxes (``max|x|`` and the amax of the RHT of ``x^T``) in one
pass and register the amaxes for that tensor; :meth:`NVFP4Quantizer.quantize_impl` then casts it
with ``nvfp4_quantize_with_amax``. The amaxes are bitwise those of ``nvte_hadamard_transform_amax``,
so every cast output is unchanged.

* :func:`add_with_rht_amax`: ``a + b``, e.g. the sum of the two gradients of a residual-stream
  tensor (see ``NVTE_FUSED_ADD_RMSNORM_NVFP4_GRAD_RHT_AMAX`` in :mod:`.fused_add_rmsnorm`).
* :func:`fused_qkv_rope_backward_with_rht_amax`: RoPE backward of the Q, K and V gradients into
  the gradient of the fused QKV projection output.
* :func:`nvfp4_quantize_rowwise_with_amax`: row-wise cast of a tensor that is cast again
  column-wise later (e.g. a linear input saved in high precision for the wgrad GEMM).

An amax is registered for one tensor object (address, size and version counter) and taken by the
first eligible cast of that tensor; an entry that is not taken is dropped after a few newer ones.
Registration only happens when a caller uses one of these producers, so nothing changes otherwise.
The registry is process-global: producer and consumer must run on the same thread.
"""

from typing import Dict, List, Optional, Tuple

import torch

import transformer_engine_torch as tex

__all__ = [
    "add_with_rht_amax",
    "fused_qkv_rope_backward_with_rht_amax",
    "nvfp4_quantize_rowwise_with_amax",
    "take_rht_amax",
]

# Entries of tensors cast right after they are produced (gradients) and of tensors cast column-wise
# in the backward pass (one per layer while the forward pass runs ahead)
_GRAD_CAPACITY = 2
_SAVED_CAPACITY = 64
_ROWS_MULTIPLE = 64
_COLS_MULTIPLE = 128


class _Entry:
    """Amaxes registered for one tensor."""

    __slots__ = ("tensor", "numel", "version", "amax", "sign_mask")

    def __init__(self, tensor: torch.Tensor, amax: torch.Tensor, sign_mask: int):
        # Holding the tensor keeps its storage alive, so no other tensor can take its address
        self.tensor = tensor
        self.numel = tensor.numel()
        self.version = tensor._version
        self.amax = amax
        self.sign_mask = sign_mask


_grad_entries: Dict[int, _Entry] = {}
_saved_entries: Dict[int, _Entry] = {}


def _register(entries: Dict[int, _Entry], capacity: int, tensor, amax, sign_mask: int) -> None:
    while len(entries) >= capacity:
        entries.pop(next(iter(entries)))
    entries[tensor.data_ptr()] = _Entry(tensor, amax, sign_mask)


def _is_plain_bf16(t) -> bool:
    return (
        type(t) is torch.Tensor  # pylint: disable=unidiomatic-typecheck
        and t.dtype == torch.bfloat16
        and t.is_cuda
        and t.is_contiguous()
        and t.dim() >= 2
        and t.data_ptr() % 16 == 0
    )


def _sign_mask(device: torch.device) -> int:
    # pylint: disable=import-outside-toplevel
    from .fused_add_rmsnorm import _rht_sign_mask

    return _rht_sign_mask(device)


def _eligible_quantizer(quantizer, sign_mask: int) -> bool:
    # pylint: disable=import-outside-toplevel
    from .fused_add_rmsnorm import _is_eligible_quantizer

    return _is_eligible_quantizer(quantizer, sign_mask)


def take_rht_amax(tensor: torch.Tensor, quantizer) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(rowwise_amax, columnwise_amax)`` a producer registered for ``tensor``, or ``None``.

    The entry is removed. It is returned only if ``tensor`` is unchanged since the producer wrote
    it and ``quantizer`` is an NVFP4 quantizer with RHT and post-RHT amax whose column-wise RHT
    uses the sign mask the amaxes were computed with.
    """
    if not (_grad_entries or _saved_entries) or not isinstance(tensor, torch.Tensor):
        return None
    ptr = tensor.data_ptr()
    entry = _grad_entries.pop(ptr, None)
    if entry is None:
        entry = _saved_entries.pop(ptr, None)
    if entry is None:
        return None
    if (
        tensor.numel() != entry.numel
        or tensor._version != entry.version
        or not _is_plain_bf16(tensor)
        or not _eligible_quantizer(quantizer, entry.sign_mask)
    ):
        return None
    return entry.amax[0:1], entry.amax[1:2]


def _addable(a, b) -> bool:
    if not (_is_plain_bf16(a) and _is_plain_bf16(b)) or a.shape != b.shape or a.device != b.device:
        return False
    n = a.shape[-1]
    return n % _COLS_MULTIPLE == 0 and a.numel() > 0 and (a.numel() // n) % _ROWS_MULTIPLE == 0


def add_with_rht_amax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a + b``; for supported BF16 inputs the NVFP4 RHT amaxes of the sum are registered for
    its cast. Supports contiguous BF16 tensors of the same shape whose flattened shape
    ``[M, N]`` has ``M`` a multiple of 64 and ``N`` a multiple of 128, on devices of compute
    capability 10.0 or newer; other inputs return ``a + b`` with nothing registered."""
    if not _addable(a, b) or torch.cuda.get_device_capability(a.device)[0] < 10:
        return a + b
    sign_mask = _sign_mask(a.device)
    out, amax = tex.add_hadamard_transform_amax(a, b, sign_mask)
    _register(_grad_entries, _GRAD_CAPACITY, out, amax, sign_mask)
    return out


def fused_qkv_rope_backward_with_rht_amax(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    split: List[int],
) -> Optional[torch.Tensor]:
    """RoPE backward of the Q, K and V gradients into the gradient of the fused QKV output, with
    the NVFP4 RHT amaxes of the result registered for its cast.

    Equals ``tex.fused_qkv_rope_backward(grad_q, grad_k, grad_v, freqs, freqs, split, sbhd, ...)``
    (non-interleaved, no context parallelism) with ``cos``/``sin`` the FP32 tables of ``freqs``
    of shape ``[>= s, 1, 1, d]``. Returns the BF16 gradient ``[s, b, h_kv, sum(split)]``, or
    ``None`` (nothing done) unless the gradients are BF16 ``[s, b, h, 128]`` tensors with
    ``s * b`` a multiple of 64, ``b`` dividing 64 and ``split = [h_q / h_kv * 128, 128, 128]``,
    on a device of compute capability 10.0 or newer.
    """
    if not all(
        type(t) is torch.Tensor  # pylint: disable=unidiomatic-typecheck
        for t in (grad_q, grad_k, grad_v, cos, sin)
    ):
        return None
    if not (grad_q.dim() == grad_k.dim() == grad_v.dim() == 4 and len(split) == 3):
        return None
    if not (grad_q.dtype == grad_k.dtype == grad_v.dtype == torch.bfloat16 and grad_q.is_cuda):
        return None
    s, b, g, d = grad_k.shape
    if d != 128 or split[1] != 128 or split[2] != 128 or grad_v.shape != grad_k.shape:
        return None
    if grad_q.shape[:2] != (s, b) or grad_q.shape[3] != 128 or g == 0 or grad_q.shape[2] % g:
        return None
    if split[0] != grad_q.shape[2] // g * 128:
        return None
    if b == 0 or (s * b) % 64 != 0 or 64 % b != 0:
        return None
    if not (
        cos.dtype == sin.dtype == torch.float32 and cos.is_contiguous() and sin.is_contiguous()
    ):
        return None
    if cos.dim() != 4 or cos.shape != sin.shape or cos.shape[1:] != (1, 1, 128) or cos.shape[0] < s:
        return None
    if cos.data_ptr() % 16 or sin.data_ptr() % 16:
        return None
    if torch.cuda.get_device_capability(grad_q.device)[0] < 10:
        return None
    sign_mask = _sign_mask(grad_q.device)
    out, amax = tex.fused_qkv_rope_backward_hadamard_transform_amax(
        grad_q, grad_k, grad_v, cos, sin, sign_mask
    )
    _register(_grad_entries, _GRAD_CAPACITY, out, amax, sign_mask)
    return out


def _rowwise_cast_supported(tensor, quantizer) -> bool:
    # pylint: disable=import-outside-toplevel
    from .tensor.nvfp4_tensor import NVFP4Quantizer

    if type(quantizer) is not NVFP4Quantizer:  # pylint: disable=unidiomatic-typecheck
        return False
    if not quantizer.rowwise_usage or quantizer.columnwise_usage or quantizer.stochastic_rounding:
        return False
    if not _eligible_quantizer(quantizer, int(quantizer.rht_matrix_random_sign_mask_t)):
        return False
    if not _is_plain_bf16(tensor):
        return False
    n = tensor.shape[-1]
    m = tensor.numel() // n if n else 0
    tiles = n // 128
    return m > 0 and m % 128 == 0 and n % 128 == 0 and tiles & (tiles - 1) == 0


def nvfp4_quantize_rowwise_with_amax(
    tensor: torch.Tensor,
    quantizer,
    rowwise_amax: torch.Tensor,
    columnwise_amax: torch.Tensor,
):
    """``tex.nvfp4_quantize_with_amax(tensor, quantizer, rowwise_amax, columnwise_amax)`` for a
    row-wise-only cast whose input is cast again column-wise later (with RHT, e.g. for a wgrad
    GEMM in the backward pass).

    For supported inputs the cast also computes the two NVFP4 RHT amaxes of ``tensor`` and, when
    autograd is recording, registers them for that later column-wise cast. Supports row-wise-only
    NVFP4 quantizers with RHT and post-RHT amax, round-to-nearest (no stochastic rounding),
    1D blocks and no amax reduction, and contiguous BF16 tensors with both flattened dimensions
    multiples of 128 and the number of columns divided by 128 a power of two, on devices of
    compute capability 10.0 or newer. Other calls run ``tex.nvfp4_quantize_with_amax``.
    """
    if (
        not torch.is_grad_enabled()
        or not _rowwise_cast_supported(tensor, quantizer)
        or not isinstance(rowwise_amax, torch.Tensor)
        or rowwise_amax.dtype != torch.float32
        or not rowwise_amax.is_cuda
        or rowwise_amax.numel() != 1
        or torch.cuda.get_device_capability(tensor.device)[0] < 10
    ):
        return tex.nvfp4_quantize_with_amax(tensor, quantizer, rowwise_amax, columnwise_amax)
    out, amax = tex.nvfp4_quantize_rowwise_with_amax_hadamard_transform_amax(
        tensor, quantizer, rowwise_amax
    )
    sign_mask = int(quantizer.rht_matrix_random_sign_mask_t)
    _register(_saved_entries, _SAVED_CAPACITY, tensor, amax, sign_mask)
    return out


def _clear() -> None:
    """Drop every registered entry (tests)."""
    _grad_entries.clear()
    _saved_entries.clear()
