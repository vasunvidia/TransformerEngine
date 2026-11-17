# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Precomputed per-group amaxes for grouped NVFP4 quantization with random Hadamard transform.

A grouped NVFP4 quantize with RHT (``tex.group_quantize`` with an NVFP4 quantizer that has
``with_rht`` and ``with_post_rht_amax``) first runs a separate amax pass over the whole BF16 input
(``nvte_group_hadamard_transform_amax_graph_safe``) for, per group,

* the pre-RHT row-wise amax ``max |x|`` and
* the post-RHT column-wise amax ``max |bf16(R^T x_b)|`` over the 16-row blocks ``x_b``
  (``R`` = the quantizer's ``rht_matrix``),

and then the RHT + cast kernel. A producer that writes the tensor (for example the kernel that
gathers the routed tokens of a mixture-of-experts layer into the grouped expert input) can compute
both amaxes while the rows are in registers and hand them over here with
:func:`set_precomputed_group_amax`. TE's grouped quantize sites that go through
:func:`group_quantize` (the fused grouped MLP's FC1 input and FC2 grad output, and the grouped-tensor
``GroupedLinear`` module's input and grad output) then call ``tex.nvfp4_group_quantize_with_amax``
with them and skip the amax pass. The result is bitwise TE's own path when the producer's amaxes
are bitwise TE's (the amax kernel's arithmetic: fp32 accumulation of the 16x16 bf16 RHT product,
rounded to bf16 before the abs max).

The handoff is keyed by the tensor's storage (data pointer, shape, dtype, device): views and
reshapes of a contiguous tensor that keep its shape (e.g. ``reshape(-1, hidden)`` of a 2-D
tensor) find it. An entry holds a reference to the tensor, so its address cannot be reused while
the entry waits, and it is consumed by the first eligible quantize of that tensor. Entries are
only used when the quantizer applies the same RHT matrix and computes these amaxes itself
(NVFP4, RHT with post-RHT amax, 1D scaling, no row-scaled / 4over6 / disabled second-level
scale); any other quantize of the tensor drops the entry and runs TE's own amax pass. At most
:data:`MAX_PENDING` entries wait; older ones are evicted.

Graph safety: the lookup is host-side Python on tensor addresses, so under CUDA graph capture it
runs once at capture time with the addresses the replays use; no device synchronization.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

import transformer_engine_torch as tex

from .nvfp4_tensor import NVFP4Quantizer, get_rht_matrix

__all__ = [
    "rht_matrix_for_group_amax",
    "set_precomputed_group_amax",
    "group_quantize",
    "precomputed_group_amax_stats",
    "clear_precomputed_group_amax",
]

MAX_PENDING = 8
"""Maximum number of precomputed amax entries waiting for their quantize."""

_PENDING: "OrderedDict[tuple, _Entry]" = OrderedDict()
_STATS: Dict[str, int] = {}


class _Entry:
    __slots__ = ("tensor", "amax", "rht_matrix")

    def __init__(self, tensor: torch.Tensor, amax: torch.Tensor, rht_matrix: torch.Tensor):
        self.tensor = tensor  # keeps the keyed storage alive while the entry waits
        self.amax = amax  # [2, num_groups] fp32: [0] pre-RHT row-wise, [1] post-RHT column-wise
        self.rht_matrix = rht_matrix


def _count(key: str) -> None:
    _STATS[key] = _STATS.get(key, 0) + 1


def _key(tensor: torch.Tensor) -> tuple:
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype, tensor.device)


def rht_matrix_for_group_amax(device: Optional[torch.device] = None) -> torch.Tensor:
    """The 16x16 BF16 RHT matrix ``R`` that the NVFP4 recipe's RHT quantizers apply (random sign
    mask on, the ``NVFP4Quantizer`` default) on ``device``. The post-RHT column-wise amax of a
    16-row block ``x_b`` is ``max |bf16(R^T x_b)|``."""
    if device is None:
        index = torch.cuda.current_device()
    else:
        device = torch.device(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
    return get_rht_matrix(True, index)


def set_precomputed_group_amax(
    tensor: torch.Tensor, amax: torch.Tensor, rht_matrix: torch.Tensor
) -> None:
    """Hand the per-group NVFP4 amaxes of ``tensor`` to its next grouped quantize.

    Args:
        tensor: The 2-D BF16 tensor that will be group-quantized (its rows grouped by the
            quantize's split sizes).
        amax: ``[2, num_groups]`` float32 on the tensor's device: ``amax[0]`` the pre-RHT
            row-wise amaxes, ``amax[1]`` the post-RHT column-wise amaxes, computed with
            ``rht_matrix``.
        rht_matrix: The RHT matrix the column-wise amaxes were computed with (see
            :func:`rht_matrix_for_group_amax`); the quantize uses the entry only if its quantizer
            applies the same matrix.
    """
    if amax.dim() != 2 or amax.shape[0] != 2 or amax.dtype != torch.float32:
        raise ValueError(f"amax must be [2, num_groups] float32, got {amax.dtype} {tuple(amax.shape)}")
    if amax.device != tensor.device or not amax.is_contiguous():
        raise ValueError("amax must be a contiguous tensor on the tensor's device")
    key = _key(tensor)
    if key in _PENDING:
        del _PENDING[key]
        _count("replaced")
    _PENDING[key] = _Entry(tensor, amax, rht_matrix)
    _count("registered")
    while len(_PENDING) > MAX_PENDING:
        _PENDING.popitem(last=False)
        _count("evicted")


def _take(tensor: torch.Tensor) -> Optional[_Entry]:
    if not _PENDING or not isinstance(tensor, torch.Tensor) or type(tensor) is not torch.Tensor:
        return None
    return _PENDING.pop(_key(tensor), None)


def _eligible(entry: _Entry, tensor: torch.Tensor, quantizer, num_groups: int) -> Optional[str]:
    """None if the entry can replace the quantizer's amax pass, else the reason it cannot."""
    if not isinstance(quantizer, NVFP4Quantizer):
        return "not_nvfp4"
    if not (quantizer.with_rht and quantizer.with_post_rht_amax):
        return "no_rht"
    if (
        quantizer.with_2d_quantization
        or quantizer.row_scaled_nvfp4
        or quantizer.disable_second_level_scale
        or getattr(quantizer, "nvfp4_use_4over6", False)
    ):
        return "quantizer_mode"
    rht = quantizer.rht_matrix
    if not (
        entry.rht_matrix is rht
        or (
            entry.rht_matrix.shape == rht.shape
            and entry.rht_matrix.dtype == rht.dtype
            and entry.rht_matrix.device == rht.device
            and entry.rht_matrix.data_ptr() == rht.data_ptr()
        )
    ):
        return "rht_matrix"
    if tuple(entry.amax.shape) != (2, int(num_groups)):
        return "num_groups"
    if tensor.dim() != 2 or not tensor.is_contiguous():
        return "layout"
    return None


def group_quantize(
    tensor: torch.Tensor,
    quantizer,
    num_groups: int,
    split_sizes: Optional[torch.Tensor],
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
):
    """``tex.group_quantize``, or ``tex.nvfp4_group_quantize_with_amax`` with the amaxes handed
    over by :func:`set_precomputed_group_amax` for this tensor (bitwise the same result when
    the amaxes are bitwise TE's; TE's grouped quantize never reduces grouped amaxes across ranks,
    so neither does this call)."""
    entry = _take(tensor)
    if entry is not None:
        reason = _eligible(entry, tensor, quantizer, num_groups)
        if reason is None:
            _count("used")
            reduce_amax = quantizer.with_amax_reduction
            quantizer.with_amax_reduction = False
            try:
                return tex.nvfp4_group_quantize_with_amax(
                    tensor,
                    quantizer,
                    int(num_groups),
                    split_sizes,
                    rowwise_amax=entry.amax[0],
                    columnwise_amax=entry.amax[1],
                    tensor_offsets=tensor_offsets,
                )
            finally:
                quantizer.with_amax_reduction = reduce_amax
        _count(f"ignored_{reason}")
    return tex.group_quantize(
        tensor,
        quantizer,
        num_groups,
        split_sizes,
        tensor_offsets=tensor_offsets,
    )


def precomputed_group_amax_stats() -> Dict[str, int]:
    """Counters: registered, used, replaced, evicted, ignored_<reason>; plus pending."""
    return dict(_STATS, pending=len(_PENDING))


def clear_precomputed_group_amax(reset_stats: bool = False) -> None:
    """Drop every waiting entry (and, optionally, the counters)."""
    _PENDING.clear()
    if reset_stats:
        _STATS.clear()
