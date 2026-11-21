# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped NVFP4 quantization with 2D (16x16 block) scaling.

``tex.split_quantize`` with a 2D NVFP4 quantizer quantizes the splits of one BF16 matrix with a
grouped amax and one grouped cast kernel (row-wise and transposed outputs of every split). The
result has to be bitwise what ``NVFP4Quantizer`` (tex.quantize) gives each split on its own,
whether ``split_quantize`` allocates the outputs or quantizes into tensors the caller owns.
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)
pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)

_BUFFERS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
    "_amax_rowwise",
    "_amax_columnwise",
)


def _quantizer(rowwise=True, columnwise=True):
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=rowwise,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
    )


def _valid(name, t, rows, cols):
    """The scale tensors are padded (128 x 4) and allocated uninitialised: compare the valid part."""
    if name == "_rowwise_scale_inv":
        return t[:rows, : cols // 16]
    if name == "_columnwise_scale_inv":
        return t[:cols, : rows // 16]
    return t


def _assert_same(got, ref, label, rows, cols):
    for name in _BUFFERS:
        a, b = getattr(got, name), getattr(ref, name)
        if a is None or b is None:
            assert a is None and b is None, f"{label}: {name} present on one side only"
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, f"{label}: {name} layout differs"
        a, b = _valid(name, a, rows, cols), _valid(name, b, rows, cols)
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), f"{label}: {name} differs"


def _data(split_sections, cols, scale=1.0, seed=0):
    torch.manual_seed(seed)
    x = torch.randn((sum(split_sections), cols), dtype=torch.bfloat16, device="cuda") * scale
    # a few outliers so the block and tensor amaxes differ
    x[:: max(1, x.shape[0] // 7), ::53] *= 9.0
    return x


@pytest.mark.parametrize("cols", [128, 384, 2944])
@pytest.mark.parametrize("split_sections", [[64] * 3, [128, 64, 192], [2944, 2944], [64, 0, 128]])
@pytest.mark.parametrize("usage", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("into_existing", [False, True])
def test_group_quantize_2d_matches_per_tensor(cols, split_sections, usage, into_existing):
    if cols == 2944 and sum(split_sections) > 4000 and usage != (True, True):
        pytest.skip("redundant large case")
    rowwise, columnwise = usage
    x = _data(split_sections, cols)
    quantizers = [_quantizer(rowwise, columnwise) for _ in split_sections]
    ref = [q(xi) if xi.numel() else None for q, xi in zip(quantizers, torch.split(x, split_sections))]

    if into_existing:
        outs = [q.make_empty((n, cols), dtype=torch.bfloat16, device="cuda") for q, n in zip(quantizers, split_sections)]
        got = tex.split_quantize(x, split_sections, quantizers, False, outs)
        assert all(g is o for g, o in zip(got, outs)), "outputs must be the tensors that were passed"
    else:
        got = tex.split_quantize(x, split_sections, quantizers)
    for i, (g, r) in enumerate(zip(got, ref)):
        if r is not None:
            _assert_same(g, r, f"split {i}", split_sections[i], cols)


def test_group_quantize_2d_into_views_of_one_buffer():
    """The owner-cast layout: every output's buffers are views into one flat byte buffer."""
    cols, split_sections = 2944, [2944] * 3
    x = _data(split_sections, cols, seed=3)
    q = _quantizer()
    quantizers = [q] * len(split_sections)
    ref = [q(xi) for xi in torch.split(x, split_sections)]

    template = [q.make_empty((n, cols), dtype=torch.bfloat16, device="cuda") for n in split_sections]
    sizes = [[getattr(t, name).numel() * getattr(t, name).element_size() for name in _BUFFERS] for t in template]
    flat = torch.full((sum(map(sum, sizes)) + 64,), 0x5A, dtype=torch.uint8, device="cuda")
    outs, off = [], 16  # 16-byte aligned views with untouched guard bytes around them
    for t, sz in zip(template, sizes):
        views = {}
        for name, n in zip(_BUFFERS, sz):
            tmpl = getattr(t, name)
            views[name] = flat[off : off + n].view(tmpl.dtype).view(tmpl.shape)
            off += n
        outs.append(
            NVFP4Tensor(
                shape=(cols, cols),
                dtype=torch.bfloat16,
                rowwise_data=views["_rowwise_data"],
                rowwise_scale_inv=views["_rowwise_scale_inv"],
                columnwise_data=views["_columnwise_data"],
                columnwise_scale_inv=views["_columnwise_scale_inv"],
                amax_rowwise=views["_amax_rowwise"],
                amax_columnwise=views["_amax_columnwise"],
                fp4_dtype=q.dtype,
                quantizer=q,
                with_gemm_swizzled_scales=False,
            )
        )
    got = tex.split_quantize(x, split_sections, quantizers, False, outs)
    for i, (g, r) in enumerate(zip(got, ref)):
        _assert_same(g, r, f"split {i}", cols, cols)
    assert torch.all(flat[:16] == 0x5A) and torch.all(flat[off:] == 0x5A), "wrote outside the views"


def test_group_quantize_2d_rejects_splits_not_multiple_of_64():
    """Splits that are not a multiple of 64 rows are rejected by the grouped amax kernel."""
    x = _data([96, 32], 128)
    quantizers = [_quantizer(), _quantizer()]
    with pytest.raises(Exception):
        tex.split_quantize(x, [96, 32], quantizers)
