# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fidelity tests for NVFP4 (non-RHT) 2D-quantize swizzled SF output.

Mirrors ``test_nvfp4_rht_quantize_swizzle_fusion.py`` but for the plain
2D-block-scaling quantize kernel used by cached weights: when the quantizer
has ``optimize_for_gemm=True``, ``with_2d_quantization=True``, ``with_rht=False``
and the shape is 128-aligned (so no SF padding is needed), the quantize kernel
should emit scale factors directly in the GEMM-swizzled layout ``cuBLAS LT``
consumes, eliminating the otherwise-required ``nvte_swizzle_scaling_factors``
pass between quantize and GEMM.

The fidelity contract is: ``quantizer_swizzle_fusion(x)`` produces SF that are
byte-equal to ``swizzle_nvfp4_scale(quantizer(x).sx)``. The ``_rowwise_data`` /
``_columnwise_data`` quantized FP4 buffers and amaxes must also be byte-equal
(the swizzle optimization changes only the SF layout, not the FP4 data itself).

The second part checks the fast 2D quantization path enabled by
``NVTE_NVFP4_FAST_2D_WEIGHT_QUANTIZATION=1`` (per-tensor amax and quantization in
two persistent kernels, ``nvte_nvfp4_2d_amax_quantize``): every output buffer must
be byte-equal to the default path, for new tensors and in-place updates (with a
noop flag), and linear layers must produce bitwise identical results.
"""

from typing import Optional, Tuple

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_torch as tex  # noqa: F401
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
    NVFP4TensorStorage,
)

from nvfp4_utils import (
    swizzle_nvfp4_scale,
    get_nvfp4_scale_shape_no_padding,
)

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _unpack_quantized_tensor(
    quantized_tensor: NVFP4TensorStorage,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract ``(qx, sx, amax_row, qx_t, sx_t, amax_col)`` for byte comparison."""
    qx, sx, amax_row = None, None, None
    qx_t, sx_t, amax_col = None, None, None
    if quantized_tensor._rowwise_data is not None:
        qx = quantized_tensor._rowwise_data.view(dtype=torch.uint8)
    if quantized_tensor._rowwise_scale_inv is not None:
        sx = quantized_tensor._rowwise_scale_inv
    if quantized_tensor._amax_rowwise is not None:
        amax_row = quantized_tensor._amax_rowwise
    if quantized_tensor._columnwise_data is not None:
        qx_t = quantized_tensor._columnwise_data.view(dtype=torch.uint8)
    if quantized_tensor._columnwise_scale_inv is not None:
        sx_t = quantized_tensor._columnwise_scale_inv
    if quantized_tensor._amax_columnwise is not None:
        amax_col = quantized_tensor._amax_columnwise
    return qx, sx, amax_row, qx_t, sx_t, amax_col


def _make_quantizer(return_rowwise: bool, return_transpose: bool) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=return_rowwise,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
    )


def _check_nvfp4_2d_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
) -> None:
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # Reference quantizer: compact SF (default, no optimize_for_gemm).
    quantizer = _make_quantizer(return_rowwise, return_transpose)

    # SUT: same quantizer but with swizzled-SF emission enabled. The kernel must
    # bake the swizzled layout in directly (no standalone swizzle pass).
    quantizer_swizzle_fusion = quantizer.copy()
    quantizer_swizzle_fusion.optimize_for_gemm = True

    sut = quantizer_swizzle_fusion(x)
    assert sut._with_gemm_swizzled_scales, (
        f"expected in-kernel swizzled SF for aligned shape ({M}, {N}); "
        "the 2D quantize kernel should have baked the swizzled layout in"
    )

    qx_swf, sx_swf, amax_row_swf, qx_t_swf, sx_t_swf, amax_col_swf = _unpack_quantized_tensor(sut)
    qx_ref, sx_ref, amax_row_ref, qx_t_ref, sx_t_ref, amax_col_ref = _unpack_quantized_tensor(
        quantizer(x)
    )

    if return_rowwise:
        # FP4 data buffer and amax must be byte-equal (swizzle only changes SF
        # layout, not the quantized data).
        torch.testing.assert_close(qx_swf, qx_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(amax_row_swf, amax_row_ref, atol=0.0, rtol=0.0)

        # SF tensor must match the swizzle of the reference compact SF.
        valid_scale_shape = get_nvfp4_scale_shape_no_padding(x.shape, False)
        assert valid_scale_shape == sx_swf.shape, (
            "rowwise SF shape mismatch; this test assumes the input shape needs no "
            f"SF padding (got valid={valid_scale_shape}, got_swf={sx_swf.shape})."
        )
        sx_ref_swizzled = swizzle_nvfp4_scale(M, N, sx_ref, columnwise=False)
        torch.testing.assert_close(sx_swf, sx_ref_swizzled, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t_swf, qx_t_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(amax_col_swf, amax_col_ref, atol=0.0, rtol=0.0)

        valid_scale_shape = get_nvfp4_scale_shape_no_padding(x.shape, True)
        assert valid_scale_shape == sx_t_swf.shape, (
            "columnwise SF shape mismatch; this test assumes the input shape needs no "
            f"SF padding (got valid={valid_scale_shape}, got_swf={sx_t_swf.shape})."
        )
        sx_t_ref_swizzled = swizzle_nvfp4_scale(M, N, sx_t_ref, columnwise=True)
        torch.testing.assert_close(sx_t_swf, sx_t_ref_swizzled, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # 128-aligned shapes (no SF padding needed) -> in-kernel swizzle eligible.
        (128, 128),
        (256, 256),
        (1024, 1024),
        (2048, 2048),
        (8192, 1024),
        (8192, 5120),
        (8192, 10240),
        (16384, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
def test_nvfp4_2d_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quantize_mode: str,
) -> None:
    if quantize_mode == "rowwise_only":
        return_rowwise, return_transpose = True, False
    elif quantize_mode == "both_directions":
        return_rowwise, return_transpose = True, True
    elif quantize_mode == "columnwise_only":
        return_rowwise, return_transpose = False, True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    _check_nvfp4_2d_quantize_swizzle_fusion(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
        (256, 512),
        # Ineligible shapes: compact scales are swizzled after quantization.
        (128, 192),
        (64, 128),
    ],
)
def test_nvfp4_2d_swizzle_fusion_end_to_end_swizzled(M: int, N: int) -> None:
    """End-to-end quantize must never crash and must always yield swizzled SF.

    Eligible (128-aligned) shapes get swizzled SF directly from the 2D quantize
    kernel; for ineligible shapes the kernel writes compact SF to scratch buffers,
    which are then swizzled into the output. Either way the final
    ``_with_gemm_swizzled_scales`` must be True.
    """
    quantizer = _make_quantizer(return_rowwise=True, return_transpose=True)
    quantizer.optimize_for_gemm = True
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    result = quantizer(x)
    assert result._with_gemm_swizzled_scales is True, (
        "End-to-end quantize expected _with_gemm_swizzled_scales=True for shape "
        f"({M}, {N}) with optimize_for_gemm=True + with_2d_quantization=True, "
        f"got {result._with_gemm_swizzled_scales}"
    )


_FAST_2D_ENV = "NVTE_NVFP4_FAST_2D_WEIGHT_QUANTIZATION"


def _make_input(shape: Tuple[int, ...], kind: str, seed: int) -> torch.Tensor:
    """BF16 input with a given value distribution."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "normal":
        x = torch.randn(shape, generator=g, device="cuda") * 0.02
    elif kind == "wide":
        # Magnitudes over ~2^-40..2^40: exercises the E4M3 scale range and saturation.
        x = torch.randn(shape, generator=g, device="cuda") * torch.exp2(
            torch.randn(shape, generator=g, device="cuda") * 10
        )
    elif kind == "sparse":
        # Mostly zeros, all-zero blocks and one large outlier.
        x = torch.randn(shape, generator=g, device="cuda")
        x = x * (torch.rand(shape, generator=g, device="cuda") < 0.05)
        flat = x.view(-1)
        flat[: min(flat.numel(), 4096)] = 0
        flat[flat.numel() // 2] = 3.0e4
    elif kind == "zeros":
        x = torch.zeros(shape, device="cuda")
    elif kind == "tiny":
        x = torch.randn(shape, generator=g, device="cuda") * 1e-38
    elif kind == "ties":
        # Values on E2M1 rounding ties after scaling.
        vals = torch.tensor(
            [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -2.5, -5.0], device="cuda"
        )
        x = vals[torch.randint(0, vals.numel(), shape, generator=g, device="cuda")]
    else:
        raise ValueError(f"Invalid input kind: {kind}")
    return x.to(torch.bfloat16)


def _assert_bitwise_equal(
    ref: NVFP4TensorStorage,
    out: NVFP4TensorStorage,
    shape_with_padded_scales: Optional[Tuple[int, ...]] = None,
) -> None:
    """Compare every buffer byte for byte. With ``shape_with_padded_scales``, compare only
    the valid (unpadded) region of compact scales; the padding is not initialized."""
    assert ref._with_gemm_swizzled_scales == out._with_gemm_swizzled_scales
    for name, a, b in zip(
        ("data", "scale_inv", "amax", "data_t", "scale_inv_t", "amax_t"),
        _unpack_quantized_tensor(ref),
        _unpack_quantized_tensor(out),
    ):
        assert (a is None) == (b is None), f"{name}: allocated in one tensor only"
        if a is None:
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if shape_with_padded_scales is not None and name.startswith("scale_inv"):
            rows, cols = get_nvfp4_scale_shape_no_padding(
                shape_with_padded_scales, columnwise=name.endswith("_t")
            )
            a, b = a[:rows, :cols], b[:rows, :cols]
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"{name} differs"


def _fast_2d_kernel_names(fn) -> list:
    """Names of the fast-path kernels that fn launches."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if "amax_quantize_2d" in e.name]


def _fast_2d_supported() -> bool:
    major, _ = torch.cuda.get_device_capability()
    return major == 10 or major == 11


_LLAMA_WEIGHT_SHAPES = [
    (6144, 4096),  # QKV
    (4096, 4096),  # attention output projection
    (28672, 4096),  # FC1
    (4096, 14336),  # FC2
    (128256, 4096),  # embedding / LM head
]
_EDGE_SHAPES = [(128, 128), (256, 384), (384, 1280), (1152, 128), (2, 128, 256)]


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "shape, kind",
    [(s, k) for s in _LLAMA_WEIGHT_SHAPES for k in ("normal", "wide")]
    + [(s, k) for s in _EDGE_SHAPES for k in ("normal", "wide", "sparse", "zeros", "tiny", "ties")],
    ids=str,
)
@pytest.mark.parametrize("columnwise", [True, False], ids=["rowwise_columnwise", "rowwise"])
@pytest.mark.parametrize("swizzled", [False, True], ids=["compact", "swizzled"])
def test_nvfp4_fast_2d_quantize(
    shape: Tuple[int, ...], kind: str, columnwise: bool, swizzled: bool, monkeypatch
) -> None:
    """Fast 2D quantization is bitwise equal to the default path."""
    quantizer = _make_quantizer(return_rowwise=True, return_transpose=columnwise)
    quantizer.optimize_for_gemm = swizzled
    x = _make_input(shape, kind, seed=0)

    monkeypatch.delenv(_FAST_2D_ENV, raising=False)
    ref = quantizer(x)
    monkeypatch.setenv(_FAST_2D_ENV, "1")
    if _fast_2d_supported():
        assert _fast_2d_kernel_names(lambda: quantizer(x)), "fast path not taken"
    _assert_bitwise_equal(ref, quantizer(x))


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", [(4096, 14336), (256, 384), (2, 128, 256)], ids=str)
@pytest.mark.parametrize("columnwise", [True, False], ids=["rowwise_columnwise", "rowwise"])
@pytest.mark.parametrize("swizzled", [False, True], ids=["compact", "swizzled"])
@pytest.mark.parametrize("noop", [None, 0.0, 1.0], ids=["no_noop", "noop_0", "noop_1"])
def test_nvfp4_fast_2d_update_quantized(
    shape: Tuple[int, ...],
    columnwise: bool,
    swizzled: bool,
    noop: Optional[float],
    monkeypatch,
) -> None:
    """In-place fast 2D quantization (update_quantized) with a noop flag."""
    quantizer = _make_quantizer(return_rowwise=True, return_transpose=columnwise)
    quantizer.optimize_for_gemm = swizzled
    monkeypatch.delenv(_FAST_2D_ENV, raising=False)
    dst_ref = quantizer(_make_input(shape, "normal", seed=0))
    dst_fast = quantizer(_make_input(shape, "normal", seed=0))
    for dst in (dst_ref, dst_fast):
        for t in _unpack_quantized_tensor(dst):
            if t is not None:
                t.view(torch.uint8).fill_(0xA5)
    noop_flag = None if noop is None else torch.full((1,), noop, device="cuda")
    x = _make_input(shape, "wide", seed=1)

    quantizer.update_quantized(x, dst_ref, noop_flag=noop_flag)
    monkeypatch.setenv(_FAST_2D_ENV, "1")
    if _fast_2d_supported():
        names = _fast_2d_kernel_names(
            lambda: quantizer.update_quantized(x, dst_fast, noop_flag=noop_flag)
        )
        assert names, "fast path not taken"
    else:
        quantizer.update_quantized(x, dst_fast, noop_flag=noop_flag)
    _assert_bitwise_equal(dst_ref, dst_fast)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", [(192, 128), (128, 192), (64, 256)], ids=str)
def test_nvfp4_fast_2d_quantize_unsupported_shape(shape: Tuple[int, int], monkeypatch) -> None:
    """Shapes that are not 128-aligned use the default path."""
    quantizer = _make_quantizer(return_rowwise=True, return_transpose=True)
    x = _make_input(shape, "normal", seed=0)
    monkeypatch.delenv(_FAST_2D_ENV, raising=False)
    ref = quantizer(x)
    monkeypatch.setenv(_FAST_2D_ENV, "1")
    assert not _fast_2d_kernel_names(lambda: quantizer(x))
    _assert_bitwise_equal(ref, quantizer(x), shape_with_padded_scales=shape)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_nvfp4_fast_2d_quantize_cuda_graph(monkeypatch) -> None:
    """Fast 2D quantization captured in a CUDA graph and replayed on new inputs."""
    monkeypatch.setenv(_FAST_2D_ENV, "1")
    quantizer = _make_quantizer(return_rowwise=True, return_transpose=True)
    quantizer.optimize_for_gemm = True
    x = _make_input((6144, 4096), "normal", seed=0)
    quantizer(x)  # Warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = quantizer(x)
    for seed in (1, 2):
        x.copy_(_make_input(x.shape, "wide", seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.delenv(_FAST_2D_ENV)
        ref = quantizer(x)
        monkeypatch.setenv(_FAST_2D_ENV, "1")
        _assert_bitwise_equal(ref, out)


def _run_linear_layers(num_steps: int = 2) -> list:
    """Forward and backward of te.Linear and a te.ops BasicLinear MLP under the
    NVFP4BlockScaling recipe, with a weight update between steps."""
    torch.manual_seed(1234)
    dtype = torch.bfloat16
    hidden, ffn, tokens = 512, 1024, 256
    linear = te.Linear(hidden, hidden, bias=False, params_dtype=dtype)
    mlp = te.ops.Sequential(
        te.ops.BasicLinear(hidden, 2 * ffn, dtype=dtype),
        te.ops.SwiGLU(),
        te.ops.BasicLinear(ffn, hidden, dtype=dtype),
    )
    modules = [linear, mlp]
    outs = []
    for step in range(num_steps):
        torch.manual_seed(100 + step)
        xs = [
            torch.randn(tokens, hidden, device="cuda", dtype=dtype, requires_grad=True)
            for _ in modules
        ]
        dys = [torch.randn(tokens, hidden, device="cuda", dtype=dtype) for _ in modules]
        with te.autocast(enabled=True, recipe=recipe.NVFP4BlockScaling()):
            ys = [m(x) for m, x in zip(modules, xs)]
        for y, dy in zip(ys, dys):
            y.backward(dy)
        outs += [y.detach() for y in ys] + [x.grad for x in xs]
        outs += [p.grad.clone() for m in modules for p in m.parameters()]
        with torch.no_grad():
            for m in modules:
                for p in m.parameters():
                    p.grad = None
                    p.add_(torch.randn_like(p) * 1e-3)
    torch.cuda.synchronize()
    weight_quantizers = [op.get_quantizer("forward", 1) for op in (mlp[0], mlp[2])]
    return outs, weight_quantizers


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_nvfp4_fast_2d_weight_quantization_linear_layers(monkeypatch) -> None:
    """Linear layers give bitwise identical outputs and gradients with the fast path, and
    te.ops.BasicLinear quantizes its weights with GEMM-swizzled scales."""
    monkeypatch.delenv(_FAST_2D_ENV, raising=False)
    ref, ref_quantizers = _run_linear_layers()
    assert not any(q.optimize_for_gemm for q in ref_quantizers)
    monkeypatch.setenv(_FAST_2D_ENV, "1")
    out, fast_quantizers = _run_linear_layers()
    if _fast_2d_supported():
        assert all(q.optimize_for_gemm for q in fast_quantizers)
    assert len(ref) == len(out)
    for i, (a, b) in enumerate(zip(ref, out)):
        assert a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16)), i
