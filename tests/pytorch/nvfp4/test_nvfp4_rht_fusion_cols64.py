# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVTE_NVFP4_RHT_FUSION_COLS64: fused RHT + NVFP4 cast for 64-column-aligned tensors.

With the switch on, single tensors whose last dimension is a multiple of 64 but not of 128
(e.g. a hidden size of 2880) take the fused RHT cast kernel instead of the unfused path
(RHT + amax, then separate rowwise and columnwise casts). Checks:

* the fused kernel is launched iff the switch is on (128-aligned shapes: always);
* without fast math and stochastic rounding the result is bitwise the unfused path's;
* with or without fast math, the result equals the fused kernel's on the input zero-padded
  to a 128-aligned width, cropped (i.e. the trailing half column tile is handled exactly);
* stochastic rounding: deterministic for a fixed seed, same error level as the unfused path;
* te.Linear forward/backward under NVFP4BlockScaling (no SR) is bitwise unchanged.
"""

import contextlib
import os

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import NVFP4Quantizer

from nvfp4_utils import get_nvfp4_scale_shape_no_padding

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)
sm100_family = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11)
pytestmark = [
    pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe),
    pytest.mark.skipif(not sm100_family, reason="RHT cast fusion kernel is SM100-family only"),
]

ENV = "NVTE_NVFP4_RHT_FUSION_COLS64"
FUSED_KERNEL = "row_col_rht_gemm_device"

# (rows, cols): cols % 64 == 0 and cols % 128 != 0, rows % 64 == 0.
# 2880 = GPT-OSS hidden size (dense QKV input, attention-proj output gradient).
COLS64_SHAPES = [
    (64, 64),
    (128, 192),
    (320, 320),
    (192, 2880),
    (320, 2880),
    (4096, 2880),
    (8192, 2880),
    (16384, 2880),
]
# 128-aligned shapes: unaffected by the switch (GPT-OSS dense QKV out 5120, proj in 4096).
ALIGNED_SHAPES = [(8192, 5120), (8192, 4096), (8192, 2944)]


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make_quantizer(rowwise=True, columnwise=True, sr=False, optimize_for_gemm=False, mask=True):
    q = NVFP4Quantizer(
        rowwise=rowwise,
        columnwise=columnwise,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=sr,
        with_random_sign_mask=mask,
    )
    q.optimize_for_gemm = optimize_for_gemm
    return q


def quantize(x, cols64, fast_math=False, seed=None, **qkw):
    q = make_quantizer(**qkw)
    with env(**{ENV: int(cols64), "NVTE_USE_FAST_MATH": int(fast_math)}):
        if seed is not None:
            torch.manual_seed(seed)
        y = q(x)
    torch.cuda.synchronize()
    return y


def parts(t, shape):
    """Valid region of every output buffer of an NVFP4 tensor (padding excluded)."""
    M, N = shape
    out = {}
    if t._rowwise_data is not None:
        out["qx"] = t._rowwise_data.view(torch.uint8)[:M, : N // 2]
        r, c = get_nvfp4_scale_shape_no_padding(shape, False)
        out["sx"] = t._rowwise_scale_inv.view(torch.uint8)[:r, :c]
        out["amax_row"] = t._amax_rowwise
    if t._columnwise_data is not None:
        out["qx_t"] = t._columnwise_data.view(torch.uint8)[:N, : M // 2]
        r, c = get_nvfp4_scale_shape_no_padding(shape, True)
        out["sx_t"] = t._columnwise_scale_inv.view(torch.uint8)[:r, :c]
        out["amax_col"] = t._amax_columnwise
    return out


def assert_parts_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape, k
        assert torch.equal(a[k], b[k]), f"{k} differs ({(a[k] != b[k]).sum().item()} entries)"


def kernel_names(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def fused_used(names):
    return any(FUSED_KERNEL in n and "group" not in n for n in names)


MODES = {"rowwise": (True, False), "columnwise": (False, True), "both": (True, True)}


@pytest.mark.parametrize("shape", COLS64_SHAPES[:4] + ALIGNED_SHAPES[:1], ids=str)
def test_fused_kernel_selection(shape):
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    for cols64 in (False, True):
        names = kernel_names(lambda: quantize(x, cols64))
        expect = cols64 or shape[1] % 128 == 0
        assert fused_used(names) == expect, (cols64, sorted(set(names)))


@pytest.mark.parametrize("shape", COLS64_SHAPES + ALIGNED_SHAPES, ids=str)
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("mask", [True, False], ids=["sign_mask", "no_sign_mask"])
@pytest.mark.parametrize("optimize_for_gemm", [False, True], ids=["compact_sf", "opt_for_gemm"])
def test_bitwise_vs_unfused(shape, mode, mask, optimize_for_gemm):
    """No fast math, no SR: switch on == switch off (unfused path for cols % 128 != 0)."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    rw, cw = MODES[mode]
    kw = dict(rowwise=rw, columnwise=cw, mask=mask, optimize_for_gemm=optimize_for_gemm)
    off = quantize(x, False, **kw)
    on = quantize(x, True, **kw)
    assert off._with_gemm_swizzled_scales == on._with_gemm_swizzled_scales
    if off._with_gemm_swizzled_scales:  # 128-aligned: compare the whole buffers
        for k in ("_rowwise_data", "_columnwise_data", "_rowwise_scale_inv",
                  "_columnwise_scale_inv", "_amax_rowwise", "_amax_columnwise"):
            a, b = getattr(off, k), getattr(on, k)
            assert (a is None) == (b is None), k
            if a is not None:
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), k
    else:
        assert_parts_equal(parts(off, shape), parts(on, shape))


@pytest.mark.parametrize("shape", COLS64_SHAPES, ids=str)
@pytest.mark.parametrize("fast_math", [False, True], ids=["exact_math", "fast_math"])
@pytest.mark.parametrize("mode", ["both", "columnwise"])
def test_matches_fused_on_padded_input(shape, fast_math, mode):
    """Switch on, cols % 128 == 64: equals the fused kernel on the input zero-padded to cols+64."""
    torch.manual_seed(1)
    M, N = shape
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    xp = torch.nn.functional.pad(x, (0, 64))
    rw, cw = MODES[mode]
    on = quantize(x, True, fast_math=fast_math, rowwise=rw, columnwise=cw)
    pad = quantize(xp, True, fast_math=fast_math, rowwise=rw, columnwise=cw)
    a, b = parts(on, shape), parts(pad, (M, N + 64))
    if "qx" in b:
        b["qx"] = b["qx"][:, : N // 2]
        b["sx"] = b["sx"][:, : N // 16]
    if "qx_t" in b:
        b["qx_t"] = b["qx_t"][:N]
        b["sx_t"] = b["sx_t"][:N]
    assert_parts_equal(a, b)


@pytest.mark.parametrize("shape", [(4096, 2880), (8192, 2880), (320, 2880)], ids=str)
def test_fast_math_difference_is_rounding_only(shape):
    """Fast math: the fused kernel casts the FP32 RHT output (no BF16 round trip), so on != off in
    general. Record the size of the difference; it must be a rounding-level effect."""
    torch.manual_seed(2)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    off = quantize(x, False, fast_math=True)
    on = quantize(x, True, fast_math=True)
    pa, pb = parts(off, shape), parts(on, shape)
    for k in ("amax_row", "amax_col"):
        assert torch.equal(pa[k], pb[k]), k
    fracs = {k: (pa[k] != pb[k]).float().mean().item() for k in ("qx", "sx", "qx_t", "sx_t")}
    print(f"\n{shape}: fast-math fraction of bytes differing on vs off: {fracs}")
    assert all(f < 0.02 for f in fracs.values()), fracs


@pytest.mark.parametrize("shape", [(4096, 2880), (8192, 2880)], ids=str)
def test_stochastic_rounding(shape):
    torch.manual_seed(3)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    a = quantize(x, True, sr=True, seed=123)
    b = quantize(x, True, sr=True, seed=123)
    assert_parts_equal(parts(a, shape), parts(b, shape))
    names = kernel_names(lambda: quantize(x, True, sr=True, seed=123))
    assert fused_used(names)
    # same error level as the unfused path (dequantize reads the rowwise data)
    off = quantize(x, False, sr=True, seed=123)
    err_on = (a.dequantize().float() - x.float()).pow(2).mean().sqrt()
    err_off = (off.dequantize().float() - x.float()).pow(2).mean().sqrt()
    assert torch.isfinite(err_on) and abs(err_on / err_off - 1) < 0.05, (err_on, err_off)


@pytest.mark.parametrize(
    "in_f, out_f", [(2880, 5120), (4096, 2880), (2880, 2880)], ids=["qkv", "proj", "sq2880"]
)
@pytest.mark.parametrize("tokens", [4096, 8192])
def test_linear_bitwise(in_f, out_f, tokens):
    """te.Linear under NVFP4BlockScaling (RHT on, SR off, no fast math): output, dgrad and
    wgrad are bitwise unchanged by the switch, and the fused kernel is used when on."""
    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)
    torch.manual_seed(4)
    lin = te.Linear(in_f, out_f, bias=False, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(tokens, in_f, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn(tokens, out_f, dtype=torch.bfloat16, device="cuda")

    def run(cols64):
        lin.weight.grad = None
        xx = x.clone().requires_grad_(True)
        with env(**{ENV: int(cols64), "NVTE_USE_FAST_MATH": 0}):
            with te.autocast(enabled=True, recipe=recipe):
                y = lin(xx)
            y.backward(dy)
        torch.cuda.synchronize()
        return y.detach().clone(), xx.grad.clone(), lin.weight.grad.clone()

    ref = run(False)
    got = run(True)
    for name, a, b in zip(("y", "dgrad", "wgrad"), ref, got):
        assert torch.equal(a, b), name
    names = kernel_names(lambda: run(True))
    assert fused_used(names)
