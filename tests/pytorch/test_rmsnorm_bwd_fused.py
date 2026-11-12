# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused single-sweep RMSNorm backward (``tex.rmsnorm_bwd_fused`` / ``nvte_rmsnorm_bwd_fused``).

On top of ``rmsnorm_bwd_add``'s residual gradient, the kernel optionally
* sums a second incoming gradient ``dz2`` into dz (one BF16 addition, as autograd's gradient
  accumulation of a normalized output with two consumers would), and
* emits the final dx quantized to MXFP8 (row-wise + column-wise, E4M3/E5M2, compact or
  GEMM-swizzled scales), so the GEMMs consuming dx skip their own quantization.

Everything is compared bitwise with the unfused sequence: BF16 ``dz + dz2``, the single-sweep
``rmsnorm_bwd(_add)`` and the MXFP8 quantizer applied to dx.
"""

import contextlib
import os

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.constants import FP8BwdTensorIdx
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

ENV = "NVTE_RMSNORM_BWD_SINGLE_SWEEP"
KERNEL = "rmsnorm_bwd_single_sweep_kernel"

_CC = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
pytestmark = pytest.mark.skipif(_CC < (9, 0), reason="single-sweep RMSNorm backward needs SM 9.0+")
needs_mxfp8 = pytest.mark.skipif(_CC < (10, 0), reason="MXFP8 needs SM 10.0+")

# rows % 128 == 0 for the quantized cases; hidden sizes with and without scale padding
SHAPES = [(8192, 3, 2880), (2048, 2880), (128, 3072), (384, 1024), (256, 160)]
FP8_DTYPES = {"e4m3": tex.DType.kFloat8E4M3, "e5m2": tex.DType.kFloat8E5M2}


@contextlib.contextmanager
def single_sweep():
    old = os.environ.get(ENV)
    os.environ[ENV] = "1"
    try:
        yield
    finally:
        if old is None:
            del os.environ[ENV]
        else:
            os.environ[ENV] = old


def make_inputs(shape, seed=4321, zero_centered_gamma=False):
    torch.manual_seed(seed)
    hidden = shape[-1]
    x = (torch.randn(shape, device="cuda") * 2.0).bfloat16()
    dz = torch.randn(shape, device="cuda").bfloat16()
    dz2 = (torch.randn(shape, device="cuda") * 0.5).bfloat16()
    add = (torch.randn(shape, device="cuda") * 3.0).bfloat16()
    # a few large and tiny blocks to exercise the scales
    dz.view(-1, hidden)[::7, :32] *= 64
    dz.view(-1, hidden)[1::11, 32:64] *= 1e-3
    center = 0.0 if zero_centered_gamma else 1.0
    gamma = (center + 0.1 * torch.randn(hidden, device="cuda")).bfloat16()
    rsigma = torch.rsqrt(x.float().pow(2).mean(dim=-1) + 1e-5).reshape(-1).contiguous()
    return dz, dz2, x, add, rsigma, gamma


def make_quantizer(fp8, swizzled):
    q = MXFP8Quantizer(FP8_DTYPES[fp8], rowwise=True, columnwise=True)
    q.optimize_for_gemm = swizzled
    q.internal = False
    return q


def reference(dz, dz2, x, add, rsigma, gamma, zcg, quantizer):
    with single_sweep():
        dz_sum = dz if dz2 is None else dz + dz2
        if add is None:
            dx, dgamma = tex.rmsnorm_bwd(dz_sum, x, rsigma, gamma, 0, zcg)
        else:
            dx, dgamma = tex.rmsnorm_bwd_add(dz_sum, x, add, rsigma, gamma, 0, zcg)
    q = quantizer(dx) if quantizer is not None else None
    return dx, dgamma, q


def assert_mxfp8_equal(a, b):
    """Data and scales bitwise. Swizzled scales: the whole buffers (the quantizer zeroes the
    padding, so must the kernel). Compact scales: the valid region (the quantizer leaves the
    padding uninitialized; the kernel writes zeros there)."""
    assert isinstance(a, MXFP8Tensor), type(a)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a._fp8_dtype == b._fp8_dtype
    assert a._with_gemm_swizzled_scales == b._with_gemm_swizzled_scales
    rows, cols = a.numel() // a.shape[-1], a.shape[-1]
    valid = {"_rowwise_scale_inv": (rows, cols // 32), "_columnwise_scale_inv": (rows // 32, cols)}
    for name in ("_rowwise_data", "_columnwise_data", "_rowwise_scale_inv", "_columnwise_scale_inv"):
        ta, tb = getattr(a, name).view(torch.uint8), getattr(b, name).view(torch.uint8)
        assert ta.shape == tb.shape, (name, ta.shape, tb.shape)
        if name in valid and not a._with_gemm_swizzled_scales:
            r, c = valid[name]
            assert not ta[:, c:].any() and not ta[r:].any(), f"{name}: padding not zero"
            ta, tb = ta[:r, :c], tb[:r, :c]
        assert torch.equal(ta, tb), f"{name} differs"


def kernels_of(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [evt.key for evt in prof.key_averages()]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("with_dz2", [False, True])
@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_fused_bf16_bitwise(shape, with_dz2, with_add, zero_centered_gamma):
    """dz2 and residual add folded in: bitwise the BF16 add + single-sweep backward."""
    dz, dz2, x, add, rsigma, gamma = make_inputs(shape, zero_centered_gamma=zero_centered_gamma)
    dz2 = dz2 if with_dz2 else None
    add = add if with_add else None
    assert tex.rmsnorm_bwd_fused_supported(dz, dz2, x, add, rsigma, gamma, zero_centered_gamma)
    dx, dgamma, q = tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, zero_centered_gamma, None)
    dx_ref, dgamma_ref, _ = reference(dz, dz2, x, add, rsigma, gamma, zero_centered_gamma, None)
    assert q is None
    assert dx.shape == x.shape and dx.dtype == torch.bfloat16
    assert torch.equal(dx, dx_ref)
    assert torch.equal(dgamma, dgamma_ref)


@needs_mxfp8
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("fp8", ["e4m3", "e5m2"])
@pytest.mark.parametrize("swizzled", [True, False])
@pytest.mark.parametrize("with_dz2", [False, True])
@pytest.mark.parametrize("with_add", [False, True])
def test_fused_mxfp8_bitwise(shape, fp8, swizzled, with_dz2, with_add):
    """MXFP8 copy of dx: bitwise the quantizer applied to dx (data, scales, padding)."""
    dz, dz2, x, add, rsigma, gamma = make_inputs(shape)
    dz2 = dz2 if with_dz2 else None
    add = add if with_add else None
    quantizer = make_quantizer(fp8, swizzled)
    dx, dgamma, q = tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)
    dx_ref, dgamma_ref, q_ref = reference(dz, dz2, x, add, rsigma, gamma, False, quantizer)
    assert torch.equal(dx, dx_ref)
    assert torch.equal(dgamma, dgamma_ref)
    assert q is not None, "quantized dx not emitted"
    assert_mxfp8_equal(q, q_ref)
    # the dequantized values agree as well
    assert torch.equal(q.dequantize(), q_ref.dequantize())


@needs_mxfp8
def test_fused_one_kernel():
    """dz2 + add + MXFP8: a single kernel (no add, no quantize kernels)."""
    dz, dz2, x, add, rsigma, gamma = make_inputs((8192, 3, 2880))
    quantizer = make_quantizer("e4m3", True)
    tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)  # warm-up
    names = kernels_of(
        lambda: tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)
    )
    assert any(KERNEL in n for n in names), names
    assert not any("elementwise" in n or "quantize" in n for n in names), names


@needs_mxfp8
def test_fused_quantized_unsupported_rows():
    """rows % 128 != 0: no quantized copy (None), dx/dgamma still bitwise."""
    dz, dz2, x, add, rsigma, gamma = make_inputs((96, 1024))
    quantizer = make_quantizer("e4m3", True)
    dx, dgamma, q = tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)
    dx_ref, dgamma_ref, _ = reference(dz, dz2, x, add, rsigma, gamma, False, None)
    assert q is None
    assert torch.equal(dx, dx_ref) and torch.equal(dgamma, dgamma_ref)


def test_fused_unsupported():
    """Shapes the single-sweep kernel does not cover are reported unsupported."""
    dz, dz2, x, add, rsigma, gamma = make_inputs((100, 1024))
    assert not tex.rmsnorm_bwd_fused_supported(dz, dz2, x, add, rsigma, gamma, False)
    dz, dz2, x, add, rsigma, gamma = make_inputs((128, 4096))
    assert not tex.rmsnorm_bwd_fused_supported(dz, dz2, x, add, rsigma, gamma, False)
    dz, dz2, x, add, rsigma, gamma = make_inputs((128, 1024))
    assert not tex.rmsnorm_bwd_fused_supported(dz, dz2.float(), x, add, rsigma, gamma, False)
    with pytest.raises(RuntimeError):
        tex.rmsnorm_bwd_fused(dz, dz2.float(), x, add, rsigma, gamma, False, None)


@needs_mxfp8
def test_fused_deterministic_and_graph():
    """Bitwise reproducible, and replayable from a CUDA graph."""
    dz, dz2, x, add, rsigma, gamma = make_inputs((8192, 3, 2880))
    quantizer = make_quantizer("e4m3", True)
    out1 = tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)
    out2 = tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)
    assert torch.equal(out1[0], out2[0]) and torch.equal(out1[1], out2[1])
    assert_mxfp8_equal(out1[2], out2[2])

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_g[0], out1[0]) and torch.equal(out_g[1], out1[1])
    assert_mxfp8_equal(out_g[2], out1[2])


@needs_mxfp8
@pytest.mark.parametrize("bias", [False])
def test_quantized_grad_into_linear(bias):
    """A TE Linear (MXFP8) handed the fused MXFP8 dx as its grad_output computes bitwise the same
    dgrad and wgrad as when it quantizes the BF16 dx itself."""
    torch.manual_seed(7)
    seq, batch, hidden = 1024, 2, 2880
    lin = te.Linear(hidden, hidden, bias=bias, params_dtype=torch.bfloat16).cuda()
    inp = torch.randn(seq, batch, hidden, device="cuda", dtype=torch.bfloat16)
    dz, dz2, x, add, rsigma, gamma = make_inputs((seq, batch, hidden))
    mx = recipe.MXFP8BlockScaling()

    def run(grad_fn):
        lin.zero_grad(set_to_none=True)
        a = inp.clone().requires_grad_(True)
        with te.autocast(enabled=True, recipe=mx):
            out = lin(a)
        quantizer = lin.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1].copy()
        quantizer.internal = False
        quantizer.set_usage(rowwise=True, columnwise=True)
        out.backward(grad_fn(quantizer))
        return a.grad.clone(), lin.weight.grad.clone()

    def fused_grad(quantizer):
        _, _, q = tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, quantizer)
        assert isinstance(q, MXFP8Tensor)
        return q

    def bf16_grad(quantizer):
        dx, _, _ = reference(dz, dz2, x, add, rsigma, gamma, False, None)
        return dx

    dgrad_q, wgrad_q = run(fused_grad)
    dgrad_b, wgrad_b = run(bf16_grad)
    assert torch.equal(dgrad_q, dgrad_b)
    assert torch.equal(wgrad_q, wgrad_b)
