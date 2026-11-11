# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Residual gradient folded into the RMSNorm backward.

* ``rmsnorm_bwd_add`` on the opt-in single-sweep kernel (NVTE_RMSNORM_BWD_SINGLE_SWEEP=1): the
  residual gradient is added to the BF16-rounded dx in the kernel's store, so the result is bitwise
  the single-sweep ``rmsnorm_bwd`` followed by a separate BF16 addition (what autograd's gradient
  accumulation computes) -- one kernel instead of two.
* ``te.ops.Sequential(MakeExtraOutput(), RMSNorm())`` (the ``BackwardAddRMSNorm`` fusion) and
  ``LayerNormLinear(return_layernorm_input=True)``: the residual connection's gradient reaches the
  module's backward and is accumulated there. Compared bitwise with the unfused modules whose input
  gradients autograd sums (single-sweep on), and within BF16 rounding on the regular kernels
  (which add in fp32 before rounding dx).
"""

import contextlib
import os

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe

ENV = "NVTE_RMSNORM_BWD_SINGLE_SWEEP"
KERNEL = "rmsnorm_bwd_single_sweep_kernel"

if torch.cuda.is_available():
    _CC = torch.cuda.get_device_capability()
else:
    _CC = (0, 0)
pytestmark = pytest.mark.skipif(
    _CC < (9, 0), reason="The single-sweep RMSNorm backward requires SM 9.0+"
)
_MXFP8_OK = _CC >= (10, 0)

SHAPES = [(8192, 3, 2880), (2048, 2880), (32, 2880), (4096, 3072), (96, 1024)]


@contextlib.contextmanager
def single_sweep(enabled: bool):
    old = os.environ.get(ENV)
    os.environ[ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            del os.environ[ENV]
        else:
            os.environ[ENV] = old


def make_inputs(shape, seed=1234, zero_centered_gamma=False):
    torch.manual_seed(seed)
    hidden = shape[-1]
    x = (torch.randn(shape, device="cuda") * 2.0).bfloat16()
    dz = torch.randn(shape, device="cuda").bfloat16()
    add = (torch.randn(shape, device="cuda") * 3.0).bfloat16()
    center = 0.0 if zero_centered_gamma else 1.0
    gamma = (center + 0.1 * torch.randn(hidden, device="cuda")).bfloat16()
    rsigma = torch.rsqrt(x.float().pow(2).mean(dim=-1) + 1e-5).reshape(-1).contiguous()
    return dz, x, add, rsigma, gamma


def kernels_of(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [evt.key for evt in prof.key_averages()]


def assert_bf16_close(actual, expected, name, addend=None):
    """Within BF16 rounding. With a residual `addend`, the regular kernels round dx + add once and
    the single-sweep kernel twice: allow one BF16 ulp of the addend's scale (cancellation)."""
    expected = expected.double()
    atol = 1e-4 * max(expected.abs().max().item(), 1.0)
    if addend is not None:
        atol = max(atol, 2**-7 * addend.abs().max().item())
    torch.testing.assert_close(
        actual.double(), expected, rtol=2**-7, atol=atol, msg=lambda m: f"{name}: {m}"
    )


# ------------------------------------------------------------------------------ rmsnorm_bwd_add
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_bwd_add_single_sweep_bitwise(shape, zero_centered_gamma):
    """Single sweep: rmsnorm_bwd_add == rmsnorm_bwd + add, bitwise, in one kernel."""
    dz, x, add, rsigma, gamma = make_inputs(shape, zero_centered_gamma=zero_centered_gamma)
    with single_sweep(True):
        dx_ref, dgamma_ref = tex.rmsnorm_bwd(dz, x, rsigma, gamma, 0, zero_centered_gamma)
        dx_ref = dx_ref + add
        dx, dgamma = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, zero_centered_gamma)
        names = kernels_of(
            lambda: tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, zero_centered_gamma)
        )
    assert any(KERNEL in n for n in names), names
    assert not any("elementwise" in n for n in names), names
    assert dx.dtype == torch.bfloat16 and dx.shape == x.shape
    assert torch.equal(dx, dx_ref), "dx: fused residual add differs from the separate addition"
    assert torch.equal(dgamma, dgamma_ref), "dgamma changed by the residual add"


@pytest.mark.parametrize("shape", [(8192, 3, 2880), (96, 1024)], ids=str)
def test_bwd_add_matches_regular_kernel(shape):
    """Single sweep vs the regular rmsnorm_bwd_add kernels: within BF16 rounding."""
    dz, x, add, rsigma, gamma = make_inputs(shape)
    with single_sweep(False):
        dx_ref, dgamma_ref = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False)
    with single_sweep(True):
        dx, dgamma = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False)
    assert_bf16_close(dx, dx_ref, "dx", addend=add)
    assert_bf16_close(dgamma, dgamma_ref, "dgamma")


def test_bwd_add_deterministic_and_graph():
    """Bitwise run to run; captured in a CUDA graph and replayed with new inputs."""
    shape = (8192, 3, 2880)
    static = [t.clone() for t in make_inputs(shape)]
    dz, x, add, rsigma, gamma = static
    with single_sweep(True):
        dx0, dg0 = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False)
        for _ in range(5):
            dx1, dg1 = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False)
            assert torch.equal(dx0, dx1) and torch.equal(dg0, dg1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gdx, gdg = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False)
        for seed in (5, 6):
            for dst, src in zip(static, make_inputs(shape, seed=seed)):
                dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            edx, edg = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False)
            assert torch.equal(gdx, edx) and torch.equal(gdg, edg), f"seed {seed}"


def _misaligned_like(t):
    buf = torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype)
    out = buf[1:].view(t.shape)
    out.copy_(t)
    return out


@pytest.mark.parametrize("case", ["misaligned_add", "rows_not_multiple_of_32"])
def test_bwd_add_fallback(case):
    """Unsupported configurations run the regular rmsnorm_bwd_add: the same bits as with single
    sweep off."""
    shape = (100, 2880) if case == "rows_not_multiple_of_32" else (64, 2880)
    dz, x, add, rsigma, gamma = make_inputs(shape)
    if case == "misaligned_add":
        add = _misaligned_like(add)
    with single_sweep(False):
        dx_ref, dg_ref = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False)
    with single_sweep(True):
        dx, dg = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False)
        names = kernels_of(lambda: tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, False))
    assert not any(KERNEL in n for n in names), names
    assert torch.equal(dx, dx_ref) and torch.equal(dg, dg_ref)


# --------------------------------------------------------- te.ops MakeExtraOutput + RMSNorm
def _residual_block_grads(norm_fn, inp, dy, dres):
    """d/dx of <dy, norm(x)> + <dres, residual(x)> where norm_fn returns (y, residual)."""
    x = inp.detach().clone().requires_grad_(True)
    y, res = norm_fn(x)
    torch.autograd.backward([y, res], [dy, dres])
    return x.grad


@pytest.mark.parametrize("enabled", [True, False])
def test_ops_make_extra_output_rmsnorm(enabled):
    """te.ops Sequential(MakeExtraOutput, RMSNorm) (BackwardAddRMSNorm) vs te.RMSNorm with the
    residual summed by autograd: bitwise with single sweep, within rounding without."""
    hidden = 2880
    norm = te.RMSNorm(hidden, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
    op_norm = te.ops.RMSNorm(hidden, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        op_norm.weight.copy_(norm.weight)
    fused = te.ops.Sequential(te.ops.MakeExtraOutput(), op_norm)
    torch.manual_seed(7)
    inp = torch.randn(2048, 3, hidden, device="cuda").bfloat16()
    dy = torch.randn_like(inp)
    dres = torch.randn_like(inp)
    with single_sweep(enabled):
        dx_ref = _residual_block_grads(lambda x: (norm(x), x), inp, dy, dres)
        dw_ref = norm.weight.grad.clone()
        dx = _residual_block_grads(fused, inp, dy, dres)
        dw = op_norm.weight.grad.clone()
        names = kernels_of(lambda: _residual_block_grads(fused, inp, dy, dres))
    if enabled:
        assert any(KERNEL in n for n in names), names
        assert torch.equal(dx, dx_ref), "dx"
        assert torch.equal(dw, dw_ref), "dgamma"
    else:
        assert_bf16_close(dx, dx_ref, "dx", addend=dres)
        assert_bf16_close(dw, dw_ref, "dgamma")


# ------------------------------------------------- LayerNormLinear(return_layernorm_input=True)
def _lnl_pair(normalization, hidden, out_features, **kwargs):
    common = dict(
        normalization=normalization,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        **kwargs,
    )
    ref = te.LayerNormLinear(hidden, out_features, **common)
    fused = te.LayerNormLinear(hidden, out_features, return_layernorm_input=True, **common)
    with torch.no_grad():
        for p_ref, p in zip(ref.parameters(), fused.parameters()):
            p_ref.normal_(1.0 if p_ref.dim() == 1 else 0.0, 0.1 if p_ref.dim() == 1 else 0.02)
            p.copy_(p_ref)
    return ref, fused


def _lnl_grads(module, fused, inp, dout, dres, steps=1):
    """Gradients of <dout, linear(norm(x))> + <dres, x> (the residual), summed over `steps`
    micro-batches (grad accumulation)."""
    module.zero_grad(set_to_none=True)
    x = inp.detach().clone().requires_grad_(True)
    outs = []
    for _ in range(steps):
        if fused:
            out, res = module(x)
            assert res.data_ptr() == x.data_ptr() and res.shape == x.shape
        else:
            out = module(x)
            res = x
        torch.autograd.backward([out, res], [dout, dres])
        outs.append(out.detach())
    return outs, x.grad, [p.grad.clone() for p in module.parameters()]


@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "mxfp8"])
@pytest.mark.parametrize("steps", [1, 2], ids=["ga1", "ga2"])
def test_layernorm_linear_return_input(normalization, enabled, fp8, steps):
    if fp8 and not _MXFP8_OK:
        pytest.skip("MXFP8 needs SM 10.0+")
    hidden, out_features = 2880, 1024
    ref, fused = _lnl_pair(normalization, hidden, out_features)
    torch.manual_seed(11)
    inp = torch.randn(1024, 3, hidden, device="cuda").bfloat16()
    dout = torch.randn(1024, 3, out_features, device="cuda").bfloat16()
    dres = torch.randn_like(inp)
    ctx = (
        te.autocast(enabled=True, recipe=recipe.MXFP8BlockScaling())
        if fp8
        else contextlib.nullcontext()
    )
    with single_sweep(enabled), ctx:
        out_ref, dx_ref, g_ref = _lnl_grads(ref, False, inp, dout, dres, steps)
        out, dx, g = _lnl_grads(fused, True, inp, dout, dres, steps)
    for o, o_ref in zip(out, out_ref):
        assert torch.equal(o, o_ref), "forward output changed"
    for p, p_ref in zip(g, g_ref):
        assert torch.equal(p, p_ref), "parameter gradient changed"
    if normalization == "LayerNorm" or enabled:
        # LayerNorm adds after the backward; single sweep folds the add bitwise
        assert torch.equal(dx, dx_ref), "input gradient"
    else:
        assert_bf16_close(dx, dx_ref, "input gradient (regular rmsnorm_bwd_add)", addend=dres)


def test_layernorm_linear_return_input_kernels():
    """With single sweep, the RMSNorm backward with the residual is one kernel and no separate
    addition of the residual gradient runs."""
    hidden = 2880
    _, fused = _lnl_pair("RMSNorm", hidden, 1024)
    inp = torch.randn(2048, 3, hidden, device="cuda").bfloat16()
    dout = torch.randn(2048, 3, 1024, device="cuda").bfloat16()
    dres = torch.randn_like(inp)
    with single_sweep(True):
        _lnl_grads(fused, True, inp, dout, dres)
        names = kernels_of(lambda: _lnl_grads(fused, True, inp, dout, dres))
    assert any(KERNEL in n for n in names), names
    assert not any("CUDAFunctor_add" in n for n in names), names


def test_layernorm_linear_return_input_unused():
    """The returned input left unused: no gradient for it, the plain backward."""
    hidden = 1024
    ref, fused = _lnl_pair("RMSNorm", hidden, 512)
    inp = torch.randn(256, hidden, device="cuda").bfloat16()
    dout = torch.randn(256, 512, device="cuda").bfloat16()
    with single_sweep(True):
        x_ref = inp.clone().requires_grad_(True)
        ref(x_ref).backward(dout)
        x = inp.clone().requires_grad_(True)
        out, _ = fused(x)
        out.backward(dout)
    assert torch.equal(x.grad, x_ref.grad)


@pytest.mark.parametrize(
    "return_bias,return_layernorm_output", [(False, False), (True, False), (True, True)]
)
def test_layernorm_linear_return_arity(return_bias, return_layernorm_output):
    """The input comes last, after the bias and the layernorm output."""
    module = te.LayerNormLinear(
        256,
        128,
        normalization="RMSNorm",
        bias=True,
        return_bias=return_bias,
        return_layernorm_output=return_layernorm_output,
        return_layernorm_input=True,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    x = torch.randn(64, 256, device="cuda").bfloat16().requires_grad_(True)
    outs = module(x)
    assert isinstance(outs, tuple)
    assert len(outs) == 2 + int(return_bias) + int(return_layernorm_output)
    assert outs[-1].data_ptr() == x.data_ptr() and outs[-1].shape == x.shape
    with torch.no_grad():
        outs = module(x)
    assert outs[-1].data_ptr() == x.data_ptr()
