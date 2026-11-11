# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests of the opt-in single-sweep RMSNorm backward (NVTE_RMSNORM_BWD_SINGLE_SWEEP=1).

The kernel computes dx and dgamma in fp32 with a different arithmetic order than the default
backward (TE or cuDNN): dx = rsigma * dz * gamma - x * k with k = sum(dz * gamma * x) * rsigma^3 / H
instead of rsigma * (dz * gamma - mean(dz * gamma * y) * y), other reduction trees and another
dgamma summation order. Its results are therefore compared with the default backward to within
BF16 rounding, not bitwise. Bitwise checks: run-to-run determinism, CUDA graph replay versus
eager, and the fallback cases (which must run the default backward).
"""

import contextlib
import os
import subprocess
import sys
import textwrap

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

ENV = "NVTE_RMSNORM_BWD_SINGLE_SWEEP"
KERNEL = "rmsnorm_bwd_single_sweep_kernel"

if torch.cuda.is_available():
    _CC = torch.cuda.get_device_capability()
else:
    _CC = (0, 0)
pytestmark = pytest.mark.skipif(
    _CC < (9, 0), reason="The single-sweep RMSNorm backward requires SM 9.0+"
)

# gpt-oss-20b: 3 x 8192 tokens per rank, hidden size 2880
SUPPORTED_SHAPES = [
    (8192, 3, 2880),
    (2048, 2880),
    (32, 2880),
    (4096, 3072),
    (96, 1024),
    (65536, 32),
]


@contextlib.contextmanager
def single_sweep(enabled: bool):
    """Set NVTE_RMSNORM_BWD_SINGLE_SWEEP (read by TE at every rmsnorm_bwd call)."""
    old = os.environ.get(ENV)
    os.environ[ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            del os.environ[ENV]
        else:
            os.environ[ENV] = old


def make_inputs(shape, dtype=torch.bfloat16, zero_centered_gamma=False, seed=1234):
    torch.manual_seed(seed)
    hidden = shape[-1]
    x = (torch.randn(shape, device="cuda") * 2.0).to(dtype)
    dz = torch.randn(shape, device="cuda").to(dtype)
    if zero_centered_gamma:
        gamma = (0.1 * torch.randn(hidden, device="cuda")).to(dtype)
    else:
        gamma = (1.0 + 0.1 * torch.randn(hidden, device="cuda")).to(dtype)
    rsigma = torch.rsqrt(x.float().pow(2).mean(dim=-1) + 1e-5).reshape(-1).contiguous()
    return dz, x, rsigma, gamma


def rmsnorm_bwd(dz, x, rsigma, gamma, zero_centered_gamma, enabled):
    with single_sweep(enabled):
        dx, dgamma = tex.rmsnorm_bwd(dz, x, rsigma, gamma, 0, zero_centered_gamma)
    return dx, dgamma


def reference_bwd(dz, x, rsigma, gamma, zero_centered_gamma):
    """fp64 reference."""
    hidden = x.shape[-1]
    x64 = x.reshape(-1, hidden).double()
    dz64 = dz.reshape(-1, hidden).double()
    rs = rsigma.double()[:, None]
    g = gamma.double() + (1.0 if zero_centered_gamma else 0.0)
    y = x64 * rs
    dy = dz64 * g
    mdyy = (dy * y).mean(dim=-1, keepdim=True)
    dx = rs * (dy - mdyy * y)
    dgamma = (dz64 * y).sum(dim=0)
    return dx.reshape(x.shape), dgamma


def ran_single_sweep(fn) -> bool:
    """Whether fn launched the single-sweep kernel."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return any(KERNEL in evt.key for evt in prof.key_averages())


def assert_bf16_close(actual, expected, name):
    """Within BF16 rounding: ~1 ulp relative, a small absolute floor for fp32 cancellation."""
    expected = expected.double()
    atol = 1e-4 * max(expected.abs().max().item(), 1.0)
    torch.testing.assert_close(
        actual.double(), expected, rtol=2**-7, atol=atol, msg=lambda m: f"{name}: {m}"
    )


@pytest.mark.parametrize("shape", SUPPORTED_SHAPES, ids=str)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_matches_default_backward(shape, zero_centered_gamma):
    dz, x, rsigma, gamma = make_inputs(shape, zero_centered_gamma=zero_centered_gamma)
    dx_ref, dgamma_ref = rmsnorm_bwd(dz, x, rsigma, gamma, zero_centered_gamma, False)
    dx, dgamma = rmsnorm_bwd(dz, x, rsigma, gamma, zero_centered_gamma, True)
    assert ran_single_sweep(
        lambda: rmsnorm_bwd(dz, x, rsigma, gamma, zero_centered_gamma, True)
    ), "the single-sweep kernel did not run"
    assert dx.dtype == torch.bfloat16 and dx.shape == x.shape
    assert dgamma.dtype == torch.bfloat16 and dgamma.shape == gamma.shape

    # Against the default backward: within BF16 rounding (not bitwise; see the module docstring).
    assert_bf16_close(dx, dx_ref, "dx vs default")
    assert_bf16_close(dgamma, dgamma_ref, "dgamma vs default")

    # Against fp64: no less accurate than the default backward (with a BF16-ulp allowance).
    dx64, dgamma64 = reference_bwd(dz, x, rsigma, gamma, zero_centered_gamma)
    assert_bf16_close(dx, dx64, "dx vs fp64")
    assert_bf16_close(dgamma, dgamma64, "dgamma vs fp64")
    for name, ours, default, exact in (
        ("dx", dx, dx_ref, dx64),
        ("dgamma", dgamma, dgamma_ref, dgamma64),
    ):
        err = (ours.double() - exact).abs().max().item()
        err_default = (default.double() - exact).abs().max().item()
        ulp = exact.abs().max().item() * 2**-8
        assert err <= err_default + ulp, f"{name}: max error {err} vs default {err_default}"


@pytest.mark.parametrize("shape", [(8192, 3, 2880), (96, 1024)], ids=str)
def test_deterministic(shape):
    """Bitwise identical across calls, including past a full turn of the 64-slot ticket ring."""
    dz, x, rsigma, gamma = make_inputs(shape)
    dx0, dgamma0 = rmsnorm_bwd(dz, x, rsigma, gamma, False, True)
    other = make_inputs(shape, seed=99)
    for i in range(70):
        if i % 7 == 3:  # interleave other inputs
            rmsnorm_bwd(*other, False, True)
        dx, dgamma = rmsnorm_bwd(dz, x, rsigma, gamma, False, True)
        assert torch.equal(dx, dx0), f"dx differs on call {i}"
        assert torch.equal(dgamma, dgamma0), f"dgamma differs on call {i}"


@pytest.mark.parametrize("shape", [(8192, 3, 2880), (4096, 3072)], ids=str)
def test_cuda_graph(shape):
    """Captured once, replayed with new inputs: bitwise the eager single-sweep results."""
    static = [t.clone() for t in make_inputs(shape)]
    with single_sweep(True):
        tex.rmsnorm_bwd(*static, 0, False)  # eager warm-up
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_dx, static_dgamma = tex.rmsnorm_bwd(*static, 0, False)
    for seed in (5, 6, 7):
        new = make_inputs(shape, seed=seed)
        for dst, src in zip(static, new):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        dx, dgamma = rmsnorm_bwd(*new, False, True)
        assert torch.equal(static_dx, dx), f"dx: replay differs from eager (seed {seed})"
        assert torch.equal(static_dgamma, dgamma), f"dgamma: replay differs (seed {seed})"


def test_cuda_graph_first_use_in_capture():
    """First use on the device inside a capture: the tickets come from the workspace (memset node).

    Runs in a fresh process so that the per-device ticket ring does not exist yet.
    """
    script = textwrap.dedent("""
        import torch
        import transformer_engine.pytorch  # noqa: F401
        import transformer_engine_torch as tex
        torch.manual_seed(0)
        shape = (8192, 3, 2880)
        x = (torch.randn(shape, device="cuda") * 2.0).bfloat16()
        dz = torch.randn(shape, device="cuda").bfloat16()
        gamma = (1.0 + 0.1 * torch.randn(shape[-1], device="cuda")).bfloat16()
        rsigma = torch.rsqrt(x.float().pow(2).mean(dim=-1) + 1e-5).reshape(-1).contiguous()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gdx, gdgamma = tex.rmsnorm_bwd(dz, x, rsigma, gamma, 0, False)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            dx, dgamma = tex.rmsnorm_bwd(dz, x, rsigma, gamma, 0, False)
            assert torch.equal(gdx, dx), "dx"
            assert torch.equal(gdgamma, dgamma), "dgamma"
        print("OK")
        """)
    env = dict(os.environ)
    env[ENV] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout + result.stderr


def _misaligned(shape, dtype=torch.bfloat16):
    """A contiguous tensor whose data pointer is 2 bytes past a 16-byte boundary."""
    numel = 1
    for s in shape:
        numel *= s
    buf = torch.randn(numel + 1, device="cuda").to(dtype)
    return buf[1:].view(shape)


FALLBACK_CASES = {
    "rows_not_multiple_of_32": dict(shape=(100, 2880)),
    "hidden_not_multiple_of_32": dict(shape=(64, 2896)),
    "hidden_above_3072": dict(shape=(64, 4096)),
    "fp32": dict(shape=(64, 2880), dtype=torch.float32),
    "fp16": dict(shape=(64, 2880), dtype=torch.float16),
    "misaligned_x": dict(shape=(64, 2880), misaligned=True),
}


@pytest.mark.parametrize("case", list(FALLBACK_CASES), ids=str)
def test_fallback(case):
    """Unsupported configurations silently run the default backward: the same bits."""
    cfg = FALLBACK_CASES[case]
    dz, x, rsigma, gamma = make_inputs(cfg["shape"], dtype=cfg.get("dtype", torch.bfloat16))
    if cfg.get("misaligned"):
        x2 = _misaligned(x.shape)
        x2.copy_(x)
        x = x2
    dx_ref, dgamma_ref = rmsnorm_bwd(dz, x, rsigma, gamma, False, False)
    dx, dgamma = rmsnorm_bwd(dz, x, rsigma, gamma, False, True)
    assert not ran_single_sweep(lambda: rmsnorm_bwd(dz, x, rsigma, gamma, False, True))
    assert torch.equal(dx, dx_ref)
    assert torch.equal(dgamma, dgamma_ref)


def test_disabled_by_default():
    dz, x, rsigma, gamma = make_inputs((2048, 2880))
    old = os.environ.pop(ENV, None)
    try:
        assert not ran_single_sweep(lambda: tex.rmsnorm_bwd(dz, x, rsigma, gamma, 0, False))
    finally:
        if old is not None:
            os.environ[ENV] = old


def _module_grads(module, inp, grad_out, enabled):
    module.zero_grad(set_to_none=True)
    x = inp.detach().clone().requires_grad_(True)
    with single_sweep(enabled):
        out = module(x)
        out.backward(grad_out)
    return x.grad, [p.grad.clone() for p in module.parameters()]


@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_te_rmsnorm_module(zero_centered_gamma):
    """te.RMSNorm (the op-fuser path, ops/basic/rmsnorm.py)."""
    hidden = 2880
    module = te.RMSNorm(
        hidden,
        eps=1e-5,
        device="cuda",
        dtype=torch.bfloat16,
        zero_centered_gamma=zero_centered_gamma,
    )
    with torch.no_grad():
        module.weight.normal_(0.0 if zero_centered_gamma else 1.0, 0.1)
    inp = torch.randn(2048, 3, hidden, device="cuda").bfloat16()
    grad_out = torch.randn_like(inp)
    dx_ref, grads_ref = _module_grads(module, inp, grad_out, False)
    dx, grads = _module_grads(module, inp, grad_out, True)
    assert ran_single_sweep(lambda: _module_grads(module, inp, grad_out, True))
    assert_bf16_close(dx, dx_ref, "dx")
    for g, g_ref in zip(grads, grads_ref):
        assert_bf16_close(g, g_ref, "dgamma")


def test_layernorm_linear_rmsnorm():
    """te.LayerNormLinear(normalization="RMSNorm") (the module path, module/layernorm_linear.py)."""
    hidden = 2880
    module = te.LayerNormLinear(
        hidden,
        1024,
        normalization="RMSNorm",
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    inp = torch.randn(2048, 3, hidden, device="cuda").bfloat16()
    grad_out = torch.randn(2048, 3, 1024, device="cuda").bfloat16()
    dx_ref, grads_ref = _module_grads(module, inp, grad_out, False)
    dx, grads = _module_grads(module, inp, grad_out, True)
    assert ran_single_sweep(lambda: _module_grads(module, inp, grad_out, True))
    assert_bf16_close(dx, dx_ref, "dx")
    for g, g_ref in zip(grads, grads_ref):
        assert_bf16_close(g, g_ref, "param grad")


_RTC_CONTEXT_SCRIPT = """
import os
import sys
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

ENV = "NVTE_RMSNORM_BWD_SINGLE_SWEEP"
first_single_sweep = sys.argv[1] == "1"


def inputs(shape, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(shape, device="cuda") * 2.0).bfloat16()
    dz = torch.randn(shape, device="cuda").bfloat16()
    gamma = (1.0 + 0.1 * torch.randn(shape[-1], device="cuda")).bfloat16()
    rsigma = torch.rsqrt(x.float().pow(2).mean(dim=-1) + 1e-5).reshape(-1).contiguous()
    return dz, x, rsigma, gamma


# 1. Large-hidden backward on the main thread (single-sweep or stock), then the stock backward:
#    loads TE's RTC norm kernels from the main thread and fills the caching allocator.
for shape in ((8192, 3, 2880), (4096, 3072)):
    os.environ[ENV] = "1" if first_single_sweep else "0"
    tex.rmsnorm_bwd(*inputs(shape), 0, False)
    os.environ[ENV] = "0"
    tex.rmsnorm_bwd(*inputs((2048, shape[-1])), 0, False)
torch.cuda.synchronize()

# 2. te.RMSNorm: its backward runs on PyTorch's autograd worker thread, whose allocations the
#    cache serves, so its first CUDA call can be a driver call of TE's RTC (occupancy query).
for hidden in (2880, 3072):
    norm = te.RMSNorm(hidden, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(2048, 3, hidden, device="cuda").bfloat16()
    grad = torch.randn_like(inp)
    results = []
    for enabled in ("0", "1"):
        os.environ[ENV] = enabled
        norm.zero_grad(set_to_none=True)
        x = inp.clone().requires_grad_(True)
        norm(x).backward(grad)
        results.append((x.grad, norm.weight.grad.clone()))
    torch.cuda.synchronize()
    (dx0, dw0), (dx1, dw1) = results
    torch.testing.assert_close(dx1.double(), dx0.double(), rtol=2**-7, atol=1e-3)
    torch.testing.assert_close(
        dw1.double(), dw0.double(), rtol=2**-7, atol=1e-4 * dw0.abs().max().item()
    )
print("OK")
"""


@pytest.mark.parametrize("first_single_sweep", [True, False])
def test_stock_backward_after_large_hidden(first_single_sweep):
    """Regression: a large-hidden backward on the main thread, then the stock backward and
    te.RMSNorm (backward on the autograd thread) in the same process.

    Used to fail in the stock half with "invalid device context" from TE's RTC occupancy query
    (rtc.cpp did not call ensure_context_exists() before cuOccupancyMaxActiveBlocksPerMultiprocessor
    / cuFuncSetAttribute). Runs in a fresh process: the failure needs a fresh autograd thread.
    """
    env = dict(os.environ)
    env.pop(ENV, None)
    result = subprocess.run(
        [sys.executable, "-c", _RTC_CONTEXT_SCRIPT, "1" if first_single_sweep else "0"],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout + result.stderr
