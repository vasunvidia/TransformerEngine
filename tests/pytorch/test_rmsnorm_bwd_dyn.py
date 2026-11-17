# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Dynamic persistent single-sweep RMSNorm backward (``NVTE_RMSNORM_BWD_DYN=1``).

The dynamic kernel must reproduce the cluster kernel (the single-sweep backward without the switch)
bit for bit: dx and dgamma of ``rmsnorm_bwd``, ``rmsnorm_bwd_add`` and ``rmsnorm_bwd_fused`` (dz2,
add, both), for row counts with partial level-1 groups and fewer 8-row units than SMs, zero-centered
gamma, repeated launches (the unit and exit counters re-arm), CUDA graph replay and two streams.
Where it cannot run (MXFP8 copy of dx, misaligned rsigma) the cluster kernel takes over.
"""

import contextlib
import os

import pytest
import torch

import transformer_engine.pytorch  # noqa: F401
import transformer_engine_torch as tex

DYN_KERNEL = "rmsnorm_bwd_dyn_kernel"
CLUSTER_KERNEL = "rmsnorm_bwd_single_sweep_kernel"

_CC = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
pytestmark = pytest.mark.skipif(_CC < (9, 0), reason="single-sweep RMSNorm backward needs SM 9.0+")

# (rows, hidden): gpt-oss small (48 full level-1 groups), a partial last group (100 row blocks),
# fewer units than SMs (28 units), other hidden sizes
SHAPES = [(24576, 2880), (3200, 2880), (224, 2880), (8192, 3072), (4096, 1024), (256, 160)]
KINDS = ["plain", "add", "dz2", "dz2_add"]


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make_inputs(rows, hidden, seed=1234, zcg=False):
    torch.manual_seed(seed)
    x = (torch.randn(rows, hidden, device="cuda") * 2.0).bfloat16()
    dz = torch.randn(rows, hidden, device="cuda").bfloat16()
    dz2 = (torch.randn(rows, hidden, device="cuda") * 0.5).bfloat16()
    add = (torch.randn(rows, hidden, device="cuda") * 3.0).bfloat16()
    gamma = ((0.1 if zcg else 1.0) * (1.0 + 0.1 * torch.randn(hidden, device="cuda"))).bfloat16()
    rsigma = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-5).contiguous()
    return x, dz, dz2, add, gamma, rsigma


def call(kind, x, dz, dz2, add, gamma, rsigma, zcg=False):
    if kind == "plain":
        dx, dg = tex.rmsnorm_bwd(dz, x, rsigma, gamma, 0, zcg)
    elif kind == "add":
        dx, dg = tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, 0, zcg)
    else:
        dx, dg, _ = tex.rmsnorm_bwd_fused(dz, dz2, x, add if kind == "dz2_add" else None, rsigma,
                                          gamma, zcg, None)
    return dx, dg


def kernels_of(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return out, names


def run(kind, inputs, dyn, zcg=False):
    with env(NVTE_RMSNORM_BWD_SINGLE_SWEEP=1, NVTE_RMSNORM_BWD_DYN=int(dyn)):
        (dx, dg), names = kernels_of(lambda: call(kind, *inputs, zcg=zcg))
    return dx, dg, names


def assert_bitwise(a, b, what):
    assert a.shape == b.shape and a.dtype == b.dtype, what
    if not torch.equal(a, b):
        nd = int((a.view(torch.int16) != b.view(torch.int16)).sum())
        raise AssertionError(f"{what}: {nd} elements differ, max abs {(a.float() - b.float()).abs().max():.3e}")


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("zcg", [False, True], ids=["gamma", "zcg"])
def test_bitwise_vs_cluster(shape, kind, zcg):
    inputs = make_inputs(*shape, zcg=zcg)
    dx0, dg0, n0 = run(kind, inputs, dyn=False, zcg=zcg)
    assert any(CLUSTER_KERNEL in n for n in n0) and not any(DYN_KERNEL in n for n in n0)
    for rep in range(3):  # the counters re-arm: every launch bitwise
        dx1, dg1, n1 = run(kind, inputs, dyn=True, zcg=zcg)
        assert any(DYN_KERNEL in n for n in n1), n1
        assert not any(CLUSTER_KERNEL in n for n in n1), n1
        assert_bitwise(dx1, dx0, f"dx rep {rep}")
        assert_bitwise(dg1, dg0, f"dgamma rep {rep}")


@pytest.mark.parametrize("kind", KINDS)
def test_vs_fp64_reference(kind):
    rows, hidden = 3200, 2880
    x, dz, dz2, add, gamma, rsigma = make_inputs(rows, hidden)
    dx, dg, names = run(kind, (x, dz, dz2, add, gamma, rsigma), dyn=True)
    assert any(DYN_KERNEL in n for n in names)
    dzr = (dz.float() + dz2.float()).bfloat16().double() if kind.startswith("dz2") else dz.double()
    xd, g, rs = x.double(), gamma.double(), rsigma.double()[:, None]
    y = xd * rs
    dzg = dzr * g
    ref = rs * (dzg - y * (dzg * y).mean(-1, keepdim=True))
    if kind in ("add", "dz2_add"):
        ref = ref + add.double()
    dg_ref = (dzr * y).sum(0)
    torch.testing.assert_close(dx.double(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dg.double(), dg_ref, rtol=2e-2, atol=1.0)


@pytest.mark.parametrize("kind", ["plain", "dz2_add"])
def test_cuda_graph_replay(kind):
    inputs = make_inputs(24576, 2880)
    dx0, dg0, _ = run(kind, inputs, dyn=False)
    with env(NVTE_RMSNORM_BWD_SINGLE_SWEEP=1, NVTE_RMSNORM_BWD_DYN=1):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                call(kind, *inputs)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dx_g, dg_g = call(kind, *inputs)
            dx_g2, dg_g2 = call(kind, *inputs)  # two launches in one graph
    for i in range(4):
        dx_g.zero_(); dg_g.zero_(); dx_g2.zero_(); dg_g2.zero_()  # noqa: E702
        graph.replay()
        torch.cuda.synchronize()
        for a, b, w in ((dx_g, dx0, "dx"), (dg_g, dg0, "dgamma"), (dx_g2, dx0, "dx#2"), (dg_g2, dg0, "dgamma#2")):
            assert_bitwise(a, b, f"replay {i} {w}")


def test_two_streams():
    a = make_inputs(24576, 2880, seed=1)
    b = make_inputs(24576, 2880, seed=2)
    ra = run("add", a, dyn=False)
    rb = run("dz2_add", b, dyn=False)
    with env(NVTE_RMSNORM_BWD_SINGLE_SWEEP=1, NVTE_RMSNORM_BWD_DYN=1):
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        for s in (s1, s2):
            s.wait_stream(torch.cuda.current_stream())
        outs = []
        for _ in range(5):
            with torch.cuda.stream(s1):
                oa = call("add", *a)
            with torch.cuda.stream(s2):
                ob = call("dz2_add", *b)
            outs.append((oa, ob))
        torch.cuda.synchronize()
    for oa, ob in outs:
        assert_bitwise(oa[0], ra[0], "stream 1 dx")
        assert_bitwise(oa[1], ra[1], "stream 1 dgamma")
        assert_bitwise(ob[0], rb[0], "stream 2 dx")
        assert_bitwise(ob[1], rb[1], "stream 2 dgamma")


def test_fallback_misaligned_rsigma():
    x, dz, dz2, add, gamma, rsigma = make_inputs(3200, 2880)
    buf = torch.empty(rsigma.numel() + 1, dtype=torch.float32, device="cuda")
    rs_off = buf[1:]  # 4-byte aligned only
    rs_off.copy_(rsigma)
    dx0, dg0, _ = run("add", (x, dz, dz2, add, gamma, rsigma), dyn=False)
    dx1, dg1, names = run("add", (x, dz, dz2, add, gamma, rs_off), dyn=True)
    assert not any(DYN_KERNEL in n for n in names) and any(CLUSTER_KERNEL in n for n in names)
    assert_bitwise(dx1, dx0, "dx")
    assert_bitwise(dg1, dg0, "dgamma")


@pytest.mark.skipif(_CC < (10, 0), reason="MXFP8 needs SM 10.0+")
def test_mxfp8_copy_stays_on_cluster_kernel():
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    x, dz, dz2, add, gamma, rsigma = make_inputs(4096, 2880)
    q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    q.internal = False

    def go():
        return tex.rmsnorm_bwd_fused(dz, dz2, x, add, rsigma, gamma, False, q)

    with env(NVTE_RMSNORM_BWD_DYN=0):
        ref, _ = kernels_of(go)
    with env(NVTE_RMSNORM_BWD_DYN=1):
        out, names = kernels_of(go)
    assert not any(DYN_KERNEL in n for n in names)
    assert_bitwise(out[0], ref[0], "dx")
    assert_bitwise(out[1], ref[1], "dgamma")
