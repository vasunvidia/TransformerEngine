# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""WeightGradStore stream mode (WeightGradStream): wgrad GEMMs issued on a side stream at put()."""

from __future__ import annotations

import gc
import os
import sys
import weakref

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import WeightGradStream
from transformer_engine.pytorch.cpp_extensions import gemm as te_gemm
from transformer_engine.pytorch.module._common import WeightGradStore

sys.path.insert(0, os.path.dirname(__file__))
from utils import make_recipe  # pylint: disable=wrong-import-position

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
_quantizations = [None] + (["mxfp8"] if mxfp8_available else [])

# One long-running kernel: ~cycles at ~2 GHz.
_SLEEP_CYCLES = 200_000_000


def _seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _init_main_grad(params, *, overwrite: bool, fill: float = 0.25):
    for p in params:
        p.main_grad = torch.full(p.shape, fill, dtype=torch.float32, device=p.device)
        p.overwrite_main_grad = overwrite
        p.grad_added_to_main_grad = False


def _reset_main_grad(params, fill: float = 0.25):
    for p in params:
        p.main_grad.fill_(fill)
        p.grad_added_to_main_grad = False
        p.grad = None


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.detach().contiguous()
    b = b.detach().contiguous()
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


# ----------------------------------------------------------------------------- WeightGradStream


def test_issue_waits_on_producer_and_does_not_block_it():
    """The wgrad reads what the producer wrote just before put(); the producer does not wait."""
    wgs = WeightGradStream(keep_alive_groups=0)
    x = torch.zeros(1 << 20, device="cuda")
    out = torch.empty_like(x)
    torch.cuda.synchronize()
    torch.cuda._sleep(_SLEEP_CYCLES)  # producer busy
    x.fill_(3.0)  # written by the producer after the sleep

    def wgrad(t, o):
        o.copy_(t * 2)
        torch.cuda._sleep(_SLEEP_CYCLES)  # long wgrad

    wgs.issue([x, out], wgrad, tag="t")
    after = torch.cuda.Event()
    after.record()  # producer work after the put
    after.synchronize()
    # The producer finished; the wgrad stream is still sleeping.
    assert not wgs.stream.query()
    wgs.join()
    torch.cuda.synchronize()
    assert torch.all(out == 6.0)
    assert wgs.stats["issued_t"] == 1


def test_hold_issues_at_next_put_and_flush():
    wgs = WeightGradStream()
    calls = []
    wgs.issue([], lambda: calls.append("a"), tag="a", hold=True)
    assert calls == [] and wgs.has_pending()
    wgs.issue([], lambda: calls.append("b"), tag="b")
    assert calls == ["a", "b"]
    wgs.issue([], lambda: calls.append("c"), tag="c", hold=True)
    assert calls == ["a", "b"]
    wgs.join()
    assert calls == ["a", "b", "c"] and not wgs.has_pending()


def test_keep_alive_groups_release_inputs():
    """Inputs stay referenced until `keep_alive_groups` newer groups exist (or join)."""
    wgs = WeightGradStream(keep_alive_groups=1)
    refs = []
    for _ in range(4):
        t = torch.ones(16, device="cuda")
        refs.append(weakref.ref(t))
        wgs.issue([t], lambda a: a.mul_(2), new_group=True)
        del t
        gc.collect()
    # groups 0, 1 released (group 3 current, group 2 kept alive)
    assert [r() is None for r in refs] == [True, True, False, False]
    assert wgs.stats["groups_released"] == 2
    wgs.join()
    gc.collect()
    assert all(r() is None for r in refs)


def test_event_ring_exhaustion_is_reported():
    wgs = WeightGradStream(num_events=4)
    wgs.issue([], lambda: None)
    wgs.issue([], lambda: None)
    with pytest.raises(RuntimeError, match="event ring"):
        wgs.issue([], lambda: None)
    wgs.join()
    wgs.issue([], lambda: None)  # a join resets the count
    wgs.join()


# ------------------------------------------------------------------------- workspace isolation


def test_per_stream_workspaces():
    wgs = WeightGradStream()
    dev = torch.cuda.current_device()
    shared = te_gemm.get_cublas_workspace(dev, False, False)
    assert te_gemm._workspace_stream_key() == 0
    with torch.cuda.stream(wgs.stream):
        key = te_gemm._workspace_stream_key()
        assert key > 0
        own = te_gemm.get_cublas_workspace(dev, False, False, key)
        assert own.data_ptr() != shared.data_ptr()
        assert (
            te_gemm._get_grouped_gemm_setup_workspace(dev, 8, key).data_ptr()
            != te_gemm._get_grouped_gemm_setup_workspace(dev, 8, 0).data_ptr()
        )
        assert (
            te_gemm._get_grouped_cublas_workspace(dev, "NT", key).data_ptr()
            != te_gemm._get_grouped_cublas_workspace(dev, "NT", 0).data_ptr()
        )
    assert te_gemm.register_cublas_workspace_stream(wgs.stream) == key  # idempotent


def test_concurrent_split_k_gemms_bitwise():
    """Split-K style GEMMs (small M,N, huge K) on the compute and the wgrad stream at once give
    the serial results bit for bit (each stream has its own cuBLAS workspace)."""
    _seed()
    wgs = WeightGradStream()
    shapes = [(64, 64, 1 << 16)] * 8
    a = [torch.randn(k, m, device="cuda", dtype=torch.bfloat16) for m, n, k in shapes]
    b = [torch.randn(k, n, device="cuda", dtype=torch.bfloat16) for m, n, k in shapes]

    def gemm(i, out):
        te.cpp_extensions.general_gemm(a[i], b[i], out_dtype=torch.float32, layout="NT", out=out, grad=True)

    ref = [torch.empty(n, m, device="cuda", dtype=torch.float32) for m, n, k in shapes]
    for i, r in enumerate(ref):
        gemm(i, r)
    torch.cuda.synchronize()
    for _ in range(5):
        out_c = [torch.empty_like(r) for r in ref]
        out_w = [torch.empty_like(r) for r in ref]
        wgs.issue([], lambda: [gemm(i, out_w[i]) for i in range(len(shapes))])
        for i in range(len(shapes)):
            gemm(i, out_c[i])
        wgs.join()
        torch.cuda.synchronize()
        for r, c, w in zip(ref, out_c, out_w):
            assert _bits_equal(r, c) and _bits_equal(r, w)


# ------------------------------------------------------------------------ modules, bitwise


def _make_module(kind, hidden, out, bias, dtype):
    if kind == "Linear":
        return te.Linear(hidden, out, bias=bias, params_dtype=dtype, fuse_wgrad_accumulation=True)
    return te.LayerNormLinear(
        hidden, out, bias=bias, params_dtype=dtype, normalization="RMSNorm", fuse_wgrad_accumulation=True
    )


def _run_dense(kind, quantization, overwrite, use_stream, *, bias=False, hold=False, steps=2):
    _seed()
    dtype = torch.bfloat16
    hidden, out_f, tokens = 256, 384, 512
    layers = [_make_module(kind, hidden, out_f, bias, dtype), _make_module(kind, out_f, hidden, bias, dtype)]
    weights = [l.weight for l in layers]
    _init_main_grad(weights, overwrite=overwrite)
    wgs = None
    if use_stream:
        wgs = WeightGradStream()
        for i, l in enumerate(layers):
            l.set_wgrad_stream(wgs, tag=f"l{i}", hold=hold and i == 1, new_group=i == 1)
    recipe = make_recipe(quantization)
    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    results = []
    for _ in range(steps):
        _reset_main_grad(weights)
        with te.autocast(enabled=quantization is not None, recipe=recipe):
            y = layers[1](layers[0](x))
        torch.cuda._sleep(_SLEEP_CYCLES // 10)  # producer lags behind the host
        y.backward(dy)
        if wgs is not None:
            wgs.join()
        torch.cuda.synchronize()
        results.append(
            [w.main_grad.clone() for w in weights]
            + [x.grad.clone()]
            + ([l.bias.grad.clone() for l in layers] if bias else [])
        )
        x.grad = None
        for l in layers:
            if bias:
                l.bias.grad = None
    return results, wgs


@pytest.mark.parametrize("kind", ["Linear", "LayerNormLinear"])
@pytest.mark.parametrize("quantization", _quantizations)
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("hold", [False, True])
def test_dense_modules_bitwise_vs_inline(kind, quantization, overwrite, hold):
    ref, _ = _run_dense(kind, quantization, overwrite, use_stream=False)
    got, wgs = _run_dense(kind, quantization, overwrite, use_stream=True, hold=hold)
    for r_step, g_step in zip(ref, got):
        for r, g in zip(r_step, g_step):
            assert _bits_equal(r, g)
    assert wgs.stats["issued"] == 2 * 2  # 2 modules x 2 steps, nothing inline
    for l_tag in ("l0", "l1"):
        assert wgs.stats[f"issued_{l_tag}"] == 2


def test_bf16_bias_epilogue_wgrad_stays_inline():
    """A BF16 wgrad GEMM that also produces the bias gradient is not deferred (nothing would
    assign that bias gradient later); results identical, nothing issued."""
    ref, _ = _run_dense("Linear", None, False, use_stream=False, bias=True)
    got, wgs = _run_dense("Linear", None, False, use_stream=True, bias=True)
    for r_step, g_step in zip(ref, got):
        for r, g in zip(r_step, g_step):
            assert _bits_equal(r, g)
    assert wgs.stats["issued"] == 0


@pytest.mark.parametrize("quantization", _quantizations)
@pytest.mark.parametrize("path", ["split_quantize", "grouped_tensor"])
def test_grouped_linear_module_bitwise(quantization, path):
    """te.GroupedLinear (MCore's unfused expert path): list path and grouped-tensor path."""
    num_gemms, hidden, out_f = 4, 256, 512
    m_splits = [128, 256, 0, 128]
    grouped = path == "grouped_tensor"

    def run(use_stream):
        _seed()
        m = te.GroupedLinear(
            num_gemms, hidden, out_f, bias=False, params_dtype=torch.bfloat16,
            fuse_wgrad_accumulation=True, use_grouped_tensor=grouped, single_grouped_weight=grouped,
        )
        params = [p for n, p in m.named_parameters() if "weight" in n]
        _init_main_grad(params, overwrite=False)
        wgs = None
        if use_stream:
            wgs = WeightGradStream()
            m.set_wgrad_stream(wgs, tag="g")
        x = torch.randn(sum(m_splits), hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(sum(m_splits), out_f, device="cuda", dtype=torch.bfloat16)
        splits = torch.tensor(m_splits, device="cuda" if grouped else "cpu", dtype=torch.int64)
        recipe = make_recipe(quantization)
        with te.autocast(enabled=quantization is not None, recipe=recipe):
            y = m(x, splits)
        torch.cuda._sleep(_SLEEP_CYCLES // 10)
        y.backward(dy)
        if wgs is not None:
            wgs.join()
        torch.cuda.synchronize()
        return [p.main_grad.clone() for p in params] + [x.grad.clone()], wgs

    try:
        ref, _ = run(False)
    except RuntimeError as e:
        if "grouped-tensor path" in str(e) or "not support" in str(e):
            pytest.skip(str(e))
        raise
    got, wgs = run(True)
    for r, g in zip(ref, got):
        assert _bits_equal(r, g)
    assert wgs.stats["issued_g"] == 1


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_fused_grouped_mlp_ops_bitwise():
    """te.ops grouped MLP (GroupedLinear + ScaledSwiGLU + GroupedLinear, MXFP8, single grouped
    weight, main_grad accumulation): both grouped wgrads on the stream, bitwise vs inline."""
    group_size, hidden, ffn, align = 4, 256, 512, 256
    split_sizes = torch.tensor([align * i for i in (2, 1, 0, 3)], dtype=torch.int, device="cuda")
    tokens = int(split_sizes.sum().item())

    def run(use_stream):
        _seed()
        recipe = make_recipe("mxfp8")
        with te.quantized_model_init(enabled=True, recipe=recipe):
            fc1 = te.ops.GroupedLinear(
                group_size, hidden, 2 * ffn, bias=False, device="cuda", dtype=torch.bfloat16,
                single_grouped_weight=True, accumulate_into_main_grad=True, split_alignment=align,
            )
            fc2 = te.ops.GroupedLinear(
                group_size, ffn, hidden, bias=False, device="cuda", dtype=torch.bfloat16,
                single_grouped_weight=True, accumulate_into_main_grad=True, split_alignment=align,
            )
            module = te.ops.Sequential(fc1, te.ops.ScaledSwiGLU(glu_interleave_size=32), fc2)
        params = [fc1.weight, fc2.weight]
        _init_main_grad(params, overwrite=True)
        wgs = None
        if use_stream:
            wgs = WeightGradStream()
            fc2.set_wgrad_stream(wgs, tag="fc2", new_group=True)
            fc1.set_wgrad_stream(wgs, tag="fc1")
            assert not fc1.need_backward_dw() and not fc2.need_backward_dw()
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(tokens, device="cuda", dtype=torch.float32, requires_grad=True)
        dy = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        with te.autocast(enabled=True, recipe=recipe):
            y = module(x, split_sizes, probs, split_sizes)
        torch.cuda._sleep(_SLEEP_CYCLES // 10)
        y.backward(dy)
        if wgs is not None:
            wgs.join()
        torch.cuda.synchronize()
        return [p.main_grad.clone() for p in params] + [x.grad.clone(), probs.grad.clone()], wgs

    ref, _ = run(False)
    got, wgs = run(True)
    for r, g in zip(ref, got):
        assert _bits_equal(r, g)
    assert wgs.stats["issued_fc1"] == 1 and wgs.stats["issued_fc2"] == 1


# ---------------------------------------------------------------------------- CUDA graphs


@pytest.mark.parametrize("quantization", _quantizations)
def test_capture_and_replay_bitwise(quantization):
    """Forward + backward + join captured in one CUDA graph from a non-default stream; the wgrad
    stream forks from and rejoins the capture. Replays are bitwise equal to eager inline runs."""
    hidden, out_f, tokens = 256, 384, 512
    recipe = make_recipe(quantization)

    def build(use_stream):
        _seed()
        l0 = te.Linear(hidden, out_f, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=True)
        l1 = te.LayerNormLinear(
            out_f, hidden, bias=False, params_dtype=torch.bfloat16, normalization="RMSNorm",
            fuse_wgrad_accumulation=True,
        )
        weights = [l0.weight, l1.weight]
        _init_main_grad(weights, overwrite=True)
        wgs = None
        if use_stream:
            wgs = WeightGradStream()
            l1.set_wgrad_stream(wgs, tag="l1", new_group=True)
            l0.set_wgrad_stream(wgs, tag="l0", hold=True)
        return (l0, l1), weights, wgs

    xs = [torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    dys = [torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

    # Reference: eager, inline wgrads.
    (l0, l1), weights, _ = build(False)
    ref = []
    for x, dy in zip(xs, dys):
        x = x.clone().requires_grad_()
        for w in weights:
            w.grad_added_to_main_grad = False
        with te.autocast(enabled=quantization is not None, recipe=recipe):
            y = l1(l0(x))
        y.backward(dy)
        torch.cuda.synchronize()
        ref.append([w.main_grad.clone() for w in weights] + [x.grad.clone()])

    # Stream mode, captured.
    (l0, l1), weights, wgs = build(True)
    static_x = torch.empty_like(xs[0]).requires_grad_()
    static_dy = torch.empty_like(dys[0])

    def step():
        for w in weights:
            w.grad_added_to_main_grad = False
        with te.autocast(enabled=quantization is not None, recipe=recipe):
            y = l1(l0(static_x))
        y.backward(static_dy)
        wgs.join()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):  # warm-up
            with torch.no_grad():
                static_x.copy_(xs[0]); static_dy.copy_(dys[0])
            static_x.grad = None
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    static_x.grad = None
    with torch.cuda.graph(graph):
        step()
    assert wgs.stats["issued_captured"] == 2
    for i, (x, dy) in enumerate(zip(xs, dys)):
        with torch.no_grad():
            static_x.copy_(x); static_dy.copy_(dy)
        graph.replay()
        torch.cuda.synchronize()
        got = [w.main_grad for w in weights] + [static_x.grad]
        for r, g in zip(ref[i], got):
            assert _bits_equal(r, g)


# ----------------------------------------------------------------------------- validation


def test_validation():
    wgs = WeightGradStream()
    lin = te.Linear(64, 64, bias=False, params_dtype=torch.bfloat16)  # no fused accumulation
    with pytest.raises(ValueError, match="fuse_wgrad_accumulation"):
        lin.set_wgrad_stream(wgs)
    mlp = te.LayerNormMLP(64, 128, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=True)
    with pytest.raises(NotImplementedError):
        mlp.set_wgrad_stream(wgs)
    store = WeightGradStore(delay_wgrad_compute=True)
    with pytest.raises(ValueError, match="backward_dw"):
        store.set_wgrad_stream(wgs)
    op = te.ops.GroupedLinear(2, 64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="accumulate_into_main_grad"):
        op.set_wgrad_stream(wgs)
    ok = te.Linear(64, 64, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=True)
    ok.set_wgrad_stream(wgs)
    assert ok.wgrad_store.stream_mode() and ok.wgrad_store.delay_wgrad_compute()
    assert not ok.need_backward_dw()
    assert not getattr(ok.weight, "skip_backward_post_hook", False)
    ok.backward_dw()  # nothing to do
    ok.set_wgrad_stream(None)
    assert not ok.wgrad_store.delay_wgrad_compute()
