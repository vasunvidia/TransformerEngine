# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tile-scheduler workspace of the grouped fused NVFP4 RHT cast kernels.

Modes: default (cudaMemsetAsync before every launch), NVTE_NVFP4_GROUP_CAST_ZERO_KERNEL=1
(1-block zeroing kernel) and NVTE_NVFP4_GROUP_CAST_SELF_RESET=1 (the kernel resets a persistent
per-stream workspace itself). Checks: bitwise identical outputs in all modes for the graph-safe
(tex.group_quantize) and the host-split (tex.split_quantize) grouped kernels, over repeated
launches, concurrent streams and CUDA graph replays; the captured graph has no memset node per
launch in the self-reset mode (node types read with the CUDA driver API).
"""

import contextlib
import ctypes
import math
import os

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)
sm100_family = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11)
pytestmark = [
    pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe),
    pytest.mark.skipif(not sm100_family, reason="grouped RHT cast kernels are SM100-family only"),
]

MODES = {
    "memset": {},
    "zero_kernel": {"NVTE_NVFP4_GROUP_CAST_ZERO_KERNEL": "1"},
    "self_reset": {
        "NVTE_NVFP4_GROUP_CAST_SELF_RESET": "1",
        "NVTE_NVFP4_GROUP_CAST_SELF_RESET_CHECK": "1",
    },
}
ALL_KEYS = sorted({k for m in MODES.values() for k in m})

# (experts, tokens, width): GPT-OSS expert widths padded to 128 multiples
# (2880 -> 2944; fc1 gradient width 2 x 2944), scaled-down and per-rank token counts.
CASES = [
    (4, 2048, 2944),
    (4, 2048, 5888),
    (32, 32768, 2944),
    (8, 8192, 5888),
    (2, 512, 128),
]


@contextlib.contextmanager
def mode_env(mode, check=True):
    old = {k: os.environ.get(k) for k in ALL_KEYS}
    try:
        for k in ALL_KEYS:
            os.environ.pop(k, None)
        for k, v in MODES[mode].items():
            if k.endswith("_CHECK") and not check:
                continue
            os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def splits_for(experts, tokens, seed, with_zero=True):
    g = torch.Generator().manual_seed(seed)
    units = tokens // 128
    cuts = sorted(torch.randint(0, units + 1, (experts - 1,), generator=g).tolist())
    s = [b - a for a, b in zip([0] + cuts, cuts + [units])]
    if with_zero and experts > 2:
        s[1] += s[0]
        s[0] = 0
    return [128 * v for v in s]


def make_quantizer(sr=False):
    return NVFP4Quantizer(
        rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True,
        stochastic_rounding=sr,
    )


def flat(tensors):
    """All output buffers, bytewise. Empty splits are skipped: their buffers (incl. amax) are
    not written by any mode."""
    out = []
    for t in tensors:
        if math.prod(t.shape) == 0:
            continue
        for k in ("_rowwise_data", "_rowwise_scale_inv", "_amax_rowwise",
                  "_columnwise_data", "_columnwise_scale_inv", "_amax_columnwise"):
            v = getattr(t, k)
            out.append(None if v is None else v.view(torch.uint8).clone())
    return out


def assert_same(a, b, what=""):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert (x is None) == (y is None), (what, i)
        if x is not None:
            assert torch.equal(x, y), f"{what}: buffer {i} differs"


class GroupQuantize:
    """Graph-safe grouped cast (device split sizes)."""

    def __init__(self, splits):
        self.st = torch.tensor(splits, dtype=torch.int64, device="cuda")
        self.n = len(splits)

    def raw(self, x, quantizer):
        return tex.group_quantize(x, quantizer, self.n, self.st)

    @staticmethod
    def tensors(raw):
        return raw.split_into_quantized_tensors()


class SplitQuantize:
    """Host-split grouped cast (tex.split_quantize -> group_row_col_rht_gemm_device)."""

    def __init__(self, splits):
        self.splits = list(splits)

    def raw(self, x, quantizer):
        return tex.split_quantize(x, self.splits, [quantizer.copy() for _ in self.splits])

    @staticmethod
    def tensors(raw):
        return raw


def run(entry, splits):
    e = ENTRY[entry](splits)
    return lambda x, q: e.tensors(e.raw(x, q))


ENTRY = {"group_quantize": GroupQuantize, "split_quantize": SplitQuantize}


@pytest.mark.parametrize("case", CASES, ids=lambda c: "x".join(map(str, c)))
@pytest.mark.parametrize("entry", list(ENTRY))
@pytest.mark.parametrize("sr", [False, True], ids=["rn", "sr"])
def test_bitwise_all_modes(case, entry, sr):
    experts, tokens, width = case
    splits = splits_for(experts, tokens, seed=tokens + width)
    torch.manual_seed(0)
    xs = [torch.randn(tokens, width, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    fn = run(entry, splits)
    results = {}
    for mode in MODES:
        with mode_env(mode):
            outs = []
            for i, x in enumerate(xs):  # consecutive launches reuse the self-reset workspace
                torch.manual_seed(100 + i)
                outs.append(flat(fn(x, make_quantizer(sr))))
            torch.cuda.synchronize()
        results[mode] = outs
    for mode in ("zero_kernel", "self_reset"):
        for i in range(len(xs)):
            assert_same(results["memset"][i], results[mode][i], f"{mode} launch {i}")


def test_self_reset_concurrent_streams():
    experts, tokens, width = 8, 8192, 2944
    splits = splits_for(experts, tokens, seed=7)
    torch.manual_seed(1)
    xs = [torch.randn(tokens, width, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    e = GroupQuantize(splits)
    with mode_env("memset"):
        ref = [flat(e.tensors(e.raw(x, make_quantizer()))) for x in xs]
    streams = [torch.cuda.Stream() for _ in range(4)]
    with mode_env("self_reset", check=False):
        for _ in range(3):
            outs = [None] * 4
            torch.cuda.synchronize()
            for i, (s, x) in enumerate(zip(streams, xs)):
                with torch.cuda.stream(s):
                    outs[i] = e.raw(x, make_quantizer())
            torch.cuda.synchronize()
            for i in range(4):
                assert_same(ref[i], flat(e.tensors(outs[i])), f"stream {i}")


# ---- CUDA graph node inspection (driver API on the graph PyTorch captured)
_CU_NODE_TYPES = {0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "graph", 5: "empty",
                  6: "wait_event", 7: "event_record", 8: "ext_sem_signal", 9: "ext_sem_wait",
                  10: "mem_alloc", 11: "mem_free", 12: "batch_mem_op", 13: "conditional"}


def graph_node_types(graph):
    """Node counts of the captured graph by type; kernel nodes also by function name (a
    captured cudaMemsetAsync may appear as a kernel node, depending on the driver)."""
    libcuda = ctypes.CDLL("libcuda.so.1")
    raw = ctypes.c_void_p(graph.raw_cuda_graph())
    n = ctypes.c_size_t(0)
    assert libcuda.cuGraphGetNodes(raw, None, ctypes.byref(n)) == 0
    nodes = (ctypes.c_void_p * n.value)()
    assert libcuda.cuGraphGetNodes(raw, nodes, ctypes.byref(n)) == 0
    counts = {"total": n.value}
    for node in nodes:
        t = ctypes.c_int(-1)
        assert libcuda.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(t)) == 0
        name = _CU_NODE_TYPES.get(t.value, str(t.value))
        if name == "kernel":
            params = (ctypes.c_void_p * 16)()  # CUDA_KERNEL_NODE_PARAMS(_v2); func is the first field
            fname = "?"
            if libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(node), params) == 0 and params[0]:
                cname = ctypes.c_char_p()
                if libcuda.cuFuncGetName(ctypes.byref(cname), ctypes.c_void_p(params[0])) == 0:
                    fname = cname.value.decode(errors="replace")
            short = ("memset_kernel" if "memset" in fname.lower() else
                     "zero_tile_scheduler_workspace" if "zero_tile_scheduler" in fname else
                     "group_rht_cast" if "group_row_col_rht_gemm" in fname else "other_kernel")
            counts["kernel:" + short] = counts.get("kernel:" + short, 0) + 1
        counts[name] = counts.get(name, 0) + 1
    return counts


def capture(fn, launches, stream=None):
    """Capture `launches` calls of fn; returns (graph, raw outputs). The graph is kept (not only
    its executable) so that its nodes can be inspected."""
    try:
        g = torch.cuda.CUDAGraph(keep_graph=True)
        keep = True
    except TypeError:  # older PyTorch keeps the graph anyway
        g = torch.cuda.CUDAGraph()
        keep = False
    with torch.cuda.graph(g, stream=stream):
        outs = [fn() for _ in range(launches)]
    g._needs_instantiate = keep
    return g, outs


def replay(g):
    if getattr(g, "_needs_instantiate", False):
        g.instantiate()
        g._needs_instantiate = False
    g.replay()


@pytest.mark.parametrize("entry", list(ENTRY))
def test_graph_nodes_and_replay(entry):
    experts, tokens, width, launches = 4, 2048, 2944, 3
    splits = splits_for(experts, tokens, seed=3)
    torch.manual_seed(2)
    x_static = torch.randn(tokens, width, dtype=torch.bfloat16, device="cuda")
    e = ENTRY[entry](splits)
    q = make_quantizer()
    counts, refs = {}, {}
    new_inputs = [torch.randn_like(x_static) for _ in range(2)]
    with mode_env("memset"):
        for i, xi in enumerate(new_inputs):
            refs[i] = flat(e.tensors(e.raw(xi, q)))
    for mode in MODES:
        with mode_env(mode, check=False):
            e.raw(x_static, q)  # warm-up (JIT, workspace creation) outside capture
            torch.cuda.synchronize()
            g, raws = capture(lambda: e.raw(x_static, q), launches, stream=torch.cuda.Stream())
            counts[mode] = graph_node_types(g)
            outs = [e.tensors(r) for r in raws]
            for i, xi in enumerate(new_inputs):
                x_static.copy_(xi)
                for _ in range(2):
                    replay(g)
                torch.cuda.synchronize()
                for o in outs:
                    assert_same(refs[i], flat(o), f"{mode} replay")
            del g
    print(f"\n{entry} graph node types ({launches} grouped casts): {counts}")
    tot = {key: counts[key]["total"] for key in counts}
    # one node in front of every grouped cast in the memset / zero-kernel modes, none in self-reset
    assert tot["memset"] - tot["self_reset"] == launches, counts
    assert tot["zero_kernel"] - tot["self_reset"] == launches, counts
    assert counts["zero_kernel"].get("kernel:zero_tile_scheduler_workspace", 0) == launches, counts
    sr = counts["self_reset"]
    assert sr.get("memset", 0) == 0 and sr.get("kernel:memset_kernel", 0) == 0, counts
    assert sr.get("kernel:zero_tile_scheduler_workspace", 0) == 0, counts
    assert sr.get("kernel:group_rht_cast", 0) == launches, counts


def test_self_reset_workspace_created_during_capture():
    """First use of a stream inside a capture: the workspace must be zero before the first
    replay, also when a second graph captured on that stream is replayed first."""
    experts, tokens, width = 4, 2048, 2944
    splits = splits_for(experts, tokens, seed=5)
    torch.manual_seed(3)
    x = torch.randn(tokens, width, dtype=torch.bfloat16, device="cuda")
    q = make_quantizer()
    e = GroupQuantize(splits)
    with mode_env("memset"):
        ref = flat(e.tensors(e.raw(x, q)))
    with mode_env("self_reset", check=False):
        e.raw(x, q)  # JIT / one-time init on the default stream
        torch.cuda.synchronize()
        s = torch.cuda.Stream()  # never used eagerly
        g1, r1 = capture(lambda: e.raw(x, q), 2, stream=s)
        g2, r2 = capture(lambda: e.raw(x, q), 2, stream=s)
        assert graph_node_types(g2).get("memset", 0) == graph_node_types(g1).get("memset", 0)
        replay(g2)
        replay(g1)
        replay(g2)
        torch.cuda.synchronize()
        for r in r1 + r2:
            assert_same(ref, flat(e.tensors(r)), "graph")
