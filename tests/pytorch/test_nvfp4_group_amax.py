# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Precomputed per-group NVFP4 amaxes (transformer_engine.pytorch.tensor.nvfp4_group_amax)."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor import nvfp4_group_amax as ga
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

fp4_available, reason_for_no_fp4 = te.is_nvfp4_available(return_reason=True)
pytestmark = pytest.mark.skipif(not fp4_available, reason=reason_for_no_fp4)


def _quantizer(sr: bool, columnwise: bool = True) -> NVFP4Quantizer:
    q = NVFP4Quantizer(
        rowwise=True,
        columnwise=columnwise,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=sr,
    )
    q.optimize_for_gemm = True
    q.internal = True
    return q


def _inputs(counts, hidden=256, extra_rows=128, seed=0):
    # TE's graph-safe grouped amax needs the total rows to be a multiple of 128
    torch.manual_seed(seed)
    rows = sum(counts) + extra_rows
    x = (torch.randn(rows, hidden, device="cuda") * 3).to(torch.bfloat16)
    x[sum(counts) :] = 0
    split = torch.tensor(counts, dtype=torch.int64, device="cuda")
    offsets = torch.cat([split.new_zeros(1), torch.cumsum(split, 0)]) * hidden
    return x, split, offsets


def _bytes(t):
    return t.detach().contiguous().view(torch.uint8).view(-1)


def _written(out, n_rows, hidden):
    n128 = n_rows // 128 * 128
    prefix = {
        "rowwise_data": n_rows * hidden // 2,
        "columnwise_data": n_rows * hidden // 2,
        "scale_inv": n128 * hidden // 16,
        "columnwise_scale_inv": n128 * hidden // 16,
    }
    res = {}
    for name in ("amax", "columnwise_amax") + tuple(prefix):
        t = getattr(out, name, None)
        res[name] = None if t is None else _bytes(t)[: prefix.get(name, t.numel() * 4)]
    return res


def _te_amax(x, q, n, split, offsets):
    out = tex.group_quantize(x, q, n, split, tensor_offsets=offsets)
    return torch.stack([out.amax.clone(), out.columnwise_amax.clone()]).contiguous()


@pytest.mark.parametrize("sr", [False, True])
@pytest.mark.parametrize("counts", [[128, 0, 256, 384], [0, 0, 128, 0], [512, 128, 0, 128, 256]])
def test_precomputed_amax_bitwise(sr, counts):
    hidden = 256
    x, split, offsets = _inputs(counts, hidden)
    n = len(counts)
    q = _quantizer(sr)
    assert q.rht_matrix is ga.rht_matrix_for_group_amax(x.device)
    amax = _te_amax(x, q, n, split, offsets)
    ga.clear_precomputed_group_amax(reset_stats=True)
    rng = torch.cuda.get_rng_state()
    ref = tex.group_quantize(x, q, n, split, tensor_offsets=offsets)
    torch.cuda.set_rng_state(rng)
    ga.set_precomputed_group_amax(x, amax, q.rht_matrix)
    # a view of the same storage with the same shape finds the entry
    out = ga.group_quantize(x.reshape(-1, hidden), q, n, split, tensor_offsets=offsets)
    assert ga.precomputed_group_amax_stats()["used"] == 1
    assert ga.precomputed_group_amax_stats()["pending"] == 0
    a, r = _written(out, sum(counts), hidden), _written(ref, sum(counts), hidden)
    for name in a:
        assert (a[name] is None) == (r[name] is None), name
        if a[name] is not None:
            assert torch.equal(a[name], r[name]), name
    # consumed: the next quantize of the tensor runs TE's amax pass
    torch.cuda.set_rng_state(rng)
    again = ga.group_quantize(x, q, n, split, tensor_offsets=offsets)
    assert ga.precomputed_group_amax_stats()["used"] == 1
    assert torch.equal(_written(again, sum(counts), hidden)["rowwise_data"], r["rowwise_data"])


def test_precomputed_amax_is_used():
    """The quantize takes the handed-over values (not its own): a scaled amax changes the scales."""
    counts, hidden = [128, 256], 256
    x, split, offsets = _inputs(counts, hidden)
    q = _quantizer(False)
    amax = _te_amax(x, q, 2, split, offsets)
    ga.clear_precomputed_group_amax(reset_stats=True)
    ga.set_precomputed_group_amax(x, (amax * 2).contiguous(), q.rht_matrix)
    out = ga.group_quantize(x, q, 2, split, tensor_offsets=offsets)
    assert torch.equal(out.amax, amax[0] * 2) and torch.equal(out.columnwise_amax, amax[1] * 2)


def test_ineligible_entries_fall_back():
    counts, hidden = [128, 256], 256
    x, split, offsets = _inputs(counts, hidden)
    q = _quantizer(False)
    amax = _te_amax(x, q, 2, split, offsets)
    ref = tex.group_quantize(x, q, 2, split, tensor_offsets=offsets)
    ga.clear_precomputed_group_amax(reset_stats=True)
    # wrong RHT matrix
    ga.set_precomputed_group_amax(x, (amax * 2).contiguous(), q.rht_matrix.clone())
    out = ga.group_quantize(x, q, 2, split, tensor_offsets=offsets)
    assert torch.equal(out.amax, ref.amax)
    # wrong group count
    ga.set_precomputed_group_amax(x, torch.zeros(2, 3, device="cuda"), q.rht_matrix)
    ga.group_quantize(x, q, 2, split, tensor_offsets=offsets)
    s = ga.precomputed_group_amax_stats()
    assert s.get("used", 0) == 0
    assert s["ignored_rht_matrix"] == 1 and s["ignored_num_groups"] == 1 and s["pending"] == 0
    # a different tensor never matches
    ga.set_precomputed_group_amax(x, amax, q.rht_matrix)
    y = x.clone()
    ga.group_quantize(y, q, 2, split, tensor_offsets=offsets)
    assert ga.precomputed_group_amax_stats()["pending"] == 1
    ga.clear_precomputed_group_amax(reset_stats=True)


def test_eviction():
    ga.clear_precomputed_group_amax(reset_stats=True)
    q = _quantizer(False)
    tensors = [torch.zeros(128, 256, device="cuda", dtype=torch.bfloat16) for _ in range(ga.MAX_PENDING + 2)]
    for t in tensors:
        ga.set_precomputed_group_amax(t, torch.zeros(2, 1, device="cuda"), q.rht_matrix)
    s = ga.precomputed_group_amax_stats()
    assert s["evicted"] == 2 and s["pending"] == ga.MAX_PENDING
    ga.clear_precomputed_group_amax(reset_stats=True)


def test_graph_capture_bitwise():
    counts, hidden = [256, 0, 128, 384], 512
    x, split, offsets = _inputs(counts, hidden)
    q = _quantizer(True)
    amax_src = _te_amax(x, q, 4, split, offsets)
    amax = torch.zeros_like(amax_src)
    ga.clear_precomputed_group_amax(reset_stats=True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ga.set_precomputed_group_amax(x, amax, q.rht_matrix)
        ga.group_quantize(x, q, 4, split, tensor_offsets=offsets)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ga.set_precomputed_group_amax(x, amax, q.rht_matrix)
        out = ga.group_quantize(x, q, 4, split, tensor_offsets=offsets)
    assert ga.precomputed_group_amax_stats()["used"] == 2
    amax.copy_(amax_src)
    g.replay()
    torch.cuda.synchronize()
    # SR draws a Philox offset per call, so compare the amaxes here and the data with an RN quantizer.
    assert torch.equal(out.amax, amax_src[0]) and torch.equal(out.columnwise_amax, amax_src[1])
    q_rn = _quantizer(False)
    ref = tex.group_quantize(x, q_rn, 4, split, tensor_offsets=offsets)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        ga.set_precomputed_group_amax(x, amax, q_rn.rht_matrix)
        out2 = ga.group_quantize(x, q_rn, 4, split, tensor_offsets=offsets)
    g2.replay()
    torch.cuda.synchronize()
    a, r = _written(out2, sum(counts), hidden), _written(ref, sum(counts), hidden)
    for name in a:
        if a[name] is not None:
            assert torch.equal(a[name], r[name]), name
    ga.clear_precomputed_group_amax(reset_stats=True)
