# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVTE_CUTEDSL_FUSED_ATTN_BWD: the CuTe DSL FP8 attention backward against cuDNN's, through DotProductAttention."""

import os

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling
from transformer_engine.pytorch.attention.dot_product_attention import cutedsl_attn_bwd

try:
    from cudnn.sdpa.bwd import sdpa_bwd_fp8_gqa_d64  # noqa: F401

    HAVE_KERNEL = True
except Exception:  # pylint: disable=broad-except
    HAVE_KERNEL = False

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10 and HAVE_KERNEL),
    reason="needs an SM10x GPU and cudnn.sdpa.bwd.sdpa_bwd_fp8_gqa_d64",
)

S, B, H, HG, D = 2048, 2, 64, 8, 64


def _run(monkeypatch, enabled, window, softmax_type, seed=0):
    monkeypatch.setenv("NVTE_DPA_FP8CS_O_in_F16", "1")
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_ATTN_BWD", "1" if enabled else "0")
    torch.manual_seed(seed)
    dpa = te.DotProductAttention(
        H, D, num_gqa_groups=HG, attn_mask_type="causal", qkv_format="sbhd", softmax_type=softmax_type,
        window_size=(window, 0),
    ).cuda().bfloat16()
    if softmax_type == "learnable":
        with torch.no_grad():
            dpa.softmax_offset.copy_(torch.randn_like(dpa.softmax_offset) * 0.5)
    q = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(S, B, HG, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(S, B, HG, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    before = cutedsl_attn_bwd._STATE["routed"]
    with te.autocast(enabled=True, recipe=Float8CurrentScaling(fp8_dpa=True)):
        o = dpa(q, k, v)
    o.backward(torch.randn_like(o) * 0.02)
    torch.cuda.synchronize()
    grads = [q.grad, k.grad, v.grad]
    if softmax_type == "learnable":
        grads.append(dpa.softmax_offset.grad)
    return grads, cutedsl_attn_bwd._STATE["routed"] - before


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("window", [-1, 127])
@pytest.mark.parametrize("softmax_type", ["vanilla", "learnable"])
def test_matches_cudnn(monkeypatch, window, softmax_type):
    ref, routed_off = _run(monkeypatch, False, window, softmax_type)
    out, routed_on = _run(monkeypatch, True, window, softmax_type)
    assert routed_off == 0
    assert routed_on == 1, "the CuTe backward was not used"
    for name, a, b in zip(("dq", "dk", "dv", "dsink"), out, ref):
        assert torch.isfinite(a.float()).all(), name
        assert _rel(a, b) < 0.02, f"{name}: rel L2 {_rel(a, b):.4f} between CuTe and cuDNN"


def test_default_off_keeps_cudnn(monkeypatch):
    monkeypatch.delenv("NVTE_CUTEDSL_FUSED_ATTN_BWD", raising=False)
    assert not cutedsl_attn_bwd.enabled()


def test_ineligible_falls_back(monkeypatch):
    """A bf16 (non-FP8) attention backward never reaches the kernel."""
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_ATTN_BWD", "1")
    dpa = te.DotProductAttention(H, D, num_gqa_groups=HG, attn_mask_type="causal", qkv_format="sbhd").cuda().bfloat16()
    q = torch.randn(256, 1, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(256, 1, HG, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(256, 1, HG, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    before = cutedsl_attn_bwd._STATE["routed"]
    dpa(q, k, v).sum().backward()
    assert cutedsl_attn_bwd._STATE["routed"] == before
