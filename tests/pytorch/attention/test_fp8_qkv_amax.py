# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 current-scaling Q/K/V quantization with a precomputed amax (``qkv_amax``).

``combine_and_quantize(..., qkv_amax=...)`` and ``DotProductAttention(..., qkv_amax=...)`` quantize
separate Q, K and V with the scale derived from an amax computed by their producer, instead of
concatenating them so that one amax pass covers all three. The FP8 bytes and scales must be
bitwise equal to the concatenated quantization.
"""

import os

import pytest
import torch

import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    DotProductAttention,
    autocast,
    get_device_compute_capability,
    is_fp8_available,
    is_mxfp8_available,
)
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    Float8Tensor,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
fp8_attn_available, reason_for_no_fp8_attn = fp8_available, reason_for_no_fp8
_cc = get_device_compute_capability()
if fp8_available and (_cc < (9, 0) or _cc >= (12, 0)):
    fp8_attn_available = False
    reason_for_no_fp8_attn = f"FP8 attention is not supported for sm{_cc[0] * 10 + _cc[1]}"


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def _reference_amax(tensors):
    """What the current-scaling quantizer computes over the concatenation (TE amax kernel)."""
    amax = torch.zeros(1, dtype=torch.float32, device=tensors[0].device)
    tex.compute_amax(torch.cat([x.reshape(-1) for x in tensors]), amax)
    return amax


def _torch_amax(tensors):
    """max |x| over all elements with NaN ignored, computed with torch."""
    vals = [torch.where(torch.isnan(x), 0.0, x.float().abs()).amax() for x in tensors]
    return torch.stack(vals).amax().reshape(1)


def _make_qkv(s, b, h, hkv, d, kind, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (
        torch.randn((s, b, n, d), generator=gen, device="cuda", dtype=torch.float32)
        for n in (h, hkv, hkv)
    )
    if kind == "zeros":
        q, k, v = (torch.zeros_like(x) for x in (q, k, v))
    elif kind == "tiny":
        q, k, v = (x * 1e-30 for x in (q, k, v))
    elif kind == "large":
        v = v * 1e30
    elif kind == "v_max":
        # the amax sits in v, the last tensor of the concatenation
        v = v * 100.0
    q, k, v = (x.to(dtype) for x in (q, k, v))
    if kind == "negzero":
        q[0] = -0.0
        k[-1] = -0.0
        v.view(-1)[::7] = -0.0
    elif kind == "inf":
        k.view(-1)[5] = float("inf")
    elif kind == "neginf":
        q.view(-1)[17] = float("-inf")
    elif kind == "nan":
        q.view(-1)[3] = float("nan")
        v.view(-1)[-1] = float("nan")
    return q, k, v


def _assert_fp8_equal(a, b, name):
    assert isinstance(a, Float8Tensor) and isinstance(b, Float8Tensor), name
    assert a.shape == b.shape and a.dtype == b.dtype, name
    assert a._fp8_dtype == b._fp8_dtype, name
    qa, qb = a._quantizer, b._quantizer
    assert type(qa) is type(qb), name
    for attr in (
        "dtype",
        "rowwise_usage",
        "columnwise_usage",
        "force_pow_2_scales",
        "amax_epsilon",
    ):
        assert getattr(qa, attr) == getattr(qb, attr), (name, attr)
    assert a._transpose is None and b._transpose is None, name
    assert a._data.shape == b._data.shape and a._data.stride() == b._data.stride(), name
    assert torch.equal(a._data, b._data), f"{name}: FP8 bytes differ"
    assert torch.equal(
        a._scale_inv.view(torch.int32), b._scale_inv.view(torch.int32)
    ), f"{name}: scale_inv differs ({a._scale_inv.item()} vs {b._scale_inv.item()})"


def _cs_quantizer(fp8_dtype, pow2, eps):
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype, device="cuda", force_pow_2_scales=pow2, amax_epsilon=eps
    )
    # as get_attention_quantizers configures the QKV quantizer
    quantizer.internal = False
    quantizer.set_usage(rowwise=True, columnwise=False)
    return quantizer


_SHAPES = {
    # s, b, h, hkv, d
    "gqa_512": (512, 2, 8, 2, 64),
    "mha_odd": (130, 3, 4, 4, 32),
    "gpt_oss": (8192, 3, 64, 8, 64),
    "mqa_tiny": (1, 1, 2, 1, 16),
}
_KINDS = ["randn", "zeros", "tiny", "large", "v_max", "negzero", "inf", "neginf", "nan"]


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("pow2_eps", [(False, 0.0), (True, 0.0), (False, 1e-3)])
@pytest.mark.parametrize("used_in_backward", [False, True])
def test_combine_and_quantize_qkv_amax(shape, kind, fp8_dtype, pow2_eps, used_in_backward):
    """Bitwise equal to the concatenated quantization: FP8 bytes, scale_inv, metadata, layout."""
    if shape == "gpt_oss" and (fp8_dtype != tex.DType.kFloat8E4M3 or pow2_eps != (False, 0.0)):
        pytest.skip("large shape: submission dtype and scaling options only")
    s, b, h, hkv, d = _SHAPES[shape]
    q, k, v = _make_qkv(s, b, h, hkv, d, kind)
    amax = _reference_amax([q, k, v])
    # The contract: max |x| with NaN ignored. The TE amax kernel and torch agree on it.
    assert torch.equal(amax, _torch_amax([q, k, v])), (amax, _torch_amax([q, k, v]))
    quantizer = _cs_quantizer(fp8_dtype, *pow2_eps)

    ref = dpa_utils.combine_and_quantize(
        "sbhd_sbhd_sbhd", q, k, v, quantizer, used_in_backward=used_in_backward
    )
    out = dpa_utils.combine_and_quantize(
        "sbhd_sbhd_sbhd", q, k, v, quantizer, used_in_backward=used_in_backward, qkv_amax=amax
    )
    assert out[3] == ref[3] == "sbhd_sbhd_sbhd" and out[4] is None and ref[4] is None
    for name, a, r in zip("qkv", out[:3], ref[:3]):
        _assert_fp8_equal(a, r, name)
    # one shared scale_inv and one uint8 buffer laid out as the concatenation, as in the cat path
    q8, k8, v8 = out[:3]
    assert q8._scale_inv is k8._scale_inv is v8._scale_inv
    base = q8._data.data_ptr()
    assert (
        k8._data.data_ptr() == base + q.numel()
        and v8._data.data_ptr() == base + q.numel() + k.numel()
    )
    # the input amax is only read
    assert torch.equal(amax, _reference_amax([q, k, v]))


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_quantize_with_qkv_amax_uses_given_amax():
    """The scale comes from qkv_amax, not from the data (the producer owns the amax)."""
    q, k, v = _make_qkv(256, 2, 4, 2, 64, "randn")
    quantizer = _cs_quantizer(tex.DType.kFloat8E4M3, False, 0.0)
    amax = _reference_amax([q, k, v])
    out = dpa_utils.combine_and_quantize("sbhd_sbhd_sbhd", q, k, v, quantizer, qkv_amax=amax * 2)
    ref = dpa_utils.combine_and_quantize("sbhd_sbhd_sbhd", q, k, v, quantizer, qkv_amax=amax)
    assert torch.equal(out[0]._scale_inv, ref[0]._scale_inv * 2)
    assert not torch.equal(out[0]._data, ref[0]._data)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_combine_and_quantize_qkv_amax_ignored_otherwise():
    """qkv_amax is ignored (bitwise the path without it) where it does not apply."""
    s, b, h, d = 256, 2, 4, 64
    q, k, v = _make_qkv(s, b, h, h, d, "randn")
    wrong_amax = torch.full((1,), 1234.5, dtype=torch.float32, device="cuda")

    # delayed scaling quantizer: the scale is the quantizer's, not derived from an amax
    ds = Float8Quantizer(
        scale=torch.full((1,), 3.0, device="cuda"),
        amax=torch.zeros(1, device="cuda"),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    a = dpa_utils.combine_and_quantize("sbhd_sbhd_sbhd", q, k, v, ds, qkv_amax=wrong_amax)
    r = dpa_utils.combine_and_quantize("sbhd_sbhd_sbhd", q, k, v, ds)
    for x, y in zip(a[:3], r[:3]):
        assert torch.equal(x._data, y._data) and torch.equal(x._scale_inv, y._scale_inv)

    # current scaling with amax reduction: the amax must be reduced first
    cs = _cs_quantizer(tex.DType.kFloat8E4M3, False, 0.0)
    cs.with_amax_reduction = True
    assert not dpa_utils._qkv_amax_usable(cs, [q, k, v])

    # current scaling with columnwise usage
    cs = _cs_quantizer(tex.DType.kFloat8E4M3, False, 0.0)
    cs.set_usage(rowwise=True, columnwise=True)
    assert not dpa_utils._qkv_amax_usable(cs, [q, k, v])

    # packed QKV (qkv_group 1): one buffer, quantized directly
    cs = _cs_quantizer(tex.DType.kFloat8E4M3, False, 0.0)
    qkv = torch.stack([q, k, v], dim=2)
    qp, kp, vp = qkv.unbind(2)
    a = dpa_utils.combine_and_quantize(
        "sb3hd", qp, kp, vp, cs, combined_qkv=qkv, qkv_amax=wrong_amax
    )
    r = dpa_utils.combine_and_quantize("sb3hd", qp, kp, vp, cs, combined_qkv=qkv)
    for x, y in zip(a[:3], r[:3]):
        assert torch.equal(x._data, y._data) and torch.equal(x._scale_inv, y._scale_inv)

    # non-contiguous separate tensors: not usable
    assert not dpa_utils._qkv_amax_usable(
        _cs_quantizer(tex.DType.kFloat8E4M3, False, 0.0), [q.transpose(0, 1), k, v]
    )

    # MXFP8: per-block scales, no per-tensor amax
    if mxfp8_available:
        mx = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        a = dpa_utils.combine_and_quantize("sbhd_sbhd_sbhd", q, k, v, mx, qkv_amax=wrong_amax)
        r = dpa_utils.combine_and_quantize("sbhd_sbhd_sbhd", q, k, v, mx)
        for x, y in zip(a[:3], r[:3]):
            for attr in (
                "_rowwise_data",
                "_rowwise_scale_inv",
                "_columnwise_data",
                "_columnwise_scale_inv",
            ):
                tx, ty = getattr(x, attr), getattr(y, attr)
                assert (tx is None) == (ty is None), attr
                assert tx is None or torch.equal(tx, ty), attr


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(1, dtype=torch.bfloat16),
        torch.ones(2),
        torch.ones(()),
        "cpu",
        1.0,
    ],
)
def test_qkv_amax_validation(bad):
    q, k, v = _make_qkv(128, 1, 2, 2, 32, "randn")
    if isinstance(bad, str):
        bad = torch.ones(1)
    elif isinstance(bad, torch.Tensor) and bad.numel() == 1 and bad.dim() == 0:
        bad = bad.cuda()  # 0-dim is fine as long as dtype/device are right
        out = dpa_utils.combine_and_quantize(
            "sbhd_sbhd_sbhd",
            q,
            k,
            v,
            _cs_quantizer(tex.DType.kFloat8E4M3, False, 0.0),
            qkv_amax=bad,
        )
        assert isinstance(out[0], Float8Tensor)
        return
    elif isinstance(bad, torch.Tensor):
        bad = bad.cuda()
    with pytest.raises(ValueError, match="qkv_amax"):
        dpa_utils.combine_and_quantize(
            "sbhd_sbhd_sbhd",
            q,
            k,
            v,
            _cs_quantizer(tex.DType.kFloat8E4M3, False, 0.0),
            qkv_amax=bad,
        )


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_quantize_with_qkv_amax_cuda_graph():
    """Capturable: amax produced and consumed inside a CUDA graph, replayed with new data."""
    s, b, h, hkv, d = 512, 2, 8, 2, 64
    q, k, v = _make_qkv(s, b, h, hkv, d, "randn")
    quantizer = _cs_quantizer(tex.DType.kFloat8E4M3, False, 0.0)

    def produce_and_quantize():
        amax = torch.zeros(1, dtype=torch.float32, device="cuda")
        tex.compute_amax(torch.cat([q.view(-1), k.view(-1), v.view(-1)]), amax)
        return dpa_utils.combine_and_quantize("sbhd_sbhd_sbhd", q, k, v, quantizer, qkv_amax=amax)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            produce_and_quantize()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = produce_and_quantize()

    for scale in (1.0, 0.25, 7.0, 0.0):
        gen = torch.Generator(device="cuda").manual_seed(int(scale * 100) + 1)
        for x in (q, k, v):
            x.copy_(torch.randn(x.shape, generator=gen, device="cuda") * scale)
        graph.replay()
        ref = dpa_utils.combine_and_quantize("sbhd_sbhd_sbhd", q, k, v, quantizer)
        for name, a, r in zip("qkv", static_out[:3], ref[:3]):
            assert torch.equal(a._data, r._data), (scale, name)
            assert torch.equal(a._scale_inv, r._scale_inv), (scale, name)


def _set_fused_attention_backend(monkeypatch):
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    _attention_backends["backend_selection_requires_update"] = True


@pytest.mark.skipif(not fp8_attn_available, reason=reason_for_no_fp8_attn)
@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("window_size", [None, (128, 0)])
def test_dpa_fp8_current_scaling_qkv_amax(monkeypatch, is_training, window_size):
    """DotProductAttention: qkv_amax gives bitwise the same output and Q/K/V gradients."""
    _set_fused_attention_backend(monkeypatch)
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    calls = []
    real = dpa_utils.quantize_with_qkv_amax

    def spy(tensors, quantizer, amax):
        calls.append(amax)
        return real(tensors, quantizer, amax)

    monkeypatch.setattr(dpa_utils, "quantize_with_qkv_amax", spy)

    s, b, h, hkv, d = 512, 2, 8, 2, 64
    fp8_recipe = recipe.Float8CurrentScaling(fp8_format=recipe.Format.HYBRID, fp8_dpa=True)
    q0, k0, v0 = _make_qkv(s, b, h, hkv, d, "randn", seed=3)
    gen = torch.Generator(device="cuda").manual_seed(4)
    dout = torch.randn((s, b, h * d), generator=gen, device="cuda").to(torch.bfloat16)

    def run(with_amax):
        # A fresh module and FP8 state per run: the S/dP tensors of FP8 attention use delayed scaling, whose
        # scales move after every call, so only first calls are comparable.
        FP8GlobalStateManager.reset()
        dpa = DotProductAttention(
            h,
            d,
            num_gqa_groups=hkv,
            attention_dropout=0.0,
            qkv_format="sbhd",
            attn_mask_type="causal",
            window_size=window_size,
        ).cuda()
        if not is_training:
            dpa.eval()
        q, k, v = (x.detach().clone().requires_grad_(is_training) for x in (q0, k0, v0))
        amax = _reference_amax([q0, k0, v0]) if with_amax else None
        with autocast(enabled=True, recipe=fp8_recipe):
            out = dpa(q, k, v, qkv_amax=amax)
        assert dpa.fp8
        if not is_training:
            return out, None
        out.backward(dout)
        return out, (q.grad, k.grad, v.grad)

    out_ref, grads_ref = run(False)
    assert not calls
    out, grads = run(True)
    assert len(calls) == 1, "quantize_with_qkv_amax was not used"
    assert torch.equal(out, out_ref)
    if is_training:
        for name, g, r in zip(("dq", "dk", "dv"), grads, grads_ref):
            assert torch.equal(g, r), name
    # repeat: identical again
    out2, grads2 = run(True)
    assert len(calls) == 2 and torch.equal(out2, out_ref)
    if is_training:
        for g, r in zip(grads2, grads_ref):
            assert torch.equal(g, r)
