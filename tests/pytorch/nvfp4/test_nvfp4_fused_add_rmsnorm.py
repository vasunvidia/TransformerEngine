# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused residual add + RMSNorm + NVFP4 RHT amax: bitwise equality with the unfused path."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import fused_add_rmsnorm as far
from transformer_engine.pytorch import precomputed_rht_amax
from transformer_engine.pytorch.fused_add_rmsnorm import fused_add_rmsnorm_nvfp4
from transformer_engine.pytorch.module._common import apply_normalization
from transformer_engine.pytorch.quantization import RecipeState
from transformer_engine.pytorch.tensor.nvfp4_tensor import get_random_sign_mask_for_rht

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)
sm100_available = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)
pytestmark = [
    pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe),
    pytest.mark.skipif(not sm100_available, reason="Requires compute capability 10.0+"),
]

HIDDEN = 4096
EPS = 1e-5
FIELDS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
    "_amax_rowwise",
    "_amax_columnwise",
)


@pytest.fixture(autouse=True)
def enable_fusion(monkeypatch):
    monkeypatch.setenv("NVTE_FUSED_ADD_RMSNORM_NVFP4", "1")
    far._pending = None
    far._last_split = None
    yield
    far._pending = None
    far._last_split = None


def recipe_input_quantizer(swizzled: bool = True, columnwise: bool = True):
    """Forward input quantizer as the NVFP4BlockScaling recipe builds it."""
    state = RecipeState.create(NVFP4BlockScaling(), mode="forward", num_quantizers=3)
    quantizer = state.make_quantizers()[0]
    quantizer.set_usage(rowwise=True, columnwise=columnwise)
    quantizer.internal = True
    quantizer.optimize_for_gemm = swizzled
    return quantizer


def make_inputs(kind: str, shape, seed: int = 1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    randn = lambda *size: torch.randn(*size, device="cuda", generator=gen)
    if kind == "randn":
        x, residual = randn(shape), randn(shape)
    elif kind == "rowscale":
        scale = torch.exp(2.0 * randn(*shape[:-1], 1))
        x, residual = randn(shape) * scale, randn(shape) * scale * 4
    elif kind == "outliers":
        x, residual = randn(shape) * 0.05, randn(shape) * 0.3
        hot = torch.randint(0, shape[-1], (8,), device="cuda", generator=gen)
        residual[..., hot] *= 300.0
        x.view(-1)[torch.randint(0, x.numel(), (1000,), device="cuda", generator=gen)] = 1e4
    elif kind == "tiny":
        x, residual = randn(shape) * 1e-30, randn(shape) * 1e-31
    elif kind == "zero_rows":
        x = randn(shape)
        residual = -x.clone()
        residual.view(-1, shape[-1])[::3] += 1.0
    else:
        raise ValueError(kind)
    weight = (1.0 + 0.2 * randn(shape[-1])).to(torch.bfloat16)
    return x.to(torch.bfloat16), residual.to(torch.bfloat16), weight


def assert_bitwise_equal(a, b, name):
    if a is None or b is None:
        assert a is b, name
        return
    assert a.shape == b.shape and a.dtype == b.dtype, name
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), name


def assert_nvfp4_equal(out, ref):
    assert type(out) is type(ref)
    assert out.shape == ref.shape
    assert out._with_gemm_swizzled_scales == ref._with_gemm_swizzled_scales
    rows, cols = out.shape.numel() // out.shape[-1], out.shape[-1]
    for field in FIELDS:
        a, b = getattr(out, field, None), getattr(ref, field, None)
        if field.endswith("scale_inv") and a is not None and not out._with_gemm_swizzled_scales:
            # Compact scales are allocated with padding that no kernel writes; compare the
            # scales of the tensor only
            valid = (rows, cols // 16) if field.startswith("_rowwise") else (cols, rows // 16)
            a, b = a[: valid[0], : valid[1]], b[: valid[0], : valid[1]]
        assert_bitwise_equal(a, b, field)


def unfused_rmsnorm(s, weight, quantizer):
    return tex.rmsnorm_fwd(s, weight, EPS, None, quantizer, tex.DType.kBFloat16, 0, False)


@pytest.mark.parametrize(
    "shape", [(16384, HIDDEN), (8192, 2, HIDDEN), (16, HIDDEN), (1040, 1, HIDDEN), (48, 2, HIDDEN)]
)
@pytest.mark.parametrize("kind", ["randn", "rowscale", "outliers", "tiny", "zero_rows"])
def test_kernel_matches_unfused(shape, kind):
    """Sum, RMSNorm output, rsigma and both amaxes, and the cast with the precomputed amaxes."""
    x, residual, weight = make_inputs(kind, shape)
    s_ref = residual + x
    z_ref, _, rsigma_ref = unfused_rmsnorm(s_ref, weight, None)
    out_ref, _, rsigma_q_ref = unfused_rmsnorm(s_ref, weight, recipe_input_quantizer())

    mask = get_random_sign_mask_for_rht(True, torch.cuda.current_device())
    for _ in range(2):  # the second launch reuses the workspace the first one left zero-filled
        s, z, rsigma, amax = tex.rmsnorm_fwd_add_rht_amax(x, residual, weight, EPS, mask, 0)
        assert_bitwise_equal(s, s_ref, "sum")
        assert_bitwise_equal(z, z_ref, "z")
        assert_bitwise_equal(rsigma, rsigma_ref, "rsigma")
        assert_bitwise_equal(rsigma, rsigma_q_ref, "rsigma (quantized norm)")
        assert_bitwise_equal(amax[0:1], out_ref._amax_rowwise, "amax rowwise")
        assert_bitwise_equal(amax[1:2], out_ref._amax_columnwise, "amax columnwise")
        out = tex.nvfp4_quantize_with_amax(z, recipe_input_quantizer(), amax[0:1], amax[1:2])
        assert_nvfp4_equal(out, out_ref)


def test_kernel_without_add():
    """With x=None the kernel normalizes residual and writes no sum."""
    _, residual, weight = make_inputs("rowscale", (4096, HIDDEN))
    z_ref, _, rsigma_ref = unfused_rmsnorm(residual, weight, None)
    out_ref, _, _ = unfused_rmsnorm(residual, weight, recipe_input_quantizer())
    mask = get_random_sign_mask_for_rht(True, torch.cuda.current_device())
    s, z, rsigma, amax = tex.rmsnorm_fwd_add_rht_amax(None, residual, weight, EPS, mask, 0)
    assert s is None
    assert_bitwise_equal(z, z_ref, "z")
    assert_bitwise_equal(rsigma, rsigma_ref, "rsigma")
    assert_bitwise_equal(amax[0:1], out_ref._amax_rowwise, "amax rowwise")
    assert_bitwise_equal(amax[1:2], out_ref._amax_columnwise, "amax columnwise")


def test_kernel_cuda_graph():
    """The kernel captures in a CUDA graph and replays without a workspace memset."""
    x, residual, weight = make_inputs("randn", (2048, HIDDEN))
    mask = get_random_sign_mask_for_rht(True, torch.cuda.current_device())
    _, _, _, amax_ref = tex.rmsnorm_fwd_add_rht_amax(x, residual, weight, EPS, mask, 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tex.rmsnorm_fwd_add_rht_amax(x, residual, weight, EPS, mask, 0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, _, _, amax = tex.rmsnorm_fwd_add_rht_amax(x, residual, weight, EPS, mask, 0)
    for _ in range(3):
        amax.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise_equal(amax, amax_ref, "amax")


@pytest.mark.parametrize("swizzled", [True, False])
@pytest.mark.parametrize("columnwise", [True, False])
@pytest.mark.parametrize("shape", [(8192, 2, HIDDEN), (48, 1, HIDDEN)])
def test_precomputed_rmsnorm(swizzled, columnwise, shape):
    """apply_normalization on the sum of a fused add returns the unfused result."""
    x, residual, weight = make_inputs("rowscale", shape)
    s_ref = residual + x
    out_ref, _, rsigma_ref = unfused_rmsnorm(
        s_ref.view(-1, HIDDEN), weight, recipe_input_quantizer(swizzled, columnwise)
    )
    s = fused_add_rmsnorm_nvfp4(x, residual, weight, EPS)
    assert s is not None
    assert_bitwise_equal(s, s_ref, "sum")
    out, mu, rsigma = apply_normalization(
        s.view(-1, HIDDEN),
        None,
        weight,
        None,
        EPS,
        recipe_input_quantizer(swizzled, columnwise),
        torch.bfloat16,
        "RMSNorm",
        0,
        False,
    )
    assert far._pending is None
    assert mu is None
    assert_bitwise_equal(rsigma, rsigma_ref, "rsigma")
    assert_nvfp4_equal(out, out_ref)


def test_mismatched_consumer_falls_back():
    """A consumer with another weight runs the unfused RMSNorm and drops the precomputed result."""
    x, residual, weight = make_inputs("randn", (256, HIDDEN))
    other_weight = weight.clone()
    s = fused_add_rmsnorm_nvfp4(x, residual, weight, EPS)
    quantizer = recipe_input_quantizer()
    assert (
        far.take_precomputed_rmsnorm(
            s, other_weight, EPS, None, quantizer, tex.DType.kBFloat16, False
        )
        is None
    )
    assert far._pending is None

    # A stale entry (sum modified in place) is not used either
    s = fused_add_rmsnorm_nvfp4(x, residual, weight, EPS)
    s.add_(1.0)
    assert (
        far.take_precomputed_rmsnorm(s, weight, EPS, None, quantizer, tex.DType.kBFloat16, False)
        is None
    )


def test_unsupported_inputs(monkeypatch):
    x, residual, weight = make_inputs("randn", (256, HIDDEN))
    assert fused_add_rmsnorm_nvfp4(x[:250], residual[:250], weight, EPS) is None  # rows % 16
    assert fused_add_rmsnorm_nvfp4(x.float(), residual.float(), weight, EPS) is None
    x2, residual2, weight2 = (t[..., :2048].contiguous() for t in (x, residual, weight))
    assert fused_add_rmsnorm_nvfp4(x2, residual2, weight2, EPS) is None  # hidden size
    monkeypatch.setenv("NVTE_FUSED_ADD_RMSNORM_NVFP4", "0")
    assert fused_add_rmsnorm_nvfp4(x, residual, weight, EPS) is None
    assert far._pending is None


@pytest.mark.parametrize("value", ["", "0", "1"])
def test_cudnn_norm_fwd_falls_back(value, monkeypatch):
    """With the cuDNN RMSNorm forward selected, the fused path is not used."""
    x, residual, weight = make_inputs("randn", (256, HIDDEN))
    monkeypatch.setenv("NVTE_NORM_FWD_USE_CUDNN", value)
    s = fused_add_rmsnorm_nvfp4(x, residual, weight, EPS)
    if value == "1":
        assert s is None
        assert far._pending is None
    else:
        assert s is not None
        assert far._pending is not None


@pytest.mark.parametrize("smem_optin", [99 * 1024, far._KERNEL_SMEM_BYTES - 1])
def test_insufficient_shared_memory_falls_back(smem_optin, monkeypatch):
    """A device with too little opt-in shared memory per block for the kernel falls back."""
    x, residual, weight = make_inputs("randn", (256, HIDDEN))
    real_props = torch.cuda.get_device_properties(x.device)

    class Props:
        major, minor = real_props.major, real_props.minor
        shared_memory_per_block_optin = smem_optin

    monkeypatch.setattr(far, "_supported_devices", {})
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args, **kwargs: Props())
    assert fused_add_rmsnorm_nvfp4(x, residual, weight, EPS) is None
    assert far._pending is None

    Props.shared_memory_per_block_optin = far._KERNEL_SMEM_BYTES
    monkeypatch.setattr(far, "_supported_devices", {})
    assert fused_add_rmsnorm_nvfp4(x, residual, weight, EPS) is not None


def test_backward_launches_no_fill():
    """The backward of the fused add only passes the gradient on: no zero-filled gradients for the
    non-differentiable RMSNorm outputs."""
    x, residual, weight = make_inputs("randn", (4096, HIDDEN))
    x.requires_grad_(True)
    residual.requires_grad_(True)
    grad = torch.randn_like(x)
    s = fused_add_rmsnorm_nvfp4(x, residual, weight, EPS)
    far._pending = None
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        s.backward(grad)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert not [n for n in names if "Memset" in n or "Fill" in n], names
    assert_bitwise_equal(x.grad, grad, "x grad")
    assert_bitwise_equal(residual.grad, grad, "residual grad")


def _run_layer(model, weight_of, x0, residual0, grad, fused, seed=1234):
    for param in model.parameters():
        param.grad = None
    x = x0.detach().clone().requires_grad_(True)
    residual = residual0.detach().clone().requires_grad_(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with te.autocast(enabled=True, recipe=NVFP4BlockScaling()):
        s = fused_add_rmsnorm_nvfp4(x, residual, weight_of(model), EPS) if fused else None
        if s is None:
            s = residual + x
        y = model(s)
    y.backward(grad)
    torch.cuda.synchronize()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    return y.detach(), x.grad, residual.grad, grads


@pytest.mark.parametrize("module", ["LayerNormLinear", "ops.RMSNorm"])
def test_layer_fused_vs_unfused(module, monkeypatch):
    """Forward output and all gradients of an RMSNorm + NVFP4 linear are bitwise unchanged."""
    if module == "LayerNormLinear":
        model = te.LayerNormLinear(
            HIDDEN, 512, eps=EPS, normalization="RMSNorm", bias=False, params_dtype=torch.bfloat16
        )
        weight_of = lambda m: m.layer_norm_weight
    else:
        model = te.ops.Sequential(
            te.ops.RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16),
            te.ops.Linear(HIDDEN, 512, bias=False, dtype=torch.bfloat16),
        )
        weight_of = lambda m: m[0].weight
    with torch.no_grad():
        weight_of(model).copy_(1.0 + 0.1 * torch.randn(HIDDEN, device="cuda"))
    x0, residual0, _ = make_inputs("randn", (2048, 2, HIDDEN), seed=5)
    grad = (torch.randn(2048, 2, 512, device="cuda") * 1e-2).to(torch.bfloat16)

    calls = []
    cast = tex.nvfp4_quantize_with_amax

    def counting_cast(*args):
        calls.append(1)
        return cast(*args)

    monkeypatch.setattr(tex, "nvfp4_quantize_with_amax", counting_cast)
    unfused = _run_layer(model, weight_of, x0, residual0, grad, fused=False)
    assert not calls
    fused = _run_layer(model, weight_of, x0, residual0, grad, fused=True)
    assert len(calls) == 1, "the RMSNorm did not use the precomputed result"
    assert far._pending is None
    assert_bitwise_equal(fused[0], unfused[0], "output")
    assert_bitwise_equal(fused[1], unfused[1], "x grad")
    assert_bitwise_equal(fused[2], unfused[2], "residual grad")
    for name in unfused[3]:
        assert_bitwise_equal(fused[3][name], unfused[3][name], name)


# Producers that emit the NVFP4 RHT amaxes of their output (precomputed_rht_amax)


def recipe_grad_quantizer(stochastic_rounding: bool = False):
    """Backward (grad output) quantizer as the NVFP4BlockScaling recipe builds it."""
    state = RecipeState.create(NVFP4BlockScaling(), mode="backward", num_quantizers=2)
    quantizer = state.make_quantizers()[0]
    quantizer.set_usage(rowwise=True, columnwise=True)
    quantizer.internal = True
    quantizer.optimize_for_gemm = True
    quantizer.stochastic_rounding = stochastic_rounding
    return quantizer


def hadamard_amaxes(t, quantizer):
    """Both amaxes as the NVFP4 cast computes them."""
    out = tex.quantize(t, quantizer)
    return out._amax_rowwise, out._amax_columnwise


@pytest.fixture
def clean_registry():
    precomputed_rht_amax._clear()
    yield
    precomputed_rht_amax._clear()


@pytest.mark.parametrize("shape", [(16384, HIDDEN), (8192, 2, HIDDEN), (64, 128), (192, 3, 256)])
@pytest.mark.parametrize("kind", ["randn", "outliers", "tiny"])
def test_add_hadamard_transform_amax(shape, kind, clean_registry):
    """The sum is torch's BF16 add and the amaxes are those of the NVFP4 RHT cast."""
    a, b, _ = make_inputs(kind, shape)
    quantizer = recipe_grad_quantizer()
    mask = int(quantizer.rht_matrix_random_sign_mask_t)
    ref_row, ref_col = hadamard_amaxes(a + b, quantizer)
    for _ in range(2):  # the second launch reuses the workspace the first one left zero-filled
        out, amax = tex.add_hadamard_transform_amax(a, b, mask)
        assert_bitwise_equal(out, a + b, "sum")
        assert_bitwise_equal(amax[0:1], ref_row, "amax rowwise")
        assert_bitwise_equal(amax[1:2], ref_col, "amax columnwise")


def test_add_with_rht_amax_registers_for_cast(clean_registry):
    """The cast of the sum takes the registered amaxes (once) and equals the unfused cast."""
    a, b, _ = make_inputs("rowscale", (4096, HIDDEN))
    quantizer = recipe_grad_quantizer()
    ref = quantizer(a + b)
    out = precomputed_rht_amax.add_with_rht_amax(a, b)
    calls = []
    cast = tex.nvfp4_quantize_with_amax
    try:
        tex.nvfp4_quantize_with_amax = lambda *args: calls.append(1) or cast(*args)
        assert_nvfp4_equal(quantizer(out), ref)
        assert calls == [1]
        assert_nvfp4_equal(quantizer(out), ref)  # taken once: computes the amaxes itself
        assert calls == [1]
    finally:
        tex.nvfp4_quantize_with_amax = cast
    # A tensor modified after its producer ran does not use the stale amaxes
    out = precomputed_rht_amax.add_with_rht_amax(a, b)
    out.mul_(2.0)
    assert precomputed_rht_amax.take_rht_amax(out, quantizer) is None
    # Unsupported shapes fall back to a + b with nothing registered
    out = precomputed_rht_amax.add_with_rht_amax(a[:100], b[:100])
    assert_bitwise_equal(out, a[:100] + b[:100], "fallback sum")
    assert precomputed_rht_amax.take_rht_amax(out, quantizer) is None


def test_add_hadamard_transform_amax_cuda_graph():
    """The add captures in a CUDA graph and replays without a workspace memset."""
    a, b, _ = make_inputs("randn", (2048, HIDDEN))
    mask = get_random_sign_mask_for_rht(True, torch.cuda.current_device())
    _, amax_ref = tex.add_hadamard_transform_amax(a, b, mask)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tex.add_hadamard_transform_amax(a, b, mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, amax = tex.add_hadamard_transform_amax(a, b, mask)
    for _ in range(3):
        amax.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise_equal(amax, amax_ref, "amax")


def _rope_inputs(s, b, groups, q_per_group, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(s + groups)
    randn = lambda *size: torch.randn(*size, device="cuda", generator=gen) * scale
    gq = randn(s, b, groups * q_per_group, 128).to(torch.bfloat16)
    gk = randn(s, b, groups, 128).to(torch.bfloat16)
    gv = randn(s, b, groups, 128).to(torch.bfloat16)
    freqs = te.attention.RotaryPositionEmbedding(128, rotary_base=500000)(s + 3).contiguous()
    cos = torch.cos(freqs).float().contiguous()
    sin = torch.sin(freqs).float().contiguous()
    return gq, gk, gv, freqs, cos, sin


@pytest.mark.parametrize(
    "s, b, groups, q_per_group", [(8192, 2, 8, 4), (256, 1, 2, 8), (96, 2, 4, 1), (64, 4, 1, 3)]
)
@pytest.mark.parametrize("scale", [1.0, 1e-3, 300.0])
def test_fused_qkv_rope_backward_with_rht_amax(s, b, groups, q_per_group, scale, clean_registry):
    """Bitwise TE's fused QKV RoPE backward (with the FP32 cos/sin tables of the frequencies),
    with the amaxes of the NVFP4 RHT cast registered for the result."""
    gq, gk, gv, freqs, cos, sin = _rope_inputs(s, b, groups, q_per_group, scale)
    split = [q_per_group * 128, 128, 128]
    ref = tex.fused_qkv_rope_backward(
        gq, gk, gv, freqs, freqs, split, tex.NVTE_QKV_Format.NVTE_SBHD, False, 1, 0
    )
    out = precomputed_rht_amax.fused_qkv_rope_backward_with_rht_amax(gq, gk, gv, cos, sin, split)
    assert out is not None and out.shape == ref.shape
    assert_bitwise_equal(out, ref, "grad qkv")
    # The QKV linear casts the gradient as [s * b, h_kv * (h_q / h_kv + 2) * d]
    quantizer = recipe_grad_quantizer()
    ref_row, ref_col = hadamard_amaxes(ref.view(s * b, -1), quantizer)
    amax = precomputed_rht_amax.take_rht_amax(out.view(s * b, -1), quantizer)
    assert amax is not None
    assert_bitwise_equal(amax[0], ref_row, "amax rowwise")
    assert_bitwise_equal(amax[1], ref_col, "amax columnwise")


def test_fused_qkv_rope_backward_with_rht_amax_unsupported(clean_registry):
    gq, gk, gv, _, cos, sin = _rope_inputs(96, 3, 2, 2)  # batch does not divide 64
    assert (
        precomputed_rht_amax.fused_qkv_rope_backward_with_rht_amax(
            gq, gk, gv, cos, sin, [256, 128, 128]
        )
        is None
    )
    gq, gk, gv, _, cos, sin = _rope_inputs(96, 2, 2, 2)
    assert (
        precomputed_rht_amax.fused_qkv_rope_backward_with_rht_amax(
            gq.float(), gk, gv, cos, sin, [256, 128, 128]
        )
        is None
    )


@pytest.mark.parametrize("swizzled", [True, False])
@pytest.mark.parametrize("shape", [(16384, HIDDEN), (8192, 2, HIDDEN), (128, 128), (256, 512)])
@pytest.mark.parametrize("kind", ["randn", "outliers", "tiny"])
def test_rowwise_cast_with_rht_amax(swizzled, shape, kind, clean_registry):
    """The row-wise cast equals nvfp4_quantize_with_amax; the later column-wise cast of the same
    input takes the registered amaxes and equals the unfused one."""
    x, _, _ = make_inputs(kind, shape)
    row_quantizer = recipe_input_quantizer(swizzled, columnwise=False)
    rowwise_amax = x.abs().amax().float().view(1)
    ref = tex.nvfp4_quantize_with_amax(x, row_quantizer, rowwise_amax, rowwise_amax)
    out = precomputed_rht_amax.nvfp4_quantize_rowwise_with_amax(
        x, row_quantizer, rowwise_amax, rowwise_amax
    )
    assert_nvfp4_equal(out, ref)

    col_quantizer = recipe_input_quantizer(swizzled, columnwise=True)
    col_quantizer.set_usage(rowwise=False, columnwise=True)
    col_ref = tex.quantize(x, col_quantizer)
    col = col_quantizer(x)
    assert not precomputed_rht_amax._saved_entries, "the column-wise cast did not take the amaxes"
    assert_nvfp4_equal(col, col_ref)


def test_rowwise_cast_with_rht_amax_registers_only_with_grad(clean_registry):
    x, _, _ = make_inputs("randn", (256, HIDDEN))
    quantizer = recipe_input_quantizer(columnwise=False)
    amax = x.abs().amax().float().view(1)
    with torch.no_grad():
        precomputed_rht_amax.nvfp4_quantize_rowwise_with_amax(x, quantizer, amax, amax)
    assert not precomputed_rht_amax._saved_entries
    precomputed_rht_amax.nvfp4_quantize_rowwise_with_amax(x, quantizer, amax, amax)
    assert len(precomputed_rht_amax._saved_entries) == 1


def test_residual_grad_rht_amax(monkeypatch, clean_registry):
    """Two chained fused adds with NVTE_FUSED_ADD_RMSNORM_NVFP4_GRAD_RHT_AMAX=1: the residual
    gradients are bitwise those of the plain adds, and the summed gradient of the first sum comes
    with its amaxes."""
    monkeypatch.setenv("NVTE_FUSED_ADD_RMSNORM_NVFP4_GRAD_RHT_AMAX", "1")
    x1, residual, weight = make_inputs("randn", (2048, HIDDEN), seed=3)
    x2, _, _ = make_inputs("randn", (2048, HIDDEN), seed=4)
    norm_grad = make_inputs("rowscale", (2048, HIDDEN), seed=5)[0]
    out_grad = make_inputs("rowscale", (2048, HIDDEN), seed=6)[0]

    def run(fused):
        leaves = [t.detach().clone().requires_grad_(True) for t in (x1, residual, x2)]
        a, r, b = leaves
        far._last_split = None
        s1 = fused_add_rmsnorm_nvfp4(a, r, weight, EPS) if fused else r + a
        far._pending = None
        s2 = fused_add_rmsnorm_nvfp4(b, s1, weight, EPS) if fused else s1 + b
        far._pending = None
        # s1 feeds an RMSNorm-like consumer and the second add
        torch.autograd.backward([s1, s2], [norm_grad, out_grad])
        return [t.grad for t in leaves]

    ref = run(fused=False)
    fused = run(fused=True)
    for name, f, r in zip(("x1", "residual", "x2"), fused, ref):
        assert_bitwise_equal(f, r, name + " grad")
    # The gradient of the first sum (the one the linear layer producing x1 casts) was registered
    entries = list(precomputed_rht_amax._grad_entries.values())
    assert len(entries) == 1, "the summed gradient has no registered amaxes"
    grad_sum = entries[0].tensor
    assert_bitwise_equal(grad_sum, ref[0], "summed gradient")
    quantizer = recipe_grad_quantizer()
    ref_row, ref_col = hadamard_amaxes(grad_sum, quantizer)
    amax = precomputed_rht_amax.take_rht_amax(grad_sum, quantizer)
    assert amax is not None
    assert_bitwise_equal(amax[0], ref_row, "amax rowwise")
    assert_bitwise_equal(amax[1], ref_col, "amax columnwise")
