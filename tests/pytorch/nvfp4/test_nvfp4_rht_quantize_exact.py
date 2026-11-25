# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# NOTE: This file is dependent on the success of test_nvfp4_quantize_exact.py.
# Separate to make sure all the functionalities are working as expected.
# Otherwise reference implementation will get messy.

# Due to the structure of NVFP4Quantizer, we need to test the RHT functionality
# together with the quantization functionality.

import functools
import os

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.reference_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import reference_utils
from transformer_engine.common.recipe import NVFP4BlockScaling

import pytest
import torch

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    contiguous: bool,
    return_rowwise: bool,
    return_transpose: bool,
    use_cpp_allocator: bool,
    swizzled_scale: bool = False,
    hadamard_dimension: int = 16,
    with_rht: bool = True,
    with_post_rht_amax: bool = True,
    with_random_sign_mask: bool = True,
) -> None:
    assert with_rht and with_post_rht_amax, "RHT and post-RHT amax reduction must be enabled."

    te_dtype = te.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    x = x.transpose(0, 1) if not contiguous else x

    # Quantize
    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=return_rowwise,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=with_rht,
        with_post_rht_amax=with_post_rht_amax,
        with_random_sign_mask=with_random_sign_mask,
    )
    if use_cpp_allocator:
        x_nvfp4_sut = nvfp4_quantizer(x)
    else:
        x_nvfp4_sut = nvfp4_quantizer.make_empty(
            x.shape, dtype=x_dtype, device=device, requires_grad=False
        )
        x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    # Extract data from NVFP4Tensor
    qx: torch.Tensor = (
        x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._rowwise_data is not None
        else None
    )
    sx: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    amax_rowwise = x_nvfp4_sut._amax_rowwise
    amax_colwise = x_nvfp4_sut._amax_columnwise

    qx = unpack_fp4(qx) if qx is not None else None
    qx_t = unpack_fp4(qx_t) if qx_t is not None else None

    # Reference quantization using NVFP4QuantizerRef with built-in RHT
    ref_quantizer = NVFP4QuantizerRef(
        dtype=reference_utils.Fp4Formats.E2M1,
        rowwise=return_rowwise,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        with_rht=with_rht,
        with_random_sign_mask=with_random_sign_mask,
    )
    x_nvfp4_ref = ref_quantizer.quantize(x)
    # Extract data from RefNVFP4Tensor
    qx_ref = (
        unpack_fp4(x_nvfp4_ref.data.view(dtype=torch.uint8))
        if x_nvfp4_ref.data is not None
        else None
    )
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    ref_amax_rowwise = x_nvfp4_ref.global_amax_row

    if return_transpose:
        assert x_nvfp4_ref.data_t is not None
        assert x_nvfp4_ref.scale_t is not None
        qx_t_ref = unpack_fp4(x_nvfp4_ref.data_t.view(dtype=torch.uint8))
        sx_t_ref = x_nvfp4_ref.scale_t.view(dtype=torch.uint8)
        # Compute transpose amax using the same reference quantizer
        x_t_for_amax = (
            ref_quantizer._apply_rht(x.t().contiguous()) if with_rht else x.t().contiguous()
        )
        ref_amax_colwise_t = torch.max(torch.abs(x_t_for_amax)).to(torch.float32).view(1)

        # SM120/121 uses TE's native single-K=16 MMA Hadamard arithmetic. cuBLAS may
        # choose a different reduction order (and applies the random-sign matrix in
        # the ATen reference orientation), so use the unfused TE kernel as the exact
        # reference for the fused TE kernel on these architectures.
        if torch.cuda.get_device_capability() in ((12, 0), (12, 1)):
            env_name = "NVTE_NVFP4_DISABLE_RHT_CAST_FUSION"
            old_value = os.environ.get(env_name)
            os.environ[env_name] = "1"
            try:
                native_quantizer = NVFP4Quantizer(
                    fp4_dtype=te_dtype,
                    rowwise=False,
                    columnwise=True,
                    with_amax_reduction=False,
                    with_rht=True,
                    with_post_rht_amax=True,
                    with_random_sign_mask=with_random_sign_mask,
                )
                native = native_quantizer(x)
            finally:
                if old_value is None:
                    os.environ.pop(env_name)
                else:
                    os.environ[env_name] = old_value
            qx_t_ref = unpack_fp4(native._columnwise_data.view(dtype=torch.uint8))
            sx_t_ref = native._columnwise_scale_inv.view(dtype=torch.uint8)
            ref_amax_colwise_t = native._amax_columnwise
    else:
        qx_t_ref = None
        sx_t_ref = None
        ref_amax_colwise_t = None

    if return_rowwise:
        torch.testing.assert_close(amax_rowwise, ref_amax_rowwise, atol=0.0, rtol=0.0)

        torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)
        # Compare only the valid portion of scale tensors (reference may not have padding)
        ref_sx_shape = sx_ref.shape
        sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
        torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(amax_colwise, ref_amax_colwise_t, atol=0.0, rtol=0.0)

        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)

        # Compare only the valid portion of transpose scale tensors
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # full tile cases
        (128, 128),
        (256, 256),
        (256, 1024),
        (1024, 256),
        # Padding required cases
        (256, 272),
        (304, 304),
        (320, 256),
        # Some larger tiles
        (2048, 2048),
        (1024, 2048),
        (2048, 1024),
        # Real shapes,
        (8192, 5120),
        (8192, 10240),
        (8192, 2560),
        (8192, 11328),
        (8192, 512),
        (8192, 3584),
        (5120, 8192),
        (10240, 8192),
        (2560, 8192),
        (11328, 8192),
        (512, 8192),
        (3584, 8192),
        (4096, 16384),
        (14336, 16384),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize(
    "with_random_sign_mask", [True, False], ids=["with_random_sign_mask", "no_random_sign_mask"]
)
def test_rht_with_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quantize_mode: str,
    use_cpp_allocator: bool,
    with_random_sign_mask: bool,
) -> None:

    if quantize_mode == "rowwise_only":
        return_rowwise = True
        return_transpose = False
    elif quantize_mode == "both_directions":
        return_rowwise = True
        return_transpose = True
    elif quantize_mode == "columnwise_only":
        return_rowwise = False
        return_transpose = True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        contiguous=True,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
        use_cpp_allocator=use_cpp_allocator,
        with_random_sign_mask=with_random_sign_mask,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (32, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize(
    "with_random_sign_mask", [True, False], ids=["with_random_sign_mask", "no_random_sign_mask"]
)
def test_nvfp4_quantization_noncontiguous_inputs(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quantize_mode: str,
    use_cpp_allocator: bool,
    with_random_sign_mask: bool,
):

    if quantize_mode == "rowwise_only":
        return_rowwise = True
        return_transpose = False
    elif quantize_mode == "both_directions":
        return_rowwise = True
        return_transpose = True
    elif quantize_mode == "columnwise_only":
        return_rowwise = False
        return_transpose = True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        contiguous=False,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
        use_cpp_allocator=use_cpp_allocator,
        with_random_sign_mask=with_random_sign_mask,
    )


def _bits(x: torch.Tensor) -> torch.Tensor:
    """Bit pattern of a tensor, for bitwise comparisons that also cover NaN."""
    return x.view({2: torch.int16, 4: torch.int32}.get(x.element_size(), torch.uint8))


def _assert_bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape
    assert torch.equal(_bits(a), _bits(b))


def _make_swiglu_backward_inputs(M: int, N: int, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU forward input [M, 2N] and incoming gradient [M, N]."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(M + N)

    def random_bits(*shape):
        return (
            torch.randint(-(2**15), 2**15, shape, device="cuda", generator=gen, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )

    if kind == "normal":
        x = torch.randn(M, 2 * N, device="cuda", generator=gen) * 1.5
        dy = torch.randn(M, N, device="cuda", generator=gen) * 3e-6
    elif kind == "wide_range":
        x = torch.randn(M, 2 * N, device="cuda", generator=gen) * torch.exp2(
            torch.randint(-20, 8, (M, 2 * N), device="cuda", generator=gen).float()
        )
        dy = torch.randn(M, N, device="cuda", generator=gen) * torch.exp2(
            torch.randint(-40, 0, (M, N), device="cuda", generator=gen).float()
        )
    elif kind in ("all_gate_values", "all_gate_values_random_bits"):
        # Every BF16 bit pattern appears in the gate half
        gate = torch.arange(M * N, device="cuda") % 2**16
        gate = gate[torch.randperm(M * N, device="cuda", generator=gen)]
        gate = gate.to(torch.int32).to(torch.int16).view(torch.bfloat16).view(M, N)
        if kind == "all_gate_values":
            linear = torch.randn(M, N, device="cuda", generator=gen) * 2
            dy = torch.randn(M, N, device="cuda", generator=gen) * 1e-5
        else:
            linear = random_bits(M, N)
            dy = random_bits(M, N)
        x = torch.cat([gate, linear.to(torch.bfloat16)], dim=1)
    else:
        raise ValueError(f"Invalid input kind: {kind}")
    return x.to(torch.bfloat16).contiguous(), dy.to(torch.bfloat16).contiguous()


def _make_rht_quantizer(stochastic_rounding: bool, swizzled_scale: bool) -> NVFP4Quantizer:
    quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=stochastic_rounding,
    )
    quantizer.optimize_for_gemm = swizzled_scale
    return quantizer


# Kernel of nvte_dswiglu_hadamard_transform_amax, and the RHT amax kernel of its unfused path
_FUSED_DSWIGLU_RHT_AMAX_KERNEL = "dswiglu_hadamard_transform_amax_kernel"
_UNFUSED_RHT_AMAX_KERNEL = "HadamardAmaxTmaKernel"


def _run_and_list_cuda_kernels(fn):
    """Run fn under the PyTorch profiler; return its result and the CUDA kernels it launched."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    names = [evt.name for evt in prof.events() if evt.device_type == torch.autograd.DeviceType.CUDA]
    assert names, "The profiler recorded no CUDA kernels"
    return result, names


def _ran(kernel: str, kernel_names: list[str]) -> bool:
    return any(kernel in name for name in kernel_names)


@functools.lru_cache(maxsize=None)
def _fused_dswiglu_rht_amax_built() -> bool:
    """Whether this build compiles the fused dSwiGLU + RHT amax kernel.

    The kernel is compiled only with NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1, a build option that
    is off by default. When that variable is set while the tests run, it states the build mode
    and the tests require the matching kernel. Otherwise the build mode is inferred from the
    kernel that a supported shape runs.
    """
    build_flag = os.getenv("NVTE_BUILD_ACTIVATION_WITH_FAST_MATH")
    if build_flag is not None:
        return bool(int(build_flag))
    x, dy = _make_swiglu_backward_inputs(64, 64, "normal")
    _, kernel_names = _run_and_list_cuda_kernels(
        lambda: tex.dswiglu_hadamard_transform_amax(dy, x, 0)
    )
    return _ran(_FUSED_DSWIGLU_RHT_AMAX_KERNEL, kernel_names)


def _fused_dswiglu_rht_amax_supported(M: int, N: int) -> bool:
    """Whether the fused kernel handles a [M, 2N] SwiGLU input on this device, in any build."""
    return M % 64 == 0 and N % 64 == 0 and torch.cuda.get_device_capability() >= (10, 0)


def _expect_fused_dswiglu_rht_amax(M: int, N: int) -> bool:
    return _fused_dswiglu_rht_amax_supported(M, N) and _fused_dswiglu_rht_amax_built()


def _assert_dswiglu_rht_amax_kernels(kernel_names: list[str], M: int, N: int) -> None:
    """The fused kernel ran when this build and device support the shape, the unfused one if not."""
    fused_ran = _ran(_FUSED_DSWIGLU_RHT_AMAX_KERNEL, kernel_names)
    unfused_ran = _ran(_UNFUSED_RHT_AMAX_KERNEL, kernel_names)
    if _expect_fused_dswiglu_rht_amax(M, N):
        assert fused_ran and not unfused_ran, f"Expected the fused kernel, ran {kernel_names}"
    else:
        assert unfused_ran and not fused_ran, f"Expected the unfused path, ran {kernel_names}"


def _skip_if_fused_dswiglu_rht_amax_not_built(M: int, N: int) -> None:
    """Report a case meant for the fused kernel as skipped when only the fallback could run."""
    if _fused_dswiglu_rht_amax_supported(M, N) and not _fused_dswiglu_rht_amax_built():
        pytest.skip(
            "Fused dSwiGLU + RHT amax kernel not built: Transformer Engine was built without"
            " NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1, so only the unfused fallback was checked"
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # Llama 3.1 8B FC1 (16384 tokens, FFN hidden size 14336)
        (16384, 14336),
        # Fused kernel: multiples of 64
        (64, 64),
        (128, 192),
        (192, 320),
        (4096, 2048),
        # Unfused fallback
        (48, 64),
        (128, 80),
    ],
)
@pytest.mark.parametrize(
    "kind", ["normal", "wide_range", "all_gate_values", "all_gate_values_random_bits"]
)
def test_dswiglu_hadamard_transform_amax(M: int, N: int, kind: str) -> None:
    """Fused SwiGLU backward + RHT amaxes equals dswiglu followed by the quantizer's amaxes."""
    if kind.startswith("all_gate_values") and M * N < 2**16:
        pytest.skip("Too few elements for every gate value")
    x, dy = _make_swiglu_backward_inputs(M, N, kind)
    quantizer = _make_rht_quantizer(stochastic_rounding=False, swizzled_scale=False)

    (dx, pre_rht_amax, post_rht_amax), kernel_names = _run_and_list_cuda_kernels(
        lambda: tex.dswiglu_hadamard_transform_amax(dy, x, quantizer.rht_matrix_random_sign_mask_t)
    )
    _assert_dswiglu_rht_amax_kernels(kernel_names, M, N)
    dx_ref = tex.dswiglu(dy, x, None)
    _assert_bitwise_equal(dx, dx_ref)

    if kind != "all_gate_values_random_bits":
        # With NaN and Inf everywhere only the SwiGLU backward is compared
        dx_ref_nvfp4 = quantizer(dx_ref)
        _assert_bitwise_equal(pre_rht_amax, dx_ref_nvfp4._amax_rowwise)
        _assert_bitwise_equal(post_rht_amax, dx_ref_nvfp4._amax_columnwise)
    _skip_if_fused_dswiglu_rht_amax_not_built(M, N)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(16384, 14336), (256, 128), (384, 256), (1024, 384), (48, 64)])
@pytest.mark.parametrize("kind", ["normal", "wide_range"])
@pytest.mark.parametrize("stochastic_rounding", [False, True], ids=["rn", "sr"])
@pytest.mark.parametrize("swizzled_scale", [False, True], ids=["compact", "swizzled"])
def test_dswiglu_nvfp4_rht_fused_amax(
    M: int,
    N: int,
    kind: str,
    stochastic_rounding: bool,
    swizzled_scale: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVTE_NVFP4_FUSE_DSWIGLU_RHT_AMAX=1 leaves the NVFP4 SwiGLU backward bitwise unchanged."""
    x, dy = _make_swiglu_backward_inputs(M, N, kind)
    quantizer = _make_rht_quantizer(stochastic_rounding, swizzled_scale)

    def swiglu_backward(fused: bool):
        monkeypatch.setenv("NVTE_NVFP4_FUSE_DSWIGLU_RHT_AMAX", "1" if fused else "0")
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        out, kernel_names = _run_and_list_cuda_kernels(lambda: tex.dswiglu(dy, x, quantizer))
        rng_state = torch.cuda.get_rng_state()
        return out, rng_state, kernel_names

    ref, ref_rng_state, ref_kernel_names = swiglu_backward(fused=False)
    out, out_rng_state, out_kernel_names = swiglu_backward(fused=True)
    assert not _ran(_FUSED_DSWIGLU_RHT_AMAX_KERNEL, ref_kernel_names)
    _assert_dswiglu_rht_amax_kernels(out_kernel_names, M, N)
    assert torch.equal(out_rng_state, ref_rng_state)
    assert out._with_gemm_swizzled_scales == ref._with_gemm_swizzled_scales
    for name in ("_rowwise_data", "_columnwise_data", "_amax_rowwise", "_amax_columnwise"):
        _assert_bitwise_equal(getattr(out, name), getattr(ref, name))
    if swizzled_scale:
        _assert_bitwise_equal(out._rowwise_scale_inv, ref._rowwise_scale_inv)
        _assert_bitwise_equal(out._columnwise_scale_inv, ref._columnwise_scale_inv)
    else:
        # Compact scales are padded and the padding is not written
        _assert_bitwise_equal(
            out._rowwise_scale_inv[:M, : 2 * N // 16], ref._rowwise_scale_inv[:M, : 2 * N // 16]
        )
        _assert_bitwise_equal(
            out._columnwise_scale_inv[: 2 * N, : (M + 15) // 16],
            ref._columnwise_scale_inv[: 2 * N, : (M + 15) // 16],
        )
    _skip_if_fused_dswiglu_rht_amax_not_built(M, N)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_swiglu_op_nvfp4_rht_fused_amax(monkeypatch: pytest.MonkeyPatch) -> None:
    """te.ops MLP with the NVFP4 recipe: identical gradients with and without the fused amax."""
    from transformer_engine.pytorch.ops.basic import swiglu as swiglu_op

    tokens, hidden, ffn_hidden = 256, 256, 512
    torch.manual_seed(0)
    model = te.ops.Sequential(
        te.ops.BasicLinear(hidden, 2 * ffn_hidden, device="cuda", dtype=torch.bfloat16),
        te.ops.SwiGLU(),
        te.ops.BasicLinear(ffn_hidden, hidden, device="cuda", dtype=torch.bfloat16),
    )
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)

    dswiglu_quantizers = []
    dswiglu = tex.dswiglu

    def recording_dswiglu(grad, fwd_input, quantizer):
        dswiglu_quantizers.append(quantizer)
        return dswiglu(grad, fwd_input, quantizer)

    monkeypatch.setattr(swiglu_op.tex, "dswiglu", recording_dswiglu)

    def grads(fused: bool):
        monkeypatch.setenv("NVTE_NVFP4_FUSE_DSWIGLU_RHT_AMAX", "1" if fused else "0")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        x_ = x.detach().clone().requires_grad_()
        for param in model.parameters():
            param.grad = None
        with te.autocast(enabled=True, recipe=NVFP4BlockScaling()):
            y = model(x_)
        _, kernel_names = _run_and_list_cuda_kernels(lambda: y.backward(dy))
        return [x_.grad] + [param.grad for param in model.parameters()], kernel_names

    ref, ref_kernel_names = grads(fused=False)
    out, out_kernel_names = grads(fused=True)
    assert not _ran(_FUSED_DSWIGLU_RHT_AMAX_KERNEL, ref_kernel_names)
    assert _ran(_FUSED_DSWIGLU_RHT_AMAX_KERNEL, out_kernel_names) == _expect_fused_dswiglu_rht_amax(
        tokens, ffn_hidden
    )
    assert any(
        isinstance(q, NVFP4Quantizer) and q.with_rht and q.with_post_rht_amax
        for q in dswiglu_quantizers
    ), "SwiGLU backward did not quantize with an NVFP4 RHT quantizer"
    for a, b in zip(out, ref):
        _assert_bitwise_equal(a, b)
    _skip_if_fused_dswiglu_rht_amax_not_built(tokens, ffn_hidden)
