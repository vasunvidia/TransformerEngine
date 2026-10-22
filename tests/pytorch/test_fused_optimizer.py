# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling, Float8BlockScaling
from transformer_engine.pytorch import MultiheadAttention, quantized_model_init, is_bf16_available
from transformer_engine.pytorch import QuantizedTensor
from transformer_engine.pytorch.optimizers import multi_tensor_adam_flat, multi_tensor_applier
from transformer_engine.pytorch.optimizers.fused_adam import (
    _MAX_TENSORS_PER_MULTI_TENSOR_ADAM_CALL,
)
from transformer_engine.pytorch.utils import gpu_autocast_ctx

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)


class TestFusedOptimizer:

    def setup_method(self, *, iters: int = 7) -> None:
        self.iters = iters
        torch.manual_seed(9876)

    def gen_param_optim(self, tensors, options, tst_options=None):

        # Adding this to make backward compatible with existing tests. Just in
        # case "tst_options" are not provided, it gets a copy of options
        # which contains the parameters for the reference optimizer
        if tst_options == None:
            tst_options = options

        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone()))

        ref_optim = self.ref_optim(ref_param, **options)
        tst_optim = self.fused_optim(tst_param, **tst_options)

        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad

    def gen_mixed_grad(self, ref_param, tst_param, scale=1.0):
        half_grads = []
        for p_ref, p_tst in zip(ref_param, tst_param):
            half_grads.append(torch.rand_like(p_ref).half())
            p_ref.grad = half_grads[-1].float() / scale
        return half_grads

    def gen_single_type_test(
        self, param_type=torch.float, device="cuda", *, skip_assert: bool = False
    ):
        nelem = 278011

        # Some ref and test optimizers may require different set of options.
        # This is a quick workaround to add that functionality while making
        # minimum changes in existing code.
        # If there is no "tst_options" field provided, safe to initialize
        # the test optimizer with the parameters of reference optimizer.
        if not hasattr(self, "tst_options"):
            self.tst_options = self.options

        tensor = torch.rand(nelem, dtype=param_type, device=device)

        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
            [tensor], self.options, self.tst_options
        )

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()
            if skip_assert:
                return
            torch.testing.assert_close(ref_param, tst_param)


class TestFusedAdam(TestFusedOptimizer):

    def setup_method(self) -> None:
        super().setup_method()
        self.options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        self.ref_optim = torch.optim.Adam
        self.fused_optim = te.optimizers.FusedAdam

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    @pytest.mark.skipif(
        not hasattr(torch.nn.Parameter(torch.empty(0)), "grad_dtype"),
        reason="PyTorch does not support parameter grad_dtype",
    )
    @pytest.mark.parametrize(
        "param_dtype,grad_dtype,master_weights",
        [
            (torch.float32, torch.bfloat16, False),
            (torch.bfloat16, torch.float32, True),
        ],
    )
    def test_capturable_mixed_param_grad_dtype(self, param_dtype, grad_dtype, master_weights):
        """Capturable Adam supports parameter and gradient tensors with different dtypes."""
        if param_dtype == torch.bfloat16 and not is_bf16_available():
            pytest.skip("BF16 is not supported")

        ref_param = torch.nn.Parameter(torch.ones(16, device="cuda", dtype=param_dtype))
        tst_param = torch.nn.Parameter(ref_param.detach().clone())
        for param in (ref_param, tst_param):
            param.grad_dtype = grad_dtype
            param.grad = torch.ones_like(param, dtype=grad_dtype)

        ref_optim = self.fused_optim(
            [ref_param], capturable=False, master_weights=master_weights, **self.options
        )
        tst_optim = self.fused_optim(
            [tst_param], capturable=True, master_weights=master_weights, **self.options
        )
        # Initialize optimizer state before CUDA graph capture.
        ref_optim.step()
        tst_optim.step()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tst_optim.step()
        graph.replay()
        ref_optim.step()

        torch.testing.assert_close(tst_param, ref_param)
        torch.testing.assert_close(
            tst_optim.state[tst_param]["exp_avg"], ref_optim.state[ref_param]["exp_avg"]
        )
        torch.testing.assert_close(
            tst_optim.state[tst_param]["exp_avg_sq"], ref_optim.state[ref_param]["exp_avg_sq"]
        )
        if master_weights:
            torch.testing.assert_close(
                tst_optim.state[tst_param]["master_param"],
                ref_optim.state[ref_param]["master_param"],
            )

    # NOTE(mkozuki): Current threshold values look too small for BFloat16.
    # TODO(mkozuki): Refactor `TestFusedOptimizer`
    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16, skip_assert=True)

    def test_bfloat16(self):
        self.gen_single_type_test(param_type=torch.bfloat16, skip_assert=True)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]

        tensors = []
        for size in sizes:
            tensors.append(torch.rand(size, dtype=torch.float, device="cuda"))
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(tensors, self.options)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

    @pytest.mark.parametrize(
        "param_dtype,master_weights,capturable",
        [
            (torch.float32, False, False),
            (torch.bfloat16, True, True),
        ],
    )
    def test_many_small_parameters(self, param_dtype, master_weights, capturable):
        """FusedAdam chunks large parameter groups without changing optimizer semantics."""
        if param_dtype == torch.bfloat16 and not is_bf16_available():
            pytest.skip("BF16 is not supported")

        # Keep this above the default 20 MiB handle-pool capacity so the test
        # fails without FusedAdam's per-call bound, not just at the slice boundary.
        num_params = 8 * _MAX_TENSORS_PER_MULTI_TENSOR_ADAM_CALL + 1
        values = torch.linspace(-1, 1, num_params, dtype=param_dtype, device="cuda")
        coefficients = torch.linspace(-0.5, 0.5, num_params, dtype=param_dtype, device="cuda")
        ref_param = torch.nn.Parameter(values.float())
        tst_params = [torch.nn.Parameter(value.reshape(1).clone()) for value in values]
        ref_optim = torch.optim.Adam([ref_param], **self.options)
        tst_optim = te.optimizers.FusedAdam(
            tst_params,
            capturable=capturable,
            master_weights=master_weights,
            **self.options,
        )

        def backward():
            ref_optim.zero_grad()
            tst_optim.zero_grad()
            ref_loss = (ref_param.to(param_dtype).float() * coefficients.float()).sum()
            tst_loss = (torch.cat(tst_params).float() * coefficients.float()).sum()
            torch.testing.assert_close(tst_loss, ref_loss)
            ref_loss.backward()
            tst_loss.backward()
            torch.testing.assert_close(
                torch.cat([param.grad.float() for param in tst_params]), ref_param.grad
            )

        backward()
        ref_optim.step()
        tst_optim.step()

        backward()
        if capturable:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tst_optim.step()
            graph.replay()
        else:
            tst_optim.step()
        ref_optim.step()

        torch.testing.assert_close(
            torch.cat([param.detach() for param in tst_params]), ref_param.detach().to(param_dtype)
        )
        torch.testing.assert_close(
            torch.cat([tst_optim.state[param]["exp_avg"] for param in tst_params]),
            ref_optim.state[ref_param]["exp_avg"],
        )
        torch.testing.assert_close(
            torch.cat([tst_optim.state[param]["exp_avg_sq"] for param in tst_params]),
            ref_optim.state[ref_param]["exp_avg_sq"],
        )
        if master_weights:
            torch.testing.assert_close(
                torch.cat([tst_optim.state[param]["master_param"] for param in tst_params]),
                ref_param,
            )

    def test_adam_option(self):
        nelem = 1
        adam_option = {
            "lr": 0.01,
            "betas": (0.6, 0.9),
            "eps": 3e-06,
            "weight_decay": 0,
            "amsgrad": False,
        }

        tensor = torch.rand(nelem, dtype=torch.float, device="cuda")
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([tensor], adam_option)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

    def test_frozen_model(self):
        nelem = 1
        adam_option = {
            "lr": 0.01,
            "betas": (0.6, 0.9),
            "eps": 3e-06,
            "weight_decay": 0,
            "amsgrad": False,
        }

        tensor = torch.rand(nelem, dtype=torch.float, device="cuda")
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([tensor], adam_option)

        # Add an empty param group which may occur for pipeline parallel p-tuning
        tst_optim.add_param_group({"params": []})

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

    @pytest.mark.parametrize("capturable", [False, True])
    def test_empty_param_group_advances_step(self, capturable):
        # An empty param group must advance its step counter like a populated one.
        # The same group can be empty on one data-parallel rank and populated on
        # another, and "step" is part of the checkpoint, so a group that stops
        # counting makes a resumed run apply a stale bias correction.
        param = torch.nn.Parameter(torch.rand(4, dtype=torch.float, device="cuda"))
        tst_optim = self.fused_optim(
            [{"params": [param]}, {"params": []}], capturable=capturable, **self.options
        )

        num_steps = 3
        for _ in range(num_steps):
            param.grad = torch.rand_like(param)
            tst_optim.step()

        populated_step, empty_step = (int(g["step"]) for g in tst_optim.param_groups)
        assert populated_step == num_steps
        assert empty_step == populated_step

        # The counter is checkpointed through the param groups, so the empty group
        # has to round trip with the same value as the populated one.
        checkpoint = tst_optim.state_dict()
        assert int(checkpoint["param_groups"][1]["step"]) == num_steps

    def test_empty_param_at_end_of_group(self):
        tensors = [
            torch.ones(4, dtype=torch.float, device="cuda"),
            torch.empty(0, dtype=torch.float, device="cuda"),
        ]
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(tensors, self.options)

        self.gen_grad(ref_param, tst_param)
        ref_optim.step()
        tst_optim.step()

        torch.testing.assert_close(ref_param, tst_param)

    def gen_precision_aware_test(
        self,
        use_fp8_params,
        param_dtype,
        use_master_weights,
        master_weight_dtype,
        grad_dtype,
        exp_avg_dtype,
        exp_avg_sq_dtype,
        store_param_remainders=False,
        model_rtol=None,
        model_atol=None,
        master_rtol=None,
        master_atol=None,
        skip_assert=False,
    ):
        build_model_context = nullcontext
        build_model_context_args = {}
        if use_fp8_params:
            build_model_context = quantized_model_init
            build_model_context_args["enabled"] = True

        with build_model_context(**build_model_context_args):
            model = MultiheadAttention(
                hidden_size=1024,
                num_attention_heads=16,
                layer_number=1,
                params_dtype=param_dtype,
                fuse_qkv_params=True,
            ).cuda()

        ref_params = []
        model_params = []

        for p in model.parameters():
            if p.requires_grad:
                ref_params.append(p.detach().clone().float())
                model_params.append(p)

        options = {
            "lr": 1,
            "betas": (0.1, 0.25),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }

        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            model_params,
            master_weights=use_master_weights,
            master_weight_dtype=master_weight_dtype,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
            use_decoupled_grad=True,
            store_param_remainders=store_param_remainders,
            **options,
        )

        def test_one_iteration(ref_optimizer, tst_optimizer):
            for p_ref, p in zip(ref_params, model_params):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone().to(grad_dtype)
            ref_optimizer.step()
            tst_optimizer.step()
            if use_master_weights and not store_param_remainders:
                master_weights_to_fp32 = [
                    tst_optim.get_unscaled_state(p, "master_param") for p in model_params
                ]
                if not skip_assert:
                    torch.testing.assert_close(
                        ref_params,
                        master_weights_to_fp32,
                        rtol=master_rtol,
                        atol=master_atol,
                        equal_nan=True,
                    )
            ref_params_to_model_dtype = [p.to(param_dtype) for p in ref_params]
            if not skip_assert:
                torch.testing.assert_close(
                    ref_params_to_model_dtype,
                    model_params,
                    rtol=model_rtol,
                    atol=model_atol,
                    equal_nan=True,
                )

        for i in range(self.iters):
            test_one_iteration(ref_optim, tst_optim)

        state_dict = tst_optim.state_dict()
        tst_optim = te.optimizers.FusedAdam(
            model_params,
            master_weights=use_master_weights,
            master_weight_dtype=master_weight_dtype,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
            use_decoupled_grad=True,
            store_param_remainders=store_param_remainders,
            **options,
        )
        tst_optim.load_state_dict(state_dict)

        for i in range(self.iters):
            test_one_iteration(ref_optim, tst_optim)

    def test_fp32_no_master(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.float32,
            use_master_weights=False,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_fp32_master(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_fp32_master_store_param_remainders(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            store_param_remainders=True,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_fp16_master(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.half,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_bf16_grad(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.bfloat16,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_fp16_exp_avg(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.half,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_bf16_exp_avg(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_fp8_exp_avg(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.uint8,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=1e-2,
            master_atol=1e-2,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_fp16_exp_avg_sq(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.half,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_bf16_exp_avg_sq(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.bfloat16,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_bf16_exp_avg_and_exp_avg_sq(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_fp8_exp_avg_sq(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.uint8,
            skip_assert=True,
        )

    @pytest.mark.skipif(not is_bf16_available(), reason="bf16 if not supported")
    def test_bf16_model_weight_cast(self):
        dtype = torch.bfloat16
        model = MultiheadAttention(
            hidden_size=1024,
            num_attention_heads=16,
            layer_number=1,
            params_dtype=dtype,
            fuse_qkv_params=True,
        ).cuda()
        ref_params = []
        model_params = []
        for p in model.parameters():
            if p.requires_grad:
                ref_params.append(p.detach().clone().float())
                model_params.append(p)
        options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            model_params, master_weights=True, use_decoupled_grad=True, **options
        )

        for i in range(self.iters):
            for p_ref, p in zip(ref_params, model_params):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone()
            ref_optim.step()
            tst_optim.step()
            master_params = [tst_optim.get_unscaled_state(p, "master_param") for p in model_params]
            torch.testing.assert_close(ref_params, master_params)
            model_params_to_fp32 = [p.float() for p in model_params]
            torch.testing.assert_close(
                ref_params, model_params_to_fp32, rtol=1e-3, atol=1e-3, equal_nan=True
            )

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_fp8_model_weight_cast(self):
        dtype = torch.bfloat16
        with quantized_model_init(enabled=True, recipe=DelayedScaling()):
            model = MultiheadAttention(
                hidden_size=1024,
                num_attention_heads=16,
                layer_number=1,
                params_dtype=dtype,
                fuse_qkv_params=True,
            ).cuda()
        ref_params = []
        model_params = []
        for p in model.parameters():
            if p.requires_grad:
                ref_params.append(p.detach().clone().float())
                model_params.append(p)
        options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            model_params, master_weights=True, use_decoupled_grad=True, **options
        )

        for i in range(self.iters):
            for p_ref, p in zip(ref_params, model_params):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone()
            ref_optim.step()
            tst_optim.step()
            master_params = [tst_optim.get_unscaled_state(p, "master_param") for p in model_params]
            torch.testing.assert_close(ref_params, master_params)
            model_params_to_fp32 = [p.float() for p in model_params]
            torch.testing.assert_close(
                ref_params, model_params_to_fp32, rtol=1e-2, atol=1e-2, equal_nan=True
            )


class TestFusedAdamMXFP8(TestFusedOptimizer):
    """FusedAdam with MXFP8BlockScaling quantized primary weights (single GPU, no FSDP)."""

    def setup_method(self) -> None:
        super().setup_method(iters=5)
        mxfp8_available, self.mxfp8_reason = te.is_mxfp8_available(return_reason=True)
        self.mxfp8_available = mxfp8_available

    def _build_model(self):
        recipe = MXFP8BlockScaling()
        with quantized_model_init(enabled=True, recipe=recipe):
            model = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        return model, recipe

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_mxfp8_linear_fused_adam_master_weights(self):
        """quantized_model_init(MXFP8) + te.Linear + FusedAdam(master_weights=True).

        Verifies:
        - Model params are MXFP8 QuantizedTensors after init
        - FP32 master weights track a reference Adam optimizer
        - Params remain QuantizedTensors after training
        - Loss decreases over training steps
        """
        if not self.mxfp8_available:
            pytest.skip(self.mxfp8_reason)

        model, recipe = self._build_model()

        # Verify weight params are QuantizedTensors (bias stays bf16)
        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(
                    p, QuantizedTensor
                ), f"Expected QuantizedTensor for {name}, got {type(p).__name__}"

        # Build reference: clone dequantized weights for a plain Adam
        ref_params = [p.detach().clone().float() for p in model.parameters()]

        options = {"lr": 5e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}
        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            list(model.parameters()),
            master_weights=True,
            master_weight_dtype=torch.float32,
            use_decoupled_grad=True,
            **options,
        )

        for _ in range(self.iters):
            for p_ref, p in zip(ref_params, model.parameters()):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone()
            ref_optim.step()
            tst_optim.step()

            # FP32 master weights should match reference Adam exactly
            master_params = [
                tst_optim.get_unscaled_state(p, "master_param") for p in model.parameters()
            ]
            torch.testing.assert_close(ref_params, master_params)

        # Weight params should still be QuantizedTensors after training
        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(
                    p, QuantizedTensor
                ), f"{name} lost QuantizedTensor type after training: {type(p).__name__}"

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_mxfp8_linear_forward_backward_step(self):
        """End-to-end: quantized_model_init + autocast forward + backward + FusedAdam.step().

        Uses te.autocast with MXFP8BlockScaling recipe for the forward pass,
        verifying the full training loop works with quantized compute.
        """
        if not self.mxfp8_available:
            pytest.skip(self.mxfp8_reason)

        model, recipe = self._build_model()

        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        batch_size, seq_len, hidden = 4, 32, 256
        x = torch.randn(batch_size, seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        losses = []
        for i in range(self.iters):
            optimizer.zero_grad(set_to_none=True)
            with te.autocast(enabled=True, recipe=recipe):
                output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.item())
            loss.backward()

            # Verify all params have non-None gradients after backward
            for name, p in model.named_parameters():
                assert p.grad is not None, f"Step {i}: {name} has no gradient after backward"
                assert (
                    p.grad.shape == p.shape
                ), f"Step {i}: {name} grad shape {p.grad.shape} != param shape {p.shape}"
                assert torch.isfinite(p.grad).all(), f"Step {i}: {name} has non-finite gradients"
                assert p.grad.any(), f"Step {i}: {name} gradient is all zeros"

            optimizer.step()

        # Verify loss decreased
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

        # Verify weight params remain QuantizedTensors
        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(
                    p, QuantizedTensor
                ), f"{name} lost QuantizedTensor type: {type(p).__name__}"

        # Verify optimizer states are float32
        for name, p in model.named_parameters():
            state = optimizer.state[p]
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32
            if "bias" not in name:
                assert state["master_param"].dtype == torch.float32


class TestFusedAdamFloat8Block(TestFusedOptimizer):
    """FusedAdam with Float8BlockScaling quantized primary weights (single GPU, no FSDP)."""

    def setup_method(self) -> None:
        super().setup_method(iters=5)
        fp8_block_available, self.fp8_block_reason = te.is_fp8_block_scaling_available(
            return_reason=True
        )
        self.fp8_block_available = fp8_block_available

    def _build_model(self):
        recipe = Float8BlockScaling()
        with quantized_model_init(enabled=True, recipe=recipe):
            model = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        return model, recipe

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_float8block_linear_fused_adam_master_weights(self):
        """quantized_model_init(Float8BlockScaling) + te.Linear + FusedAdam(master_weights=True).

        Verifies:
        - Model params are QuantizedTensors after init
        - FP32 master weights track a reference Adam optimizer
        - Params remain QuantizedTensors after training
        """
        if not self.fp8_block_available:
            pytest.skip(self.fp8_block_reason)

        model, recipe = self._build_model()

        # Verify weight params are QuantizedTensors (bias stays bf16)
        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(
                    p, QuantizedTensor
                ), f"Expected QuantizedTensor for {name}, got {type(p).__name__}"

        # Build reference: clone dequantized weights for a plain Adam
        ref_params = [p.detach().clone().float() for p in model.parameters()]

        options = {"lr": 5e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}
        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            list(model.parameters()),
            master_weights=True,
            master_weight_dtype=torch.float32,
            use_decoupled_grad=True,
            **options,
        )

        for _ in range(self.iters):
            for p_ref, p in zip(ref_params, model.parameters()):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone()
            ref_optim.step()
            tst_optim.step()

            # FP32 master weights should match reference Adam exactly
            master_params = [
                tst_optim.get_unscaled_state(p, "master_param") for p in model.parameters()
            ]
            torch.testing.assert_close(ref_params, master_params)

        # Weight params should still be QuantizedTensors after training
        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(
                    p, QuantizedTensor
                ), f"{name} lost QuantizedTensor type after training: {type(p).__name__}"

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_float8block_linear_forward_backward_step(self):
        """End-to-end: quantized_model_init + autocast forward + backward + FusedAdam.step().

        Uses te.autocast with Float8BlockScaling recipe for the forward pass,
        verifying the full training loop works with quantized compute.
        """
        if not self.fp8_block_available:
            pytest.skip(self.fp8_block_reason)

        model, recipe = self._build_model()

        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        batch_size, seq_len, hidden = 4, 32, 256
        x = torch.randn(batch_size, seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        target = torch.randn_like(x)

        losses = []
        for i in range(self.iters):
            optimizer.zero_grad(set_to_none=True)
            with te.autocast(enabled=True, recipe=recipe):
                output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.item())
            loss.backward()

            # Verify all params have non-None gradients after backward
            for name, p in model.named_parameters():
                assert p.grad is not None, f"Step {i}: {name} has no gradient after backward"
                assert (
                    p.grad.shape == p.shape
                ), f"Step {i}: {name} grad shape {p.grad.shape} != param shape {p.shape}"
                assert torch.isfinite(p.grad).all(), f"Step {i}: {name} has non-finite gradients"
                assert p.grad.any(), f"Step {i}: {name} gradient is all zeros"

            optimizer.step()

        # Verify loss decreased
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

        # Verify weight params remain QuantizedTensors
        for name, p in model.named_parameters():
            if "bias" not in name:
                assert isinstance(
                    p, QuantizedTensor
                ), f"{name} lost QuantizedTensor type: {type(p).__name__}"

        # Verify optimizer states are float32
        for name, p in model.named_parameters():
            state = optimizer.state[p]
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32
            if "bias" not in name:
                assert state["master_param"].dtype == torch.float32


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Raw bits of a tensor, for bitwise comparisons."""
    return tensor.view({4: torch.int32, 2: torch.int16}[tensor.element_size()])


class TestMultiTensorAdamFlat:
    """multi_tensor_adam_flat is bitwise equal to the multi-tensor path it replaces.

    The reference is the update of a distributed-optimizer shard with per-parameter FP32 tensors:
    cast each gradient segment to FP32, optionally scale the FP32 gradients with
    multi_tensor_scale_tensor (gradient clipping), step FusedAdam on FP32 parameters (FP32
    moments, no master weights, non-capturable), and copy the parameters back into the model
    parameter buffer.
    """

    # (number of elements, padding before the parameter): segment boundaries cut vectors, and
    # the padding must be neither read nor written.
    layout = [(4096, 0), (1000, 7), (3, 1), (65536 + 5, 64), (1, 0), (2880, 3), (1536, 13)]
    group_options = [
        {"lr": 1e-3, "weight_decay": 0.1, "betas": (0.9, 0.95), "eps": 1e-5},
        {"lr": 5e-4, "weight_decay": 0.0, "betas": (0.8, 0.99), "eps": 1e-8},
    ]
    max_norm = 1.0

    @pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32, None])
    @pytest.mark.parametrize("clip", [True, False])
    @pytest.mark.parametrize("adam_w_mode", [True, False])
    @pytest.mark.parametrize("misaligned", [False, True])
    def test_matches_fused_adam(self, grad_dtype, param_dtype, clip, adam_w_mode, misaligned):
        if torch.bfloat16 in (grad_dtype, param_dtype) and not is_bf16_available():
            pytest.skip("BF16 is not supported")
        torch.manual_seed(1234)
        device = "cuda"

        # Flat layout: segments (start, end, group) with padding in between.
        segments = []
        pos = 0
        for i, (numel, padding) in enumerate(self.layout):
            pos += padding
            segments.append((pos, pos + numel, i % len(self.group_options)))
            pos += numel
        numel = pos + 5
        # A one-element offset makes every buffer misaligned for vector access.
        offset = 1 if misaligned else 0

        def flat(dtype, fill=None):
            buf = torch.empty(numel + offset, dtype=dtype, device=device)
            if fill is None:
                buf.normal_()
            else:
                buf.fill_(fill)
            return buf[offset:]

        grad = flat(grad_dtype)
        master = flat(torch.float32)
        exp_avg = flat(torch.float32, 0.0)
        exp_avg_sq = flat(torch.float32, 0.0)
        param = flat(param_dtype, -7.0) if param_dtype is not None else None
        segments_t = torch.tensor(segments, dtype=torch.int64, device=device)

        # Reference: per-parameter FP32 tensors and FusedAdam.
        ref_params = [torch.nn.Parameter(master[s:e].clone()) for s, e, _ in segments]
        ref_groups = [
            {"params": [p for p, (_, _, g) in zip(ref_params, segments) if g == i], **options}
            for i, options in enumerate(self.group_options)
        ]
        ref_optim = te.optimizers.FusedAdam(ref_groups, adam_w_mode=adam_w_mode)

        is_gap = torch.ones(numel, dtype=torch.bool, device=device)
        for s, e, _ in segments:
            is_gap[s:e] = False

        for step in range(1, 4):
            # Large gradients, so that clipping changes them.
            grad.copy_(torch.randn(numel, device=device) * 10)
            # FP32 copies (as the default path makes), scaled in place by the reference.
            ref_grads = [grad[s:e].to(torch.float32, copy=True) for s, e, _ in segments]
            grad_scale = None
            if clip:
                overflow = torch.zeros(1, dtype=torch.int, device=device)
                norm, _ = multi_tensor_applier(
                    te.optimizers.multi_tensor_l2norm, overflow, [ref_grads], False
                )
                grad_scale = self.max_norm / (norm + 1.0e-6)
                grad_scale.clamp_max_(1.0)
                assert grad_scale.item() < 1.0
                multi_tensor_applier(
                    te.optimizers.multi_tensor_scale_tensor,
                    overflow,
                    [ref_grads, ref_grads],
                    grad_scale,
                )
            for ref_param, ref_grad in zip(ref_params, ref_grads):
                ref_param.grad = ref_grad
            ref_optim.step()

            before = {
                "master": master.clone(),
                "exp_avg": exp_avg.clone(),
                "exp_avg_sq": exp_avg_sq.clone(),
            }
            if param is not None:
                ref_param_buf = param.clone()
                for ref_param, (s, e, _) in zip(ref_params, segments):
                    ref_param_buf[s:e].copy_(ref_param.detach())
                before["param"] = param.clone()

            multi_tensor_adam_flat(
                grad,
                param,
                master,
                exp_avg,
                exp_avg_sq,
                segments_t,
                grad_scale,
                lr=[options["lr"] for options in self.group_options],
                beta1=[options["betas"][0] for options in self.group_options],
                beta2=[options["betas"][1] for options in self.group_options],
                epsilon=[options["eps"] for options in self.group_options],
                weight_decay=[options["weight_decay"] for options in self.group_options],
                step=[step] * len(self.group_options),
                bias_correction=[1] * len(self.group_options),
                mode=1 if adam_w_mode else 0,
            )

            for ref_param, (s, e, _) in zip(ref_params, segments):
                state = ref_optim.state[ref_param]
                assert torch.equal(_bits(master[s:e]), _bits(ref_param.detach()))
                assert torch.equal(_bits(exp_avg[s:e]), _bits(state["exp_avg"]))
                assert torch.equal(_bits(exp_avg_sq[s:e]), _bits(state["exp_avg_sq"]))
            if param is not None:
                assert torch.equal(_bits(param), _bits(ref_param_buf))
            # Padding is neither read nor written.
            for name, buf in (("master", master), ("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
                assert torch.equal(_bits(buf[is_gap]), _bits(before[name][is_gap])), name
            if param is not None:
                assert torch.equal(_bits(param[is_gap]), _bits(before["param"][is_gap]))

    def test_rejects_too_many_groups(self):
        buf = torch.zeros(8, device="cuda")
        segments = torch.tensor([[0, 8, 0]], dtype=torch.int64, device="cuda")
        num_groups = 33
        with pytest.raises(RuntimeError, match="parameter groups"):
            multi_tensor_adam_flat(
                buf,
                None,
                buf.clone(),
                buf.clone(),
                buf.clone(),
                segments,
                None,
                lr=[1e-3] * num_groups,
                beta1=[0.9] * num_groups,
                beta2=[0.999] * num_groups,
                epsilon=[1e-8] * num_groups,
                weight_decay=[0.0] * num_groups,
                step=[1] * num_groups,
                bias_correction=[1] * num_groups,
                mode=1,
            )


class TestFusedSGD(TestFusedOptimizer):

    def setup_method(self) -> None:
        super().setup_method()
        self.options = {"lr": 0.25, "momentum": 0.125}
        self.ref_optim = torch.optim.SGD
        self.fused_optim = te.optimizers.FusedSGD

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16)

    def test_empty_param_at_end_of_group(self):
        tensors = [
            torch.ones(4, dtype=torch.float, device="cuda"),
            torch.empty(0, dtype=torch.float, device="cuda"),
        ]
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(tensors, self.options)

        self.gen_grad(ref_param, tst_param)
        ref_optim.step()
        tst_optim.step()

        torch.testing.assert_close(ref_param, tst_param)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.reshape(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class TestAdamTest:

    def setup_method(self, *, seed: int = 0) -> None:
        torch.manual_seed(seed)

        self.model = Model().cuda()
        self.model_ = Model().cuda()
        self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

        self.lr = 0.00001
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def test_grad_scaler(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(params_, lr=self.lr, capturable=False)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        scaler_ = torch.amp.GradScaler("cuda", enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with gpu_autocast_ctx(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # DUT
            with gpu_autocast_ctx(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad,
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def test_grad_scaler_capturable(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(params_, lr=self.lr, capturable=True)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        scaler_ = torch.amp.GradScaler("cuda", enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with gpu_autocast_ctx(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # DUT
            with gpu_autocast_ctx(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad,
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def test_grad_scaler_capturable_master(self):
        # Cast conv layers to FP16
        for m in self.model_.modules():
            if m.__class__ in [torch.nn.Conv2d]:
                m.half()
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        master_weights = [p.float() for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(
            params_, lr=self.lr, capturable=True, master_weights=master_weights
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        scaler_ = torch.amp.GradScaler("cuda", enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with gpu_autocast_ctx(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # DUT
            with gpu_autocast_ctx(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight,
                        m_.weight.float(),
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad.float(),
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def test_native(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(params_, lr=self.lr, capturable=False)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            y = self.model(x)
            loss = ((gt - y) ** 2).mean()

            loss.backward()
            self.optimizer.step()

            # DUT
            y = self.model_(x)
            loss_ = ((gt_ - y) ** 2).mean()

            loss_.backward()
            optimizer_.step()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad,
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def test_large_tensor(self):
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.memory.mem_get_info()[0] < 60 * 1024**3:
            pytest.skip("Insufficient available memory")
        t = torch.zeros(2359332864, dtype=torch.half, device="cuda")
        t2 = torch.zeros(2359332864, dtype=torch.half, device="cuda")
        grad = torch.randn_like(t)
        t.grad = grad
        t2.grad = grad
        params = [t]
        params2 = [t2]
        optimizer = te.optimizers.FusedAdam(params, lr=self.lr)
        optimizer.step()
        optimizer2 = torch.optim.Adam(params2, lr=self.lr)
        torch.testing.assert_close(t, t2)
        torch.cuda.synchronize()
