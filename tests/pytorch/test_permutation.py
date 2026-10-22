# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import random

import torch
import pytest
from typing import Dict, List

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    moe_permute as te_permute,
    moe_permute_with_probs as te_permute_with_probs,
    moe_permute_and_pad_with_probs as te_permute_and_pad_with_probs,
    moe_unpermute as te_unpermute,
    moe_sort_chunks_by_index as te_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs as te_sort_chunks_by_index_with_probs,
)
from transformer_engine.pytorch import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    Float8BlockQuantizer,
    MXFP8Quantizer,
)
from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding
import copy

from utils import assert_close

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def pytorch_permute_index_map(tokens, indices, num_out_tokens: int = None):
    """
    Permute the tokens based on the indices. Token with the same index will be grouped together.
    The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.

    Args:
        tokens: torch.Tensor
            The input token tensor.
        indices: torch.Tensor
            The token to expert indices tensor, should have a shape of [num_tokens] or [num_tokens, topk].
        num_out_tokens: int, optional
            The effective output token count, when enabling the capacity factor, should equal the number of tokens not dropped.
            By default, set to None, meaning no tokens are dropped.

    Returns:
        torch.Tensor:
            The permuted tensor.
        torch.Tensor:
            The sorted_indices corresponding permuted tensor.
    """
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    num_out_tokens = num_out_tokens if num_out_tokens is not None else flatten_indices.size(0)

    permuted_tokens = tokens.index_select(0, sorted_indices[:num_out_tokens] // topk)
    return permuted_tokens, sorted_indices


def pytorch_unpermute_index_map(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
):
    """
    Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their
    corresponding probabilities.

    Args:
        permuted_tokens: torch.Tensor
            The tensor of permuted tokens to be unpermuted.
        sorted_indices: torch.Tensor
            The tensor of sorted indices used to unpermute the tokens.
        probs: torch.Tensor, optional
            The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will
            be merged with their respective probabilities.

    Returns:
        torch.Tensor:
            The unpermuted tokens, optionally merged with probabilities.
    """

    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = sorted_indices.size(0)
        topk = 1
    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )

    unpermuted_tokens.index_copy_(0, sorted_indices[: permuted_tokens.size(0)], permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)
    return unpermuted_tokens


def pytorch_permute_mask_map(tokens, routing_map):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def pytorch_unpermute_mask_map(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype
    )
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens


def pytorch_sort_chunks_by_index(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
):
    """
    Split and sort the input tensor based on the split_sizes and sorted indices.
    return a tuple of (output, row_id_map). row_id_map is only used when fused=True.
    """
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    return output


def dtype_tols(te_dtype: te.DType) -> Dict[str, float]:
    """Estimated tolerances for a datatype

    Based on tolerances for torch.testing.assert_close.

    """
    if te_dtype == te.DType.kFloat32:
        return dict(rtol=1.0e-6, atol=1.0e-6)
    if te_dtype == te.DType.kFloat16:
        return dict(rtol=3.0e-3, atol=1.0e-5)
    if te_dtype == te.DType.kBFloat16:
        return dict(rtol=2.0e-2, atol=1.0e-5)
    if te_dtype == te.DType.kFloat8E5M2 or te_dtype == te.DType.kFloat8E4M3:
        return dict(rtol=2.0e-1, atol=1.0e-1)
    raise ValueError(f"Unsuppored dtype ({te_dtype})")


def backward_wrapper(
    act, backward_input, forward_input=[], retain_graph=True, accumulate_grad=False
):
    # Set forward_input.grad to None to avoid grad accumulation.
    if accumulate_grad == False:
        for i in forward_input:
            i.grad = None
    return act.backward(backward_input, retain_graph=retain_graph)


def _maybe_compile(fn, use_torch_compile):
    """Wrap fn with torch.compile(fullgraph=True) if requested."""
    if use_torch_compile:
        torch._dynamo.reset()
        import torch._functorch.config as functorch_config

        functorch_config.donated_buffer = False
        return torch.compile(fn, fullgraph=True)
    return fn


def _test_permutation_index_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_probs,
    BENCHMARK=False,
    use_torch_compile=False,
):
    if not with_probs and topK > 1:
        pytest.skip("Only permutations with topK=1 and without probabilities are supported.")

    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "index map:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == te.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == te.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == te.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_permute_bwd_input = torch.rand((num_out_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_unpermute_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_permute_fwd_input.requires_grad_(True)

    if num_tokens > 0:
        indices = torch.stack([torch.randperm(num_expert)[:topK] for _ in range(num_tokens)])
    else:
        indices = torch.empty((num_tokens, topK))
    indices = indices.to(torch.int32).cuda()

    probs = None
    if with_probs:
        probs = torch.rand(num_tokens, topK).cuda()
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums
        probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_permute_output, sorted_indices = pytorch_permute_index_map(
        pytorch_permute_fwd_input, indices, num_out_tokens
    )
    pytorch_permute_output.backward(pytorch_permute_bwd_input, retain_graph=True)

    pytorch_unpermute_fwd_input = pytorch_permute_output.detach()
    pytorch_unpermute_fwd_input.requires_grad_(True)

    pytorch_unpermute_output = pytorch_unpermute_index_map(
        pytorch_unpermute_fwd_input, sorted_indices, probs=probs
    )
    pytorch_unpermute_output.backward(pytorch_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE Permutation
    #
    ###################################################################################################################################
    te_permute_fwd_input = pytorch_permute_fwd_input.detach()
    te_permute_fwd_input.requires_grad_(True)
    te_permute_bwd_input = pytorch_permute_bwd_input.detach()

    _permute = _maybe_compile(
        lambda inp, idx, num_out, max_token: te_permute(
            inp, idx, num_out, max_token, map_type="index"
        ),
        use_torch_compile,
    )
    te_permute_output, row_id_map = _permute(te_permute_fwd_input, indices, num_out_tokens, -1)
    te_permute_output.backward(te_permute_bwd_input, retain_graph=True)

    te_probs = None
    if with_probs:
        te_probs = probs.detach()
        te_probs.requires_grad_(True)
    te_unpermute_fwd_input = te_permute_output.detach()
    te_unpermute_fwd_input.requires_grad_(True)
    te_unpermute_bwd_input = pytorch_unpermute_bwd_input.detach()

    _unpermute = _maybe_compile(
        lambda inp, row_map, probs_val: te_unpermute(inp, row_map, probs_val, map_type="index"),
        use_torch_compile,
    )
    te_unpermute_output = _unpermute(te_unpermute_fwd_input, row_id_map, te_probs)
    te_unpermute_output.backward(te_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    te_permute_output_ = te_permute_output.float()
    te_permute_fwd_input_grad = te_permute_fwd_input.grad.float()
    te_unpermute_output_ = te_unpermute_output.float()
    te_unpermute_fwd_input_grad = te_unpermute_fwd_input.grad.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            pytorch_permute_output.float(),
            te_permute_output_,
            msg=f"Mismatch in te_permute fwd",
        )
        torch.testing.assert_close(
            pytorch_permute_fwd_input.grad.float(),
            te_permute_fwd_input_grad,
            msg=f"Mismatch in te_permute bwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_unpermute_output.float(),
            te_unpermute_output_,
            msg=f"Mismatch in te_unpermute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_unpermute_fwd_input.grad.float(),
            te_unpermute_fwd_input_grad,
            msg=f"Mismatch in te_unpermute bwd",
            **tols,
        )
        if with_probs:
            torch.testing.assert_close(
                probs.grad.float(),
                te_probs.grad.float(),
                msg=f"Mismatch in te_unpermute bwd",
                **tols,
            )

    if not pytorch_permute_fwd_input.numel():
        print("Empty pytorch_permute_fwd_input activation test passed.")
        return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        t1 = perf_test_cuda_kernel(
            lambda: pytorch_permute_index_map(pytorch_permute_fwd_input, indices, num_out_tokens)
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_permute(te_permute_fwd_input, indices, num_out_tokens, map_type="index")
        )
        print(f"permute\t\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_permute_output,
                pytorch_permute_bwd_input,
                forward_input=[pytorch_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_permute_output,
                te_permute_bwd_input,
                forward_input=[te_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"permute\t\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: pytorch_unpermute_index_map(
                pytorch_unpermute_fwd_input, sorted_indices, probs=probs
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_unpermute(te_unpermute_fwd_input, row_id_map, te_probs, map_type="index")
        )
        print(f"unpermute\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_unpermute_output,
                pytorch_unpermute_bwd_input,
                forward_input=(
                    [pytorch_unpermute_fwd_input, probs]
                    if with_probs
                    else [pytorch_unpermute_fwd_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_unpermute_output,
                te_unpermute_bwd_input,
                forward_input=(
                    [te_unpermute_fwd_input, te_probs] if with_probs else [te_unpermute_fwd_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"unpermute\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")


def _test_permutation_mask_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_probs,
    BENCHMARK=False,
    use_torch_compile=False,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "mask map:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == te.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == te.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == te.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_permute_bwd_input = torch.rand((num_out_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_unpermute_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_permute_fwd_input.requires_grad_(True)

    restore_shape = pytorch_permute_fwd_input.shape

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    probs = None
    if with_probs:
        probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums
        probs = probs.to(dtype)
        probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_permute_output, sorted_indices = pytorch_permute_mask_map(
        pytorch_permute_fwd_input, routing_map
    )
    pytorch_permute_output.backward(pytorch_permute_bwd_input, retain_graph=True)

    pytorch_unpermute_fwd_input = pytorch_permute_output.detach()
    pytorch_unpermute_fwd_input.requires_grad_(True)

    pytorch_unpermute_output = pytorch_unpermute_mask_map(
        pytorch_unpermute_fwd_input, sorted_indices, restore_shape, probs, routing_map
    )
    pytorch_unpermute_output.backward(pytorch_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE Permutation
    #
    ###################################################################################################################################
    te_permute_fwd_input = pytorch_permute_fwd_input.detach()
    te_permute_fwd_input.requires_grad_(True)
    te_permute_bwd_input = pytorch_permute_bwd_input.detach()

    _permute = _maybe_compile(
        lambda inp, rmap, n_out: te_permute(inp, rmap, num_out_tokens=n_out, map_type="mask"),
        use_torch_compile,
    )
    te_permute_output, row_id_map = _permute(te_permute_fwd_input, routing_map, num_out_tokens)
    te_permute_output.backward(te_permute_bwd_input, retain_graph=True)

    te_probs = None
    if with_probs:
        te_probs = probs.detach()
        te_probs.requires_grad_(True)
    te_unpermute_fwd_input = te_permute_output.detach()
    te_unpermute_fwd_input.requires_grad_(True)
    te_unpermute_bwd_input = pytorch_unpermute_bwd_input.detach()

    _unpermute = _maybe_compile(
        lambda inp, row_map, p, rs: te_unpermute(inp, row_map, p, rs, map_type="mask"),
        use_torch_compile,
    )
    te_unpermute_output = _unpermute(te_unpermute_fwd_input, row_id_map, te_probs, restore_shape)
    te_unpermute_output.backward(te_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    te_permute_output_ = te_permute_output.float()
    te_permute_fwd_input_grad = te_permute_fwd_input.grad.float()
    te_unpermute_output_ = te_unpermute_output.float()
    te_unpermute_fwd_input_grad = te_unpermute_fwd_input.grad.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            pytorch_permute_output.float(),
            te_permute_output_,
            msg=f"Mismatch in te_permute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_permute_fwd_input.grad.float(),
            te_permute_fwd_input_grad,
            msg=f"Mismatch in te_permute bwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_unpermute_output.float(),
            te_unpermute_output_,
            msg=f"Mismatch in te_unpermute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_unpermute_fwd_input.grad.float(),
            te_unpermute_fwd_input_grad,
            msg=f"Mismatch in te_unpermute bwd",
            **tols,
        )
        if with_probs:
            torch.testing.assert_close(
                probs.grad.float(),
                te_probs.grad.float(),
                msg=f"Mismatch in te_unpermute bwd",
                **tols,
            )

    if not pytorch_permute_fwd_input.numel():
        print("Empty pytorch_permute_fwd_input activation test passed.")
        return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        t1 = perf_test_cuda_kernel(
            lambda: pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_permute(
                te_permute_fwd_input, routing_map, num_out_tokens=num_out_tokens, map_type="mask"
            )
        )
        print(f"permute\t\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_permute_output,
                pytorch_permute_bwd_input,
                forward_input=[pytorch_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_permute_output,
                te_permute_bwd_input,
                forward_input=[te_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"permute\t\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: pytorch_unpermute_mask_map(
                pytorch_unpermute_fwd_input, sorted_indices, restore_shape, probs, routing_map
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_unpermute(
                te_unpermute_fwd_input, row_id_map, te_probs, restore_shape, map_type="mask"
            )
        )
        print(f"unpermute\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_unpermute_output,
                pytorch_unpermute_bwd_input,
                forward_input=(
                    [pytorch_unpermute_fwd_input, probs]
                    if with_probs
                    else [pytorch_unpermute_fwd_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_unpermute_output,
                te_unpermute_bwd_input,
                forward_input=(
                    [te_unpermute_fwd_input, te_probs] if with_probs else [te_unpermute_fwd_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"unpermute\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")


def _test_permutation_and_padding_mask_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_merging_probs=False,
    align_size=16,
    BENCHMARK=False,
    use_torch_compile=False,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens is None:
        num_out_tokens = num_tokens * topK

    print(
        "permutation and padding:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK}"
        f" with_merging_probs:{with_merging_probs} align_size:{align_size} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == te.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == te.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == te.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    probs = probs.to(dtype)
    probs.requires_grad_(True)

    tokens_per_expert = routing_map.sum(dim=0).cpu()
    target_tokens_per_expert = (torch.ceil(tokens_per_expert / align_size) * align_size).long()
    num_permute_pad_out_tokens = target_tokens_per_expert.sum().item()

    permute_pad_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    permute_pad_bwd_input = torch.rand(
        (num_permute_pad_out_tokens, hidden_size), dtype=dtype
    ).cuda()
    unpermute_unpad_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    permute_pad_fwd_input.requires_grad_(True)

    restore_shape = permute_pad_fwd_input.shape
    ###################################################################################################################################
    #
    # moe_permute_with_probs and Fp8Padding, moe_unpermute and Fp8Unpadding
    #
    ###################################################################################################################################
    # permute + padding
    permuted_output, permuted_probs, row_id_map = te_permute_with_probs(
        permute_pad_fwd_input,
        probs,
        routing_map,
        num_out_tokens=num_out_tokens,
    )
    tokens_per_expert_list = tokens_per_expert.tolist()
    fp8_padding = Fp8Padding(num_expert, align_size)
    permuted_paded_output, _ = fp8_padding(permuted_output, tokens_per_expert_list)
    permuted_paded_probs, _ = fp8_padding(permuted_probs.unsqueeze(-1), tokens_per_expert_list)

    permuted_paded_output.backward(permute_pad_bwd_input, retain_graph=True)

    # unpadding + unpermute

    unpermute_unpad_fwd_input = permuted_paded_output.detach()
    unpermute_unpad_fwd_input.requires_grad_(True)

    fp8_unpadding = Fp8Unpadding(num_expert, align_size)
    unpaded_output = fp8_unpadding(unpermute_unpad_fwd_input, tokens_per_expert_list)

    probs_naive = probs
    unpermuted_unpaded_output = te_unpermute(
        unpaded_output,
        row_id_map,
        merging_probs=probs_naive if with_merging_probs else None,
        restore_shape=restore_shape,
    )

    unpermuted_unpaded_output.backward(unpermute_unpad_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # fusion moe_permute_with_probs and Fp8Padding, fusion fusion moe_unpermute and Fp8Unpadding
    #
    ###################################################################################################################################
    # fusion permute_and_pad
    fusion_permute_and_pad_fwd_input = permute_pad_fwd_input.detach()
    fusion_permute_and_pad_fwd_input.requires_grad_(True)
    probs_fusion = probs_naive.detach().clone()
    probs_fusion.requires_grad_(True)

    (
        fusion_permuted_padded_output,
        fusion_permuted_padded_probs,
        row_id_map,
        pad_offsets,
        target_tokens_per_expert,
    ) = te_permute_and_pad_with_probs(
        fusion_permute_and_pad_fwd_input,
        probs_fusion,
        routing_map,
        tokens_per_expert,
        align_size,
    )
    fusion_permuted_padded_probs = fusion_permuted_padded_probs.unsqueeze(-1)

    fusion_permute_pad_bwd_input = permute_pad_bwd_input.detach()
    fusion_permuted_padded_output.backward(fusion_permute_pad_bwd_input, retain_graph=True)

    # fusion unpad and unpermute
    fusion_unpermute_unpad_fwd_input = fusion_permuted_padded_output.detach()
    fusion_unpermute_unpad_fwd_input.requires_grad_(True)

    fusion_unpermuted_unpaded_output = te_unpermute(
        fusion_unpermute_unpad_fwd_input,
        row_id_map,
        merging_probs=probs_fusion if with_merging_probs else None,
        restore_shape=restore_shape,
        pad_offsets=pad_offsets,
    )

    fusion_unpermute_bwd_input = unpermute_unpad_bwd_input.detach()
    fusion_unpermuted_unpaded_output.backward(fusion_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    permuted_paded_output_ = permuted_paded_output.float()
    fusion_permuted_padded_output_ = fusion_permuted_padded_output.float()
    permute_pad_fwd_input_grad = permute_pad_fwd_input.grad.float()
    fusion_permute_and_pad_fwd_input_grad = fusion_permute_and_pad_fwd_input.grad.float()

    unpermuted_unpaded_output_ = unpermuted_unpaded_output.float()
    fusion_unpermuted_unpaded_output_ = fusion_unpermuted_unpaded_output.float()
    unpermute_unpad_fwd_input_grad = unpermute_unpad_fwd_input.grad.float()
    fusion_unpermute_unpad_fwd_input_grad = fusion_unpermute_unpad_fwd_input.grad.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            permuted_paded_output_,
            fusion_permuted_padded_output_,
            msg=f"Mismatch in te_permute_and_pad fwd",
            **tols,
        )
        torch.testing.assert_close(
            permute_pad_fwd_input_grad,
            fusion_permute_and_pad_fwd_input_grad,
            msg=f"Mismatch in te_permute_and_pad bwd",
            **tols,
        )
        torch.testing.assert_close(
            unpermuted_unpaded_output_,
            fusion_unpermuted_unpaded_output_,
            msg=f"Mismatch in te_unpermute fwd",
            **tols,
        )
        torch.testing.assert_close(
            unpermute_unpad_fwd_input_grad,
            fusion_unpermute_unpad_fwd_input_grad,
            msg=f"Mismatch in te_unpermute bwd",
            **tols,
        )
        torch.testing.assert_close(
            permuted_paded_probs.float(),
            fusion_permuted_padded_probs.float(),
            msg=f"Mismatch in te_permute_and_pad bwd",
            **tols,
        )
        if with_merging_probs:
            torch.testing.assert_close(
                probs_naive.grad.float(),
                probs_fusion.grad.float(),
                msg=f"Mismatch in te_unpermute bwd",
                **tols,
            )

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:

        def permute_and_pad():
            permuted_output, permuted_probs, row_id_map = te_permute_with_probs(
                permute_pad_fwd_input,
                probs,
                routing_map,
                num_out_tokens=num_out_tokens,
            )
            fp8_padding(permuted_output, tokens_per_expert_list)
            fp8_padding(permuted_probs.unsqueeze(-1), tokens_per_expert_list)

        def fusion_permute_and_pad():
            (
                fusion_permuted_padded_output,
                fusion_permuted_padded_probs,
                row_id_map,
                pad_offsets,
                target_tokens_per_expert,
            ) = te_permute_and_pad_with_probs(
                fusion_permute_and_pad_fwd_input,
                probs,
                routing_map,
                tokens_per_expert,
                align_size,
            )
            fusion_permuted_padded_probs = fusion_permuted_padded_probs.unsqueeze(-1)

        t1 = perf_test_cuda_kernel(lambda: permute_and_pad())

        t2 = perf_test_cuda_kernel(lambda: fusion_permute_and_pad())

        print(f"permute_and_pad\t\tfwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                permuted_paded_output,
                permute_pad_bwd_input,
                forward_input=[permute_pad_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                fusion_permuted_padded_output,
                fusion_permute_pad_bwd_input,
                forward_input=[fusion_permute_and_pad_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"permute_and_pad\t\tbwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")

        def unpad_unpermute():
            unpaded_output = fp8_unpadding(unpermute_unpad_fwd_input, tokens_per_expert_list)
            unpermuted_unpaded_output = te_unpermute(
                unpaded_output, row_id_map, restore_shape=restore_shape
            )

            unpermuted_unpaded_output.backward(unpermute_unpad_bwd_input, retain_graph=True)

        t1 = perf_test_cuda_kernel(lambda: unpad_unpermute())
        t2 = perf_test_cuda_kernel(
            lambda: te_unpermute(
                fusion_unpermute_unpad_fwd_input,
                row_id_map,
                restore_shape=restore_shape,
                pad_offsets=pad_offsets,
            )
        )
        print(f"unpermute_and_unpad\tfwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                unpermuted_unpaded_output,
                unpermute_unpad_bwd_input,
                forward_input=([unpermute_unpad_fwd_input, probs]),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                fusion_unpermuted_unpaded_output,
                fusion_unpermute_bwd_input,
                forward_input=([fusion_unpermute_unpad_fwd_input, probs]),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"unpermute_and_unpad\tbwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")


def _test_permutation_and_padding_with_merging_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    align_size=16,
    BENCHMARK=False,
    use_torch_compile=False,
):
    """
    Test the combination of merging_probs AND pad_offsets together in moe_unpermute.
    This specifically tests the backward pass fix where pad_offsets must be used
    when computing gradients with merging_probs.
    """
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "permutation and padding with merging probs:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} align_size:{align_size} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == te.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == te.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == te.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    probs = probs.to(dtype)
    probs.requires_grad_(True)

    tokens_per_expert = routing_map.sum(dim=0).cpu()
    target_tokens_per_expert = (torch.ceil(tokens_per_expert / align_size) * align_size).long()
    num_permute_pad_out_tokens = target_tokens_per_expert.sum().item()

    permute_pad_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    permute_pad_bwd_input = torch.rand(
        (num_permute_pad_out_tokens, hidden_size), dtype=dtype
    ).cuda()
    unpermute_unpad_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    permute_pad_fwd_input.requires_grad_(True)

    restore_shape = permute_pad_fwd_input.shape
    ###################################################################################################################################
    #
    # Reference: moe_permute_with_probs + Fp8Padding, then Fp8Unpadding + moe_unpermute with merging_probs
    #
    ###################################################################################################################################
    # permute + padding
    permuted_output, permuted_probs, row_id_map = te_permute_with_probs(
        permute_pad_fwd_input,
        probs,
        routing_map,
        num_out_tokens=num_out_tokens,
    )
    tokens_per_expert_list = tokens_per_expert.tolist()
    fp8_padding = Fp8Padding(num_expert, align_size)
    permuted_paded_output, _ = fp8_padding(permuted_output, tokens_per_expert_list)

    permuted_paded_output.backward(permute_pad_bwd_input, retain_graph=True)

    # Reference: unpadding + unpermute WITH merging_probs
    ref_unpermute_fwd_input = permuted_paded_output.detach()
    ref_unpermute_fwd_input.requires_grad_(True)

    ref_probs = probs.detach()
    ref_probs.requires_grad_(True)

    fp8_unpadding = Fp8Unpadding(num_expert, align_size)
    unpaded_output = fp8_unpadding(ref_unpermute_fwd_input, tokens_per_expert_list)
    ref_unpermuted_output = te_unpermute(
        unpaded_output, row_id_map, ref_probs, restore_shape=restore_shape
    )

    ref_unpermuted_output.backward(unpermute_unpad_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Fused: moe_permute_and_pad_with_probs, then moe_unpermute with BOTH merging_probs AND pad_offsets
    #
    ###################################################################################################################################
    # fusion permute_and_pad
    fusion_permute_fwd_input = permute_pad_fwd_input.detach()
    fusion_permute_fwd_input.requires_grad_(True)
    fusion_probs = probs.detach()
    fusion_probs.requires_grad_(True)

    (
        fusion_permuted_padded_output,
        fusion_permuted_padded_probs,
        fused_row_id_map,
        pad_offsets,
        _,
    ) = te_permute_and_pad_with_probs(
        fusion_permute_fwd_input,
        fusion_probs,
        routing_map,
        tokens_per_expert,
        align_size,
    )

    fusion_permute_pad_bwd_input = permute_pad_bwd_input.detach()
    fusion_permuted_padded_output.backward(fusion_permute_pad_bwd_input, retain_graph=True)

    # Fused: unpermute with BOTH merging_probs AND pad_offsets
    fusion_unpermute_fwd_input = fusion_permuted_padded_output.detach()
    fusion_unpermute_fwd_input.requires_grad_(True)

    fusion_merging_probs = probs.detach()
    fusion_merging_probs.requires_grad_(True)

    fusion_unpermuted_output = te_unpermute(
        fusion_unpermute_fwd_input,
        fused_row_id_map,
        fusion_merging_probs,
        restore_shape=restore_shape,
        pad_offsets=pad_offsets,
    )

    fusion_unpermute_bwd_input = unpermute_unpad_bwd_input.detach()
    fusion_unpermuted_output.backward(fusion_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    # Check forward pass
    ref_unpermuted_output_ = ref_unpermuted_output.float()
    fusion_unpermuted_output_ = fusion_unpermuted_output.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            ref_unpermuted_output_,
            fusion_unpermuted_output_,
            msg=f"Mismatch in te_unpermute with merging_probs and pad_offsets fwd",
            **tols,
        )

        # Check backward pass - activation gradients
        ref_unpermute_fwd_input_grad = ref_unpermute_fwd_input.grad.float()
        fusion_unpermute_fwd_input_grad = fusion_unpermute_fwd_input.grad.float()

        torch.testing.assert_close(
            ref_unpermute_fwd_input_grad,
            fusion_unpermute_fwd_input_grad,
            msg=f"Mismatch in te_unpermute with merging_probs and pad_offsets bwd (act_grad)",
            **tols,
        )

        # Check backward pass - probs gradients
        ref_probs_grad = ref_probs.grad.float()
        fusion_probs_grad = fusion_merging_probs.grad.float()

        torch.testing.assert_close(
            ref_probs_grad,
            fusion_probs_grad,
            msg=f"Mismatch in te_unpermute with merging_probs and pad_offsets bwd (probs_grad)",
            **tols,
        )

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:

        def ref_unpad_unpermute():
            unpaded = fp8_unpadding(ref_unpermute_fwd_input, tokens_per_expert_list)
            return te_unpermute(unpaded, row_id_map, ref_probs, restore_shape=restore_shape)

        def fused_unpermute():
            return te_unpermute(
                fusion_unpermute_fwd_input,
                fused_row_id_map,
                fusion_merging_probs,
                restore_shape=restore_shape,
                pad_offsets=pad_offsets,
            )

        t1 = perf_test_cuda_kernel(lambda: ref_unpad_unpermute())
        t2 = perf_test_cuda_kernel(lambda: fused_unpermute())
        print(f"unpermute_unpad_with_probs\tfwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                ref_unpermuted_output,
                unpermute_unpad_bwd_input,
                forward_input=[ref_unpermute_fwd_input, ref_probs],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                fusion_unpermuted_output,
                fusion_unpermute_bwd_input,
                forward_input=[fusion_unpermute_fwd_input, fusion_merging_probs],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"unpermute_unpad_with_probs\tbwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")


def _test_permutation_mask_map_fp8(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    recipe,
    use_torch_compile=False,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    if recipe.delayed():
        quantizer = Float8Quantizer(
            scale=torch.full([1], 1.0).cuda().squeeze(),
            amax=torch.full([1], 1.0).cuda(),
            fp8_dtype=te_dtype,
        )
    elif recipe.float8_current_scaling():
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=te_dtype,
            device=torch.device("cuda"),
            columnwise=False,
        )
    elif recipe.float8_block_scaling():
        quantizer = Float8BlockQuantizer(
            fp8_dtype=te_dtype,
            rowwise=True,
            columnwise=False,
            amax_epsilon=0.0,
            force_pow_2_scales=True,  # Fp8 sub-channel a2a requires e8 scales
            block_scaling_dim=1,  # 1x128 scaling
        )
    elif recipe.mxfp8():
        quantizer = MXFP8Quantizer(
            fp8_dtype=te_dtype,
            rowwise=True,
            columnwise=False,
        )
    else:
        raise ValueError("Unsupported FP8 recipe")

    permute_fwd_input = torch.rand(
        size=(num_tokens, hidden_size), dtype=torch.float32, device="cuda"
    )
    # Make an empty fp8 tensor
    permute_fwd_input_fp8 = quantizer.make_empty(
        permute_fwd_input.shape,
        dtype=permute_fwd_input.dtype,
        device=permute_fwd_input.device,
    )
    # quantize the tensor
    quantizer.update_quantized(permute_fwd_input, permute_fwd_input_fp8)
    if recipe.float8_block_scaling():
        pytorch_permute_fwd_input = copy.deepcopy(permute_fwd_input_fp8._rowwise_data)
        pytorch_permute_fwd_scale_input = copy.deepcopy(
            permute_fwd_input_fp8._rowwise_scale_inv.T.contiguous()
        )
    elif recipe.mxfp8():
        pytorch_permute_fwd_input = copy.deepcopy(permute_fwd_input_fp8._rowwise_data)
        pytorch_permute_fwd_scale_input = copy.deepcopy(
            permute_fwd_input_fp8._rowwise_scale_inv.contiguous()
        )
    else:
        pytorch_permute_fwd_input = copy.deepcopy(permute_fwd_input_fp8._data)
        pytorch_permute_fwd_scale_input = None

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    # PyTorch Permutaion
    pytorch_permute_output, _ = pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
    if pytorch_permute_fwd_scale_input is not None:
        pytorch_permute_scale_output, _ = pytorch_permute_mask_map(
            pytorch_permute_fwd_scale_input, routing_map
        )

    # TE Permutation
    _permute = _maybe_compile(
        lambda inp, rmap, n_out: te_permute(inp, rmap, num_out_tokens=n_out, map_type="mask"),
        use_torch_compile,
    )
    permute_output, _ = _permute(permute_fwd_input_fp8, routing_map, num_out_tokens)
    if recipe.float8_block_scaling():
        te_permute_output = permute_output._rowwise_data
        te_permute_scale_output = permute_output._rowwise_scale_inv.T.contiguous()
    elif recipe.mxfp8():
        te_permute_output = permute_output._rowwise_data
        te_permute_scale_output = permute_output._rowwise_scale_inv.contiguous()
    else:
        te_permute_output = permute_output._data
        te_permute_scale_output = None

    # check the permute output
    torch.testing.assert_close(
        pytorch_permute_output,
        te_permute_output,
        atol=0,
        rtol=0,
    )
    if recipe.float8_block_scaling() or recipe.mxfp8():
        torch.testing.assert_close(
            pytorch_permute_scale_output,
            te_permute_scale_output,
            atol=0,
            rtol=0,
        )


def _test_moe_chunk_sort(
    te_dtype,
    num_tokens,
    num_expert,
    tp_size,
    hidden_size,
    BENCHMARK=False,
    use_torch_compile=False,
):
    print(
        "chunk permute:"
        f" token:{num_tokens} hidden_size:{hidden_size} num_expert:{num_expert} tp_size:{tp_size} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == te.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == te.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == te.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    pytorch_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_fwd_input.requires_grad_(True)

    _split_sizes = [0] * (num_expert * tp_size)
    for _ in range(num_tokens):
        idx = random.randint(0, num_expert * tp_size - 1)
        _split_sizes[idx] += 1
    split_sizes = torch.tensor(_split_sizes, dtype=torch.int32).ravel()
    split_sizes_cuda = split_sizes.to(device="cuda")

    _sorted_idxs = torch.arange(num_expert * tp_size, dtype=torch.int32)
    sorted_idxs = _sorted_idxs.reshape(tp_size, num_expert).T.ravel()
    sorted_idxs_cuda = sorted_idxs.to(device="cuda")

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_output = pytorch_sort_chunks_by_index(pytorch_fwd_input, split_sizes, sorted_idxs)
    pytorch_output.backward(pytorch_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE Permutation
    #
    ###################################################################################################################################
    te_fwd_input = pytorch_fwd_input.detach()
    te_fwd_input.requires_grad_(True)
    te_bwd_input = pytorch_bwd_input.detach()

    _sort = _maybe_compile(
        lambda inp, ss, si: te_sort_chunks_by_index(inp, ss, si),
        use_torch_compile,
    )
    te_output = _sort(te_fwd_input, split_sizes_cuda, sorted_idxs_cuda)
    te_output.backward(te_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    te_output_ = te_output.float()
    te_fwd_input_grad = te_fwd_input.grad.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            pytorch_output.float(),
            te_output_,
            msg=f"Mismatch in te_permute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_fwd_input.grad.float(),
            te_fwd_input_grad,
            msg=f"Mismatch in te_permute bwd",
            **tols,
        )

    if not pytorch_fwd_input.numel():
        print("Empty pytorch_fwd_input activation test passed.")
        return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        t1 = perf_test_cuda_kernel(
            lambda: pytorch_sort_chunks_by_index(pytorch_fwd_input, split_sizes, sorted_idxs)
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_sort_chunks_by_index(te_fwd_input, split_sizes_cuda, sorted_idxs_cuda)
        )
        print(f"chunk sort\t\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_output,
                pytorch_bwd_input,
                forward_input=[pytorch_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_output,
                te_bwd_input,
                forward_input=[te_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"chunk sort\t\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")


def _test_permutation_mask_map_alongside_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    tp_size,
    BENCHMARK=False,
    use_torch_compile=False,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "mask map alongside probs:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == te.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == te.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == te.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_unpermute_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_permute_fwd_input.requires_grad_(True)

    restore_shape = pytorch_permute_fwd_input.shape

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    probs = probs.to(dtype)
    probs.requires_grad_(True)

    split_sizes = [0] * (num_expert * tp_size)
    for i in range(num_out_tokens):
        idx = random.randint(0, num_expert * tp_size - 1)
        split_sizes[idx] += 1
    split_sizes = torch.tensor(split_sizes, dtype=torch.int32)
    split_sizes_cuda = split_sizes.to(device="cuda")

    _sorted_idxs = torch.arange(num_expert * tp_size, dtype=torch.int32)
    sorted_idxs = _sorted_idxs.reshape(tp_size, num_expert).T.ravel()
    sorted_idxs_cuda = sorted_idxs.to(device="cuda")

    split_sizes_2 = [split_sizes[i] for i in sorted_idxs.tolist()]
    split_sizes_2 = torch.tensor(split_sizes_2, dtype=torch.int32)
    split_sizes_2_cuda = split_sizes_2.to(device="cuda")

    sorted_idxs_2 = [0] * (num_expert * tp_size)
    for i in range(num_expert * tp_size):
        sorted_idxs_2[sorted_idxs[i]] = i
    sorted_idxs_2 = torch.tensor(sorted_idxs_2, dtype=torch.int32)
    sorted_idxs_2_cuda = sorted_idxs_2.to(device="cuda")

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_permute_output, sorted_indices = pytorch_permute_mask_map(
        pytorch_permute_fwd_input, routing_map
    )

    pytorch_permute_output = pytorch_sort_chunks_by_index(
        pytorch_permute_output, split_sizes, sorted_idxs
    )

    pytorch_permute_output = pytorch_sort_chunks_by_index(
        pytorch_permute_output, split_sizes_2, sorted_idxs_2
    )

    pytorch_unpermute_output = pytorch_unpermute_mask_map(
        pytorch_permute_output, sorted_indices, restore_shape, probs, routing_map
    )
    pytorch_unpermute_output.backward(pytorch_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE Permutation
    #
    ###################################################################################################################################
    te_permute_fwd_input = pytorch_permute_fwd_input.detach()
    te_permute_fwd_input.requires_grad_(True)

    te_unpermute_bwd_input = pytorch_unpermute_bwd_input.detach()
    te_probs = probs.detach()
    te_probs.requires_grad_(True)

    def _alongside_probs_fn(fwd_inp, t_probs, rmap, ss1, si1, ss2, si2):
        out, pprobs, rid = te_permute_with_probs(
            fwd_inp, t_probs, rmap, num_out_tokens=num_out_tokens
        )
        out, pprobs = te_sort_chunks_by_index_with_probs(out, pprobs, ss1, si1)
        out_dtype = out.dtype
        out = out * pprobs.unsqueeze(-1)
        out = out.to(dtype=out_dtype)
        out = te_sort_chunks_by_index(out, ss2, si2)
        out = te_unpermute(out, rid, restore_shape=restore_shape, map_type="mask")
        return out

    _fn = _maybe_compile(_alongside_probs_fn, use_torch_compile)
    te_unpermute_output = _fn(
        te_permute_fwd_input,
        te_probs,
        routing_map,
        split_sizes_cuda,
        sorted_idxs_cuda,
        split_sizes_2_cuda,
        sorted_idxs_2_cuda,
    )
    te_unpermute_output.backward(te_unpermute_bwd_input, retain_graph=True)

    ###############################################################################################

    tols = dtype_tols(te_dtype)

    te_permute_fwd_input_grad = te_permute_fwd_input.grad.float()
    te_unpermute_output_ = te_unpermute_output.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            pytorch_unpermute_output.float(),
            te_unpermute_output_,
            msg=f"Mismatch in fused_unpermute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_permute_fwd_input.grad.float(),
            te_permute_fwd_input_grad,
            msg=f"Mismatch in fused_permute bwd",
            **tols,
        )
        torch.testing.assert_close(
            probs.grad.float(), te_probs.grad.float(), msg=f"Mismatch in prob grad", **tols
        )

    if BENCHMARK:
        t1 = perf_test_cuda_kernel(
            lambda: te_permute_with_probs(
                te_permute_fwd_input, te_probs, routing_map, num_out_tokens=num_out_tokens
            )
        )
        print(f"permute\t\tfwd: TE: {t1:.3f} ms")

        te_permute_output, te_permuted_probs, row_id_map = te_permute_with_probs(
            te_permute_fwd_input,
            te_probs,
            routing_map,
            num_out_tokens=num_out_tokens,
        )
        te_permute_bwd_input = torch.rand((num_out_tokens, hidden_size), dtype=dtype).cuda()
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_permute_output,
                te_permute_bwd_input,
                forward_input=[te_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"permute\t\tbwd: TE: {t2:.3f} ms")

        chunk_sort_fwd_input = te_permute_output.detach()
        chunk_sort_fwd_input.requires_grad_(True)
        chunk_sort_fwd_probs = te_permuted_probs.detach()
        chunk_sort_fwd_probs.requires_grad_(True)
        t1 = perf_test_cuda_kernel(
            lambda: te_sort_chunks_by_index_with_probs(
                chunk_sort_fwd_input, chunk_sort_fwd_probs, split_sizes_cuda, sorted_idxs_cuda
            )
        )
        print(f"chunk sort\t\tfwd: TE: {t1:.3f} ms")

        chunk_sort_output, _ = te_sort_chunks_by_index_with_probs(
            chunk_sort_fwd_input, chunk_sort_fwd_probs, split_sizes_cuda, sorted_idxs_cuda
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                chunk_sort_output,
                te_permute_bwd_input,
                forward_input=[chunk_sort_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"chunk sort\t\tbwd: TE: {t2:.3f} ms")


def perf_test_cuda_kernel(cuda_kernel_fn):
    if torch.cuda.is_available():
        # create CUDA event
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # warmup
        for _ in range(50):
            cuda_kernel_fn()

        start_event.record()
        for _ in range(100):
            cuda_kernel_fn()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms / 100
    else:
        pytest.skip("CUDA is not available.")


# TE tensor dtypes
_te_dtypes: List[te.DType] = [te.DType.kFloat32, te.DType.kFloat16]
if te.is_bf16_available():
    _te_dtypes.append(te.DType.kBFloat16)


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_index_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    use_torch_compile,
):
    if use_torch_compile and (num_expert != 7 or topK != 2):
        pytest.skip("torch.compile tested with single config only")
    with_probs = True
    BENCHMARK = False

    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_index_map_empty_input(te_dtype, use_torch_compile):
    with_probs = True
    BENCHMARK = False

    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=0,
        num_expert=8,
        hidden_size=4096,
        topK=2,
        num_out_tokens=0,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_mask_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    use_torch_compile,
):
    if use_torch_compile and (num_expert != 7 or topK != 2):
        pytest.skip("torch.compile tested with single config only")
    with_probs = True
    BENCHMARK = False

    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16, torch.bfloat16])
# 4096 tokens span several 1024-token blocks of the row-map kernels
@pytest.mark.parametrize("num_tokens", [64, 4096])
def test_permutation_mask_map_capacity_drop(torch_dtype, num_tokens):
    """num_out_tokens smaller than routing_map.sum(): over-capacity entries must be
    dropped (as the CUDA index-map path does), not written past the output buffer."""
    num_expert, hidden_size = 8, 128
    torch.manual_seed(1234)

    routing_map = (torch.rand(num_tokens, num_expert) > 0.5).bool().cuda()
    total_routed = int(routing_map.sum())
    capacity = total_routed - 17
    assert capacity > 0

    inp = torch.randn((num_tokens, hidden_size)).cuda().to(torch_dtype)

    out, row_id_map = te_permute(inp, routing_map, num_out_tokens=capacity, map_type="mask")

    # reference: expert-major order of routed (expert, token) pairs, first
    # `capacity` destination rows kept
    token_idx = torch.nonzero(routing_map.T.contiguous(), as_tuple=False)[:, 1]
    ref = inp[token_idx[:capacity]]
    assert out.shape[0] == capacity
    torch.testing.assert_close(out.float(), ref.float(), atol=0, rtol=0)

    # the row map must mark exactly the over-capacity entries as dropped
    n_routed = row_id_map[:, 2 * num_expert]
    assert int(n_routed.sum()) == capacity

    # unpermute consumes the same map: each token gets the sum of its kept copies
    unperm = te_unpermute(out, row_id_map, restore_shape=inp.shape, map_type="mask")
    kept_flat = torch.zeros(num_expert * num_tokens, dtype=torch.bool, device=inp.device)
    routed_flat_idx = torch.nonzero(routing_map.T.contiguous().flatten(), as_tuple=False)[:, 0]
    kept_flat[routed_flat_idx[:capacity]] = True
    kept_map = kept_flat.view(num_expert, num_tokens).T.contiguous()
    ref_unperm = (inp.float().unsqueeze(1) * kept_map.unsqueeze(-1).float()).sum(1)
    assert_close(unperm, ref_unperm)


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_out_tokens", [None])
@pytest.mark.parametrize(
    "num_tokens, num_expert, hidden_size, topK",
    [
        (4096, 8, 1280, 2),
        (4096, 64, 4096, 6),
        (4096, 256, 7168, 6),
        (4096, 512, 9216, 8),
    ],
)
@pytest.mark.parametrize("with_merging_probs", [True, False])
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_and_padding_mask_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_merging_probs,
    use_torch_compile,
):
    if use_torch_compile and (num_expert != 8 or topK != 2):
        pytest.skip("torch.compile tested with single config only")
    BENCHMARK = False

    _test_permutation_and_padding_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_merging_probs=with_merging_probs,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_out_tokens", [None])
@pytest.mark.parametrize(
    "num_tokens, num_expert, hidden_size, topK",
    [
        (4096, 8, 1280, 2),
        (4096, 64, 4096, 6),
        (4096, 256, 7168, 6),
        (4096, 512, 9216, 8),
    ],
)
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_and_padding_with_merging_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    use_torch_compile,
):
    """Test moe_unpermute backward pass with BOTH merging_probs AND pad_offsets."""
    if use_torch_compile and (num_expert != 8 or topK != 2):
        pytest.skip("torch.compile tested with single config only")
    BENCHMARK = False

    _test_permutation_and_padding_with_merging_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_mask_map_empty_input(te_dtype, use_torch_compile):
    with_probs = True
    BENCHMARK = False

    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=0,
        num_expert=8,
        hidden_size=4096,
        topK=2,
        num_out_tokens=0,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_mask_map_alongside_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    tp_size,
    use_torch_compile,
):
    if use_torch_compile and (num_expert != 7 or topK != 2 or tp_size != 1):
        pytest.skip("torch.compile tested with single config only")
    _test_permutation_mask_map_alongside_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        tp_size=tp_size,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_mask_map_alongside_probs_empty_input(te_dtype, use_torch_compile):
    _test_permutation_mask_map_alongside_probs(
        te_dtype=te_dtype,
        num_tokens=0,
        num_expert=8,
        hidden_size=4096,
        topK=2,
        num_out_tokens=0,
        tp_size=2,
        use_torch_compile=use_torch_compile,
    )


# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
fp8_recipes = [
    recipe.MXFP8BlockScaling(),
    recipe.DelayedScaling(),
    recipe.Float8CurrentScaling(),
    recipe.Float8BlockScaling(),
]


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("te_dtype", [te.DType.kFloat8E4M3, te.DType.kFloat8E5M2])
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
@pytest.mark.parametrize("recipe", fp8_recipes)
def test_permutation_mask_map_fp8(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    recipe,
):
    if recipe.mxfp8() and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if recipe.float8_block_scaling() and not fp8_block_scaling_available:
        pytest.skip(reason_for_no_fp8_block_scaling)

    _test_permutation_mask_map_fp8(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        recipe=recipe,
        use_torch_compile=False,  # FP8 permutation is not yet supported under torch.compile
    )


def _padded_permutation_row_index(routing_map, align_size, num_rows):
    """Source token of every row of the expert-major, per-expert padded permutation that
    moe_permute_and_pad_with_probs produces (-1 on padding rows and on the unused tail)."""
    num_tokens, num_expert = routing_map.shape
    routed = routing_map.to(torch.int64)
    counts = routed.sum(dim=0)
    padded_counts = (counts + align_size - 1) // align_size * align_size
    starts = torch.cumsum(padded_counts, dim=0) - padded_counts
    dest = starts.unsqueeze(0) + torch.cumsum(routed, dim=0) - routed
    tokens = torch.arange(num_tokens, device=routing_map.device).unsqueeze(1).expand_as(dest)
    row_index = torch.full((num_rows,), -1, dtype=torch.int32, device=routing_map.device)
    row_index[dest[routing_map]] = tokens[routing_map].to(torch.int32)
    return row_index, padded_counts


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "num_tokens, num_expert, hidden_size, topK, align_size, tail_rows",
    [
        (4096, 32, 2880, 4, 128, 0),  # gpt-oss shape: hidden size not a multiple of 128
        (4096, 32, 2880, 4, 256, 512),  # unused capacity tail past the last expert
        (1024, 7, 1024, 2, 128, 128),
        (96, 16, 512, 1, 128, 0),  # some experts receive no token
    ],
)
@pytest.mark.parametrize("columnwise", [False, True])
@pytest.mark.parametrize("optimize_for_gemm", [False, True])
def test_group_quantize_row_gather_mxfp8(
    dtype, num_tokens, num_expert, hidden_size, topK, align_size, tail_rows, columnwise,
    optimize_for_gemm,
):
    """group_quantize(x, row_index=...) is bitwise permute-and-pad + group_quantize."""
    import transformer_engine_torch as tex

    torch.manual_seed(seed)
    scores = torch.rand(num_tokens, num_expert, device="cuda")
    routing_map = torch.zeros(num_tokens, num_expert, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, torch.topk(scores, topK, dim=1).indices, True)
    if num_tokens < num_expert * 8:
        routing_map[:, : num_expert // 2] = False  # leave some experts without tokens
    probs = scores * routing_map
    inp = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    inp[:, ::97] *= 1000.0  # make the per-block scales differ

    tokens_per_expert = routing_map.sum(dim=0)
    permuted, _, _, _, padded_counts = te_permute_and_pad_with_probs(
        inp, probs, routing_map, tokens_per_expert, align_size
    )
    num_used = int(padded_counts.sum().item())
    num_rows = num_used + tail_rows
    row_index, padded_counts_ref = _padded_permutation_row_index(
        routing_map, align_size, num_rows
    )
    assert torch.equal(padded_counts.to(padded_counts_ref.device), padded_counts_ref)

    # The permuted operand as a dispatcher hands it to the grouped GEMM: the rows past the
    # last expert are left unwritten (garbage).
    ref_rows = torch.randn(num_rows, hidden_size, dtype=dtype, device="cuda") * 1e4
    ref_rows[:num_used] = permuted[:num_used]
    gathered = inp[row_index.clamp(min=0).long()] * (row_index >= 0).unsqueeze(1).to(dtype)
    torch.testing.assert_close(gathered[:num_used], ref_rows[:num_used], rtol=0, atol=0)

    first_dims = padded_counts_ref.to(torch.int64)
    tensor_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64, device="cuda"), torch.cumsum(first_dims, 0)]
    ) * hidden_size

    def quantize(x, rows):
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=columnwise)
        quantizer.optimize_for_gemm = optimize_for_gemm
        quantizer.internal = True
        return tex.group_quantize(
            x, quantizer, num_expert, first_dims, tensor_offsets=tensor_offsets, row_index=rows
        )

    ref = quantize(ref_rows, None)
    out = quantize(inp, row_index)
    assert tuple(out.logical_shape) == (num_rows, hidden_size)

    # Rows past the last expert are not quantized by either path, so only the used prefix of
    # the data and scale buffers is defined. It is a prefix in both scale layouts.
    rowwise_scale_cols = (hidden_size // 32 + 3) // 4 * 4
    columnwise_scale_cols = (hidden_size + 127) // 128 * 128
    checks = [
        ("rowwise_data", num_used * hidden_size),
        ("scale_inv", num_used * rowwise_scale_cols),
    ]
    if columnwise:
        checks += [
            ("columnwise_data", num_used * hidden_size),
            ("columnwise_scale_inv", num_used // 32 * columnwise_scale_cols),
        ]
    for name, numel in checks:
        a = getattr(ref, name).reshape(-1).view(torch.uint8)[:numel]
        b = getattr(out, name).reshape(-1).view(torch.uint8)[:numel]
        assert torch.equal(a, b), f"{name}: {int((a != b).sum())} of {numel} bytes differ"


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize(
    "num_groups, rows_per_group, cols",
    [(4, 160, 128), (32, 2880, 256)],
    ids=("4x160x128", "32x2880x256"),
)
def test_group_quantize_row_gather_mxfp8_uniform_partial_chunks(num_groups, rows_per_group, cols):
    """Row-gather quantize into a uniform grouped tensor (no first_dims) whose rows per group are
    not a multiple of 128: bitwise equal to group_quantize of the gathered rows (both clip the
    partial last chunk of every group)."""
    import transformer_engine_torch as tex

    torch.manual_seed(seed)
    rows = num_groups * rows_per_group
    num_src = rows // 2 + 7
    inp = torch.randn(num_src, cols, dtype=torch.bfloat16, device="cuda")
    inp[:, ::5] *= 1000.0  # make the per-block scales differ
    row_index = torch.randint(-1, num_src, (rows,), dtype=torch.int32, device="cuda")
    # Rows with a negative index are +0 (a multiply by 0 would give -0 for negative inputs).
    gathered = torch.where(
        (row_index >= 0).unsqueeze(1), inp[row_index.clamp(min=0).long()], torch.zeros_like(inp[:1])
    )

    def quantize(x, rows_idx):
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        quantizer.internal = True
        return tex.group_quantize(x, quantizer, num_groups, None, row_index=rows_idx)

    ref = quantize(gathered, None)
    out = quantize(inp, row_index)
    assert tuple(out.logical_shape) == (rows, cols)
    rowwise_scale_cols = (cols // 32 + 3) // 4 * 4
    columnwise_scale_cols = (cols + 127) // 128 * 128
    for name, numel in (
        ("rowwise_data", rows * cols),
        ("columnwise_data", rows * cols),
        ("scale_inv", rows * rowwise_scale_cols),
        ("columnwise_scale_inv", rows // 32 * columnwise_scale_cols),
    ):
        a = getattr(ref, name).reshape(-1).view(torch.uint8)[:numel]
        b = getattr(out, name).reshape(-1).view(torch.uint8)[:numel]
        assert torch.equal(a, b), f"{name}: {int((a != b).sum())} of {numel} bytes differ"


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize(
    "num_groups, rows_per_group, optimize_for_gemm, match",
    [
        (8, 48, False, "divisible by 32"),  # a 32-row MXFP8 block would straddle two groups
        (4, 160, True, "GEMM-swizzled"),  # swizzled scales need whole 128-row tiles per group
    ],
    ids=("rows_not_multiple_of_32", "swizzled_rows_not_multiple_of_128"),
)
def test_group_quantize_row_gather_mxfp8_rejects_unsupported_uniform_rows(
    num_groups, rows_per_group, optimize_for_gemm, match
):
    """The row-gather entry applies the same uniform-shape row-count checks as group_quantize."""
    import transformer_engine_torch as tex

    rows = num_groups * rows_per_group
    inp = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    row_index = torch.zeros(rows, dtype=torch.int32, device="cuda")
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = optimize_for_gemm
    quantizer.internal = True
    with pytest.raises(RuntimeError, match=match):
        tex.group_quantize(inp, quantizer, num_groups, None, row_index=row_index)


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_index_map_topk1_no_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    use_torch_compile,
):
    if use_torch_compile and num_expert != 7:
        pytest.skip("torch.compile tested with single config only")
    topK = 1
    num_out_tokens = None
    with_probs = False
    BENCHMARK = False

    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_permutation_mask_map_topk1_no_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    use_torch_compile,
):
    if use_torch_compile and num_expert != 7:
        pytest.skip("torch.compile tested with single config only")
    topK = 1
    num_out_tokens = None
    with_probs = False
    BENCHMARK = False

    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_chunk_permutation(
    te_dtype,
    num_tokens,
    num_expert,
    tp_size,
    hidden_size,
    use_torch_compile,
):
    if use_torch_compile and (num_expert != 7 or tp_size != 2):
        pytest.skip("torch.compile tested with single config only")
    BENCHMARK = False

    _test_moe_chunk_sort(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        tp_size=tp_size,
        hidden_size=hidden_size,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_chunk_permutation_empty_input(te_dtype, use_torch_compile):
    BENCHMARK = False

    _test_moe_chunk_sort(
        te_dtype=te_dtype,
        num_tokens=0,
        num_expert=8,
        tp_size=2,
        hidden_size=4096,
        BENCHMARK=BENCHMARK,
        use_torch_compile=use_torch_compile,
    )


# ---------------------------------------------------------------------------------------------
# Deterministic tile selection of the permutation Triton kernels (NVTE_PERMUTATION_BLOCK_SIZE)
# ---------------------------------------------------------------------------------------------

from transformer_engine.common.triton import permutation as _te_triton_perm_common
from transformer_engine.pytorch.triton import permutation as _te_triton_perm

_TILE_ENV = "NVTE_PERMUTATION_BLOCK_SIZE"
_TILES = _te_triton_perm_common._PERMUTATION_AUTOTUNE_BLOCK_SIZES
# Every fixed tile, the shape rule, and Triton's autotuning (the default).
_TILE_MODES = [str(t) for t in _TILES] + ["heuristic", "autotune"]
# (kernel, hidden_size, flags, tile) of the shape rule (see permutation_block_size).
_PP = {"permute_probs": True}
_MP = {"merging_probs": True}
_HEURISTIC_EXPECTED = [
    ("unpermute", 2880, {}, 1024),
    ("unpermute", 7168, {}, 1024),
    ("unpermute", 512, {}, 512),
    ("unpermute", 100, {}, 128),
    ("unpermute", 1, {}, 64),
    ("unpermute", 2880, _PP, 4096),
    ("unpermute", 2880, _MP, 4096),
    ("unpermute", 1000, _PP, 1024),
    ("unpermute", 5120, _PP, 2048),
    ("unpermute", 7168, _MP, 4096),
    ("permute", 2880, {}, 4096),
    ("permute", 2880, _PP, 4096),
    ("permute", 2048, {}, 2048),
    ("permute", 6144, _PP, 2048),
    ("permute", 16384, {}, 4096),
    ("sort_chunks", 2880, {}, 4096),
    ("sort_chunks", 5120, _PP, 2048),
    ("unpermute_bwd_with_merging_probs", 1024, {}, 2048),
    ("unpermute_bwd_with_merging_probs", 2880, {}, 4096),
    ("unpermute_bwd_with_merging_probs", 7168, {}, 2048),
]
_tile_dtypes = [torch.float32, torch.float16]
if te.is_bf16_available():
    _tile_dtypes.append(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Integer view of a tensor, so that comparisons are bitwise (-0 != +0, NaN payloads)."""
    int_dtype = {4: torch.int32, 2: torch.int16, 1: torch.int8}[t.element_size()]
    return t.contiguous().view(int_dtype)


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, what: str):
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    diff = int((_bits(actual) != _bits(expected)).sum())
    assert diff == 0, f"{what}: {diff} of {actual.numel()} elements differ bitwise"


def _tile_case(num_tokens, num_experts, top_k, hidden_size, dtype, seed=1234):
    """A mask-map ``row_id_map`` plus inputs that stress the per-element summation.

    Tokens with 0 routes, 1 route and up to ``top_k`` routes; experts in ascending order and
    distinct random source rows. Inputs span a wide exponent range and contain -0 rows, subnormals
    of ``dtype`` and, for 64 tokens, the rows (1, 2^-24, 2^-24, -1), whose fp32 sum is 0 in map
    order and 2^-23 in the reverse order (so a different summation order would be detected).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = torch.randint(1, top_k + 1, (num_tokens,), device="cuda", generator=g, dtype=torch.int32)
    n[: num_tokens // 16] = 0
    n[num_tokens // 16 : num_tokens // 16 + 64] = top_k
    experts = (
        torch.rand(num_tokens, num_experts, device="cuda", generator=g)
        .argsort(dim=-1)[:, :top_k]
        .sort(dim=-1)
        .values
    )
    num_rows = num_tokens * top_k
    rows = torch.randperm(num_rows, device="cuda", generator=g).view(num_tokens, top_k)
    row_id_map = torch.full((num_tokens, 2 * num_experts + 1), -1, dtype=torch.int32, device="cuda")
    row_id_map[:, :top_k] = rows.int()
    row_id_map[:, num_experts : num_experts + top_k] = experts.int()
    row_id_map[:, 2 * num_experts] = n

    max_exp = 8 if dtype == torch.float16 else 24
    x = torch.randn(num_rows, hidden_size, device="cuda", generator=g)
    x = x * torch.exp2(
        torch.randint(-max_exp, max_exp + 1, x.shape, device="cuda", generator=g).float()
    )
    x = x.to(dtype)
    x[::7] = -0.0
    x[1::11, ::3] = torch.finfo(dtype).tiny / 4  # subnormal in dtype
    x[2::13, 1::5] = -torch.finfo(dtype).tiny / 8
    special = rows[num_tokens // 16 : num_tokens // 16 + 64]
    for k, value in enumerate((1.0, 2.0**-24, 2.0**-24, -1.0)[:top_k]):
        x[special[:, k]] = value
    return n, experts, rows, row_id_map, x


def _ref_unpermute_rows(x, rows, n, num_tokens, hidden_size):
    """fp32 sum of the routed rows from +0, in map order, rounded once (the kernel's contract)."""
    acc = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device="cuda")
    for k in range(rows.size(1)):
        routed = (n > k).unsqueeze(-1)
        acc = acc + torch.where(routed, x[rows[:, k]].float(), 0.0)
    return acc.to(x.dtype)


def _assert_close_to_scale(actual, expected, scale, rel, dtype):
    """|actual - expected| <= rel * scale + the smallest normal of dtype (scale: sum of |terms|)."""
    err = (actual.float() - expected.float()).abs()
    bound = rel * scale.float() + torch.finfo(dtype).tiny
    bad = int((err > bound).sum())
    assert bad == 0, f"{bad} elements exceed rel {rel} of the term magnitude"


def _run_tile_modes(monkeypatch, fn):
    """Run ``fn`` under every NVTE_PERMUTATION_BLOCK_SIZE mode; returns {mode: outputs}."""
    outs = {}
    for mode in _TILE_MODES:
        monkeypatch.setenv(_TILE_ENV, mode)
        outs[mode] = fn()
    monkeypatch.delenv(_TILE_ENV, raising=False)
    outs["unset"] = fn()
    return outs


def test_permutation_block_size_env(monkeypatch):
    """Parsing of NVTE_PERMUTATION_BLOCK_SIZE."""
    fixed = _te_triton_perm._fixed_block_size
    monkeypatch.delenv(_TILE_ENV, raising=False)
    assert fixed("unpermute", 2880) is None
    for unset in ("", "autotune", " AutoTune "):
        monkeypatch.setenv(_TILE_ENV, unset)
        assert fixed("unpermute", 2880) is None
    for tile in _TILES:
        monkeypatch.setenv(_TILE_ENV, str(tile))
        assert fixed("unpermute", 2880, permute_probs=True) == tile
        assert fixed("unpermute_bwd_with_merging_probs", 7168) == tile
    monkeypatch.setenv(_TILE_ENV, "heuristic")
    for kernel in ("permute", "unpermute", "unpermute_bwd_with_merging_probs", "sort_chunks"):
        for hidden_size in (1, 63, 1000, 2880, 4096, 7168, 16384):
            for permute_probs in (False, True):
                tile = fixed(kernel, hidden_size, permute_probs=permute_probs)
                assert tile in _TILES
                assert tile == _te_triton_perm_common.permutation_block_size(
                    kernel, hidden_size, permute_probs=permute_probs
                )
    for bad in ("1", "32", "100", "8192", "-1024", "fixed", "1024.0"):
        monkeypatch.setenv(_TILE_ENV, bad)
        with pytest.raises(ValueError, match=_TILE_ENV):
            fixed("unpermute", 2880)
    with pytest.raises(ValueError):
        _te_triton_perm_common.permutation_block_size("unknown_kernel", 2880)
    with pytest.raises(ValueError):
        _te_triton_perm_common.permutation_autotune_config(96)


def test_permutation_block_size_heuristic_rule():
    """The documented tiles of the shape rule (common/triton/permutation.py)."""
    rule = _te_triton_perm_common.permutation_block_size
    for kernel, hidden_size, flags, expected in _HEURISTIC_EXPECTED:
        assert rule(kernel, hidden_size, **flags) == expected, (kernel, hidden_size, flags)


@pytest.mark.parametrize("dtype", _tile_dtypes)
@pytest.mark.parametrize("hidden_size", [2880, 1000])
@pytest.mark.parametrize("variant", ["rows", "permute_probs", "merging_probs"])
def test_unpermute_bitwise_across_block_sizes(monkeypatch, dtype, hidden_size, variant):
    """_unpermute_kernel: the tile only splits the hidden dimension, so every tile (fixed, the
    shape rule, or autotuned) gives the same bits; rows-only sums equal an fp32 map-order
    reference bit for bit."""
    num_tokens, num_experts, top_k = 1024, 32, 4
    n, experts, rows, row_id_map, x = _tile_case(num_tokens, num_experts, top_k, hidden_size, dtype)
    g = torch.Generator(device="cuda").manual_seed(99)
    permuted_probs = merging_probs = None
    if variant == "permute_probs":
        permuted_probs = torch.rand(x.size(0), device="cuda", generator=g)
        permuted_probs[::5] = -0.0
        permuted_probs[1::9] = 2.0**-140  # fp32 subnormal
    if variant == "merging_probs":
        merging_probs = torch.rand(num_tokens, num_experts, device="cuda", generator=g)

    def run():
        out, out_probs = _te_triton_perm.unpermute_with_mask_map(
            x, row_id_map, merging_probs, permuted_probs, None, num_tokens, num_experts, hidden_size
        )
        return out, out_probs

    outs = _run_tile_modes(monkeypatch, run)
    ref_out, ref_probs = outs["64"]
    for mode, (out, out_probs) in outs.items():
        _assert_bitwise(out, ref_out, f"{variant} output, mode {mode} vs 64")
        if permuted_probs is not None:
            _assert_bitwise(out_probs, ref_probs, f"unpermuted probs, mode {mode} vs 64")
        else:
            assert out_probs is None

    if variant != "merging_probs":
        expected = _ref_unpermute_rows(x, rows, n, num_tokens, hidden_size)
        _assert_bitwise(ref_out, expected, "output vs fp32 map-order reference")
    else:
        acc = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device="cuda")
        abs_acc = torch.zeros_like(acc)
        for k in range(top_k):
            p = merging_probs.gather(1, experts[:, k : k + 1])
            term = torch.where((n > k).unsqueeze(-1), x[rows[:, k]].float() * p, 0.0)
            acc += term
            abs_acc += term.abs()
        # The kernel may contract the multiply-add (FMA): compare against the magnitude of the
        # terms rather than bitwise.
        _assert_close_to_scale(ref_out.float(), acc, abs_acc, 2.0**-6, dtype)
    if permuted_probs is not None:
        expected_probs = torch.zeros(num_tokens, num_experts, device="cuda")
        for k in range(top_k):
            tok = torch.nonzero(n > k).squeeze(-1)
            expected_probs[tok, experts[tok, k]] = permuted_probs[rows[tok, k]]
        _assert_bitwise(ref_probs, expected_probs, "unpermuted probs vs reference")


@pytest.mark.parametrize("dtype", _tile_dtypes)
@pytest.mark.parametrize("hidden_size", [2880, 1000])
@pytest.mark.parametrize("with_probs", [False, True])
def test_permute_bitwise_across_block_sizes(monkeypatch, dtype, hidden_size, with_probs):
    """_permute_kernel copies rows: identical bits for every tile, equal to the source rows."""
    num_tokens, num_experts, top_k = 1024, 32, 4
    n, experts, rows, row_id_map, x = _tile_case(num_tokens, num_experts, top_k, hidden_size, dtype)
    inp = x[:num_tokens].contiguous()
    probs = None
    if with_probs:
        probs = torch.rand(num_tokens, num_experts, device="cuda") + 0.5  # no zero probs
    num_out_tokens = num_tokens * top_k
    routed = torch.cat([rows[n > k, k] for k in range(top_k)])

    def run():
        out, _, out_probs = _te_triton_perm.permute_with_mask_map(
            inp,
            row_id_map,
            probs,
            None,
            None,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
            0,
        )
        # Rows that no token routes to stay uninitialised: compare the routed rows only.
        return out[routed], (out_probs[routed] if out_probs is not None else None)

    outs = _run_tile_modes(monkeypatch, run)
    expected = torch.cat([inp[n > k] for k in range(top_k)])
    for mode, (out, out_probs) in outs.items():
        _assert_bitwise(out, expected, f"permuted rows, mode {mode}")
        if with_probs:
            exp_probs = torch.cat(
                [probs[n > k].gather(1, experts[n > k, k : k + 1]).squeeze(1) for k in range(top_k)]
            )
            _assert_bitwise(out_probs, exp_probs, f"permuted probs, mode {mode}")


@pytest.mark.parametrize("dtype", _tile_dtypes)
@pytest.mark.parametrize("hidden_size", [2880, 1000])
def test_unpermute_bwd_with_merging_probs_block_sizes(monkeypatch, dtype, hidden_size):
    """_unpermute_bwd_with_merging_probs_kernel: the activation gradient is elementwise (same bits
    for every tile). The merging-prob gradient is a reduction over the hidden dimension whose order
    follows the tile, so it may differ in the last bits between tiles; with a fixed tile or the
    shape rule it is reproducible."""
    num_tokens, num_experts, top_k = 1024, 32, 4
    n, experts, rows, row_id_map, x = _tile_case(num_tokens, num_experts, top_k, hidden_size, dtype)
    g = torch.Generator(device="cuda").manual_seed(7)
    grad = torch.randn(num_tokens, hidden_size, device="cuda", generator=g).to(dtype)
    merging_probs = torch.rand(num_tokens, num_experts, device="cuda", generator=g)
    routed = torch.cat([rows[n > k, k] for k in range(top_k)])

    def run():
        act_grad, probs_grad = _te_triton_perm.unpermute_with_mask_map_bwd_with_merging_probs(
            grad,
            row_id_map,
            x,
            merging_probs,
            None,
            num_tokens,
            num_experts,
            x.size(0),
            hidden_size,
        )
        return act_grad[routed], probs_grad

    outs = _run_tile_modes(monkeypatch, run)
    # Magnitude of the dot-product terms of every (token, expert) prob gradient.
    abs_dot = torch.zeros(num_tokens, num_experts, device="cuda")
    for k in range(top_k):
        tok = torch.nonzero(n > k).squeeze(-1)
        abs_dot[tok, experts[tok, k]] = (x[rows[tok, k]].float() * grad[tok].float()).abs().sum(-1)
    ref_act, ref_probs_grad = outs["64"]
    for mode, (act_grad, probs_grad) in outs.items():
        _assert_bitwise(act_grad, ref_act, f"act grad, mode {mode} vs 64")
        _assert_close_to_scale(probs_grad, ref_probs_grad, abs_dot, 1e-4, torch.float32)
    monkeypatch.setenv(_TILE_ENV, "heuristic")
    again = run()
    rule_tile = _te_triton_perm_common.permutation_block_size(
        "unpermute_bwd_with_merging_probs", hidden_size
    )
    _assert_bitwise(again[1], outs["heuristic"][1], "probs grad, heuristic run-to-run")
    _assert_bitwise(again[1], outs[str(rule_tile)][1], "probs grad, heuristic vs its tile")


@pytest.mark.parametrize("dtype", _tile_dtypes)
@pytest.mark.parametrize("with_probs", [False, True])
@pytest.mark.parametrize("is_forward", [False, True])
def test_sort_chunks_bitwise_across_block_sizes(monkeypatch, dtype, with_probs, is_forward):
    """_sort_chunks_by_map_kernel copies rows: identical bits for every tile."""
    num_rows, hidden_size = 4096, 2880
    _, _, _, _, x = _tile_case(1024, 32, 4, hidden_size, dtype)
    perm = torch.randperm(num_rows, device="cuda").int()
    probs = torch.rand(num_rows, device="cuda") if with_probs else None

    def run():
        return _te_triton_perm.sort_chunks_by_map(x, perm, probs, num_rows, hidden_size, is_forward)

    outs = _run_tile_modes(monkeypatch, run)
    if is_forward:
        expected = torch.empty_like(x)
        expected[perm.long()] = x
    else:
        expected = x[perm.long()]
    for mode, (out, out_probs) in outs.items():
        _assert_bitwise(out, expected, f"sorted rows, mode {mode}")
        if with_probs:
            exp_probs = torch.empty_like(probs)
            if is_forward:
                exp_probs[perm.long()] = probs
            else:
                exp_probs = probs[perm.long()]
            _assert_bitwise(out_probs, exp_probs, f"sorted probs, mode {mode}")


@pytest.mark.parametrize("with_merging_probs", [False, True])
def test_permutation_block_size_public_api(monkeypatch, with_merging_probs):
    """moe_permute_with_probs + moe_unpermute forward and backward through the public API: the
    same bits with every tile mode (the merging-prob gradient is compared with a tolerance, see
    test_unpermute_bwd_with_merging_probs_block_sizes)."""
    dtype = torch.bfloat16 if te.is_bf16_available() else torch.float16
    num_tokens, num_experts, top_k, hidden_size = 1024, 32, 4, 2880
    g = torch.Generator(device="cuda").manual_seed(3)
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=g)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, scores.topk(top_k, dim=-1).indices, True)
    routing_map[: num_tokens // 16] = False  # tokens routed nowhere
    num_out_tokens = int(routing_map.sum())
    inp0 = torch.randn(num_tokens, hidden_size, device="cuda", generator=g).to(dtype)
    probs0 = torch.rand(num_tokens, num_experts, device="cuda", generator=g)
    w_out = torch.randn(num_tokens, hidden_size, device="cuda", generator=g).to(dtype)
    w_probs = torch.randn(num_out_tokens, device="cuda", generator=g)
    merging0 = torch.rand(num_tokens, num_experts, device="cuda", generator=g)

    def run():
        inp = inp0.clone().requires_grad_(True)
        probs = probs0.clone().requires_grad_(True)
        merging = merging0.clone().requires_grad_(True) if with_merging_probs else None
        permuted, permuted_probs, row_id_map = te_permute_with_probs(
            inp, probs, routing_map, num_out_tokens=num_out_tokens
        )
        out = te_unpermute(permuted, row_id_map, merging_probs=merging, restore_shape=inp.shape)
        loss = (out.float() * w_out.float()).sum() + (permuted_probs * w_probs).sum()
        loss.backward()
        return out, inp.grad, probs.grad, (merging.grad if merging is not None else None)

    outs = _run_tile_modes(monkeypatch, run)
    ref = outs["64"]
    for mode, (out, inp_grad, probs_grad, merging_grad) in outs.items():
        _assert_bitwise(out, ref[0], f"unpermuted output, mode {mode}")
        _assert_bitwise(inp_grad, ref[1], f"input grad, mode {mode}")
        _assert_bitwise(probs_grad, ref[2], f"probs grad, mode {mode}")
        if with_merging_probs:
            torch.testing.assert_close(merging_grad, ref[3], atol=1e-2, rtol=1e-3)


def test_permutation_fixed_block_size_skips_autotuning(monkeypatch):
    """With a fixed tile the autotuner is not consulted (no benchmarking, no cache entry); unset
    restores autotuning."""
    num_tokens, num_experts, top_k = 256, 32, 4
    hidden_size = 2888  # a hidden size no other test uses: a fresh autotune key
    _, _, _, row_id_map, x = _tile_case(num_tokens, num_experts, top_k, hidden_size, torch.float32)
    kernel = _te_triton_perm_common._unpermute_kernel
    if not hasattr(kernel, "cache"):
        pytest.skip("_unpermute_kernel is not wrapped by triton.autotune")

    def keys():
        return {k for k in kernel.cache if hidden_size in tuple(k)}

    def run():
        return _te_triton_perm.unpermute_with_mask_map(
            x, row_id_map, None, None, None, num_tokens, num_experts, hidden_size
        )[0]

    assert not keys()
    monkeypatch.setenv(_TILE_ENV, "heuristic")
    fixed = run()
    monkeypatch.setenv(_TILE_ENV, "128")
    fixed_128 = run()
    torch.cuda.synchronize()
    assert not keys(), "a fixed tile must not run the autotuner"
    monkeypatch.delenv(_TILE_ENV)
    tuned = run()
    assert keys(), "unset must autotune"
    _assert_bitwise(fixed, tuned, "heuristic vs autotuned")
    _assert_bitwise(fixed_128, tuned, "128 vs autotuned")


@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS", "0") != "1",
    reason="Benchmark test - run with: RUN_BENCHMARK_TESTS=1 pytest -k single_case",
)
def test_permutation_single_case():
    print("GPU:", torch.cuda.get_device_name(0))

    # te_dtype = te.DType.kFloat32
    # te_dtype = te.DType.kFloat16
    te_dtype = te.DType.kBFloat16

    num_tokens = 12
    num_expert = 4
    hidden_size = 16
    topK = 2
    num_out_tokens = num_tokens * topK - 1
    with_probs = True
    Benchmark = True

    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=Benchmark,
    )

    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=Benchmark,
    )

    _test_permutation_and_padding_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        BENCHMARK=Benchmark,
    )

    _test_permutation_and_padding_with_merging_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        BENCHMARK=Benchmark,
    )

    _test_moe_chunk_sort(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        tp_size=4,
        hidden_size=hidden_size,
        BENCHMARK=Benchmark,
    )

    _test_permutation_mask_map_alongside_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        tp_size=4,
    )


def benchmark_single_case(
    te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, ep_size, tp_size
):
    torch.cuda.nvtx.range_push(
        f"{num_tokens}-{num_expert}-{hidden_size}-{topK}-{ep_size}-{tp_size}"
    )

    torch.cuda.nvtx.range_push("permutation_index_map_with_probs")
    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=True,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("permutation_mask_map_with_probs")
    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=True,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("permutation_mask_map_without_probs")
    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=False,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("permutation_and_padding_mask_map")
    _test_permutation_and_padding_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("permutation_and_padding_with_merging_probs")
    _test_permutation_and_padding_with_merging_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("permutation_mask_map_alongside_probs")
    _test_permutation_mask_map_alongside_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        tp_size=tp_size,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()


@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS", "0") != "1",
    reason="Benchmark test - run with: RUN_BENCHMARK_TESTS=1 pytest -k benchmark",
)
def test_benchmark_multiple_cases():
    """Benchmark test - skipped by default. Run with: RUN_BENCHMARK_TESTS=1 pytest -k benchmark"""
    print("GPU:", torch.cuda.get_device_name(0))

    # te_dtype = te.DType.kFloat32
    # te_dtype = te.DType.kFloat16
    te_dtype = te.DType.kBFloat16

    ep_size = 64
    tp_size = 2
    num_tokens = 4096
    num_expert = 256
    hidden_size = 7168
    topK = 8
    num_out_tokens = num_tokens * topK
    benchmark_single_case(
        te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, ep_size, tp_size
    )

    ep_size = 8
    tp_size = 1
    num_tokens = 8192 * 2
    num_expert = 128
    hidden_size = 4096
    topK = 6
    num_out_tokens = num_tokens * topK
    benchmark_single_case(
        te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, ep_size, tp_size
    )

    ep_size = 64
    tp_size = 2
    num_tokens = 16384
    num_expert = 4
    hidden_size = 7168
    topK = 1
    num_out_tokens = num_tokens * topK
    benchmark_single_case(
        te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, ep_size, tp_size
    )


if __name__ == "__main__":
    test_benchmark_multiple_cases()
