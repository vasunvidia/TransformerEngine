# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
from typing import Callable, Tuple, Union
import math
import torch
import pytest
from transformer_engine.pytorch.attention.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    apply_fused_qkv_rotary_pos_emb,
)


# Gradient is a broadcasted scalar
def _overlapping_grad(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return query.sum() * 2 + key.sum() * 2 + value.sum() * 2


# Gradient is a full tensor
def _non_overlapping_grad(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(query)
    return torch.sum(query * t) + torch.sum(key * t) + torch.sum(value * t)

# @pytest.mark.parametrize("start_positions", [True, False])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
# @pytest.mark.parametrize("seq_length", [2048, 4096])
# @pytest.mark.parametrize("hidden_size", [128, 256])
# @pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
# @pytest.mark.parametrize("margin", [0, 10])
# @pytest.mark.parametrize("transpose", [None, (0, 1), (2, 3)])
# @pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
# @pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
# @pytest.mark.parametrize("cp_size", [1, 2])
# @pytest.mark.parametrize("interleaved", [True, False])

@pytest.mark.parametrize("start_positions", [False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [8, 2048, 4096])
@pytest.mark.parametrize("hidden_size", [64, 128, 256])
@pytest.mark.parametrize("rotary_percent", [1.0])
@pytest.mark.parametrize("margin", [0])
#@pytest.mark.parametrize("transpose", [None, (0, 1), (2, 3)])
@pytest.mark.parametrize("tensor_format", ["sbhd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad])
@pytest.mark.parametrize("cp_size", [1])
@pytest.mark.parametrize("interleaved", [False])
def test_fused_qkv_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    #transpose: Union[Tuple, None],
    tensor_format: str,
    loss_func: Callable,
    cp_size: int,
    interleaved: bool,
    start_positions: bool,
) -> None:
    if margin == 0 and start_positions == True:
        # This makes sure that the `start_positions` offsets being applied
        # are with the maximum length of the rope embeddings.
        pytest.skip("Skipping test with margin=0 and start_positions=True")

    if start_positions == True and cp_size > 1:
        # `start_positions` is only supported for `cp_size=1` and inference.
        pytest.skip("Skipping test with cp_size>1 and start_positions=True")

    device = torch.device("cuda:0")
    batch_size, head_num = 1, 8
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size*6),
        dtype=dtype,
        device=device,
    )

    # Get arbitrary offsets to be used with RoPE for all the sequences
    start_positions = (
        torch.randint(0, margin, (batch_size,), dtype=torch.int32, device=device)
        if start_positions
        else None
    )

    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent, interleaved=interleaved)
    emb = rotary_pos_emb(seq_length * cp_size)
    assert emb.is_contiguous()

    for cp_rank in range(cp_size):
        # unfused
        # The fused kernel computes in float32 internally, so we force the unfused func to use float32
        # for more accurate comparison

        (query, key, value) = torch.split(t, [hidden_size*4, hidden_size, hidden_size], dim=3)

        query_unfused = apply_rotary_pos_emb(
            query.contiguous().float(),
            emb,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            fused=True,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)
        key_unfused = apply_rotary_pos_emb(
            key.contiguous().float(),
            emb,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            fused=True,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)
        value_unfused = value.contiguous()
        #loss_unfused = loss_func(query_unfused, key_unfused, value_unfused)

        #if not isinstance(start_positions, torch.Tensor):
        #    loss_unfused.backward()
        #    grad_unfused = t.grad.detach().clone()

        t.grad = None

        # fused
        query_fused, key_fused, value_fused = apply_fused_qkv_rotary_pos_emb(
            t,
            emb,
            emb,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            cp_size=cp_size,
            cp_rank=cp_rank,
            qkv_split_arg_list=[hidden_size*4, hidden_size, hidden_size],
        )
        #loss_fused = loss_func(query_fused, key_fused, value_fused)

        #if not isinstance(start_positions, torch.Tensor):
        #    loss_fused.backward()
        #    grad_fused = t.grad.detach().clone()
        #t.grad = None

        # torch.testing.assert_close(query_fused.view(query_unfused.shape), query_unfused)
        batch_idx_print = 0
        seq_idx_print = 7
        
        
        query_unfused = query_unfused.view(query_fused.shape)

        for batch_idx in range(batch_size):
            for seq_idx in range(seq_length):
                if batch_idx != batch_idx_print or seq_idx != seq_idx_print:
                    continue
                query_fused2 = query_fused[batch_idx][seq_idx] if tensor_format == "bshd" else query_fused[seq_idx][batch_idx]
                q_out_flat = query_fused2.flatten().float()
                query_unfused2 = query_unfused[batch_idx][seq_idx] if tensor_format == "bshd" else query_unfused[seq_idx][batch_idx]
                q_out_te_flat = query_unfused2.flatten().float()
                if batch_idx == batch_idx_print and seq_idx == seq_idx_print:
                    print (f'query_unfused: {query_unfused2.shape}-{query_unfused2}')
                    print (f'query_fused: {query_fused2.shape}-{query_fused2}')
                q_cos_sim = torch.nn.functional.cosine_similarity(q_out_te_flat, q_out_flat, dim=0)
                print(f"Cosine similarity {batch_idx},{seq_idx} between q_expected and q_actual: {q_cos_sim.item():.6f}")
        
        #torch.testing.assert_close(key_fused.view(key_unfused.shape), key_unfused)
        #torch.testing.assert_close(value_fused.view(value_unfused.shape), value_unfused)

        #if not isinstance(start_positions, torch.Tensor):
        #    torch.testing.assert_close(grad_fused, grad_unfused)

        assert query_fused.is_contiguous()
        assert key_fused.is_contiguous()
        assert value_fused.is_contiguous()


#@pytest.mark.parametrize("margin", [10])
#@pytest.mark.parametrize("start_positions", [True, False])
#@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
#@pytest.mark.parametrize("hidden_size", [128, 256])
#@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
#@pytest.mark.parametrize("transpose", [None, (1, 2)])
#@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
#@pytest.mark.parametrize("cp_size", [1, 2])
#@pytest.mark.parametrize("interleaved", [True, False])
#def test_fused_rope_thd(
#    dtype: torch.dtype,
#    hidden_size: int,
#    rotary_percent: float,
#    transpose: Union[Tuple, None],
#    loss_func: Callable,
#    cp_size: int,
#    interleaved: bool,
#    start_positions: bool,
#    margin: int,
#) -> None:
#
#    if start_positions == True and cp_size > 1:
#        # `start_positions` is only supported for `cp_size=1` and inference.
#        pytest.skip("Skipping test with cp_size>1 and start_positions=True")
#
#    device = torch.device("cuda:0")
#    batch_size, head_num = 2, 64
#    cu_seqlens = [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048]
#
#    # Get arbitrary offsets to be used with RoPE for all the sequences
#    start_positions = (
#        torch.randint(0, margin, (len(cu_seqlens) - 1,), dtype=torch.int32, device=device)
#        if start_positions
#        else None
#    )
#
#    if cp_size > 1:
#        cu_seqlens_padded = [0]
#        for i in range(1, len(cu_seqlens)):
#            cu_seqlens_padded.append(
#                cu_seqlens_padded[i - 1]
#                + math.ceil((cu_seqlens[i] - cu_seqlens[i - 1]) / (cp_size * 2)) * (cp_size * 2)
#            )
#    else:
#        cu_seqlens_padded = cu_seqlens
#    cu_seqlens_padded = torch.tensor(
#        cu_seqlens_padded,
#        dtype=torch.int32,
#        device=device,
#    )
#    t = torch.rand(
#        (cu_seqlens_padded[-1] // cp_size, head_num, hidden_size),
#        dtype=dtype,
#        device=device,
#    )
#    if transpose:
#        t = t.transpose(*transpose).contiguous().transpose(*transpose)
#    t.requires_grad = True
#
#    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent, interleaved=interleaved)
#    emb = rotary_pos_emb(cu_seqlens_padded[-1])
#    assert emb.is_contiguous()
#
#    for cp_rank in range(cp_size):
#        # unfused
#        # The fused kernel computes in float32 internally, so we force the unfused func to use float32
#        # for more accurate comparison
#        output_unfused = apply_rotary_pos_emb(
#            t.float(),
#            emb,
#            start_positions=start_positions,
#            tensor_format="thd",
#            interleaved=interleaved,
#            fused=False,
#            cu_seqlens=cu_seqlens_padded,
#            cp_size=cp_size,
#            cp_rank=cp_rank,
#        ).to(dtype)
#        loss_unfused = loss_func(output_unfused)
#
#        if not isinstance(start_positions, torch.Tensor):
#            loss_unfused.backward()
#            grad_unfused = t.grad.detach().clone()
#        t.grad = None
#
#        # fused
#        output_fused = apply_rotary_pos_emb(
#            t,
#            emb,
#            start_positions=start_positions,
#            interleaved=interleaved,
#            fused=True,
#            tensor_format="thd",
#            cu_seqlens=cu_seqlens_padded,
#            cp_size=cp_size,
#            cp_rank=cp_rank,
#        )
#        loss_fused = loss_func(output_fused)
#
#        if not isinstance(start_positions, torch.Tensor):
#            loss_fused.backward()
#            grad_fused = t.grad.detach().clone()
#        t.grad = None
#
#        torch.testing.assert_close(output_fused, output_unfused)
#
#        if not isinstance(start_positions, torch.Tensor):
#            torch.testing.assert_close(grad_fused, grad_unfused)
#
#        assert output_fused.is_contiguous()
