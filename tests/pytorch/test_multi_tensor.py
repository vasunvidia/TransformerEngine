# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch
import transformer_engine_torch as tex
from transformer_engine.pytorch import is_mxfp8_available
from transformer_engine.pytorch.optimizers import MultiTensorApply

from references.quantize_scale_calc import scale_from_amax_tensor


input_size_pairs = [
    (7777 * 77, 555 * 555),
    (777, 555),
    (555, 2048 * 32 + 1),
    (2048 * 32 + 1, 555),
    (555, 2048 * 32),
    (2048 * 32, 555),
    (33333, 555),
    (555, 33333),
]
appliers = [MultiTensorApply(2048 * 32), MultiTensorApply(333), MultiTensorApply(33333)]
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)


@pytest.mark.parametrize("input_size_pair", input_size_pairs)
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
def test_multi_tensor_scale(input_size_pair, applier, repeat, in_type, out_type, inplace):
    if inplace is True and (out_type is not in_type):
        pytest.skip("inplace=True and out_type != in_type is not supported.")
    elif (in_type == torch.float16 and out_type == torch.bfloat16) or (
        in_type == torch.bfloat16 and out_type == torch.float16
    ):
        pytest.skip("float16 to bfloat16 is not necessary and vice versa.")

    device = torch.device("cuda")
    scale = 4.0
    overflow_buf = torch.zeros(1, dtype=torch.int32, device=device)
    ref = torch.tensor([1.0], dtype=torch.float32, device=device)
    sizea, sizeb = input_size_pair

    def downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=False):
        overflow_buf.zero_()
        a = torch.full([sizea], scale, dtype=torch.float32, device=device)
        b = torch.full([sizeb], scale, dtype=torch.float32, device=device)

        out_list = []
        for i in range(repeat):
            out_list += [a.clone().to(out_type), b.clone().to(out_type)]

        if inplace:
            in_list = out_list
        else:
            in_list = [out.clone().to(in_type) for out in out_list]

        applier(tex.multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)

        assert all([torch.allclose(out, ref.to(out_type)) for out in out_list])
        assert overflow_buf.item() == 0

    def find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        t,
        ind,
        val,
        inplace=False,
    ):
        overflow_buf.zero_()
        a = torch.full([sizea], scale, dtype=torch.float32, device=device)
        b = torch.full([sizeb], scale, dtype=torch.float32, device=device)

        out_list = []
        for i in range(repeat):
            out_list += [a.clone().to(out_type), b.clone().to(out_type)]

        if inplace:
            in_list = out_list
        else:
            in_list = [out.clone().to(in_type) for out in out_list]

        applier(tex.multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)

        overflow_buf.zero_()
        in_list[t][ind] = val
        applier(tex.multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)
        assert overflow_buf.item() > 0

    downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=inplace)
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        0,
        0,
        float("nan"),
        inplace=inplace,
    )
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        2 * repeat - 1,
        sizeb - 1,
        float("inf"),
        inplace=inplace,
    )
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        2 * (repeat // 2),
        sizea // 2,
        float("inf"),
        inplace=inplace,
    )


@pytest.mark.parametrize("input_size_pair", input_size_pairs)
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
def test_multi_tensor_scale_tensor(input_size_pair, applier, repeat, in_type, out_type, inplace):
    if inplace is True and (out_type is not in_type):
        pytest.skip("inplace=True and out_type != in_type is not supported.")
    elif (in_type == torch.float16 and out_type == torch.bfloat16) or (
        in_type == torch.bfloat16 and out_type == torch.float16
    ):
        pytest.skip("float16 to bfloat16 is not necessary and vice versa.")

    device = torch.device("cuda")
    scale = 4.0
    inv_scale_cuda = torch.tensor([1.0 / scale], dtype=torch.float32, device=device)
    overflow_buf = torch.zeros(1, dtype=torch.int32, device=device)
    ref = torch.tensor([1.0], dtype=torch.float32, device=device)
    sizea, sizeb = input_size_pair

    def downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=False):
        overflow_buf.zero_()
        a = torch.full([sizea], scale, dtype=torch.float32, device=device)
        b = torch.full([sizeb], scale, dtype=torch.float32, device=device)

        out_list = []
        for _ in range(repeat):
            out_list += [a.clone().to(out_type), b.clone().to(out_type)]

        if inplace:
            in_list = out_list
        else:
            in_list = [out.clone().to(in_type) for out in out_list]

        applier(tex.multi_tensor_scale_tensor, overflow_buf, [in_list, out_list], inv_scale_cuda)

        assert all([torch.allclose(out, ref.to(out_type)) for out in out_list])
        assert overflow_buf.item() == 0

    def find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        t,
        ind,
        val,
        inplace=False,
    ):
        overflow_buf.zero_()
        a = torch.full([sizea], scale, dtype=torch.float32, device=device)
        b = torch.full([sizeb], scale, dtype=torch.float32, device=device)

        out_list = []
        for _ in range(repeat):
            out_list += [a.clone().to(out_type), b.clone().to(out_type)]

        if inplace:
            in_list = out_list
        else:
            in_list = [out.clone().to(in_type) for out in out_list]

        applier(tex.multi_tensor_scale_tensor, overflow_buf, [in_list, out_list], inv_scale_cuda)

        overflow_buf.zero_()
        in_list[t][ind] = val
        applier(tex.multi_tensor_scale_tensor, overflow_buf, [in_list, out_list], inv_scale_cuda)
        assert overflow_buf.item() > 0

    downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=inplace)
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        0,
        0,
        float("nan"),
        inplace=inplace,
    )
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        2 * repeat - 1,
        sizeb - 1,
        float("inf"),
        inplace=inplace,
    )
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        2 * (repeat // 2),
        sizea // 2,
        float("inf"),
        inplace=inplace,
    )


@pytest.mark.parametrize("input_size_pair", input_size_pairs)
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("per_tensor", [False, True])
def test_multi_tensor_l2norm(input_size_pair, applier, repeat, in_type, per_tensor):
    sizea, sizeb = input_size_pair
    device = torch.device("cuda")
    val = 4.0
    overflow_buf = torch.zeros(1, dtype=torch.int32, device=device)

    overflow_buf.zero_()
    a = torch.full([sizea], val, dtype=torch.float32, device=device)
    b = torch.full([sizeb], val, dtype=torch.float32, device=device)

    in_list = []
    for i in range(repeat):
        in_list += [a.clone().to(in_type), b.clone().to(in_type)]

    if per_tensor:
        norm, norm_per_tensor = applier(tex.multi_tensor_l2norm, overflow_buf, [in_list], True)
        normab = torch.cat((a.norm().view(1), b.norm().view(1)))
        norm_per_tensor = norm_per_tensor.view(-1, 2)
    else:
        norm, _ = applier(tex.multi_tensor_l2norm, overflow_buf, [in_list], False)

    reference = torch.full(
        [(sizea + sizeb) * repeat], val, dtype=torch.float32, device=device
    ).norm()

    torch.testing.assert_close(norm, reference.broadcast_to(norm.shape))
    if per_tensor:
        torch.testing.assert_close(norm_per_tensor, normab.broadcast_to(norm_per_tensor.shape))
    assert overflow_buf.item() == 0


def _assert_l2norm_single_launch_bitwise(applier, tensors, per_tensor):
    """multi_tensor_l2norm_single_launch returns the bits of multi_tensor_l2norm."""
    device = tensors[0].device
    overflow_ref = torch.zeros(1, dtype=torch.int32, device=device)
    overflow_out = torch.zeros(1, dtype=torch.int32, device=device)
    norm_ref, per_tensor_ref = applier(tex.multi_tensor_l2norm, overflow_ref, [tensors], per_tensor)
    norm, norm_per_tensor = applier(
        tex.multi_tensor_l2norm_single_launch, overflow_out, [tensors], per_tensor
    )
    assert torch.equal(
        norm.view(torch.int32), norm_ref.view(torch.int32)
    ), f"norm {norm.item()!r} != {norm_ref.item()!r}"
    if per_tensor:
        assert torch.equal(norm_per_tensor.view(torch.int32), per_tensor_ref.view(torch.int32))
    assert overflow_out.item() == overflow_ref.item()


def _random_list(sizes, dtype, device, seed):
    """Random tensors with magnitudes spread over 6 decades (sums depend on the order)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i, n in enumerate(sizes):
        scale = 10.0 ** (6.0 * ((i * 7919) % 101) / 101 - 3.0)
        out.append((torch.randn(n, generator=gen, device=device) * scale).to(dtype))
    return out


@pytest.mark.parametrize("input_size_pair", input_size_pairs)
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("per_tensor", [False, True])
def test_multi_tensor_l2norm_single_launch(input_size_pair, applier, repeat, in_type, per_tensor):
    sizea, sizeb = input_size_pair
    tensors = _random_list([sizea, sizeb] * repeat, in_type, torch.device("cuda"), seed=repeat)
    _assert_l2norm_single_launch_bitwise(applier, tensors, per_tensor)


# Lists that hit every way multi_tensor_apply<1> ends a launch (chunk size 64, so 320 blocks are
# 320 x 64 elements): a tensor that fills the launch on its last chunk, tensors split across one
# or several launches, 110 tensors, 110 tensors after a split (the carried tensor counts), both
# limits at once, and more than 128 tensors (several single-launch kernels).
_layout_cases = {
    "fill_on_last_chunk": [64 * 320, 5, 64 * 319 + 1, 64],
    "split_mid_tensor": [64 * 100, 64 * 300 + 3, 64 * 1000 + 17, 9],
    "tensors_full": [1 + (i % 64) for i in range(250)],
    "tensors_full_after_split": [64 * 319, 64 * 2] + [7] * 120 + [64 * 330],
    "both_limits": [64 * 2] * 109 + [64 * 102] + [3, 64 * 641, 1],
    "many_launches": [(i * 2654435761) % (64 * 700) + 1 for i in range(300)],
}


@pytest.mark.parametrize("case", list(_layout_cases))
@pytest.mark.parametrize("in_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("per_tensor", [False, True])
def test_multi_tensor_l2norm_single_launch_layouts(case, in_type, per_tensor):
    tensors = _random_list(_layout_cases[case], in_type, torch.device("cuda"), seed=7)
    _assert_l2norm_single_launch_bitwise(MultiTensorApply(64), tensors, per_tensor)


def _bucket_shard_views(buffers, bucket_numel, dp_size, rank, dtype, device, seed):
    """Gradient views of one data-parallel rank, as Megatron's distributed optimizer computes the
    grad norm over them: for every grad buffer, params laid out in reverse order at 64-element
    boundaries, a new bucket once bucket_numel is reached, every bucket (padded to a multiple of
    dp_size x 128 elements) split into dp_size equal shards, and the views of each param on the
    rank's shard. Views are cut at shard boundaries, so many chunks are partial."""
    gen = torch.Generator(device=device).manual_seed(seed)
    views = []
    for param_sizes in buffers:
        buckets, current, offset = [], [], 0
        for n in reversed(param_sizes):
            current.append((offset, n))
            offset = (offset + n + 63) // 64 * 64
            if offset >= bucket_numel:
                buckets.append((current, offset))
                current, offset = [], 0
        if current:
            buckets.append((current, offset))
        for params, numel in buckets:
            numel = (numel + dp_size * 128 - 1) // (dp_size * 128) * (dp_size * 128)
            shard = numel // dp_size
            lo, hi = rank * shard, (rank + 1) * shard
            buf = torch.randn(shard, generator=gen, device=device).mul_(1e-2).to(dtype)
            for start, n in params:
                a, b = max(start, lo), min(start + n, hi)
                if a < b:
                    views.append(buf[a - lo : b - lo])
    return views


def _moe_grad_buffers(layers, hidden, ffn, experts, heads, kv_heads, head_dim, vocab):
    """Parameter sizes of the dense and the expert grad buffer of a GPT-OSS-like MoE model
    (single grouped expert weights)."""
    qkv = (heads + 2 * kv_heads) * head_dim
    dense_layer = [
        hidden,  # input layer norm
        qkv * hidden,
        qkv,
        hidden * heads * head_dim,
        hidden,
        heads,  # attention sinks
        hidden,  # pre-MLP layer norm
        experts * hidden,  # router
        experts,
    ]
    expert_layer = [
        experts * 2 * ffn * hidden,  # fc1 (gated)
        experts * 2 * ffn,
        experts * hidden * ffn,  # fc2
        experts * hidden,
    ]
    dense = [vocab * hidden] + dense_layer * layers + [hidden, vocab * hidden]
    return [dense, expert_layer * layers]


@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize("in_type", [torch.float32, torch.bfloat16])
def test_multi_tensor_l2norm_single_launch_bucket_shards(rank, in_type):
    # GPT-OSS-20B structure (24 layers, 32 experts, GQA 64/8) at 1/10 of the hidden size and 1/100
    # of the 768M-element buckets, DP 8: about 25M elements, 26 bucket shards, 46-74 views per rank.
    buffers = _moe_grad_buffers(
        layers=24, hidden=288, ffn=288, experts=32, heads=64, kv_heads=8, head_dim=8, vocab=2011
    )
    views = _bucket_shard_views(buffers, 7_680_000, 8, rank, in_type, torch.device("cuda"), rank)
    chunks = sum((v.numel() + 2048 * 32 - 1) // (2048 * 32) for v in views)
    assert chunks > 320  # several multi_tensor_l2norm launches
    _assert_l2norm_single_launch_bitwise(MultiTensorApply(2048 * 32), views, False)


@pytest.mark.parametrize("bad", ["inf", "nan"])
def test_multi_tensor_l2norm_single_launch_nonfinite(bad):
    tensors = _random_list([555, 2048 * 32 + 1, 33333] * 3, torch.bfloat16, torch.device("cuda"), 1)
    tensors[4][1000] = float(bad)
    _assert_l2norm_single_launch_bitwise(MultiTensorApply(2048 * 32), tensors, True)


def test_multi_tensor_l2norm_single_launch_cuda_graph():
    applier = MultiTensorApply(2048 * 32)
    tensors = _random_list(
        [7777 * 77, 555, 2048 * 32 + 1] * 40, torch.bfloat16, torch.device("cuda"), 3
    )
    overflow = torch.zeros(1, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        applier(tex.multi_tensor_l2norm_single_launch, overflow, [tensors], False)  # warm-up
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm, _ = applier(tex.multi_tensor_l2norm_single_launch, overflow, [tensors], False)
    for seed in (11, 12):
        new = _random_list([t.numel() for t in tensors], torch.bfloat16, torch.device("cuda"), seed)
        for t, v in zip(tensors, new):
            t.copy_(v)
        graph.replay()
        norm_ref, _ = applier(tex.multi_tensor_l2norm, overflow, [tensors], False)
        assert torch.equal(norm.view(torch.int32), norm_ref.view(torch.int32))


def test_multi_tensor_l2norm_single_launch_rejects_mixed_dtypes():
    tensors = [torch.ones(10, device="cuda"), torch.ones(10, device="cuda", dtype=torch.bfloat16)]
    overflow = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="one dtype"):
        MultiTensorApply(2048 * 32)(
            tex.multi_tensor_l2norm_single_launch, overflow, [tensors], False
        )


def _l2norm_split(chunk_size, tensors, groups, output=None):
    """Split single-launch L2 norm: the partials of every group of tensor indices (in the given
    order), then the finalize. Returns (norm, block-sum array)."""
    first_slot, size = tex.multi_tensor_l2norm_single_launch_slots(chunk_size, [tensors])
    assert len(first_slot) == len(tensors)
    if output is None:
        output = torch.zeros(size, dtype=torch.float32, device=tensors[0].device)
    assert output.numel() == size
    overflow = torch.zeros(1, dtype=torch.int32, device=tensors[0].device)
    for group in groups:
        tex.multi_tensor_l2norm_single_launch_partials(
            chunk_size,
            overflow,
            [[tensors[i] for i in group]],
            output,
            [first_slot[i] for i in group],
        )
    return tex.multi_tensor_l2norm_single_launch_finalize(output), output


def _split_groups(num_tensors, num_groups, seed):
    """Tensor indices in num_groups groups: contiguous runs of random length, issued in a random
    order, plus one group of interleaved indices."""
    gen = torch.Generator().manual_seed(seed)
    cuts = sorted(torch.randperm(max(num_tensors - 1, 1), generator=gen)[: num_groups - 1].tolist())
    bounds = [0] + [c + 1 for c in cuts] + [num_tensors]
    groups = [list(range(a, b)) for a, b in zip(bounds[:-1], bounds[1:]) if a < b]
    order = torch.randperm(len(groups), generator=gen).tolist()
    groups = [groups[i] for i in order]
    if len(groups) > 2:  # merge two groups into one strided group
        groups = [sorted(groups[0] + groups[1], reverse=True)] + groups[2:]
    return groups


@pytest.mark.parametrize("case", list(_layout_cases))
@pytest.mark.parametrize("in_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_groups", [1, 3, 17])
def test_multi_tensor_l2norm_single_launch_split(case, in_type, num_groups):
    applier = MultiTensorApply(64)
    tensors = _random_list(_layout_cases[case], in_type, torch.device("cuda"), seed=7)
    overflow = torch.zeros(1, dtype=torch.int32, device="cuda")
    norm_ref, _ = applier(tex.multi_tensor_l2norm, overflow, [tensors], False)
    groups = _split_groups(len(tensors), num_groups, seed=num_groups)
    norm, output = _l2norm_split(64, tensors, groups)
    assert torch.equal(norm.view(torch.int32), norm_ref.view(torch.int32))
    # The block-sum array is reused: every stored slot is overwritten by the next evaluation.
    new = _random_list(_layout_cases[case], in_type, torch.device("cuda"), seed=8)
    norm_ref, _ = applier(tex.multi_tensor_l2norm, overflow, [new], False)
    norm, _ = _l2norm_split(64, new, list(reversed(groups)), output)
    assert torch.equal(norm.view(torch.int32), norm_ref.view(torch.int32))


@pytest.mark.parametrize("rank", [0, 5])
def test_multi_tensor_l2norm_single_launch_split_bucket_shards(rank):
    # As test_multi_tensor_l2norm_single_launch_bucket_shards, partials per bucket shard.
    buffers = _moe_grad_buffers(
        layers=24, hidden=288, ffn=288, experts=32, heads=64, kv_heads=8, head_dim=8, vocab=2011
    )
    views = _bucket_shard_views(
        buffers, 7_680_000, 8, rank, torch.bfloat16, torch.device("cuda"), rank
    )
    # Group the views by their shard (one storage per bucket shard), last bucket first.
    by_storage = {}
    for i, v in enumerate(views):
        by_storage.setdefault(v.untyped_storage().data_ptr(), []).append(i)
    groups = list(reversed(list(by_storage.values())))
    assert len(groups) > 10
    overflow = torch.zeros(1, dtype=torch.int32, device="cuda")
    norm_ref, _ = MultiTensorApply(2048 * 32)(tex.multi_tensor_l2norm, overflow, [views], False)
    norm, _ = _l2norm_split(2048 * 32, views, groups)
    assert torch.equal(norm.view(torch.int32), norm_ref.view(torch.int32))


def test_multi_tensor_l2norm_single_launch_split_cuda_graph():
    sizes = [7777 * 77, 555, 2048 * 32 + 1] * 40
    tensors = _random_list(sizes, torch.bfloat16, torch.device("cuda"), 3)
    first_slot, size = tex.multi_tensor_l2norm_single_launch_slots(2048 * 32, [tensors])
    output = torch.zeros(size, dtype=torch.float32, device="cuda")
    overflow = torch.zeros(1, dtype=torch.int32, device="cuda")
    groups = _split_groups(len(tensors), 9, seed=1)

    def run():
        for group in groups:
            tex.multi_tensor_l2norm_single_launch_partials(
                2048 * 32, overflow, [[tensors[i] for i in group]], output,
                [first_slot[i] for i in group],
            )
        return tex.multi_tensor_l2norm_single_launch_finalize(output)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()  # warm-up
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm = run()
    for seed in (11, 12):
        new = _random_list(sizes, torch.bfloat16, torch.device("cuda"), seed)
        for t, v in zip(tensors, new):
            t.copy_(v)
        graph.replay()
        norm_ref, _ = MultiTensorApply(2048 * 32)(
            tex.multi_tensor_l2norm, overflow, [tensors], False
        )
        assert torch.equal(norm.view(torch.int32), norm_ref.view(torch.int32))


def test_multi_tensor_l2norm_single_launch_split_checks():
    tensors = _random_list([555, 2048 * 32 + 1], torch.bfloat16, torch.device("cuda"), 1)
    first_slot, size = tex.multi_tensor_l2norm_single_launch_slots(2048 * 32, [tensors])
    assert first_slot == [0, 1] and size == 320
    output = torch.zeros(size, dtype=torch.float32, device="cuda")
    overflow = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="outside the block-sum array"):
        tex.multi_tensor_l2norm_single_launch_partials(
            2048 * 32, overflow, [tensors[1:]], output, [319]
        )
    with pytest.raises(RuntimeError, match="one slot per tensor"):
        tex.multi_tensor_l2norm_single_launch_partials(
            2048 * 32, overflow, [tensors], output, [0]
        )
    with pytest.raises(RuntimeError, match="multiple of 320"):
        tex.multi_tensor_l2norm_single_launch_finalize(output[:100])


@pytest.mark.parametrize("input_size_pair", input_size_pairs)
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("per_tensor", [False, True])
def test_multi_tensor_unscale_l2norm(input_size_pair, applier, repeat, in_type, per_tensor):
    sizea, sizeb = input_size_pair
    device = torch.device("cuda")
    val = 4.0
    inv_scale = 0.5
    inv_scale_cuda = torch.tensor([inv_scale], dtype=torch.float32, device=device)
    overflow_buf = torch.zeros(1, dtype=torch.int32, device=device)

    overflow_buf.zero_()
    a = torch.full([sizea], val, dtype=torch.float32, device=device)
    b = torch.full([sizeb], val, dtype=torch.float32, device=device)

    in_list = []
    for i in range(repeat):
        in_list += [a.clone().to(in_type), b.clone().to(in_type)]

    if per_tensor:
        norm, norm_per_tensor = applier(
            tex.multi_tensor_unscale_l2norm,
            overflow_buf,
            [in_list],
            inv_scale_cuda,
            True,
        )
        normab = torch.cat(((a * inv_scale).norm().view(1), (b * inv_scale).norm().view(1)))
        norm_per_tensor = norm_per_tensor.view(-1, 2)
    else:
        norm, _ = applier(
            tex.multi_tensor_unscale_l2norm,
            overflow_buf,
            [in_list],
            inv_scale_cuda,
            True,
        )

    reference = torch.full(
        [(sizea + sizeb) * repeat], val * inv_scale, dtype=torch.float32, device=device
    ).norm()

    torch.testing.assert_close(norm, reference.broadcast_to(norm.shape))
    if per_tensor:
        torch.testing.assert_close(norm_per_tensor, normab.broadcast_to(norm_per_tensor.shape))
    assert overflow_buf.item() == 0


@pytest.mark.parametrize("input_size_pair", input_size_pairs + [(1, 1)])
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("pow_2_scales", [False, True])
@pytest.mark.parametrize("epsilon", [0.0, 100.0])
def test_multi_tensor_compute_scale_and_scale_inv(
    input_size_pair, applier, repeat, fp8_dtype, pow_2_scales, epsilon
):
    sizea, sizeb = input_size_pair
    device = torch.device("cuda")
    overflow_buf = torch.zeros(1, dtype=torch.int32, device=device)
    a = torch.randn([sizea], dtype=torch.float32, device=device).abs()
    b = torch.randn([sizeb], dtype=torch.float32, device=device).abs()
    max_fp8 = torch.finfo(fp8_dtype).max

    amax_list = []
    for i in range(repeat):
        amax_list += [a.clone(), b.clone()]

    scale_list = [torch.empty_like(x) for x in amax_list]
    scale_inv_list = [torch.empty_like(x) for x in amax_list]

    applier(
        tex.multi_tensor_compute_scale_and_scale_inv,
        overflow_buf,
        [amax_list, scale_list, scale_inv_list],
        max_fp8,
        pow_2_scales,
        epsilon,
    )

    for amax, scale, scale_inv in zip(amax_list, scale_list, scale_inv_list):
        scale_ref, scale_inv_ref, _ = scale_from_amax_tensor(
            torch.float32, amax, fp8_dtype, eps=epsilon, pow_2_scales=pow_2_scales
        )
        torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
        torch.testing.assert_close(scale_inv, scale_inv_ref, rtol=0, atol=0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("input_size_pair", input_size_pairs + [(1, 1)])
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
def test_multi_tensor_compute_scale_inv_e8m0(input_size_pair, applier, repeat):
    sizea, sizeb = input_size_pair
    device = torch.device("cuda")
    a = torch.randn([sizea], dtype=torch.bfloat16, device=device).abs()
    b = torch.randn([sizeb], dtype=torch.bfloat16, device=device).abs()

    amax_list = []
    for _ in range(repeat):
        amax_list += [a.clone(), b.clone()]
    scale_inv_list = [torch.empty_like(x).to(torch.uint8) for x in amax_list]

    applier(
        tex.multi_tensor_compute_scale_inv_e8m0,
        None,  # overflow_buf
        [amax_list, scale_inv_list],
    )

    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    for amax, scale_inv in zip(amax_list, scale_inv_list):
        scale_inv_u32 = (amax.float() / max_fp8).view(torch.int)
        exponent = scale_inv_u32 // 2**23
        mantissa = scale_inv_u32 & 0x7FFFFF
        exponent += (
            ((mantissa > 0) & (exponent != 0xFE)) & ~((exponent == 0) & (mantissa <= 0x400000))
        ).to(torch.int)
        torch.testing.assert_close(exponent.to(torch.uint8), scale_inv)
