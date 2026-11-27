# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernels for an FP8 (E4M3) LM-head backward with row-scaled, stochastically rounded
cross-entropy gradients."""

import triton
import triton.language as tl
from triton.language.random import philox as _philox

E4M3_MAX = 448.0


@triton.jit
def _stochastic_round_e4m3(a, u):
    """Unbiased stochastic rounding of fp32 ``a`` (|a| <= 448) onto the E4M3 grid, including the
    subnormal range (spacing 2^-9 below 2^-6), with uniforms ``u`` in (0, 1). Returns an exactly
    representable fp32 value."""
    aa = tl.abs(a)
    e = (aa.to(tl.int32, bitcast=True) >> 23) & 0xFF
    e = tl.maximum(e, 121)
    ulp = ((e - 3) << 23).to(tl.float32, bitcast=True)
    inv_ulp = ((257 - e) << 23).to(tl.float32, bitcast=True)
    r = tl.floor(aa * inv_ulp + u) * ulp
    r = tl.minimum(r, 448.0)
    return tl.where(a < 0, -r, r)


@triton.jit
def _philox_key(rng_state_ptr):
    """Philox key and the two high counter words from a (seed, offset) int64 pair of the CUDA
    generator; one Philox call per subsequence uses 4 of the reserved offsets."""
    seed = tl.load(rng_state_ptr).to(tl.uint64, bitcast=True)
    offset = tl.load(rng_state_ptr + 1).to(tl.uint64, bitcast=True)
    blk = offset >> 2
    return seed, (blk & 0xFFFFFFFF).to(tl.uint32), (blk >> 32).to(tl.uint32)


@triton.jit(do_not_specialize=["rank"])
def cross_entropy_fp8_grad_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride_b,
    Y_stride_s,
    loss_ptr,
    L_stride_b,
    L_stride_s,
    scale_ptr,
    Q_ptr,
    Q_stride,
    row_scale_ptr,
    rng_state_ptr,
    rank,
    n_batch,
    n_cols,
    ignore_idx,
    BLOCK_SIZE: tl.constexpr,
    GRAD_BLOCK: tl.constexpr,
):
    """Cross-entropy loss of one row of logits and its gradient as E4M3 with a row scale.

    Row ``r`` holds token (s, b) = (r // n_batch, r % n_batch) of a sequence-first [s, b] layout,
    whose target and loss are read and written with [b, s] strides. The gradient
    ``g = (softmax - onehot) * scale`` satisfies |g| <= (1 - p_y) * scale, so it is quantized as
    ``g * 448 / ((1 - p_y) * scale)`` with stochastic rounding and the dequantization scale
    ``(1 - p_y) * scale / 448`` goes to row_scale_ptr. Ignored rows get a zero gradient, a zero
    scale and a zero loss.
    """
    row = tl.program_id(0).to(tl.int64)
    s_idx = row // n_batch
    b_idx = row - s_idx * n_batch
    y = tl.load(Y_ptr + b_idx * Y_stride_b + s_idx * Y_stride_s)
    X_ptr += row * X_stride
    Q_ptr += row * Q_stride
    loss_out = loss_ptr + b_idx * L_stride_b + s_idx * L_stride_s
    # The gradient pass works on 8 contiguous chunks of QR elements per GRAD_BLOCK; one Philox
    # call (4 x 32 bits) gives the 16-bit uniforms of the 8 elements at the same chunk position.
    QR: tl.constexpr = GRAD_BLOCK // 8
    rq = tl.arange(0, QR)
    if y == ignore_idx:
        for i in range(0, n_cols, QR):
            offs = i + rq
            tl.store(
                Q_ptr + offs,
                tl.zeros([QR], dtype=tl.float32).to(Q_ptr.dtype.element_ty),
                mask=offs < n_cols,
            )
        tl.store(row_scale_ptr + row, 0.0)
        tl.store(loss_out, 0.0)
        return
    # Online softmax statistics
    m = float("-inf")
    d = 0.0
    x_y = tl.load(X_ptr + y).to(tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
    # Gradient pass
    scale = tl.load(scale_ptr)
    p_y = tl.exp(x_y - m) / d
    amax = tl.maximum(1.0 - p_y, 5.9604644775390625e-08)
    qs = 448.0 / amax
    c = qs / d
    a_y = tl.maximum((p_y - 1.0) * qs, -448.0)
    tl.store(row_scale_ptr + row, amax * (scale / 448.0))
    seed, c0, c1 = _philox_key(rng_state_ptr)
    c3 = rank.to(tl.int64).to(tl.uint32)
    n_it = (n_cols + GRAD_BLOCK - 1) // GRAD_BLOCK
    sub_base = (row * n_it * QR).to(tl.uint32)
    zero = rq.to(tl.uint32) * 0
    for i in range(0, n_cols, GRAD_BLOCK):
        sub = sub_base + ((i // GRAD_BLOCK) * QR).to(tl.uint32) + rq.to(tl.uint32)
        k0, k1, k2, k3 = _philox(seed, zero + c0, zero + c1, sub, zero + c3)
        for jc in tl.static_range(8):
            offs = i + jc * QR + rq
            mask = offs < n_cols
            x = tl.load(X_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
            a = tl.where(offs == y, a_y, tl.exp(x - m) * c)
            if jc // 2 == 0:
                k = k0
            elif jc // 2 == 1:
                k = k1
            elif jc // 2 == 2:
                k = k2
            else:
                k = k3
            if jc % 2 == 0:
                h = k & 0xFFFF
            else:
                h = k >> 16
            u = (h.to(tl.float32) + 0.5) * 1.52587890625e-05
            q = _stochastic_round_e4m3(a, u)
            tl.store(Q_ptr + offs, q.to(Q_ptr.dtype.element_ty), mask=mask)
    loss = -(x_y - m - tl.log(d))
    tl.store(loss_out, loss)


@triton.jit
def amax_kernel(x_ptr, n, amax_ptr, BLOCK: tl.constexpr):
    """max(|x|) into amax_ptr (zero-initialized fp32)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.atomic_max(amax_ptr, tl.max(acc), sem="relaxed")


@triton.jit
def quantize_fp8_kernel(x_ptr, n, amax_ptr, q_ptr, scale_inv_ptr, BLOCK: tl.constexpr):
    """E4M3 with round-to-nearest and the per-tensor scale 448 / amax; writes amax / 448."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    amax = tl.load(amax_ptr)
    qs = tl.where(amax > 0.0, 448.0 / amax, 0.0)
    if pid == 0:
        tl.store(scale_inv_ptr, amax / 448.0)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(x * qs, -448.0), 448.0)
        tl.store(q_ptr + offs, q.to(q_ptr.dtype.element_ty), mask=offs < n)


@triton.jit
def row_scaled_amax_kernel(
    x_ptr, row_scale_ptr, n_rows, H, amax_ptr, ROWS: tl.constexpr, BLOCK_H: tl.constexpr
):
    """max(|diag(row_scale) . x|) into amax_ptr (zero-initialized fp32)."""
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for r in range(pid * ROWS, tl.minimum(pid * ROWS + ROWS, n_rows)):
        s = tl.abs(tl.load(row_scale_ptr + r))
        for h in range(0, H, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            x = tl.load(x_ptr + r.to(tl.int64) * H + offs, mask=offs < H, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(x * s))
    tl.atomic_max(amax_ptr, tl.max(acc), sem="relaxed")


@triton.jit(do_not_specialize=["rank"])
def row_scaled_quantize_fp8_kernel(
    x_ptr,
    row_scale_ptr,
    n_rows,
    H,
    amax_ptr,
    q_ptr,
    scale_inv_ptr,
    rng_state_ptr,
    rank,
    ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """E4M3 of diag(row_scale) . x * 448 / amax with stochastic rounding; writes amax / 448.
    One Philox call covers 4 chunks of BLOCK_H / 4 elements."""
    pid = tl.program_id(0)
    amax = tl.load(amax_ptr)
    qs = tl.where(amax > 0.0, 448.0 / amax, 0.0)
    if pid == 0:
        tl.store(scale_inv_ptr, amax / 448.0)
    QH: tl.constexpr = BLOCK_H // 4
    rq = tl.arange(0, QH)
    n_hb = (H + BLOCK_H - 1) // BLOCK_H
    seed, c0, c1 = _philox_key(rng_state_ptr)
    c3 = rank.to(tl.int64).to(tl.uint32)
    zero = rq.to(tl.uint32) * 0
    for r in range(pid * ROWS, tl.minimum(pid * ROWS + ROWS, n_rows)):
        s = tl.load(row_scale_ptr + r) * qs
        base = r.to(tl.int64) * H
        for hb in range(0, n_hb):
            sub = ((r.to(tl.int64) * n_hb + hb) * QH).to(tl.uint32) + rq.to(tl.uint32)
            k0, k1, k2, k3 = _philox(seed, zero + c0, zero + c1, sub, zero + c3)
            for jc in tl.static_range(4):
                offs = hb * BLOCK_H + jc * QH + rq
                x = tl.load(x_ptr + base + offs, mask=offs < H, other=0.0).to(tl.float32)
                a = x * s
                if jc == 0:
                    k = k0
                elif jc == 1:
                    k = k1
                elif jc == 2:
                    k = k2
                else:
                    k = k3
                u = ((k >> 8).to(tl.float32) + 0.5) * 5.9604644775390625e-08
                q = _stochastic_round_e4m3(a, u)
                tl.store(q_ptr + base + offs, q.to(q_ptr.dtype.element_ty), mask=offs < H)


@triton.jit
def row_scale_kernel(
    y_ptr, row_scale_ptr, g_ptr, out_ptr, n_rows, H, ROWS: tl.constexpr, BLOCK_H: tl.constexpr
):
    """out = diag(row_scale) . y * g."""
    pid = tl.program_id(0)
    g = tl.load(g_ptr).to(tl.float32)
    for r in range(pid * ROWS, tl.minimum(pid * ROWS + ROWS, n_rows)):
        s = tl.load(row_scale_ptr + r) * g
        for h in range(0, H, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            y = tl.load(y_ptr + r.to(tl.int64) * H + offs, mask=offs < H, other=0.0)
            tl.store(
                out_ptr + r.to(tl.int64) * H + offs,
                (y * s).to(out_ptr.dtype.element_ty),
                mask=offs < H,
            )
