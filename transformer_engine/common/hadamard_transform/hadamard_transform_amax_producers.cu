/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file hadamard_transform_amax_producers.cu
 *  \brief Producers of BF16 tensors that also emit the tensor's NVFP4 RHT amaxes.
 *
 *  An NVFP4 cast with random Hadamard transform and post-RHT amax first runs
 *  nvte_hadamard_transform_amax over its input (one more read of the tensor). The kernels here
 *  write a tensor and reduce the same two amaxes in the same pass, so the cast can take them:
 *
 *  - nvte_add_hadamard_transform_amax: out = a + b (bf16(float(a) + float(b)), the elementwise
 *    BF16 add), e.g. the accumulation of the two gradients of a residual-stream tensor.
 *  - nvte_fused_qkv_rope_backward_hadamard_transform_amax: RoPE backward of the Q, K and V
 *    gradients into the gradient of the fused QKV projection output (SBHD, non-interleaved,
 *    fp32 cos/sin tables, out = fma(g, cos, g_rot * sin) rounded to BF16).
 *  - nvte_quantize_rowwise_with_amax_hadamard_transform_amax: row-wise NVFP4 cast (round to
 *    nearest, given per-tensor amax) of an input, plus the two amaxes of the input for its later
 *    column-wise RHT cast.
 *
 *  The amaxes are bitwise those of nvte_hadamard_transform_amax on the written tensor: the same
 *  wmma.m16n16k16 BF16 MMA against the same sign-masked 16x16 Hadamard fragments on
 *  16-token x 16-column blocks, the same BF16 rounding of the result and the same
 *  max.xorsign.abs reduction (max is order independent).
 *
 *  The grids are not persistent, so communication kernels running beside them find SM resources.
 *  Amaxes end in a deterministic two-level reduction (per-CTA partials or an atomicMax pair); the
 *  last CTA writes them and re-arms the counter, so a zero-filled workspace stays zero-filled and
 *  is reused (and captured in a CUDA graph) without a memset.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <transformer_engine/hadamard_transform.h>

#include <algorithm>
#include <cstdint>

#include "common/common.h"
#include "common/util/cuda_runtime.h"

namespace transformer_engine {
namespace hadamard_amax_producers {

constexpr int kTileT = 64;  // tokens per tile (four 16-token RHT groups)
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBufChunks = kTileT * 64 / 8;  // 16-byte chunks per 64 x 64 bf16 buffer (512)

__device__ __forceinline__ uint32_t swz(uint32_t row, uint32_t chunk) {
  // TE swizzle_128B_atom_32B: 64 bf16 (8 x 16B chunks) per row.
  return row * 8 + (chunk ^ ((row * 2) % 8));
}

__device__ __forceinline__ uint32_t bmax(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t bmax4(uint4 v) { return bmax(bmax(v.x, v.y), bmax(v.z, v.w)); }

__device__ __forceinline__ void st_na(void *p, uint4 v) {
  asm volatile("st.global.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ __forceinline__ float4 ldg_f4(const float *p) {
  return __ldg(reinterpret_cast<const float4 *>(p));
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// --- Mirrors of hadamard_transform_utils.cuh and HadamardAmaxTmaKernel ---------------------------

__device__ __forceinline__ void get_hadamard_t_fragment(uint32_t *had_frag_t,
                                                        uint16_t random_sign_mask_t) {
  const int32_t tid = threadIdx.x % 32;
  float temp_t[2];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const uint32_t r = i * 8 + tid / 4;
#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const uint32_t c = j * 8 + k + tid % 4 * 2;
        const int32_t base_sign = __popc(r & c);
        const int32_t sign_t = ((random_sign_mask_t >> c) ^ base_sign);
        temp_t[k] = copysignf(0.25f, __int_as_float(sign_t << 31));
      }
      asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t"
                   : "=r"(had_frag_t[i * 2 + j])
                   : "f"(temp_t[1]), "f"(temp_t[0]));
    }
  }
}

// Transposed-RHT amax (packed bf16x2) of one 16-token x 16-column block of a swizzled 64 x 64
// buffer; `addr` is the lane's ldmatrix row pointer.
__device__ __forceinline__ uint32_t rht_t_amax16x16(const uint4 *addr, const uint32_t *b) {
  uint32_t a0, a1, a2, a3;
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(smem_addr));
  const uint32_t zero = 0;
  uint32_t t0, t1, t2, t3, t4, t5, t6, t7;
  // a_frag order (0, 2, 1, 3) as in TE ComputeKernel's transposed branch.
  asm volatile(
      "wmma.mma.sync.aligned.row.row.m16n16k16.f32.bf16.bf16.f32 \n"
      "{%0, %1, %2, %3, %4, %5, %6, %7}, \n"
      "{%8, %9, %10, %11}, \n"
      "{%12, %13, %14, %15}, \n"
      "{%16, %17, %18, %19, %20, %21, %22, %23};\n\t"
      : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3), "=r"(t4), "=r"(t5), "=r"(t6), "=r"(t7)
      : "r"(a0), "r"(a2), "r"(a1), "r"(a3), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(zero),
        "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero));
  uint32_t c0, c1, c2, c3;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c0) : "r"(t1), "r"(t0));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c1) : "r"(t3), "r"(t2));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c2) : "r"(t5), "r"(t4));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c3) : "r"(t7), "r"(t6));
  return bmax(bmax(c0, c2), bmax(c1, c3));
}

__device__ __forceinline__ float unpack_max(uint32_t packed) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162 *>(&packed);
  float fa = __bfloat162float(v.x);
  float fb = __bfloat162float(v.y);
  float r;
  asm volatile("max.xorsign.abs.f32 %0, %1, %2;\n\t" : "=f"(r) : "f"(fa), "f"(fb));
  return fabsf(r);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Per-CTA maxima -> partials; the last CTA writes amax_out[0..1] and re-arms the counter.
__device__ __forceinline__ void finish_amax_f(float row_f, uint32_t col_packed,
                                              float *__restrict__ amax_out,
                                              float *__restrict__ partials,
                                              unsigned int *__restrict__ counter) {
  __shared__ float red[2 * kWarps];
  __shared__ bool is_last;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  float fr = warp_max(row_f);
  float fc = warp_max(unpack_max(col_packed));
  if (lane == 0) {
    red[warp] = fr;
    red[kWarps + warp] = fc;
  }
  __syncthreads();
  if (warp == 0) {
    float r = lane < kWarps ? red[lane] : 0.f;
    float c = lane < kWarps ? red[kWarps + lane] : 0.f;
    r = warp_max(r);
    c = warp_max(c);
    if (lane == 0) {
      partials[2 * blockIdx.x] = r;
      partials[2 * blockIdx.x + 1] = c;
      __threadfence();
      const unsigned int prev = atomicAdd(counter, 1u);
      is_last = (prev == gridDim.x - 1);
    }
  }
  __syncthreads();
  if (is_last) {
    __threadfence();
    float r = 0.f, c = 0.f;
    for (int b = threadIdx.x; b < gridDim.x; b += kThreads) {
      r = fmaxf(r, __ldcg(partials + 2 * b));
      c = fmaxf(c, __ldcg(partials + 2 * b + 1));
    }
    r = warp_max(r);
    c = warp_max(c);
    __syncthreads();
    if (lane == 0) {
      red[warp] = r;
      red[kWarps + warp] = c;
    }
    __syncthreads();
    if (warp == 0) {
      float rr = lane < kWarps ? red[lane] : 0.f;
      float cc = lane < kWarps ? red[kWarps + lane] : 0.f;
      rr = warp_max(rr);
      cc = warp_max(cc);
      if (lane == 0) {
        amax_out[0] = rr;
        amax_out[1] = cc;
        *counter = 0u;
      }
    }
  }
}

__device__ __forceinline__ void finish_amax(uint32_t row_packed, uint32_t col_packed,
                                            float *__restrict__ amax_out,
                                            float *__restrict__ partials,
                                            unsigned int *__restrict__ counter) {
  finish_amax_f(unpack_max(row_packed), col_packed, amax_out, partials, counter);
}

// bf16x2 + bf16x2 in fp32, rounded to nearest even (torch's bf16 add: float(a) + float(b)).
__device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("{\n"
      ".reg .b32 a0, a1, b0, b1;\n"
      "shl.b32 a0, %1, 16;\n"
      "and.b32 a1, %1, 0xffff0000;\n"
      "shl.b32 b0, %2, 16;\n"
      "and.b32 b1, %2, 0xffff0000;\n"
      "add.rn.f32 a0, a0, b0;\n"
      "add.rn.f32 a1, a1, b1;\n"
      "cvt.rn.bf16x2.f32 %0, a1, a0;\n"
      "}\n"
      : "=r"(r)
      : "r"(a), "r"(b));
  return r;
}

// ------------------------------------------------------------------------------------------------
// out[M, N] = a + b, amax(|out|), amax(|RHT_t(out)|). One 64 x 128 tile per CTA (256 threads,
// 8 x 16 B loads per thread issued up front), 16 KiB of shared memory for the MMA only. Like
// torch's elementwise add, CTAs are short and light, so the NCCL kernels launched on the side
// streams find SM resources at once. Amaxes: atomicMax into a zeroed pair acc[0..1] (skipped when
// the CTA cannot raise it); the last CTA copies it out and re-zeroes it.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 ld_nc(const void *p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void finish_amax_atomic(float fr, float fc, float *__restrict__ amax_out,
                                                   float *__restrict__ acc,
                                                   unsigned int *__restrict__ counter) {
  __shared__ float red[2 * kWarps];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  fr = warp_max(fr);
  fc = warp_max(fc);
  if (lane == 0) {
    red[warp] = fr;
    red[kWarps + warp] = fc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float r = red[0], c = red[kWarps];
#pragma unroll
    for (int w = 1; w < kWarps; w++) {
      r = fmaxf(r, red[w]);
      c = fmaxf(c, red[kWarps + w]);
    }
    int *ia = reinterpret_cast<int *>(acc);
    if (__float_as_int(r) > __ldcg(ia)) atomicMax(ia, __float_as_int(r));
    if (__float_as_int(c) > __ldcg(ia + 1)) atomicMax(ia + 1, __float_as_int(c));
    __threadfence();
    const unsigned int prev = atomicAdd(counter, 1u);
    if (prev == gridDim.x - 1) {
      __threadfence();
      amax_out[0] = __int_as_float(atomicExch(ia, 0));
      amax_out[1] = __int_as_float(atomicExch(ia + 1, 0));
      *counter = 0u;
    }
  }
}

__global__ void __launch_bounds__(kThreads)
    add_rht_amax_reg_kernel(const __nv_bfloat16 *__restrict__ a,
                            const __nv_bfloat16 *__restrict__ b, __nv_bfloat16 *__restrict__ out,
                            float *__restrict__ amax_out, float *__restrict__ acc,
                            unsigned int *__restrict__ counter, int64_t n, int tiles_h,
                            uint16_t sign_mask_t) {
  __shared__ __align__(128) uint4 sbuf[2 * kBufChunks];  // two 64 x 64 halves
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int t = blockIdx.x;
  const int64_t base =
      static_cast<int64_t>(t / tiles_h) * kTileT * n + static_cast<int64_t>(t % tiles_h) * 128;
  // Thread -> chunk tid % 16 (16 x 8 columns) of rows tid / 16 + 16 r, r = 0..3.
  const int cc = tid % 16, r0 = tid / 16;
  uint4 av[4], bv[4];
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const int64_t o = base + static_cast<int64_t>(r0 + 16 * r) * n + cc * 8;
    av[r] = ld_nc(a + o);
    bv[r] = ld_nc(b + o);
  }
  uint32_t had_t[4];
  get_hadamard_t_fragment(had_t, sign_mask_t);
  uint32_t row_amax = 0;
  uint4 *half = sbuf + (cc / 8) * kBufChunks;
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const int row = r0 + 16 * r;
    uint4 sm;
    sm.x = add2(av[r].x, bv[r].x);
    sm.y = add2(av[r].y, bv[r].y);
    sm.z = add2(av[r].z, bv[r].z);
    sm.w = add2(av[r].w, bv[r].w);
    row_amax = bmax(row_amax, bmax4(sm));
    half[swz(row, cc % 8)] = sm;
    st_na(out + base + static_cast<int64_t>(row) * n + cc * 8, sm);
  }
  __syncthreads();
  const int buf = warp / 4, ws = warp % 4;
  const uint32_t sidx = swz(lane % 16, lane / 16 + ws * 2);
  const uint4 *sb = sbuf + buf * kBufChunks;
  uint32_t col_amax = 0;
#pragma unroll
  for (int rb = 0; rb < 4; rb++) {
    col_amax = bmax(col_amax, rht_t_amax16x16(&sb[rb * 16 * 8 + sidx], had_t));
  }
  finish_amax_atomic(unpack_max(row_amax), unpack_max(col_amax), amax_out, acc, counter);
}

// ------------------------------------------------------------------------------------------------
// RoPE backward of grad Q [T, G*HQ, D], grad K / V [T, G, D] (D = 128, T tokens in (s, b) order)
// into the QKV GEMM output gradient [T, G, (HQ + 2) * D], plus its amaxes. A tile is 64 tokens x
// one 128-column head of the output (two 64 x 64 buffers: the low and the high half of the head);
// tiles are token block major, head minor; with tiles_per_cta = hq + 2 a CTA takes the heads of
// one KV group of a token block and loads that block's cos/sin into registers once.
//   out[d <  64] = fma(g[d], cos[d], g[d + 64] * sin[d + 64])
//   out[d >= 64] = fma(g[d], cos[d], g[d - 64] * -sin[d - 64])      (V heads: out = g)
// ------------------------------------------------------------------------------------------------
// cos / sin of one position at columns c .. c + 7 (low half) and c + 64 .. c + 71 (high half).
struct Trig {
  float cl[8], sl[8], chh[8], sh[8];
  __device__ __forceinline__ void load(const float *cos_p, const float *sin_p, int c) {
    const float4 a0 = ldg_f4(cos_p + c), a1 = ldg_f4(cos_p + c + 4);
    const float4 b0 = ldg_f4(sin_p + c), b1 = ldg_f4(sin_p + c + 4);
    const float4 c0 = ldg_f4(cos_p + c + 64), c1 = ldg_f4(cos_p + c + 68);
    const float4 d0 = ldg_f4(sin_p + c + 64), d1 = ldg_f4(sin_p + c + 68);
    cl[0] = a0.x;
    cl[1] = a0.y;
    cl[2] = a0.z;
    cl[3] = a0.w;
    cl[4] = a1.x;
    cl[5] = a1.y;
    cl[6] = a1.z;
    cl[7] = a1.w;
    sl[0] = b0.x;
    sl[1] = b0.y;
    sl[2] = b0.z;
    sl[3] = b0.w;
    sl[4] = b1.x;
    sl[5] = b1.y;
    sl[6] = b1.z;
    sl[7] = b1.w;
    chh[0] = c0.x;
    chh[1] = c0.y;
    chh[2] = c0.z;
    chh[3] = c0.w;
    chh[4] = c1.x;
    chh[5] = c1.y;
    chh[6] = c1.z;
    chh[7] = c1.w;
    sh[0] = d0.x;
    sh[1] = d0.y;
    sh[2] = d0.z;
    sh[3] = d0.w;
    sh[4] = d1.x;
    sh[5] = d1.y;
    sh[6] = d1.z;
    sh[7] = d1.w;
  }
};

__device__ __forceinline__ void rope8(const uint4 lo, const uint4 hi, const Trig &tr, uint4 &olo,
                                      uint4 &ohi) {
  const float *cl = tr.cl, *sl = tr.sl, *chh = tr.chh, *sh = tr.sh;
  const uint32_t xl[4] = {lo.x, lo.y, lo.z, lo.w};
  const uint32_t xh[4] = {hi.x, hi.y, hi.z, hi.w};
  uint32_t ol[4], oh[4];
#pragma unroll
  for (int w = 0; w < 4; w++) {
    float l[2], h[2], rl[2], rh[2];
    l[0] = __uint_as_float(xl[w] << 16);
    l[1] = __uint_as_float(xl[w] & 0xffff0000u);
    h[0] = __uint_as_float(xh[w] << 16);
    h[1] = __uint_as_float(xh[w] & 0xffff0000u);
#pragma unroll
    for (int e = 0; e < 2; e++) {
      const int i = 2 * w + e;
      rl[e] = __fmaf_rn(l[e], cl[i], __fmul_rn(h[e], sh[i]));
      rh[e] = __fmaf_rn(h[e], chh[i], __fmul_rn(l[e], -sl[i]));
    }
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(ol[w]) : "f"(rl[1]), "f"(rl[0]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(oh[w]) : "f"(rh[1]), "f"(rh[0]));
  }
  olo = make_uint4(ol[0], ol[1], ol[2], ol[3]);
  ohi = make_uint4(oh[0], oh[1], oh[2], oh[3]);
}

// kPair: batch is even, so the thread's two rows (2k, 2k + 1) share one position, whose cos/sin
// stay in registers while the CTA walks the heads of a token block.
template <int kStages, bool kPair>
__global__ void __launch_bounds__(kThreads)
    qkv_rope_bwd_rht_amax_kernel(const __nv_bfloat16 *__restrict__ gq,
                                 const __nv_bfloat16 *__restrict__ gk,
                                 const __nv_bfloat16 *__restrict__ gv,
                                 const float *__restrict__ cosv, const float *__restrict__ sinv,
                                 __nv_bfloat16 *__restrict__ out, float *__restrict__ amax_out,
                                 float *__restrict__ partials, unsigned int *__restrict__ counter,
                                 int batch, int groups, int hq, int num_tiles, uint16_t sign_mask_t,
                                 int tiles_per_cta) {
  constexpr int D = 128;
  extern __shared__ __align__(128) uint4 ring[];  // [kStages][lo, hi][kBufChunks]
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int buf = warp / 4, ws = warp % 4;
  const uint32_t sidx = swz(lane % 16, lane / 16 + ws * 2);
  const int heads = groups * (hq + 2);
  const int64_t out_ld = static_cast<int64_t>(heads) * D;

  // Thread -> rows (2 * (tid / 8), +1) of the tile, 8-column chunk tid % 8 of each half.
  const int ch = tid % 8;
  const int r0 = 2 * (tid / 8);
  const uint32_t sw0 = swz(r0, ch), sw1 = swz(r0 + 1, ch);

  // (tile) -> source row pointer of token tok0 and its token stride (elements), output column.
  auto src_of = [&](int t, const __nv_bfloat16 *&src, int64_t &src_ld, int &kind, int64_t &ocol,
                    int64_t &tok0) {
    const int tb = t / heads, hh = t % heads;
    const int g = hh / (hq + 2), j = hh % (hq + 2);
    tok0 = static_cast<int64_t>(tb) * kTileT;
    if (j < hq) {
      src_ld = static_cast<int64_t>(groups) * hq * D;
      src = gq + tok0 * src_ld + static_cast<int64_t>(g * hq + j) * D;
      kind = 0;
    } else {
      src_ld = static_cast<int64_t>(groups) * D;
      src = (j == hq ? gk : gv) + tok0 * src_ld + static_cast<int64_t>(g) * D;
      kind = j == hq ? 0 : 1;
    }
    ocol = static_cast<int64_t>(hh) * D;
  };
  auto issue = [&](int t, int st) {
    const __nv_bfloat16 *src;
    int64_t src_ld, ocol, tok0;
    int kind;
    src_of(t, src, src_ld, kind, ocol, tok0);
    uint4 *s_lo = ring + (st * 2) * kBufChunks;
    uint4 *s_hi = s_lo + kBufChunks;
    const __nv_bfloat16 *p0 = src + static_cast<int64_t>(r0) * src_ld + ch * 8;
    const __nv_bfloat16 *p1 = p0 + src_ld;
    cp_async16(s_lo + sw0, p0);
    cp_async16(s_hi + sw0, p0 + 64);
    cp_async16(s_lo + sw1, p1);
    cp_async16(s_hi + sw1, p1 + 64);
  };

  uint32_t had_t[4];
  get_hadamard_t_fragment(had_t, sign_mask_t);

  // This CTA's tiles: [first, first + count).
  const int first = blockIdx.x * tiles_per_cta;
  const int count = min(tiles_per_cta, num_tiles - first);
#pragma unroll
  for (int p = 0; p < kStages - 1; p++) {
    if (p < count) issue(first + p, p);
    cp_async_commit();
  }

  uint32_t row_amax = 0, col_amax = 0;
  int st = 0;
  int64_t cached_tok0 = -1;
  Trig tr0;
  for (int k = 0; k < count; k++) {
    cp_async_wait<kStages - 2>();
    const int tile = first + k;
    uint4 *s_lo = ring + (st * 2) * kBufChunks;
    uint4 *s_hi = s_lo + kBufChunks;
    const __nv_bfloat16 *src;
    int64_t src_ld, ocol, tok0;
    int kind;
    src_of(tile, src, src_ld, kind, ocol, tok0);
    uint4 lo0 = s_lo[sw0], hi0 = s_hi[sw0], lo1 = s_lo[sw1], hi1 = s_hi[sw1];
    if (kind == 0) {
      if (tok0 != cached_tok0) {
        const int64_t pos0 = (tok0 + r0) / batch;
        tr0.load(cosv + pos0 * D, sinv + pos0 * D, ch * 8);
        cached_tok0 = tok0;
      }
      uint4 a, b2;
      rope8(lo0, hi0, tr0, a, b2);
      lo0 = a;
      hi0 = b2;
      if constexpr (kPair) {
        rope8(lo1, hi1, tr0, a, b2);
      } else {
        const int64_t pos1 = (tok0 + r0 + 1) / batch;
        Trig tr1;
        tr1.load(cosv + pos1 * D, sinv + pos1 * D, ch * 8);
        rope8(lo1, hi1, tr1, a, b2);
      }
      lo1 = a;
      hi1 = b2;
      s_lo[sw0] = lo0;
      s_hi[sw0] = hi0;
      s_lo[sw1] = lo1;
      s_hi[sw1] = hi1;
    }
    row_amax = bmax(row_amax, bmax(bmax(bmax4(lo0), bmax4(hi0)), bmax(bmax4(lo1), bmax4(hi1))));
    __nv_bfloat16 *o0 = out + (tok0 + r0) * out_ld + ocol + ch * 8;
    __nv_bfloat16 *o1 = o0 + out_ld;
    st_na(o0, lo0);
    st_na(o0 + 64, hi0);
    st_na(o1, lo1);
    st_na(o1 + 64, hi1);
    const int nst = st == 0 ? kStages - 1 : st - 1;
    __syncthreads();
    const uint4 *sbuf = buf == 0 ? s_lo : s_hi;
#pragma unroll
    for (int rb = 0; rb < 4; rb++) {
      col_amax = bmax(col_amax, rht_t_amax16x16(&sbuf[rb * 16 * 8 + sidx], had_t));
    }
    if (k + kStages - 1 < count) issue(first + k + kStages - 1, nst);
    cp_async_commit();
    st = st + 1 == kStages ? 0 : st + 1;
  }
  cp_async_wait<0>();
  finish_amax(row_amax, col_amax, amax_out, partials, counter);
}

// ------------------------------------------------------------------------------------------------
// Row-wise NVFP4 cast (round to nearest, no RHT) of x[M, N] with the given row-wise amax, plus the
// amaxes of x (pre-RHT and transposed RHT) for its later column-wise RHT cast. The cast arithmetic
// is that of the row path of row_col_rht_gemm_device (row_cast_col_hadamard_transform_cast_fusion.cu,
// core_nvfp4.cuh): per tensor enc = fminf(6 * 448 / amax, FLT_MAX) (1 if amax == 0 or enc == 0),
// mult = enc * (1/6), dec = 1 / enc; per 16-value block code = e4m3_rn_satfinite(amax16 * mult),
// scale = min_nan_propagating(1 / (fp8(code) * dec), FLT_MAX) (from a 256-entry table per CTA),
// values e2m1_rn_satfinite(x * scale) with an IEEE (non-flushing) multiply, as that row path.
// Scales in TE's GEMM swizzled layout (128 x 4 blocks of 512 B) or plain [M, N / 16]. Bitwise equal
// to it for every finite input (incl. denormals); a 16-value block holding a NaN gets different codes
// than it (a NaN activation has already broken the step either way).
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void global_scales(float amax, float &dec, float &mult) {
  float e = __fdiv_rn(6.0f * 448.0f, amax);
  e = fminf(e, 3.402823466e38f);
  if (amax == 0.0f || e == 0.0f) e = 1.0f;
  dec = __fdiv_rn(1.0f, e);
  mult = __fmul_rn(e, 1.0f / 6.0f);
}

__device__ __forceinline__ float lut_entry(uint32_t code, float dec) {
  float r;
  asm("{\n"
      ".reg .f32 u, q;\n"
      ".reg .b16 h, ulo, uhi;\n"
      ".reg .b32 uu;\n"
      "cvt.u16.u32 h, %1;\n"
      "cvt.rn.f16x2.e4m3x2 uu, h;\n"
      "mov.b32 {ulo, uhi}, uu;\n"
      "cvt.f32.f16 u, ulo;\n"
      "mul.ftz.f32 q, u, %2;\n"
      "div.rn.f32 %0, 0f3F800000, q;\n"
      "}\n"
      : "=f"(r)
      : "r"(code), "f"(dec));
  return isnan(r) ? r : fminf(r, 3.402823466e38f);
}

__device__ __forceinline__ uint32_t sf_code(float amax16, float mult) {
  uint32_t c;
  asm("{\n"
      ".reg .f32 p;\n"
      ".reg .b16 h;\n"
      "mul.ftz.f32 p, %1, %2;\n"
      "cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, p;\n"
      "cvt.u32.u16 %0, h;\n"
      "}\n"
      : "=r"(c)
      : "f"(amax16), "f"(mult));
  return c & 0xffu;
}

// 8 bf16x2 words -> 16 e2m1 codes (two 32-bit words, element 0 in the low nibble).
__device__ __forceinline__ uint2 quant16_rn(const uint32_t *w, float s) {
  uint2 r;
  asm("{\n"
      ".reg .b32 f<16>;\n"
      ".reg .b8 b<8>;\n"
      "shl.b32 f0, %2, 16;\n and.b32 f1, %2, 0xffff0000;\n"
      "shl.b32 f2, %3, 16;\n and.b32 f3, %3, 0xffff0000;\n"
      "shl.b32 f4, %4, 16;\n and.b32 f5, %4, 0xffff0000;\n"
      "shl.b32 f6, %5, 16;\n and.b32 f7, %5, 0xffff0000;\n"
      "shl.b32 f8, %6, 16;\n and.b32 f9, %6, 0xffff0000;\n"
      "shl.b32 f10, %7, 16;\n and.b32 f11, %7, 0xffff0000;\n"
      "shl.b32 f12, %8, 16;\n and.b32 f13, %8, 0xffff0000;\n"
      "shl.b32 f14, %9, 16;\n and.b32 f15, %9, 0xffff0000;\n"
      "mul.rn.f32 f0, f0, %10;\n mul.rn.f32 f1, f1, %10;\n"
      "mul.rn.f32 f2, f2, %10;\n mul.rn.f32 f3, f3, %10;\n"
      "mul.rn.f32 f4, f4, %10;\n mul.rn.f32 f5, f5, %10;\n"
      "mul.rn.f32 f6, f6, %10;\n mul.rn.f32 f7, f7, %10;\n"
      "mul.rn.f32 f8, f8, %10;\n mul.rn.f32 f9, f9, %10;\n"
      "mul.rn.f32 f10, f10, %10;\n mul.rn.f32 f11, f11, %10;\n"
      "mul.rn.f32 f12, f12, %10;\n mul.rn.f32 f13, f13, %10;\n"
      "mul.rn.f32 f14, f14, %10;\n mul.rn.f32 f15, f15, %10;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, f1, f0;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, f3, f2;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, f5, f4;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, f7, f6;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b4, f9, f8;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b5, f11, f10;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b6, f13, f12;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b7, f15, f14;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "mov.b32 %1, {b4, b5, b6, b7};\n"
      "}\n"
      : "=r"(r.x), "=r"(r.y)
      : "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]),
        "f"(s));
  return r;
}

// |max| of 16 bf16 values as fp32 (exact).
__device__ __forceinline__ float amax16_f32(const uint32_t *w) {
  uint32_t m =
      bmax(bmax(bmax(w[0], w[1]), bmax(w[2], w[3])), bmax(bmax(w[4], w[5]), bmax(w[6], w[7])));
  float lo = __uint_as_float(m << 16), hi = __uint_as_float(m & 0xffff0000u);
  return fmaxf(fabsf(lo), fabsf(hi));
}

template <int kStages, bool kSwizzled>
__global__ void __launch_bounds__(kThreads)
    row_cast_rht_amax_kernel(const __nv_bfloat16 *__restrict__ x, const float *__restrict__ amax_in,
                             uint8_t *__restrict__ qdata, uint8_t *__restrict__ qsf,
                             float *__restrict__ amax_out, float *__restrict__ partials,
                             unsigned int *__restrict__ counter, int64_t n, int tiles_h,
                             int num_tiles, uint16_t sign_mask_t, int tiles_h_log2,
                             int tiles_per_cta) {
  // Tile: 64 tokens x 128 columns = two swizzled 64 x 64 buffers.
  extern __shared__ __align__(128) uint4 ring[];  // [kStages][2][kBufChunks]
  __shared__ float lut[256];
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int buf = warp / 4, ws = warp % 4;
  const uint32_t sidx = swz(lane % 16, lane / 16 + ws * 2);

  float dec, mult;
  global_scales(__ldg(amax_in), dec, mult);
  lut[tid] = lut_entry(static_cast<uint32_t>(tid), dec);

  // Thread -> token row tid / 4, 16-value block tid % 4 of each 64-column half.
  const int row = tid / 4, blk = tid % 4;
  const uint32_t sw0 = swz(row, 2 * blk), sw1 = swz(row, 2 * blk + 1);
  const uint32_t off = static_cast<uint32_t>(row * n + blk * 16);
  // tiles_h is a power of two (2^tiles_h_log2)
  auto tile_base = [&](int t) -> int64_t {
    return static_cast<int64_t>(t >> tiles_h_log2) * kTileT * n +
           static_cast<int64_t>(t & (tiles_h - 1)) * 128;
  };
  auto issue = [&](int t, int st) {
    const __nv_bfloat16 *p = x + tile_base(t) + off;
    uint4 *s = ring + st * 2 * kBufChunks;
    cp_async16(s + sw0, p);
    cp_async16(s + sw1, p + 8);
    cp_async16(s + kBufChunks + sw0, p + 64);
    cp_async16(s + kBufChunks + sw1, p + 72);
  };

  uint32_t had_t[4];
  get_hadamard_t_fragment(had_t, sign_mask_t);

  __syncthreads();  // lut
  // This CTA's tiles: [first, first + count).
  const int first = blockIdx.x * tiles_per_cta;
  const int count = min(tiles_per_cta, num_tiles - first);
#pragma unroll
  for (int p = 0; p < kStages - 1; p++) {
    if (p < count) issue(first + p, p);
    cp_async_commit();
  }

  const int64_t sf_ld = n / 16;
  float row_amax = 0.f;
  uint32_t col_amax = 0;
  int st = 0;
  for (int k = 0; k < count; k++) {
    cp_async_wait<kStages - 2>();
    const int tile = first + k;
    const uint4 *s = ring + st * 2 * kBufChunks;
    const int64_t token = static_cast<int64_t>(tile >> tiles_h_log2) * kTileT + row;
    const int64_t col0 = static_cast<int64_t>(tile & (tiles_h - 1)) * 128;
    uint2 q[2];
    uint32_t word[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const uint4 v0 = s[h * kBufChunks + sw0], v1 = s[h * kBufChunks + sw1];
      const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
      const float a16 = amax16_f32(w);
      row_amax = fmaxf(row_amax, a16);
      const uint32_t code = sf_code(a16, mult);
      q[h] = quant16_rn(w, lut[code]);
      word[h] = code << (8 * blk);
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int64_t col = col0 + h * 64 + blk * 16;
      *reinterpret_cast<uint2 *>(qdata + token * (n / 2) + col / 2) = q[h];
      // the 4 codes of this token's 64 columns (lanes 4r .. 4r + 3) -> one 32-bit word
      uint32_t wd = word[h];
      wd |= __shfl_xor_sync(0xffffffffu, wd, 1);
      wd |= __shfl_xor_sync(0xffffffffu, wd, 2);
      if (blk == 0) {
        int64_t a;
        const int64_t c64 = (col0 + h * 64) / 64;
        if constexpr (kSwizzled) {
          a = ((token / 128) * (n / 64) + c64) * 512 + (token % 32) * 16 + ((token % 128) / 32) * 4;
        } else {
          a = token * sf_ld + c64 * 4;
        }
        *reinterpret_cast<uint32_t *>(qsf + a) = wd;
      }
    }
    const int nst = st == 0 ? kStages - 1 : st - 1;
    __syncthreads();
    const uint4 *sbuf = s + buf * kBufChunks;
#pragma unroll
    for (int rb = 0; rb < 4; rb++) {
      col_amax = bmax(col_amax, rht_t_amax16x16(&sbuf[rb * 16 * 8 + sidx], had_t));
    }
    if (k + kStages - 1 < count) issue(first + k + kStages - 1, nst);
    cp_async_commit();
    st = st + 1 == kStages ? 0 : st + 1;
  }
  cp_async_wait<0>();
  finish_amax_f(row_amax, col_amax, amax_out, partials, counter);
}

// Workspace: the arrival counter, then 2 floats per CTA (the last pair is the atomicMax
// accumulator of the add kernel).
constexpr size_t kCounterBytes = 16;
constexpr int kMaxGrid = 65535;
constexpr size_t kWorkspaceBytes = kCounterBytes + 2 * sizeof(float) * (kMaxGrid + 1);
constexpr int kStages = 3;           // cp.async ring depth of the RoPE and cast kernels
constexpr int kCastTilesPerCta = 8;  // 64 x 128 tiles per CTA of the cast kernel

// Returns true (and sets the workspace shape) when the call only queries the workspace size.
bool query_workspace(Tensor *workspace) {
  if (workspace->data.dptr == nullptr) {
    workspace->data.shape = {kWorkspaceBytes};
    workspace->data.dtype = DType::kByte;
    return true;
  }
  NVTE_CHECK(workspace->data.dtype == DType::kByte, "Workspace must be a byte tensor.");
  NVTE_CHECK(workspace->data.numel() >= kWorkspaceBytes, "Workspace is too small (expected ",
             kWorkspaceBytes, " bytes, got ", workspace->data.numel(), ").");
  return false;
}

unsigned int *workspace_counter(const Tensor &workspace) {
  return reinterpret_cast<unsigned int *>(workspace.data.dptr);
}

float *workspace_partials(const Tensor &workspace) {
  return reinterpret_cast<float *>(reinterpret_cast<unsigned char *>(workspace.data.dptr) +
                                   kCounterBytes);
}

void check_bf16(const Tensor &t, const char *name) {
  NVTE_CHECK(t.data.dptr != nullptr, name, " is not allocated.");
  NVTE_CHECK(t.data.dtype == DType::kBFloat16, name, " must be a BF16 tensor, but dtype is ",
             to_string(t.data.dtype), ".");
  NVTE_CHECK(reinterpret_cast<uintptr_t>(t.data.dptr) % 16 == 0, name, " must be 16 B aligned.");
}

void check_amax(const Tensor &amax) {
  NVTE_CHECK(
      amax.data.dptr != nullptr && amax.data.dtype == DType::kFloat32 && amax.data.numel() >= 2,
      "amax must be an FP32 tensor with 2 elements.");
}

void check_grid(int64_t grid) {
  NVTE_CHECK(grid > 0 && grid <= kMaxGrid, "Tensor too large for the workspace (", grid,
             " CTAs, at most ", kMaxGrid, ").");
}

template <typename K>
void set_smem(K kernel, size_t bytes) {
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace hadamard_amax_producers

void add_hadamard_transform_amax(const Tensor &a, const Tensor &b, Tensor *out, Tensor *amax,
                                 Tensor *workspace, const uint16_t random_sign_mask_t,
                                 cudaStream_t stream) {
  using namespace hadamard_amax_producers;
  if (query_workspace(workspace)) {
    return;
  }
  check_bf16(a, "a");
  check_bf16(b, "b");
  check_bf16(*out, "out");
  check_amax(*amax);
  NVTE_CHECK(a.data.shape == b.data.shape && a.data.shape == out->data.shape,
             "a, b and out must have the same shape.");
  const auto [m, n] = a.flat_2d_dims();
  NVTE_CHECK(m % kTileT == 0 && n % 128 == 0, "The number of rows (", m,
             ") must be a multiple of 64 and the number of columns (", n, ") a multiple of 128.");
  NVTE_CHECK(64 * n < (size_t{1} << 31), "Too many columns.");
  NVTE_CHECK(cuda::sm_arch(cuda::current_device()) >= 100,
             "add_hadamard_transform_amax requires SM 10.0+.");
  if (m == 0) {
    return;
  }
  const int64_t grid = static_cast<int64_t>(m / kTileT) * static_cast<int64_t>(n / 128);
  check_grid(grid);
  float *partials = workspace_partials(*workspace);
  add_rht_amax_reg_kernel<<<static_cast<int>(grid), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(a.data.dptr),
      reinterpret_cast<const __nv_bfloat16 *>(b.data.dptr),
      reinterpret_cast<__nv_bfloat16 *>(out->data.dptr), reinterpret_cast<float *>(amax->data.dptr),
      partials + 2 * kMaxGrid, workspace_counter(*workspace), static_cast<int64_t>(n),
      static_cast<int>(n / 128), random_sign_mask_t);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_qkv_rope_backward_hadamard_transform_amax(const Tensor &grad_q, const Tensor &grad_k,
                                                     const Tensor &grad_v, const Tensor &cos,
                                                     const Tensor &sin, Tensor *grad_qkv,
                                                     Tensor *amax, Tensor *workspace,
                                                     const uint16_t random_sign_mask_t,
                                                     cudaStream_t stream) {
  using namespace hadamard_amax_producers;
  if (query_workspace(workspace)) {
    return;
  }
  constexpr size_t D = 128;
  check_bf16(grad_q, "grad_q");
  check_bf16(grad_k, "grad_k");
  check_bf16(grad_v, "grad_v");
  check_bf16(*grad_qkv, "grad_qkv");
  check_amax(*amax);
  const auto &qs = grad_q.data.shape;
  const auto &ks = grad_k.data.shape;
  NVTE_CHECK(qs.size() == 4 && ks.size() == 4 && grad_v.data.shape == ks,
             "grad_q, grad_k and grad_v must be [s, b, h, d] with equal K and V shapes.");
  NVTE_CHECK(qs[3] == D && ks[3] == D, "Only a head dimension of 128 is supported.");
  NVTE_CHECK(qs[0] == ks[0] && qs[1] == ks[1], "Q and K/V must have the same s and b.");
  const size_t groups = ks[2];
  NVTE_CHECK(groups > 0 && qs[2] % groups == 0, "Q heads must be a multiple of the KV heads.");
  const size_t hq = qs[2] / groups;
  const size_t batch = qs[1];
  const size_t n_tok = qs[0] * batch;
  NVTE_CHECK(n_tok % kTileT == 0 && batch > 0 && kTileT % batch == 0,
             "s * b must be a multiple of 64 and b must divide 64.");
  NVTE_CHECK(grad_qkv->data.numel() == n_tok * groups * (hq + 2) * D,
             "grad_qkv must have s * b * h_kv * (h_q / h_kv + 2) * d elements.");
  for (const Tensor *t : {&cos, &sin}) {
    NVTE_CHECK(
        t->data.dptr != nullptr && t->data.dtype == DType::kFloat32 && t->data.numel() >= qs[0] * D,
        "cos and sin must be FP32 tables with d values for each of the s positions.");
    NVTE_CHECK(reinterpret_cast<uintptr_t>(t->data.dptr) % 16 == 0,
               "cos and sin must be 16 B aligned.");
  }
  NVTE_CHECK(cuda::sm_arch(cuda::current_device()) >= 100,
             "fused_qkv_rope_backward_hadamard_transform_amax requires SM 10.0+.");
  if (n_tok == 0) {
    return;
  }
  // A CTA takes the heads of two KV groups of a 64-token block
  const int heads = static_cast<int>(groups * (hq + 2));
  const int64_t num_tiles = static_cast<int64_t>(n_tok / kTileT) * heads;
  const int tiles_per_cta = static_cast<int>(2 * (hq + 2));
  const int64_t grid = (num_tiles + tiles_per_cta - 1) / tiles_per_cta;
  check_grid(grid);
  const size_t smem = static_cast<size_t>(kStages) * 2 * kBufChunks * sizeof(uint4);
  const bool pair = batch % 2 == 0;
  auto kernel = pair ? qkv_rope_bwd_rht_amax_kernel<kStages, true>
                     : qkv_rope_bwd_rht_amax_kernel<kStages, false>;
  set_smem(kernel, smem);
  kernel<<<static_cast<int>(grid), kThreads, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(grad_q.data.dptr),
      reinterpret_cast<const __nv_bfloat16 *>(grad_k.data.dptr),
      reinterpret_cast<const __nv_bfloat16 *>(grad_v.data.dptr),
      reinterpret_cast<const float *>(cos.data.dptr),
      reinterpret_cast<const float *>(sin.data.dptr),
      reinterpret_cast<__nv_bfloat16 *>(grad_qkv->data.dptr),
      reinterpret_cast<float *>(amax->data.dptr), workspace_partials(*workspace),
      workspace_counter(*workspace), static_cast<int>(batch), static_cast<int>(groups),
      static_cast<int>(hq), static_cast<int>(num_tiles), random_sign_mask_t, tiles_per_cta);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void quantize_rowwise_with_amax_hadamard_transform_amax(const Tensor &input, Tensor *output,
                                                        Tensor *amax, Tensor *workspace,
                                                        const uint16_t random_sign_mask_t,
                                                        cudaStream_t stream) {
  using namespace hadamard_amax_producers;
  if (query_workspace(workspace)) {
    return;
  }
  check_bf16(input, "input");
  check_amax(*amax);
  NVTE_CHECK(output->scaling_mode == NVTE_NVFP4_1D_SCALING, "Output must be an NVFP4 tensor.");
  NVTE_CHECK(output->has_data() && !output->has_columnwise_data(),
             "Output must hold row-wise data only.");
  NVTE_CHECK(output->amax.dptr != nullptr && output->amax.dtype == DType::kFloat32,
             "Output must hold the FP32 row-wise amax used for the cast.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Output scales are not allocated.");
  const auto [m, n] = input.flat_2d_dims();
  NVTE_CHECK(m % 128 == 0 && n % 128 == 0, "The dimensions (", m, ", ", n,
             ") must be multiples of 128.");
  const size_t tiles_h = n / 128;
  NVTE_CHECK((tiles_h & (tiles_h - 1)) == 0, "The number of columns divided by 128 (", tiles_h,
             ") must be a power of two.");
  NVTE_CHECK(64 * n < (size_t{1} << 31), "Too many columns.");
  NVTE_CHECK(output->data.numel() * typeToNumBits(output->data.dtype) >= m * n * 4,
             "Output row-wise data is too small.");
  NVTE_CHECK(output->scale_inv.numel() * typeToNumBits(output->scale_inv.dtype) >= m * n / 2,
             "Output row-wise scales are too small.");
  NVTE_CHECK(cuda::sm_arch(cuda::current_device()) >= 100,
             "quantize_rowwise_with_amax_hadamard_transform_amax requires SM 10.0+.");
  if (m == 0) {
    return;
  }
  int lg = 0;
  while ((size_t{1} << lg) < tiles_h) lg++;
  const int64_t num_tiles = static_cast<int64_t>(tiles_h) * static_cast<int64_t>(m / kTileT);
  const int64_t grid = (num_tiles + kCastTilesPerCta - 1) / kCastTilesPerCta;
  check_grid(grid);
  const size_t smem = static_cast<size_t>(kStages) * 2 * kBufChunks * sizeof(uint4);
  const bool swizzled = output->with_gemm_swizzled_scales;
  auto kernel =
      swizzled ? row_cast_rht_amax_kernel<kStages, true> : row_cast_rht_amax_kernel<kStages, false>;
  set_smem(kernel, smem);
  kernel<<<static_cast<int>(grid), kThreads, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data.dptr),
      reinterpret_cast<const float *>(output->amax.dptr),
      reinterpret_cast<uint8_t *>(output->data.dptr),
      reinterpret_cast<uint8_t *>(output->scale_inv.dptr),
      reinterpret_cast<float *>(amax->data.dptr), workspace_partials(*workspace),
      workspace_counter(*workspace), static_cast<int64_t>(n), static_cast<int>(tiles_h),
      static_cast<int>(num_tiles), random_sign_mask_t, lg, kCastTilesPerCta);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine

void nvte_add_hadamard_transform_amax(const NVTETensor a, const NVTETensor b, NVTETensor output,
                                      NVTETensor amax, NVTETensor workspace, int random_sign_mask_t,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_add_hadamard_transform_amax);
  using namespace transformer_engine;
  add_hadamard_transform_amax(*convertNVTETensorCheck(a), *convertNVTETensorCheck(b),
                              convertNVTETensorCheck(output), convertNVTETensorCheck(amax),
                              convertNVTETensorCheck(workspace),
                              static_cast<uint16_t>(random_sign_mask_t), stream);
}

void nvte_fused_qkv_rope_backward_hadamard_transform_amax(
    const NVTETensor grad_q, const NVTETensor grad_k, const NVTETensor grad_v, const NVTETensor cos,
    const NVTETensor sin, NVTETensor grad_qkv, NVTETensor amax, NVTETensor workspace,
    int random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_qkv_rope_backward_hadamard_transform_amax);
  using namespace transformer_engine;
  fused_qkv_rope_backward_hadamard_transform_amax(
      *convertNVTETensorCheck(grad_q), *convertNVTETensorCheck(grad_k),
      *convertNVTETensorCheck(grad_v), *convertNVTETensorCheck(cos), *convertNVTETensorCheck(sin),
      convertNVTETensorCheck(grad_qkv), convertNVTETensorCheck(amax),
      convertNVTETensorCheck(workspace), static_cast<uint16_t>(random_sign_mask_t), stream);
}

void nvte_quantize_rowwise_with_amax_hadamard_transform_amax(const NVTETensor input,
                                                             NVTETensor output, NVTETensor amax,
                                                             NVTETensor workspace,
                                                             int random_sign_mask_t,
                                                             cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_rowwise_with_amax_hadamard_transform_amax);
  using namespace transformer_engine;
  quantize_rowwise_with_amax_hadamard_transform_amax(
      *convertNVTETensorCheck(input), convertNVTETensorCheck(output), convertNVTETensorCheck(amax),
      convertNVTETensorCheck(workspace), static_cast<uint16_t>(random_sign_mask_t), stream);
}
