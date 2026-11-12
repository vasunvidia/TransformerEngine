/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file rmsnorm_bwd_single_sweep.cu
 *  \brief Opt-in single-sweep RMSNorm backward (NVTE_RMSNORM_BWD_SINGLE_SWEEP=1).
 *
 *  One kernel computes dx and dgamma, reading dz and x from global memory exactly once,
 *  instead of the default backward kernel plus the dgamma finalize kernel:
 *
 *   - One CTA per 8 rows, a thread-block cluster of 4 CTAs per 32-row block. The CTA's
 *     8 x H bf16 slabs of dz and x are loaded by two TMA bulk copies (one mbarrier) and stay
 *     in shared memory for the whole kernel. Thread t owns columns 8t..8t+7 (H <= 3072).
 *   - Sweep 1 (shared memory): per row, k = sum_j(dz * g * x) * rsigma^3 / H, reduced in a
 *     fixed order (warp butterfly, then the 12 warp sums in order).
 *   - Sweep 2 (shared memory): dx = rsigma * (dz * g) - x * k, rounded to bf16 and stored;
 *     the per-column dgamma partial sum_r(dz * x * rsigma) over the CTA's 8 rows stays in
 *     registers.
 *   - The 4 CTAs of a cluster exchange their dgamma partials through distributed shared memory:
 *     CTA q of the cluster sums column slice q over the 4 CTAs (in rank order) and writes one
 *     fp32 partial row per 32-row block into the workspace.
 *   - The column reduction over the row blocks is finished in the same kernel: per-CTA
 *     "last arrival" tickets drive a 3-level tree (>= 16 row blocks -> level-1 group partial,
 *     <= 8 groups -> level-2 partial, the last CTA -> dgamma in bf16). Every level is a sum in a
 *     fixed order by the single last-arriving CTA, so dgamma is deterministic and independent
 *     of CTA scheduling. The last arrival of every ticket resets it to zero, so the tickets are
 *     zero again when the kernel exits.
 *
 *  Ticket storage and CUDA graphs: the tickets live in a per-device ring of slots allocated
 *  once (outside stream capture) and kept zero by the kernel itself, so a launch needs no
 *  memset and a captured launch replays without one. Consecutive launches rotate through the
 *  ring, so kernels running concurrently on different streams do not share a slot. If the ring
 *  has not been allocated yet when the first launch on a device happens inside a stream capture
 *  (cudaMalloc is not allowed there), the tickets are taken from the workspace instead and
 *  zeroed by a captured cudaMemsetAsync.
 *
 *  Numerics: dx and dgamma are computed in fp32 and rounded to bf16 once. The arithmetic order
 *  differs from the default kernels (dx: rsigma * dz * g - x * k instead of
 *  rsigma * (dz * g - mean(dz * g * y) * y); different row-reduction trees; a different
 *  dgamma summation order), so results are not bitwise identical to the default
 *  Transformer Engine or cuDNN backends but agree to within bf16 rounding. Both outputs are
 *  bitwise reproducible from run to run.
 *
 *  Residual gradient (rmsnorm_bwd_add): with an `add` tensor, the kernel adds it to dx in the
 *  final store, dx = bf16(bf16(dx) + add), with the sum formed in fp32 and rounded once. That is
 *  exactly what the single-sweep rmsnorm_bwd followed by a separate BF16 addition (PyTorch's
 *  autograd gradient accumulation, `dx + add`) produces, bit for bit, so folding a residual
 *  gradient into the normalization backward removes the addition kernel without changing the
 *  result. (The regular rmsnorm_bwd_add kernels instead add in fp32 before the single rounding
 *  of dx.) `add` is read with one 16-byte load per thread and row, next to the dx store.
 *
 *  Fused backward (nvte_rmsnorm_bwd_fused, independent of NVTE_RMSNORM_BWD_SINGLE_SWEEP):
 *   - `dz2`: a second incoming gradient (the normalized output feeding two consumers, e.g. a
 *     MoE router and the experts). The kernel forms dz = bf16(dz + dz2) (one fp32 sum, rounded
 *     once) before the backward, bitwise what autograd's gradient accumulation of the two would
 *     hand a separate backward. dz2 is read with one 16-byte load per thread and row, issued
 *     before the wait for the dz/x slabs.
 *   - `dx_q`: the final dx (after the residual gradient) also quantized to MXFP8, row-wise and
 *     column-wise (E4M3 or E5M2, compact or GEMM-swizzled scales, padding zeroed). The values
 *     and scales are those of the MXFP8 quantizer applied to the bf16 dx: per 32-element block
 *     amax -> E8M0 scale (float_to_e8m0(amax / fp8_max)) -> cvt.rn.satfinite(x * 2^-scale).
 *     The row-wise quantization runs in the dx store loop (a 32-column block is 4 adjacent threads,
 *     shuffle max); a column-wise block is the 32 rows of the cluster, whose 4 per-CTA column
 *     maxima are pushed into every peer's shared memory ahead of one split cluster barrier.
 *  With a residual gradient, its 8-row slab is prefetched into L2 (one bulk prefetch, issued with
 *  the slab loads), so the loads at the dx store hit L2.
 *     This lets the consumer of dx (e.g. the GEMMs of the preceding Linear) skip its own
 *     quantization pass over dx.
 *
 *  Origin: the single-sweep cluster kernel of the gpt-oss-20b "plus" workload
 *  (rmsnorm_bwd_cuda.cu, rmsnorm_bwd_single_sweep_kernel<8, HAS_RES, false, false>, NVIDIA
 *  first-party), without the MXFP8 gradient emission and dbias, dgamma written to the output
 *  tensor (not accumulated into main_grad).
 */

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "../../cast/mxfp8/swizzle.cuh"
#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx_arch_spec.cuh"
#include "../../util/system.h"
#include "../../utils.cuh"
#include "../common.h"

namespace transformer_engine {
namespace normalization {
namespace {

namespace cg = cooperative_groups;

constexpr int kThreads = 384;         // 12 warps; one thread per 8-column group
constexpr int kWarps = kThreads / 32;
constexpr int kMaxCols = kThreads * 8;  // 3072
constexpr int kRows = 8;              // rows per CTA
constexpr int kBlockRows = 32;        // rows per cluster (one dgamma partial row)
constexpr int kCluster = kBlockRows / kRows;  // CTAs per cluster
constexpr int kGroupMin = 16;         // minimum row blocks per level-1 group
constexpr int kMaxGroups = 64;        // level-1 groups
constexpr int kFan2 = 8;              // level-1 groups per level-2 group
constexpr int kMaxGroups2 = (kMaxGroups + kFan2 - 1) / kFan2;
// int32 per ticket slot: [final ticket | level-1 tickets | level-2 tickets]
constexpr int kTicketSlot = 1 + kMaxGroups + kMaxGroups2;
constexpr int kTicketRing = 64;       // slots per device
constexpr size_t kWorkspaceAlign = 128;

struct Params {
  const uint16_t *dz;     // [M, N] bf16
  const uint16_t *x;      // [M, N] bf16
  const float *rsigma;    // [M]
  const uint16_t *gamma;  // [N] bf16
  const uint16_t *add;    // [M, N] bf16 residual gradient added to dx, or nullptr
  uint16_t *dx;           // [M, N] bf16
  uint16_t *dgamma;       // [N] bf16
  float *part;            // [M / 32, N] fp32 partials of the 32-row blocks
  float *part2;           // [ngroups, N] fp32 level-1 partials
  float *part3;           // [ceil(ngroups / 8), N] fp32 level-2 partials
  int *tickets;           // [kTicketSlot], zero at rest
  int M, N, nb32, group, ngroups;
  int zero_centered;
  // Fused backward (nvte_rmsnorm_bwd_fused)
  const uint16_t *dz2;    // [M, N] bf16 second incoming gradient summed into dz, or nullptr
  uint8_t *q_row;         // [M, N] fp8 row-wise MXFP8 dx (only with the quantizing kernel)
  uint8_t *q_col;         // [M, N] fp8 column-wise MXFP8 dx
  uint8_t *s_row;         // e8m0 row-wise scales [M, s_row_cols]
  uint8_t *s_col;         // e8m0 column-wise scales [M / 32, s_col_cols]
  int s_row_cols, s_col_cols;  // padded scale columns (multiples of 4 and of 128)
  int swizzled;           // scales in GEMM-swizzled layout
};

// Row blocks per level-1 group and the number of level-1 groups (<= kMaxGroups).
struct Plan {
  int nb32, group, ngroups, ngroups2;
};
Plan make_plan(const size_t rows) {
  Plan p;
  p.nb32 = static_cast<int>(rows / kBlockRows);
  p.group = std::max(kGroupMin, DIVUP(p.nb32, kMaxGroups));
  p.ngroups = DIVUP(p.nb32, p.group);
  p.ngroups2 = DIVUP(p.ngroups, kFan2);
  return p;
}

size_t align_up(const size_t n) { return DIVUP(n, kWorkspaceAlign) * kWorkspaceAlign; }

// Shared memory: [dz slab 8 x N bf16 | x slab 8 x N bf16 |
//                 dgamma inbox kCluster x N / kCluster fp32 | misc]
constexpr size_t misc_bytes() {
  return 16 /*mbarrier*/ + kWarps * kRows * 4 /*warp row sums*/ + kRows * 4 /*k*/ +
         kRows * 4 /*rsigma*/ + 16 /*flag*/;
}
size_t smem_bytes(const int N) {
  return static_cast<size_t>(kRows) * N * 2 * 2 + static_cast<size_t>(N) * 4 + misc_bytes();
}

// ------------------------------------------------------------------------------- device helpers
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count)
               : "memory");
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t *bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_g2s(void *dst, const void *src, uint32_t bytes,
                                         uint64_t *bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void bulk_prefetch_l2(const void *src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ uint32_t mapa_u32(uint32_t smem_addr, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_dsmem_f4(uint32_t addr, float4 v) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(v.x),
               "f"(v.y), "f"(v.z), "f"(v.w)
               : "memory");
}
__device__ __forceinline__ float bf2f(uint32_t packed, int hi) {
  return __uint_as_float(hi ? (packed & 0xFFFF0000u) : (packed << 16));
}
__device__ __forceinline__ uint16_t f2bf(float f) {
  __nv_bfloat16 b = __float2bfloat16_rn(f);
  return *reinterpret_cast<uint16_t *>(&b);
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t *>(&h);
}
__device__ __forceinline__ float4 ld_cg_f4(const float *p) {
  return __ldcg(reinterpret_cast<const float4 *>(p));
}
// 8 bf16 sums a + b, each formed in fp32 and rounded once (= a BF16 tensor addition).
__device__ __forceinline__ uint4 add_bf16x8(const uint4 a, const uint4 b) {
  const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
  const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    o[i] = pack_bf16x2(__fadd_rn(bf2f(aw[i], 0), bf2f(bw[i], 0)),
                       __fadd_rn(bf2f(aw[i], 1), bf2f(bw[i], 1)));
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}

// One CTA sums `nrows` fp32 rows of stride N into dst, rows in order (deterministic).
__device__ __forceinline__ void reduce_rows_f32(const float *base, int nrows, int N, float *dst) {
  for (int i = threadIdx.x; i < (N >> 2); i += kThreads) {
    const int c = i * 4;
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int j = 0;
    for (; j + 4 <= nrows; j += 4) {  // 4 independent loads in flight per thread
      const float4 v0 = ld_cg_f4(base + static_cast<size_t>(j + 0) * N + c);
      const float4 v1 = ld_cg_f4(base + static_cast<size_t>(j + 1) * N + c);
      const float4 v2 = ld_cg_f4(base + static_cast<size_t>(j + 2) * N + c);
      const float4 v3 = ld_cg_f4(base + static_cast<size_t>(j + 3) * N + c);
      s.x += v0.x; s.y += v0.y; s.z += v0.z; s.w += v0.w;  // NOLINT(*)
      s.x += v1.x; s.y += v1.y; s.z += v1.z; s.w += v1.w;  // NOLINT(*)
      s.x += v2.x; s.y += v2.y; s.z += v2.z; s.w += v2.w;  // NOLINT(*)
      s.x += v3.x; s.y += v3.y; s.z += v3.z; s.w += v3.w;  // NOLINT(*)
    }
    for (; j < nrows; ++j) {
      const float4 v = ld_cg_f4(base + static_cast<size_t>(j) * N + c);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;  // NOLINT(*)
    }
    *reinterpret_cast<float4 *>(dst + c) = s;
  }
}

// ---- MXFP8 emission of dx (fused backward)
// kQuant: 0 = none, 1 = E4M3, 2 = E5M2.
template <int kQuant>
using QuantOType = std::conditional_t<kQuant == 2, fp8e5m2, fp8e4m3>;

// Eight BF16 values (4 packed pairs) times per-value multipliers, converted to eight FP8 values
// (element 0 in the lowest byte), as the MXFP8 quantizer does: cvt.rn.satfinite(x * 2^-scale)
// (the BF16 -> FP32 conversion is exact).
template <int kQuant>
__device__ __forceinline__ uint2 quantize_8x(const uint32_t (&w)[4], const float (&m)[8]) {
  using OTypex2 = std::conditional_t<kQuant == 2, ptx::fp8e5m2x2, ptx::fp8e4m3x2>;
  OTypex2 o[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    ptx::mul_cvt_2x(o[k], reinterpret_cast<const ptx::bf16x2 &>(w[k]),
                    ptx::floatx2{m[2 * k], m[2 * k + 1]});
  }
  const auto u = [&](int k) {
    return static_cast<uint32_t>(reinterpret_cast<const uint16_t &>(o[k]));
  };
  return make_uint2(u(0) | (u(1) << 16), u(2) | (u(3) << 16));
}

// Per-lane max of |a| and |b| on packed BF16 pairs (the magnitude is exact; the sign is dropped
// by the caller).
__device__ __forceinline__ uint32_t absmax_bf16x2(const uint32_t a, const uint32_t b) {
  uint32_t d;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ float abs_lo(const uint32_t w) { return fabsf(bf2f(w, 0)); }
__device__ __forceinline__ float abs_hi(const uint32_t w) { return fabsf(bf2f(w, 1)); }

// Row-wise scale index of (row, scale column j); column-wise of (block row j, column).
__device__ __forceinline__ size_t row_scale_idx(const Params &p, int row, int j) {
  return p.swizzled ? dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx(row, j, p.s_row_cols / 4)
                    : static_cast<size_t>(row) * p.s_row_cols + j;
}
__device__ __forceinline__ size_t col_scale_idx(const Params &p, int blk32, int col) {
  return p.swizzled ? dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx(
                          col, blk32, (p.nb32 + 3) / 4)
                    : static_cast<size_t>(blk32) * p.s_col_cols + col;
}

// The CTA's dx store with MXFP8 emission. Per row: the final dx (dz slab, plus the residual
// gradient) is stored in BF16, kept in the dz slab, and quantized row-wise right away (a 32-column
// block is 4 adjacent threads; N % 32 == 0, so no block mixes owners and idle threads); the
// per-column maxima over the CTA's rows accumulate in registers. Column-wise blocks are the
// cluster's 32 rows: every CTA pushes its maxima into each peer's inbox (`inbox`, kCluster x N
// floats in the no longer used x slab), one split cluster barrier (row-wise scale padding in
// between), then each CTA quantizes its rows column-wise from the dz slab. Called by every thread
// of every CTA of the cluster.
template <int kQuant>
__device__ __forceinline__ void store_dx_quantized(const Params &p, uint16_t *s_dz, uint32_t *inbox,
                                                   const int rank, const int row0,
                                                   const int blk32) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr float kMaxNormRcp = Quantized_Limits<QuantOType<kQuant>>::max_norm_rcp;
  const int N = p.N;
  const int tid = threadIdx.x;
  const bool owner = tid < (N >> 3);
  const int c0 = tid * 8;
  const bool has_add = p.add != nullptr;
  uint32_t cmax[4] = {0u, 0u, 0u, 0u};  // per-column |max| over the CTA's rows, BF16 pairs

  uint4 a[kRows];
  if (owner && has_add) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {  // all loads in flight before the first store
      a[r] = __ldg(reinterpret_cast<const uint4 *>(p.add + static_cast<size_t>(row0 + r) * N + c0));
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    uint32_t w[4] = {0u, 0u, 0u, 0u};
    float amax = 0.0f;
    if (owner) {
      uint4 *slot = reinterpret_cast<uint4 *>(s_dz + r * N + c0);
      uint4 v = *slot;
      if (has_add) {
        v = add_bf16x8(v, a[r]);
        *slot = v;
      }
      *reinterpret_cast<uint4 *>(p.dx + static_cast<size_t>(row0 + r) * N + c0) = v;
      w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;  // NOLINT(*)
#pragma unroll
      for (int k = 0; k < 4; ++k) cmax[k] = absmax_bf16x2(cmax[k], w[k]);
      const uint32_t m2 = absmax_bf16x2(absmax_bf16x2(w[0], w[1]), absmax_bf16x2(w[2], w[3]));
      amax = fmaxf(abs_lo(m2), abs_hi(m2));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (owner) {
      const e8m0_t e = ptx::float_to_e8m0(amax * kMaxNormRcp);
      const float mult = ptx::exp2f_rcp<float>(e);
      const float m[8] = {mult, mult, mult, mult, mult, mult, mult, mult};
      *reinterpret_cast<uint2 *>(p.q_row + static_cast<size_t>(row0 + r) * N + c0) =
          quantize_8x<kQuant>(w, m);
      if ((tid & 3) == 0) p.s_row[row_scale_idx(p, row0 + r, tid >> 2)] = e;
    }
  }

  // Column maxima to every CTA of the cluster (slot `rank` of each inbox: kCluster x N / 2 pairs).
  if (owner) {
#pragma unroll
    for (int q = 0; q < kCluster; ++q) {
      const uint32_t dst = mapa_u32(smem_u32(inbox + rank * (N >> 1) + (c0 >> 1)), q);
      asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(dst), "r"(cmax[0]),
                   "r"(cmax[1]), "r"(cmax[2]), "r"(cmax[3])
                   : "memory");
    }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  {  // row-wise scale padding (columns N / 32 .. s_row_cols - 1) of this CTA's rows
    const int npad = p.s_row_cols - (N >> 5);
    for (int t = tid; t < kRows * npad; t += kThreads) {
      p.s_row[row_scale_idx(p, row0 + t / npad, (N >> 5) + t % npad)] = 0;
    }
  }
  if (rank == kCluster - 1) {  // column-wise scale padding (columns N ..) of this 32-row block
    for (int t = tid; t < p.s_col_cols - N; t += kThreads) {
      p.s_col[col_scale_idx(p, blk32, N + t)] = 0;
    }
  }
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

  if (owner) {
#pragma unroll
    for (int q = 0; q < kCluster; ++q) {
      const uint4 v = *reinterpret_cast<const uint4 *>(inbox + q * (N >> 1) + (c0 >> 1));
      cmax[0] = absmax_bf16x2(cmax[0], v.x);
      cmax[1] = absmax_bf16x2(cmax[1], v.y);
      cmax[2] = absmax_bf16x2(cmax[2], v.z);
      cmax[3] = absmax_bf16x2(cmax[3], v.w);
    }
    e8m0_t e[8];
    float m[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      e[2 * k] = ptx::float_to_e8m0(abs_lo(cmax[k]) * kMaxNormRcp);
      e[2 * k + 1] = ptx::float_to_e8m0(abs_hi(cmax[k]) * kMaxNormRcp);
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) m[c] = ptx::exp2f_rcp<float>(e[c]);
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const uint4 v = *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
      *reinterpret_cast<uint2 *>(p.q_col + static_cast<size_t>(row0 + r) * N + c0) =
          quantize_8x<kQuant>(w, m);
    }
    if (rank == 0) {
#pragma unroll
      for (int c = 0; c < 8; ++c) p.s_col[col_scale_idx(p, blk32, c0 + c)] = e[c];
    }
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// Blocking ticket (levels 2 and 3): fences this CTA's stores, returns true (CTA-uniform) for
// the last arrival, which also re-arms the ticket.
__device__ __forceinline__ bool ticket_last(int *counter, int target, int *s_flag) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const int old = atomicAdd(counter, 1);
    const int last = (old == target - 1) ? 1 : 0;
    if (last) *counter = 0;  // every arrival is in; the kernel boundary publishes the reset
    __threadfence();
    *s_flag = last;
  }
  __syncthreads();
  return *s_flag != 0;
}

#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// ------------------------------------------------------------------------------- the kernel
template <int kQuant>
__global__ void __launch_bounds__(kThreads, 2) rmsnorm_bwd_single_sweep_kernel(const Params p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(128) uint8_t smem[];

  const int N = p.N;
  const int NV = N >> 3;            // 8-column groups
  const int cslice = N / kCluster;  // column slice reduced by each CTA of the cluster
  uint16_t *s_dz = reinterpret_cast<uint16_t *>(smem);
  uint16_t *s_x = reinterpret_cast<uint16_t *>(smem + static_cast<size_t>(kRows) * N * 2);
  float *s_dg_in = reinterpret_cast<float *>(smem + static_cast<size_t>(kRows) * N * 4);
  uint8_t *misc = smem + static_cast<size_t>(kRows) * N * 4 + static_cast<size_t>(N) * 4;
  uint64_t *mbar = reinterpret_cast<uint64_t *>(misc);
  float *s_red = reinterpret_cast<float *>(misc + 16);
  float *s_k = s_red + kWarps * kRows;
  float *s_rsigma = s_k + kRows;
  int *s_flag = reinterpret_cast<int *>(s_rsigma + kRows);

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int blk = blockIdx.x;
  const int blk32 = blk / kCluster;
  const int row0 = blk * kRows;
  const bool owner = tid < NV;  // this thread owns columns 8 * tid .. 8 * tid + 7
  const int c0 = tid * 8;

  // ---- TMA bulk loads of the dz / x slabs (the 8 rows are contiguous in global memory)
  if (tid == 0) {
    mbar_init(mbar, 1);
    fence_barrier_init();
  }
  __syncthreads();
  // Peers may write into this CTA's inbox only after it has started: first half of a split
  // cluster barrier, completed just before the pushes.
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  if (tid == 0) {
    mbar_expect_tx(mbar, static_cast<uint32_t>(2 * kRows * N * 2));
    bulk_g2s(s_dz, p.dz + static_cast<size_t>(row0) * N, static_cast<uint32_t>(kRows * N * 2),
             mbar);
    bulk_g2s(s_x, p.x + static_cast<size_t>(row0) * N, static_cast<uint32_t>(kRows * N * 2),
             mbar);
    // The residual gradient is read only at the dx store: start bringing its slab into L2 now.
    if (p.add != nullptr) {
      bulk_prefetch_l2(p.add + static_cast<size_t>(row0) * N, static_cast<uint32_t>(kRows * N * 2));
    }
  }
  if (tid < kRows) s_rsigma[tid] = p.rsigma[row0 + tid];
  float g[8];
  {
    uint4 gv = make_uint4(0u, 0u, 0u, 0u);
    if (owner) gv = __ldg(reinterpret_cast<const uint4 *>(p.gamma + c0));
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    const float add = p.zero_centered ? 1.0f : 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) g[c] = bf2f(gw[c >> 1], c & 1) + add;
  }
  // Second incoming gradient: loads in flight while the slabs arrive, summed into the dz slab.
  const bool has_dz2 = p.dz2 != nullptr;
  uint4 d2[kRows];
  if (has_dz2 && owner) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      d2[r] = __ldg(reinterpret_cast<const uint4 *>(p.dz2 + static_cast<size_t>(row0 + r) * N + c0));
    }
  }
  __syncthreads();  // s_rsigma
  mbar_wait(mbar, 0);
  if (has_dz2 && owner) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      uint4 *dst = reinterpret_cast<uint4 *>(s_dz + r * N + c0);
      *dst = add_bf16x8(*dst, d2[r]);
    }
  }

  // ---- sweep 1: per-row sum of dz * g * x
  {
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;
    if (owner) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint4 a = *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
        const uint4 b = *reinterpret_cast<const uint4 *>(s_x + r * N + c0);
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
        const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) sum += bf2f(aw[c >> 1], c & 1) * g[c] * bf2f(bw[c >> 1], c & 1);
        acc[r] = sum;
      }
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) s_red[warp * kRows + r] = acc[r];
    }
    __syncthreads();
    if (tid < kRows) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) sum += s_red[w * kRows + tid];
      const float rs = s_rsigma[tid];
      s_k[tid] = sum * (rs * rs * rs) / static_cast<float>(N);
    }
    __syncthreads();
  }

  // ---- sweep 2: dx -> bf16; per-column dgamma over the CTA's rows in registers,
  // pushed to the inbox of the peer that owns the column slice
  {
    float dg[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) dg[c] = 0.0f;
    if (owner) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint4 a = *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
        const uint4 b = *reinterpret_cast<const uint4 *>(s_x + r * N + c0);
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
        const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
        const float rs = s_rsigma[r];
        const float k = s_k[r];
        uint32_t ow[4];
#pragma unroll
        for (int c = 0; c < 8; c += 2) {
          float v2[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float dzf = bf2f(aw[c >> 1], e);
            const float xf = bf2f(bw[c >> 1], e);
            v2[e] = rs * (dzf * g[c + e]) - xf * k;
            dg[c + e] += dzf * xf * rs;
          }
          ow[c >> 1] = pack_bf16x2(v2[0], v2[1]);
        }
        // in place of dz in shared memory; stored to global after the level-1 ticket is issued
        *reinterpret_cast<uint4 *>(s_dz + r * N + c0) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
      }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // every peer started
    if (owner) {
      const int q = c0 / cslice;  // the peer whose column slice holds my 8 columns
      const uint32_t dst = mapa_u32(smem_u32(s_dg_in + rank * cslice + (c0 - q * cslice)), q);
      st_dsmem_f4(dst, make_float4(dg[0], dg[1], dg[2], dg[3]));
      st_dsmem_f4(dst + 16u, make_float4(dg[4], dg[5], dg[6], dg[7]));
    }
  }
  cluster.sync();  // every push has landed; from here on only local shared memory is read

  // ---- the 32-row block's partial row (this CTA's column slice), peers summed in rank order
  {
    float *part_row = p.part + static_cast<size_t>(blk32) * N;
    for (int i = tid; i < (cslice >> 2); i += kThreads) {
      const int cc = i * 4;
      float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (int q = 0; q < kCluster; ++q) {
        const float4 v = *reinterpret_cast<const float4 *>(s_dg_in + q * cslice + cc);
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;  // NOLINT(*)
      }
      *reinterpret_cast<float4 *>(part_row + rank * cslice + cc) = sum;
    }
  }

  // ---- the level-1 ticket, issued now by thread 0 alone: its fence and the atomic's latency
  // overlap the dx stores below
  __syncthreads();
  const int g1 = blk32 / p.group;
  const int gsize1 = min(p.group, p.nb32 - g1 * p.group);
  if (tid == 0) {
    __threadfence();
    const int old = atomicAdd(p.tickets + 1 + g1, 1);
    const int last = (old == gsize1 * kCluster - 1) ? 1 : 0;
    if (last) p.tickets[1 + g1] = 0;
    s_flag[0] = last;
  }

  // ---- dx (bf16, staged in shared memory) to global, plus the residual gradient if given
  if constexpr (kQuant != 0) {
    // with its MXFP8 copy (the x slab is the column-maxima inbox from here on)
    store_dx_quantized<kQuant>(p, s_dz, reinterpret_cast<uint32_t *>(s_x), rank, row0, blk32);
  } else if (owner) {
    if (p.add != nullptr) {
      uint4 a[kRows];
#pragma unroll
      for (int r = 0; r < kRows; ++r) {  // all loads in flight before the first store
        a[r] = __ldg(reinterpret_cast<const uint4 *>(p.add + static_cast<size_t>(row0 + r) * N + c0));
      }
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        *reinterpret_cast<uint4 *>(p.dx + static_cast<size_t>(row0 + r) * N + c0) =
            add_bf16x8(*reinterpret_cast<const uint4 *>(s_dz + r * N + c0), a[r]);
      }
    } else {
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        *reinterpret_cast<uint4 *>(p.dx + static_cast<size_t>(row0 + r) * N + c0) =
            *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
      }
    }
  }

  // ---- in-kernel dgamma reduction: 3-level tree of fixed-order sums, each level finished by
  // the last-arriving CTA alone
  __syncthreads();
  if (s_flag[0] == 0) return;
  if (tid == 0) __threadfence();  // acquire side of the level-1 ticket
  __syncthreads();
  reduce_rows_f32(p.part + static_cast<size_t>(g1) * p.group * N, gsize1, N,
                  p.part2 + static_cast<size_t>(g1) * N);
  const int g2 = g1 / kFan2;
  const int gsize2 = min(kFan2, p.ngroups - g2 * kFan2);
  if (!ticket_last(p.tickets + 1 + kMaxGroups + g2, gsize2, s_flag)) return;
  reduce_rows_f32(p.part2 + static_cast<size_t>(g2) * kFan2 * N, gsize2, N,
                  p.part3 + static_cast<size_t>(g2) * N);
  const int ngroups2 = (p.ngroups + kFan2 - 1) / kFan2;
  if (!ticket_last(p.tickets, ngroups2, s_flag)) return;
  for (int i = tid; i < (N >> 2); i += kThreads) {
    const int c = i * 4;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int gg = 0; gg < ngroups2; ++gg) {
      const float4 v = ld_cg_f4(p.part3 + static_cast<size_t>(gg) * N + c);
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;  // NOLINT(*)
    }
    p.dgamma[c + 0] = f2bf(sum.x);
    p.dgamma[c + 1] = f2bf(sum.y);
    p.dgamma[c + 2] = f2bf(sum.z);
    p.dgamma[c + 3] = f2bf(sum.w);
  }
#else
  NVTE_DEVICE_THREAD0_ERROR("Single-sweep RMSNorm backward requires SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// ------------------------------------------------------------------------------- host helpers
// Per-device facts checked once: the kernel's cluster launch is possible and fits.
struct DeviceSupport {
  bool ok = false;
  std::string why;
};

const DeviceSupport &device_support(const int device_id) {
  static std::vector<DeviceSupport> cache(cuda::num_devices());
  static std::vector<std::once_flag> flags(cuda::num_devices());
  std::call_once(flags[device_id], [&]() {
    DeviceSupport &s = cache[device_id];
    if (cuda::sm_arch(device_id) < 90) {
      s.why = "requires SM 9.0+ (thread-block clusters, TMA bulk copies)";
      return;
    }
    const size_t need = smem_bytes(kMaxCols);
    if (cuda::max_shared_memory_per_block_optin(device_id) < need) {
      s.why = concat_strings("requires ", need, " bytes of shared memory per block");
      return;
    }
    s.ok = true;
  });
  return cache[device_id];
}

// Per-device ring of ticket slots, zero at rest (each launch leaves its slot zeroed).
struct TicketRing {
  int *ptr = nullptr;
  std::atomic<uint32_t> next{0};
};

TicketRing &ticket_ring(const int device_id) {
  static std::vector<TicketRing> rings(cuda::num_devices());
  return rings[device_id];
}

// A slot of the device's ticket ring, allocating the ring on first use; nullptr when the ring
// does not exist yet and cannot be allocated now (inside a stream capture).
int *ticket_slot(const int device_id, cudaStream_t stream) {
  static std::mutex mutex;
  TicketRing &ring = ticket_ring(device_id);
  int *base = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (ring.ptr == nullptr) {
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &status));
      if (status != cudaStreamCaptureStatusNone) {
        return nullptr;
      }
      int *p = nullptr;
      const size_t bytes = sizeof(int) * kTicketSlot * kTicketRing;
      NVTE_CHECK_CUDA(cudaMalloc(&p, bytes));
      NVTE_CHECK_CUDA(cudaMemset(p, 0, bytes));  // synchronous: zero before any launch uses it
      ring.ptr = p;
    }
    base = ring.ptr;
  }
  const uint32_t slot = ring.next.fetch_add(1, std::memory_order_relaxed) % kTicketRing;
  return base + static_cast<size_t>(slot) * kTicketSlot;
}

struct WorkspaceLayout {
  size_t part, part2, part3, tickets, total;
};

WorkspaceLayout workspace_layout(const size_t rows, const size_t cols) {
  const Plan plan = make_plan(rows);
  WorkspaceLayout w;
  w.part = 0;
  w.part2 = w.part + align_up(sizeof(float) * plan.nb32 * cols);
  w.part3 = w.part2 + align_up(sizeof(float) * plan.ngroups * cols);
  w.tickets = w.part3 + align_up(sizeof(float) * plan.ngroups2 * cols);
  w.total = w.tickets + align_up(sizeof(int) * kTicketSlot);
  return w;
}

}  // namespace

bool is_supported_by_rmsnorm_bwd_single_sweep(const Tensor &dz, const Tensor &x,
                                               const Tensor &rsigma, const Tensor &gamma,
                                               const Tensor &dx, const Tensor &dgamma,
                                               const Tensor *add, const bool zero_centered_gamma,
                                               std::string *reason, const Tensor *dz2,
                                               const Tensor *dx_q) {
  auto fail = [&](const std::string &why) {
    if (reason != nullptr) *reason = why;
    return false;
  };
  const auto bf16 = DType::kBFloat16;
  if (x.data.dtype != bf16 || dz.data.dtype != bf16 || gamma.data.dtype != bf16 ||
      dx.data.dtype != bf16 || dgamma.data.dtype != bf16) {
    return fail(concat_strings("needs BF16 x, dz, gamma, dx and dgamma (got x ",
                               to_string(x.data.dtype), ", gamma ", to_string(gamma.data.dtype),
                               ")"));
  }
  if (rsigma.data.dtype != DType::kFloat32) {
    return fail("needs FP32 rsigma");
  }
  if (x.data.shape.empty()) {
    return fail("needs at least a 1-D input");
  }
  const auto [rows, cols] = x.flat_2d_dims();
  if (rows == 0 || rows % kBlockRows != 0) {
    return fail(concat_strings("needs a positive number of rows divisible by ", kBlockRows,
                               " (got ", rows, ")"));
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return fail("too many rows");
  }
  if (cols == 0 || cols % 32 != 0 || cols > static_cast<size_t>(kMaxCols)) {
    return fail(concat_strings("needs a hidden size divisible by 32 and at most ", kMaxCols,
                               " (got ", cols, ")"));
  }
  if (!is_ptr_aligned(dz.data.dptr, x.data.dptr, dx.data.dptr, gamma.data.dptr) ||
      !is_ptr_aligned<4>(rsigma.data.dptr) || !is_ptr_aligned<2>(dgamma.data.dptr)) {
    return fail("needs 16-byte aligned dz, x, dx and gamma");
  }
  if (add != nullptr) {
    if (add->data.dtype != bf16) {
      return fail(concat_strings("needs a BF16 residual gradient (got ", to_string(add->data.dtype),
                                 ")"));
    }
    if (add->data.shape != x.data.shape) {
      return fail("needs a residual gradient shaped like x");
    }
    if (!is_ptr_aligned(add->data.dptr)) {
      return fail("needs a 16-byte aligned residual gradient");
    }
  }
  if (dz2 != nullptr) {
    if (dz2->data.dtype != bf16 || dz2->data.shape != dz.data.shape) {
      return fail("needs a BF16 second gradient shaped like dz");
    }
    if (!is_ptr_aligned(dz2->data.dptr)) {
      return fail("needs a 16-byte aligned second gradient");
    }
  }
  if (dx_q != nullptr) {
    if (!is_mxfp8_scaling(dx_q->scaling_mode)) {
      return fail("quantized dx: needs MXFP8 scaling");
    }
    if (!dx_q->has_data() || !dx_q->has_columnwise_data()) {
      return fail("quantized dx: needs row-wise and column-wise data");
    }
    const DType qt = dx_q->data.dtype;
    if ((qt != DType::kFloat8E4M3 && qt != DType::kFloat8E5M2) ||
        dx_q->columnwise_data.dtype != qt) {
      return fail("quantized dx: needs E4M3 or E5M2 data in both directions");
    }
    if (rows % 128 != 0) {
      return fail(concat_strings("quantized dx: needs rows divisible by 128 (got ", rows, ")"));
    }
    if (dx_q->data.numel() != rows * cols || dx_q->columnwise_data.numel() != rows * cols) {
      return fail("quantized dx: data shaped unlike x");
    }
    const Shape srow{rows, DIVUP(cols / 32, size_t{4}) * 4};
    const Shape scol{DIVUP(rows / 32, size_t{4}) * 4, DIVUP(cols, size_t{128}) * 128};
    if (dx_q->scale_inv.shape != srow || dx_q->columnwise_scale_inv.shape != scol) {
      return fail("quantized dx: unexpected scale shapes");
    }
    if (!is_ptr_aligned<8>(dx_q->data.dptr, dx_q->columnwise_data.dptr)) {
      return fail("quantized dx: needs 8-byte aligned data");
    }
    if (cuda::sm_arch(cuda::current_device()) < 100) {
      return fail("quantized dx: requires SM 10.0+");
    }
  }
  if (zero_centered_gamma && use_zero_centered_gamma_in_weight_dtype()) {
    return fail("zero-centered gamma in weight dtype (NVTE_ZERO_CENTERED_GAMMA_IN_WTYPE)");
  }
  const DeviceSupport &device = device_support(cuda::current_device());
  if (!device.ok) {
    return fail(device.why);
  }
  return true;
}

size_t rmsnorm_bwd_single_sweep_workspace_bytes(const size_t rows, const size_t cols) {
  return workspace_layout(rows, cols).total;
}

void rmsnorm_bwd_single_sweep(const Tensor &dz, const Tensor &x, const Tensor &rsigma,
                              const Tensor &gamma, const Tensor *add, Tensor *dx, Tensor *dgamma,
                              void *workspace, const bool zero_centered_gamma, cudaStream_t stream,
                              const Tensor *dz2, Tensor *dx_q) {
  const auto [rows, cols] = x.flat_2d_dims();
  const Plan plan = make_plan(rows);
  const WorkspaceLayout layout = workspace_layout(rows, cols);
  NVTE_CHECK(is_ptr_aligned(workspace), "Workspace must be 16-byte aligned.");
  auto *ws = static_cast<uint8_t *>(workspace);

  Params p;
  p.dz = static_cast<const uint16_t *>(dz.data.dptr);
  p.x = static_cast<const uint16_t *>(x.data.dptr);
  p.rsigma = static_cast<const float *>(rsigma.data.dptr);
  p.gamma = static_cast<const uint16_t *>(gamma.data.dptr);
  p.add = add != nullptr ? static_cast<const uint16_t *>(add->data.dptr) : nullptr;
  p.dx = static_cast<uint16_t *>(dx->data.dptr);
  p.dgamma = static_cast<uint16_t *>(dgamma->data.dptr);
  p.part = reinterpret_cast<float *>(ws + layout.part);
  p.part2 = reinterpret_cast<float *>(ws + layout.part2);
  p.part3 = reinterpret_cast<float *>(ws + layout.part3);
  p.M = static_cast<int>(rows);
  p.N = static_cast<int>(cols);
  p.nb32 = plan.nb32;
  p.group = plan.group;
  p.ngroups = plan.ngroups;
  p.zero_centered = zero_centered_gamma ? 1 : 0;
  p.dz2 = dz2 != nullptr ? static_cast<const uint16_t *>(dz2->data.dptr) : nullptr;
  p.q_row = p.q_col = p.s_row = p.s_col = nullptr;
  p.s_row_cols = p.s_col_cols = 0;
  p.swizzled = 0;
  int quant = 0;
  if (dx_q != nullptr) {
    quant = dx_q->data.dtype == DType::kFloat8E5M2 ? 2 : 1;
    p.q_row = static_cast<uint8_t *>(dx_q->data.dptr);
    p.q_col = static_cast<uint8_t *>(dx_q->columnwise_data.dptr);
    p.s_row = static_cast<uint8_t *>(dx_q->scale_inv.dptr);
    p.s_col = static_cast<uint8_t *>(dx_q->columnwise_scale_inv.dptr);
    p.s_row_cols = static_cast<int>(dx_q->scale_inv.shape[1]);
    p.s_col_cols = static_cast<int>(dx_q->columnwise_scale_inv.shape[1]);
    p.swizzled = dx_q->with_gemm_swizzled_scales ? 1 : 0;
  }

  p.tickets = ticket_slot(cuda::current_device(), stream);
  if (p.tickets == nullptr) {
    // First use on this device inside a stream capture: the tickets come from the workspace and
    // are zeroed by a (captured) memset.
    p.tickets = reinterpret_cast<int *>(ws + layout.tickets);
    NVTE_CHECK_CUDA(cudaMemsetAsync(p.tickets, 0, sizeof(int) * kTicketSlot, stream));
  }

  const size_t smem = smem_bytes(p.N);
  auto kernel = quant == 2   ? rmsnorm_bwd_single_sweep_kernel<2>
                : quant == 1 ? rmsnorm_bwd_single_sweep_kernel<1>
                             : rmsnorm_bwd_single_sweep_kernel<0>;
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(smem)));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(rows / kRows), 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = kCluster;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  NVTE_CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, p));
}

}  // namespace normalization
}  // namespace transformer_engine
