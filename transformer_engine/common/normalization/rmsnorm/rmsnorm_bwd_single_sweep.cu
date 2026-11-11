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
 *  Origin: the single-sweep cluster kernel of the gpt-oss-20b "plus" workload
 *  (rmsnorm_bwd_cuda.cu, rmsnorm_bwd_single_sweep_kernel<8, false, false, false>, NVIDIA
 *  first-party), reduced to the plain backward: no residual-gradient add, no MXFP8 gradient
 *  emission, no dbias, dgamma written to the output tensor (not accumulated into main_grad).
 */

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
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
  uint16_t *dx;           // [M, N] bf16
  uint16_t *dgamma;       // [N] bf16
  float *part;            // [M / 32, N] fp32 partials of the 32-row blocks
  float *part2;           // [ngroups, N] fp32 level-1 partials
  float *part3;           // [ceil(ngroups / 8), N] fp32 level-2 partials
  int *tickets;           // [kTicketSlot], zero at rest
  int M, N, nb32, group, ngroups;
  int zero_centered;
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
  __syncthreads();  // s_rsigma
  mbar_wait(mbar, 0);

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

  // ---- dx (bf16, staged in shared memory) to global
  if (owner) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      *reinterpret_cast<uint4 *>(p.dx + static_cast<size_t>(row0 + r) * N + c0) =
          *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
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
                                               const bool zero_centered_gamma,
                                               std::string *reason) {
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
                              const Tensor &gamma, Tensor *dx, Tensor *dgamma, void *workspace,
                              const bool zero_centered_gamma, cudaStream_t stream) {
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

  p.tickets = ticket_slot(cuda::current_device(), stream);
  if (p.tickets == nullptr) {
    // First use on this device inside a stream capture: the tickets come from the workspace and
    // are zeroed by a (captured) memset.
    p.tickets = reinterpret_cast<int *>(ws + layout.tickets);
    NVTE_CHECK_CUDA(cudaMemsetAsync(p.tickets, 0, sizeof(int) * kTicketSlot, stream));
  }

  const size_t smem = smem_bytes(p.N);
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(rmsnorm_bwd_single_sweep_kernel,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
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
  NVTE_CHECK_CUDA(cudaLaunchKernelEx(&cfg, rmsnorm_bwd_single_sweep_kernel, p));
}

}  // namespace normalization
}  // namespace transformer_engine
