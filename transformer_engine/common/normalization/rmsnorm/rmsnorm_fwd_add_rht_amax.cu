/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Residual add + RMSNorm forward + NVFP4 random-Hadamard-transform amaxes in one pass.
 *
 *   s     = bf16(float(x) + float(residual))
 *   rs    = 1 / sqrt(mean(s^2) + eps)          (statistics of the tuned RMSNorm forward kernel)
 *   z     = bf16(gamma * (rs * s))
 *   amax  = { max|z|, max|bf16(RHT16(z^T))| }  (nvte_hadamard_transform_amax: pre-RHT amax and
 *                                               amax of the column-wise RHT)
 *
 * Every output is bitwise the one of the unfused chain (bf16 add, nvte_rmsnorm_fwd,
 * nvte_hadamard_transform_amax):
 *  - The per-row statistics use the per-thread element assignment, reduction tree and expression
 *    forms of the tuned RMSNorm forward kernel for hidden size 4096 in BF16 (CTAS_PER_ROW=1,
 *    WARPS_N=4, BYTES_PER_LDG=16): see rmsnorm_fwd_kernels.cuh and Stats/Reducer in utils.cuh.
 *    The block-level reduction uses a per-row-slot named barrier in place of __syncthreads; the
 *    reduction tree is unchanged.
 *  - The RHT amax applies the same wmma m16n16k16 instruction to the same fragments as
 *    HadamardAmaxTmaKernel (hadamard_transform_utils.cuh), so every RHT value is bitwise the one
 *    the unfused kernel takes the maximum of.
 *
 * Persistent CTAs (at most one per SM) walk groups of 16 consecutive rows (the RHT block of the
 * column-wise copy spans 16 rows). A producer warp streams the rows of x and residual into a ring
 * of shared-memory stages with bulk copies; kWarpsM row slots of 4 warps each run the RMSNorm
 * row code. The BF16 z rows of a group stay in shared memory and, once the 16 rows are done, the
 * consumer warps take the 16x16 tiles of z^T through the wmma. Per-CTA maxima are reduced by the
 * last CTA, which also resets the arrival counter, so a zero-filled workspace stays zero-filled
 * and can be reused (and captured in a CUDA graph) without a memset.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <transformer_engine/normalization.h>

#include <algorithm>

#include "../../common.h"
#include "../../hadamard_transform/hadamard_transform_utils.cuh"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"

namespace transformer_engine {
namespace normalization {
namespace add_rht_amax {

constexpr int kWarp = 32;
constexpr int kHidden = 4096;
constexpr int kWarpsN = 4;                        // tuned RMSNorm config for hidden size 4096
constexpr int kNumElts = 8;                       // 16 B per load / 2 B
constexpr int kThreadsPerRow = kWarpsN * kWarp;   // 128
constexpr int kVecCols = kHidden / kNumElts;      // 512
constexpr int kVecColsPerLdg = kThreadsPerRow;    // 128 (CTAS_PER_ROW == 1)
constexpr int kLdgs = kVecCols / kVecColsPerLdg;  // 4
constexpr int kElts = kLdgs * kNumElts;           // 32 per thread per row
constexpr int kGroupRows = 16;                    // RHT dimension
constexpr int kTileStride = kHidden + 8;          // BF16 elements; 16 B pad: conflict-free ldmatrix
constexpr int kRowBytes = kHidden * 2;            // 8 KiB per BF16 row
constexpr int kWarpsM = 4;                        // row slots per CTA
constexpr int kStagesPerSlot = 1;                 // bulk-copy stages per row slot
constexpr size_t kWorkspaceCounterBytes = 16;     // arrival counter, padded to 16 B

// Shared memory layout: kWarpsM row slots (4 warps each) + 1 producer warp. Every slot owns a ring
// of kStagesPerSlot stages (one x row + one residual row each) filled by its own producer lane, so
// a slot always waits on its stages in order.
struct Layout {
  static constexpr int kStages = kWarpsM * kStagesPerSlot;
  static constexpr int kConsumers = kWarpsM * kThreadsPerRow;
  static constexpr int kThreads = kConsumers + kWarp;
  static constexpr int kNumWarps = kThreads / kWarp;
  static constexpr size_t kTileBytes = sizeof(__nv_bfloat16) * kGroupRows * kTileStride;
  static constexpr size_t kStageBytes = 2 * kRowBytes;
  static constexpr size_t kStagesOff = kTileBytes;  // 16 B aligned (kTileStride * 2 % 16 == 0)
  static constexpr size_t kBarOff = kStagesOff + kStages * kStageBytes;
  static constexpr size_t kStatsOff = kBarOff + 2 * kStages * sizeof(uint64_t);
  static constexpr size_t kRedOff = kStatsOff + sizeof(float2) * 2 * kWarpsM * kWarpsN;
  static constexpr size_t kBytes = kRedOff + sizeof(float) * 2 * kNumWarps;
};

// Reducer<float, 1, WARPS_M, 1>::allreduce_ with Sum<float>
__device__ __forceinline__ float warp_allreduce_sum(float data) {
#pragma unroll
  for (int it = 1; it < kWarp; it *= 2) {
    data = data + warp_shuffle_xor(data, it);
  }
  return data;
}

// Stats<float, 1, WARPS_M, 1>::compute (warp level), N = kElts
__device__ __forceinline__ float2 warp_stats(const float (&elts)[kElts], const float rn) {
  float m = 0.f;
#pragma unroll
  for (int it = 0; it < kElts; it++) {
    m += elts[it];
  }
  m = warp_allreduce_sum(m) * rn;

  float m2 = 0.f;
#pragma unroll
  for (int it = 0; it < kElts; it++) {
    float diff = (elts[it] - m);
    m2 += diff * diff;
  }
  m2 = warp_allreduce_sum(m2);
  return make_float2(m, m2);
}

// Stats<float, 1, WARPS_M, WARPS_N>::compute (block level). smem_slot points at this row slot's
// 2 * kWarpsN float2 (two alternating buffers, as the smem0_/smem1_ buffers of Stats).
__device__ __forceinline__ float2 block_stats(const float (&elts)[kElts], float2 *smem_slot,
                                              bool &use0, int warp_n, int lane, int barrier_id) {
  float2 *smem = use0 ? smem_slot : smem_slot + kWarpsN;
  use0 = !use0;
  constexpr float warp_rn = 1.f / static_cast<float>(kElts * kWarp);
  float2 ws = warp_stats(elts, warp_rn);
  if (lane == 0) {
    smem[warp_n] = ws;
  }
  ptx::numbered_barrier_sync(kThreadsPerRow, barrier_id);  // __syncthreads over the row's warps

  float n = 0.f;
  float m = 0.f;
  float m2 = 0.f;
  if (lane < kWarpsN) {
    float2 result = smem[lane];
    n = kElts * kWarp;
    m = result.x;
    m2 = result.y;
  }
  warp_chan_upd_dynamic(m, m2, n, kWarpsN);
  return make_float2(m, m2);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int d = 16; d > 0; d /= 2) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, d));
  }
  return v;
}

__device__ __forceinline__ float bf16lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Round-to-nearest conversion of two floats into one bf16x2 word: lo -> bits 0..15
__device__ __forceinline__ uint32_t pack_bf16x2_rn(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint32_t bf16x2_absmax(uint32_t acc, uint32_t v) {
  uint32_t r;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(acc), "r"(v));
  return r;
}

__global__ void __launch_bounds__(Layout::kThreads, 1)
    add_rmsnorm_fwd_rht_amax_kernel(const __nv_bfloat16 *__restrict__ x,  // null: no add
                                    const __nv_bfloat16 *__restrict__ residual,
                                    const __nv_bfloat16 *__restrict__ gamma,
                                    __nv_bfloat16 *__restrict__ sum_out,  // null when no add
                                    __nv_bfloat16 *__restrict__ z_out,
                                    float *__restrict__ rsigma_out,
                                    float *__restrict__ amax_out,  // [2]: pre-RHT, RHT column-wise
                                    float *__restrict__ partials,  // [2 * gridDim.x]
                                    unsigned int *__restrict__ counter, const int rows,
                                    const float epsilon, const uint16_t random_sign_mask_t) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using L = Layout;
  constexpr int kNumWarps = L::kNumWarps;
  constexpr int kConsumerWarps = L::kConsumers / kWarp;
  static_assert(kGroupRows % kWarpsM == 0, "row slots must tile the 16-row group");
  static_assert(kNumWarps <= kWarp, "final reduction assumes at most 32 warps");
  static_assert(kWarpsM <= kWarp, "one producer lane per row slot");
  constexpr int kRowsPerSlot = kGroupRows / kWarpsM;
  constexpr int kColTiles = kHidden / 16;     // 16x16 tiles per group
  constexpr int kGroupBarrier = 1 + kWarpsM;  // ids 1..kWarpsM are the row slots

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16 *tile = reinterpret_cast<__nv_bfloat16 *>(smem_raw);  // [16][kTileStride]
  unsigned char *stages = smem_raw + L::kStagesOff;
  uint64_t *full_bar = reinterpret_cast<uint64_t *>(smem_raw + L::kBarOff);
  uint64_t *empty_bar = full_bar + L::kStages;
  float2 *stats_smem = reinterpret_cast<float2 *>(smem_raw + L::kStatsOff);
  float *red_smem = reinterpret_cast<float *>(smem_raw + L::kRedOff);  // [2][kNumWarps]
  __shared__ bool is_last_cta;

  const int tidx = threadIdx.x;
  const int lane = tidx % kWarp;
  const int warp = tidx / kWarp;
  const bool has_add = x != nullptr;
  const uint32_t stage_tx = has_add ? 2 * kRowBytes : kRowBytes;
  const int num_groups = rows / kGroupRows;

  if (tidx == 0) {
    for (int s = 0; s < L::kStages; s++) {
      ptx::mbarrier_init(&full_bar[s], 1);
      ptx::mbarrier_init(&empty_bar[s], kThreadsPerRow);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  __syncthreads();

  uint32_t amax_row_reg = 0u;  // bf16x2 +0.0
  uint32_t amax_t_reg = 0u;    // bf16x2 +0.0

  if (warp == kConsumerWarps) {
    // Producer warp: lane m streams row slot m's rows into slot m's ring
    if (lane < kWarpsM) {
      const int m = lane;
      uint32_t j = 0;
      for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        for (int i = 0; i < kRowsPerSlot; i++, j++) {
          const int s = m * kStagesPerSlot + static_cast<int>(j % kStagesPerSlot);
          const uint32_t phase = (j / kStagesPerSlot) & 1u;
          ptx::mbarrier_wait_parity(&empty_bar[s], phase ^ 1u);
          ptx::mbarrier_arrive_expect_tx(&full_bar[s], stage_tx);
          const size_t off = static_cast<size_t>(g * kGroupRows + m + i * kWarpsM) * kHidden;
          unsigned char *st = stages + s * L::kStageBytes;
          ptx::cp_async_bulk_tensor_1d_global_to_shared(
              reinterpret_cast<uint64_t *>(st), reinterpret_cast<const uint64_t *>(residual + off),
              kRowBytes, &full_bar[s]);
          if (has_add) {
            ptx::cp_async_bulk_tensor_1d_global_to_shared(
                reinterpret_cast<uint64_t *>(st + kRowBytes),
                reinterpret_cast<const uint64_t *>(x + off), kRowBytes, &full_bar[s]);
          }
        }
      }
    }
  } else {
    // Consumers: RMSNorm row code, then the group's RHT amax
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;
    const int c = warp_n * kWarp + lane;  // bidn == 0
    const int slot_barrier = 1 + warp_m;
    float2 *stats_slot = stats_smem + warp_m * 2 * kWarpsN;
    bool use0 = true;

    float gf[kElts];
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      const uint4 gv = __ldg(reinterpret_cast<const uint4 *>(gamma) + c + it * kVecColsPerLdg);
      const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
      for (int p = 0; p < 4; p++) {
        gf[it * kNumElts + 2 * p] = bf16lo(gw[p]);
        gf[it * kNumElts + 2 * p + 1] = bf16hi(gw[p]);
      }
    }
    uint32_t had_frag_t[4];
    get_hadamard_matrix_fragment<false, true, false, false>(nullptr, 0, had_frag_t,
                                                            random_sign_mask_t);
    constexpr float rn = 1.f / static_cast<float>(kHidden);

    uint32_t j = 0;
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
      const int row0 = g * kGroupRows;
#pragma unroll 1
      for (int i = 0; i < kRowsPerSlot; i++, j++) {
        const int lrow = warp_m + i * kWarpsM;
        const int row = row0 + lrow;
        const int s = warp_m * kStagesPerSlot + static_cast<int>(j % kStagesPerSlot);
        const uint32_t phase = (j / kStagesPerSlot) & 1u;
        const uint4 *st_r = reinterpret_cast<const uint4 *>(stages + s * L::kStageBytes);
        const uint4 *st_x = st_r + kVecCols;

        ptx::mbarrier_wait_parity(&full_bar[s], phase);
        uint4 xs[kLdgs];
#pragma unroll
        for (int it = 0; it < kLdgs; it++) {
          xs[it] = st_r[c + it * kVecColsPerLdg];
        }
        if (has_add) {
#pragma unroll
          for (int it = 0; it < kLdgs; it++) {
            const uint4 va = st_x[c + it * kVecColsPerLdg];
            uint32_t *xw = reinterpret_cast<uint32_t *>(&xs[it]);
            const uint32_t aw[4] = {va.x, va.y, va.z, va.w};
#pragma unroll
            for (int p = 0; p < 4; p++) {
              // residual + x: FP32 add of the BF16 operands, rounded to BF16
              xw[p] = pack_bf16x2_rn(bf16lo(xw[p]) + bf16lo(aw[p]), bf16hi(xw[p]) + bf16hi(aw[p]));
            }
          }
        }
        ptx::mbarrier_arrive(&empty_bar[s]);  // stage consumed (values are in registers)

        float xf[kElts];
#pragma unroll
        for (int it = 0; it < kLdgs; it++) {
          const uint32_t *xw = reinterpret_cast<const uint32_t *>(&xs[it]);
#pragma unroll
          for (int p = 0; p < 4; p++) {
            xf[it * kNumElts + 2 * p] = bf16lo(xw[p]);
            xf[it * kNumElts + 2 * p + 1] = bf16hi(xw[p]);
          }
        }

        const size_t base = static_cast<size_t>(row) * kVecCols + c;
        if (has_add) {
#pragma unroll
          for (int it = 0; it < kLdgs; it++) {
            reinterpret_cast<uint4 *>(sum_out)[base + it * kVecColsPerLdg] = xs[it];
          }
        }

        const float2 sts = block_stats(xf, stats_slot, use0, warp_n, lane, slot_barrier);
        const float mu = sts.x;
        const float m2 = sts.y;
        const float rs = rsqrtf(rn * m2 + mu * mu + epsilon);
        if (warp_n == 0 && lane == 0) {
          rsigma_out[row] = rs;
        }

#pragma unroll
        for (int it = 0; it < kLdgs; it++) {
          uint32_t zw[4];
#pragma unroll
          for (int p = 0; p < 4; p++) {
            const int e = it * kNumElts + 2 * p;
            // RMSNorm forward: y_ij = rs * x_ij; temp_output = g_ij * y_ij; output_t(temp_output)
            const float y0 = rs * (xf[e]);
            const float y1 = rs * (xf[e + 1]);
            const float t0 = gf[e] * y0;
            const float t1 = gf[e + 1] * y1;
            zw[p] = pack_bf16x2_rn(t0, t1);
            amax_row_reg = bf16x2_absmax(amax_row_reg, zw[p]);
          }
          const uint4 zv = make_uint4(zw[0], zw[1], zw[2], zw[3]);
          reinterpret_cast<uint4 *>(z_out)[base + it * kVecColsPerLdg] = zv;
          *reinterpret_cast<uint4 *>(tile + lrow * kTileStride +
                                     (c + it * kVecColsPerLdg) * kNumElts) = zv;
        }
      }
      ptx::numbered_barrier_sync(L::kConsumers, kGroupBarrier);

      // amax of bf16(RHT16 over the 16 rows) per column, with the HadamardAmaxTmaKernel wmma
      const int ld_row = lane % 16;
      const int ld_col = (lane / 16) * 8;
#pragma unroll 4
      for (int t = warp; t < kColTiles; t += kConsumerWarps) {
        __nv_bfloat16 *p = tile + ld_row * kTileStride + t * 16 + ld_col;
        uint32_t a0, a1, a2, a3;
        ldmatrix_x4_m8n8_shared_b16<true>(a0, a1, a2, a3, p);
        uint32_t c0, c1, c2, c3, tmp;
        mma_m16_n16_k16_b16_b16_b16_noacc<true>(a0, a2, a1, a3, had_frag_t[0], had_frag_t[1],
                                                had_frag_t[2], had_frag_t[3], c0, c1, c2, c3, tmp);
        amax_t_reg = bf16x2_absmax(amax_t_reg, tmp);
      }
      ptx::numbered_barrier_sync(L::kConsumers, kGroupBarrier);  // tile is reused by next group
    }
  }

  // CTA reduction, then the last CTA reduces all partials
  float amax_row;
  float amax_t;
  unpack_max_of_packed_bf16(amax_row_reg, amax_row);
  unpack_max_of_packed_bf16(amax_t_reg, amax_t);
  float wr = warp_max(amax_row);
  float wt = warp_max(amax_t);
  if (lane == 0) {
    red_smem[warp] = wr;
    red_smem[kNumWarps + warp] = wt;
  }
  __syncthreads();
  if (warp == 0) {
    float r = lane < kNumWarps ? red_smem[lane] : 0.f;
    float t = lane < kNumWarps ? red_smem[kNumWarps + lane] : 0.f;
    r = warp_max(r);
    t = warp_max(t);
    if (lane == 0) {
      partials[2 * blockIdx.x] = r;
      partials[2 * blockIdx.x + 1] = t;
      __threadfence();
      unsigned int prev = atomicAdd(counter, 1u);
      is_last_cta = (prev == gridDim.x - 1);
    }
  }
  __syncthreads();
  if (is_last_cta) {
    __threadfence();
    float r = 0.f;
    float t = 0.f;
    for (int b = tidx; b < gridDim.x; b += L::kThreads) {
      r = fmaxf(r, __ldcg(partials + 2 * b));
      t = fmaxf(t, __ldcg(partials + 2 * b + 1));
    }
    r = warp_max(r);
    t = warp_max(t);
    __syncthreads();
    if (lane == 0) {
      red_smem[warp] = r;
      red_smem[kNumWarps + warp] = t;
    }
    __syncthreads();
    if (warp == 0) {
      float rr = lane < kNumWarps ? red_smem[lane] : 0.f;
      float tt = lane < kNumWarps ? red_smem[kNumWarps + lane] : 0.f;
      rr = warp_max(rr);
      tt = warp_max(tt);
      if (lane == 0) {
        amax_out[0] = rr;
        amax_out[1] = tt;
        *counter = 0u;  // leave the workspace zero-filled for the next launch
      }
    }
  }
#else
  NVTE_DEVICE_ERROR("add_rmsnorm_fwd_rht_amax_kernel is only supported on SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

size_t workspace_bytes(const int multiprocessorCount) {
  return kWorkspaceCounterBytes + 2 * sizeof(float) * static_cast<size_t>(multiprocessorCount);
}

void check_bf16_rows(const Tensor &t, const char *name, const Shape &shape) {
  NVTE_CHECK(t.data.dtype == DType::kBFloat16, name, " must be a BF16 tensor, but dtype is ",
             to_string(t.data.dtype), ".");
  NVTE_CHECK(t.data.shape == shape, name, " has invalid shape (expected ", shape, ", got ",
             t.data.shape, ").");
  NVTE_CHECK(t.data.dptr != nullptr, name, " is not allocated.");
  NVTE_CHECK(reinterpret_cast<uintptr_t>(t.data.dptr) % 16 == 0, name, " must be 16 B aligned.");
}

}  // namespace add_rht_amax
}  // namespace normalization

void rmsnorm_fwd_add_rht_amax(const Tensor *x, const Tensor &residual, const Tensor &gamma,
                              const float epsilon, Tensor *sum, Tensor *z, Tensor *rsigma,
                              Tensor *amax, Tensor *workspace, const int multiprocessorCount,
                              const uint16_t random_sign_mask_t, cudaStream_t stream) {
  using namespace normalization::add_rht_amax;
  using L = Layout;

  NVTE_CHECK(multiprocessorCount > 0, "multiprocessorCount must be positive.");
  NVTE_CHECK(residual.data.shape.size() >= 2, "residual must have at least two dimensions.");
  const auto [rows, cols] = residual.flat_2d_dims();
  NVTE_CHECK(cols == static_cast<size_t>(kHidden), "Only a hidden size of ", kHidden,
             " is supported (got ", cols, ").");
  NVTE_CHECK(rows > 0 && rows % kGroupRows == 0, "The number of rows (", rows,
             ") must be a positive multiple of ", kGroupRows, ".");
  NVTE_CHECK(rows <= static_cast<size_t>(INT32_MAX), "Too many rows.");

  // Workspace size query
  const size_t ws_bytes = workspace_bytes(multiprocessorCount);
  if (workspace->data.dptr == nullptr) {
    workspace->data.shape = {ws_bytes};
    workspace->data.dtype = DType::kByte;
    return;
  }
  NVTE_CHECK(workspace->data.dtype == DType::kByte, "Workspace must be a byte tensor.");
  NVTE_CHECK(workspace->data.numel() >= ws_bytes, "Workspace is too small (expected ", ws_bytes,
             " bytes, got ", workspace->data.numel(), ").");

  const Shape &shape = residual.data.shape;
  check_bf16_rows(residual, "residual", shape);
  check_bf16_rows(*z, "z", shape);
  check_bf16_rows(gamma, "gamma", Shape{cols});
  const bool has_add = x != nullptr && x->data.dptr != nullptr;
  if (has_add) {
    check_bf16_rows(*x, "x", shape);
    NVTE_CHECK(sum != nullptr, "sum is required when x is given.");
    check_bf16_rows(*sum, "sum", shape);
  }
  NVTE_CHECK(rsigma->data.dtype == DType::kFloat32 && rsigma->data.shape == Shape{rows},
             "rsigma must be an FP32 tensor of shape ", Shape{rows}, ".");
  NVTE_CHECK(amax->data.dtype == DType::kFloat32 && amax->data.numel() >= 2,
             "amax must be an FP32 tensor with 2 elements.");
  NVTE_CHECK(epsilon >= 0.f, "Epsilon must be non-negative.");
  NVTE_CHECK(cuda::sm_arch(cuda::current_device()) >= 100,
             "rmsnorm_fwd_add_rht_amax requires SM 10.0+.");

  const int num_groups = static_cast<int>(rows / kGroupRows);
  const int grid = std::min(multiprocessorCount, num_groups);
  auto *ws = reinterpret_cast<unsigned char *>(workspace->data.dptr);
  auto *counter = reinterpret_cast<unsigned int *>(ws);
  auto *partials = reinterpret_cast<float *>(ws + kWorkspaceCounterBytes);

  auto kernel = add_rmsnorm_fwd_rht_amax_kernel;
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(L::kBytes)));
  kernel<<<grid, L::kThreads, L::kBytes, stream>>>(
      has_add ? reinterpret_cast<const __nv_bfloat16 *>(x->data.dptr) : nullptr,
      reinterpret_cast<const __nv_bfloat16 *>(residual.data.dptr),
      reinterpret_cast<const __nv_bfloat16 *>(gamma.data.dptr),
      has_add ? reinterpret_cast<__nv_bfloat16 *>(sum->data.dptr) : nullptr,
      reinterpret_cast<__nv_bfloat16 *>(z->data.dptr), reinterpret_cast<float *>(rsigma->data.dptr),
      reinterpret_cast<float *>(amax->data.dptr), partials, counter, static_cast<int>(rows),
      epsilon, random_sign_mask_t);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine

void nvte_rmsnorm_fwd_add_rht_amax(const NVTETensor x, const NVTETensor residual,
                                   const NVTETensor gamma, const float epsilon, NVTETensor sum,
                                   NVTETensor z, NVTETensor rsigma, NVTETensor amax,
                                   NVTETensor workspace, const int multiprocessorCount,
                                   const int random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(nvte_rmsnorm_fwd_add_rht_amax);
  using namespace transformer_engine;
  rmsnorm_fwd_add_rht_amax(convertNVTETensor(x), *convertNVTETensorCheck(residual),
                           *convertNVTETensorCheck(gamma), epsilon, convertNVTETensor(sum),
                           convertNVTETensorCheck(z), convertNVTETensorCheck(rsigma),
                           convertNVTETensorCheck(amax), convertNVTETensorCheck(workspace),
                           multiprocessorCount, static_cast<uint16_t>(random_sign_mask_t), stream);
}
