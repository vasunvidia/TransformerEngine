/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file amax_quantize_2d_nvfp4.cu
 *  \brief Per-tensor amax and NVFP4 2D (16x16 block) quantization in two persistent kernels.
 *
 *  Produces the same bytes as nvte_compute_amax_with_config followed by nvte_quantize_v2 with
 *  2D quantization (round-to-nearest, no RHT, E4M3 block scales with the default 448 bound):
 *  rowwise and columnwise FP4 data, compact or GEMM-swizzled scales, and both amaxes.
 *
 *  1. amax_kernel: max|x| over the BF16 input in 32 KiB chunks, one partial per CTA.
 *  2. quantize_kernel: 128x128 tiles, three consumer groups of four warps. Reduces the
 *     partials to the per-tensor amax, computes each 16x16 block scale, and writes the
 *     rowwise data, the columnwise data (the nibble transpose of the rowwise data, since 2D
 *     block scales are shared by both directions), both scale matrices (compact or directly
 *     in the GEMM-swizzled layout) and the amaxes.
 *
 *  Both kernels run one CTA per SM and stream the input through a six-stage ring of 32 KiB
 *  shared-memory buffers filled by TMA bulk copies issued by a producer warp. The quantize
 *  kernel visits the tiles in the reverse order of the amax kernel, which marks the tail of
 *  the input as L2 evict-last, so that the tail is re-read from L2.
 *
 *  The arithmetic follows core_nvfp4.cuh (compute_global_encode_scaling_factor_FP4,
 *  compute_decoding_scaling_factor) and quantize_transpose_nvfp4.cuh
 *  (quantize_transpose_nvfp4_2D_kernel) bit for bit.
 */

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>

#include "../../common.h"
#include "../../util/cuda_driver.h"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx.cuh"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace amax_quantize_2d {

constexpr int kTile = 128;                           // Quantize-kernel tile: 128x128 BF16
constexpr int kStageBytes = kTile * kTile * 2;       // One tile (32 KiB) per ring stage
constexpr int kStages = 6;                           // Ring depth
constexpr int kRingBytes = kStages * kStageBytes;    //
constexpr int kSmemBytes = kRingBytes + 1024;        // + alignment slack
constexpr int kAmaxConsumers = 256;                  // Amax kernel: consumer threads
constexpr int kAmaxThreads = kAmaxConsumers + 32;    //   + one producer warp
constexpr int kGroups = 3;                           // Quantize kernel: consumer groups
constexpr int kStagesPerGroup = kStages / kGroups;   //   each owning its ring stages
constexpr int kQuantConsumers = kGroups * 128;       //   of four warps
constexpr int kQuantThreads = kQuantConsumers + 32;  //   + one producer warp
constexpr int kColStageBytes = kTile * kTile / 2;    // Per group: columnwise FP4 tile
constexpr int kQuantSmemBytes = kSmemBytes + kGroups * kColStageBytes;
// The amax kernel marks the last kL2KeepBytes of the input L2 evict-last, as the quantize
// kernel reads them first.
constexpr size_t kL2KeepBytes = size_t{64} << 20;

static_assert(kStagesPerGroup * kGroups == kStages, "Stages must split evenly across groups");

#if FP4_TYPE_SUPPORTED

__device__ __forceinline__ uint8_t *align_smem_1024(uint8_t *p) {
  return reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(p) + 1023) & ~uintptr_t(1023));
}

__device__ __forceinline__ void init_ring(uint64_t *full, uint64_t *empty, uint32_t empty_count) {
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < kStages; ++i) {
      ptx::mbarrier_init(&full[i], 1);
      ptx::mbarrier_init(&empty[i], empty_count);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  __syncthreads();
}

// 1D bulk copy global -> shared with an L2 cache policy.
__device__ __forceinline__ void bulk_copy_g2s(void *dst, const void *src, uint32_t bytes,
                                              uint64_t *mbar, uint64_t policy) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1], %2, [%3], %4;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
      "l"(src), "r"(bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))), "l"(policy)
      : "memory");
#else
  NVTE_DEVICE_ERROR("bulk_copy_g2s is only supported on SM 10.0+.");
#endif
}

// 1D bulk copy global -> shared with the default L2 policy.
__device__ __forceinline__ void bulk_copy_g2s(void *dst, const void *src, uint32_t bytes,
                                              uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
      "l"(src), "r"(bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
      : "memory");
#else
  NVTE_DEVICE_ERROR("bulk_copy_g2s is only supported on SM 10.0+.");
#endif
}

// 2D TMA tile copy global -> shared with an L2 cache policy.
__device__ __forceinline__ void tma_2d_g2s(void *dst, const CUtensorMap *map, int x, int y,
                                           uint64_t *mbar, uint64_t policy) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(
          static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y),
      "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))), "l"(policy)
      : "memory");
#else
  NVTE_DEVICE_ERROR("tma_2d_g2s is only supported on SM 10.0+.");
#endif
}

// max(|a|, |b|) per BF16 lane. The sign bit is the xor of the signs; callers clear it.
__device__ __forceinline__ uint32_t abs_max_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t abs_max_bf16x8(uint32_t acc, const uint4 &v) {
  acc = abs_max_bf16x2(acc, v.x);
  acc = abs_max_bf16x2(acc, v.y);
  acc = abs_max_bf16x2(acc, v.z);
  acc = abs_max_bf16x2(acc, v.w);
  return acc;
}

__device__ __forceinline__ float abs_max_bf16x2_to_float(uint32_t acc) {
  const float lo = __uint_as_float((acc & 0x7fffu) << 16);
  const float hi = __uint_as_float(acc & 0x7fff0000u);
  return fmaxf(lo, hi);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, d));
  return v;
}

// Kernel 1: per-CTA max|x|. CTA b reduces the 32 KiB chunks b, b + grid, b + 2 * grid, ...
__global__ void __launch_bounds__(kAmaxThreads, 1)
    amax_kernel(const uint8_t *__restrict__ input, const int num_chunks, const int keep_from,
                float *__restrict__ partials, const float *__restrict__ noop) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) return;
  extern __shared__ uint8_t dshmem[];
  uint8_t *ring = align_smem_1024(dshmem);
  __shared__ __align__(8) uint64_t full[kStages], empty[kStages];
  __shared__ float red[kAmaxConsumers / 32];
  init_ring(full, empty, kAmaxConsumers);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int my_chunks =
      (num_chunks - 1 - static_cast<int>(blockIdx.x)) / static_cast<int>(gridDim.x) + 1;
  if (warp == kAmaxConsumers / 32) {  // Producer
    if (lane == 0) {
      for (int k = 0; k < my_chunks; ++k) {
        const int s = k % kStages;
        if (k >= kStages) ptx::mbarrier_wait_parity(&empty[s], ((k / kStages) - 1) & 1);
        ptx::mbarrier_arrive_expect_tx(&full[s], kStageBytes);
        const size_t c = static_cast<size_t>(blockIdx.x) + static_cast<size_t>(k) * gridDim.x;
        const uint8_t *src = input + c * kStageBytes;
        if (static_cast<int>(c) >= keep_from) {
          bulk_copy_g2s(ring + s * kStageBytes, src, kStageBytes, &full[s],
                        ptx::create_l2_policy_evict_last());
        } else {
          bulk_copy_g2s(ring + s * kStageBytes, src, kStageBytes, &full[s]);
        }
      }
    }
    return;
  }
  constexpr int kVecsPerThread = kStageBytes / 16 / kAmaxConsumers;
  uint32_t acc = 0;
  for (int k = 0; k < my_chunks; ++k) {
    const int s = k % kStages;
    ptx::mbarrier_wait_parity(&full[s], (k / kStages) & 1);
    const uint4 *b = reinterpret_cast<const uint4 *>(ring + s * kStageBytes);
    uint4 v[kVecsPerThread];
#pragma unroll
    for (int j = 0; j < kVecsPerThread; ++j) v[j] = b[threadIdx.x + j * kAmaxConsumers];
    // Order the generic-proxy reads before the producer's next TMA write into this stage.
    ptx::fence_proxy_async_shared_cta();
    ptx::mbarrier_arrive(&empty[s]);
#pragma unroll
    for (int j = 0; j < kVecsPerThread; ++j) acc = abs_max_bf16x8(acc, v[j]);
  }
  float m = warp_reduce_max(abs_max_bf16x2_to_float(acc));
  if (lane == 0) red[warp] = m;
  ptx::numbered_barrier_sync(kAmaxConsumers, 1);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 1; i < kAmaxConsumers / 32; ++i) m = fmaxf(m, red[i]);
    partials[blockIdx.x] = m;
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// compute_global_encode_scaling_factor_FP4 (core_nvfp4.cuh) for E4M3 scales (bound 448).
__device__ __forceinline__ float global_encode_scale(const float global_amax) {
  constexpr float fp8_max = 448.0f;
  constexpr float fp4_max = 6.0f;
  float s = fp8_max * fp4_max / global_amax;
  s = fminf(s, FLT_MAX);
  if (global_amax == 0.0f || s == 0.0f) return 1.0f;
  return s;
}

// Eight BF16 values (one uint4, element 0 in the low half of .x) times scale -> eight E2M1
// nibbles (element 0 in the low nibble): FP32 multiply, then cvt.rn.satfinite.e2m1x2.
__device__ __forceinline__ uint32_t mul_cvt_bf16x8_to_fp4x8(const uint4 &v, const float s) {
  uint32_t out = 0;
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    const float f0 = __uint_as_float(v.x << 16) * s, f1 = __uint_as_float(v.x & 0xffff0000u) * s;
    const float f2 = __uint_as_float(v.y << 16) * s, f3 = __uint_as_float(v.y & 0xffff0000u) * s;
    const float f4 = __uint_as_float(v.z << 16) * s, f5 = __uint_as_float(v.z & 0xffff0000u) * s;
    const float f6 = __uint_as_float(v.w << 16) * s, f7 = __uint_as_float(v.w & 0xffff0000u) * s;
    asm("{\n\t.reg.b8 b0, b1, b2, b3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n\t"
        "mov.b32 %0, {b0, b1, b2, b3};\n\t}"
        : "=r"(out)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3), "f"(f4), "f"(f5), "f"(f6), "f"(f7));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return out;
}

// In-place transpose of an 8x8 matrix of nibbles: r[i] nibble j -> r[j] nibble i.
__device__ __forceinline__ void transpose_8x8_nibbles(uint32_t *r) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t a = r[i], b = r[i + 4];
    r[i] = __byte_perm(a, b, 0x5410);
    r[i + 4] = __byte_perm(a, b, 0x7632);
  }
#pragma unroll
  for (int g = 0; g < 8; g += 4) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t a = r[g + i], b = r[g + i + 2];
      r[g + i] = __byte_perm(a, b, 0x6240);
      r[g + i + 2] = __byte_perm(a, b, 0x7351);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const uint32_t a = r[i], b = r[i + 1];
    r[i] = (a & 0x0f0f0f0fu) | ((b << 4) & 0xf0f0f0f0u);
    r[i + 1] = ((a >> 4) & 0x0f0f0f0fu) | (b & 0xf0f0f0f0u);
  }
}

// gemm_swizzled_scale_idx (swizzle.cuh): byte offset of the 16-byte row d3 of the 512-byte
// swizzled block (128-row block rb, 4-column group cbg).
__device__ __forceinline__ size_t swizzled_chunk_offset(size_t rb, size_t cbg, size_t cbg_cnt,
                                                        int d3) {
  return ((rb * cbg_cnt + cbg) * 32 + d3) * 16;
}

struct QuantizeParams {
  int rows, cols;
  const float *partials;
  int num_partials;
  uint8_t *row_data;
  uint8_t *row_scale;
  float *row_amax;
  uint8_t *col_data;
  uint8_t *col_scale;
  float *col_amax;
  int row_scale_stride;  // Compact layout: bytes per scale row
  int col_scale_stride;
  const float *noop;
  int tiles_x;
  int num_tiles;
};

// Tiles in the reverse order of the amax kernel's chunks.
__device__ __forceinline__ int tile_index(const QuantizeParams &p, int k) {
  return p.num_tiles - 1 - (static_cast<int>(blockIdx.x) + k * static_cast<int>(gridDim.x));
}

// Kernel 2. Producer warp: 2D TMA of 128x128 BF16 tiles into the ring. Consumer group g takes
// the CTA's tiles g, g + kGroups, ... and owns its own kStagesPerGroup ring stages. Within a
// group, warp w / half-warp h owns the 16-row strip 2w+h and lane % 16 the 8-column chunk,
// i.e. one 16x8 half of a 16x16 block per thread.
template <bool SWIZZLED, bool DO_COL>
__global__ void __launch_bounds__(kQuantThreads, 1)
    quantize_kernel(const __grid_constant__ CUtensorMap tensor_map, const QuantizeParams p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (p.noop != nullptr && p.noop[0] == 1.0f) return;
  extern __shared__ uint8_t dshmem[];
  uint8_t *ring = align_smem_1024(dshmem);
  uint8_t *col_stage_all = ring + kRingBytes;
  __shared__ __align__(8) uint64_t full[kStages], empty[kStages];
  __shared__ float red[kQuantConsumers / 32];
  __shared__ __align__(16) uint8_t sc[kGroups][2][8][8];   // [group][parity][strip][col block]
  __shared__ __align__(16) uint8_t scT[kGroups][2][8][8];  // [group][parity][col block][strip]
  init_ring(full, empty, 128);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int my_tiles =
      (p.num_tiles - 1 - static_cast<int>(blockIdx.x)) / static_cast<int>(gridDim.x) + 1;

  if (warp == kQuantConsumers / 32) {  // Producer
    if (lane == 0) {
      const uint64_t policy = ptx::create_l2_policy_evict_first(1.0f);
      for (int k = 0; k < my_tiles; ++k) {
        const int j = k / kGroups;  // The consumer group's j-th tile
        const int s = (k % kGroups) * kStagesPerGroup + j % kStagesPerGroup;
        if (j >= kStagesPerGroup) {
          ptx::mbarrier_wait_parity(&empty[s], ((j / kStagesPerGroup) - 1) & 1);
        }
        ptx::mbarrier_arrive_expect_tx(&full[s], kStageBytes);
        const int t = tile_index(p, k);
        tma_2d_g2s(ring + s * kStageBytes, &tensor_map, (t % p.tiles_x) * kTile,
                   (t / p.tiles_x) * kTile, &full[s], policy);
      }
    }
    return;
  }

  // Per-tensor amax from the partials of the amax kernel.
  float pm = 0.0f;
  for (int i = threadIdx.x; i < p.num_partials; i += kQuantConsumers) {
    pm = fmaxf(pm, p.partials[i]);
  }
  pm = warp_reduce_max(pm);
  if (lane == 0) red[warp] = pm;
  ptx::numbered_barrier_sync(kQuantConsumers, 1);
  float amax = red[0];
#pragma unroll
  for (int i = 1; i < kQuantConsumers / 32; ++i) amax = fmaxf(amax, red[i]);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *p.row_amax = amax;
    if (p.col_amax != nullptr) *p.col_amax = amax;
  }
  const float S_enc = global_encode_scale(amax);
  const float S_dec = 1.0 / S_enc;  // Division in double, as compute_decoding_scaling_factor
  constexpr float fp4_max_inv = 1.0f / 6.0f;
  const float S_enc_6 = S_enc * fp4_max_inv;

  const int g = warp >> 2;
  const int gt = threadIdx.x & 127;
  const int strip = (warp & 3) * 2 + (lane >> 4);  // 16-row strip of the tile, 0..7
  const int c8 = lane & 15;                        // 8-column chunk of the tile, 0..15
  const size_t cols = static_cast<size_t>(p.cols);
  const size_t rows_half = static_cast<size_t>(p.rows) / 2;

  int it = 0;
  for (int k = g; k < my_tiles; k += kGroups, ++it) {
    const int s = g * kStagesPerGroup + it % kStagesPerGroup;
    const int t = tile_index(p, k);
    const int row0 = (t / p.tiles_x) * kTile, col0 = (t % p.tiles_x) * kTile;
    ptx::mbarrier_wait_parity(&full[s], (it / kStagesPerGroup) & 1);
    const uint4 *src =
        reinterpret_cast<const uint4 *>(ring + s * kStageBytes) + strip * 16 * (kTile / 8) + c8;
    uint4 v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = src[i * (kTile / 8)];
    // Release the stage as soon as it is in registers. The proxy fence orders these
    // generic-proxy reads before the producer's next TMA write into the stage.
    ptx::fence_proxy_async_shared_cta();
    ptx::mbarrier_arrive(&empty[s]);

    // 16x16 block amax: this thread's 16x8 half and its neighbour's (lane ^ 1).
    uint32_t acc = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) acc = abs_max_bf16x8(acc, v[i]);
    float block_amax = abs_max_bf16x2_to_float(acc);
    block_amax = fmaxf(block_amax, __shfl_xor_sync(0xffffffffu, block_amax, 1));

    // compute_decoding_scaling_factor and the 2D kernel's block_scale_inverse.
    const float dec = block_amax * S_enc_6;
    const __nv_fp8_e4m3 s_fp8 = static_cast<__nv_fp8_e4m3>(fminf(dec, FLT_MAX));
    const float inv = fminf(1.0f / (static_cast<float>(s_fp8) * S_dec), FLT_MAX);

    uint32_t r[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = mul_cvt_bf16x8_to_fp4x8(v[i], inv);

    {  // Rowwise data [rows, cols / 2]: each half-warp row store is 64 contiguous bytes.
      uint32_t *dst = reinterpret_cast<uint32_t *>(p.row_data + (row0 + strip * 16) * (cols / 2) +
                                                   (col0 + c8 * 8) / 2);
#pragma unroll
      for (int i = 0; i < 16; ++i) dst[i * (cols / 8)] = r[i];
    }
    // Scale factors, staged per group and double-buffered by tile parity.
    uint8_t(*scb)[8] = sc[g][it & 1];
    uint8_t(*scTb)[8] = scT[g][it & 1];
    if ((c8 & 1) == 0) {
      const uint8_t bits = s_fp8.__x;
      scb[strip][c8 >> 1] = bits;
      scTb[c8 >> 1][strip] = bits;
    }
    if constexpr (DO_COL) {
      // Columnwise data [cols, rows / 2]: nibble i of the 8-byte word of column c is row
      // row0 + strip * 16 + i. The tile's [128 cols][64 B] is staged in shared memory so the
      // global stores are 16 B per lane. The 8-byte slot of strip s in column c is at slot
      // s ^ ((c >> 3) & 7), and the two half-warps write opposite column parities, to keep
      // shared-memory bank conflicts low.
      transpose_8x8_nibbles(r);
      transpose_8x8_nibbles(r + 8);
      uint8_t *cst = col_stage_all + g * kColStageBytes;
      ptx::numbered_barrier_sync(128, 2 + g);  // The group is done reading the previous tile
      const int hw = lane >> 4;
      const int slot = strip ^ (c8 & 7);
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) {
        const uint32_t lo = hw ? r[cc ^ 1] : r[cc];
        const uint32_t hi = hw ? r[8 + (cc ^ 1)] : r[8 + cc];
        *reinterpret_cast<uint2 *>(cst + (c8 * 8 + (cc ^ hw)) * 64 + slot * 8) = make_uint2(lo, hi);
      }
      ptx::numbered_barrier_sync(128, 2 + g);
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const int idx = q4 * 128 + gt;  // 16-byte chunk: column idx / 4, quarter idx % 4
        const int c = idx >> 2, q = idx & 3, kx = (c >> 3) & 7;
        uint4 w = *reinterpret_cast<const uint4 *>(cst + c * 64 + ((q ^ (kx >> 1)) << 4));
        if (kx & 1) w = make_uint4(w.z, w.w, w.x, w.y);
        *reinterpret_cast<uint4 *>(p.col_data + (col0 + c) * rows_half + row0 / 2 + q * 16) = w;
      }
    } else {
      ptx::numbered_barrier_sync(128, 2 + g);
    }
    if constexpr (SWIZZLED) {
      // Each tile covers whole 512-byte swizzled blocks: 128 rows x 2 groups of 4 scale
      // columns, for the rowwise and the columnwise matrix. Threads 0..63 write rowwise,
      // threads 64..127 columnwise.
      const int u = gt & 63;
      const int gg = u >> 5, d3 = u & 31, hb = d3 >> 4;
      if (gt < 64) {
        const size_t cbg_cnt = (cols / 16 + 3) / 4;
        uint4 w;
        w.x = *reinterpret_cast<const uint32_t *>(&scb[0 + hb][4 * gg]);
        w.y = *reinterpret_cast<const uint32_t *>(&scb[2 + hb][4 * gg]);
        w.z = *reinterpret_cast<const uint32_t *>(&scb[4 + hb][4 * gg]);
        w.w = *reinterpret_cast<const uint32_t *>(&scb[6 + hb][4 * gg]);
        *reinterpret_cast<uint4 *>(
            p.row_scale + swizzled_chunk_offset(row0 / 128, col0 / 64 + gg, cbg_cnt, d3)) = w;
      } else if (DO_COL) {
        const size_t cbg_cnt = (static_cast<size_t>(p.rows) / 16 + 3) / 4;
        uint4 w;
        w.x = *reinterpret_cast<const uint32_t *>(&scTb[0 + hb][4 * gg]);
        w.y = *reinterpret_cast<const uint32_t *>(&scTb[2 + hb][4 * gg]);
        w.z = *reinterpret_cast<const uint32_t *>(&scTb[4 + hb][4 * gg]);
        w.w = *reinterpret_cast<const uint32_t *>(&scTb[6 + hb][4 * gg]);
        *reinterpret_cast<uint4 *>(
            p.col_scale + swizzled_chunk_offset(col0 / 128, row0 / 64 + gg, cbg_cnt, d3)) = w;
      }
    } else {
      // Compact: rowwise [rows, stride], columnwise [cols, stride]; 8 scales per tile row.
      *reinterpret_cast<uint64_t *>(
          p.row_scale + static_cast<size_t>(row0 + gt) * p.row_scale_stride + col0 / 16) =
          *reinterpret_cast<const uint64_t *>(&scb[gt >> 4][0]);
      if constexpr (DO_COL) {
        *reinterpret_cast<uint64_t *>(
            p.col_scale + static_cast<size_t>(col0 + gt) * p.col_scale_stride + row0 / 16) =
            *reinterpret_cast<const uint64_t *>(&scTb[gt >> 4][0]);
      }
    }
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

#endif  // FP4_TYPE_SUPPORTED

inline size_t num_partials(const Tensor &input) {
  const size_t num_chunks = input.numel() * 2 / kStageBytes;
  return std::max<size_t>(1, std::min<size_t>(num_chunks, cuda::sm_count()));
}

inline bool shape_is(const SimpleTensor &t, size_t d0, size_t d1) {
  return t.shape.size() == 2 && t.shape[0] == d0 && t.shape[1] == d1;
}

bool is_supported(const Tensor &input, const Tensor &output) {
#if FP4_TYPE_SUPPORTED
  const int arch = cuda::sm_arch();
  if (arch < 100 || arch > 110) return false;
  if (cuda::max_shared_memory_per_block_optin() < static_cast<size_t>(kQuantSmemBytes) + 2048) {
    return false;
  }
  if (output.scaling_mode != NVTE_NVFP4_1D_SCALING || output.row_scaled_nvfp4) return false;
  if (input.dtype() != DType::kBFloat16 || input.data.dptr == nullptr || input.numel() == 0) {
    return false;
  }
  if (!is_aligned_ptr(input.data.dptr, TMA_GMEM_ALIGNMENT) || input.data.shape.size() < 2) {
    return false;
  }
  const auto [rows, cols] = input.flat_2d_dims();
  if (rows % kTile != 0 || cols % kTile != 0 || rows >= (size_t{1} << 31) ||
      cols >= (size_t{1} << 31) || input.numel() * 2 / kStageBytes >= (size_t{1} << 31)) {
    return false;
  }
  // Rowwise output is required; columnwise output is optional.
  if (!output.has_data() || output.data.dtype != DType::kFloat4E2M1 ||
      output.scale_inv.dptr == nullptr || output.scale_inv.dtype != DType::kFloat8E4M3 ||
      !shape_is(output.scale_inv, rows, cols / 16)) {
    return false;
  }
  if (output.has_columnwise_data() && (output.columnwise_data.dtype != DType::kFloat4E2M1 ||
                                       output.columnwise_scale_inv.dptr == nullptr ||
                                       output.columnwise_scale_inv.dtype != DType::kFloat8E4M3 ||
                                       !shape_is(output.columnwise_scale_inv, cols, rows / 16))) {
    return false;
  }
  // Per-tensor amax with the default E4M3 scale bound.
  if (output.amax.dptr == nullptr || output.amax.numel() != 1 ||
      (output.columnwise_amax.dptr != nullptr && output.columnwise_amax.numel() != 1)) {
    return false;
  }
  return output.get_nvfp4_scale_max() == 448;
#else
  return false;
#endif  // FP4_TYPE_SUPPORTED
}

#if FP4_TYPE_SUPPORTED
CUtensorMap make_tile_map(const Tensor &input, size_t rows, size_t cols) {
  // L2 256B promotion, unlike create_2D_tensor_map: the tile rows are 256-byte segments.
  static PFN_cuTensorMapEncodeTiled_v12000 encode = []() {
    void *ptr = cuda_driver::get_symbol("cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  CUtensorMap map{};
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * 2};
  const cuuint32_t box[2] = {kTile, kTile};
  const cuuint32_t elem_strides[2] = {1, 1};
  NVTE_CHECK_CUDA_DRIVER(encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, input.data.dptr, dims,
                                strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

// Kernel 2: quantize, with the per-tensor amax the max of partials[0, num_partials).
void launch_quantize(const Tensor &input, Tensor *output, const float *partials, int num_partials,
                     const float *noop, cudaStream_t stream) {
  const auto [rows, cols] = input.flat_2d_dims();
  const bool do_col = output->has_columnwise_data();
  QuantizeParams p;
  p.rows = static_cast<int>(rows);
  p.cols = static_cast<int>(cols);
  p.partials = partials;
  p.num_partials = num_partials;
  p.row_data = reinterpret_cast<uint8_t *>(output->data.dptr);
  p.row_scale = reinterpret_cast<uint8_t *>(output->scale_inv.dptr);
  p.row_amax = reinterpret_cast<float *>(output->amax.dptr);
  p.col_data = do_col ? reinterpret_cast<uint8_t *>(output->columnwise_data.dptr) : nullptr;
  p.col_scale = do_col ? reinterpret_cast<uint8_t *>(output->columnwise_scale_inv.dptr) : nullptr;
  // As nvte_compute_amax_with_config, write the columnwise amax whenever it is allocated.
  p.col_amax = reinterpret_cast<float *>(output->columnwise_amax.dptr);
  p.row_scale_stride = static_cast<int>(output->scale_inv.shape[1]);
  p.col_scale_stride = do_col ? static_cast<int>(output->columnwise_scale_inv.shape[1]) : 0;
  p.noop = noop;
  p.tiles_x = static_cast<int>(cols / kTile);
  p.num_tiles = static_cast<int>((rows / kTile) * (cols / kTile));

  checkCuDriverContext(stream);
  const CUtensorMap tensor_map = make_tile_map(input, rows, cols);
  const unsigned grid = static_cast<unsigned>(std::min(p.num_tiles, cuda::sm_count()));
  auto launch = [&](auto kernel) {
    NVTE_CHECK_CUDA(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kQuantSmemBytes));
    kernel<<<grid, kQuantThreads, kQuantSmemBytes, stream>>>(tensor_map, p);
  };
  const bool swizzled = output->with_gemm_swizzled_scales;
  if (swizzled) {
    if (do_col) {
      launch(quantize_kernel<true, true>);
    } else {
      launch(quantize_kernel<true, false>);
    }
  } else {
    if (do_col) {
      launch(quantize_kernel<false, true>);
    } else {
      launch(quantize_kernel<false, false>);
    }
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

const float *noop_ptr(const Tensor *noop) {
  return (noop != nullptr && noop->data.dptr != nullptr)
             ? reinterpret_cast<const float *>(noop->data.dptr)
             : nullptr;
}
#endif  // FP4_TYPE_SUPPORTED

void amax_quantize(const Tensor &input, Tensor *output, Tensor *workspace, const Tensor *noop,
                   cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_CHECK(is_supported(input, *output),
             "nvte_nvfp4_2d_amax_quantize does not support this input/output combination; "
             "check nvte_nvfp4_2d_amax_quantize_supported first.");
  const size_t partials_count = num_partials(input);
  if (workspace->data.dptr == nullptr) {
    workspace->data.shape = {partials_count};
    workspace->data.dtype = DType::kFloat32;
    return;
  }
  NVTE_CHECK(workspace->data.dtype == DType::kFloat32 && workspace->numel() >= partials_count,
             "Workspace must be a float32 tensor with at least ", partials_count, " elements.");
  float *partials = reinterpret_cast<float *>(workspace->data.dptr);

  // Kernel 1: amax partials.
  const int num_chunks = static_cast<int>(input.numel() * 2 / kStageBytes);
  const int keep_chunks =
      static_cast<int>(std::min<size_t>(num_chunks, kL2KeepBytes / kStageBytes));
  NVTE_CHECK_CUDA(
      cudaFuncSetAttribute(amax_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  amax_kernel<<<static_cast<unsigned>(partials_count), kAmaxThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const uint8_t *>(input.data.dptr), num_chunks, num_chunks - keep_chunks,
      partials, noop_ptr(noop));
  NVTE_CHECK_CUDA(cudaGetLastError());

  launch_quantize(input, output, partials, static_cast<int>(partials_count), noop_ptr(noop),
                  stream);
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

void quantize_with_amax(const Tensor &input, const Tensor &amax, Tensor *output, const Tensor *noop,
                        cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_CHECK(is_supported(input, *output),
             "nvte_nvfp4_2d_quantize_with_amax does not support this input/output combination; "
             "check nvte_nvfp4_2d_amax_quantize_supported first.");
  NVTE_CHECK(amax.data.dptr != nullptr && amax.dtype() == DType::kFloat32 && amax.numel() == 1,
             "nvte_nvfp4_2d_quantize_with_amax expects a float32 amax with one element.");
  // The amax is the quantize kernel's single partial.
  launch_quantize(input, output, reinterpret_cast<const float *>(amax.data.dptr), 1, noop_ptr(noop),
                  stream);
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

// Sharded 2D quantization (nvte_nvfp4_2d_shard_*): each data-parallel rank quantizes the weight
// elements of its shard into its chunk of a gathered buffer, and the gathered chunks are expanded
// into NVFP4 tensors. The arithmetic is quantize_kernel's.
constexpr int kMaxShardRanks = 64;
constexpr int kShardPackThreads = 128;  // Eight half-warps, one 16x128 unit each
constexpr int kShardPackCtasPerSm = 8;
constexpr int kShardUnpackThreads = 128;
constexpr int kShardUnpackCtasPerSm = 32;

struct ShardUnpackParams {
  const uint8_t *gathered;
  int64_t chunk_bytes;
  int64_t shard;
  int64_t param_start;
  int64_t scale_base[kMaxShardRanks];
  const int32_t *amax;
  int rows, cols;
  uint8_t *row_data;
  uint8_t *row_scale;
  float *row_amax;
  uint8_t *col_data;
  uint8_t *col_scale;
  float *col_amax;
  int row_scale_stride, col_scale_stride;
  const float *noop;
  int tiles_x, num_tiles;
};

#if FP4_TYPE_SUPPORTED

// One CTA per entry; thread t reduces the owned elements of 16x16 blocks t, t + blockDim.x, ...
__global__ void shard_partial_amax_kernel(const NVTENVFP4ShardPartialEntry *__restrict__ entries,
                                          int32_t *__restrict__ amax) {
  const NVTENVFP4ShardPartialEntry e = entries[blockIdx.x];
  const uint16_t *x = reinterpret_cast<const uint16_t *>(e.row_group);
  for (int cb = threadIdx.x; cb < e.cols / 16; cb += blockDim.x) {
    uint32_t m = 0;
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        const int64_t idx = static_cast<int64_t>(i) * e.cols + cb * 16 + j;
        if (idx >= e.own_begin && idx < e.own_end)
          m = max(m, static_cast<uint32_t>(x[idx] & 0x7fffu));
      }
    }
    amax[e.slot + cb] = static_cast<int32_t>(m << 16);
  }
}

// A half-warp quantizes one 16x128 unit (16 rows of a row group, 128 columns); lane c8 of the
// half-warp one 16x8 half of a 16x16 block, as in quantize_kernel. Each half-warp takes a
// contiguous range of units, and all threads run the same number of iterations (the block amax
// shuffle is warp-wide).
__global__ void __launch_bounds__(kShardPackThreads)
    shard_pack_kernel(const NVTENVFP4ShardPackItem *__restrict__ items, int num_items,
                      int64_t num_units, const int32_t *__restrict__ amax_buf) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int hw = threadIdx.x >> 4, c8 = threadIdx.x & 15;
  const int hw_per_cta = blockDim.x >> 4;
  const int64_t num_hw = static_cast<int64_t>(gridDim.x) * hw_per_cta;
  const int64_t per_hw = (num_units + num_hw - 1) / num_hw;
  const int64_t u_begin = (static_cast<int64_t>(blockIdx.x) * hw_per_cta + hw) * per_hw;
  // Last item whose first unit is at or before u_begin.
  int cur = 0;
  {
    int lo = 0, hi = num_items - 1;
    const int64_t ub = u_begin < num_units ? u_begin : num_units - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (items[mid].first_unit <= ub) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    cur = lo;
  }
  NVTENVFP4ShardPackItem it = items[cur];
  int64_t next_first = cur + 1 < num_items ? items[cur + 1].first_unit : INT64_MAX;
  for (int64_t j = 0; j < per_hw; ++j) {
    const bool valid = u_begin + j < num_units;
    const int64_t u = valid ? u_begin + j : num_units - 1;
    while (u >= next_first) {
      ++cur;
      it = items[cur];
      next_first = cur + 1 < num_items ? items[cur + 1].first_unit : INT64_MAX;
    }
    const int units_per_row_group = it.cols / 128;
    const int64_t local = u - it.first_unit;
    const int rg = it.row_group_first + static_cast<int>(local / units_per_row_group);
    const int strip = static_cast<int>(local % units_per_row_group);
    const int64_t cols = it.cols;
    const int64_t e0 = static_cast<int64_t>(rg) * 16 * cols + strip * 128 + c8 * 8;
    const uint16_t *x = reinterpret_cast<const uint16_t *>(it.weight);
    uint4 v[16];
    bool own[16];
    uint32_t acc = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int64_t e = e0 + i * cols;
      own[i] = valid && e >= it.own_begin && e < it.own_end;
      v[i] = own[i] ? *reinterpret_cast<const uint4 *>(x + e) : make_uint4(0, 0, 0, 0);
      acc = abs_max_bf16x8(acc, v[i]);
    }
    float block_amax = abs_max_bf16x2_to_float(acc);
    block_amax = fmaxf(block_amax, __shfl_xor_sync(0xffffffffu, block_amax, 1));
    // A row group cut by the shard's first or last boundary takes its block amaxes over the
    // whole row group from the reduced partials.
    int cut = -1;
    if (rg == it.row_group_first && it.cut_begin_slot >= 0) cut = it.cut_begin_slot;
    if (rg == it.row_group_last && it.cut_end_slot >= 0) cut = it.cut_end_slot;
    if (cut >= 0) block_amax = __int_as_float(amax_buf[cut + strip * 8 + (c8 >> 1)]);
    const float S_enc = global_encode_scale(__int_as_float(amax_buf[it.amax_slot]));
    const float S_dec = 1.0 / S_enc;  // Division in double, as compute_decoding_scaling_factor
    constexpr float fp4_max_inv = 1.0f / 6.0f;
    const float dec = block_amax * (S_enc * fp4_max_inv);
    const __nv_fp8_e4m3 s_fp8 = static_cast<__nv_fp8_e4m3>(fminf(dec, FLT_MAX));
    const float inv = fminf(1.0f / (static_cast<float>(s_fp8) * S_dec), FLT_MAX);
    uint8_t *chunk = reinterpret_cast<uint8_t *>(it.chunk);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (own[i]) {
        const int64_t e = e0 + i * cols + it.data_offset;
        *reinterpret_cast<uint32_t *>(chunk + e / 2) = mul_cvt_bf16x8_to_fp4x8(v[i], inv);
      }
    }
    if (valid && (c8 & 1) == 0 && rg >= it.row_group_own_begin && rg < it.row_group_own_end) {
      chunk[it.scale_offset + static_cast<int64_t>(rg - it.row_group_own_begin) * (cols / 16) +
            strip * 8 + (c8 >> 1)] = s_fp8.__x;
    }
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// One 128x128 tile per iteration; thread layout and output stores as in quantize_kernel.
template <bool SWIZZLED, bool DO_COL>
__global__ void __launch_bounds__(kShardUnpackThreads)
    shard_unpack_kernel(const __grid_constant__ ShardUnpackParams p) {
  if (p.noop != nullptr && p.noop[0] == 1.0f) return;
  __shared__ __align__(16) uint8_t col_stage[kTile * kTile / 2];
  __shared__ __align__(16) uint8_t sc[8][8];
  __shared__ __align__(16) uint8_t scT[8][8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gt = threadIdx.x;
  const int strip = (warp & 3) * 2 + (lane >> 4);
  const int c8 = lane & 15;
  const int64_t cols = p.cols;
  const size_t rows_half = static_cast<size_t>(p.rows) / 2;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const float a = __int_as_float(p.amax[0]);
    *p.row_amax = a;
    if (p.col_amax != nullptr) *p.col_amax = a;
  }
  for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
    const int row0 = (t / p.tiles_x) * kTile, col0 = (t % p.tiles_x) * kTile;
    __syncthreads();  // The previous tile's shared-memory readers are done
    uint32_t r[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int64_t e =
          p.param_start + static_cast<int64_t>(row0 + strip * 16 + i) * cols + col0 + c8 * 8;
      const int64_t k = e / p.shard;
      r[i] = *reinterpret_cast<const uint32_t *>(p.gathered + k * p.chunk_bytes +
                                                 (e - k * p.shard) / 2);
    }
    const int rg = row0 / 16 + strip;
    const int64_t k = (p.param_start + static_cast<int64_t>(rg) * 16 * cols) / p.shard;
    const uint8_t bits = p.gathered[k * p.chunk_bytes + p.scale_base[k] +
                                    static_cast<int64_t>(rg) * (cols / 16) + col0 / 16 + (c8 >> 1)];
    {
      uint32_t *dst = reinterpret_cast<uint32_t *>(p.row_data + (row0 + strip * 16) * (cols / 2) +
                                                   (col0 + c8 * 8) / 2);
#pragma unroll
      for (int i = 0; i < 16; ++i) dst[i * (cols / 8)] = r[i];
    }
    if ((c8 & 1) == 0) {
      sc[strip][c8 >> 1] = bits;
      scT[c8 >> 1][strip] = bits;
    }
    if constexpr (DO_COL) {
      transpose_8x8_nibbles(r);
      transpose_8x8_nibbles(r + 8);
      __syncthreads();
      const int hw = lane >> 4;
      const int slot = strip ^ (c8 & 7);
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) {
        const uint32_t lo = hw ? r[cc ^ 1] : r[cc];
        const uint32_t hi = hw ? r[8 + (cc ^ 1)] : r[8 + cc];
        *reinterpret_cast<uint2 *>(col_stage + (c8 * 8 + (cc ^ hw)) * 64 + slot * 8) =
            make_uint2(lo, hi);
      }
      __syncthreads();
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const int idx = q4 * 128 + gt;
        const int c = idx >> 2, q = idx & 3, kx = (c >> 3) & 7;
        uint4 w = *reinterpret_cast<const uint4 *>(col_stage + c * 64 + ((q ^ (kx >> 1)) << 4));
        if (kx & 1) w = make_uint4(w.z, w.w, w.x, w.y);
        *reinterpret_cast<uint4 *>(p.col_data + (col0 + c) * rows_half + row0 / 2 + q * 16) = w;
      }
    } else {
      __syncthreads();
    }
    if constexpr (SWIZZLED) {
      const int u = gt & 63;
      const int gg = u >> 5, d3 = u & 31, hb = d3 >> 4;
      if (gt < 64) {
        const size_t cbg_cnt = (cols / 16 + 3) / 4;
        uint4 w;
        w.x = *reinterpret_cast<const uint32_t *>(&sc[0 + hb][4 * gg]);
        w.y = *reinterpret_cast<const uint32_t *>(&sc[2 + hb][4 * gg]);
        w.z = *reinterpret_cast<const uint32_t *>(&sc[4 + hb][4 * gg]);
        w.w = *reinterpret_cast<const uint32_t *>(&sc[6 + hb][4 * gg]);
        *reinterpret_cast<uint4 *>(
            p.row_scale + swizzled_chunk_offset(row0 / 128, col0 / 64 + gg, cbg_cnt, d3)) = w;
      } else if (DO_COL) {
        const size_t cbg_cnt = (static_cast<size_t>(p.rows) / 16 + 3) / 4;
        uint4 w;
        w.x = *reinterpret_cast<const uint32_t *>(&scT[0 + hb][4 * gg]);
        w.y = *reinterpret_cast<const uint32_t *>(&scT[2 + hb][4 * gg]);
        w.z = *reinterpret_cast<const uint32_t *>(&scT[4 + hb][4 * gg]);
        w.w = *reinterpret_cast<const uint32_t *>(&scT[6 + hb][4 * gg]);
        *reinterpret_cast<uint4 *>(
            p.col_scale + swizzled_chunk_offset(col0 / 128, row0 / 64 + gg, cbg_cnt, d3)) = w;
      }
    } else {
      *reinterpret_cast<uint64_t *>(
          p.row_scale + static_cast<size_t>(row0 + gt) * p.row_scale_stride + col0 / 16) =
          *reinterpret_cast<const uint64_t *>(&sc[gt >> 4][0]);
      if constexpr (DO_COL) {
        *reinterpret_cast<uint64_t *>(
            p.col_scale + static_cast<size_t>(col0 + gt) * p.col_scale_stride + row0 / 16) =
            *reinterpret_cast<const uint64_t *>(&scT[gt >> 4][0]);
      }
    }
  }
}

#endif  // FP4_TYPE_SUPPORTED

void shard_partial_amax(const NVTENVFP4ShardPartialEntry *entries, size_t num_entries,
                        int32_t *amax, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  if (num_entries == 0) return;
  NVTE_CHECK(entries != nullptr && amax != nullptr, "Entries and amax must be device pointers.");
  shard_partial_amax_kernel<<<static_cast<unsigned>(num_entries), 128, 0, stream>>>(entries, amax);
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

void shard_pack(const NVTENVFP4ShardPackItem *items, size_t num_items, int64_t num_units,
                const int32_t *amax, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  if (num_items == 0 || num_units == 0) return;
  NVTE_CHECK(items != nullptr && amax != nullptr, "Items and amax must be device pointers.");
  NVTE_CHECK(num_items < (size_t{1} << 31), "Too many items.");
  const int64_t ctas = std::min<int64_t>(
      (num_units + 15) / 16, static_cast<int64_t>(cuda::sm_count()) * kShardPackCtasPerSm);
  shard_pack_kernel<<<static_cast<unsigned>(ctas), kShardPackThreads, 0, stream>>>(
      items, static_cast<int>(num_items), num_units, amax);
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

void shard_unpack(const uint8_t *gathered, int64_t chunk_bytes, int64_t shard, int64_t param_start,
                  const int64_t *scale_base, size_t num_ranks, const int32_t *amax, size_t rows,
                  size_t cols, Tensor *output, const Tensor *noop, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_CHECK(num_ranks >= 1 && num_ranks <= static_cast<size_t>(kMaxShardRanks),
             "nvte_nvfp4_2d_shard_unpack supports 1 to ", kMaxShardRanks, " ranks.");
  NVTE_CHECK(gathered != nullptr && amax != nullptr && scale_base != nullptr,
             "Gathered buffer, amax and scale offsets must be given.");
  NVTE_CHECK(chunk_bytes > 0 && shard > 0 && shard % 16 == 0 && param_start >= 0,
             "Invalid shard layout.");
  NVTE_CHECK(rows % kTile == 0 && cols % kTile == 0 && rows < (size_t{1} << 31) &&
                 cols < (size_t{1} << 31),
             "Rows and columns must be multiples of ", kTile, ".");
  NVTE_CHECK(output->scaling_mode == NVTE_NVFP4_1D_SCALING && !output->row_scaled_nvfp4,
             "Output must be an NVFP4 tensor.");
  NVTE_CHECK(output->has_data() && output->data.dtype == DType::kFloat4E2M1 &&
                 output->data.numel() == rows * cols && output->scale_inv.dptr != nullptr &&
                 output->scale_inv.dtype == DType::kFloat8E4M3 &&
                 shape_is(output->scale_inv, rows, cols / 16),
             "Output rowwise data and scales do not match a [", rows, ", ", cols, "] weight.");
  const bool do_col = output->has_columnwise_data();
  NVTE_CHECK(!do_col || (output->columnwise_data.dtype == DType::kFloat4E2M1 &&
                         output->columnwise_data.numel() == rows * cols &&
                         output->columnwise_scale_inv.dptr != nullptr &&
                         output->columnwise_scale_inv.dtype == DType::kFloat8E4M3 &&
                         shape_is(output->columnwise_scale_inv, cols, rows / 16)),
             "Output columnwise data and scales do not match a [", rows, ", ", cols, "] weight.");
  NVTE_CHECK(output->amax.dptr != nullptr && output->amax.numel() == 1 &&
                 (output->columnwise_amax.dptr == nullptr || output->columnwise_amax.numel() == 1),
             "Output must have a per-tensor amax.");
  ShardUnpackParams p;
  p.gathered = gathered;
  p.chunk_bytes = chunk_bytes;
  p.shard = shard;
  p.param_start = param_start;
  for (int i = 0; i < kMaxShardRanks; ++i) {
    p.scale_base[i] = i < static_cast<int>(num_ranks) ? scale_base[i] : 0;
  }
  p.amax = amax;
  p.rows = static_cast<int>(rows);
  p.cols = static_cast<int>(cols);
  p.row_data = reinterpret_cast<uint8_t *>(output->data.dptr);
  p.row_scale = reinterpret_cast<uint8_t *>(output->scale_inv.dptr);
  p.row_amax = reinterpret_cast<float *>(output->amax.dptr);
  p.col_data = do_col ? reinterpret_cast<uint8_t *>(output->columnwise_data.dptr) : nullptr;
  p.col_scale = do_col ? reinterpret_cast<uint8_t *>(output->columnwise_scale_inv.dptr) : nullptr;
  p.col_amax = reinterpret_cast<float *>(output->columnwise_amax.dptr);
  p.row_scale_stride = static_cast<int>(output->scale_inv.shape[1]);
  p.col_scale_stride = do_col ? static_cast<int>(output->columnwise_scale_inv.shape[1]) : 0;
  p.noop = noop_ptr(noop);
  p.tiles_x = static_cast<int>(cols / kTile);
  p.num_tiles = static_cast<int>((rows / kTile) * (cols / kTile));
  const unsigned grid = static_cast<unsigned>(std::min<int64_t>(
      p.num_tiles, static_cast<int64_t>(cuda::sm_count()) * kShardUnpackCtasPerSm));
  auto launch = [&](auto kernel) { kernel<<<grid, kShardUnpackThreads, 0, stream>>>(p); };
  if (output->with_gemm_swizzled_scales) {
    if (do_col) {
      launch(shard_unpack_kernel<true, true>);
    } else {
      launch(shard_unpack_kernel<true, false>);
    }
  } else {
    if (do_col) {
      launch(shard_unpack_kernel<false, true>);
    } else {
      launch(shard_unpack_kernel<false, false>);
    }
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace amax_quantize_2d
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

int nvte_nvfp4_2d_amax_quantize_supported(const NVTETensor input, const NVTETensor output) {
  using namespace transformer_engine;
  return dispatch::nvfp4::amax_quantize_2d::is_supported(*convertNVTETensorCheck(input),
                                                         *convertNVTETensorCheck(output))
             ? 1
             : 0;
}

void nvte_nvfp4_2d_amax_quantize(const NVTETensor input, NVTETensor output, NVTETensor workspace,
                                 const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_2d_amax_quantize);
  using namespace transformer_engine;
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }
  NVTE_CHECK(!quant_config_cpp.stochastic_rounding,
             "nvte_nvfp4_2d_amax_quantize does not support stochastic rounding.");
  NVTE_CHECK(quant_config_cpp.nvfp4_4over6_mode == kNVTENVFP44Over6Disabled,
             "nvte_nvfp4_2d_amax_quantize does not support 4over6 quantization.");
  const Tensor *noop = nullptr;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }
  dispatch::nvfp4::amax_quantize_2d::amax_quantize(*convertNVTETensorCheck(input),
                                                   convertNVTETensorCheck(output),
                                                   convertNVTETensorCheck(workspace), noop, stream);
}

void nvte_nvfp4_2d_quantize_with_amax(const NVTETensor input, const NVTETensor amax,
                                      NVTETensor output, const NVTEQuantizationConfig quant_config,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_2d_quantize_with_amax);
  using namespace transformer_engine;
  const Tensor *noop = nullptr;
  if (quant_config != nullptr) {
    const auto &config = *reinterpret_cast<QuantizationConfig *>(quant_config);
    NVTE_CHECK(!config.stochastic_rounding,
               "nvte_nvfp4_2d_quantize_with_amax does not support stochastic rounding.");
    NVTE_CHECK(config.nvfp4_4over6_mode == kNVTENVFP44Over6Disabled,
               "nvte_nvfp4_2d_quantize_with_amax does not support 4over6 quantization.");
    if (config.noop_tensor != nullptr) noop = convertNVTETensorCheck(config.noop_tensor);
  }
  dispatch::nvfp4::amax_quantize_2d::quantize_with_amax(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(amax), convertNVTETensorCheck(output),
      noop, stream);
}

void nvte_nvfp4_2d_shard_partial_amax(const NVTENVFP4ShardPartialEntry *entries, size_t num_entries,
                                      int32_t *amax, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_2d_shard_partial_amax);
  transformer_engine::dispatch::nvfp4::amax_quantize_2d::shard_partial_amax(entries, num_entries,
                                                                            amax, stream);
}

void nvte_nvfp4_2d_shard_pack(const NVTENVFP4ShardPackItem *items, size_t num_items,
                              int64_t num_units, const int32_t *amax, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_2d_shard_pack);
  transformer_engine::dispatch::nvfp4::amax_quantize_2d::shard_pack(items, num_items, num_units,
                                                                    amax, stream);
}

void nvte_nvfp4_2d_shard_unpack(const uint8_t *gathered, int64_t chunk_bytes, int64_t shard,
                                int64_t param_start, const int64_t *scale_base, size_t num_ranks,
                                const int32_t *amax, size_t rows, size_t cols, NVTETensor output,
                                const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_2d_shard_unpack);
  using namespace transformer_engine;
  const Tensor *noop = nullptr;
  if (quant_config != nullptr) {
    const auto &config = *reinterpret_cast<QuantizationConfig *>(quant_config);
    if (config.noop_tensor != nullptr) noop = convertNVTETensorCheck(config.noop_tensor);
  }
  dispatch::nvfp4::amax_quantize_2d::shard_unpack(gathered, chunk_bytes, shard, param_start,
                                                  scale_base, num_ranks, amax, rows, cols,
                                                  convertNVTETensorCheck(output), noop, stream);
}
