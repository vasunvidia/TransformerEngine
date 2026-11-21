/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_2d_nvfp4.cuh
 *  \brief Grouped NVFP4 cast with 2D (16x16) block scaling: row-wise and transposed outputs of many
 *         tensors in one launch.
 *
 *  The input is one BF16 matrix whose rows are split into tensors (split_sections); every tensor
 *  is quantized like nvte_quantize_v2 with nvfp4_2d_quantization would quantize it on its own:
 *  16x16 block amax, e4m3 block decode scale from the tensor's global amax, round-to-nearest
 *  e2m1, a row-wise [rows, cols/2] data buffer with [rows, cols/16] scales and, when requested,
 *  the transposed [cols, rows/2] data with [cols, rows/16] scales. The block arithmetic is
 *  core::compute_decoding_scaling_factor and ptx::mul_cvt_bf16_to_fp4_4x, the helpers the
 *  single-tensor kernels use, so the bytes are identical to the per-tensor cast.
 *
 *  Every output tensor brings its own data / scale / amax pointers (the tensors need not be laid
 *  out at a regular stride), and the amax of every tensor has to be computed before the launch
 *  (nvte_group_amax). Scales are in the unswizzled (compact) layout.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_2D_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_2D_NVFP4_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <vector>

#include "../../common.h"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace group_quantize_2d_kernel {

using namespace ptx;

constexpr int kMaxTensors2D = 64;  // the argument struct stays well below the 4 KB parameter limit
constexpr int kBlock = 16;         // 16x16 scaling blocks
constexpr int kTile = 64;          // a CTA quantizes a 64 x 64 tile: 4 x 4 scaling blocks
constexpr int kThreads = 256;
constexpr int kPitch = kTile + 8;  // shared-memory row pitch in elements (16-byte multiple)

struct Args {
  const bf16 *input;
  const float *noop;
  int num_tensors;
  int cols;
  int max_row_tiles;
  // Per tensor
  size_t input_row_offset[kMaxTensors2D];  // first row of the tensor in the input matrix
  int rows[kMaxTensors2D];
  uint8_t *rowwise_data[kMaxTensors2D];
  void *rowwise_scale[kMaxTensors2D];
  int rowwise_scale_stride[kMaxTensors2D];
  uint8_t *colwise_data[kMaxTensors2D];
  void *colwise_scale[kMaxTensors2D];
  int colwise_scale_stride[kMaxTensors2D];
  const float *rowwise_amax[kMaxTensors2D];
  const float *colwise_amax[kMaxTensors2D];
};

// Packs 16 BF16 values (as two 128-bit halves) into 8 FP4 bytes: the helper the single-tensor
// kernels use, four elements at a time.
__device__ __forceinline__ uint2 cast_16(const uint4 lo, const uint4 hi, const float2 scale_2x) {
  const uint64_t e0 = (static_cast<uint64_t>(lo.y) << 32) | lo.x;
  const uint64_t e1 = (static_cast<uint64_t>(lo.w) << 32) | lo.z;
  const uint64_t e2 = (static_cast<uint64_t>(hi.y) << 32) | hi.x;
  const uint64_t e3 = (static_cast<uint64_t>(hi.w) << 32) | hi.z;
  const fp4e2m1x4 q0 = ptx::mul_cvt_bf16_to_fp4_4x<false>(e0, scale_2x, 0);
  const fp4e2m1x4 q1 = ptx::mul_cvt_bf16_to_fp4_4x<false>(e1, scale_2x, 0);
  const fp4e2m1x4 q2 = ptx::mul_cvt_bf16_to_fp4_4x<false>(e2, scale_2x, 0);
  const fp4e2m1x4 q3 = ptx::mul_cvt_bf16_to_fp4_4x<false>(e3, scale_2x, 0);
  uint2 out;
  out.x = static_cast<uint32_t>(reinterpret_cast<const uint16_t &>(q0)) |
          (static_cast<uint32_t>(reinterpret_cast<const uint16_t &>(q1)) << 16);
  out.y = static_cast<uint32_t>(reinterpret_cast<const uint16_t &>(q2)) |
          (static_cast<uint32_t>(reinterpret_cast<const uint16_t &>(q3)) << 16);
  return out;
}

// grid = (cols / 64, max_row_tiles, num_tensors); CTA (x, y, z) quantizes the 64 x 64 tile at tile
// column x and tile row y of tensor z. The tile is staged in shared memory, so the row-wise
// output (32 contiguous bytes per row) and the transposed output (32 contiguous bytes per column)
// are both written as whole, aligned sectors.
template <typename ScaleType, bool RETURN_ROWWISE, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(kThreads) group_quantize_2d_nvfp4_kernel(const Args args) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (args.noop != nullptr && args.noop[0] == 1.0f) {
    return;
  }
  const int t = blockIdx.z;
  const int rows = args.rows[t];
  if (blockIdx.y * kTile >= rows) {
    return;
  }
  const int cols = args.cols;
  const int row0 = blockIdx.y * kTile;  // first row of the tile within the tensor
  const int col0 = blockIdx.x * kTile;  // first column of the tile
  const int tid = threadIdx.x;

  __shared__ __align__(16) bf16 tile[kTile][kPitch];
  __shared__ float block_amax[4][4];  // [row block][column block]

  // 1. Load the tile: 64 rows x 8 segments of 8 elements; each thread loads two.
  const bf16 *in = args.input + (args.input_row_offset[t] + row0) * static_cast<size_t>(cols) + col0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tid + i * kThreads;
    const int r = idx >> 3;
    const int seg = idx & 7;
    *reinterpret_cast<uint4 *>(&tile[r][seg * 8]) =
        *reinterpret_cast<const uint4 *>(in + static_cast<size_t>(r) * cols + seg * 8);
  }
  __syncthreads();

  // 2. Amax of each of the 16 scaling blocks: 16 threads per block, one row of it each.
  {
    const int b = tid >> 4;  // block 0..15
    const int l = tid & 15;  // row of the block
    const int rb = b >> 2, cb = b & 3;
    const uint4 lo = *reinterpret_cast<const uint4 *>(&tile[rb * 16 + l][cb * 16]);
    const uint4 hi = *reinterpret_cast<const uint4 *>(&tile[rb * 16 + l][cb * 16 + 8]);
    const bf16 *v = reinterpret_cast<const bf16 *>(&lo);
    const bf16 *w = reinterpret_cast<const bf16 *>(&hi);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      amax = fmaxf(amax, fabsf(static_cast<float>(v[i])));
      amax = fmaxf(amax, fabsf(static_cast<float>(w[i])));
    }
#pragma unroll
    for (int delta = 8; delta >= 1; delta >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
    }
    if (l == 0) {
      block_amax[rb][cb] = amax;
    }
  }
  __syncthreads();

  // Global scales exactly as the single-tensor 2D kernel derives them.
  const float *amax_row = args.rowwise_amax[t];
  const float *amax_col = args.colwise_amax[t];
  const float S_enc_rowwise =
      (amax_row == nullptr) ? 1.0f
                            : core::compute_global_encode_scaling_factor_FP4<ScaleType>(*amax_row);
  const float S_dec_rowwise = 1.0 / S_enc_rowwise;
  const float S_enc_colwise =
      (amax_col == nullptr) ? S_enc_rowwise
                            : core::compute_global_encode_scaling_factor_FP4<ScaleType>(*amax_col);
  const float S_dec_colwise = 1.0 / S_enc_colwise;
  constexpr float float_max = detail::TypeExtrema<float>::max;

  // 3. Row-wise: thread (row, quarter) casts 16 elements of its row (one scaling block's row).
  if constexpr (RETURN_ROWWISE) {
    const int r = tid >> 2;
    const int q = tid & 3;
    const int rb = r >> 4;
    const ScaleType S_dec_b =
        core::compute_decoding_scaling_factor<ScaleType>(block_amax[rb][q], S_enc_rowwise);
    const float inv = fminf(1.0f / (static_cast<float>(S_dec_b) * S_dec_rowwise), float_max);
    const uint4 lo = *reinterpret_cast<const uint4 *>(&tile[r][q * 16]);
    const uint4 hi = *reinterpret_cast<const uint4 *>(&tile[r][q * 16 + 8]);
    const uint2 packed = cast_16(lo, hi, float2{inv, inv});
    const size_t trow = static_cast<size_t>(row0 + r);
    *reinterpret_cast<uint2 *>(args.rowwise_data[t] + trow * (cols / 2) + col0 / 2 + q * 8) = packed;
    reinterpret_cast<ScaleType *>(args.rowwise_scale[t])[trow * args.rowwise_scale_stride[t] +
                                                         col0 / kBlock + q] = S_dec_b;
  }

  // 4. Transposed: thread (column, quarter) casts 16 rows of its column; rows [16 q, 16 q + 16).
  if constexpr (RETURN_TRANSPOSE) {
    const int c = tid >> 2;
    const int q = tid & 3;
    const ScaleType S_dec_b =
        core::compute_decoding_scaling_factor<ScaleType>(block_amax[q][c >> 4], S_enc_colwise);
    const float inv = fminf(1.0f / (static_cast<float>(S_dec_b) * S_dec_colwise), float_max);
    uint4 lo, hi;
    bf16 *l8 = reinterpret_cast<bf16 *>(&lo);
    bf16 *h8 = reinterpret_cast<bf16 *>(&hi);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      l8[i] = tile[q * 16 + i][c];
      h8[i] = tile[q * 16 + 8 + i][c];
    }
    const uint2 packed = cast_16(lo, hi, float2{inv, inv});
    const size_t tcol = static_cast<size_t>(col0 + c);
    *reinterpret_cast<uint2 *>(args.colwise_data[t] + tcol * (rows / 2) + row0 / 2 + q * 8) = packed;
    reinterpret_cast<ScaleType *>(args.colwise_scale[t])[tcol * args.colwise_scale_stride[t] +
                                                         row0 / kBlock + q] = S_dec_b;
  }
#else
  NVTE_DEVICE_ERROR("Grouped 2D NVFP4 quantization requires sm_100 or newer.");
#endif  // __CUDA_ARCH__ >= 1000
}

}  // namespace group_quantize_2d_kernel

// Quantizes input (rows = sum(split_sections), cols) into output_list with 2D block scaling. The
// amax of every output tensor (row-wise, and column-wise when that output is requested) must have
// been computed already. Supported: BF16 input, no stochastic rounding, compact (unswizzled)
// scales, cols % 64 == 0 and every split a multiple of 64 rows.
inline void group_quantize_2d(const Tensor &input, const Tensor *noop,
                              std::vector<Tensor *> &output_list, const size_t *split_sections,
                              size_t num_tensors, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace group_quantize_2d_kernel;
  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(input.dtype() == DType::kBFloat16,
             "Grouped 2D NVFP4 quantization requires a BF16 input.");
  NVTE_CHECK(num_tensors == output_list.size(),
             "Number of output tensors should match number of tensors.");
  const auto [rows_total, cols] = input.flat_2d_dims();
  NVTE_CHECK(cols % kTile == 0, "Grouped 2D NVFP4 quantization requires cols % 64 == 0, got ", cols);
  NVTE_CHECK(reinterpret_cast<uintptr_t>(input.data.dptr) % 16 == 0,
             "Grouped 2D NVFP4 quantization requires a 16-byte aligned input.");

  size_t start = 0;
  size_t i = 0;
  while (i < num_tensors) {
    Args args{};
    args.input = reinterpret_cast<const bf16 *>(input.data.dptr);
    args.noop = (noop != nullptr) ? reinterpret_cast<const float *>(noop->data.dptr) : nullptr;
    args.cols = static_cast<int>(cols);
    args.num_tensors = 0;
    args.max_row_tiles = 0;
    bool want_rowwise = false;
    bool want_transpose = false;
    DType scale_dtype = DType::kFloat8E4M3;
    bool scale_dtype_set = false;
    for (; i < num_tensors && args.num_tensors < kMaxTensors2D; ++i) {
      const size_t rows = split_sections[i];
      if (rows == 0) {
        continue;
      }
      NVTE_CHECK(rows % kTile == 0, "Grouped 2D NVFP4 quantization requires every split to be a "
                 "multiple of 64 rows, got ", rows);
      Tensor *out = output_list[i];
      NVTE_CHECK(!out->with_gemm_swizzled_scales,
                 "Grouped 2D NVFP4 quantization writes compact (unswizzled) scales.");
      const bool row_i = out->has_data();
      const bool col_i = out->has_columnwise_data();
      NVTE_CHECK(row_i || col_i, "Output tensor ", i, " has neither row-wise nor column-wise data.");
      if (args.num_tensors == 0) {
        want_rowwise = row_i;
        want_transpose = col_i;
      }
      NVTE_CHECK(row_i == want_rowwise && col_i == want_transpose,
                 "All outputs of a grouped 2D NVFP4 quantization need the same usages.");
      const DType sdt = row_i ? out->scale_inv.dtype : out->columnwise_scale_inv.dtype;
      if (!scale_dtype_set) {
        scale_dtype = sdt;
        scale_dtype_set = true;
      }
      NVTE_CHECK(sdt == scale_dtype, "All NVFP4 scale tensors must have the same dtype.");

      const int n = args.num_tensors;
      args.input_row_offset[n] = start;
      args.rows[n] = static_cast<int>(rows);
      if (row_i) {
        NVTE_CHECK(out->data.dptr != nullptr && out->scale_inv.dptr != nullptr,
                   "Row-wise output ", i, " is missing data or scales.");
        NVTE_CHECK(reinterpret_cast<uintptr_t>(out->data.dptr) % 8 == 0,
                   "Row-wise data of output ", i, " must be 8-byte aligned.");
        args.rowwise_data[n] = reinterpret_cast<uint8_t *>(out->data.dptr);
        args.rowwise_scale[n] = out->scale_inv.dptr;
        args.rowwise_scale_stride[n] = static_cast<int>(out->scale_inv.shape[1]);
        NVTE_CHECK(out->scale_inv.shape.size() == 2 &&
                       static_cast<size_t>(out->scale_inv.shape[1]) >= cols / kBlock,
                   "Row-wise scale tensor ", i, " is too small.");
      }
      if (col_i) {
        NVTE_CHECK(out->columnwise_data.dptr != nullptr && out->columnwise_scale_inv.dptr != nullptr,
                   "Column-wise output ", i, " is missing data or scales.");
        NVTE_CHECK(reinterpret_cast<uintptr_t>(out->columnwise_data.dptr) % 8 == 0,
                   "Column-wise data of output ", i, " must be 8-byte aligned.");
        args.colwise_data[n] = reinterpret_cast<uint8_t *>(out->columnwise_data.dptr);
        args.colwise_scale[n] = out->columnwise_scale_inv.dptr;
        args.colwise_scale_stride[n] = static_cast<int>(out->columnwise_scale_inv.shape[1]);
        NVTE_CHECK(out->columnwise_scale_inv.shape.size() == 2 &&
                       static_cast<size_t>(out->columnwise_scale_inv.shape[1]) >= rows / kBlock,
                   "Column-wise scale tensor ", i, " is too small.");
      }
      args.rowwise_amax[n] = reinterpret_cast<const float *>(out->amax.dptr);
      args.colwise_amax[n] = reinterpret_cast<const float *>(out->columnwise_amax.dptr);
      args.max_row_tiles = std::max(args.max_row_tiles, static_cast<int>(rows / kTile));
      args.num_tensors++;
      start += rows;
    }
    // Rows of the tensors that were skipped as empty sit at no offset, so only advance past the
    // chunk's non-empty tensors (empty ones add 0 rows).
    if (args.num_tensors == 0) {
      continue;
    }

    const dim3 grid(cols / kTile, args.max_row_tiles, args.num_tensors);
    TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(scale_dtype, ScaleType, {
      TRANSFORMER_ENGINE_SWITCH_CONDITION(want_rowwise, RETURN_ROWWISE, {
        TRANSFORMER_ENGINE_SWITCH_CONDITION(want_transpose, RETURN_TRANSPOSE, {
          auto kernel = group_quantize_2d_nvfp4_kernel<ScaleType, RETURN_ROWWISE, RETURN_TRANSPOSE>;
          kernel<<<grid, kThreads, 0, stream>>>(args);
        });
      });
    });
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_2D_NVFP4_CUH_
