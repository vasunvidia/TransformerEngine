/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dswiglu_hadamard_transform_amax.cu
 *  \brief SwiGLU backward fused with the NVFP4 RHT amaxes of its output.
 *
 *  One kernel computes dact = dswiglu(grad, input), writes it and reduces the two amaxes
 *  that the NVFP4 RHT recipe needs before it casts dact: the pre-RHT amax (max |dact|) and
 *  the amax of the 16x16 RHT of dact^T. Without the fusion these come from nvte_dswiglu and
 *  nvte_hadamard_transform_amax, which reads dact back from global memory.
 *
 *  The results are bitwise equal to that unfused pair:
 *  - dSwiGLU: the inline PTX below repeats, operation for operation, what the bf16 TMA
 *    kernel of nvte_dswiglu (cast_fp8_gated_kernel with silu / dsilu) executes when the
 *    activation kernels are compiled with --use_fast_math (NVTE_BUILD_ACTIVATION_WITH_FAST_MATH).
 *    Explicit .rn rounding keeps ptxas from contracting the multiplies into FMAs. Other
 *    builds, and devices where nvte_dswiglu does not use the TMA kernel, take the unfused path.
 *  - RHT amax: the same wmma MMA against the same sign-masked 16x16 Hadamard fragments, the
 *    same bf16 rounding and the same max.xorsign.abs reduction as HadamardAmaxTmaKernel.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/hadamard_transform.h>

#include <algorithm>
#include <climits>
#include <cstdint>

#include "common/common.h"
#include "common/util/cuda_runtime.h"
#include "common/utils.cuh"
#include "hadamard_transform_utils.cuh"

namespace transformer_engine {
namespace {

// Set by the build when the activation kernels are compiled with --use_fast_math
#ifdef NVTE_ACTIVATION_WITH_FAST_MATH
constexpr bool kActivationWithFastMath = true;
#else
constexpr bool kActivationWithFastMath = false;
#endif

constexpr int kTileRows = 64;  // Tokens per tile
constexpr int kTileCols = 64;  // Columns of the gate half per tile (a tile also covers the
                               // matching 64 columns of the linear half)
constexpr int kThreads = 256;
constexpr int kMinBlocksPerSM = 3;
constexpr int kChunks = kTileRows * kTileCols / 8;  // 16-byte chunks per 64x64 bf16 tile
constexpr int kStages = 3;                          // cp.async ring depth
// Per stage: gate-half tile, linear-half tile, grad tile
constexpr size_t kDynamicSmemBytes = static_cast<size_t>(kStages) * 3 * kChunks * sizeof(uint4);

// Sigmoid of the activation kernels' fast-math dSwiGLU: s = rcp.rn.ftz(1 + ex2.approx.ftz(-x *
// log2(e))). rcp.rn.ftz compiles to MUFU.RCP, a Newton step and a branch to a slow path for
// t >= 2^126 (t = 1 + e >= 1 here). This branchless form runs the fast path on t clamped to
// FLT_MAX; for every t >= 2^126 it returns +0 like the slow path (1/t is subnormal and flushed).
// The two forms agree bitwise for all 65536 bf16 inputs.
// clang-format off
#define NVTE_DSWIGLU_SIGMOID(x, t, s, e)          \
  "mul.rn.ftz.f32 " t ", " x ", 0fBFB8AA3B;\n"    \
  "ex2.approx.ftz.f32 " t ", " t ";\n"            \
  "add.rn.ftz.f32 " t ", " t ", 0f3F800000;\n"    \
  "min.ftz.f32 " t ", " t ", 0f7F7FFFFF;\n"       \
  "rcp.approx.ftz.f32 " s ", " t ";\n"            \
  "neg.f32 " t ", " t ";\n"                       \
  "fma.rn.f32 " e ", " t ", " s ", 0f3F800000;\n" \
  "fma.rn.f32 " s ", " s ", " e ", " s ";\n"

// Rest of the dSwiGLU element given s: act = x * s; dact = fma(act, 1 - s, s);
// d(gate) = up * (grad * dact); d(linear) = grad * act.
#define NVTE_DSWIGLU_TAIL(x, y, g, s, t, a, da, dl) \
  "mul.rn.ftz.f32 " a ", " x ", " s ";\n"           \
  "sub.rn.ftz.f32 " t ", 0f3F800000, " s ";\n"      \
  "fma.rn.ftz.f32 " t ", " a ", " t ", " s ";\n"    \
  "mul.rn.ftz.f32 " t ", " g ", " t ";\n"           \
  "mul.rn.ftz.f32 " da ", " y ", " t ";\n"          \
  "mul.rn.ftz.f32 " dl ", " g ", " a ";\n"

// Two dSwiGLU elements from one bf16x2 word each of the gate input x, the linear input y and
// the incoming gradient g. Returns bf16x2 words of d(gate) and d(linear).
__device__ __forceinline__ void dswiglu_bf16x2(uint32_t x2, uint32_t y2, uint32_t g2,
                                               uint32_t &dgate2, uint32_t &dlinear2) {
  asm("{\n"
      ".reg .b32 x<2>, y<2>, g<2>, t<2>, s<2>, e<2>, a<2>, da<2>, dl<2>;\n"
      "shl.b32 x0, %2, 16;\n"
      "and.b32 x1, %2, 0xffff0000;\n"
      "shl.b32 y0, %3, 16;\n"
      "and.b32 y1, %3, 0xffff0000;\n"
      "shl.b32 g0, %4, 16;\n"
      "and.b32 g1, %4, 0xffff0000;\n"
      NVTE_DSWIGLU_SIGMOID("x0", "t0", "s0", "e0")
      NVTE_DSWIGLU_SIGMOID("x1", "t1", "s1", "e1")
      NVTE_DSWIGLU_TAIL("x0", "y0", "g0", "s0", "t0", "a0", "da0", "dl0")
      NVTE_DSWIGLU_TAIL("x1", "y1", "g1", "s1", "t1", "a1", "da1", "dl1")
      "cvt.rn.bf16x2.f32 %0, da1, da0;\n"
      "cvt.rn.bf16x2.f32 %1, dl1, dl0;\n"
      "}\n"
      : "=r"(dgate2), "=r"(dlinear2)
      : "r"(x2), "r"(y2), "r"(g2));
}

#undef NVTE_DSWIGLU_SIGMOID
#undef NVTE_DSWIGLU_TAIL
// clang-format on

__device__ __forceinline__ uint32_t max_xorsign_abs_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ void store_no_allocate(void *ptr, uint4 v) {
  asm volatile("st.global.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ __forceinline__ void cp_async_16B(void *smem, const void *gmem) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  }
  return v;
}

// input: [rows, 2 * cols] (gate half | linear half), grad: [rows, cols], dact: [rows, 2 * cols].
// Persistent CTAs stride over tiles of 64 rows x 64 gate columns; a tile yields two 64x64 dact
// tiles (columns c0 and cols + c0). A 3-stage cp.async ring holds the x, y and grad tiles in the
// 128B_ATOM_32B swizzle of HadamardAmaxTmaKernel. dSwiGLU is computed in place (d(gate) over x,
// d(linear) over grad), then each half of the CTA runs the transposed-RHT amax MMA over one of
// the two 64x64 buffers (4 warps per buffer, one 16-column strip per warp, 4 row blocks).
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSM) dswiglu_hadamard_transform_amax_kernel(
    const bf16 *__restrict__ input, const bf16 *__restrict__ grad, bf16 *__restrict__ dact,
    float *__restrict__ pre_rht_amax, float *__restrict__ transposed_rht_amax, int64_t cols,
    int col_tiles, int num_tiles, uint16_t random_sign_mask_t) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  extern __shared__ __align__(128) uint4 ring[];  // [kStages][x, y, grad][kChunks]
  __shared__ float pre_rht_amax_smem[kThreads / 32];
  __shared__ float transposed_rht_amax_smem[kThreads / 32];

  const int tid = threadIdx.x;
  const int64_t input_ld = 2 * cols;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int buffer_idx = warp / 4;
  const int warp_in_buffer = warp % 4;
  const uint32_t strip_idx = swizzle_128B_atom_32B(lane % 16, lane / 16 + warp_in_buffer * 2);

  // Each thread moves two 16-byte chunks of every 64x64 tile
  uint32_t input_offset[2], grad_offset[2], smem_idx[2];
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int chunk = tid + r * kThreads;
    const int row = chunk / 8;
    const int col_chunk = chunk % 8;
    input_offset[r] = static_cast<uint32_t>(row * input_ld + col_chunk * 8);
    grad_offset[r] = static_cast<uint32_t>(row * cols + col_chunk * 8);
    smem_idx[r] = swizzle_128B_atom_32B(row, col_chunk);
  }
  auto issue_loads = [&](int tile, int stage) {
    const int64_t row0 = static_cast<int64_t>(tile / col_tiles) * kTileRows;
    const int64_t col0 = static_cast<int64_t>(tile % col_tiles) * kTileCols;
    const bf16 *input_tile = input + row0 * input_ld + col0;
    const bf16 *grad_tile = grad + row0 * cols + col0;
    uint4 *x_smem = ring + (stage * 3 + 0) * kChunks;
#pragma unroll
    for (int r = 0; r < 2; r++) {
      cp_async_16B(x_smem + smem_idx[r], input_tile + input_offset[r]);
      cp_async_16B(x_smem + kChunks + smem_idx[r], input_tile + input_offset[r] + cols);
      cp_async_16B(x_smem + 2 * kChunks + smem_idx[r], grad_tile + grad_offset[r]);
    }
  };

  uint32_t had_frag_i[4];
  uint32_t had_frag_t[4];
  get_hadamard_matrix_fragment<false, true, false, false>(had_frag_i, 0, had_frag_t,
                                                          random_sign_mask_t);

#pragma unroll
  for (int p = 0; p < kStages - 1; p++) {
    const int tile = blockIdx.x + p * gridDim.x;
    if (tile < num_tiles) {
      issue_loads(tile, p);
    }
    cp_async_commit();
  }

  uint32_t local_pre_rht_amax = 0;
  uint32_t local_transposed_amax = 0;
  int stage = 0;
  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    cp_async_wait<kStages - 2>();
    uint4 *x_smem = ring + (stage * 3 + 0) * kChunks;
    uint4 *y_smem = x_smem + kChunks;
    uint4 *g_smem = x_smem + 2 * kChunks;
    const int64_t row0 = static_cast<int64_t>(tile / col_tiles) * kTileRows;
    const int64_t col0 = static_cast<int64_t>(tile % col_tiles) * kTileCols;
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const uint4 xv = x_smem[smem_idx[r]];
      const uint4 yv = y_smem[smem_idx[r]];
      const uint4 gv = g_smem[smem_idx[r]];
      uint4 dgate, dlinear;
      dswiglu_bf16x2(xv.x, yv.x, gv.x, dgate.x, dlinear.x);
      dswiglu_bf16x2(xv.y, yv.y, gv.y, dgate.y, dlinear.y);
      dswiglu_bf16x2(xv.z, yv.z, gv.z, dgate.z, dlinear.z);
      dswiglu_bf16x2(xv.w, yv.w, gv.w, dgate.w, dlinear.w);
      local_pre_rht_amax = max_xorsign_abs_bf16x2(
          local_pre_rht_amax, max_xorsign_abs_bf16x2(max_xorsign_abs_bf16x2(dgate.x, dgate.y),
                                                     max_xorsign_abs_bf16x2(dgate.z, dgate.w)));
      local_pre_rht_amax = max_xorsign_abs_bf16x2(
          local_pre_rht_amax, max_xorsign_abs_bf16x2(max_xorsign_abs_bf16x2(dlinear.x, dlinear.y),
                                                     max_xorsign_abs_bf16x2(dlinear.z, dlinear.w)));
      x_smem[smem_idx[r]] = dgate;
      g_smem[smem_idx[r]] = dlinear;
      bf16 *dact_ptr = dact + row0 * input_ld + col0 + input_offset[r];
      store_no_allocate(dact_ptr, dgate);
      store_no_allocate(dact_ptr + cols, dlinear);
    }
    // The dSwiGLU results of this stage are visible, and all reads of the stage refilled below
    // (consumed in the previous iteration) are complete.
    __syncthreads();
    uint4 *rht_in = buffer_idx == 0 ? x_smem : g_smem;
#pragma unroll
    for (int rb = 0; rb < 4; rb++) {
      uint32_t a_frag[4];
      uint32_t c_frag[4];
      uint32_t temp_amax;
      ldmatrix_x4_m8n8_shared_b16<true>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                        rht_in + rb * 16 * 8 + strip_idx);
      // Fragment order as in the transposed branch of HadamardAmaxTmaKernel
      mma_m16_n16_k16_b16_b16_b16_noacc<true>(
          a_frag[0], a_frag[2], a_frag[1], a_frag[3], had_frag_t[0], had_frag_t[1], had_frag_t[2],
          had_frag_t[3], c_frag[0], c_frag[1], c_frag[2], c_frag[3], temp_amax);
      local_transposed_amax = max_xorsign_abs_bf16x2(local_transposed_amax, temp_amax);
    }
    const int next_tile = tile + (kStages - 1) * gridDim.x;
    const int next_stage = stage == 0 ? kStages - 1 : stage - 1;  // (stage + kStages - 1) % kStages
    if (next_tile < num_tiles) {
      issue_loads(next_tile, next_stage);
    }
    cp_async_commit();
    stage = stage + 1 == kStages ? 0 : stage + 1;
  }
  cp_async_wait<0>();

  float pre_rht_amax_f, transposed_amax_f;
  unpack_max_of_packed_bf16(local_pre_rht_amax, pre_rht_amax_f);
  unpack_max_of_packed_bf16(local_transposed_amax, transposed_amax_f);
  pre_rht_amax_f = warp_max(pre_rht_amax_f);
  transposed_amax_f = warp_max(transposed_amax_f);
  if (lane == 0) {
    pre_rht_amax_smem[warp] = pre_rht_amax_f;
    transposed_rht_amax_smem[warp] = transposed_amax_f;
  }
  __syncthreads();
  if (warp == 0) {
    float v_pre = lane < kThreads / 32 ? pre_rht_amax_smem[lane] : 0.0f;
    float v_t = lane < kThreads / 32 ? transposed_rht_amax_smem[lane] : 0.0f;
    v_pre = warp_max(v_pre);
    v_t = warp_max(v_t);
    // Non-negative floats order like their bit patterns
    if (lane == 0 && pre_rht_amax != nullptr) {
      atomicMax(reinterpret_cast<int *>(pre_rht_amax), __float_as_int(v_pre));
    }
    if (lane == 0 && transposed_rht_amax != nullptr) {
      atomicMax(reinterpret_cast<int *>(transposed_rht_amax), __float_as_int(v_t));
    }
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__global__ void zero_amaxes_kernel(float *pre_rht_amax, float *transposed_rht_amax) {
  if (pre_rht_amax != nullptr) {
    *pre_rht_amax = 0.0f;
  }
  if (transposed_rht_amax != nullptr) {
    *transposed_rht_amax = 0.0f;
  }
}

bool is_16B_aligned(const void *ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; }

// Whether the fused kernel reproduces nvte_dswiglu + nvte_hadamard_transform_amax bitwise.
bool can_fuse(const Tensor &grad, const Tensor &input, const Tensor &output,
              const Tensor &amax_output) {
  // The kernel mirrors the fast-math activation kernels only
  if (!kActivationWithFastMath) {
    return false;
  }
  // nvte_dswiglu uses its TMA kernel (the one the kernel mirrors) from compute capability 10.0
  if (!is_supported_by_CC_100()) {
    return false;
  }
  if (grad.scaling_mode != NVTE_DELAYED_TENSOR_SCALING ||
      input.scaling_mode != NVTE_DELAYED_TENSOR_SCALING ||
      output.scaling_mode != NVTE_DELAYED_TENSOR_SCALING) {
    return false;
  }
  if (grad.dtype() != DType::kBFloat16 || input.dtype() != DType::kBFloat16 || !output.has_data() ||
      output.data.dtype != DType::kBFloat16 || output.has_columnwise_data()) {
    return false;
  }
  const SimpleTensor &pre_rht_amax = amax_output.amax;
  const SimpleTensor &transposed_amax = amax_output.columnwise_amax;
  if (pre_rht_amax.dptr == nullptr && transposed_amax.dptr == nullptr) {
    return false;
  }
  if ((pre_rht_amax.dptr != nullptr && pre_rht_amax.dtype != DType::kFloat32) ||
      (transposed_amax.dptr != nullptr && transposed_amax.dtype != DType::kFloat32)) {
    return false;
  }
  if (input.dim() < 2 || grad.dim() < 2 || output.dim() < 2) {
    return false;
  }
  const size_t rows = input.flat_first_dim();
  const size_t two_cols = input.flat_last_dim();
  const size_t cols = two_cols / 2;
  if (two_cols % 2 != 0 || grad.flat_first_dim() != rows || grad.flat_last_dim() != cols ||
      output.flat_first_dim() != rows || output.flat_last_dim() != two_cols) {
    return false;
  }
  if (rows == 0 || rows % kTileRows != 0 || cols % kTileCols != 0) {
    return false;
  }
  // In-tile offsets are 32-bit and the tile index is an int
  if (two_cols >= (size_t{1} << 24) ||
      (rows / kTileRows) * (cols / kTileCols) > static_cast<size_t>(INT_MAX)) {
    return false;
  }
  return is_16B_aligned(input.data.dptr) && is_16B_aligned(grad.data.dptr) &&
         is_16B_aligned(output.data.dptr);
}

void dswiglu_hadamard_transform_amax(const Tensor &grad, const Tensor &input, Tensor *output,
                                     Tensor *amax_output, uint16_t random_sign_mask_t,
                                     cudaStream_t stream) {
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim() / 2;
  const int col_tiles = static_cast<int>(cols / kTileCols);
  const int num_tiles = col_tiles * static_cast<int>(rows / kTileRows);

  float *pre_rht_amax = reinterpret_cast<float *>(amax_output->amax.dptr);
  float *transposed_amax = reinterpret_cast<float *>(amax_output->columnwise_amax.dptr);
  zero_amaxes_kernel<<<1, 1, 0, stream>>>(pre_rht_amax, transposed_amax);
  NVTE_CHECK_CUDA(cudaGetLastError());

  const int grid = std::min(num_tiles, cuda::sm_count() * kMinBlocksPerSM);
  auto kernel = dswiglu_hadamard_transform_amax_kernel;
  NVTE_CHECK_CUDA(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes));
  kernel<<<grid, kThreads, kDynamicSmemBytes, stream>>>(
      reinterpret_cast<const bf16 *>(input.data.dptr),
      reinterpret_cast<const bf16 *>(grad.data.dptr), reinterpret_cast<bf16 *>(output->data.dptr),
      pre_rht_amax, transposed_amax, static_cast<int64_t>(cols), col_tiles, num_tiles,
      random_sign_mask_t);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace
}  // namespace transformer_engine

void nvte_dswiglu_hadamard_transform_amax(const NVTETensor grad, const NVTETensor input,
                                          NVTETensor output, NVTETensor amax_output,
                                          int random_sign_mask, int random_sign_mask_t,
                                          cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswiglu_hadamard_transform_amax);
  using namespace transformer_engine;
  const Tensor &grad_tensor = *convertNVTETensorCheck(grad);
  const Tensor &input_tensor = *convertNVTETensorCheck(input);
  Tensor *output_tensor = convertNVTETensorCheck(output);
  Tensor *amax_tensor = convertNVTETensorCheck(amax_output);
  if (can_fuse(grad_tensor, input_tensor, *output_tensor, *amax_tensor)) {
    CheckInputTensor(grad_tensor, "grad");
    CheckInputTensor(input_tensor, "input");
    CheckOutputTensor(*output_tensor, "output");
    dswiglu_hadamard_transform_amax(grad_tensor, input_tensor, output_tensor, amax_tensor,
                                    static_cast<uint16_t>(random_sign_mask_t), stream);
    return;
  }
  // Unfused: same results
  nvte_dswiglu(grad, input, output, stream);
  nvte_hadamard_transform_amax(output, amax_output, random_sign_mask, random_sign_mask_t, stream);
}
