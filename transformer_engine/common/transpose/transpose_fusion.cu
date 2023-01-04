/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transpose.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include <type_traits>
#include "../utils.cuh"
#include "../common.h"

namespace transformer_engine {

template <bool full_tile, int nvec_in, int nvec_out,
          typename IVec, typename CType, typename OType>
inline __device__ void dgelu_regs(const IVec (&in)[nvec_out],
                                  OType *output_cast_tile,
                                  const size_t current_place,
                                  const size_t stride,
                                  CType  &max,  // NOLINT(*)
                                  const CType scale,
                                  const bool valid_store) {
  using T = OType;
  using OVecC = Vec<T, nvec_in>;

#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    OVecC out_cast;
#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      const CType tmp = in[i].data.elt[j];
      const T elt_o = T(scale * tmp);

      out_cast.data.elt[j]     = elt_o;

      __builtin_assume(max >= 0);
      max = fmaxf(fabsf(tmp), max);
    }
    if (full_tile || valid_store) {
      out_cast.store_to(output_cast_tile, current_place + stride * i);
    }
  }
}

template <bool full_tile, int nvec_in, int nvec_out,
          typename IVec, typename OVec, typename CVec, typename CType>
inline __device__ void dgelu_and_transpose_regs_partial_dbias(const IVec (&in)[nvec_out],
                                                             OVec (&out_trans)[nvec_in],
                                                             CVec &out_dbias,  // NOLINT(*)
                                                             typename OVec::type *output_cast_tile,
                                                             const size_t current_place,
                                                             const size_t stride,
                                                             CType  &max,  // NOLINT(*)
                                                             const CType scale,
                                                             const int dbias_shfl_src_lane,
                                                             const bool valid_store) {
  using T = typename OVec::type;
  using OVecC = Vec<T, nvec_in>;

  CVec step_dbias; step_dbias.clear();

#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    OVecC out_cast;
#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      const CType tmp = in[i].data.elt[j];
      const T elt_o = T(scale * tmp);

      /* dbias: thread tile local accumulation */
      step_dbias.data.elt[j] += tmp;

      out_cast.data.elt[j]     = elt_o;
      out_trans[j].data.elt[i] = elt_o;  // thread tile transpose

      __builtin_assume(max >= 0);
      max = fmaxf(fabsf(tmp), max);
    }
    if (full_tile || valid_store) {
      out_cast.store_to(output_cast_tile, current_place + stride * i);
    }
  }

#pragma unroll
  for (unsigned int j = 0; j < nvec_in; ++j) {
    CType elt = step_dbias.data.elt[j];
    elt = __shfl_sync(0xffffffff, elt, dbias_shfl_src_lane);  // shuffle data in warp
    out_dbias.data.elt[j] += elt;
  }
}

template <bool full_tile, int nvec_in, int nvec_out,
          typename IVec, typename OVec, typename CVec, typename CType>
inline __device__ void transpose_regs_partial_dbias(const IVec (&in)[nvec_out],
                                                             OVec (&out_trans)[nvec_in],
                                                             CVec &out_dbias,  // NOLINT(*)
                                                             const size_t stride,
                                                             const CType scale_inv,
                                                             const int dbias_shfl_src_lane,
                                                             const bool valid_store) {
  using T = typename OVec::type;
  using OVecC = Vec<T, nvec_in>;

  CVec step_dbias; step_dbias.clear();

#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    OVecC out_cast;
#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      const CType tmp = CType(in[i].data.elt[j]) * scale_inv;
      const T elt_o = in[i].data.elt[j];

      /* dbias: thread tile local accumulation */
      step_dbias.data.elt[j] += tmp;

      out_trans[j].data.elt[i] = elt_o;  // thread tile transpose
    }
  }

#pragma unroll
  for (unsigned int j = 0; j < nvec_in; ++j) {
    CType elt = step_dbias.data.elt[j];
    elt = __shfl_sync(0xffffffff, elt, dbias_shfl_src_lane);  // shuffle data in warp
    out_dbias.data.elt[j] += elt;
  }
}

// STUFF TO TUNE
constexpr unsigned int n_warps_per_tile = 4;
constexpr int desired_load_size = 8;
constexpr int desired_store_size = 8;

constexpr unsigned int max_threads_per_block = 256;
static_assert(n_warps_per_tile * THREADS_PER_WARP <= max_threads_per_block);
constexpr unsigned int cast_transpose_num_threads = n_warps_per_tile * THREADS_PER_WARP;

namespace {

template <typename IType, typename OType, typename CType>
struct TDBiasParam {
    using InputType = IType;
    using OutputType = OType;
    using ComputeType = CType;
    const IType *input;
    OType *output_t;
    const CType *scale_inv;
    CType *workspace;
};

template <typename IType, typename IType2, typename IType3, typename OType, typename CType>
struct TDBiasDGeluParam {
    using InputType = IType;
    using InputType2 = IType2;
    using InputType3 = IType3;
    using OutputType = OType;
    using ComputeType = CType;
    const IType *input;
    const IType2 *gelu_input;
    const IType3 *gelu_output;
    const CType *gelu_output_scale_inv;
    OType *output_c;
    OType *output_t;
    const CType *scale_ptr;
    CType *amax;
    CType *scale_inv;
    CType *workspace;
};

template <typename IType, typename IType2, /*typename IType3,*/ typename OType, typename CType>
struct TDGeluParam {
    using InputType = IType;
    using InputType2 = IType2;
//    using InputType3 = IType3;
    using OutputType = OType;
    using ComputeType = CType;
    const IType *input;
    const IType2 *gelu_input;
//    const IType3 *gelu_output;
//    const CType *gelu_output_scale_inv;
    OType *output_c;
    const CType *scale_ptr;
    CType *amax;
    CType *scale_inv;
};

}  // namespace

template <int nvec_in, int nvec_out, typename Param>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
transpose_dbias_kernel(const Param param,
                            const size_t row_length,
                            const size_t num_rows,
                            const size_t num_tiles) {
  using IType = typename Param::InputType;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = row_length / (nvec_in * THREADS_PER_WARP);
  // const size_t num_tiles_y = num_rows / (nvec * THREADS_PER_WARP);
  const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                         warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType * const my_input_tile = param.input + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
                                                    THREADS_PER_WARP;
  OType * const my_output_t_tile = param.output_t + (tile_id_y * nvec_out +
                                                     tile_id_x * num_rows * nvec_in) *
                                                    THREADS_PER_WARP;
  CType * const my_partial_dbias_tile = param.workspace +
                                        (tile_id_x * (nvec_in * THREADS_PER_WARP) +
                                         tile_id_y * row_length);

  OVec * const my_scratch = reinterpret_cast<OVec *>(scratch) +
                            (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) *
                            (THREADS_PER_WARP + 1);

  CVec * const my_dbias_scratch = reinterpret_cast<CVec *>(scratch);

  IVec in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];
  CVec partial_dbias;

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                           warp_id_in_tile * n_iterations) %
                          THREADS_PER_WARP;
  const CType scale_inv = param.scale_inv != nullptr ? *param.scale_inv : 1;

  partial_dbias.clear();

#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        in[current_in][j].load_from(my_input_tile,
                                    current_stride + my_place_in + stride * (nvec_out + j));
      }
    }
    OVec out_trans[nvec_in];  // NOLINT(*)
    transpose_regs_partial_dbias<true>(in[current_in ^ 1], out_trans,
                                                partial_dbias,
                                                stride, scale_inv,
                                                (my_id_in_warp + i +
                                                 warp_id_in_tile * n_iterations) %
                                                THREADS_PER_WARP,
                                                true);

#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      out_space[i][j].data.vec = out_trans[j].data.vec;
    }
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                  j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
               THREADS_PER_WARP;
    current_stride = i * output_stride +
                     warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                              current_stride + my_place);
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }

  my_dbias_scratch[threadIdx.x] = partial_dbias;
  __syncthreads();
  // TODO(ptredak): check if the regular reduction is better
  if (warp_id_in_tile == 0) {
#pragma unroll
    for (unsigned int i = 1; i < n_warps_per_tile; ++i) {
      CVec tmp = my_dbias_scratch[threadIdx.x + i * THREADS_PER_WARP];
#pragma unroll
      for (unsigned int j = 0; j < nvec_in; ++j) {
        partial_dbias.data.elt[j] += tmp.data.elt[j];
      }
    }

    partial_dbias.store_to(my_partial_dbias_tile, my_id_in_warp);
  }

}

template <int nvec_in, int nvec_out, typename Param>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
transpose_dbias_kernel_notaligned(const Param param,
                                       const size_t row_length,
                                       const size_t num_rows,
                                       const size_t num_tiles) {
  using IType = typename Param::InputType;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = (row_length + nvec_in * THREADS_PER_WARP - 1) /
                             (nvec_in * THREADS_PER_WARP);
  const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                         warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType * const my_input_tile = param.input + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
                                                    THREADS_PER_WARP;
  OType * const my_output_t_tile = param.output_t + (tile_id_y * nvec_out +
                                                     tile_id_x * num_rows * nvec_in) *
                                                    THREADS_PER_WARP;
  CType * const my_partial_dbias_tile = param.workspace +
                                        (tile_id_x * (nvec_in * THREADS_PER_WARP) +
                                         tile_id_y * row_length);

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  const size_t row_length_rest = stride - tile_id_x * THREADS_PER_WARP;
  const size_t row_height_rest = output_stride - tile_id_y * THREADS_PER_WARP;
  const unsigned int tile_length = row_length_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                      : row_length_rest;
  const unsigned int tile_height = row_height_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                      : row_height_rest;

  OVec * const my_scratch = reinterpret_cast<OVec *>(scratch) +
                            (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) *
                            (THREADS_PER_WARP + 1);

  CVec * const my_dbias_scratch = reinterpret_cast<CVec *>(scratch);

  IVec in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];
  CVec partial_dbias;

  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                           warp_id_in_tile * n_iterations) %
                          THREADS_PER_WARP;
  const CType scale_inv = param.scale_inv != nullptr ? *param.scale_inv : 1;

  partial_dbias.clear();

  {
    const bool valid_load = my_place < tile_length &&
                            warp_id_in_tile * n_iterations < tile_height;
#pragma unroll
    for (unsigned int i = 0; i < nvec_out; ++i) {
      if (valid_load) {
        in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
      } else {
        in[0][i].clear();
      }
    }
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
      const bool valid_load = my_place_in < tile_length &&
                              warp_id_in_tile * n_iterations + i + 1 < tile_height;
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        if (valid_load) {
          in[current_in][j].load_from(my_input_tile,
                                      current_stride + my_place_in + stride * (nvec_out + j));
        } else {
          in[current_in][j].clear();
        }
      }
    }
    OVec out_trans[nvec_in];  // NOLINT(*)
    const bool valid_store = my_place < tile_length &&
                             warp_id_in_tile * n_iterations + i < tile_height;
    transpose_regs_partial_dbias<false>(in[current_in ^ 1], out_trans,
                                                partial_dbias,
                                                stride, scale_inv,
                                                (my_id_in_warp + i +
                                                 warp_id_in_tile * n_iterations) %
                                                THREADS_PER_WARP,
                                                valid_store);

#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      out_space[i][j].data.vec = out_trans[j].data.vec;
    }
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                  j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
               THREADS_PER_WARP;
    current_stride = i * output_stride +
                     warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; warp_id_in_tile * n_iterations + j < tile_length; ++j) {
      const bool valid_store = my_place < tile_height;
      if (valid_store) {
        my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                                current_stride + my_place);
      }
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }

  my_dbias_scratch[threadIdx.x] = partial_dbias;
  __syncthreads();
  // TODO(ptredak): check if the regular reduction is better
  if (warp_id_in_tile == 0) {
#pragma unroll
    for (unsigned int i = 1; i < n_warps_per_tile; ++i) {
      CVec tmp = my_dbias_scratch[threadIdx.x + i * THREADS_PER_WARP];
#pragma unroll
      for (unsigned int j = 0; j < nvec_in; ++j) {
        partial_dbias.data.elt[j] += tmp.data.elt[j];
      }
    }

    if (my_id_in_warp < tile_length) {
      partial_dbias.store_to(my_partial_dbias_tile, my_id_in_warp);
    }
  }

}

constexpr size_t reduce_dbias_num_threads = 256;

template<int nvec, typename ComputeType, typename OutputType>
__global__ void
__launch_bounds__(reduce_dbias_num_threads)
reduce_dbias_kernel(OutputType*  const dbias_output,
                    const ComputeType* const dbias_partial,
                    const int row_length,
                    const int num_rows) {
  using ComputeVec = Vec<ComputeType, nvec>;
  using OutputVec  = Vec<OutputType,  nvec>;

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id * nvec >= row_length) return;

  const ComputeType* const thread_in_base  = dbias_partial + thread_id * nvec;
  OutputType*  const thread_out_base = dbias_output  + thread_id * nvec;

  const int stride_in_vec = row_length / nvec;

  ComputeVec ldg_vec;
  ComputeVec acc_vec; acc_vec.clear();
  for (int i = 0; i < num_rows; ++i) {
    ldg_vec.load_from(thread_in_base, i * stride_in_vec);
#pragma unroll
    for (int e = 0; e < nvec; ++e) {
      acc_vec.data.elt[e] += ldg_vec.data.elt[e];
    }
  }

  OutputVec  stg_vec;
#pragma unroll
  for (int e = 0; e < nvec; ++e) {
    stg_vec.data.elt[e] = OutputType(acc_vec.data.elt[e]);
  }
  stg_vec.store_to(thread_out_base, 0);
}

void populate_transpose_dbias_workspace_config(const Tensor &input, /*cast*/
                                                    Tensor* workspace,
                                                    const int nvec_out) {
  const size_t row_length = input.data.shape[1];
  const size_t num_rows   = input.data.shape[0];

  const size_t tile_size_y = (nvec_out * THREADS_PER_WARP);
  NVTE_CHECK(num_rows % nvec_out == 0, "Unsupported shape.");

  const size_t num_rows_partial_dbias = DIVUP(num_rows, tile_size_y);

  workspace->data.shape = {num_rows_partial_dbias, row_length};
  workspace->data.dtype = DType::kFloat32;
}

template <typename BiasType>
void reduce_dbias(const Tensor &workspace, Tensor *dbias,
                  const size_t row_length, const size_t num_rows, const int nvec_out,
                  cudaStream_t stream) {
  constexpr int reduce_dbias_store_bytes  = 8;  // stg.64
  constexpr int reduce_dbias_nvec         = reduce_dbias_store_bytes / sizeof(BiasType);

  NVTE_CHECK(row_length % reduce_dbias_nvec == 0, "Unsupported shape.");

  const size_t reduce_dbias_row_length = row_length;
  const size_t reduce_dbias_num_rows   = DIVUP(num_rows,
                                               static_cast<size_t>(nvec_out *
                                                                   THREADS_PER_WARP));
  const size_t reduce_dbias_num_blocks = DIVUP(row_length,
                                               reduce_dbias_num_threads * reduce_dbias_nvec);

  reduce_dbias_kernel<reduce_dbias_nvec, fp32, BiasType>
    <<<reduce_dbias_num_blocks,
    reduce_dbias_num_threads,
    0,
    stream>>>(
        reinterpret_cast<BiasType *>(dbias->data.dptr),
        reinterpret_cast<const fp32 *>(workspace.data.dptr),
        reduce_dbias_row_length,
        reduce_dbias_num_rows);
}

void fp8_transpose_dbias(const Tensor &input,
                          Tensor *transposed_output,
                          Tensor *dbias,
                          Tensor *workspace,
                          cudaStream_t stream) {
  CheckInputTensor(input, "fp8_transpose_dbias_input");
  CheckOutputTensor(*transposed_output, "transposed_output");
  CheckOutputTensor(*dbias, "dbias");

  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(transposed_output->data.shape.size() == 2, "T output must have 2 dimensions.");
  const size_t row_length = input.data.shape[1];
  const size_t num_rows = input.data.shape[0];

  NVTE_CHECK(transposed_output->data.shape[0] == row_length, "Wrong dimension of T output.");
  NVTE_CHECK(transposed_output->data.shape[1] == num_rows, "Wrong dimension of T output.");

  NVTE_CHECK(transposed_output->data.dtype == input.data.dtype, "T output must have the same type as input.");
  NVTE_CHECK(dbias->data.shape == std::vector<size_t>{ row_length }, "Wrong shape of DBias.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dbias->data.dtype, BiasType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(input.data.dtype, Type,
      constexpr int type_size = sizeof(Type);
      constexpr int nvec_in = desired_load_size / type_size;
      constexpr int nvec_out = desired_store_size / type_size;

      if (workspace->data.dptr == nullptr) {
        populate_transpose_dbias_workspace_config(input, workspace, nvec_out);
        return;
      }

      NVTE_CHECK(row_length % nvec_in  == 0, "Unsupported shape.");
      NVTE_CHECK(num_rows   % nvec_out == 0, "Unsupported shape.");
      const size_t n_tiles = DIVUP(row_length, static_cast<size_t>(nvec_in * THREADS_PER_WARP)) *
                             DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
      const size_t n_warps_per_block = cast_transpose_num_threads / THREADS_PER_WARP;
      const size_t n_blocks = DIVUP(n_tiles * n_warps_per_tile, n_warps_per_block);

      const bool full_tile = row_length % (nvec_in * THREADS_PER_WARP) == 0 &&
                             num_rows % (nvec_out * THREADS_PER_WARP) == 0;

      using ComputeType = fp32;
      constexpr size_t shared_size_transpose = cast_transpose_num_threads / n_warps_per_tile *
                                               (THREADS_PER_WARP + 1) *
                                               sizeof(Vec<Type, nvec_out>);
      constexpr size_t shared_size_dbias = cast_transpose_num_threads *
                                           sizeof(Vec<ComputeType, nvec_in>);
      static_assert(shared_size_transpose >= shared_size_dbias);
      using Param = TDBiasParam<Type, Type, ComputeType>;
      Param param;
      param.input     = reinterpret_cast<const Type *>(input.data.dptr);
      param.output_t  = reinterpret_cast<Type *>(transposed_output->data.dptr);
      param.scale_inv = reinterpret_cast<const ComputeType *>(transposed_output->scale_inv.dptr);
      param.workspace = reinterpret_cast<ComputeType *>(workspace->data.dptr);

      if (full_tile) {
        cudaFuncSetAttribute(transpose_dbias_kernel<nvec_in, nvec_out, Param>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
        transpose_dbias_kernel<nvec_in, nvec_out, Param>
          <<<n_blocks,
             cast_transpose_num_threads,
             shared_size_transpose,
             stream>>>(param, row_length, num_rows, n_tiles);
      } else {
        cudaFuncSetAttribute(transpose_dbias_kernel_notaligned<nvec_in, nvec_out, Param>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
        transpose_dbias_kernel_notaligned<nvec_in, nvec_out, Param>
          <<<n_blocks,
             cast_transpose_num_threads,
             shared_size_transpose,
             stream>>>(param, row_length, num_rows, n_tiles);
      }

      reduce_dbias<BiasType>(*workspace, dbias, row_length, num_rows, nvec_out, stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

namespace {

template <typename CType, typename IType>
__device__ inline CType dgelu_old(const IType val/*, const CType scale_inv*/) {
    CType cval = CType(val)/* * scale_inv*/;
    const CType tanh_out = tanhf(0.79788456f * cval * (1.f + 0.044715f * cval * cval));
    return 0.5f * cval * ((1.f - tanh_out * tanh_out) *
                          (0.79788456f + 0.1070322243f * cval * cval)) +
           0.5f * (1.f + tanh_out);
}
template <typename CType, typename IType, typename IType2>
__device__ inline CType dgelu(const IType val, const IType2 gelu_out, const CType scale_inv) {
    CType cval = val;
    CType y = CType(gelu_out) * scale_inv;
    const CType tanh_out = 2.f * y / cval - 1.f;
    return y/cval * (1.f + 2.f * (cval - y) * (0.79788456f + 0.1070322243f * cval * cval));
}

}  // namespace

template <int nvec_in, int nvec_out, typename Param>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
transpose_dbias_dgelu_kernel(const Param param,
                                  const size_t row_length,
                                  const size_t num_rows,
                                  const size_t num_tiles) {
  using IType = typename Param::InputType;
  using IType2 = typename Param::InputType2;
  using IType3 = typename Param::InputType3;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using IVec2 = Vec<IType2, nvec_in>;
  using IVec3 = Vec<IType3, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = row_length / (nvec_in * THREADS_PER_WARP);
  // const size_t num_tiles_y = num_rows / (nvec * THREADS_PER_WARP);
  const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                         warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType * const my_input_tile = param.input + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
    THREADS_PER_WARP;
  const IType2 * const my_gelu_input_tile = param.gelu_input +
                                            (tile_id_x * nvec_in +
                                             tile_id_y * row_length * nvec_out) *
                                            THREADS_PER_WARP;
  const IType3 * const my_gelu_output_tile = param.gelu_output +
                                            (tile_id_x * nvec_in +
                                             tile_id_y * row_length * nvec_out) *
                                            THREADS_PER_WARP;
  OType * const my_output_c_tile = param.output_c + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
                                                    THREADS_PER_WARP;
  OType * const my_output_t_tile = param.output_t + (tile_id_y * nvec_out +
                                                     tile_id_x * num_rows * nvec_in) *
                                                    THREADS_PER_WARP;
  CType * const my_partial_dbias_tile = param.workspace +
                                        (tile_id_x * (nvec_in * THREADS_PER_WARP) +
                                         tile_id_y * row_length);

  OVec * const my_scratch = reinterpret_cast<OVec *>(scratch) +
                            (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) *
                            (THREADS_PER_WARP + 1);

  CVec * const my_dbias_scratch = reinterpret_cast<CVec *>(scratch);

  IVec in[2][nvec_out];
  IVec2 gelu_in[2][nvec_out];
  IVec3 gelu_out[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];
  CVec partial_dbias;

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                           warp_id_in_tile * n_iterations) %
                          THREADS_PER_WARP;
  CType max = 0;
  const CType scale = param.scale_ptr != nullptr ? *param.scale_ptr : 1;
  const CType gelu_output_scale_inv = param.gelu_output_scale_inv != nullptr ? *param.gelu_output_scale_inv : 1;

  partial_dbias.clear();

#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
    gelu_in[0][i].load_from(my_gelu_input_tile, current_stride + my_place + stride * i);
    gelu_out[0][i].load_from(my_gelu_output_tile, current_stride + my_place + stride * i);
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const size_t current_place = current_stride + my_place;
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        in[current_in][j].load_from(my_input_tile,
                                    current_stride + my_place_in + stride * (nvec_out + j));
        gelu_in[current_in][j].load_from(my_gelu_input_tile,
                                         current_stride + my_place_in +
                                         stride * (nvec_out + j));
        gelu_out[current_in][j].load_from(my_gelu_output_tile,
                                         current_stride + my_place_in +
                                         stride * (nvec_out + j));
      }
    }
    CVec after_dgelu[nvec_out];  // NOLINT(*)
#pragma unroll
    for (unsigned int j = 0; j < nvec_out; ++j) {
#pragma unroll
      for (unsigned int k = 0; k < nvec_in; ++k) {
        after_dgelu[j].data.elt[k] = dgelu<CType>(gelu_in[current_in ^ 1][j].data.elt[k], gelu_out[current_in ^ 1][j].data.elt[k], gelu_output_scale_inv) *
                                     CType(in[current_in ^ 1][j].data.elt[k]);
      }
    }
    OVec out_trans[nvec_in];  // NOLINT(*)
    dgelu_and_transpose_regs_partial_dbias<true>(after_dgelu, out_trans,
                                                partial_dbias, my_output_c_tile,
                                                current_place, stride, max, scale,
                                                (my_id_in_warp + i +
                                                 warp_id_in_tile * n_iterations) %
                                                THREADS_PER_WARP,
                                                true);

#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      out_space[i][j].data.vec = out_trans[j].data.vec;
    }
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                  j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
               THREADS_PER_WARP;
    current_stride = i * output_stride +
                     warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                              current_stride + my_place);
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }

  my_dbias_scratch[threadIdx.x] = partial_dbias;
  __syncthreads();
  // TODO(ptredak): check if the regular reduction is better
  if (warp_id_in_tile == 0) {
#pragma unroll
    for (unsigned int i = 1; i < n_warps_per_tile; ++i) {
      CVec tmp = my_dbias_scratch[threadIdx.x + i * THREADS_PER_WARP];
#pragma unroll
      for (unsigned int j = 0; j < nvec_in; ++j) {
        partial_dbias.data.elt[j] += tmp.data.elt[j];
      }
    }

    partial_dbias.store_to(my_partial_dbias_tile, my_id_in_warp);
  }

  /* warp tile amax reduce*/
  max = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(max, warp_id);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    if (param.amax != nullptr) atomicMaxFloat(param.amax, max);
    if (param.scale_inv != nullptr) reciprocal<CType>(param.scale_inv, scale);
  }
}

template <int nvec_in, int nvec_out, typename Param>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
transpose_dbias_dgelu_kernel_notaligned(const Param param,
                                             const size_t row_length,
                                             const size_t num_rows,
                                             const size_t num_tiles) {
  using IType = typename Param::InputType;
  using IType2 = typename Param::InputType2;
  using IType3 = typename Param::InputType3;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using IVec2 = Vec<IType2, nvec_in>;
  using IVec3 = Vec<IType3, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = (row_length + nvec_in * THREADS_PER_WARP - 1) /
                             (nvec_in * THREADS_PER_WARP);
  const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                         warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType * const my_input_tile = param.input + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
                                                    THREADS_PER_WARP;
  const IType2 * const my_gelu_input_tile = param.gelu_input +
                                            (tile_id_x * nvec_in +
                                             tile_id_y * row_length * nvec_out) *
                                            THREADS_PER_WARP;
  const IType3 * const my_gelu_output_tile = param.gelu_output +
                                            (tile_id_x * nvec_in +
                                             tile_id_y * row_length * nvec_out) *
                                            THREADS_PER_WARP;
  OType * const my_output_c_tile = param.output_c + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
                                                    THREADS_PER_WARP;
  OType * const my_output_t_tile = param.output_t + (tile_id_y * nvec_out +
                                                     tile_id_x * num_rows * nvec_in) *
                                                    THREADS_PER_WARP;
  CType * const my_partial_dbias_tile = param.workspace +
                                        (tile_id_x * (nvec_in * THREADS_PER_WARP) +
                                         tile_id_y * row_length);

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  const size_t row_length_rest = stride - tile_id_x * THREADS_PER_WARP;
  const size_t row_height_rest = output_stride - tile_id_y * THREADS_PER_WARP;
  const unsigned int tile_length = row_length_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                      : row_length_rest;
  const unsigned int tile_height = row_height_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                      : row_height_rest;

  OVec * const my_scratch = reinterpret_cast<OVec *>(scratch) +
                            (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) *
                            (THREADS_PER_WARP + 1);

  CVec * const my_dbias_scratch = reinterpret_cast<CVec *>(scratch);

  IVec in[2][nvec_out];
  IVec2 gelu_in[2][nvec_out];
  IVec3 gelu_out[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];
  CVec partial_dbias;

  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                           warp_id_in_tile * n_iterations) %
                          THREADS_PER_WARP;
  CType max = 0;
  const CType scale = param.scale_ptr != nullptr ? *param.scale_ptr : 1;
  const CType gelu_output_scale_inv = param.gelu_output_scale_inv != nullptr ? *param.gelu_output_scale_inv : 1;

  partial_dbias.clear();

  {
    const bool valid_load = my_place < tile_length &&
                            warp_id_in_tile * n_iterations < tile_height;
#pragma unroll
    for (unsigned int i = 0; i < nvec_out; ++i) {
      if (valid_load) {
        in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
        gelu_in[0][i].load_from(my_gelu_input_tile, current_stride + my_place + stride * i);
        gelu_out[0][i].load_from(my_gelu_output_tile, current_stride + my_place + stride * i);
      } else {
        in[0][i].clear();
        gelu_in[0][i].clear();
        gelu_out[0][i].clear();
      }
    }
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const size_t current_place = current_stride + my_place;
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
      const bool valid_load = my_place_in < tile_length &&
                              warp_id_in_tile * n_iterations + i + 1 < tile_height;
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        if (valid_load) {
          in[current_in][j].load_from(my_input_tile,
                                      current_stride + my_place_in + stride * (nvec_out + j));
          gelu_in[current_in][j].load_from(my_gelu_input_tile,
                                           current_stride + my_place_in +
                                           stride * (nvec_out + j));
          gelu_out[current_in][j].load_from(my_gelu_output_tile,
                                           current_stride + my_place_in +
                                           stride * (nvec_out + j));
        } else {
          in[current_in][j].clear();
          gelu_in[current_in][j].clear();
          gelu_out[current_in][j].clear();
        }
      }
    }
    CVec after_dgelu[nvec_out];  // NOLINT(*)
#pragma unroll
    for (unsigned int j = 0; j < nvec_out; ++j) {
#pragma unroll
      for (unsigned int k = 0; k < nvec_in; ++k) {
        after_dgelu[j].data.elt[k] = dgelu<CType>(gelu_in[current_in ^ 1][j].data.elt[k], gelu_out[current_in ^ 1][j].data.elt[k], gelu_output_scale_inv) *
                                     CType(in[current_in ^ 1][j].data.elt[k]);
      }
    }
    OVec out_trans[nvec_in];  // NOLINT(*)
    const bool valid_store = my_place < tile_length &&
                             warp_id_in_tile * n_iterations + i < tile_height;
    dgelu_and_transpose_regs_partial_dbias<false>(after_dgelu, out_trans,
                                                partial_dbias, my_output_c_tile,
                                                current_place, stride, max, scale,
                                                (my_id_in_warp + i +
                                                 warp_id_in_tile * n_iterations) %
                                                THREADS_PER_WARP,
                                                valid_store);

#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      out_space[i][j].data.vec = out_trans[j].data.vec;
    }
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                  j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
               THREADS_PER_WARP;
    current_stride = i * output_stride +
                     warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; warp_id_in_tile * n_iterations + j < tile_length; ++j) {
      const bool valid_store = my_place < tile_height;
      if (valid_store) {
        my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                                current_stride + my_place);
      }
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }

  my_dbias_scratch[threadIdx.x] = partial_dbias;
  __syncthreads();
  // TODO(ptredak): check if the regular reduction is better
  if (warp_id_in_tile == 0) {
#pragma unroll
    for (unsigned int i = 1; i < n_warps_per_tile; ++i) {
      CVec tmp = my_dbias_scratch[threadIdx.x + i * THREADS_PER_WARP];
#pragma unroll
      for (unsigned int j = 0; j < nvec_in; ++j) {
        partial_dbias.data.elt[j] += tmp.data.elt[j];
      }
    }

    if (my_id_in_warp < tile_length) {
      partial_dbias.store_to(my_partial_dbias_tile, my_id_in_warp);
    }
  }

  /* warp tile amax reduce*/
  max = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(max, warp_id);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    if (param.amax != nullptr) atomicMaxFloat(param.amax, max);
    if (param.scale_inv != nullptr) reciprocal<CType>(param.scale_inv, scale);
  }
}

void transpose_dbias_dgelu(const Tensor &input,
                                const Tensor &gelu_input,
                                const Tensor &gelu_output,
                                Tensor *cast_output,
                                Tensor *transposed_output,
                                Tensor *dbias,
                                Tensor *workspace,
                                cudaStream_t stream) {
  CheckInputTensor(input, "cast_transpose_dbias_dgelu_input");
  CheckInputTensor(gelu_input, "gelu_input");
  CheckInputTensor(gelu_output, "gelu_output");
  CheckOutputTensor(*cast_output, "cast_output");
  CheckOutputTensor(*transposed_output, "transposed_output");
  CheckOutputTensor(*dbias, "dbias");

  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(cast_output->data.shape.size() == 2, "C output must have 2 dimensions.");
  NVTE_CHECK(transposed_output->data.shape.size() == 2,
             "T output must have 2 dimensions.");
  NVTE_CHECK(input.data.shape == cast_output->data.shape,
             "Input and C output must have the same shape.");
  const size_t row_length = input.data.shape[1];
  const size_t num_rows = input.data.shape[0];

  NVTE_CHECK(transposed_output->data.shape[0] == row_length, "Wrong dimension of T output.");
  NVTE_CHECK(transposed_output->data.shape[1] == num_rows, "Wrong dimension of T output.");

  NVTE_CHECK(cast_output->data.dtype == transposed_output->data.dtype,
             "C and T outputs need to have the same type.");
  NVTE_CHECK(cast_output->amax.dptr == transposed_output->amax.dptr,
             "C and T outputs need to share amax tensor.");
  NVTE_CHECK(cast_output->scale.dptr == transposed_output->scale.dptr,
             "C and T outputs need to share scale tensor.");
  NVTE_CHECK(cast_output->scale_inv.dptr == transposed_output->scale_inv.dptr,
             "C and T outputs need to share scale inverse tensor.");

  NVTE_CHECK(dbias->data.dtype == input.data.dtype, "DBias must have the same type as input.");
  NVTE_CHECK(dbias->data.shape == std::vector<size_t>{ row_length }, "Wrong shape of DBias.");

//  NVTE_CHECK(input.data.dtype == gelu_input.data.dtype, "Types of both inputs must match.");
  NVTE_CHECK(input.data.shape == gelu_input.data.shape, "Shapes of inputs must match.");
  NVTE_CHECK(input.data.shape == gelu_output.data.shape, "Shapes of inputs must match.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(gelu_output.data.dtype, InputType3,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(cast_output->data.dtype, OutputType,
        using InputType2 = InputType;
        /* dgelu fusion kernel uses more registers */
        constexpr int desired_load_size_dgelu = 4;
        constexpr int desired_store_size_dgelu = 4;
        constexpr int itype_size = sizeof(InputType);
        constexpr int otype_size = sizeof(OutputType);
        constexpr int nvec_in = desired_load_size_dgelu / itype_size;
        constexpr int nvec_out = desired_store_size_dgelu / otype_size;

        if (workspace->data.dptr == nullptr) {
          populate_transpose_dbias_workspace_config(input, workspace, nvec_out);
          return;
        }

        NVTE_CHECK(row_length % nvec_in  == 0, "Unsupported shape.");
        NVTE_CHECK(num_rows   % nvec_out == 0, "Unsupported shape.");
        const size_t n_tiles = DIVUP(row_length, static_cast<size_t>(nvec_in * THREADS_PER_WARP)) *
                               DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
        const size_t n_warps_per_block = cast_transpose_num_threads / THREADS_PER_WARP;
        const size_t n_blocks = DIVUP(n_tiles * n_warps_per_tile, n_warps_per_block);

        const bool full_tile = row_length % (nvec_in * THREADS_PER_WARP) == 0 &&
                               num_rows % (nvec_out * THREADS_PER_WARP) == 0;

        using ComputeType = fp32;
        constexpr size_t shared_size_transpose = cast_transpose_num_threads / n_warps_per_tile *
        (THREADS_PER_WARP + 1) *
        sizeof(Vec<OutputType, nvec_out>);
        constexpr size_t shared_size_dbias = cast_transpose_num_threads *
        sizeof(Vec<ComputeType, nvec_in>);
        static_assert(shared_size_transpose >= shared_size_dbias);
        using Param = TDBiasDGeluParam<InputType, InputType2, InputType3, OutputType, ComputeType>;
        Param param;
        param.input = reinterpret_cast<const InputType *>(input.data.dptr);
        param.gelu_input = reinterpret_cast<const InputType2 *>(gelu_input.data.dptr);
        param.gelu_output = reinterpret_cast<const InputType3 *>(gelu_output.data.dptr);
        param.gelu_output_scale_inv = reinterpret_cast<const ComputeType *>(gelu_output.scale_inv.dptr);
        param.output_c = reinterpret_cast<OutputType *>(cast_output->data.dptr);
        param.output_t = reinterpret_cast<OutputType *>(transposed_output->data.dptr);
        param.scale_ptr = reinterpret_cast<const ComputeType *>(cast_output->scale.dptr);
        param.amax = reinterpret_cast<ComputeType *>(cast_output->amax.dptr);
        param.scale_inv = reinterpret_cast<ComputeType *>(cast_output->scale_inv.dptr);
        param.workspace = reinterpret_cast<ComputeType *>(workspace->data.dptr);
        if (full_tile) {
          cudaFuncSetAttribute(transpose_dbias_dgelu_kernel<nvec_in, nvec_out, Param>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);
          transpose_dbias_dgelu_kernel<nvec_in, nvec_out, Param>
            <<<n_blocks,
            cast_transpose_num_threads,
            shared_size_transpose,
            stream>>>(param, row_length, num_rows, n_tiles);
        } else {
          cudaFuncSetAttribute(transpose_dbias_dgelu_kernel_notaligned<nvec_in, nvec_out, Param>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);
          transpose_dbias_dgelu_kernel_notaligned<nvec_in, nvec_out, Param>
            <<<n_blocks,
            cast_transpose_num_threads,
            shared_size_transpose,
            stream>>>(param, row_length, num_rows, n_tiles);
        }

        reduce_dbias<InputType>(*workspace, dbias, row_length, num_rows, nvec_out, stream);
      ); // NOLINT(*)
    ); // NOLINT(*)
  );  // NOLINT(*)
}

template <int nvec_in, int nvec_out, typename Param>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
fp8_dgelu_kernel(const Param param,
                                  const size_t row_length,
                                  const size_t num_rows,
                                  const size_t num_tiles) {
  using IType = typename Param::InputType;
  using IType2 = typename Param::InputType2;
//  using IType3 = typename Param::InputType3;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using IVec2 = Vec<IType2, nvec_in>;
//  using IVec3 = Vec<IType3, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = row_length / (nvec_in * THREADS_PER_WARP);
  // const size_t num_tiles_y = num_rows / (nvec * THREADS_PER_WARP);
  const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                         warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType * const my_input_tile = param.input + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
    THREADS_PER_WARP;
  const IType2 * const my_gelu_input_tile = param.gelu_input +
                                            (tile_id_x * nvec_in +
                                             tile_id_y * row_length * nvec_out) *
                                            THREADS_PER_WARP;
//  const IType3 * const my_gelu_output_tile = param.gelu_output +
//                                            (tile_id_x * nvec_in +
//                                             tile_id_y * row_length * nvec_out) *
//                                            THREADS_PER_WARP;
  OType * const my_output_c_tile = param.output_c + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
                                                    THREADS_PER_WARP;
//#define DOUBLE_BUF
#ifdef DOUBLE_BUF
  IVec in[2][nvec_out];
  IVec2 gelu_in[2][nvec_out];
#else
  IVec in[nvec_out];
  IVec2 gelu_in[nvec_out];
#endif
//  IVec3 gelu_out[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];

  const size_t stride = row_length / nvec_in;
  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                           warp_id_in_tile * n_iterations) %
                          THREADS_PER_WARP;
  CType max = 0;
  const CType scale = param.scale_ptr != nullptr ? *param.scale_ptr : 1;
//  const CType gelu_output_scale_inv = param.gelu_output_scale_inv != nullptr ? *param.gelu_output_scale_inv : 1;

#ifdef DOUBLE_BUF
#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
    gelu_in[0][i].load_from(my_gelu_input_tile, current_stride + my_place + stride * i);
//    gelu_out[0][i].load_from(my_gelu_output_tile, current_stride + my_place + stride * i);
  }
#else
  for (unsigned int i = 0; i < nvec_out; ++i) {
    in[i].load_from(my_input_tile, current_stride + my_place + stride * i);
    gelu_in[i].load_from(my_gelu_input_tile, current_stride + my_place + stride * i);
//    gelu_out[0][i].load_from(my_gelu_output_tile, current_stride + my_place + stride * i);
  }
#endif

#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const size_t current_place = current_stride + my_place;
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
#ifdef DOUBLE_BUF
    if (i < n_iterations - 1) {
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        in[current_in][j].load_from(my_input_tile,
                                    current_stride + my_place_in + stride * (nvec_out + j));
        gelu_in[current_in][j].load_from(my_gelu_input_tile,
                                         current_stride + my_place_in +
                                         stride * (nvec_out + j));
//        gelu_out[current_in][j].load_from(my_gelu_output_tile,
//                                         current_stride + my_place_in +
//                                         stride * (nvec_out + j));
      }
    }
#endif //DOUBLE_BUF
    CVec after_dgelu[nvec_out];  // NOLINT(*)
#pragma unroll
    for (unsigned int j = 0; j < nvec_out; ++j) {
#pragma unroll
      for (unsigned int k = 0; k < nvec_in; ++k) {
#ifdef DOUBLE_BUF
        after_dgelu[j].data.elt[k] = dgelu_old<CType>(gelu_in[current_in ^ 1][j].data.elt[k]/*, gelu_out[current_in ^ 1][j].data.elt[k], gelu_output_scale_inv*/) *
                                     CType(in[current_in ^ 1][j].data.elt[k]);
#else
        after_dgelu[j].data.elt[k] = dgelu_old<CType>(gelu_in[j].data.elt[k]/*, gelu_out[current_in ^ 1][j].data.elt[k], gelu_output_scale_inv*/) *
                                     CType(in[j].data.elt[k]);
#endif
      }
    }
    dgelu_regs<true, nvec_in>(after_dgelu,
                     my_output_c_tile,
                     current_place, stride, max, scale,
                     true);

#ifndef DOUBLE_BUF
    if (i < n_iterations - 1) {
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        in[j].load_from(my_input_tile,
                                    current_stride + my_place_in + stride * (nvec_out + j));
        gelu_in[j].load_from(my_gelu_input_tile,
                                         current_stride + my_place_in +
                                         stride * (nvec_out + j));
//        gelu_out[current_in][j].load_from(my_gelu_output_tile,
//                                         current_stride + my_place_in +
//                                         stride * (nvec_out + j));
      }
    }
#endif //DOUBLE_BUF
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
  }

  /* warp tile amax reduce*/
  max = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(max, warp_id);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    if (param.amax != nullptr) atomicMaxFloat(param.amax, max);
    if (param.scale_inv != nullptr) reciprocal<CType>(param.scale_inv, scale);
  }
}

template <int nvec_in, int nvec_out, typename Param>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
fp8_dgelu_kernel_notaligned(const Param param,
                                             const size_t row_length,
                                             const size_t num_rows,
                                             const size_t num_tiles) {
  using IType = typename Param::InputType;
  using IType2 = typename Param::InputType2;
//  using IType3 = typename Param::InputType3;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using IVec2 = Vec<IType2, nvec_in>;
//  using IVec3 = Vec<IType3, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = (row_length + nvec_in * THREADS_PER_WARP - 1) /
                             (nvec_in * THREADS_PER_WARP);
  const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                         warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType * const my_input_tile = param.input + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
                                                    THREADS_PER_WARP;
  const IType2 * const my_gelu_input_tile = param.gelu_input +
                                            (tile_id_x * nvec_in +
                                             tile_id_y * row_length * nvec_out) *
                                            THREADS_PER_WARP;
//  const IType3 * const my_gelu_output_tile = param.gelu_output +
//                                            (tile_id_x * nvec_in +
//                                             tile_id_y * row_length * nvec_out) *
//                                            THREADS_PER_WARP;
  OType * const my_output_c_tile = param.output_c + (tile_id_x * nvec_in +
                                                     tile_id_y * row_length * nvec_out) *
                                                    THREADS_PER_WARP;

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  const size_t row_length_rest = stride - tile_id_x * THREADS_PER_WARP;
  const size_t row_height_rest = output_stride - tile_id_y * THREADS_PER_WARP;
  const unsigned int tile_length = row_length_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                      : row_length_rest;
  const unsigned int tile_height = row_height_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                      : row_height_rest;

  IVec in[2][nvec_out];
  IVec2 gelu_in[2][nvec_out];
//  IVec3 gelu_out[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];

  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                           warp_id_in_tile * n_iterations) %
                          THREADS_PER_WARP;
  CType max = 0;
  const CType scale = param.scale_ptr != nullptr ? *param.scale_ptr : 1;
//  const CType gelu_output_scale_inv = param.gelu_output_scale_inv != nullptr ? *param.gelu_output_scale_inv : 1;


  {
    const bool valid_load = my_place < tile_length &&
                            warp_id_in_tile * n_iterations < tile_height;
#pragma unroll
    for (unsigned int i = 0; i < nvec_out; ++i) {
      if (valid_load) {
        in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
        gelu_in[0][i].load_from(my_gelu_input_tile, current_stride + my_place + stride * i);
//        gelu_out[0][i].load_from(my_gelu_output_tile, current_stride + my_place + stride * i);
      } else {
        in[0][i].clear();
        gelu_in[0][i].clear();
//        gelu_out[0][i].clear();
      }
    }
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const size_t current_place = current_stride + my_place;
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
      const bool valid_load = my_place_in < tile_length &&
                              warp_id_in_tile * n_iterations + i + 1 < tile_height;
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        if (valid_load) {
          in[current_in][j].load_from(my_input_tile,
                                      current_stride + my_place_in + stride * (nvec_out + j));
          gelu_in[current_in][j].load_from(my_gelu_input_tile,
                                           current_stride + my_place_in +
                                           stride * (nvec_out + j));
//          gelu_out[current_in][j].load_from(my_gelu_output_tile,
//                                           current_stride + my_place_in +
//                                           stride * (nvec_out + j));
        } else {
          in[current_in][j].clear();
          gelu_in[current_in][j].clear();
//          gelu_out[current_in][j].clear();
        }
      }
    }
    CVec after_dgelu[nvec_out];  // NOLINT(*)
#pragma unroll
    for (unsigned int j = 0; j < nvec_out; ++j) {
#pragma unroll
      for (unsigned int k = 0; k < nvec_in; ++k) {
        after_dgelu[j].data.elt[k] = dgelu_old<CType>(gelu_in[current_in ^ 1][j].data.elt[k]/*, gelu_out[current_in ^ 1][j].data.elt[k], gelu_output_scale_inv*/) *
                                     CType(in[current_in ^ 1][j].data.elt[k]);
      }
    }
    const bool valid_store = my_place < tile_length &&
                             warp_id_in_tile * n_iterations + i < tile_height;
    dgelu_regs<false, nvec_in>(after_dgelu,
                      my_output_c_tile,
                      current_place, stride, max, scale,
                      valid_store);

    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
  }

  /* warp tile amax reduce*/
  max = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(max, warp_id);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    if (param.amax != nullptr) atomicMaxFloat(param.amax, max);
    if (param.scale_inv != nullptr) reciprocal<CType>(param.scale_inv, scale);
  }
}

void fp8_dgelu(const Tensor &input,
                                const Tensor &gelu_input,
//                                const Tensor &gelu_output,
                                Tensor *cast_output,
                                cudaStream_t stream) {
  CheckInputTensor(input, "fp8_dgelu_input");
  CheckInputTensor(gelu_input, "gelu_input");
//  CheckInputTensor(gelu_output, "gelu_output");
  CheckOutputTensor(*cast_output, "cast_output");

  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(cast_output->data.shape.size() == 2, "C output must have 2 dimensions.");
  NVTE_CHECK(input.data.shape == cast_output->data.shape,
             "Input and C output must have the same shape.");
  const size_t row_length = input.data.shape[1];
  const size_t num_rows = input.data.shape[0];


//  NVTE_CHECK(input.data.dtype == gelu_input.data.dtype, "Types of both inputs must match.");
  NVTE_CHECK(input.data.shape == gelu_input.data.shape, "Shapes of inputs must match.");
//  NVTE_CHECK(input.data.shape == gelu_output.data.shape, "Shapes of inputs must match.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, InputType,
//    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(gelu_output.data.dtype, InputType3,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(cast_output->data.dtype, OutputType,
        using InputType2 = InputType;
        /* dgelu fusion kernel uses more registers */
        constexpr int desired_load_size_dgelu = 4;
        constexpr int desired_store_size_dgelu = 4;
        constexpr int itype_size = sizeof(InputType);
        constexpr int otype_size = sizeof(OutputType);
        constexpr int nvec_in = desired_load_size_dgelu / itype_size;
        constexpr int nvec_out = desired_store_size_dgelu / otype_size;

        NVTE_CHECK(row_length % nvec_in  == 0, "Unsupported shape.");
        NVTE_CHECK(num_rows   % nvec_out == 0, "Unsupported shape.");
        const size_t n_tiles = DIVUP(row_length, static_cast<size_t>(nvec_in * THREADS_PER_WARP)) *
                               DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
        const size_t n_warps_per_block = cast_transpose_num_threads / THREADS_PER_WARP;
        const size_t n_blocks = DIVUP(n_tiles * n_warps_per_tile, n_warps_per_block);

        const bool full_tile = row_length % (nvec_in * THREADS_PER_WARP) == 0 &&
                               num_rows % (nvec_out * THREADS_PER_WARP) == 0;

        using ComputeType = fp32;
        using Param = TDGeluParam<InputType, InputType2, /*InputType3,*/ OutputType, ComputeType>;
        Param param;
        param.input = reinterpret_cast<const InputType *>(input.data.dptr);
        param.gelu_input = reinterpret_cast<const InputType2 *>(gelu_input.data.dptr);
//        param.gelu_output = reinterpret_cast<const InputType3 *>(gelu_output.data.dptr);
//        param.gelu_output_scale_inv = reinterpret_cast<const ComputeType *>(gelu_output.scale_inv.dptr);
        param.output_c = reinterpret_cast<OutputType *>(cast_output->data.dptr);
        param.scale_ptr = reinterpret_cast<const ComputeType *>(cast_output->scale.dptr);
        param.amax = reinterpret_cast<ComputeType *>(cast_output->amax.dptr);
        param.scale_inv = reinterpret_cast<ComputeType *>(cast_output->scale_inv.dptr);
        if (full_tile) {
          cudaFuncSetAttribute(fp8_dgelu_kernel<nvec_in, nvec_out, Param>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);
          fp8_dgelu_kernel<nvec_in, nvec_out, Param>
            <<<n_blocks,
            cast_transpose_num_threads,
            1024,
            stream>>>(param, row_length, num_rows, n_tiles);
        } else {
          cudaFuncSetAttribute(fp8_dgelu_kernel_notaligned<nvec_in, nvec_out, Param>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);
          fp8_dgelu_kernel_notaligned<nvec_in, nvec_out, Param>
            <<<n_blocks,
            cast_transpose_num_threads,
            1024,
            stream>>>(param, row_length, num_rows, n_tiles);
        }

      ); // NOLINT(*)
//    ); // NOLINT(*)
  );  // NOLINT(*)
}
}  // namespace transformer_engine

void nvte_fp8_transpose_dbias(const NVTETensor input,
                               NVTETensor transposed_output,
                               NVTETensor dbias,
                               NVTETensor workspace,
                               cudaStream_t stream) {
  using namespace transformer_engine;
  fp8_transpose_dbias(*reinterpret_cast<const Tensor*>(input),
                       reinterpret_cast<Tensor*>(transposed_output),
                       reinterpret_cast<Tensor*>(dbias),
                       reinterpret_cast<Tensor*>(workspace),
                       stream);
}

void nvte_transpose_dbias_dgelu(const NVTETensor input,
                                const NVTETensor gelu_input,
                                const NVTETensor gelu_output,
                                NVTETensor dgelu_output,
                                NVTETensor transposed_output,
                                NVTETensor dbias,
                                NVTETensor workspace,
                                cudaStream_t stream) {
  using namespace transformer_engine;
  transpose_dbias_dgelu(*reinterpret_cast<const Tensor*>(input),
                             *reinterpret_cast<const Tensor*>(gelu_input),
                             *reinterpret_cast<const Tensor*>(gelu_output),
                             reinterpret_cast<Tensor*>(dgelu_output),
                             reinterpret_cast<Tensor*>(transposed_output),
                             reinterpret_cast<Tensor*>(dbias),
                             reinterpret_cast<Tensor*>(workspace),
                             stream);
}
void nvte_dgelu(const NVTETensor input,
                const NVTETensor gelu_input,
//                const NVTETensor gelu_output,
                NVTETensor dgelu_output,
                cudaStream_t stream) {
  using namespace transformer_engine;
  fp8_dgelu(*reinterpret_cast<const Tensor*>(input),
                             *reinterpret_cast<const Tensor*>(gelu_input),
//                             *reinterpret_cast<const Tensor*>(gelu_output),
                             reinterpret_cast<Tensor*>(dgelu_output),
                             stream);
}
