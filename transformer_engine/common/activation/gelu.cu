/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include "../utils.cuh"
#include "../common.h"
#include <cstdlib>
#include <../util/vectorized_pointwise.h>

namespace transformer_engine {

namespace detail {

struct GELUParam {
  const fp32 *scale_inv;
};

__device__ inline fp32 gelu(fp32 value, const GELUParam &) {
  return value * (0.5F + 0.5F * tanhf(value * (0.79788456F + 0.03567741F * value * value)));
}
__device__ inline fp32 gelu_dequantize(fp32 value, const GELUParam &p) {
  value = value * (*(p.scale_inv));
  return value * (0.5F + 0.5F * tanhf(value * (0.79788456F + 0.03567741F * value * value)));
}

}

void gelu_cast(const Tensor &input,
               Tensor *output,
               cudaStream_t stream) {
  CheckInputTensor(input, "gelu_input");
  CheckOutputTensor(*output, "gelu_output");
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.data.shape == output->data.shape, "Input and output shapes must match.");
  const size_t tot_elts = input.data.shape[1] * input.data.shape[0];

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->data.dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      detail::GELUParam p;
      p.scale_inv = reinterpret_cast<fp32*>(input.scale_inv.dptr);
      VectorizedUnaryKernelLauncher<nvec, detail::GELUParam, detail::gelu>(
        reinterpret_cast<const IType*>(input.data.dptr),
        reinterpret_cast<OType*>(output->data.dptr),
        reinterpret_cast<const fp32*>(output->scale.dptr),
        reinterpret_cast<fp32*>(output->scale_inv.dptr),
        reinterpret_cast<fp32*>(output->amax.dptr),
        tot_elts,
	p,
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

constexpr int gelu_kernel_threads = 512;

template <int nvec, bool aligned,
          typename ComputeType,
          typename Param,
          typename InputType,
          typename OutputType>
__launch_bounds__(gelu_kernel_threads)
__global__ void fp8_gelu_kernel(const InputType *input,
                             OutputType *output,
                             const ComputeType *scale,
                             ComputeType *scale_inv,
                             ComputeType *amax,
                             Param p,
                             const size_t N,
                             const size_t num_aligned_elements) {
  VectorizedLoader<InputType, nvec, aligned> loader(input, N);
  VectorizedStorer<OutputType, nvec, aligned> storer(output, N);
  ComputeType max = 0;
  ComputeType s = 0;
  if (scale != nullptr) s = *scale;
  if (blockIdx.x == 0 && threadIdx.x == 0 && scale_inv != nullptr) {
    reciprocal<ComputeType>(scale_inv, s);
  }
  const int warp_id = threadIdx.x / THREADS_PER_WARP;

  const size_t M = num_aligned_elements;

  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const ComputeType val = static_cast<ComputeType>(loader.separate()[i]) * (*(p.scale_inv));
      // ComputeType temp = OP(val, p);
      ComputeType t0 = 0.79788456F + 0.03567741F * val * val;
      ComputeType tanh = tanhf(val * t0);
      ComputeType temp = val * (0.5F + 0.5F * tanh);
      __builtin_assume(max >= 0);
      max = fmaxf(fabsf(temp), max);

      temp = temp * s;

      storer.separate()[i] = static_cast<OutputType>(temp);
    }
    storer.store(tid, N);
  }
  /* warp tile amax reduce*/
  max = reduce_max<gelu_kernel_threads / THREADS_PER_WARP>(max, warp_id);

  if (threadIdx.x == 0 && amax != nullptr) {
      static_assert(std::is_same<ComputeType, float>::value);
      atomicMaxFloat(amax, max);
  }
}

template <int nvec, typename Param,
          typename InputType,
          typename OutputType>
void fp8_gelu_fp8input(const InputType *input,
                       OutputType *output,
                       const fp32 *scale,
                       fp32 *scale_inv,
                       fp32 *amax,
                       const size_t N,
                       const Param params,
                       cudaStream_t stream) {
  if (N != 0) {
    auto align = CheckAlignment(N, nvec, input, output);

    size_t num_aligned_elements = get_num_aligned_elements(input, N, nvec,
                                                           sizeof(InputType));
    constexpr size_t threads = gelu_kernel_threads;
    size_t num_blocks = DIVUP(num_aligned_elements, threads);
    constexpr size_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    switch (align) {
      case Alignment::SAME_ALIGNED:
        fp8_gelu_kernel<nvec, true, fp32, Param><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, N, num_aligned_elements);
        break;
      case Alignment::SAME_UNALIGNED:
        fp8_gelu_kernel<nvec, false, fp32, Param><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, N, num_aligned_elements);
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        fp8_gelu_kernel<1, true, fp32, Param><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, N, N);
        break;
      }
    }
  }
}

void gelu_cast_fp8input(const Tensor &input,
               Tensor *output,
               cudaStream_t stream) {
  CheckInputTensor(input, "gelu_input");
  CheckOutputTensor(*output, "gelu_output");
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.data.shape == output->data.shape, "Input and output shapes must match.");
  const size_t tot_elts = input.data.shape[1] * input.data.shape[0];

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(input.data.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output->data.dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      detail::GELUParam p;
      p.scale_inv = reinterpret_cast<fp32*>(input.scale_inv.dptr);
      fp8_gelu_fp8input<nvec, detail::GELUParam>(
        reinterpret_cast<const IType*>(input.data.dptr),
        reinterpret_cast<OType*>(output->data.dptr),
        reinterpret_cast<const fp32*>(output->scale.dptr),
        reinterpret_cast<fp32*>(output->scale_inv.dptr),
        reinterpret_cast<fp32*>(output->amax.dptr),
        tot_elts,
        p,
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_gelu(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream) {
  using namespace transformer_engine;
  gelu_cast(*reinterpret_cast<const Tensor*>(input),
            reinterpret_cast<Tensor*>(output),
            stream);
}
void nvte_gelu_fp8input(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream) {
  using namespace transformer_engine;
  gelu_cast_fp8input(*reinterpret_cast<const Tensor*>(input),
            reinterpret_cast<Tensor*>(output),
            stream);
}
