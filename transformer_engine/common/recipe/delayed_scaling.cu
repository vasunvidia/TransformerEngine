/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <cmath>
#include <string>

#include "../common.h"
#include "../util/logging.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {
namespace delayed_scaling_recipe {

namespace {

// amax value to use for updating scaling factor
enum class AmaxComputeAlgo { INVALID, MOST_RECENT, MAX };

const char* dtype_name(DType dtype) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, Type,
    return TypeInfo<Type>::name;
  );  // NOLINT(*)
  return "";
}

// Maximum representable value of an FP8 dtype
inline float fp8_dtype_max(DType dtype) {
  switch (dtype) {
  case DType::kFloat8E4M3: return 448;
  case DType::kFloat8E5M2: return 57344;
  default:
    NVTE_ERROR("Expected FP8 dtype, but got ", dtype_name(dtype));
  }
  return 0;
}

// struct for amax parameters
struct AmaxParam {
  int num_scale = 0;
  float* amax_history = nullptr;
  float* scale = nullptr;
  float* scale_inv = nullptr;
  unsigned char* scale_inv_mask = nullptr;
};

// dummy struct for kernel_bulk's other params
struct OtherParams {
  float* a;
  size_t b;
  AmaxComputeAlgo c;
  float d;
};

#if CUDART_VERSION >= 12010
constexpr size_t max_constant_memory_per_kernel = 32768;
constexpr size_t AMAX_PARAMS_LIMIT = (
  max_constant_memory_per_kernel - sizeof(OtherParams)) / sizeof(AmaxParam);
#else
constexpr size_t max_constant_memory_per_kernel = 4096;
constexpr size_t AMAX_PARAMS_LIMIT = (
  max_constant_memory_per_kernel - sizeof(OtherParams)) / sizeof(AmaxParam);
#endif

struct AmaxParams {
  AmaxParam param[AMAX_PARAMS_LIMIT];
};

namespace amax_and_scale_update_impl {

// CUDA block size
constexpr size_t bsize = 256;

/* CUDA kernel to update amax history and FP8 scaling factors
 *
 * Block dims: bsize x 1 x 1
 *
 * Grid dims: num_scales x 1 x 1
 */
__global__ void __launch_bounds__(bsize)
kernel(const float* amax_history_ptr,
       const float* scale_ptr,
       const float* scale_inv_ptr,
       const unsigned char* scale_inv_mask_ptr,
       const float* skip_scale_inv_update_ptr,
       float* updated_amax_history_ptr,
       float* updated_scale_ptr,
       float* updated_scale_inv_ptr,
       size_t amax_history_length,
       size_t amax_history_stride,
       AmaxComputeAlgo amax_compute_algo,
       float scaled_max) {
  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;

  // Update amax
  float amax = 0;
  {
    // Roll amax history
    const auto* amax_history = amax_history_ptr + bid;
    auto* updated_amax_history = updated_amax_history_ptr + bid;
    const auto last_amax = amax_history[0];
    const auto& length = amax_history_length;
    const auto& stride = amax_history_stride;
    for (size_t off = 0; off < length; off += bsize) {
      const size_t i = off + tid;
      float a = 0;
      if (i < length) {
        a = (i < length - 1) ? amax_history[(i+1)*stride] : last_amax;
        amax = fmaxf(amax, a);
      }
      __syncthreads();  // In case roll is in-place
      if (i < length) {
        updated_amax_history[i*stride] = (i > 0) ? a : 0;
      }
    }

    // Compute amax to use for scaling factor
    switch (amax_compute_algo) {
    case AmaxComputeAlgo::MOST_RECENT:
      amax = last_amax;
      break;
    case AmaxComputeAlgo::MAX:
      {
        __shared__ float shared_amax[bsize];
        shared_amax[tid] = amax;
        __syncthreads();
#pragma unroll
        for (size_t off = bsize / 2; off > 0; off /= 2) {
          if (tid < off) {
            shared_amax[tid] = fmaxf(shared_amax[tid], shared_amax[tid + off]);
          }
          __syncthreads();
        }
        amax = shared_amax[tid];
      }
      break;
    default:
      amax = 0;
    }
  }

  // Update scale and scale inverse
  if (tid == 0) {
    // Update scale
    float scale;
    if (isfinite(amax) && amax > 0) {
      scale = scaled_max / amax;
    } else {
      scale = scale_ptr[bid];
    }
    updated_scale_ptr[bid] = scale;

    bool update_weight_scale_inv;
    if (skip_scale_inv_update_ptr == nullptr) {
      update_weight_scale_inv = scale_inv_mask_ptr == nullptr;
    } else {
      update_weight_scale_inv = skip_scale_inv_update_ptr[0] == 0.0f;
    }

    // Update scale inverse
    float scale_inv;
    if (update_weight_scale_inv || scale_inv_mask_ptr[bid]) {
      scale_inv = 1 / scale;
    } else {
      scale_inv = scale_inv_ptr[bid];
    }
    updated_scale_inv_ptr[bid] = scale_inv;
  }
}

/* CUDA kernel to bulk-update amax history and FP8 scaling factors
 *
 * Block dims: bsize x 1 x 1
 *
 * Grid dims: num_tensors x 1 x 1
 */
__global__ void __launch_bounds__(bsize)
kernel_bulk(
       float* amax_reduction_buffer,
       AmaxParams p,
       size_t amax_history_length,
       AmaxComputeAlgo amax_compute_algo,
       float scaled_max) {
  const size_t bid = blockIdx.x;
  const size_t tid = threadIdx.x;
  const int num_scale = p.param[bid].num_scale;

  for (int count = 0; count < num_scale; count++) {
    // Update amax
    float amax = 0;
    {
      // Roll amax history
      const auto& length = amax_history_length;
      const auto& stride = p.param[bid].num_scale;
      auto* amax_history = p.param[bid].amax_history+count;
      const auto last_amax = ((amax_reduction_buffer != nullptr)
            && (amax_reduction_buffer[bid*stride+count] != 0)) ?
            amax_reduction_buffer[bid*stride+count] : amax_history[0];
      for (size_t off = 0; off < length; off += bsize) {
        const size_t i = off + tid;
        float a = 0;
        if (i < length) {
          a = (i < length - 1) ? amax_history[(i+1)*stride] : last_amax;
          amax = fmaxf(amax, a);
        }
        __syncthreads();  // In case roll is in-place
        if (i < length) {
          amax_history[i*stride] = (i > 0) ? a : 0;
        }
      }

      // Compute amax to use for scaling factor
      switch (amax_compute_algo) {
      case AmaxComputeAlgo::MOST_RECENT:
        amax = last_amax;
        break;
      case AmaxComputeAlgo::MAX:
        {
          __shared__ float shared_amax[bsize];
          shared_amax[tid] = amax;
          __syncthreads();
#pragma unroll
          for (size_t off = bsize / 2; off > 0; off /= 2) {
            if (tid < off) {
              shared_amax[tid] = fmaxf(shared_amax[tid], shared_amax[tid + off]);
            }
            __syncthreads();
          }
          amax = shared_amax[tid];
        }
        break;
      default:
        amax = 0;
      }
    }

    // Update scale and scale inverse
    if (tid == 0) {
      // Update scale
      float scale;
      if (isfinite(amax) && amax > 0) {
        scale = scaled_max / amax;
      } else {
        scale = p.param[bid].scale[count];
      }
      p.param[bid].scale[count] = scale;
      // Update scale inverse
      float scale_inv;
      if (p.param[bid].scale_inv_mask == nullptr || p.param[bid].scale_inv_mask[count]) {
        scale_inv = 1 / scale;
      } else {
        scale_inv = p.param[bid].scale_inv[count];
      }
      p.param[bid].scale_inv[count] = scale_inv;
    }
  }
}

}  // namespace amax_and_scale_update_impl


}  // namespace

void amax_and_scale_update(const Tensor &amax_history,
                           const Tensor &scale,
                           const Tensor &scale_inv,
                           const Tensor &scale_inv_mask,
                           const Tensor &skip_scale_inv_update,
                           Tensor *updated_amax_history_,
                           Tensor *updated_scale_,
                           Tensor *updated_scale_inv_,
                           const std::string &amax_compute_algo,
                           DType fp8_dtype,
                           float margin,
                           cudaStream_t stream) {
  auto& updated_amax_history = *updated_amax_history_;
  auto& updated_scale = *updated_scale_;
  auto& updated_scale_inv = *updated_scale_inv_;

  // Number of elements in tensor
  auto numel = [] (const Tensor &tensor) -> size_t {
    size_t acc = 1;
    for (const auto& dim : tensor.data.shape) {
      acc *= dim;
    }
    return acc;
  };

  // Check tensors
  NVTE_CHECK(amax_history.data.shape.size() == 2,
             "Found ", amax_history.data.shape.size(), " dims");
  const size_t amax_history_length = amax_history.data.shape[0];
  const size_t num_scales = amax_history.data.shape[1];
  NVTE_CHECK(amax_history.data.dtype == DType::kFloat32,
             "Found ", dtype_name(amax_history.data.dtype), ".");
  NVTE_CHECK(numel(scale) == num_scales,
             "Expected ", num_scales, " elements, ",
             "but found ", numel(scale), ".");
  NVTE_CHECK(scale.data.dtype == DType::kFloat32,
             "Found ", dtype_name(scale.data.dtype), ".");
  if (scale_inv_mask.data.dptr != nullptr) {
    NVTE_CHECK(numel(scale_inv) == num_scales,
               "Expected ", num_scales, " elements, ",
               "but found ", numel(scale_inv), ".");
    NVTE_CHECK(scale_inv.data.dtype == DType::kFloat32);
    NVTE_CHECK(numel(scale_inv_mask) == num_scales,
               "Expected ", num_scales, " elements, ",
               "but found ", numel(scale_inv_mask), ".");
    NVTE_CHECK(scale_inv_mask.data.dtype == DType::kByte,
               "Found ", dtype_name(scale_inv_mask.data.dtype), ".");
  }
  if (skip_scale_inv_update.data.dptr != nullptr) {
    NVTE_CHECK(numel(skip_scale_inv_update) == 1,
               "Expected 1 element, ",
               "but found ", numel(skip_scale_inv_update), ".");
    NVTE_CHECK(skip_scale_inv_update.data.dtype == DType::kFloat32);
    NVTE_CHECK(scale_inv_mask.data.dptr != nullptr);
  }
  NVTE_CHECK(updated_amax_history.data.shape.size() == 2,
             "Found ", updated_amax_history.data.shape.size(), " dims.");
  NVTE_CHECK(updated_amax_history.data.shape[0] == amax_history_length,
             "Expected ", amax_history_length, ", ",
             "but found ", updated_amax_history.data.shape[0]);
  NVTE_CHECK(updated_amax_history.data.shape[1] == num_scales,
             "Expected ", num_scales, ", ",
             "but found ", updated_amax_history.data.shape[1]);
  NVTE_CHECK(updated_amax_history.data.dtype == DType::kFloat32,
             "Got ", dtype_name(updated_amax_history.data.dtype), ".");
  NVTE_CHECK(numel(updated_scale) == num_scales,
             "Expected ", num_scales, " elements, ",
             "but found ", numel(updated_scale), ".");
  NVTE_CHECK(updated_scale.data.dtype == DType::kFloat32,
             "Got ", dtype_name(updated_scale.data.dtype), ".");
  NVTE_CHECK(numel(updated_scale_inv) == num_scales,
             "Expected ", num_scales, " elements, ",
             "but found ", numel(updated_scale_inv), ".");
  NVTE_CHECK(updated_scale_inv.data.dtype == DType::kFloat32,
             "Got ", dtype_name(updated_scale_inv.data.dtype), ".");

  // amax value to use for updating scaling factor
  AmaxComputeAlgo amax_compute_algo_ = AmaxComputeAlgo::INVALID;
  if (amax_compute_algo == "max") {
    amax_compute_algo_ = AmaxComputeAlgo::MAX;
  } else if (amax_compute_algo == "most_recent") {
    amax_compute_algo_ = AmaxComputeAlgo::MOST_RECENT;
  } else {
    NVTE_ERROR("Unsupported amax compute algorithm (", amax_compute_algo, ")");
  }

  // Expected maximum value after scale is applied
  const float scaled_max = fp8_dtype_max(fp8_dtype) * std::pow(2.f, -margin);

  // Launch CUDA kernel
  constexpr size_t block_size = amax_and_scale_update_impl::bsize;
  const size_t grid_size = num_scales;
  amax_and_scale_update_impl::kernel
    <<<grid_size, block_size, 0, stream>>>(
      static_cast<const float*>(amax_history.data.dptr),
      static_cast<const float*>(scale.data.dptr),
      static_cast<const float*>(scale_inv.data.dptr),
      static_cast<const unsigned char*>(scale_inv_mask.data.dptr),
      static_cast<const float*>(skip_scale_inv_update.data.dptr),
      static_cast<float*>(updated_amax_history.data.dptr),
      static_cast<float*>(updated_scale.data.dptr),
      static_cast<float*>(updated_scale_inv.data.dptr),
      amax_history_length,
      num_scales,
      amax_compute_algo_,
      scaled_max);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void amax_and_scale_update_after_reduction(const Tensor &amax_reduction_buffer,
                           std::vector<Tensor*> amax_histories,
                           std::vector<Tensor*> scales,
                           std::vector<Tensor*> scale_invs,
                           std::vector<Tensor*> scale_inv_masks,
                           const std::string &amax_compute_algo,
                           DType fp8_dtype,
                           float margin,
                           cudaStream_t stream) {
  using namespace transformer_engine;
  // get sm and cuda version
  const int device_id = cuda::current_device();
  const int sm_arch_ = cuda::sm_arch(device_id);
  int cuda_runtime_version = 0;
  cudaRuntimeGetVersion(&cuda_runtime_version);

  // amax value to use for updating scaling factor
  AmaxComputeAlgo amax_compute_algo_ = AmaxComputeAlgo::INVALID;
  if (amax_compute_algo == "max") {
    amax_compute_algo_ = AmaxComputeAlgo::MAX;
  } else if (amax_compute_algo == "most_recent") {
    amax_compute_algo_ = AmaxComputeAlgo::MOST_RECENT;
  } else {
    NVTE_ERROR("Unsupported amax compute algorithm (", amax_compute_algo, ")");
  }

  // Expected maximum value after scale is applied
  const float scaled_max = fp8_dtype_max(fp8_dtype) * std::pow(2.f, -margin);

  // Number of elements in tensor
  auto numel = [] (const Tensor *tensor) -> size_t {
    size_t acc = 1;
    for (const auto& dim : tensor->data.shape) {
      acc *= dim;
    }
    return acc;
  };

  // Number of tensors in the bulk
  const size_t num_tensors = amax_histories.size();
  const int num_kernels = (num_tensors+AMAX_PARAMS_LIMIT-1)/AMAX_PARAMS_LIMIT;
  size_t amax_history_length = 0;
  if (num_tensors > 0) {
    amax_history_length = amax_histories[0]->data.shape[0];
  }

  // amax parameters
  float* amax_buffer = static_cast<float*>(amax_reduction_buffer.data.dptr);
  AmaxParams p;
  for (int iter = 0; iter < num_kernels; iter++) {
    size_t kernel_num_scales = 0;
    size_t kernel_num_tensors = (iter == (num_kernels -1))
          ? num_tensors % AMAX_PARAMS_LIMIT: AMAX_PARAMS_LIMIT;
    for (size_t pi = 0; pi < kernel_num_tensors; pi++) {
      size_t i = iter * AMAX_PARAMS_LIMIT + pi;

      // Check tensors
      int num_scale = amax_histories[i]->data.shape[1];
      NVTE_CHECK(amax_histories[i]->data.dtype == DType::kFloat32,
                 "Found ", dtype_name(amax_histories[i]->data.dtype), ".");
      NVTE_CHECK(amax_histories[i]->data.shape.size() == 2,
                 "Found ", amax_histories[i]->data.shape.size(), " dims");
      NVTE_CHECK(numel(amax_histories[i]) == amax_history_length * num_scale,
                 "Expected ", amax_history_length * num_scale, " elements, ",
                 "but found ", numel(amax_histories[i]), ".");
      NVTE_CHECK(scales[i]->data.dtype == DType::kFloat32,
                 "Found ", dtype_name(scales[i]->data.dtype), ".");
      NVTE_CHECK(scales[i]->data.shape.size() == 1,
                 "Found ", scales[i]->data.shape.size(), " dims");
      NVTE_CHECK(numel(scales[i]) == num_scale,
                 "Expected ", num_scale, " elements, ",
                 "Found ", numel(scales[i]), ".");
      if (scale_inv_masks[i]->data.dptr != nullptr) {
        NVTE_CHECK(scale_invs[i]->data.dtype == DType::kFloat32,
                   "Found ", dtype_name(scale_invs[i]->data.dtype), ".");
        NVTE_CHECK(scale_invs[i]->data.shape.size() == 1,
                   "Found ", scale_invs[i]->data.shape.size(), " dims");
        NVTE_CHECK(numel(scale_invs[i]) == num_scale,
                   "Expected ", num_scale, " elements, ",
                   "but found ", numel(scale_invs[i]), ".");
        NVTE_CHECK(scale_inv_masks[i]->data.dtype == DType::kByte,
                   "Found ", dtype_name(scale_inv_masks[i]->data.dtype), ".");
        NVTE_CHECK(scale_inv_masks[i]->data.shape.size() == 1,
                   "Found ", scale_inv_masks[i]->data.shape.size(), " dims");
        NVTE_CHECK(numel(scale_inv_masks[i]) == num_scale,
                   "Expected ", num_scale, " elements, ",
                   "but found ", numel(scale_inv_masks[i]), ".");
      }

      // amax parameters
      kernel_num_scales += num_scale;
      p.param[pi].num_scale = num_scale;
      p.param[pi].amax_history = static_cast<float*>(amax_histories[i]->data.dptr);
      p.param[pi].scale = static_cast<float*>(scales[i]->data.dptr);
      p.param[pi].scale_inv = static_cast<float*>(scale_invs[i]->data.dptr);
      p.param[pi].scale_inv_mask = static_cast<unsigned char*>(scale_inv_masks[i]->data.dptr);
    }

    // Launch CUDA kernel
    size_t grid_size = kernel_num_tensors;
    const size_t block_size = amax_and_scale_update_impl::bsize;
    amax_and_scale_update_impl::kernel_bulk
      <<<grid_size, block_size, 0, stream>>>(
        amax_buffer,
        p,
        amax_history_length,
        amax_compute_algo_,
        scaled_max);
    NVTE_CHECK_CUDA(cudaGetLastError());

    // shift amax buffer pointer
    if (amax_buffer != nullptr) {
      amax_buffer += kernel_num_scales;
    }
  }
}


}  // namespace delayed_scaling_recipe
}  // namespace transformer_engine

void nvte_delayed_scaling_recipe_amax_and_scale_update(const NVTETensor amax_history,
                                                       const NVTETensor scale,
                                                       const NVTETensor scale_inv,
                                                       const NVTETensor scale_inv_mask,
                                                       const NVTETensor skip_scale_inv_update,
                                                       NVTETensor updated_amax_history,
                                                       NVTETensor updated_scale,
                                                       NVTETensor updated_scale_inv,
                                                       const char *amax_compute_algo,
                                                       NVTEDType fp8_dtype,
                                                       float margin,
                                                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_delayed_scaling_recipe_amax_and_scale_update);
  using namespace transformer_engine;
  delayed_scaling_recipe::amax_and_scale_update(
    *reinterpret_cast<const Tensor*>(amax_history),
    *reinterpret_cast<const Tensor*>(scale),
    *reinterpret_cast<const Tensor*>(scale_inv),
    *reinterpret_cast<const Tensor*>(scale_inv_mask),
    *reinterpret_cast<const Tensor*>(skip_scale_inv_update),
    reinterpret_cast<Tensor*>(updated_amax_history),
    reinterpret_cast<Tensor*>(updated_scale),
    reinterpret_cast<Tensor*>(updated_scale_inv),
    amax_compute_algo,
    static_cast<DType>(fp8_dtype),
    margin,
    stream);
}

void nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
                           const NVTETensor amax_reduction_buffer,
                           std::vector<NVTETensor> amax_histories,
                           std::vector<NVTETensor> scales,
                           std::vector<NVTETensor> scale_invs,
                           std::vector<NVTETensor> scale_inv_masks,
                           const char *amax_compute_algo,
                           NVTEDType fp8_dtype,
                           float margin,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction);
  using namespace transformer_engine;
  size_t num_tensors = amax_histories.size();
  std::vector<Tensor*> t_amax_histories, t_scales, t_scale_invs, t_scale_inv_masks;
  for (size_t i = 0; i < num_tensors; i++) {
    t_amax_histories.push_back(reinterpret_cast<Tensor*>(amax_histories[i]));
    t_scales.push_back(reinterpret_cast<Tensor*>(scales[i]));
    t_scale_invs.push_back(reinterpret_cast<Tensor*>(scale_invs[i]));
    t_scale_inv_masks.push_back(reinterpret_cast<Tensor*>(scale_inv_masks[i]));
  }
  delayed_scaling_recipe::amax_and_scale_update_after_reduction(
    *reinterpret_cast<const Tensor*>(amax_reduction_buffer),
    t_amax_histories,
    t_scales,
    t_scale_invs,
    t_scale_inv_masks,
    amax_compute_algo,
    static_cast<DType>(fp8_dtype),
    margin,
    stream);
}
