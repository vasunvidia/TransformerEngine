/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "../common.h"
#include "adam_flat.cuh"

static_assert(transformer_engine::multi_tensor_adam_flat::kMaxGroups ==
                  NVTE_MULTI_TENSOR_ADAM_FLAT_MAX_GROUPS,
              "kMaxGroups must match NVTE_MULTI_TENSOR_ADAM_FLAT_MAX_GROUPS");
static_assert(transformer_engine::multi_tensor_adam_flat::kNumHParams ==
                  NVTE_MULTI_TENSOR_ADAM_FLAT_NUM_HPARAMS,
              "kNumHParams must match NVTE_MULTI_TENSOR_ADAM_FLAT_NUM_HPARAMS");

namespace transformer_engine {
namespace multi_tensor_adam_flat {

namespace {

// One tile of kBlockSize * kVec elements per CUDA block: the fastest of the configurations
// measured on GB300 (about 7 TB/s; persistent grids and larger tiles were not faster).
constexpr int kBlockSize = 256;
constexpr int kUnroll = 1;

bool has_data(const Tensor *t) { return t != nullptr && t->data.dptr != nullptr; }

bool is_aligned(const void *ptr, size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <int kVec, typename GradT, typename ParamT, bool kWriteParam, bool kDeviceHParams>
void launch(const Tensor &grad, Tensor *param, Tensor &master_param, Tensor &exp_avg,
            Tensor &exp_avg_sq, const Tensor &segments, const float *grad_scale,
            const GroupArgs &groups, const float *hparams, const int mode, cudaStream_t stream) {
  constexpr int64_t kTile = static_cast<int64_t>(kBlockSize) * kVec * kUnroll;
  constexpr int64_t kMaxBlocks = std::numeric_limits<int32_t>::max();
  const int64_t numel = static_cast<int64_t>(grad.numel());
  const int64_t num_blocks = std::min(DIVUP(numel, kTile), kMaxBlocks);
  ParamT *param_ptr = kWriteParam ? reinterpret_cast<ParamT *>(param->data.dptr) : nullptr;
  adam_flat_kernel<kBlockSize, kVec, kUnroll, GradT, ParamT, kWriteParam, kDeviceHParams>
      <<<num_blocks, kBlockSize, 0, stream>>>(
          reinterpret_cast<const GradT *>(grad.data.dptr), param_ptr,
          reinterpret_cast<float *>(master_param.data.dptr),
          reinterpret_cast<float *>(exp_avg.data.dptr),
          reinterpret_cast<float *>(exp_avg_sq.data.dptr),
          reinterpret_cast<const int64_t *>(segments.data.dptr),
          static_cast<int64_t>(segments.data.shape[0]), numel, grad_scale, groups, mode, hparams);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename GradT, typename ParamT, bool kWriteParam, bool kDeviceHParams>
void dispatch_vec(const Tensor &grad, Tensor *param, Tensor &master_param, Tensor &exp_avg,
                  Tensor &exp_avg_sq, const Tensor &segments, const float *grad_scale,
                  const GroupArgs &groups, const float *hparams, const int mode,
                  cudaStream_t stream) {
  // Tiles start at multiples of the vector size, so aligned base pointers give aligned vectors.
  constexpr int kVec = 4;
  bool aligned = is_aligned(grad.data.dptr, kVec * sizeof(GradT)) &&
                 is_aligned(master_param.data.dptr, kVec * sizeof(float)) &&
                 is_aligned(exp_avg.data.dptr, kVec * sizeof(float)) &&
                 is_aligned(exp_avg_sq.data.dptr, kVec * sizeof(float));
  if constexpr (kWriteParam) {
    aligned = aligned && is_aligned(param->data.dptr, kVec * sizeof(ParamT));
  }
  if (aligned) {
    launch<kVec, GradT, ParamT, kWriteParam, kDeviceHParams>(grad, param, master_param, exp_avg,
                                                             exp_avg_sq, segments, grad_scale,
                                                             groups, hparams, mode, stream);
  } else {
    launch<1, GradT, ParamT, kWriteParam, kDeviceHParams>(grad, param, master_param, exp_avg,
                                                          exp_avg_sq, segments, grad_scale, groups,
                                                          hparams, mode, stream);
  }
}

// Checks shared by both variants. Returns the gradient-scale pointer (nullptr: no scaling).
const float *check_buffers(const Tensor &grad, const Tensor *param, const Tensor &master_param,
                           const Tensor &exp_avg, const Tensor &exp_avg_sq, const Tensor &segments,
                           const Tensor *grad_scale, const size_t num_groups, const int mode) {
  const size_t numel = grad.numel();
  NVTE_CHECK(master_param.numel() == numel, "Master parameters have ", master_param.numel(),
             " elements, but gradients have ", numel);
  NVTE_CHECK(exp_avg.numel() == numel, "First moments have ", exp_avg.numel(),
             " elements, but gradients have ", numel);
  NVTE_CHECK(exp_avg_sq.numel() == numel, "Second moments have ", exp_avg_sq.numel(),
             " elements, but gradients have ", numel);
  NVTE_CHECK(master_param.dtype() == DType::kFloat32, "Master parameters must be Float32, but got ",
             to_string(master_param.dtype()));
  NVTE_CHECK(exp_avg.dtype() == DType::kFloat32, "First moments must be Float32, but got ",
             to_string(exp_avg.dtype()));
  NVTE_CHECK(exp_avg_sq.dtype() == DType::kFloat32, "Second moments must be Float32, but got ",
             to_string(exp_avg_sq.dtype()));
  if (has_data(param)) {
    NVTE_CHECK(param->numel() == numel, "Model parameters have ", param->numel(),
               " elements, but gradients have ", numel);
  }
  NVTE_CHECK(segments.dtype() == DType::kInt64, "Segments must be Int64, but got ",
             to_string(segments.dtype()));
  NVTE_CHECK(segments.data.shape.size() == 2 && segments.data.shape[1] == 3,
             "Segments must have shape [num_segments, 3]");
  const float *grad_scale_ptr = nullptr;
  if (has_data(grad_scale)) {
    NVTE_CHECK(grad_scale->dtype() == DType::kFloat32 && grad_scale->numel() == 1,
               "Gradient scale must be a Float32 tensor with one element");
    grad_scale_ptr = reinterpret_cast<const float *>(grad_scale->data.dptr);
  }
  NVTE_CHECK(num_groups > 0 && num_groups <= static_cast<size_t>(kMaxGroups),
             "Number of parameter groups must be in [1, ", kMaxGroups, "], but got ", num_groups);
  NVTE_CHECK(mode == 0 || mode == 1, "Adam mode must be 0 (L2) or 1 (AdamW), but got ", mode);
  return grad_scale_ptr;
}

// Group hyperparameters. The bias corrections are computed on the host exactly as in
// multi_tensor_adam_cuda, so the update matches the non-capturable multi-tensor Adam.
GroupArgs make_group_args(const float *lr, const float *beta1, const float *beta2,
                          const float *epsilon, const float *weight_decay, const int *step,
                          const int *bias_correction, const size_t num_groups) {
  NVTE_CHECK(num_groups > 0 && num_groups <= static_cast<size_t>(kMaxGroups),
             "Number of parameter groups must be in [1, ", kMaxGroups, "], but got ", num_groups);
  GroupArgs groups{};
  groups.num_groups = static_cast<int>(num_groups);
  for (size_t i = 0; i < num_groups; i++) {
    float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
    if (bias_correction[i] == 1) {
      bias_correction1 = 1 - std::pow(beta1[i], step[i]);
      bias_correction2 = 1 - std::pow(beta2[i], step[i]);
    }
    groups.lr[i] = lr[i];
    groups.beta1[i] = beta1[i];
    groups.beta2[i] = beta2[i];
    groups.beta1_correction[i] = bias_correction1;
    groups.beta2_correction[i] = bias_correction2;
    groups.epsilon[i] = epsilon[i];
    groups.weight_decay[i] = weight_decay[i];
  }
  return groups;
}

template <bool kDeviceHParams>
void dispatch(const Tensor &grad, Tensor *param, Tensor &master_param, Tensor &exp_avg,
              Tensor &exp_avg_sq, const Tensor &segments, const float *grad_scale_ptr,
              const GroupArgs &groups, const float *hparams, const int mode,
              cudaStream_t stream) {
  if (grad.numel() == 0 || segments.data.shape[0] == 0) {
    return;
  }
  const bool write_param = has_data(param);
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      grad.dtype(), grad_type,
      if (write_param) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            param->dtype(), param_type,
            dispatch_vec<grad_type, param_type, true, kDeviceHParams>(
                grad, param, master_param, exp_avg, exp_avg_sq, segments, grad_scale_ptr, groups,
                hparams, mode, stream););
      } else {
        dispatch_vec<grad_type, float, false, kDeviceHParams>(
            grad, nullptr, master_param, exp_avg, exp_avg_sq, segments, grad_scale_ptr, groups,
            hparams, mode, stream);
      });
}

}  // namespace

void multi_tensor_adam_flat_cuda(const Tensor &grad, Tensor *param, Tensor &master_param,
                                 Tensor &exp_avg, Tensor &exp_avg_sq, const Tensor &segments,
                                 const Tensor *grad_scale, const float *lr, const float *beta1,
                                 const float *beta2, const float *epsilon,
                                 const float *weight_decay, const int *step,
                                 const int *bias_correction, const size_t num_groups,
                                 const int mode, cudaStream_t stream) {
  const float *grad_scale_ptr = check_buffers(grad, param, master_param, exp_avg, exp_avg_sq,
                                              segments, grad_scale, num_groups, mode);
  const GroupArgs groups = make_group_args(lr, beta1, beta2, epsilon, weight_decay, step,
                                           bias_correction, num_groups);
  dispatch<false>(grad, param, master_param, exp_avg, exp_avg_sq, segments, grad_scale_ptr, groups,
                  nullptr, mode, stream);
}

void multi_tensor_adam_flat_capturable_cuda(const Tensor &grad, Tensor *param,
                                            Tensor &master_param, Tensor &exp_avg,
                                            Tensor &exp_avg_sq, const Tensor &segments,
                                            const Tensor *grad_scale, const Tensor &hparams,
                                            const int mode, cudaStream_t stream) {
  NVTE_CHECK(hparams.dtype() == DType::kFloat32, "Hyperparameters must be Float32, but got ",
             to_string(hparams.dtype()));
  NVTE_CHECK(hparams.data.shape.size() == 2 &&
                 hparams.data.shape[0] == static_cast<size_t>(kNumHParams),
             "Hyperparameters must have shape [", static_cast<int>(kNumHParams),
             ", num_groups]");
  NVTE_CHECK(hparams.data.dptr != nullptr, "Hyperparameters must not be empty");
  const size_t num_groups = hparams.data.shape[1];
  const float *grad_scale_ptr = check_buffers(grad, param, master_param, exp_avg, exp_avg_sq,
                                              segments, grad_scale, num_groups, mode);
  GroupArgs groups{};
  groups.num_groups = static_cast<int>(num_groups);
  dispatch<true>(grad, param, master_param, exp_avg, exp_avg_sq, segments, grad_scale_ptr, groups,
                 reinterpret_cast<const float *>(hparams.data.dptr), mode, stream);
}

void multi_tensor_adam_flat_hparams(const float *lr, const float *beta1, const float *beta2,
                                    const float *epsilon, const float *weight_decay,
                                    const int *step, const int *bias_correction,
                                    const size_t num_groups, float *hparams) {
  const GroupArgs groups = make_group_args(lr, beta1, beta2, epsilon, weight_decay, step,
                                           bias_correction, num_groups);
  const size_t n = num_groups;
  for (size_t i = 0; i < n; i++) {
    hparams[kLr * n + i] = groups.lr[i];
    hparams[kBeta1 * n + i] = groups.beta1[i];
    hparams[kBeta2 * n + i] = groups.beta2[i];
    hparams[kBeta1Correction * n + i] = groups.beta1_correction[i];
    hparams[kBeta2Correction * n + i] = groups.beta2_correction[i];
    hparams[kEpsilon * n + i] = groups.epsilon[i];
    hparams[kWeightDecay * n + i] = groups.weight_decay[i];
  }
}

}  // namespace multi_tensor_adam_flat
}  // namespace transformer_engine

void nvte_multi_tensor_adam_flat_cuda(const NVTETensor grad, NVTETensor param,
                                      NVTETensor master_param, NVTETensor exp_avg,
                                      NVTETensor exp_avg_sq, const NVTETensor segments,
                                      const NVTETensor grad_scale, const float *lr,
                                      const float *beta1, const float *beta2, const float *epsilon,
                                      const float *weight_decay, const int *step,
                                      const int *bias_correction, const size_t num_groups,
                                      const int mode, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_flat_cuda);
  using namespace transformer_engine;

  multi_tensor_adam_flat::multi_tensor_adam_flat_cuda(
      *convertNVTETensorCheck(grad), convertNVTETensor(param),
      *convertNVTETensorCheck(master_param), *convertNVTETensorCheck(exp_avg),
      *convertNVTETensorCheck(exp_avg_sq), *convertNVTETensorCheck(segments),
      convertNVTETensor(grad_scale), lr, beta1, beta2, epsilon, weight_decay, step, bias_correction,
      num_groups, mode, stream);
}

void nvte_multi_tensor_adam_flat_capturable_cuda(const NVTETensor grad, NVTETensor param,
                                                 NVTETensor master_param, NVTETensor exp_avg,
                                                 NVTETensor exp_avg_sq, const NVTETensor segments,
                                                 const NVTETensor grad_scale,
                                                 const NVTETensor hparams, const int mode,
                                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_flat_capturable_cuda);
  using namespace transformer_engine;

  multi_tensor_adam_flat::multi_tensor_adam_flat_capturable_cuda(
      *convertNVTETensorCheck(grad), convertNVTETensor(param),
      *convertNVTETensorCheck(master_param), *convertNVTETensorCheck(exp_avg),
      *convertNVTETensorCheck(exp_avg_sq), *convertNVTETensorCheck(segments),
      convertNVTETensor(grad_scale), *convertNVTETensorCheck(hparams), mode, stream);
}

void nvte_multi_tensor_adam_flat_hparams(const float *lr, const float *beta1, const float *beta2,
                                         const float *epsilon, const float *weight_decay,
                                         const int *step, const int *bias_correction,
                                         const size_t num_groups, float *hparams) {
  NVTE_API_CALL(nvte_multi_tensor_adam_flat_hparams);
  using namespace transformer_engine;

  multi_tensor_adam_flat::multi_tensor_adam_flat_hparams(
      lr, beta1, beta2, epsilon, weight_decay, step, bias_correction, num_groups, hparams);
}
