/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_adam_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists,
                              num_tensors, lr, beta1, beta2, epsilon, step, mode, bias_correction,
                              weight_decay, at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_param_remainder_cuda(int chunk_size, at::Tensor noop_flag,
                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_adam_param_remainder_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors, lr, beta1,
      beta2, epsilon, step, mode, bias_correction, weight_decay, at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_fp8_cuda(int chunk_size, at::Tensor noop_flag,
                                std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, DType fp8_dtype) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_adam_fp8_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(),
                                  num_lists, num_tensors, lr, beta1, beta2, epsilon, step, mode,
                                  bias_correction, weight_decay, static_cast<NVTEDType>(fp8_dtype),
                                  at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_capturable_cuda(int chunk_size, at::Tensor noop_flag,
                                       std::vector<std::vector<at::Tensor>> tensor_lists,
                                       at::Tensor lr, const float beta1, const float beta2,
                                       const float epsilon, at::Tensor step, const int mode,
                                       const int bias_correction, const float weight_decay,
                                       at::Tensor inv_scale) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      lr_cu.data(), beta1, beta2, epsilon, step_cu.data(), mode, bias_correction, weight_decay,
      inv_scale_cu.data(), at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_capturable_master_cuda(int chunk_size, at::Tensor noop_flag,
                                              std::vector<std::vector<at::Tensor>> tensor_lists,
                                              at::Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, at::Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              at::Tensor inv_scale) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_master_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      lr_cu.data(), beta1, beta2, epsilon, step_cu.data(), mode, bias_correction, weight_decay,
      inv_scale_cu.data(), at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_flat_cuda(at::Tensor grad, std::optional<at::Tensor> param,
                                 at::Tensor master_param, at::Tensor exp_avg, at::Tensor exp_avg_sq,
                                 at::Tensor segments, std::optional<at::Tensor> grad_scale,
                                 const std::vector<float> &lr, const std::vector<float> &beta1,
                                 const std::vector<float> &beta2, const std::vector<float> &epsilon,
                                 const std::vector<float> &weight_decay,
                                 const std::vector<int> &step,
                                 const std::vector<int> &bias_correction, const int mode) {
  const size_t num_groups = lr.size();
  NVTE_CHECK(beta1.size() == num_groups && beta2.size() == num_groups &&
                 epsilon.size() == num_groups && weight_decay.size() == num_groups &&
                 step.size() == num_groups && bias_correction.size() == num_groups,
             "Every per-group hyperparameter list must have one value per parameter group");
  for (const auto &t : {grad, master_param, exp_avg, exp_avg_sq, segments}) {
    NVTE_CHECK(t.is_contiguous(), "multi_tensor_adam_flat expects contiguous tensors");
  }
  auto grad_cu = makeTransformerEngineTensor(grad);
  auto master_param_cu = makeTransformerEngineTensor(master_param);
  auto exp_avg_cu = makeTransformerEngineTensor(exp_avg);
  auto exp_avg_sq_cu = makeTransformerEngineTensor(exp_avg_sq);
  auto segments_cu = makeTransformerEngineTensor(segments);
  auto param_cu = TensorWrapper();  // empty: do not write model parameters
  if (param.has_value()) {
    NVTE_CHECK(param->is_contiguous(), "multi_tensor_adam_flat expects contiguous tensors");
    param_cu = makeTransformerEngineTensor(*param);
  }
  auto grad_scale_cu = TensorWrapper();  // empty: do not scale gradients
  if (grad_scale.has_value()) {
    grad_scale_cu = makeTransformerEngineTensor(*grad_scale);
  }

  nvte_multi_tensor_adam_flat_cuda(
      grad_cu.data(), param_cu.data(), master_param_cu.data(), exp_avg_cu.data(),
      exp_avg_sq_cu.data(), segments_cu.data(), grad_scale_cu.data(), lr.data(), beta1.data(),
      beta2.data(), epsilon.data(), weight_decay.data(), step.data(), bias_correction.data(),
      num_groups, mode, at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
