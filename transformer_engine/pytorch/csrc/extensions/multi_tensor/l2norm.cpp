/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  auto output = at::zeros({320}, float_options);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;
  auto ret = at::empty({1}, output.options());

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = at::empty({ntensors}, float_options);
  } else {
    output_per_tensor = at::empty({0}, float_options);
    ret_per_tensor = at::empty({0}, float_options);
  }

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto output_cu = makeTransformerEngineTensor(output);
  auto output_per_tensor_cu = makeTransformerEngineTensor(output_per_tensor);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto ret_per_tensor_cu = makeTransformerEngineTensor(ret_per_tensor);

  nvte_multi_tensor_l2norm_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists,
                                num_tensors, output_cu.data(), output_per_tensor_cu.data(),
                                ret_cu.data(), ret_per_tensor_cu.data(), per_tensor,
                                max_chunks_per_tensor, at::cuda::getCurrentCUDAStream());

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_single_launch_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = at::empty({ntensors}, float_options);
  } else {
    output_per_tensor = at::empty({0}, float_options);
    ret_per_tensor = at::empty({0}, float_options);
  }

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  // Block sums per launch of the multi-launch path: [launches, 320], zero-initialized.
  const size_t output_size = nvte_multi_tensor_l2norm_single_launch_output_size(
      chunk_size, tensor_lists_ptr.data(), num_lists, num_tensors);
  auto output = at::zeros({static_cast<int64_t>(output_size)}, float_options);
  auto ret = at::empty({1}, output.options());

  auto output_cu = makeTransformerEngineTensor(output);
  auto output_per_tensor_cu = makeTransformerEngineTensor(output_per_tensor);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto ret_per_tensor_cu = makeTransformerEngineTensor(ret_per_tensor);

  nvte_multi_tensor_l2norm_single_launch_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      output_cu.data(), output_per_tensor_cu.data(), ret_cu.data(), ret_per_tensor_cu.data(),
      per_tensor, max_chunks_per_tensor, at::cuda::getCurrentCUDAStream());

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}

std::tuple<std::vector<int64_t>, int64_t> multi_tensor_l2norm_single_launch_slots(
    int chunk_size, std::vector<std::vector<at::Tensor>> tensor_lists) {
  NVTE_CHECK(tensor_lists.size() == 1, "multi_tensor_l2norm_single_launch expects 1 tensor list.");
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  std::vector<int64_t> first_slot(num_tensors);
  const size_t output_size = nvte_multi_tensor_l2norm_single_launch_slots(
      chunk_size, tensor_lists_ptr.data(), num_lists, num_tensors, first_slot.data());
  return {first_slot, static_cast<int64_t>(output_size)};
}

void multi_tensor_l2norm_single_launch_partials_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::Tensor output, std::vector<int64_t> first_slot) {
  NVTE_CHECK(tensor_lists.size() == 1, "multi_tensor_l2norm_single_launch expects 1 tensor list.");
  NVTE_CHECK(first_slot.size() == tensor_lists[0].size(), "Expected one slot per tensor, got ",
             first_slot.size(), " slots for ", tensor_lists[0].size(), " tensors.");
  if (tensor_lists[0].empty()) return;
  NVTE_CHECK(output.is_cuda() && output.is_contiguous(), "output must be a contiguous CUDA tensor.");
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto output_cu = makeTransformerEngineTensor(output);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  nvte_multi_tensor_l2norm_single_launch_partials_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      first_slot.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());
}

at::Tensor multi_tensor_l2norm_single_launch_finalize_cuda(at::Tensor output) {
  NVTE_CHECK(output.is_cuda() && output.is_contiguous(), "output must be a contiguous CUDA tensor.");
  auto ret = at::empty({1}, output.options());
  auto output_cu = makeTransformerEngineTensor(output);
  auto ret_cu = makeTransformerEngineTensor(ret);
  nvte_multi_tensor_l2norm_single_launch_finalize_cuda(output_cu.data(), ret_cu.data(),
                                                       at::cuda::getCurrentCUDAStream());
  return ret;
}

std::tuple<at::Tensor, at::Tensor> multi_tensor_unscale_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::Tensor inv_scale, at::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  auto output = at::zeros({320}, float_options);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  // Create output tensors for multi scale L2 norm kernel.
  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = at::empty({ntensors}, float_options);
  } else {
    output_per_tensor = at::empty({0}, float_options);
    ret_per_tensor = at::empty({0}, float_options);
  }

  auto ret = at::empty({1}, output.options());

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto output_cu = makeTransformerEngineTensor(output);
  auto output_per_tensor_cu = makeTransformerEngineTensor(output_per_tensor);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto ret_per_tensor_cu = makeTransformerEngineTensor(ret_per_tensor);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_unscale_l2norm_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      output_cu.data(), output_per_tensor_cu.data(), ret_cu.data(), ret_per_tensor_cu.data(),
      inv_scale_cu.data(), per_tensor, max_chunks_per_tensor, at::cuda::getCurrentCUDAStream());

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}

}  // namespace transformer_engine::pytorch
