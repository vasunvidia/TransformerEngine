/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/hadamard_transform.h>

#include <map>
#include <mutex>
#include <tuple>
#include <utility>

#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

namespace {

// Zero-filled workspace of the NVFP4 RHT amax producers, one per device and stream. The kernels
// leave it zero-filled, so launches on the same stream reuse it without a memset (and a CUDA
// graph can capture them).
TensorWrapper hadamard_amax_producer_workspace(const TensorWrapper &query) {
  static std::mutex mutex;
  static std::map<std::pair<int, cudaStream_t>, at::Tensor> workspaces;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const std::pair<int, cudaStream_t> key{stream.device_index(), stream.stream()};
  const auto shape = convertShape(query.shape());
  const auto bytes = static_cast<int64_t>(product(shape));
  std::lock_guard<std::mutex> lock(mutex);
  auto &workspace = workspaces[key];
  if (!workspace.defined() || workspace.numel() < bytes) {
    workspace =
        at::zeros({bytes}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA, key.first));
  }
  return makeTransformerEngineTensor(workspace.data_ptr(), shape, query.dtype());
}

void check_bf16_input(const at::Tensor &t, const char *name) {
  NVTE_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
             " must be a contiguous CUDA BF16 tensor.");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> add_hadamard_transform_amax(const at::Tensor &a,
                                                               const at::Tensor &b,
                                                               int random_sign_mask_t) {
  init_extension();
  check_bf16_input(a, "a");
  check_bf16_input(b, "b");
  at::cuda::CUDAGuard device_guard(a.device());
  auto out = at::empty_like(a);
  auto amax = at::empty({2}, a.options().dtype(at::kFloat));
  auto a_cu = makeTransformerEngineTensor(a);
  auto b_cu = makeTransformerEngineTensor(b);
  auto out_cu = makeTransformerEngineTensor(out);
  auto amax_cu = makeTransformerEngineTensor(amax);
  const auto stream = at::cuda::getCurrentCUDAStream();
  TensorWrapper workspace;
  nvte_add_hadamard_transform_amax(a_cu.data(), b_cu.data(), out_cu.data(), amax_cu.data(),
                                   workspace.data(), random_sign_mask_t, stream);
  workspace = hadamard_amax_producer_workspace(workspace);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_add_hadamard_transform_amax(a_cu.data(), b_cu.data(), out_cu.data(), amax_cu.data(),
                                     workspace.data(), random_sign_mask_t, stream);
  });
  return {out, amax};
}

std::tuple<at::Tensor, at::Tensor> fused_qkv_rope_backward_hadamard_transform_amax(
    const at::Tensor &grad_q, const at::Tensor &grad_k, const at::Tensor &grad_v,
    const at::Tensor &cos, const at::Tensor &sin, int random_sign_mask_t) {
  init_extension();
  const auto q = grad_q.contiguous();
  const auto k = grad_k.contiguous();
  const auto v = grad_v.contiguous();
  check_bf16_input(q, "grad_q");
  check_bf16_input(k, "grad_k");
  check_bf16_input(v, "grad_v");
  NVTE_CHECK(q.dim() == 4 && k.dim() == 4, "grad_q and grad_k must be [s, b, h, d].");
  NVTE_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos and sin must be contiguous.");
  at::cuda::CUDAGuard device_guard(q.device());
  const int64_t groups = k.size(2);
  NVTE_CHECK(groups > 0 && q.size(2) % groups == 0, "Q heads must be a multiple of KV heads.");
  const int64_t width = (q.size(2) / groups + 2) * q.size(3);
  auto out = at::empty({q.size(0), q.size(1), groups, width}, q.options());
  auto amax = at::empty({2}, q.options().dtype(at::kFloat));
  auto q_cu = makeTransformerEngineTensor(q);
  auto k_cu = makeTransformerEngineTensor(k);
  auto v_cu = makeTransformerEngineTensor(v);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto out_cu = makeTransformerEngineTensor(out);
  auto amax_cu = makeTransformerEngineTensor(amax);
  const auto stream = at::cuda::getCurrentCUDAStream();
  TensorWrapper workspace;
  nvte_fused_qkv_rope_backward_hadamard_transform_amax(
      q_cu.data(), k_cu.data(), v_cu.data(), cos_cu.data(), sin_cu.data(), out_cu.data(),
      amax_cu.data(), workspace.data(), random_sign_mask_t, stream);
  workspace = hadamard_amax_producer_workspace(workspace);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_qkv_rope_backward_hadamard_transform_amax(
        q_cu.data(), k_cu.data(), v_cu.data(), cos_cu.data(), sin_cu.data(), out_cu.data(),
        amax_cu.data(), workspace.data(), random_sign_mask_t, stream);
  });
  return {out, amax};
}

std::tuple<py::object, at::Tensor> nvfp4_quantize_rowwise_with_amax_hadamard_transform_amax(
    const at::Tensor &tensor, py::handle quantizer, const at::Tensor &rowwise_amax) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();
  check_bf16_input(tensor, "tensor");
  NVTE_CHECK(tensor.dim() >= 2, "Tensor must be at least 2D.");
  NVTE_CHECK(rowwise_amax.is_cuda() && rowwise_amax.scalar_type() == at::kFloat &&
                 rowwise_amax.numel() == 1,
             "rowwise_amax must be a CUDA FP32 scalar tensor.");
  NVTE_CHECK(IsNVFP4Quantizers(quantizer.ptr()), "Only NVFP4 quantizers are supported.");
  auto quantizer_cpp = convert_quantizer(quantizer);
  at::cuda::CUDAGuard device_guard(tensor.device());

  auto input_cu = makeTransformerEngineTensor(tensor);
  const auto shape = getTensorShape(tensor);
  auto [output_cu, output_py] = quantizer_cpp->create_tensor(shape, input_cu.dtype());
  NVTE_CHECK(output_cu.get_amax().data_ptr != nullptr,
             "The quantizer must produce row-wise data only.");
  output_cu.set_amax(rowwise_amax.data_ptr(), DType::kFloat32, getTensorShape(rowwise_amax));
  output_py.attr("_amax_rowwise") = py::cast(rowwise_amax);
  auto amax = at::empty({2}, tensor.options().dtype(at::kFloat));
  auto amax_cu = makeTransformerEngineTensor(amax);
  const auto *nvfp4_quantizer_cpp = static_cast<const NVFP4Quantizer *>(quantizer_cpp.get());
  const int sign_mask_t = nvfp4_quantizer_cpp->rht_matrix_random_sign_mask_t;
  const auto stream = at::cuda::getCurrentCUDAStream();
  TensorWrapper workspace;
  nvte_quantize_rowwise_with_amax_hadamard_transform_amax(
      input_cu.data(), output_cu.data(), amax_cu.data(), workspace.data(), sign_mask_t, stream);
  workspace = hadamard_amax_producer_workspace(workspace);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_rowwise_with_amax_hadamard_transform_amax(
        input_cu.data(), output_cu.data(), amax_cu.data(), workspace.data(), sign_mask_t, stream);
  });
  return {output_py, amax};
}

}  // namespace transformer_engine::pytorch
