/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common/util/system.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

std::vector<py::object> layernorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                      const at::Tensor &mu, const at::Tensor &rsigma,
                                      const at::Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma) {
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &mu_ = mu.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  auto dbeta = at::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto mu_cu = makeTransformerEngineTensor(mu_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);
  auto dbeta_cu = makeTransformerEngineTensor(dbeta);

  // This call populates tensors with the required config.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                       dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Actual call to bwd kernel.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                       dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(dx), py::cast(dgamma), py::cast(dbeta)};
}

std::vector<py::object> layernorm_fwd(py::handle input, py::handle weight, MaybeTensor bias,
                                      float eps, py::object out, py::handle quantizer,
                                      DType out_dtype, const int sm_margin,
                                      const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch::detail;

  // Ensure that cuDNN handle is created on the correct device,
  // overriding torch.cuda.set_device calls from user side.
  // Assumes all tensors passed are on the same device.
  at::cuda::CUDAGuard device_guard(input.cast<at::Tensor>().device());

  // Input and param tensors
  auto none = py::none();
  const TensorWrapper &input_nvte = makeTransformerEngineTensor(input, none);
  const TensorWrapper &weight_nvte = makeTransformerEngineTensor(weight, none);
  TensorWrapper bias_nvte;
  if (bias.has_value()) {
    bias_nvte = makeTransformerEngineTensor(*bias);
  }

  // Tensor dimensions
  const auto shape = nvte_shape_to_vector(input_nvte.shape());
  const auto [outer_size, inner_size] = get_2d_dims(shape);

  // Tensors to save for backward pass
  at::Tensor mu_py = at::empty({static_cast<int64_t>(outer_size)}, at::CUDA(at::kFloat));
  at::Tensor rsigma_py = at::empty({static_cast<int64_t>(outer_size)}, at::CUDA(at::kFloat));
  TensorWrapper mu_nvte = makeTransformerEngineTensor(mu_py);
  TensorWrapper rsigma_nvte = makeTransformerEngineTensor(rsigma_py);

  // Quantizer
  auto quantizer_cpp = convert_quantizer(quantizer);

  // Choose implementation
  enum class Impl {
    // Compute norm in high precision, then quantize
    UNFUSED,
    // Compute norm directly
    FULLY_FUSED,
    // Compute norm and amax in high precision, then quantize to FP8
    FUSED_NORM_AMAX_FP8,
    // Compute norm and amax in high precision, then quantize to NVFP4
    FUSED_NORM_AMAX_NVFP4
  };
  Impl impl = Impl::UNFUSED;
  if (quantizer.is_none() || IsFloat8Quantizers(quantizer.ptr())) {
    impl = Impl::FULLY_FUSED;
  } else if (IsMXFP8Quantizers(quantizer.ptr())) {
    if (transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN") && outer_size % 128 == 0 &&
        inner_size % 128 == 0) {
      // cuDNN MXFP8 kernel requires full 128x128 tiles
      impl = Impl::FULLY_FUSED;
    }
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr()) &&
             !transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")) {
    auto fp8_quantizer_cpp = dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
    NVTE_CHECK(fp8_quantizer_cpp != nullptr, "Could not cast to FP8 current scaling quantizer");
    impl = Impl::FUSED_NORM_AMAX_FP8;
  } else if (detail::IsNVFP4Quantizers(quantizer.ptr())) {
    auto nvfp4_quantizer_cpp = dynamic_cast<NVFP4Quantizer *>(quantizer_cpp.get());
    NVTE_CHECK(nvfp4_quantizer_cpp != nullptr, "Could not cast to NVFP4 quantizer");
    if (nvfp4_quantizer_cpp->row_scaled_nvfp4 ||
        (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax)) {
      // Amax is handled within NVFP4 quantizer
      impl = Impl::UNFUSED;
    } else if (nvfp4_quantizer_cpp->disable_second_level_scale) {
      // No need for amax
      impl = Impl::UNFUSED;
    } else if (!transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")) {
      // TE kernel supports amax output
      impl = Impl::FUSED_NORM_AMAX_NVFP4;
    }
  }

  // Output tensor
  TensorWrapper out_nvte;
  if (out.is_none()) {
    if (impl == Impl::FULLY_FUSED) {
      // FP8 has no special logic to optimize for GEMM, MXFP8 cuDNN
      // kernel does not support GEMM swizzled scales
      quantizer_cpp->optimize_for_gemm = false;
    }
    std::tie(out_nvte, out) = quantizer_cpp->create_tensor(shape, out_dtype);
  } else {
    out_nvte = makeTransformerEngineTensor(out, quantizer);
  }

  // Construct unquantized output tensor if needed
  TensorWrapper unquantized_out_nvte;
  py::object unquantized_out;
  at::Tensor amax_buf;
  TensorWrapper *kernel_out_nvte = &out_nvte;
  switch (impl) {
    case Impl::UNFUSED: {
      NoneQuantizer q{none};
      std::tie(unquantized_out_nvte, unquantized_out) = q.create_tensor(shape, out_dtype);
      kernel_out_nvte = &unquantized_out_nvte;
    } break;
    case Impl::FUSED_NORM_AMAX_FP8: {
      auto fp8_quantizer_cpp = static_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
      std::tie(unquantized_out_nvte, unquantized_out, amax_buf) =
          fp8_quantizer_cpp->create_unquantized_tensor_with_amax(shape, out_dtype);
      kernel_out_nvte = &unquantized_out_nvte;
    } break;
    case Impl::FUSED_NORM_AMAX_NVFP4: {
      auto nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
      std::tie(unquantized_out_nvte, unquantized_out) =
          nvfp4_quantizer_cpp->create_unquantized_tensor_with_amax(out_nvte, out_dtype);
      kernel_out_nvte = &unquantized_out_nvte;
    } break;
    default: {
    }
  }

  // Query workspace size
  TensorWrapper workspace;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_fwd(input_nvte.data(), weight_nvte.data(), bias_nvte.data(), eps,
                       kernel_out_nvte->data(), mu_nvte.data(), rsigma_nvte.data(),
                       workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Allocate workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_fwd(input_nvte.data(), weight_nvte.data(), bias_nvte.data(), eps,
                       kernel_out_nvte->data(), mu_nvte.data(), rsigma_nvte.data(),
                       workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Quantize output if needed
  switch (impl) {
    case Impl::UNFUSED: {
      quantizer_cpp->quantize(unquantized_out_nvte, out_nvte);
    } break;
    case Impl::FUSED_NORM_AMAX_FP8: {
      auto fp8_quantizer_cpp = static_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
      fp8_quantizer_cpp->quantize_with_amax(unquantized_out_nvte, out_nvte, amax_buf);
    } break;
    case Impl::FUSED_NORM_AMAX_NVFP4: {
      auto nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
      nvfp4_quantizer_cpp->quantize_with_amax(unquantized_out_nvte, out_nvte);
    } break;
    default: {
    }
  }

  return {out, py::cast(mu_py), py::cast(rsigma_py)};
}

std::vector<py::object> rmsnorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                    const at::Tensor &rsigma, const at::Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma) {
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  // This call populates tensors with the required config.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                     dgamma_cu.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Actual call to bwd kernel.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                     dgamma_cu.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(dx), py::cast(dgamma)};
}

std::vector<py::object> rmsnorm_bwd_add(const at::Tensor &dz, const at::Tensor &x,
                                        const at::Tensor &add, const at::Tensor &rsigma,
                                        const at::Tensor &gamma, const int sm_margin,
                                        const bool zero_centered_gamma) {
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &add_ = add.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto add_cu = makeTransformerEngineTensor(add_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  // This call populates tensors with the required config.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd_add(dz_cu.data(), x_cu.data(), add_cu.data(), rsigma_cu.data(),
                         gamma_cu.data(), dx_cu.data(), dgamma_cu.data(), workspace.data(),
                         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                         zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Actual call to bwd kernel.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd_add(dz_cu.data(), x_cu.data(), add_cu.data(), rsigma_cu.data(),
                         gamma_cu.data(), dx_cu.data(), dgamma_cu.data(), workspace.data(),
                         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                         zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(dx), py::cast(dgamma)};
}

namespace {

// TensorWrappers of the fused RMSNorm backward's inputs (absent optional tensors: empty wrapper).
struct RMSNormBwdFusedInputs {
  at::Tensor dz, dz2, x, add, rsigma, gamma;
  TensorWrapper dz_cu, dz2_cu, x_cu, add_cu, rsigma_cu, gamma_cu;
  bool has_dz2 = false, has_add = false;
  RMSNormBwdFusedInputs(const at::Tensor &dz_, const std::optional<at::Tensor> &dz2_,
                        const at::Tensor &x_, const std::optional<at::Tensor> &add_,
                        const at::Tensor &rsigma_, const at::Tensor &gamma_)
      : dz(dz_.contiguous()),
        x(x_.contiguous()),
        rsigma(rsigma_.contiguous()),
        gamma(gamma_.contiguous()) {
    dz_cu = makeTransformerEngineTensor(dz);
    x_cu = makeTransformerEngineTensor(x);
    rsigma_cu = makeTransformerEngineTensor(rsigma);
    gamma_cu = makeTransformerEngineTensor(gamma);
    if (dz2_.has_value()) {
      dz2 = dz2_->contiguous();
      dz2_cu = makeTransformerEngineTensor(dz2);
      has_dz2 = true;
    }
    if (add_.has_value()) {
      add = add_->contiguous();
      add_cu = makeTransformerEngineTensor(add);
      has_add = true;
    }
  }
  NVTETensor dz2_ptr() { return has_dz2 ? dz2_cu.data() : nullptr; }
  NVTETensor add_ptr() { return has_add ? add_cu.data() : nullptr; }
};

}  // namespace

bool rmsnorm_bwd_fused_supported(const at::Tensor &dz, const std::optional<at::Tensor> &dz2,
                                 const at::Tensor &x, const std::optional<at::Tensor> &add,
                                 const at::Tensor &rsigma, const at::Tensor &gamma,
                                 const bool zero_centered_gamma) {
  RMSNormBwdFusedInputs in(dz, dz2, x, add, rsigma, gamma);
  // dx / dgamma are allocated like x / gamma (fresh, aligned): stand in with x / gamma.
  return nvte_rmsnorm_bwd_fused_supported(in.dz_cu.data(), in.dz2_ptr(), in.x_cu.data(),
                                          in.add_ptr(), in.rsigma_cu.data(), in.gamma_cu.data(),
                                          in.x_cu.data(), in.gamma_cu.data(), nullptr,
                                          zero_centered_gamma);
}

std::vector<py::object> rmsnorm_bwd_fused(const at::Tensor &dz,
                                          const std::optional<at::Tensor> &dz2,
                                          const at::Tensor &x, const std::optional<at::Tensor> &add,
                                          const at::Tensor &rsigma, const at::Tensor &gamma,
                                          const bool zero_centered_gamma, py::handle quantizer) {
  at::cuda::CUDAGuard device_guard(x.device());
  RMSNormBwdFusedInputs in(dz, dz2, x, add, rsigma, gamma);
  auto dx = at::empty_like(in.x);
  auto dgamma = at::empty_like(in.gamma);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  // Quantized copy of dx, when the quantizer is given and the kernel supports it (else None:
  // the caller quantizes dx itself).
  py::object dx_q_py = py::none();
  TensorWrapper dx_q_cu;
  NVTETensor dx_q_ptr = nullptr;
  if (!quantizer.is_none()) {
    auto quantizer_cpp = convert_quantizer(quantizer);
    const auto shape = getTensorShape(in.x);
    std::tie(dx_q_cu, dx_q_py) = quantizer_cpp->create_tensor(shape, GetTransformerEngineDType(
                                                                         in.x.scalar_type()));
    if (nvte_rmsnorm_bwd_fused_supported(in.dz_cu.data(), in.dz2_ptr(), in.x_cu.data(),
                                         in.add_ptr(), in.rsigma_cu.data(), in.gamma_cu.data(),
                                         dx_cu.data(), dgamma_cu.data(), dx_q_cu.data(),
                                         zero_centered_gamma)) {
      dx_q_ptr = dx_q_cu.data();
    } else {
      dx_q_py = py::none();
    }
  }

  TensorWrapper workspace;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd_fused(in.dz_cu.data(), in.dz2_ptr(), in.x_cu.data(), in.add_ptr(),
                           in.rsigma_cu.data(), in.gamma_cu.data(), dx_cu.data(), dgamma_cu.data(),
                           dx_q_ptr, workspace.data(), zero_centered_gamma,
                           at::cuda::getCurrentCUDAStream());
  });
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd_fused(in.dz_cu.data(), in.dz2_ptr(), in.x_cu.data(), in.add_ptr(),
                           in.rsigma_cu.data(), in.gamma_cu.data(), dx_cu.data(), dgamma_cu.data(),
                           dx_q_ptr, workspace.data(), zero_centered_gamma,
                           at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(dx), py::cast(dgamma), dx_q_py};
}

std::vector<py::object> rmsnorm_fwd(const py::handle &input, const py::handle &weight, float eps,
                                    py::object out, py::handle quantizer, DType out_dtype,
                                    const int sm_margin, const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch::detail;

  // Ensure that cuDNN handle is created on the correct device,
  // overriding torch.cuda.set_device calls from user side.
  // Assumes all tensors passed are on the same device.
  at::cuda::CUDAGuard device_guard(input.cast<at::Tensor>().device());

  // Input and param tensors
  auto none = py::none();
  const TensorWrapper &input_nvte = makeTransformerEngineTensor(input, none);
  const TensorWrapper &weight_nvte = makeTransformerEngineTensor(weight, none);

  // Tensor dimensions
  const auto shape = nvte_shape_to_vector(input_nvte.shape());
  const auto [outer_size, inner_size] = get_2d_dims(shape);

  // Tensors to save for backward pass
  at::Tensor rsigma_py = at::empty({static_cast<int64_t>(outer_size)}, at::CUDA(at::kFloat));
  TensorWrapper rsigma_nvte = makeTransformerEngineTensor(rsigma_py);

  // Quantizer
  auto quantizer_cpp = convert_quantizer(quantizer);

  // Choose implementation
  enum class Impl {
    // Compute norm in high precision, then quantize
    UNFUSED,
    // Compute norm directly
    FULLY_FUSED,
    // Compute norm directly, with quantization scales in compact format
    FUSED_NORM_QUANT_UNSWIZZLED,
    // Compute norm and amax in high precision, then quantize to FP8
    FUSED_NORM_AMAX_FP8,
    // Compute norm and amax in high precision, then quantize to NVFP4
    FUSED_NORM_AMAX_NVFP4
  };
  Impl impl = Impl::UNFUSED;
  if (quantizer.is_none() || IsFloat8Quantizers(quantizer.ptr())) {
    impl = Impl::FULLY_FUSED;
  } else if (IsMXFP8Quantizers(quantizer.ptr())) {
    auto mxfp8_quantizer_cpp = dynamic_cast<MXFP8Quantizer *>(quantizer_cpp.get());
    NVTE_CHECK(mxfp8_quantizer_cpp != nullptr, "Could not cast to MXFP8 quantizer");
    // The fused MXFP8 kernels quantize 1D blocks and require full 128x128 tiles
    if (!mxfp8_quantizer_cpp->with_2d_quantization && outer_size % 128 == 0 &&
        inner_size % 128 == 0) {
      if (transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")) {
        impl = Impl::FUSED_NORM_QUANT_UNSWIZZLED;
      } else {
        // Transformer Engine's fused RMSNorm + MXFP8 kernel
        impl = Impl::FULLY_FUSED;
      }
    }
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr()) &&
             !transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")) {
    auto fp8_quantizer_cpp = dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
    NVTE_CHECK(fp8_quantizer_cpp != nullptr, "Could not cast to FP8 current scaling quantizer");
    impl = Impl::FUSED_NORM_AMAX_FP8;
  } else if (detail::IsNVFP4Quantizers(quantizer.ptr())) {
    auto nvfp4_quantizer_cpp = dynamic_cast<NVFP4Quantizer *>(quantizer_cpp.get());
    NVTE_CHECK(nvfp4_quantizer_cpp != nullptr, "Could not cast to NVFP4 quantizer");
    if (nvfp4_quantizer_cpp->row_scaled_nvfp4 ||
        (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax)) {
      // Amax is handled within NVFP4 quantizer
      impl = Impl::UNFUSED;
    } else if (nvfp4_quantizer_cpp->disable_second_level_scale) {
      // No need for amax
      impl = Impl::UNFUSED;
    } else if (!transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")) {
      // TE kernel supports amax output
      impl = Impl::FUSED_NORM_AMAX_NVFP4;
    }
  }

  // Output tensor
  TensorWrapper out_nvte;
  if (out.is_none()) {
    if (impl == Impl::FUSED_NORM_QUANT_UNSWIZZLED) {
      quantizer_cpp->optimize_for_gemm = false;
    }
    std::tie(out_nvte, out) = quantizer_cpp->create_tensor(shape, out_dtype);
  } else {
    out_nvte = makeTransformerEngineTensor(out, quantizer);
  }

  // Construct unquantized output tensor if needed
  TensorWrapper unquantized_out_nvte;
  py::object unquantized_out;
  at::Tensor amax_buf;
  TensorWrapper *kernel_out_nvte = &out_nvte;
  switch (impl) {
    case Impl::UNFUSED: {
      NoneQuantizer q{none};
      std::tie(unquantized_out_nvte, unquantized_out) = q.create_tensor(shape, out_dtype);
      kernel_out_nvte = &unquantized_out_nvte;
    } break;
    case Impl::FUSED_NORM_AMAX_FP8: {
      auto fp8_quantizer_cpp = static_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
      std::tie(unquantized_out_nvte, unquantized_out, amax_buf) =
          fp8_quantizer_cpp->create_unquantized_tensor_with_amax(shape, out_dtype);
      kernel_out_nvte = &unquantized_out_nvte;
    } break;
    case Impl::FUSED_NORM_AMAX_NVFP4: {
      auto nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
      std::tie(unquantized_out_nvte, unquantized_out) =
          nvfp4_quantizer_cpp->create_unquantized_tensor_with_amax(out_nvte, out_dtype);
      kernel_out_nvte = &unquantized_out_nvte;
    } break;
    default: {
    }
  }

  // Query workspace size
  TensorWrapper workspace;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_fwd(input_nvte.data(), weight_nvte.data(), eps, kernel_out_nvte->data(),
                     rsigma_nvte.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Allocate workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_fwd(input_nvte.data(), weight_nvte.data(), eps, kernel_out_nvte->data(),
                     rsigma_nvte.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Quantize output if needed
  switch (impl) {
    case Impl::UNFUSED: {
      quantizer_cpp->quantize(unquantized_out_nvte, out_nvte);
    } break;
    case Impl::FUSED_NORM_AMAX_FP8: {
      auto fp8_quantizer_cpp = static_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
      fp8_quantizer_cpp->quantize_with_amax(unquantized_out_nvte, out_nvte, amax_buf);
    } break;
    case Impl::FUSED_NORM_AMAX_NVFP4: {
      auto nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
      nvfp4_quantizer_cpp->quantize_with_amax(unquantized_out_nvte, out_nvte);
    } break;
    default: {
    }
  }

  return {out, py::none(), py::cast(rsigma_py)};
}

}  // namespace transformer_engine::pytorch
