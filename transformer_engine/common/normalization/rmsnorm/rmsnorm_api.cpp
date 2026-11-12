/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <string>

#include "../../common.h"
#include "../common.h"
#include "transformer_engine/normalization.h"
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/transpose.h"

namespace transformer_engine {

using namespace normalization;

namespace {

// Whether rmsnorm_bwd takes the opt-in single-sweep kernel (NVTE_RMSNORM_BWD_SINGLE_SWEEP=1).
// When it is enabled but cannot take a configuration, the regular backward runs instead and
// the reason is printed once per distinct reason.
// `add` is the residual gradient of rmsnorm_bwd_add (nullptr for rmsnorm_bwd).
bool select_rmsnorm_bwd_single_sweep(const Tensor &dz, const Tensor &x, const Tensor &rsigma,
                                     const Tensor &gamma, const Tensor &dx, const Tensor &dgamma,
                                     const Tensor *add, const bool zero_centered_gamma) {
  if (!use_rmsnorm_bwd_single_sweep()) {
    return false;
  }
  std::string reason;
  if (is_supported_by_rmsnorm_bwd_single_sweep(dz, x, rsigma, gamma, dx, dgamma, add,
                                               zero_centered_gamma, &reason)) {
    return true;
  }
  static std::mutex mutex;
  static std::set<std::string> reported;
  std::lock_guard<std::mutex> lock(mutex);
  if (reported.insert(reason).second) {
    NVTE_WARN("NVTE_RMSNORM_BWD_SINGLE_SWEEP: using the regular RMSNorm backward: ", reason,
              ". (Printed once per reason.)");
  }
  return false;
}

}  // namespace

void rmsnorm_fwd(const Tensor &x, const Tensor &gamma, const float epsilon, Tensor *z,
                 Tensor *rsigma, Tensor *workspace, const int multiprocessorCount,
                 const bool zero_centered_gamma, cudaStream_t stream) {
  // Check for unsupported configurations
  if (is_fp8_dtype(z->data.dtype) && !is_delayed_tensor_scaling(z->scaling_mode) &&
      !is_mxfp8_scaling(z->scaling_mode)) {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(z->scaling_mode) + ".");
  }
  NVTE_CHECK(!x.data.shape.empty(), "x must have at least one dimension.");
  const auto [rows, cols] = x.flat_2d_dims();

  NVTE_CHECK(gamma.data.shape == Shape{cols}, "Gamma has invalid shape (expected ", Shape{cols},
             ", got ", gamma.data.shape, ").");
  NVTE_CHECK(epsilon >= 0.f, "Epsilon must be non-negative.");

  NVTE_CHECK(z->data.shape == x.data.shape, "Output tensor must have the same shape as x.");

  NVTE_CHECK(rsigma->data.shape == Shape{rows}, "RSigma has invalid shape (expected ", Shape{rows},
             ", got ", rsigma->data.shape, ").");
  NVTE_CHECK(rsigma->data.dtype == DType::kFloat32, "RSigma must be a float32 tensor.");

  if (workspace->data.numel() != 0) {
    CheckInputTensor(x, "x");
    CheckInputTensor(gamma, "gamma");

    CheckOutputTensor(*z, "z");
    CheckOutputTensor(*rsigma, "rsigma");
  }

  // MXFP8 output: Transformer Engine's fused kernel, which alone writes GEMM-swizzled scales,
  // unless cuDNN is requested or the kernel does not support the tensors.
  if (is_mxfp8_scaling(z->scaling_mode)) {
    const bool te_supported = is_supported_by_te_rmsnorm_fwd_mxfp8(x, gamma, *z);
    if (z->with_gemm_swizzled_scales || (!use_cudnn_norm_fwd() && te_supported)) {
      NVTE_CHECK(te_supported,
                 "RMSNorm forward with GEMM-swizzled MXFP8 scales requires FP32, BF16 or FP16 "
                 "input and weight, and rows and columns that are multiples of 128.");
      if (workspace->data.numel() == 0) {
        // The fused kernel needs no workspace, but a zero size denotes a workspace query.
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
      }
      rmsnorm_fwd_mxfp8(x, gamma, epsilon, z, rsigma, multiprocessorCount, zero_centered_gamma,
                        stream);
      return;
    }
  }

  NVTE_Norm_Backend norm_backend;
  bool is_aligned = true;
  bool cudnn_backend = use_cudnn_norm_fwd() || is_mxfp8_scaling(z->scaling_mode);

  if (!is_fp8_dtype(z->data.dtype) && z->amax.dptr != nullptr) {
    NVTE_CHECK(!cudnn_backend,
               "cuDNN does not currently support amax output for non quantized output");
  }

  bool training =
      is_delayed_tensor_scaling(z->scaling_mode) || (z->columnwise_data).dptr != nullptr;

  bool gamma_in_weight_dtype = false;
  if (cudnn_backend) {
    // TODO: add check for GPU ARCH
    norm_backend = NVTE_Norm_Backend::Cudnn;
    gamma_in_weight_dtype = use_zero_centered_gamma_in_weight_dtype();
  } else {
    norm_backend = NVTE_Norm_Backend::Te;
    is_aligned = is_ptr_aligned(z->data.dptr, x.data.dptr, gamma.data.dptr, rsigma->data.dptr);
  }

  auto plan = NormalizationPlanRegistry::getInstance().getNormalizationPlan(
      norm_backend, NVTE_Norm_Type::RMSNorm, NVTE_Norm_Stage::Forward,
      gamma.data.dtype,  // wtype
      x.data.dtype,      // itype
      z->data.dtype,     // otype
      rows,              // batch_size
      cols,              // hidden_size
      multiprocessorCount, zero_centered_gamma, is_aligned, z->scaling_mode, training,
      gamma_in_weight_dtype);

  if (workspace->data.numel() == 0) {
    workspace->data.shape = plan->getWorkspaceShape();
    workspace->data.dtype = DType::kByte;
    return;
  }

  NVTE_CHECK(workspace->data.shape == plan->getWorkspaceShape());
  NVTE_CHECK(
      !is_block_scaling(z->scaling_mode) || (!training || z->columnwise_scale_inv.dptr != nullptr),
      "Columnwise scale_inv must be allocated for NormFwdTraining!");
  plan->execute(z, x.data.dptr, gamma.data.dptr, nullptr /*beta*/, nullptr /*mu*/,
                reinterpret_cast<void *>(const_cast<float *>(&epsilon)), rsigma->data.dptr,
                workspace->data.dptr, stream);

  // Compute FP8 transpose if required
  if (z->has_columnwise_data() && is_tensor_scaling(z->scaling_mode)) {
    NVTETensor transpose_data = nvte_create_tensor(z->scaling_mode);
    auto *t = convertNVTETensor(transpose_data);
    t->data = z->columnwise_data;

    nvte_transpose(static_cast<NVTETensor>(*z), transpose_data, stream);
    nvte_destroy_tensor(transpose_data);
  }

  return;
}

void rmsnorm_bwd(const Tensor &dz, const Tensor &x, const Tensor &rsigma, const Tensor &gamma,
                 Tensor *dx, Tensor *dgamma, Tensor *workspace, const int multiprocessorCount,
                 const bool zero_centered_gamma, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(dz.data.dtype == gamma.data.dtype);
  NVTE_CHECK(rsigma.data.dtype == DType::kFloat32);

  NVTE_CHECK(!x.data.shape.empty(), "x must have at least one dimension.");
  const auto [rows, cols] = x.flat_2d_dims();
  NVTE_CHECK(dz.data.shape == x.data.shape);
  NVTE_CHECK(rsigma.data.shape == Shape{rows}, "RSigma has invalid shape (expected ", Shape{rows},
             ", got ", rsigma.data.shape, ").");

  NVTE_CHECK(gamma.data.shape == Shape{cols}, "Gamma has invalid shape (expected ", Shape{cols},
             ", got ", gamma.data.shape, ").");

  NVTE_CHECK(dx->data.shape == x.data.shape);
  NVTE_CHECK(dx->data.dtype == x.data.dtype);

  NVTE_CHECK(dgamma->data.shape == gamma.data.shape);
  NVTE_CHECK(dgamma->data.dtype == gamma.data.dtype);

  if (workspace->data.numel() != 0) {
    CheckInputTensor(dz, "dz");
    CheckInputTensor(x, "x");
    CheckInputTensor(rsigma, "rsigma");
    CheckInputTensor(gamma, "gamma");
    CheckOutputTensor(*dx, "dx");
    CheckOutputTensor(*dgamma, "dgamma");
  }

  // Opt-in single-sweep kernel (NVTE_RMSNORM_BWD_SINGLE_SWEEP=1), ahead of the TE and cuDNN
  // backends. It takes the workspace query and the launch alike; a workspace sized for another
  // backend (the setting changed between the two calls) sends the launch to the regular path.
  if (select_rmsnorm_bwd_single_sweep(dz, x, rsigma, gamma, *dx, *dgamma, nullptr,
                                      zero_centered_gamma)) {
    const size_t bytes = rmsnorm_bwd_single_sweep_workspace_bytes(rows, cols);
    if (workspace->data.numel() == 0) {
      workspace->data.shape = {bytes};
      workspace->data.dtype = DType::kByte;
      return;
    }
    if (workspace->data.dtype == DType::kByte && workspace->data.shape == Shape{bytes}) {
      rmsnorm_bwd_single_sweep(dz, x, rsigma, gamma, nullptr, dx, dgamma, workspace->data.dptr,
                               zero_centered_gamma, stream);
      return;
    }
  }

  NVTE_Norm_Backend norm_backend;
  bool is_aligned = true;
  bool gamma_in_weight_dtype = false;
  if (use_cudnn_norm_bwd()) {
    // TODO: add check for GPU ARCH
    norm_backend = NVTE_Norm_Backend::Cudnn;
    gamma_in_weight_dtype = use_zero_centered_gamma_in_weight_dtype();
  } else {
    norm_backend = NVTE_Norm_Backend::Te;
    is_aligned = is_ptr_aligned(x.data.dptr, gamma.data.dptr, rsigma.data.dptr, dx->data.dptr,
                                dz.data.dptr, dgamma->data.dptr);
  }
  auto plan = NormalizationPlanRegistry::getInstance().getNormalizationPlan(
      norm_backend, NVTE_Norm_Type::RMSNorm, NVTE_Norm_Stage::Backward,
      gamma.data.dtype,  // wtype
      x.data.dtype,      // itype
      gamma.data.dtype,  // otype
      rows,              // batch_size
      cols,              // hidden_size
      multiprocessorCount, zero_centered_gamma, is_aligned, NVTE_DELAYED_TENSOR_SCALING, true,
      gamma_in_weight_dtype);

  if (workspace->data.numel() == 0) {
    workspace->data.shape = plan->getWorkspaceShape();
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_CHECK(workspace->data.shape == plan->getWorkspaceShape());
    plan->execute(x.data.dptr, gamma.data.dptr, nullptr /*mu*/, rsigma.data.dptr, dx->data.dptr,
                  dz.data.dptr, nullptr /*add*/, nullptr /*dbeta*/, dgamma->data.dptr,
                  workspace->data.dptr, stream);
  }
  return;
}

void rmsnorm_bwd_add(const Tensor &dz, const Tensor &x, const Tensor &add, const Tensor &rsigma,
                     const Tensor &gamma, Tensor *dx, Tensor *dgamma, Tensor *workspace,
                     const int multiprocessorCount, const bool zero_centered_gamma,
                     cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(dz.data.dtype == gamma.data.dtype);
  NVTE_CHECK(add.data.dtype == gamma.data.dtype);
  NVTE_CHECK(rsigma.data.dtype == DType::kFloat32);

  NVTE_CHECK(!x.data.shape.empty(), "x must have at least one dimension.");
  const auto [rows, cols] = x.flat_2d_dims();
  NVTE_CHECK(dz.data.shape == x.data.shape);
  NVTE_CHECK(rsigma.data.shape == Shape{rows}, "RSigma has invalid shape (expected ", Shape{rows},
             ", got ", rsigma.data.shape, ").");
  NVTE_CHECK(add.data.shape == x.data.shape);

  NVTE_CHECK(gamma.data.shape == Shape{cols}, "Gamma has invalid shape (expected ", Shape{cols},
             ", got ", gamma.data.shape, ").");

  NVTE_CHECK(dx->data.shape == x.data.shape);
  NVTE_CHECK(dx->data.dtype == x.data.dtype);

  NVTE_CHECK(dgamma->data.shape == gamma.data.shape);
  NVTE_CHECK(dgamma->data.dtype == gamma.data.dtype);

  if (workspace->data.numel() != 0) {
    CheckInputTensor(dz, "dz");
    CheckInputTensor(x, "x");
    CheckInputTensor(add, "add");
    CheckInputTensor(rsigma, "rsigma");
    CheckInputTensor(gamma, "gamma");
    CheckOutputTensor(*dx, "dx");
    CheckOutputTensor(*dgamma, "dgamma");
  }

  // Opt-in single-sweep kernel with the residual gradient folded into its dx store (bitwise
  // equal to its rmsnorm_bwd followed by a separate BF16 addition); see rmsnorm_bwd above.
  if (select_rmsnorm_bwd_single_sweep(dz, x, rsigma, gamma, *dx, *dgamma, &add,
                                      zero_centered_gamma)) {
    const size_t bytes = rmsnorm_bwd_single_sweep_workspace_bytes(rows, cols);
    if (workspace->data.numel() == 0) {
      workspace->data.shape = {bytes};
      workspace->data.dtype = DType::kByte;
      return;
    }
    if (workspace->data.dtype == DType::kByte && workspace->data.shape == Shape{bytes}) {
      rmsnorm_bwd_single_sweep(dz, x, rsigma, gamma, &add, dx, dgamma, workspace->data.dptr,
                               zero_centered_gamma, stream);
      return;
    }
  }

  NVTE_Norm_Backend norm_backend;
  bool is_aligned = true;
  bool gamma_in_weight_dtype = false;
  if (use_cudnn_norm_bwd()) {
    norm_backend = NVTE_Norm_Backend::Cudnn;
    gamma_in_weight_dtype = use_zero_centered_gamma_in_weight_dtype();
  } else {
    norm_backend = NVTE_Norm_Backend::Te;
    // TE backend does not currently support zero_centered_gamma_in_weight_dtype
    NVTE_CHECK(!use_zero_centered_gamma_in_weight_dtype(),
               "zero_centered_gamma_in_weight_dtype is currently not supported "
               "for rmsnorm_bwd_add with TE backend");
    is_aligned = is_ptr_aligned(x.data.dptr, gamma.data.dptr, rsigma.data.dptr, dx->data.dptr,
                                dz.data.dptr, dgamma->data.dptr, add.data.dptr);
  }

  auto plan = NormalizationPlanRegistry::getInstance().getNormalizationPlan(
      norm_backend, NVTE_Norm_Type::RMSNorm, NVTE_Norm_Stage::BackwardAdd,
      gamma.data.dtype,  // wtype
      x.data.dtype,      // itype
      gamma.data.dtype,  // otype
      rows,              // batch_size
      cols,              // hidden_size
      multiprocessorCount, zero_centered_gamma, is_aligned, NVTE_DELAYED_TENSOR_SCALING, true,
      gamma_in_weight_dtype);

  if (workspace->data.numel() == 0) {
    workspace->data.shape = plan->getWorkspaceShape();
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_CHECK(workspace->data.shape == plan->getWorkspaceShape());
    plan->execute(x.data.dptr, gamma.data.dptr, nullptr /*mu*/, rsigma.data.dptr, dx->data.dptr,
                  dz.data.dptr, add.data.dptr, nullptr /*dbeta*/, dgamma->data.dptr,
                  workspace->data.dptr, stream);
  }
  return;
}

}  // namespace transformer_engine

void nvte_rmsnorm_fwd(const NVTETensor x,      // Nxhidden_size
                      const NVTETensor gamma,  // hidden_size
                      const float epsilon, NVTETensor z, NVTETensor rsigma, NVTETensor workspace,
                      const int multiprocessorCount, const bool zero_centered_gamma,
                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_rmsnorm_fwd);
  using namespace transformer_engine;
  rmsnorm_fwd(*convertNVTETensorCheck(x), *convertNVTETensorCheck(gamma), epsilon,
              convertNVTETensor(z), convertNVTETensor(rsigma), convertNVTETensor(workspace),
              multiprocessorCount, zero_centered_gamma, stream);
}

void nvte_rmsnorm_bwd(const NVTETensor dz,      // Nxhidden_size
                      const NVTETensor x,       // Nxhidden_size
                      const NVTETensor rsigma,  // N, FP32!
                      const NVTETensor gamma,   // hidden_size
                      NVTETensor dx, NVTETensor dgamma, NVTETensor workspace,
                      const int multiprocessorCount, const bool zero_centered_gamma,
                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_rmsnorm_bwd);
  using namespace transformer_engine;
  rmsnorm_bwd(*convertNVTETensorCheck(dz), *convertNVTETensorCheck(x),
              *convertNVTETensorCheck(rsigma), *convertNVTETensorCheck(gamma),
              convertNVTETensor(dx), convertNVTETensor(dgamma), convertNVTETensor(workspace),
              multiprocessorCount, zero_centered_gamma, stream);
}

void nvte_rmsnorm_bwd_add(const NVTETensor dz,      // Nxhidden_size
                          const NVTETensor x,       // Nxhidden_size
                          const NVTETensor add,     // Nxhidden_size
                          const NVTETensor rsigma,  // N, FP32!
                          const NVTETensor gamma,   // hidden_size
                          NVTETensor dx, NVTETensor dgamma, NVTETensor workspace,
                          const int multiprocessorCount, const bool zero_centered_gamma,
                          cudaStream_t stream) {
  NVTE_API_CALL(nvte_rmsnorm_bwd_add);
  using namespace transformer_engine;
  rmsnorm_bwd_add(*convertNVTETensorCheck(dz), *convertNVTETensorCheck(x),
                  *convertNVTETensorCheck(add), *convertNVTETensorCheck(rsigma),
                  *convertNVTETensorCheck(gamma), convertNVTETensor(dx), convertNVTETensor(dgamma),
                  convertNVTETensor(workspace), multiprocessorCount, zero_centered_gamma, stream);
}

bool nvte_rmsnorm_bwd_uses_single_sweep(const NVTETensor dz, const NVTETensor x,
                                        const NVTETensor rsigma, const NVTETensor gamma,
                                        const NVTETensor dx, const NVTETensor dgamma,
                                        const bool zero_centered_gamma) {
  NVTE_API_CALL(nvte_rmsnorm_bwd_uses_single_sweep);
  using namespace transformer_engine;
  using namespace transformer_engine::normalization;
  return use_rmsnorm_bwd_single_sweep() &&
         is_supported_by_rmsnorm_bwd_single_sweep(
             *convertNVTETensorCheck(dz), *convertNVTETensorCheck(x),
             *convertNVTETensorCheck(rsigma), *convertNVTETensorCheck(gamma),
             *convertNVTETensorCheck(dx), *convertNVTETensorCheck(dgamma), nullptr,
             zero_centered_gamma, nullptr);
}

namespace {
const transformer_engine::Tensor *optional_tensor(const NVTETensor t) {
  return t == nullptr ? nullptr : transformer_engine::convertNVTETensorCheck(t);
}
}  // namespace

bool nvte_rmsnorm_bwd_fused_supported(const NVTETensor dz, const NVTETensor dz2,
                                      const NVTETensor x, const NVTETensor add,
                                      const NVTETensor rsigma, const NVTETensor gamma,
                                      const NVTETensor dx, const NVTETensor dgamma,
                                      const NVTETensor dx_quantized,
                                      const bool zero_centered_gamma) {
  NVTE_API_CALL(nvte_rmsnorm_bwd_fused_supported);
  using namespace transformer_engine;
  using namespace transformer_engine::normalization;
  return is_supported_by_rmsnorm_bwd_single_sweep(
      *convertNVTETensorCheck(dz), *convertNVTETensorCheck(x), *convertNVTETensorCheck(rsigma),
      *convertNVTETensorCheck(gamma), *convertNVTETensorCheck(dx), *convertNVTETensorCheck(dgamma),
      optional_tensor(add), zero_centered_gamma, nullptr, optional_tensor(dz2),
      optional_tensor(dx_quantized));
}

void nvte_rmsnorm_bwd_fused(const NVTETensor dz, const NVTETensor dz2, const NVTETensor x,
                            const NVTETensor add, const NVTETensor rsigma, const NVTETensor gamma,
                            NVTETensor dx, NVTETensor dgamma, NVTETensor dx_quantized,
                            NVTETensor workspace, const bool zero_centered_gamma,
                            cudaStream_t stream) {
  NVTE_API_CALL(nvte_rmsnorm_bwd_fused);
  using namespace transformer_engine;
  using namespace transformer_engine::normalization;
  const Tensor &dz_t = *convertNVTETensorCheck(dz);
  const Tensor &x_t = *convertNVTETensorCheck(x);
  const Tensor &rsigma_t = *convertNVTETensorCheck(rsigma);
  const Tensor &gamma_t = *convertNVTETensorCheck(gamma);
  Tensor *dx_t = convertNVTETensorCheck(dx);
  Tensor *dgamma_t = convertNVTETensorCheck(dgamma);
  Tensor *workspace_t = convertNVTETensorCheck(workspace);
  const Tensor *dz2_t = optional_tensor(dz2);
  const Tensor *add_t = optional_tensor(add);
  Tensor *dxq_t = dx_quantized == nullptr ? nullptr : convertNVTETensorCheck(dx_quantized);
  std::string reason;
  NVTE_CHECK(is_supported_by_rmsnorm_bwd_single_sweep(dz_t, x_t, rsigma_t, gamma_t, *dx_t,
                                                      *dgamma_t, add_t, zero_centered_gamma,
                                                      &reason, dz2_t, dxq_t),
             "nvte_rmsnorm_bwd_fused: unsupported configuration: ", reason);
  const auto [rows, cols] = x_t.flat_2d_dims();
  const size_t bytes = rmsnorm_bwd_single_sweep_workspace_bytes(rows, cols);
  if (workspace_t->data.numel() == 0) {
    workspace_t->data.shape = {bytes};
    workspace_t->data.dtype = DType::kByte;
    return;
  }
  NVTE_CHECK(workspace_t->data.dtype == DType::kByte && workspace_t->data.shape == Shape{bytes},
             "nvte_rmsnorm_bwd_fused: unexpected workspace");
  rmsnorm_bwd_single_sweep(dz_t, x_t, rsigma_t, gamma_t, add_t, dx_t, dgamma_t,
                           workspace_t->data.dptr, zero_centered_gamma, stream, dz2_t, dxq_t);
}
