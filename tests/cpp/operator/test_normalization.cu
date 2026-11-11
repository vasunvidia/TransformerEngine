/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"
#include "test_normalization.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename InputType, typename OutputType>
void performTest(const size_t N, const size_t H, const bool zero_centered_gamma,
                 const NormType norm_type, const bool use_cudnn,
                 const bool zero_centered_gamma_in_weight_dtype, const bool fused_bwd_add) {
  if (sizeof(InputType) < sizeof(OutputType)) {
    GTEST_SKIP() << "LN kernel does not support OutputType > InputType";
  }

  if (norm_type == LayerNorm && fused_bwd_add) {
    GTEST_SKIP() << "Fused LN backward+add not currently supported";
  }

  if (fused_bwd_add && zero_centered_gamma_in_weight_dtype) {
    GTEST_SKIP() << "zero_centered_gamma_in_weight_dtype not currently supported "
                 << "in fused norm backward+add";
  }

  if (getDeviceComputeCapability() < hopperComputeCapability && use_cudnn) {
    GTEST_SKIP() << "cuDNN normalizations not supported on pre-Hopper GPUs yet!";
  }

  if (fused_bwd_add && use_cudnn && (cudnnGetVersion() < 92100)) {
    GTEST_SKIP() << "cuDNN < 9.21 does not support fused RMSNorm backward+add";
  }

  using WeightType = InputType;
  DType itype = TypeInfo<InputType>::dtype;
  DType wtype = TypeInfo<WeightType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  if ((itype == DType::kBFloat16 && otype == DType::kFloat16) ||
      (itype == DType::kFloat16 && otype == DType::kBFloat16)) {
    GTEST_SKIP() << "LN kernel does not support mixing Float16 and BFloat16";
  }

  Tensor input("input", std::vector<size_t>{ N, H }, itype);
  Tensor z("z", std::vector<size_t>{ N, H }, otype);
  Tensor gamma("gamma", std::vector<size_t>{ H }, wtype);
  Tensor beta("beta", std::vector<size_t>{ H }, wtype);
  Tensor mu("mu", std::vector<size_t>{ N }, DType::kFloat32);
  Tensor rsigma("rsigma", std::vector<size_t>{ N }, DType::kFloat32);
  Tensor dz("dz", std::vector<size_t>{ N, H }, wtype);
  Tensor bwd_add("bwd_add", std::vector<size_t>{ N, H }, wtype);
  Tensor dx("dx", std::vector<size_t>{ N, H }, itype);
  Tensor dgamma("dgamma", std::vector<size_t>{ H }, wtype);
  Tensor dbeta("dbeta", std::vector<size_t>{ H }, wtype);
  Tensor workspace_fwd, workspace_bwd;

  fillUniform(&input);
  fillUniform(&gamma);
  fillUniform(&beta);
  setRandomScale(&z);
  fillUniform(&dz);
  if (fused_bwd_add) {
    fillUniform(&bwd_add);
  } else {
    fillCase<WeightType>(&bwd_add, zeros);
  }

  std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(N * H);
  std::unique_ptr<float[]> ref_mu = std::make_unique<float[]>(N);
  std::unique_ptr<float[]> ref_rsigma = std::make_unique<float[]>(N);
  std::unique_ptr<InputType[]> ref_dx = std::make_unique<InputType[]>(N * H);
  std::unique_ptr<WeightType[]> ref_dgamma = std::make_unique<InputType[]>(H);
  std::unique_ptr<WeightType[]> ref_dbeta = std::make_unique<InputType[]>(H);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  if ((!use_cudnn || !zero_centered_gamma) && zero_centered_gamma_in_weight_dtype) {
    // Skip duplicate tests when zero_centered_gamma_in_weight_dtype is true and won't affect the implementation
    GTEST_SKIP() << "Zero-centered gamma in weight dtype is only supported with cuDNN backend";
  }

  if (use_cudnn){
    nvte_enable_cudnn_norm_fwd(true);
    nvte_enable_cudnn_norm_bwd(true);

    // Zero-centered gamma in weight dtype only supported by CuDNN backend currently
    if (zero_centered_gamma_in_weight_dtype) {
      nvte_enable_zero_centered_gamma_in_weight_dtype(true);
    } else {
      nvte_enable_zero_centered_gamma_in_weight_dtype(false);
    }
  }

  // Forward kernel
  float epsilon = 1e-5;
  if (norm_type == LayerNorm){
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       z.data(), mu.data(), rsigma.data(), workspace_fwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, 0);
    workspace_fwd = Tensor("workspace", workspace_fwd.rowwise_shape(), workspace_fwd.dtype());
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       z.data(), mu.data(), rsigma.data(), workspace_fwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, 0);

    nvte_layernorm_bwd(dz.data(), input.data(),
                       mu.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), dbeta.data(),
                       workspace_bwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, 0);
    workspace_bwd = Tensor("workspace", workspace_bwd.rowwise_shape(), workspace_bwd.dtype());
    nvte_layernorm_bwd(dz.data(), input.data(),
                       mu.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), dbeta.data(),
                       workspace_bwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, 0);
  } else {
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace_fwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, 0);
    workspace_fwd = Tensor("workspace", workspace_fwd.rowwise_shape(), workspace_fwd.dtype());
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace_fwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, 0);

    if (fused_bwd_add) {
      nvte_rmsnorm_bwd_add(dz.data(), input.data(), bwd_add.data(), rsigma.data(), gamma.data(),
                           dx.data(), dgamma.data(), workspace_bwd.data(),
                           prop.multiProcessorCount, zero_centered_gamma, 0);
      workspace_bwd = Tensor("workspace", workspace_bwd.rowwise_shape(), workspace_bwd.dtype());
      nvte_rmsnorm_bwd_add(dz.data(), input.data(), bwd_add.data(), rsigma.data(), gamma.data(),
                           dx.data(), dgamma.data(), workspace_bwd.data(),
                           prop.multiProcessorCount, zero_centered_gamma, 0);
    } else {
      nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), workspace_bwd.data(), prop.multiProcessorCount,
                       zero_centered_gamma, 0);
      workspace_bwd = Tensor("workspace", workspace_bwd.rowwise_shape(), workspace_bwd.dtype());
      nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), workspace_bwd.data(), prop.multiProcessorCount,
                       zero_centered_gamma, 0);
    }
  }

  if (use_cudnn){
    nvte_enable_cudnn_norm_fwd(false);
    nvte_enable_cudnn_norm_bwd(false);

    // Zero-centered gamma in weight dtype only supported by CuDNN backend currently
    if (zero_centered_gamma_in_weight_dtype) {
      nvte_enable_zero_centered_gamma_in_weight_dtype(false);
    }
  }

  // Reference implementations
  // use the GPU stats to tighten the tolerances
  mu.to_cpu();
  rsigma.to_cpu();
  float ref_amax;
  compute_ref_stats(norm_type, input.rowwise_cpu_dptr<InputType>(), ref_mu.get(),
                    ref_rsigma.get(), N, H, epsilon);
  float ref_scale = isFp8Type(otype) ? z.scale() : 1.f;
  compute_ref_output(norm_type, input.rowwise_cpu_dptr<InputType>(),
                     gamma.rowwise_cpu_dptr<WeightType>(),
                     beta.rowwise_cpu_dptr<WeightType>(),
                     ref_output.get(),
                     mu.rowwise_cpu_dptr<float>(),
                     rsigma.rowwise_cpu_dptr<float>(),
                     N, H,
                     &ref_amax,
                     ref_scale,
                     zero_centered_gamma,
                     use_cudnn,
                     zero_centered_gamma_in_weight_dtype);
  compute_ref_backward(norm_type, dz.rowwise_cpu_dptr<WeightType>(),
                       bwd_add.rowwise_cpu_dptr<WeightType>(),
                       input.rowwise_cpu_dptr<InputType>(),
                       mu.rowwise_cpu_dptr<float>(), rsigma.rowwise_cpu_dptr<float>(),
                       gamma.rowwise_cpu_dptr<WeightType>(),
                       ref_dx.get(), ref_dgamma.get(), ref_dbeta.get(),
                       N, H, zero_centered_gamma,
                       use_cudnn,
                       zero_centered_gamma_in_weight_dtype);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
  if (isFp8Type(otype)) {
    compareResults("amax", z.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / ref_scale;
    compareResults("scale_inv", z.rowwise_scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }

  auto [atol_stats, rtol_stats] = getTolerances(DType::kFloat32);
  rtol_stats = 5e-5;
  compareResults("mu", mu, ref_mu.get(), true, atol_stats, rtol_stats);
  compareResults("rsigma", rsigma, ref_rsigma.get(), true, atol_stats, rtol_stats);

  auto [atol, rtol] = getTolerances(otype);
  if (otype == DType::kFloat32) {
    atol = 5e-7;
  }
  compareResults("output", z, ref_output.get(), true, atol, rtol);

  double atol_bwd = 5e-4;
  double rtol_bwd = 5e-4;
  compareResults("dx", dx, ref_dx.get(), true, atol_bwd, rtol_bwd);
  compareResults("dgamma", dgamma, ref_dgamma.get(), true, atol_bwd, rtol_bwd);
  compareResults("dbeta", dbeta, ref_dbeta.get(), true, atol_bwd, rtol_bwd);
}

std::vector<std::pair<size_t, size_t>> test_cases = {
  {71, 229},
  {29, 541},
  {768, 6144},
  {2048, 12288},
};

}  // namespace

class NormTestSuite : public ::testing::TestWithParam<std::tuple<bool,
                                                                 NormType,
                                                                 transformer_engine::DType,
                                                                 transformer_engine::DType,
                                                                 std::pair<size_t, size_t>,
                                                                 bool,
                                                                 bool,
                                                                 bool>> {};

TEST_P(NormTestSuite, TestNorm) {
  using namespace transformer_engine;
  using namespace test;

  const bool use_cudnn = std::get<0>(GetParam());
  const NormType norm_type = std::get<1>(GetParam());
  const DType input_type = std::get<2>(GetParam());
  const DType output_type = std::get<3>(GetParam());
  const auto size = std::get<4>(GetParam());
  const bool zero_centered_gamma = std::get<5>(GetParam());
  const bool cudnn_zero_centered_gamma_in_weight_dtype = std::get<6>(GetParam());
  const bool fused_bwd_add = std::get<7>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
    if (output_type == DType::kFloat8E4M3) {
      performTest<InputType, fp8e4m3>(
        size.first,
        size.second,
        zero_centered_gamma,
        norm_type,
        use_cudnn,
        cudnn_zero_centered_gamma_in_weight_dtype,
        fused_bwd_add
      );
    } else {
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
        output_type, OutputType,
        performTest<InputType, OutputType>(
          size.first,
          size.second,
          zero_centered_gamma,
          norm_type,
          use_cudnn,
          cudnn_zero_centered_gamma_in_weight_dtype,
          fused_bwd_add
        );
      );
    }
  );
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  NormTestSuite,
  ::testing::Combine(
    ::testing::Values(true, false),
    ::testing::Values(NormType::LayerNorm, NormType::RMSNorm),
    ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
    ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16, DType::kFloat8E4M3),
    ::testing::ValuesIn(test_cases),
    ::testing::Values(false, true),
    ::testing::Values(false, true),
    ::testing::Values(false, true)),
  [](const testing::TestParamInfo<NormTestSuite::ParamType>& info) {
    auto backend = std::get<0>(info.param) == false ? "Te" : "Cudnn";
    std::string name =
      backend +
      normToString.at(std::get<1>(info.param)) + "_" +
      test::typeName(std::get<2>(info.param)) + "X" +
      test::typeName(std::get<3>(info.param)) + "X" +
      std::to_string(std::get<4>(info.param).first) + "X" +
      std::to_string(std::get<4>(info.param).second) + "X" +
      std::to_string(std::get<5>(info.param)) + "X" +
      std::to_string(std::get<6>(info.param)) + "X" +
      std::to_string(std::get<7>(info.param));
    return name;
  });

// Opt-in single-sweep RMSNorm backward (NVTE_RMSNORM_BWD_SINGLE_SWEEP /
// nvte_enable_rmsnorm_bwd_single_sweep). BF16 only. Checks that the kernel is selected exactly
// for the supported shapes, that dx and dgamma match the CPU reference and the regular backward
// to within BF16 rounding (the arithmetic order differs, so the bits may differ), and that two
// runs give bitwise identical results. Unsupported shapes must fall back to the regular backward
// and give exactly its results.
namespace {

void run_rmsnorm_bwd(Tensor &dz, Tensor &x, Tensor &rsigma, Tensor &gamma, Tensor &dx,
                     Tensor &dgamma, const int sm_count, const bool zero_centered_gamma) {
  Tensor workspace;
  nvte_rmsnorm_bwd(dz.data(), x.data(), rsigma.data(), gamma.data(), dx.data(), dgamma.data(),
                   workspace.data(), sm_count, zero_centered_gamma, 0);
  workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
  nvte_rmsnorm_bwd(dz.data(), x.data(), rsigma.data(), gamma.data(), dx.data(), dgamma.data(),
                   workspace.data(), sm_count, zero_centered_gamma, 0);
}

bool bitwise_equal(Tensor &a, Tensor &b, const size_t numel) {
  a.to_cpu();
  b.to_cpu();
  return std::memcmp(a.rowwise_cpu_dptr<bf16>(), b.rowwise_cpu_dptr<bf16>(),
                     numel * sizeof(bf16)) == 0;
}

}  // namespace

class RMSNormBwdSingleSweepTestSuite
    : public ::testing::TestWithParam<std::tuple<std::pair<size_t, size_t>, bool>> {};

TEST_P(RMSNormBwdSingleSweepTestSuite, TestRMSNormBwdSingleSweep) {
  if (getDeviceComputeCapability() < hopperComputeCapability) {
    GTEST_SKIP() << "The single-sweep RMSNorm backward requires SM 9.0+";
  }
  const size_t N = std::get<0>(GetParam()).first;
  const size_t H = std::get<0>(GetParam()).second;
  const bool zero_centered_gamma = std::get<1>(GetParam());
  const bool expect_single_sweep = N % 32 == 0 && H % 32 == 0 && H <= 3072;
  const DType t = DType::kBFloat16;

  Tensor input("input", std::vector<size_t>{N, H}, t);
  Tensor z("z", std::vector<size_t>{N, H}, t);
  Tensor gamma("gamma", std::vector<size_t>{H}, t);
  Tensor rsigma("rsigma", std::vector<size_t>{N}, DType::kFloat32);
  Tensor dz("dz", std::vector<size_t>{N, H}, t);
  Tensor dx_default("dx_default", std::vector<size_t>{N, H}, t);
  Tensor dgamma_default("dgamma_default", std::vector<size_t>{H}, t);
  Tensor dx("dx", std::vector<size_t>{N, H}, t);
  Tensor dgamma("dgamma", std::vector<size_t>{H}, t);
  Tensor dx2("dx2", std::vector<size_t>{N, H}, t);
  Tensor dgamma2("dgamma2", std::vector<size_t>{H}, t);
  fillUniform(&input);
  fillUniform(&gamma);
  fillUniform(&dz);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const float epsilon = 1e-5;
  Tensor workspace_fwd;
  nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon, z.data(), rsigma.data(),
                   workspace_fwd.data(), prop.multiProcessorCount, zero_centered_gamma, 0);
  workspace_fwd = Tensor("workspace", workspace_fwd.rowwise_shape(), workspace_fwd.dtype());
  nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon, z.data(), rsigma.data(),
                   workspace_fwd.data(), prop.multiProcessorCount, zero_centered_gamma, 0);

  nvte_enable_rmsnorm_bwd_single_sweep(false);
  EXPECT_FALSE(nvte_rmsnorm_bwd_uses_single_sweep(dz.data(), input.data(), rsigma.data(),
                                                  gamma.data(), dx.data(), dgamma.data(),
                                                  zero_centered_gamma));
  run_rmsnorm_bwd(dz, input, rsigma, gamma, dx_default, dgamma_default, prop.multiProcessorCount,
                  zero_centered_gamma);

  nvte_enable_rmsnorm_bwd_single_sweep(true);
  EXPECT_EQ(nvte_rmsnorm_bwd_uses_single_sweep(dz.data(), input.data(), rsigma.data(),
                                               gamma.data(), dx.data(), dgamma.data(),
                                               zero_centered_gamma),
            expect_single_sweep);
  run_rmsnorm_bwd(dz, input, rsigma, gamma, dx, dgamma, prop.multiProcessorCount,
                  zero_centered_gamma);
  run_rmsnorm_bwd(dz, input, rsigma, gamma, dx2, dgamma2, prop.multiProcessorCount,
                  zero_centered_gamma);
  nvte_enable_rmsnorm_bwd_single_sweep(false);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Deterministic: two runs, the same bits.
  EXPECT_TRUE(bitwise_equal(dx, dx2, N * H)) << "dx differs between two runs";
  EXPECT_TRUE(bitwise_equal(dgamma, dgamma2, H)) << "dgamma differs between two runs";
  if (!expect_single_sweep) {
    // Fallback: the regular backward, bit for bit.
    EXPECT_TRUE(bitwise_equal(dx, dx_default, N * H)) << "fallback dx differs";
    EXPECT_TRUE(bitwise_equal(dgamma, dgamma_default, H)) << "fallback dgamma differs";
  }

  // Against the CPU reference (the regular backward's tolerances).
  rsigma.to_cpu();
  std::vector<bf16> zeros(N * H, static_cast<bf16>(0.f));
  std::vector<float> mu(N, 0.f);
  std::unique_ptr<bf16[]> ref_dx = std::make_unique<bf16[]>(N * H);
  std::unique_ptr<bf16[]> ref_dgamma = std::make_unique<bf16[]>(H);
  std::unique_ptr<bf16[]> ref_dbeta = std::make_unique<bf16[]>(H);
  compute_ref_backward(RMSNorm, dz.rowwise_cpu_dptr<bf16>(), zeros.data(),
                       input.rowwise_cpu_dptr<bf16>(), mu.data(), rsigma.rowwise_cpu_dptr<float>(),
                       gamma.rowwise_cpu_dptr<bf16>(), ref_dx.get(), ref_dgamma.get(),
                       ref_dbeta.get(), N, H, zero_centered_gamma, false, false);
  const double atol_bwd = 5e-4;
  const double rtol_bwd = 5e-4;
  compareResults("dx", dx, ref_dx.get(), true, atol_bwd, rtol_bwd);
  compareResults("dgamma", dgamma, ref_dgamma.get(), true, atol_bwd, rtol_bwd);

  // Against the regular backward: within BF16 rounding (adjacent values allowed).
  dx_default.to_cpu();
  dgamma_default.to_cpu();
  compareResults("dx_vs_default", dx, dx_default.rowwise_cpu_dptr<bf16>(), true, atol_bwd,
                 rtol_bwd);
  compareResults("dgamma_vs_default", dgamma, dgamma_default.rowwise_cpu_dptr<bf16>(), true,
                 atol_bwd, rtol_bwd);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  RMSNormBwdSingleSweepTestSuite,
  ::testing::Combine(
    ::testing::Values(
      // supported: gpt-oss-20b (3 x 8192 tokens, hidden 2880) and other shapes
      std::make_pair<size_t, size_t>(24576, 2880),
      std::make_pair<size_t, size_t>(2048, 2880),
      std::make_pair<size_t, size_t>(32, 2880),
      std::make_pair<size_t, size_t>(4096, 3072),
      std::make_pair<size_t, size_t>(96, 1024),
      std::make_pair<size_t, size_t>(65536, 32),
      // fallback: rows not divisible by 32, hidden size not divisible by 32, hidden size > 3072
      std::make_pair<size_t, size_t>(71, 2880),
      std::make_pair<size_t, size_t>(64, 2896),
      std::make_pair<size_t, size_t>(64, 4096)),
    ::testing::Values(false, true)),
  [](const testing::TestParamInfo<RMSNormBwdSingleSweepTestSuite::ParamType>& info) {
    return std::to_string(std::get<0>(info.param).first) + "X" +
           std::to_string(std::get<0>(info.param).second) + "X" +
           std::to_string(std::get<1>(info.param));
  });
