/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file adam_flat.cuh
 *  \brief Adam update of flat buffers partitioned into parameter-group segments.
 *
 *  All buffers are indexed by the same element offset: element i of the gradient, of the FP32
 *  master parameter, of both moments and of the model parameter belong to the same parameter
 *  element. The elements are partitioned into segments [start, end) with a parameter-group index;
 *  elements outside every segment (e.g. alignment padding) are neither read nor written.
 *
 *  Work is split into tiles of kBlock * kVec * kUnroll elements. A CUDA block finds the segments
 *  overlapping its tile with a binary search over the sorted segment table and updates each
 *  overlap with vectorized loads and stores; only the vectors cut by a segment boundary take the
 *  scalar path.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_MULTI_TENSOR_ADAM_FLAT_CUH_
#define TRANSFORMER_ENGINE_COMMON_MULTI_TENSOR_ADAM_FLAT_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "../utils.cuh"

namespace transformer_engine {
namespace multi_tensor_adam_flat {

/*! Maximum number of parameter groups of one launch. */
constexpr int kMaxGroups = 32;

/*! Rows of the device hyperparameter table of the capturable variant: a FP32 array of shape
 *  [kNumHParams, num_groups] in this row order. */
enum HParamRow : int {
  kLr = 0,
  kBeta1,
  kBeta2,
  kBeta1Correction,
  kBeta2Correction,
  kEpsilon,
  kWeightDecay,
  kNumHParams
};

/*! Per-group hyperparameters, passed by value to the kernel. */
struct GroupArgs {
  float lr[kMaxGroups];
  float beta1[kMaxGroups];
  float beta2[kMaxGroups];
  float beta1_correction[kMaxGroups];
  float beta2_correction[kMaxGroups];
  float epsilon[kMaxGroups];
  float weight_decay[kMaxGroups];
  int num_groups;
};

/*! Element-wise Adam update.
 *
 *  Computes exactly the operations of AdamFunctor in adam.cu as compiled with --use_fast_math
 *  (this header must be compiled with that flag too, like adam.cu): with g the gradient,
 *    L2 mode:    g = fma(decay, p, g)
 *    m = fma(beta1, m, (1 - beta1) * g)
 *    v = fma(beta2, v, ((1 - beta2) * g) * g)
 *    update = (m / bias_correction1) / (sqrt(v / bias_correction2) + epsilon)
 *    AdamW mode: update = fma(decay, p, update)
 *    p = fma(-lr, update, p)
 *  where / and sqrt are the approximate fast-math operations. AdamFunctor writes the same update
 *  as plain expressions and lets the compiler contract them into these multiply-adds; here the
 *  contractions are explicit, because the compiler's choice of which product to fuse into
 *  "a * b + c * d" depends on the surrounding code and would otherwise differ between kernels.
 */
__device__ __forceinline__ void adam_update(float &r_g, float &r_p, float &r_m, float &r_v,
                                            const float beta1, const float beta2,
                                            const float beta1_correction,
                                            const float beta2_correction, const float epsilon,
                                            const float lr, const int mode, const float decay) {
  if (mode == 0) {  // L2
    r_g = __fmaf_rn(decay, r_p, r_g);
  }
  r_m = __fmaf_rn(beta1, r_m, __fmul_rn(1.0f - beta1, r_g));
  r_v = __fmaf_rn(beta2, r_v, __fmul_rn(__fmul_rn(1.0f - beta2, r_g), r_g));
  const float next_m_unbiased = r_m / beta1_correction;
  const float next_v_unbiased = r_v / beta2_correction;
  const float denom = __fadd_rn(sqrtf(next_v_unbiased), epsilon);
  float update = next_m_unbiased / denom;
  if (mode != 0) {  // weight decay
    update = __fmaf_rn(decay, r_p, update);
  }
  r_p = __fmaf_rn(-lr, update, r_p);
}

/*! Gradient value used by the update.
 *
 *  The product is rounded on its own (never contracted into an FMA), exactly as the separate
 *  scaling kernel (multi_tensor_scale) writes the scaled FP32 gradient that AdamFunctor reads.
 */
template <typename GradT>
__device__ __forceinline__ float load_grad(const GradT g, const bool has_scale, const float scale) {
  const float r_g = static_cast<float>(g);
  return has_scale ? __fmul_rn(r_g, scale) : r_g;
}

/*! Index of the first segment with end > pos (segment ends increase with the index). */
__device__ __forceinline__ int64_t first_segment_ending_after(const int64_t *segments,
                                                              const int64_t num_segments,
                                                              const int64_t pos) {
  int64_t lo = 0;
  int64_t hi = num_segments;
  while (lo < hi) {
    const int64_t mid = (lo + hi) >> 1;
    if (__ldg(segments + 3 * mid + 1) <= pos) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/*! Adam update of one launch.
 *
 *  kDeviceHParams = false: the per-group hyperparameters are the values in `groups`.
 *  kDeviceHParams = true (capturable variant): they are read from the device table `hparams`
 *  (FP32, [kNumHParams, groups.num_groups], see HParamRow); only groups.num_groups is used. The
 *  arithmetic is the same, so equal hyperparameter values give bitwise equal results.
 */
template <int kBlock, int kVec, int kUnroll, typename GradT, typename ParamT, bool kWriteParam,
          bool kDeviceHParams = false>
__global__ void __launch_bounds__(kBlock)
    adam_flat_kernel(const GradT *__restrict__ g, ParamT *__restrict__ p,
                     float *__restrict__ p_master, float *__restrict__ m, float *__restrict__ v,
                     const int64_t *__restrict__ segments, const int64_t num_segments,
                     const int64_t numel, const float *__restrict__ grad_scale,
                     const GroupArgs groups, const int mode,
                     const float *__restrict__ hparams = nullptr) {
  constexpr int64_t kTile = static_cast<int64_t>(kBlock) * kVec * kUnroll;
  const bool has_scale = grad_scale != nullptr;
  const float scale = has_scale ? *grad_scale : 1.0f;
  const int64_t num_tiles = (numel + kTile - 1) / kTile;

  for (int64_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const int64_t tile_start = tile * kTile;
    const int64_t tile_end = min(tile_start + kTile, numel);
    for (int64_t s = first_segment_ending_after(segments, num_segments, tile_start);
         s < num_segments; ++s) {
      const int64_t seg_start = __ldg(segments + 3 * s);
      if (seg_start >= tile_end) break;
      const int64_t begin = max(seg_start, tile_start);
      const int64_t end = min(__ldg(segments + 3 * s + 1), tile_end);
      const int group = static_cast<int>(__ldg(segments + 3 * s + 2));
      if (group < 0 || group >= groups.num_groups) continue;
      float lr, beta1, beta2, beta1_correction, beta2_correction, epsilon, decay;
      if constexpr (kDeviceHParams) {
        const int n = groups.num_groups;
        lr = __ldg(hparams + kLr * n + group);
        beta1 = __ldg(hparams + kBeta1 * n + group);
        beta2 = __ldg(hparams + kBeta2 * n + group);
        beta1_correction = __ldg(hparams + kBeta1Correction * n + group);
        beta2_correction = __ldg(hparams + kBeta2Correction * n + group);
        epsilon = __ldg(hparams + kEpsilon * n + group);
        decay = __ldg(hparams + kWeightDecay * n + group);
      } else {
        lr = groups.lr[group];
        beta1 = groups.beta1[group];
        beta2 = groups.beta2[group];
        beta1_correction = groups.beta1_correction[group];
        beta2_correction = groups.beta2_correction[group];
        epsilon = groups.epsilon[group];
        decay = groups.weight_decay[group];
      }

      // Whole vectors inside [begin, end): vectorized loads, update, vectorized stores.
      float r_g[kUnroll][kVec];
      float r_p[kUnroll][kVec];
      float r_m[kUnroll][kVec];
      float r_v[kUnroll][kVec];
      bool whole[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; u++) {
        const int64_t i = tile_start + (static_cast<int64_t>(u) * kBlock + threadIdx.x) * kVec;
        whole[u] = i >= begin && i + kVec <= end;
        if (whole[u]) {
          Vec<GradT, kVec> g_vec;
          Vec<float, kVec> p_vec, m_vec, v_vec;
          g_vec.load_from(g + i);
          p_vec.load_from(p_master + i);
          m_vec.load_from(m + i);
          v_vec.load_from(v + i);
#pragma unroll
          for (int j = 0; j < kVec; j++) {
            r_g[u][j] = load_grad(g_vec.data.elt[j], has_scale, scale);
            r_p[u][j] = p_vec.data.elt[j];
            r_m[u][j] = m_vec.data.elt[j];
            r_v[u][j] = v_vec.data.elt[j];
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; u++) {
        if (whole[u]) {
#pragma unroll
          for (int j = 0; j < kVec; j++) {
            adam_update(r_g[u][j], r_p[u][j], r_m[u][j], r_v[u][j], beta1, beta2, beta1_correction,
                        beta2_correction, epsilon, lr, mode, decay);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; u++) {
        if (whole[u]) {
          const int64_t i = tile_start + (static_cast<int64_t>(u) * kBlock + threadIdx.x) * kVec;
          Vec<float, kVec> p_vec, m_vec, v_vec;
#pragma unroll
          for (int j = 0; j < kVec; j++) {
            p_vec.data.elt[j] = r_p[u][j];
            m_vec.data.elt[j] = r_m[u][j];
            v_vec.data.elt[j] = r_v[u][j];
          }
          p_vec.store_to(p_master + i);
          m_vec.store_to(m + i);
          v_vec.store_to(v + i);
          if constexpr (kWriteParam) {
            Vec<ParamT, kVec> out_vec;
#pragma unroll
            for (int j = 0; j < kVec; j++) {
              out_vec.data.elt[j] = static_cast<ParamT>(r_p[u][j]);
            }
            out_vec.store_to(p + i);
          }
        }
      }

      // Vectors cut by a segment boundary: element by element.
      if constexpr (kVec > 1) {
#pragma unroll
        for (int u = 0; u < kUnroll; u++) {
          if (!whole[u]) {
            const int64_t i = tile_start + (static_cast<int64_t>(u) * kBlock + threadIdx.x) * kVec;
            const int64_t lo = max(i, begin);
            const int64_t hi = min(i + kVec, end);
            for (int64_t e = lo; e < hi; e++) {
              float s_g = load_grad(g[e], has_scale, scale);
              float s_p = p_master[e];
              float s_m = m[e];
              float s_v = v[e];
              adam_update(s_g, s_p, s_m, s_v, beta1, beta2, beta1_correction, beta2_correction,
                          epsilon, lr, mode, decay);
              p_master[e] = s_p;
              m[e] = s_m;
              v[e] = s_v;
              if constexpr (kWriteParam) {
                p[e] = static_cast<ParamT>(s_p);
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace multi_tensor_adam_flat
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_MULTI_TENSOR_ADAM_FLAT_CUH_
