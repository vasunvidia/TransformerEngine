/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "philox.h"
#include <transformer_engine/layer_norm.h>
#include <vector>
#include "../common.h"

/*

Supported Type combinations:

input    residual  output
=========================
fp16     fp16      fp16
bf16     bf16      bf16
fp16     fp8       fp16
bf16     fp8       bf16

Remarks:
Output type = Input type
Compute always in FP32

*/

namespace transformer_engine {

template <typename input_t, typename residual_t>
__global__ void dropout_add_impl(
    const input_t *input,
    const residual_t *res,
    input_t *out,
    input_t *out_mask,
    const uint64_t *rng_state,
    const fp32 *r_scale_inv,
    fp32 p_dropout,
    int cols) {
    int tidx_global = blockIdx.x * blockDim.x + threadIdx.x;
    Philox ph(rng_state[0], tidx_global, rng_state[1]);
    const input_t *inp_ptr = input + blockIdx.x * cols;
    const residual_t *res_ptr = res + blockIdx.x * cols;
    input_t *out_ptr = out + blockIdx.x * cols;
    input_t *out_mask_ptr = out_mask + blockIdx.x * cols;

    fp32 p_keep = 1.f - p_dropout;
    bool p_keep_eq_0 = p_keep == 0;
    const fp32 rp_keep = 1.f / ((fp32)(p_keep_eq_0) + p_keep);
    #pragma unroll
    for (int i=threadIdx.x*4; i<cols; i+=blockDim.x*4) {
        const float4 tmp = uniform4(ph());
        const fp32 rand[4] = {tmp.x, tmp.y, tmp.z, tmp.w};
        for (int j=0; j<4; j++) {
            int idx = i + j;
            if (idx < cols) {
                const fp32 dmask = rand[j] <= p_keep ? rp_keep : 0.f;
                fp32 result = dmask * static_cast<fp32>(inp_ptr[idx]);
                result += static_cast<fp32>(res_ptr[idx]) * (*r_scale_inv);
                out_ptr[idx] = static_cast<input_t>(result);
                out_mask_ptr[idx] = static_cast<input_t>(dmask);
            }
        }
    }
}

template<typename input_t, typename residual_t>
void dispatch_dropout_add(
    const input_t *input,
    const residual_t *res,
    input_t *out,
    input_t *out_mask,
    const uint64_t *rng_state,
    const fp32* r_scale_inv,
    int rows,
    int cols,
    float p_dropout,
    cudaStream_t stream) {
    constexpr int threads_per_block = 128;
    dropout_add_impl<input_t, residual_t> <<<rows, threads_per_block, 0, stream>>> (input, res, out, out_mask, rng_state, r_scale_inv, p_dropout, cols);
}

void dropout_add(const Tensor& input,
                 const Tensor& residual,
                 const float p_dropout,
                 Tensor* output,
                 Tensor* output_mask,
		 const Tensor& rng_state,
                 cudaStream_t stream
) {
    CheckInputTensor(input, "input");
    CheckInputTensor(residual, "residual");

    CheckOutputTensor(*output, "output");
    NVTE_CHECK(is_fp8_dtype(residual.data.dtype),
             "Residual must have FP8 type.");

    NVTE_CHECK(input.data.shape.size() == 2);

    const size_t rows = input.data.shape[0];
    const size_t cols = input.data.shape[1];

    NVTE_CHECK(p_dropout >= 0.f);
    NVTE_CHECK(output->data.shape == input.data.shape);

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(residual.data.dtype, RType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
        const fp32* r_scale_inv = reinterpret_cast<const fp32*>(residual.scale_inv.dptr);
        dispatch_dropout_add<IType, RType>(
            reinterpret_cast<const IType*>(input.data.dptr),
            reinterpret_cast<const RType*>(residual.data.dptr),
            reinterpret_cast<IType*>(output->data.dptr),
            reinterpret_cast<IType*>(output_mask->data.dptr),
            reinterpret_cast<const uint64_t*>(rng_state.data.dptr),
            r_scale_inv,
            rows,
	    cols,
            p_dropout,
            stream);
      );  // NOLINT(*)
    );  // NOLINT(*)

    // Set the kernel runtime parameters.
    //params.scale = z->scale.dptr;

    return;
}

}  // namespace transformer_engine

void nvte_dropout_add(const NVTETensor input,
                      const NVTETensor residual,
                      const float p_dropout,
                      NVTETensor output,
                      NVTETensor output_mask,
		      cudaStream_t stream,
		      const NVTETensor rng_state) {
  NVTE_API_CALL(nvte_dropout_add);
  using namespace transformer_engine;
  dropout_add(*reinterpret_cast<const Tensor*>(input),
              *reinterpret_cast<const Tensor*>(residual),
              p_dropout,
              reinterpret_cast<Tensor*>(output),
              reinterpret_cast<Tensor*>(output_mask),
	      *reinterpret_cast<const Tensor*>(rng_state),
	      stream);
}

