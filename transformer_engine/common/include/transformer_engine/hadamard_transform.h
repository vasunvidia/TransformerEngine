/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file hadamard_transform.h
 *  \brief Functions for Hadamard transforms.
 */

#ifndef TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_
#define TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Perform a randomized Hadamard transform on the input tensor.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input              Input tensor to apply Hadamard transform.
 *  \param[in,out]  output             Output tensor.
 *  \param[in]      random_sign_mask   16-bit sign mask.
 *  \param[in]      random_sign_mask_t 16-bit sign mask.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_hadamard_transform(const NVTETensor input, NVTETensor output, int random_sign_mask,
                             int random_sign_mask_t, cudaStream_t stream);

/*! \brief Perform the absolute maximum reduction on the input tensor with/without
 *         randomized hadamard transform. The rowwise result is the absolute maximum
 *         of the input tensor. The columnwise result is the absolute maximum of the
 *         input tensor transposed and applied randomized hadamard transformation.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input              Input tensor to apply Hadamard transform.
 *  \param[in,out]  output             Output tensor.
 *  \param[in]      random_sign_mask   16-bit sign mask.
 *  \param[in]      random_sign_mask_t 16-bit sign mask.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_hadamard_transform_amax(const NVTETensor input, NVTETensor output, int random_sign_mask,
                                  int random_sign_mask_t, cudaStream_t stream);

/*! \brief Compute the SwiGLU backward and the RHT amaxes of its result in one pass.
 *  This function is experimental and the API is not stable.
 *  Equivalent to nvte_dswiglu(grad, input, output) followed by
 *  nvte_hadamard_transform_amax(output, amax_output), bitwise. This is the amax
 *  pass of NVFP4 quantization with RHT applied to the SwiGLU backward output. A
 *  fused kernel is used when it reproduces the unfused results exactly (BF16
 *  tensors, both dimensions of grad multiples of 64, activation kernels built
 *  with NVTE_BUILD_ACTIVATION_WITH_FAST_MATH, compute capability 10.0 or newer);
 *  otherwise the two functions run in sequence.
 *  \param[in]      grad               Incoming gradient of shape [N, H].
 *  \param[in]      input              Forward input of shape [N, H * 2].
 *  \param[in,out]  output             SwiGLU backward output of shape [N, H * 2].
 *  \param[in,out]  amax_output        Tensor whose row-wise amax receives the amax of
 *                                     output and whose column-wise amax receives the
 *                                     amax of the RHT of the transposed output.
 *  \param[in]      random_sign_mask   16-bit sign mask.
 *  \param[in]      random_sign_mask_t 16-bit sign mask.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_dswiglu_hadamard_transform_amax(const NVTETensor grad, const NVTETensor input,
                                          NVTETensor output, NVTETensor amax_output,
                                          int random_sign_mask, int random_sign_mask_t,
                                          cudaStream_t stream);

/*! \brief Add two BF16 tensors and compute the NVFP4 RHT amaxes of the sum in one pass.
 *  This function is experimental and the API is not stable.
 *  output = a + b (each element bf16(float(a) + float(b)), as an elementwise BF16 add).
 *  amax[0] receives max|output| and amax[1] the amax of the 16-point RHT of output^T, bitwise
 *  equal to nvte_hadamard_transform_amax(output, ...) with random_sign_mask_t.
 *  Supports BF16 tensors whose flattened shape [M, N] has M a multiple of 64 and N a multiple
 *  of 128, on devices of compute capability 10.0 or newer.
 *
 *  Calling this function with workspace set to an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values. The
 *  workspace must be zero-filled before its first use; the operation leaves it zero-filled, so it
 *  can be reused by later calls on the same stream without a memset (the same workspace serves
 *  all functions of this group).
 *
 *  \param[in]      a                  First input tensor.
 *  \param[in]      b                  Second input tensor of the same shape.
 *  \param[out]     output             Sum of the same shape.
 *  \param[out]     amax               FP32 tensor with 2 elements.
 *  \param[in,out]  workspace          Workspace tensor.
 *  \param[in]      random_sign_mask_t 16-bit sign mask of the column-wise RHT.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_add_hadamard_transform_amax(const NVTETensor a, const NVTETensor b, NVTETensor output,
                                      NVTETensor amax, NVTETensor workspace, int random_sign_mask_t,
                                      cudaStream_t stream);

/*! \brief RoPE backward of the Q, K and V gradients into the gradient of the fused QKV
 *  projection output, with the NVFP4 RHT amaxes of the result.
 *  This function is experimental and the API is not stable.
 *  For the SBHD layout without interleaving or context parallelism: with g a Q or K head
 *  gradient, d < 64: out[d] = fma(g[d], cos[d], g[d + 64] * sin[d + 64]) and d >= 64:
 *  out[d] = fma(g[d], cos[d], -(g[d - 64] * sin[d - 64])) in FP32, rounded to BF16; V heads
 *  are copied. The output is laid out [s, b, h_kv, (h_q / h_kv + 2) * d] (the Q heads of a KV
 *  group, then its K and V heads). amax[0] receives max|out| and amax[1] the amax of the 16-point
 *  RHT of out^T, bitwise equal to nvte_hadamard_transform_amax(out, ...).
 *  Supports BF16 gradients with head dimension 128, s * b a multiple of 64 and b dividing 64, on
 *  devices of compute capability 10.0 or newer. The workspace works as in
 *  nvte_add_hadamard_transform_amax.
 *
 *  \param[in]      grad_q             Q gradient of shape [s, b, h_q, d].
 *  \param[in]      grad_k             K gradient of shape [s, b, h_kv, d].
 *  \param[in]      grad_v             V gradient of shape [s, b, h_kv, d].
 *  \param[in]      cos                FP32 cosine table with d values per position (>= s).
 *  \param[in]      sin                FP32 sine table of the same shape.
 *  \param[out]     grad_qkv           Gradient of the fused QKV output.
 *  \param[out]     amax               FP32 tensor with 2 elements.
 *  \param[in,out]  workspace          Workspace tensor.
 *  \param[in]      random_sign_mask_t 16-bit sign mask of the column-wise RHT.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_fused_qkv_rope_backward_hadamard_transform_amax(
    const NVTETensor grad_q, const NVTETensor grad_k, const NVTETensor grad_v, const NVTETensor cos,
    const NVTETensor sin, NVTETensor grad_qkv, NVTETensor amax, NVTETensor workspace,
    int random_sign_mask_t, cudaStream_t stream);

/*! \brief Row-wise NVFP4 cast with a given amax, plus the NVFP4 RHT amaxes of the input.
 *  This function is experimental and the API is not stable.
 *  The row-wise data and scales of output are bitwise those of the NVFP4 quantization (round to
 *  nearest, 1D 16-value blocks, E4M3 scales, no RHT on the row-wise data) with the per-tensor
 *  amax held in output's amax, for finite inputs. amax[0] receives max|input| and amax[1] the
 *  amax of the 16-point RHT of input^T (as nvte_hadamard_transform_amax), for a later
 *  column-wise RHT cast of the same input.
 *  Supports BF16 inputs with both flattened dimensions multiples of 128 and N / 128 a power of
 *  two, on devices of compute capability 10.0 or newer. The workspace works as in
 *  nvte_add_hadamard_transform_amax.
 *
 *  \param[in]      input              Input tensor.
 *  \param[in,out]  output             NVFP4 tensor with row-wise data only and its amax set.
 *  \param[out]     amax               FP32 tensor with 2 elements.
 *  \param[in,out]  workspace          Workspace tensor.
 *  \param[in]      random_sign_mask_t 16-bit sign mask of the column-wise RHT.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_quantize_rowwise_with_amax_hadamard_transform_amax(const NVTETensor input,
                                                             NVTETensor output, NVTETensor amax,
                                                             NVTETensor workspace,
                                                             int random_sign_mask_t,
                                                             cudaStream_t stream);

/*! \brief Perform the columnwise hadamard transform cast fusion.
 *
 *  \deprecated This function has been deprecated in favor of nvte_quantize_with_hadamard_transform.
 *
 *  \param[in]      input           Input tensor to apply Hadamard transform.
 *  \param[in,out]  output          Output tensor.
 *  \param[in]      hadamard_matrix Hadamard matrix.
 *  \param[in]      quant_config    Quantization configuration.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_hadamard_transform_cast_fusion_columnwise(const NVTETensor input, NVTETensor output,
                                                    const NVTETensor hadamard_matrix,
                                                    const NVTEQuantizationConfig quant_config,
                                                    cudaStream_t stream);

/*! \brief Perform the regular rowwise cast and columnwise hadamard transform cast fusion.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input           Input tensor to apply Hadamard transform.
 *  \param[in,out]  output          Output tensor.
 *  \param[in]      hadamard_matrix Hadamard matrix.
 *  \param[in]      quant_config    Quantization configuration.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_quantize_with_hadamard_transform(const NVTETensor input, NVTETensor output,
                                           const NVTETensor hadamard_matrix,
                                           const NVTEQuantizationConfig quant_config,
                                           cudaStream_t stream);

/*! \brief Split a tensor along dimension 0 and compute RHT amaxes for each split.
 *
 *  This function is experimental and the API is not stable.
 *
 *  This is intended for quantizing to NVFP4 with random Hadamard
 *  transforms (RHT). For each tensor split, compute the maximum
 *  absolute value (amax) and populate the row-wise amax of the
 *  corresponding output tensor. Also, compute the amax after a
 *  transposed RHT and populate the column-wise amax of the
 *  corresponding output tensor.
 *
 *  \param[in]      input               Input tensor.
 *  \param[in,out]  outputs             Array of NVFP4 output tensors. Only the row-wise and
 *                                      column-wise amaxes are updated.
 *  \param[in]      split_sections      Size of each tensor split along dimension 0.
 *  \param[in]      num_tensors         Number of tensor splits.
 *  \param[in]      random_sign_mask    16-bit sign mask for RHT.
 *  \param[in]      random_sign_mask_t  16-bit sign mask for transposed RHT.
 *  \param[in]      stream              CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_amax(const NVTETensor input, NVTETensor* outputs,
                                        const size_t* split_sections, size_t num_tensors,
                                        int random_sign_mask, int random_sign_mask_t,
                                        cudaStream_t stream);

/*! \brief Grouped-tensor amax with Hadamard transform (graph safe, device-managed grouping).
 *
 *  This function is experimental and the API is not stable.
 *
 *  This API assumes that the split info (grouping of tensors) is on device and unknown to the host;
 *  therefore, this is a graph safe API and the grouped-tensor argument is passed as a single device structure.
 *
 *  \param[in]      input               NVTEGroupedTensor representing grouped input tensors.
 *  \param[in,out]  output              NVTEGroupedTensor for output amax (row/col). Only the row-wise and
 *                                      column-wise amaxes are updated.
 *  \param[in]      random_sign_mask    16-bit sign mask for RHT.
 *  \param[in]      random_sign_mask_t  16-bit sign mask for transposed RHT.
 *  \param[in]      stream              CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_amax_graph_safe(const NVTEGroupedTensor input,
                                                   NVTEGroupedTensor output, int random_sign_mask,
                                                   int random_sign_mask_t, cudaStream_t stream);

/*!
 * \brief Perform the grouped-tensor columnwise Hadamard transform cast fusion operation.
 *
 *  This function is experimental and the API is not stable. Group_ prefix means contiguous input concatenated
 *
 *  \param[in]      input             Input tensor to apply Hadamard transform.
 *  \param[in,out]  outputs           Array of output tensors.
 *  \param[in]      hadamard_matrix   Hadamard matrix to use for transformation.
 *  \param[in]      split_sections    Array specifying splits in dimension 0 for each output tensor.
 *  \param[in]      num_tensors       Number of output tensors, must be > 0.
 *  \param[in]      quant_config      Quantization configuration.
 *  \param[in]      stream            CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_cast_fusion_columnwise(
    const NVTETensor input, NVTETensor* outputs, const NVTETensor hadamard_matrix,
    const size_t* split_sections, size_t num_tensors, const NVTEQuantizationConfig quant_config,
    cudaStream_t stream);

/*!
 * \brief Perform the grouped-tensor row quantize (without Hadamard) and columnwise Hadamard transform cast fusion operation.
 *
 *  This function is experimental and the API is not stable. Group_ prefix means contiguous input concatenated
 *
 *  \param[in]      input             Input tensor to apply Hadamard transform.
 *  \param[in,out]  outputs           Array of output tensors.
 *  \param[in]      hadamard_matrix   Hadamard matrix to use for transformation.
 *  \param[in]      split_sections    Array specifying splits in dimension 0 for each output tensor.
 *  \param[in]      num_tensors       Number of output tensors, must be > 0.
 *  \param[in]      quant_config      Quantization configuration.
 *  \param[in]      quant_workspace   Workspace buffer. Must be at least 4 bytes.
 *  \param[in]      stream            CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_cast_fusion(const NVTETensor input, NVTETensor* outputs,
                                               const NVTETensor hadamard_matrix,
                                               const size_t* split_sections, size_t num_tensors,
                                               const NVTEQuantizationConfig quant_config,
                                               NVTETensor quant_workspace, cudaStream_t stream);

/*!
 * \brief Perform the grouped-tensor Hadamard transform cast fusion operation in graph-safe mode.
 *
 *  This function is experimental and the API is not stable. Group_ prefix means contiguous input concatenated.
 *
 *  \param[in]      input             NVTEGroupedTensor representing grouped input tensors.
 *  \param[in,out]  output            NVTEGroupedTensor for output (row/column-wise quantized results).
 *  \param[in]      hadamard_matrix   Hadamard matrix to use for transformation.
 *  \param[in]      quant_config      Quantization configuration.
 *  \param[in]      quant_workspace   Workspace buffer. Must be at least 4 bytes.
 *  \param[in]      stream            CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_cast_fusion_graph_safe(
    const NVTEGroupedTensor input, NVTEGroupedTensor output, const NVTETensor hadamard_matrix,
    const NVTEQuantizationConfig quant_config, NVTETensor quant_workspace, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_
