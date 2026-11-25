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
