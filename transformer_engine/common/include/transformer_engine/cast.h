/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast.h
 *  \brief Functions to cast to/from FP8/MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_H_
#define TRANSFORMER_ENGINE_CAST_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Quantize the tensor
 *
 *  The type of quantized tensor in the output depends on the scaling mode of the output
 *  tensor.
 *
 *  Supported formats are:
 *
 *  1) MXFP8 scaling (for compute capability 10.0 or newer)
 *
 *  The MXFP8 implementation is per the microscaling format MXFP8 defined by the OCP specification:
 *  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *
 *
 *  Supported modes of MXFP8 scaling (live scaling) for scaling mode NVTE_MXFP8_1D_SCALING
 *      a) Rowwise scaling (along the dim=0) computes one set of the output data, which includes:
 *          - the scaled output tensor
 *          - the corresponding scaling factors
 *         The scaling factors are computed for blocks of the shape [1,32]
 *         (i.e., each scaling factor spans 32 contiguous elements along rows).
 *
 *      b) Columwise scaling (along the dim=1) computes one set of the output data.
 *         The scaling factors are computed for blocks of the shape [32,1]
 *         (i.e., each scaling factor spans 32 contiguous elements along columns).
 *
 *      c) Both rowwise AND columnwise scaling (along the dim=0 and the dim=1)
 *         computes two sets of the output data: both 1) and 2).
 *
 *  The shape of the MX block must be specified in the 'output' argument,
 *  and can be either [1,32] or [32,1] as no other shapes are currently supported.
 *
 *  To cast the input tensor to the MXFP8, the scaling_mode.delayed_scaling parameter
 *  of the output tensor should be set to 0.
 *
 *  2) NVTE_DELAYED_TENSOR_SCALING that quantize the entire tensor
 *  using a single scaling factor. The absolute maximum value of the tensor should
 *  be precalculated either online (current scaling) or based on a tensor history
 *  (delayed scaling). The calls to nvte_quantize scale based on that data value.
 *  Note the NVTE_DELAYED_TENSOR_SCALING NVTEScalingMode is reused for online
 *  per tensor scaling.
 *
 *
 *  3) FP8 block scaling formats NVTE_BLOCK_SCALING_1D and NVTE_BLOCK_SCALING_2D
 *  for compute capability of at least 9.0. These modes quantize the tensor by blocks
 *  of size 1x128 (with columnwise mode of 128x1) and 128x128 respectively.
 *
 *  The supported modes are:
 *      a) Rowwise scaling yields output data:
 *          - the scaled output tensor in fp8 coefficients with identical shape to the
 *            input tensor.
 *          - Scale factors which are computed for either 1D 1x128 or 2D 128x128 blocks.
 *      b) Columnwise scaling yields output data:
 *          - the scaled output tensor in fp8 coefficients with a shape equivalent to
 *            the transpose of the input tensor.
 *          - Scale factors which are calculated for either 1D 128x1 or 2D 128x128 blocks
 *            of the input tensor.
 *      c) Both: In which both tensors and both scales are calculated.
 *
 *  This quantization mode includes both the calculation of the scaling factors
 *  per-tile and quantization of the row and/or columnwise tiles. No precalculated
 *  absolute max is required. The scaling factors are also rounded to powers of 2.
 */

/*! \brief Casts input tensor to FP8/MXFP8/BlockwiseFP8.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8/BlockwiseFP8 tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Casts input grouped tensor.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *         NVFP4 output requires row scaling, BF16 input, and dimensions that are multiples of 128.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in,out] output           Output grouped tensor.
 *  \param[in]     quant_config     Quantization configuration.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Casts input tensor to FP8/MXFP8/BlockwiseFP8, providing the option to immediately exit the kernel
 *         based on the value of the 'noop' tensor.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output quantized tensor.
 *  \param[out]     noop             Noop tensor.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream);

/*! \brief Casts input tensor to quantized output tensor, with advanced quantization options.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output quantized tensor.
 *  \param[in]      quant_config     Quantization configuration.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Computes the per-tensor amax of a BF16 tensor and casts it to NVFP4 with 2D
 *         (16x16) block scaling.
 *
 *  The result is bitwise identical to nvte_compute_amax_with_config followed by
 *  nvte_quantize_v2 with 2D quantization: rowwise data, optional columnwise data, compact or
 *  GEMM-swizzled scales, and the rowwise and (if allocated) columnwise amax. It uses two
 *  persistent kernels and is faster for large tensors such as weights. Use
 *  nvte_nvfp4_2d_amax_quantize_supported to check that an input/output combination is
 *  supported.
 *
 *  Calling this function with the workspace being an empty tensor will not perform the
 *  operation, but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input BF16 tensor to be cast.
 *  \param[in,out] output           Output NVFP4 tensor.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     quant_config     Quantization configuration. Only the noop tensor is used;
 *                                  stochastic rounding and 4over6 are not supported.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_nvfp4_2d_amax_quantize(const NVTETensor input, NVTETensor output, NVTETensor workspace,
                                 const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Whether nvte_nvfp4_2d_amax_quantize supports the given input and output tensors.
 *
 *  Requires compute capability 10.x, a BF16 input whose flattened dimensions are multiples of
 *  128, rowwise output data, E4M3 scales with the default scale bound, and a per-tensor amax.
 *
 *  \param[in]     input            Input tensor.
 *  \param[in]     output           Output NVFP4 tensor.
 *
 *  \return 1 if supported, 0 otherwise.
 */
int nvte_nvfp4_2d_amax_quantize_supported(const NVTETensor input, const NVTETensor output);

/*! \brief Casts a BF16 tensor to NVFP4 with 2D (16x16) block scaling, given its per-tensor amax.
 *
 *  As nvte_nvfp4_2d_amax_quantize, but the per-tensor amax max|input| is read from \p amax
 *  (one FP32 value on the device, e.g. computed where the tensor was last written) instead of
 *  being computed: only the quantize kernel runs. The result is bitwise identical to
 *  nvte_nvfp4_2d_amax_quantize when \p amax holds max|input|. Supports the same input/output
 *  combinations (nvte_nvfp4_2d_amax_quantize_supported).
 *
 *  \param[in]     input            Input BF16 tensor to be cast.
 *  \param[in]     amax             FP32 tensor with one element: max|input|.
 *  \param[in,out] output           Output NVFP4 tensor.
 *  \param[in]     quant_config     Quantization configuration. Only the noop tensor is used.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_nvfp4_2d_quantize_with_amax(const NVTETensor input, const NVTETensor amax,
                                      NVTETensor output, const NVTEQuantizationConfig quant_config,
                                      cudaStream_t stream);

/*! \brief One 16-row group of a weight cut by a data-parallel shard boundary (sharded NVFP4 2D
 *         quantization, see nvte_nvfp4_2d_shard_pack).
 */
struct NVTENVFP4ShardPartialEntry {
  const void *row_group; /*!< BF16 data of the first element of the row group */
  int64_t own_begin;     /*!< First element of the row group this rank owns (relative) */
  int64_t own_end;       /*!< One past the last owned element (relative) */
  int32_t cols;          /*!< Number of columns of the weight */
  int32_t slot;          /*!< First of cols / 16 int32 block-amax slots */
};

/*! \brief The part of one weight that one data-parallel rank quantizes into its chunk of a
 *         gathered buffer (sharded NVFP4 2D quantization, see nvte_nvfp4_2d_shard_pack).
 */
struct NVTENVFP4ShardPackItem {
  const void *weight;      /*!< BF16 weight [rows, cols], row-major */
  void *chunk;             /*!< This rank's chunk of the gathered buffer */
  int64_t own_begin;       /*!< First weight element this rank owns */
  int64_t own_end;         /*!< One past the last owned element */
  int64_t data_offset;     /*!< Weight element + data_offset = element index in the rank's shard */
  int64_t scale_offset;    /*!< Chunk byte offset of the scale of (row_group_own_begin, block 0) */
  int64_t first_unit;      /*!< Index of the item's first 16x128 unit among all items */
  int32_t cols;            /*!< Number of columns of the weight (multiple of 128) */
  int32_t row_group_first; /*!< First 16-row group that intersects the owned elements */
  int32_t row_group_own_begin; /*!< First row group whose first element this rank owns */
  int32_t row_group_own_end;   /*!< One past the last such row group */
  int32_t amax_slot;           /*!< int32 slot holding the per-tensor amax (FP32 bit pattern) */
  int32_t cut_begin_slot;      /*!< Block-amax slots of a row group cut at own_begin, or -1 */
  int32_t cut_end_slot;        /*!< Block-amax slots of a row group cut at own_end, or -1 */
  int32_t row_group_last;      /*!< Last 16-row group that intersects the owned elements */
};

/*! \brief Partial 16x16 block amaxes of the row groups a shard boundary cuts.
 *
 *  For each entry, writes into amax[slot + j], j < cols / 16, the FP32 bit pattern (as int32) of
 *  max|x| over the elements of block j of the row group that this rank owns (0 if none). A MAX
 *  all-reduce of \p amax over the data-parallel group then yields the full block amaxes.
 *
 *  \param[in]     entries          Device array of NVTENVFP4ShardPartialEntry.
 *  \param[in]     num_entries      Number of entries.
 *  \param[in,out] amax             Device int32 array of slots.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_nvfp4_2d_shard_partial_amax(const NVTENVFP4ShardPartialEntry *entries, size_t num_entries,
                                      int32_t *amax, cudaStream_t stream);

/*! \brief Quantizes the elements of a data-parallel shard to NVFP4 with 2D block scaling.
 *
 *  Sharded NVFP4 2D quantization lets each data-parallel rank quantize only the weight elements
 *  it owns (e.g. the shard a distributed optimizer updates) and all-gather NVFP4 data and scales
 *  instead of BF16 weights. Every item writes, into its rank's chunk, the FP4 data of its owned
 *  elements (at (element + data_offset) / 2) and the E4M3 scales of the row groups whose first
 *  element it owns. The per-tensor amax and the block amaxes of row groups cut by a shard
 *  boundary are read from \p amax (FP32 bit patterns), which must hold their values over the
 *  whole weight. The quantized values are bitwise identical to nvte_nvfp4_2d_amax_quantize of
 *  the whole weight. nvte_nvfp4_2d_shard_unpack expands the gathered chunks.
 *
 *  \param[in]     items            Device array of NVTENVFP4ShardPackItem, by first_unit.
 *  \param[in]     num_items        Number of items.
 *  \param[in]     num_units        Total number of 16x128 units of all items.
 *  \param[in]     amax             Device int32 array of amax slots.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_nvfp4_2d_shard_pack(const NVTENVFP4ShardPackItem *items, size_t num_items,
                              int64_t num_units, const int32_t *amax, cudaStream_t stream);

/*! \brief Expands one weight of a gathered buffer of sharded NVFP4 2D chunks into an NVFP4 tensor.
 *
 *  Rank k's chunk starts at k * chunk_bytes. Element e of the weight (flat index param_start + e
 *  within the gathered bucket) belongs to rank k = (param_start + e) / shard, its FP4 nibble is at
 *  byte (param_start + e - k * shard) / 2 of the chunk, and the scale of row group g, block j at
 *  byte scale_base[k] + g * (cols / 16) + j, where k owns the first element of row group g. Writes
 *  the rowwise data and scales, and if allocated the columnwise data and scales (compact or
 *  GEMM-swizzled), and the amaxes from \p amax. The output equals nvte_nvfp4_2d_amax_quantize of
 *  the full weight.
 *
 *  \param[in]     gathered         Gathered chunks (uint8).
 *  \param[in]     chunk_bytes      Bytes per rank chunk.
 *  \param[in]     shard            Elements per rank shard.
 *  \param[in]     param_start      Weight's first element within the gathered bucket.
 *  \param[in]     scale_base       Host array of num_ranks scale offsets.
 *  \param[in]     num_ranks        Number of data-parallel ranks (at most 64).
 *  \param[in]     amax             Device int32 slot holding the per-tensor amax (FP32 bits).
 *  \param[in]     rows             Rows of the weight (multiple of 128).
 *  \param[in]     cols             Columns of the weight (multiple of 128).
 *  \param[in,out] output           Output NVFP4 tensor of shape [rows, cols].
 *  \param[in]     quant_config     Quantization configuration. Only the noop tensor is used.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_nvfp4_2d_shard_unpack(const uint8_t *gathered, int64_t chunk_bytes, int64_t shard,
                                int64_t param_start, const int64_t *scale_base, size_t num_ranks,
                                const int32_t *amax, size_t rows, size_t cols, NVTETensor output,
                                const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Casts input tensor to MXFP8. Additionally, reduces the input along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                         NVTETensor workspace, cudaStream_t stream);

/*! \brief Casts input grouped tensor to MXFP8. Additionally, reduces the input along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     quant_config     Quantization configuration.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                               NVTEGroupedTensor dbias, NVTETensor workspace,
                               const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Computes backward of GeLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the GeLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dgelu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of GeLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the GeLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_dgelu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                     NVTEGroupedTensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Computes backward of SiLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the SiLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of SiLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the SiLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_dsilu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                     NVTEGroupedTensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Computes backward of ReLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the ReLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_drelu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of ReLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the ReLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_drelu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                     NVTEGroupedTensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Computes backward of Quick GeLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Quick GeLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dqgelu(const NVTETensor input, const NVTETensor act_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream);

/*! \brief Computes backward of Quick GeLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Quick GeLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_dqgelu(const NVTEGroupedTensor input,
                                      const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                      NVTEGroupedTensor dbias, NVTETensor workspace,
                                      cudaStream_t stream);

/*! \brief Computes backward of Squared ReLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Squared ReLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dsrelu(const NVTETensor input, const NVTETensor act_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream);

/*! \brief Computes backward of Squared ReLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Squared ReLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_dsrelu(const NVTEGroupedTensor input,
                                      const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                      NVTEGroupedTensor dbias, NVTETensor workspace,
                                      cudaStream_t stream);

/*! \brief Casts input tensor from reduced to higher precision.
 *         In case of the MXFP8 dequantization, the dequantized values are stored to the rowwise
 *         data of the output tensor, regardless of whether the row- or columnwise scaling is used.
 *
 *  \param[in]     input     Input FP8/MXFP8 tensor to be cast.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Casts input grouped tensor from reduced to higher precision.
 *         In case of the MXFP8 dequantization, the dequantized values are stored to the rowwise
 *         data of the output tensor, regardless of whether the row- or columnwise scaling is used.
 *
 *  \param[in]     input     Input grouped FP8/MXFP8 tensor to be cast.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_dequantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                           cudaStream_t stream);

/*! \brief Requantizes a grouped tensor. Currently supports MXFP8 1D scaling only.
 *
 * Equivalent to grouped MXFP8 dequantization followed by columnwise quantization,
 * using BF16 intermediates, with optional BF16 dequantized output.
 *
 * Input and output must describe the same grouped shapes. Input must contain
 * rowwise FP8 data with compact E8M0 scales. Output must contain columnwise E4M3
 * data with compact or GEMM-swizzled E8M0 scales. Optional output rowwise data
 * must alias the input. Optional output rowwise scales must alias the input
 * scales for matching layouts, or use separate storage for different layouts.
 * Each group's row and column counts must be divisible by 128.
 *
 *  \param[in]     input          Compact rowwise MXFP8 grouped tensor.
 *  \param[in,out] output         Columnwise MXFP8 E4M3 grouped tensor with
 *                                optional rowwise data and scales.
 *  \param[out]    dequantized    Optional BF16 tensor with the input logical shape;
 *                                pass NULL or an unallocated tensor to skip it.
 *  \param[in]     quant_config   Unused; may be NULL.
 *  \param[in]     stream         CUDA stream used for the operation.
 */
void nvte_group_requantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                           NVTETensor dequantized, const NVTEQuantizationConfig quant_config,
                           cudaStream_t stream);

/*! \brief Casts multiple input tensors to quantized output tensors.
 *
 *  \param[in]      inputs          List of input tensors to be cast.
 *  \param[in,out]  outputs         List of output quantized tensors.
 *  \param[in]      quant_config    (Optional) Quantization configurations.
 *  \param[in]      num_tensors     Number of input and output tensors.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_multi_tensor_quantize(const NVTETensor *inputs, NVTETensor *outputs,
                                const NVTEQuantizationConfig quant_config, const size_t num_tensors,
                                cudaStream_t stream);

/*! \brief Casts grouped input tensor to quantized output tensors.
 *
 *  \param[in]      input           Input tensor to be cast.
 *  \param[in,out]  outputs         Output quantized tensors.
 *  \param[in]      split_sections  Split sections of the input tensor.
 *  \param[in]      num_tensors     Number of output tensors.
 *  \param[in]      quant_config    (Optional) Quantization configurations.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_group_nvfp4_quantize_with_amax(const NVTETensor input, NVTETensor *outputs,
                                         const size_t *split_sections, size_t num_tensors,
                                         const NVTEQuantizationConfig quant_config,
                                         cudaStream_t stream);

/*! \deprecated Use nvte_group_amax instead. */
void nvte_group_nvfp4_compute_amax(const NVTETensor input, NVTETensor *outputs,
                                   const size_t *split_sections, size_t num_tensors,
                                   cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
