/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/logging.h>
#include <transformer_engine/gemm.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include "../common.h"
#include <iostream>
#include <string>

namespace {

cudaDataType_t get_cuda_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return CUDA_R_16F;
    case DType::kFloat32:
      return CUDA_R_32F;
    case DType::kBFloat16:
      return CUDA_R_16BF;
    case DType::kFloat8E4M3:
      return CUDA_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return CUDA_R_8F_E5M2;
    default:
      NVTE_ERROR("Invalid type");
  }
}

}  // namespace

namespace transformer_engine {

/* CAUTION : must match cublasLtMatmulTile_t */
const char * const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8"   ,
    "8x32"   ,
    "16x16"  ,
    "32x8"   ,
    "8x64"   ,
    "16x32"  ,
    "32x16"  ,
    "64x8"   ,
    "32x32"  ,
    "32x64"  ,
    "64x32"  ,
    "32x128" ,
    "64x64"  ,
    "128x32" ,
    "64x128" ,
    "128x64" ,
    "64x256" ,
    "128x128",
    "256x64" ,
    "64x512" ,
    "128x256",
    "256x128",
    "512x64" ,
    "64x96"  ,
    "96x64"  ,
    "96x64"  ,
    "128x160",
    "160x128",
    "192x128",
    "128x192",
    "128x96" 
};

const char * const matmulClusterShape[] = {
    "AUTO", "na", "1x1x1", "2x1x1", "4x1x1",
    "1x2x1", "2x2x1", "4x2x1", "1x4x1", "2x4x1",
    "4x4x1", "8x1x1", "1x8x1", "8x2x1", "2x8x1",
    "16x1x1", "1x16x1", "3x1x1", "5x1x1", "6x1x1",
    "7x1x1", "9x1x1", "10x1x1", "11x1x1", "12x1x1",
    "13x1x1", "14x1x1", "15x1x1", "3x2x1", "5x2x1",
    "6x2x1", "7x2x1", "1x3x1", "2x3x1", "3x3x1",
    "4x3x1", "5x3x1", "3x4x1", "1x5x1", "2x5x1",
    "3x5x1", "1x6x1", "2x6x1", "1x7x1", "2x7x1",
    "1x9x1", "1x10x1", "1x11x1", "1x12x1", "1x13x1",
    "1x14x1", "1x15x1"
};

// Utility function to print customMatmulPerf_t structure
static bool printPerfStructure(const std::string &name, const cublasLtMatmulAlgo_t &algo, bool findLargeTileCga2, int m, int n, int k) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;
    uint16_t cga;

    const cublasLtMatmulAlgo_t *matmulAlgo = &algo;
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cga, sizeof(cga), NULL);

    if (findLargeTileCga2) {
        if (tile>=20 && tile<=24 && (cga==3 || cga==5/* || (m==1024 && n==1024 && k==1024)*/)) {
              printf("T %s [%dx%dx%d]: algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d stages=%d cluster_shape=%u (%s)}\n",
                name.c_str(), m, n, k, algoId, tile, matmulTileName[tile],
                numSplitsK, reductionScheme,
                swizzle, customOption, stages, cga, matmulClusterShape[cga]);
              return true;
        }
              printf(" %s -----x----- [%dx%dx%d]: algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d stages=%d cluster_shape=%u (%s)}\n",
                name.c_str(), m, n, k, algoId, tile, matmulTileName[tile],
                numSplitsK, reductionScheme,
                swizzle, customOption, stages, cga, matmulClusterShape[cga]);
        return false;
    }
    printf("F %s: [%dx%dx%d]: algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d stages=%d cluster_shape=%u (%s)}\n",
        name.c_str(), m, n, k, algoId, tile, matmulTileName[tile],
        numSplitsK, reductionScheme,
        swizzle, customOption, stages, cga, matmulClusterShape[cga]);
    return true;
}

void cublas_gemm(const Tensor *inputA,
                 const Tensor *inputB,
                 Tensor *outputD,
                 const Tensor *inputBias,
                 Tensor *outputPreGelu,
                 int m, int n, int k,
                 int lda, int ldb, int ldd,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 bool grad,
                 void* workspace,
                 size_t workspaceSize,
                 bool accumulate,
                 bool use_split_accumulator,
                 int math_sm_count,
                 cudaStream_t stream,
                 const std::string &name
) {
  int num_heuristics = (math_sm_count > 0) ? 10 : 1;
  void *A = inputA->data.dptr;
  void *A_scale_inverse = inputA->scale_inv.dptr;
  void *B = inputB->data.dptr;
  void *B_scale_inverse = inputB->scale_inv.dptr;
  void *C = outputD->data.dptr;
  void *D = outputD->data.dptr;
  void *D_scale = outputD->scale.dptr;
  void *D_amax = outputD->amax.dptr;
  void *bias_ptr = inputBias->data.dptr;
  const bool bias = bias_ptr != nullptr;
  void *pre_gelu_out = outputPreGelu->data.dptr;
  const bool gelu = pre_gelu_out != nullptr;
  const bool use_fp8 = is_fp8_dtype(inputA->data.dtype) ||
                       is_fp8_dtype(inputB->data.dtype);
  const cudaDataType_t A_type = get_cuda_dtype(inputA->data.dtype);
  const cudaDataType_t B_type = get_cuda_dtype(inputB->data.dtype);
  const cudaDataType_t D_type = get_cuda_dtype(outputD->data.dtype);
  const cudaDataType_t bias_type = get_cuda_dtype(inputBias->data.dtype);

  NVTE_CHECK(!is_fp8_dtype(inputA->data.dtype) || A_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");
  NVTE_CHECK(!is_fp8_dtype(inputB->data.dtype) || B_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");

  // check consistency of arguments:
  // if fp8 is desired, context cannot be null
  // fp8 + gelu fusion + fp8 aux is unavailable right now.
  if (use_fp8 && gelu) {
    NVTE_CHECK(!is_fp8_dtype(outputPreGelu->data.dtype),
             "fp8 Aux output for gemm + gelu fusion not supported!");
  }
  if (is_fp8_dtype(outputD->data.dtype)) {
    NVTE_CHECK(!accumulate,
             "Accumulation mode not supported with FP8 GEMM output!");
  }

  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  cublasLtHandle_t handle;
  NVTE_CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t       operationDesc = nullptr;
  cublasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int                             returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult[10];
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t) ldd;

  // default to tf32 except for e5m2 inputs where the config is not supported
  cublasComputeType_t gemm_compute_type = (A_type == CUDA_R_8F_E5M2 || B_type == CUDA_R_8F_E5M2)
                                          ? CUBLAS_COMPUTE_32F
                                          : CUBLAS_COMPUTE_32F_FAST_TF32;

  // Create matrix descriptors. Not setting any extra attributes.
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type,
                                               transa == CUBLAS_OP_N ? m : k,
                                               transa == CUBLAS_OP_N ? k : m,
                                               lda));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type,
                                               transb == CUBLAS_OP_N ? k : n,
                                               transb == CUBLAS_OP_N ? n : k,
                                               ldb));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                   &transa, sizeof(transa)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                   &transb, sizeof(transb)));


  // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
  // Note: gelu fusion isn't available right now, and we don't need
  // amax(D) either (next op is high precision).
  if (use_fp8) {
    // Split accumulator.
    const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                     &fastAccuMode,
                                                     sizeof(fastAccuMode)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                     &A_scale_inverse,
                                                     sizeof(A_scale_inverse)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                     &B_scale_inverse,
                                                     sizeof(B_scale_inverse)));
    if (is_fp8_dtype(outputD->data.dtype)) {
      // Accumulation mode not supported for FP8 output
      C = nullptr;
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                       CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                       &D_scale,
                                                       sizeof(D_scale)));
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                       CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                                       &D_amax,
                                                       sizeof(D_amax)));
      // For FP8 output, cuBLAS requires C_type to be same as bias_type
      NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, bias_type, m, n, ldd));
    } else {
      NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
    }
    if (bias) {
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                       CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                       &bias_type, sizeof(bias_type)));
    }
  } else {
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
  }

  if (bias && gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                     &bias_ptr, sizeof(bias_ptr)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
            &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                     &ld_gelumat, sizeof(ld_gelumat)));
    const cudaDataType_t aux_type = get_cuda_dtype(outputPreGelu->data.dtype);
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                                                     &aux_type, sizeof(aux_type)));
  } else if (bias) {
    if (grad) {
      // grad output is always input B
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
    } else {
      epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                     &bias_ptr, sizeof(bias_ptr)));
  } else if (gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
            &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                     &ld_gelumat, sizeof(ld_gelumat)));
  }

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                   CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue)));

  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspaceSize, sizeof(workspaceSize)));

  // Set math SM count
  if (math_sm_count != 0) {
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
          &math_sm_count, sizeof(math_sm_count)));
  }

  NVTE_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc,
                                                   Ddesc, preference, num_heuristics, heuristicResult,
                                                   &returnedResults));
  if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");
  int i=0;
  printf ("returnedResults %d\n", returnedResults);
  for (; i<returnedResults; i++) {
    if (printPerfStructure(name, heuristicResult[i].algo, true, m, n, k))
      break;
  }
  if (i==returnedResults) {
    i=0;
    printPerfStructure(name, heuristicResult[i].algo, false, m, n, k);
  }

  // D = alpha * (A * B) + beta * C

  NVTE_CHECK_CUBLAS(cublasLtMatmul(handle,
                                   operationDesc,
                                   static_cast<const void*>(&one),         /* alpha */
                                   A,                                      /* A */
                                   Adesc,
                                   B,                                      /* B */
                                   Bdesc,
                                   static_cast<const void*>(&beta),        /* beta */
                                   C,                                      /* C */
                                   Cdesc,
                                   D,                                      /* D */
                                   Ddesc,
                                   &(heuristicResult[i].algo),                  /* algo */
                                   workspace,                              /* workspace */
                                   workspaceSize,
                                   stream));                               /* stream */


  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
}

}  // namespace transformer_engine

void nvte_cublas_gemm(const NVTETensor A,
                      const NVTETensor B,
                      NVTETensor D,
                      const NVTETensor bias,
                      NVTETensor pre_gelu_out,
                      bool transa,
                      bool transb,
                      bool grad,
                      NVTETensor workspace,
                      bool accumulate,
                      bool use_split_accumulator,
                      int math_sm_count,
                      cudaStream_t stream,
                      const std::string &name) {
  NVTE_API_CALL(nvte_cublas_gemm);
  using namespace transformer_engine;
  const Tensor *inputA = reinterpret_cast<const Tensor*>(A);
  const Tensor *inputB = reinterpret_cast<const Tensor*>(B);
  Tensor *outputD = reinterpret_cast<Tensor*>(D);
  const Tensor *biasTensor = reinterpret_cast<const Tensor*>(bias);
  Tensor *outputGelu = reinterpret_cast<Tensor*>(pre_gelu_out);
  Tensor *wspace = reinterpret_cast<Tensor*>(workspace);

  const int m = transa ? inputA->data.shape[0] : inputA->data.shape[1];
  const int k = transa ? inputA->data.shape[1] : inputA->data.shape[0];
  const int n = transb ? inputB->data.shape[1] : inputB->data.shape[0];
  int lda, ldb, ldd;
  if (transa && !transb) {  // TN
    lda = k;
    ldb = k;
    ldd = m;
  } else if (!transa && !transb) {  // NN
    lda = m;
    ldb = k;
    ldd = m;
  } else if (!transa && transb) {  // NT
    lda = m;
    ldb = n;
    ldd = m;
  } else {  // TT
    NVTE_ERROR("TT layout not allowed.");
  }

  cublas_gemm(inputA,
              inputB,
              outputD,
              biasTensor,
              outputGelu,
              m, n, k,
              lda, ldb, ldd,
              (transa) ? CUBLAS_OP_T : CUBLAS_OP_N,
              (transb) ? CUBLAS_OP_T : CUBLAS_OP_N,
              grad, wspace->data.dptr,
              wspace->data.shape[0],
              accumulate, use_split_accumulator,
              math_sm_count,
              stream, name);
}
