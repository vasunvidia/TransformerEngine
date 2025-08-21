/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_rope.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"

namespace transformer_engine {

template <typename scalar_t>
__device__ void fused_rope_block_forward(const scalar_t *src, const float *freqs, scalar_t *dst,
                                         const bool interleaved, const int s_id,
                                         const int offset_block, const int offset_block_dst,
                                         const int h, const int d, const int d2, const int stride_h,
                                         const int stride_d, const int o_stride_h,
                                         const int o_stride_d) {
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos, v_sin;
    sincosf(freqs[s_id * d2 + d_id], &v_sin, &v_cos);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate;
      if (!interleaved) {
        v_src_rotate = (d_id + d2 / 2 < d2)
                           ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                           : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
                           // d_id + 1
                           ? -static_cast<float>(src[offset_src + stride_d])
                           // d_id - 1
                           : static_cast<float>(src[offset_src - stride_d]);
      }
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
//      if (s_id == 7 && threadIdx.y == 3 && h_id==7) {
//        printf ("fused_rope_block_forward  ID[%d,%d],[%d,%d] h_id: %d, d_id: %doffset_src: %d, offset_dst: %d, v_src: %f, v_cos: %f, v_sin: %f, v_src_rotate: %f dst_value: %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, h_id, d_id, offset_src, offset_dst, v_src, v_cos, v_sin, v_src_rotate, (float)dst[offset_dst]);
//      }
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_rope_block_backward(const scalar_t *src, const float *freqs, scalar_t *dst,
                                          const bool interleaved, const int s_id,
                                          const int offset_block, const int offset_block_dst,
                                          const int h, const int d, const int d2,
                                          const int stride_h, const int stride_d,
                                          const int o_stride_h, const int o_stride_d) {
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos = cosf(freqs[s_id * d2 + d_id]);
    float v_sin;
    if (!interleaved) {
      v_sin = (d_id + d2 / 2 < d2) ? sinf(freqs[s_id * d2 + d_id + d2 / 2])
                                   : -sinf(freqs[s_id * d2 + d_id + d2 / 2 - d2]);
    } else {
      v_sin =
          (d_id % 2 == 0) ? sinf(freqs[s_id * d2 + d_id + 1]) : -sinf(freqs[s_id * d2 + d_id - 1]);
    }
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate;
      if (!interleaved) {
        v_src_rotate = (d_id + d2 / 2 < d2)
                           ? static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                           : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
                           // d_id + 1
                           ? static_cast<float>(src[offset_src + stride_d])
                           // d_id - 1
                           : static_cast<float>(src[offset_src - stride_d]);
      }
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_qkv_rope_block_forward(const scalar_t *src, const float *freqs, scalar_t *out,
                                         const int s_id, const int offset_block, const int offset_block_dst,
                                         const int h, const int d, const int d2, const int row_offset, const int row_length) {
    // d_id: tid.x -> 128
    extern __shared__ float shared_mem_cos_sin[];
//    if (threadIdx.x == 0 && threadIdx.y == 7 && s_id == 7) {
//        printf ("fused_qkv_rope_block_forward row_offset: %d, row_length: %d, offset_block: %d, offset_block_dst: %d\n", row_offset, row_length, offset_block, offset_block_dst);
//    }
//    // Split the shared memory into cos and sin parts
//    float* shared_mem_cos = shared_mem_cos_sin;
//    float* shared_mem_sin = shared_mem_cos_sin + d2;
//    if (freqs != nullptr) {
//        int tid = threadIdx.x * blockDim.y + threadIdx.y;
//        for (int i = tid; i < d2; i+=blockDim.x * blockDim.y) {
//            sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
//        }
//    }
//    __syncthreads();

    #pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        #pragma unroll
        for (int i = 0; i < row_length; i+=d2) {
            #pragma unroll
            for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      
        // h_id: tid.y -> 8
                int offset_src = offset_block + h_id * d + (row_offset + i) + d_id;
                int offset_dst = offset_block_dst + h_id * row_length + i + d_id;


                if (freqs != nullptr) {
                    float v_cos, v_sin;
                    //v_cos = shared_mem_cos[d_id];
                    //v_sin = shared_mem_sin[d_id];
		    sincosf(freqs[s_id * d2 + d_id], &v_sin, &v_cos);
                    //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                    //    printf ("h_id: %d, i: %d, d_id: %d, offset_src: %d, offset_dst: %d, src: %f freq: %f sin: %f cos: %f\n", h_id, i, d_id, offset_src, offset_dst, (float)src[offset_src], freqs[s_id * d2 + d_id], v_sin, v_cos);
                    //}    
                    float v_src = src[offset_src];
                    float v_src_rotate = (d_id + d2 / 2 < d2)
                                 ? -static_cast<float>(src[offset_src + (d2 / 2)])
                                 : static_cast<float>(src[offset_src + (d2 / 2 - d2)]);
                    out[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
//                    if (s_id == 7 && threadIdx.y == 7) {
//                      printf ("fused_qkv_rope_block_forward SIZE [%d,%d] ID[%d,%d],[%d,%d] h_id: %d, i: %d, d_id: %d, offset_src: %d, offset_dst: %d, v_src: %f, v_cos: %f, v_sin: %f, v_src_rotate: %f dst_value: %f\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, h_id, i, d_id, offset_src, offset_dst, v_src, v_cos, v_sin, v_src_rotate, (float)out[offset_dst]);
//                    }
                } else {
                    out[offset_dst] = src[offset_src];
                }

            }
        }
    }
}

template <typename scalar_t>
__global__ void fused_rope_forward_kernel(const scalar_t *src, const int *cu_seqlens,
                                          const float *freqs, const int *start_positions,
                                          scalar_t *dst, const bool interleaved, const int cp_size,
                                          const int cp_rank, const int s, const int h, const int d,
                                          const int d2, const int stride_s_or_t, const int stride_b,
                                          const int stride_h, const int stride_d,
                                          const int o_stride_s_or_t, const int o_stride_b,
                                          const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst;
  int cur_seqlens;
  if (cu_seqlens != nullptr) {  // THD
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {  // SBHD/BSHD
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }

  int s_id_for_freqs;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs = s_id + cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs =
          cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 + s_id - cur_seqlens / 2;
    }
  } else {
    int begin_offset = (start_positions == nullptr) ? 0 : start_positions[b_id];
    s_id_for_freqs = s_id + begin_offset;
  }
//  if (s_id == 7 && b_id == 0 && threadIdx.x==0 && threadIdx.y==0) {
//    printf ("Size [%d,%d] ID[%d,%d] fused_rope_forward_kernel s_id: %d, b_id: %d, s_id_for_freqs: %d, d %d d2 %d\n", blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, s_id, b_id, s_id_for_freqs, d, d2);
//  }
  fused_rope_block_forward(src, freqs, dst, interleaved, s_id_for_freqs, offset_block,
                           offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(
    const scalar_t *src, const int *cu_seqlens, const float *freqs, scalar_t *dst,
    const bool interleaved, const int cp_size, const int cp_rank, const int s, const int h,
    const int d, const int d2, const int stride_s_or_t, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s_or_t, const int o_stride_b, const int o_stride_h,
    const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst;
  int cur_seqlens;
  if (cu_seqlens != nullptr) {  // THD
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {  // SBHD/BSHD
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }

  int s_id_for_freqs;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs = s_id + cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs =
          cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 + s_id - cur_seqlens / 2;
    }
  } else {
    s_id_for_freqs = s_id;
  }

  fused_rope_block_backward(src, freqs, dst, interleaved, s_id_for_freqs, offset_block,
                            offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_qkv_rope_forward_kernel(const scalar_t *qkv_input, const float *q_freqs, const float *k_freqs,
    scalar_t *q_out, scalar_t *k_out, scalar_t *v_out,
    const NVTE_QKV_Format qkv_format, const int s, const int b, const int h, const int d,
    const int d2, const int q_split_arg, const int k_split_arg, const int v_split_arg) {
    //if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //    printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
    //    printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
    //}
    int s_id = blockIdx.x, b_id = blockIdx.y;
    //if (s_id == 7 && b_id == 0) {
    //  printf ("Size [%d,%d] ID[%d,%d] fused_qkv_rope_forward_kernel s_id: %d, b_id: %d\n", blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, s_id, b_id);
    //}
    int offset_block, offset_block_dst_q, offset_block_dst_k, offset_block_dst_v;
    if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
        offset_block = s_id * b * h * d + b_id * h * d;
        offset_block_dst_q = s_id * b * h * q_split_arg + b_id * h * q_split_arg;
        offset_block_dst_k = s_id * b * h * k_split_arg + b_id * h * k_split_arg;
        offset_block_dst_v = s_id * b * h * v_split_arg + b_id * h * v_split_arg;
    } else {
        offset_block = b_id * s * h * d + s_id * h * d;
        offset_block_dst_q = b_id * s * h * q_split_arg + s_id * h * q_split_arg;
        offset_block_dst_k = b_id * s * h * k_split_arg + s_id * h * k_split_arg;
        offset_block_dst_v = b_id * s * h * v_split_arg + s_id * h * v_split_arg;
    }
    
    int q_limit = q_split_arg;
    int k_limit = q_limit + k_split_arg;
        //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x < 16 && blockIdx.y < 2) {
        //printf("[%d,%d] offset_block: %d, offset_block_dst_q: %d, offset_block_dst_k: %d, offset_block_dst_v: %d\n", blockIdx.x, blockIdx.y, offset_block, offset_block_dst_q, offset_block_dst_k, offset_block_dst_v);
    //}
    fused_qkv_rope_block_forward(
        qkv_input, q_freqs, q_out,
        s_id, offset_block, offset_block_dst_q,
        h, d, d2, 0, q_split_arg);
    fused_qkv_rope_block_forward(
        qkv_input, k_freqs, k_out,
        s_id, offset_block, offset_block_dst_k,
        h, d, d2, q_limit, k_split_arg);
    fused_qkv_rope_block_forward(
        qkv_input, nullptr, v_out,
        s_id, offset_block, offset_block_dst_v,
        h, d, d2, k_limit, v_split_arg);

}

template <typename scalar_t>
__device__ void fused_qkv_rope_block_backward(const scalar_t *grad_out, const float *freqs, scalar_t *out,
                                         const int s_id, const int offset_block, const int offset_block_dst,
                                         const int h, const int d, const int d2, const int row_offset, const int row_length) {
    extern __shared__ float shared_mem_cos_sin[];
    
    // Split the shared memory into cos and sin parts
    float* shared_mem_cos = shared_mem_cos_sin;
    float* shared_mem_sin = shared_mem_cos_sin + d2;
    if (freqs != nullptr) {
        int tid = threadIdx.x * blockDim.y + threadIdx.y;
        for (int i = tid; i < d2; i+=blockDim.x * blockDim.y) {
            sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        #pragma unroll
        for (int i = 0; i < row_length; i+=d2) {
            #pragma unroll
            for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      
                int offset_dst = offset_block + h_id * d + (row_offset + i) + d_id;
                int offset_src = offset_block_dst + h_id * row_length + i + d_id;
                //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 1) {
                //    printf("h_id: %d, i: %d, d_id: %d, offset_src: %d, offset_dst: %d, grad_out: %f, freqs: %f\n", h_id, i, d_id, offset_src, offset_dst, (float)grad_out[offset_src]);
                //}
                if (freqs != nullptr) {
                    float v_cos, v_sin;
                    v_cos = shared_mem_cos[d_id];
                    v_sin = shared_mem_sin[d_id];
                    v_sin = (d_id + d2 / 2 < d2) ? shared_mem_sin[d_id + d2 / 2] : -shared_mem_sin[d_id + d2 / 2 - d2];
                    //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                    //    printf ("h_id: %d, i: %d, d_id: %d, offset_src: %d, offset_dst: %d, src: %f freq: %f sin: %f cos: %f\n", h_id, i, d_id, offset_src, offset_dst, (float)src[offset_src], freqs[s_id * d2 + d_id], v_sin, v_cos);
                    //}    
                    float v_src = grad_out[offset_src];
                    float v_src_rotate = (d_id + d2 / 2 < d2)
                                 ? static_cast<float>(grad_out[offset_src + (d2 / 2)])
                                 : static_cast<float>(grad_out[offset_src + (d2 / 2 - d2)]);
                    out[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
                } else {
                    out[offset_dst] = grad_out[offset_src];
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void fused_qkv_rope_backward_kernel(const scalar_t *grad_out_q, const scalar_t *grad_out_k, const scalar_t *grad_out_v,
    const float *q_freqs, const float *k_freqs, scalar_t *qkv_grad,
    const NVTE_QKV_Format qkv_format, const int s, const int b, const int h, const int d,
    const int d2, const int q_split_arg, const int k_split_arg, const int v_split_arg) {

    int s_id = blockIdx.x, b_id = blockIdx.y;
    int offset_block, offset_block_dst_q, offset_block_dst_k, offset_block_dst_v;

    if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
        offset_block = s_id * b * h * d + b_id * h * d;
        offset_block_dst_q = s_id * b * h * q_split_arg + b_id * h * q_split_arg;
        offset_block_dst_k = s_id * b * h * k_split_arg + b_id * h * k_split_arg;
        offset_block_dst_v = s_id * b * h * v_split_arg + b_id * h * v_split_arg;
    } else {
        offset_block = b_id * s * h * d + s_id * h * d;
        offset_block_dst_q = b_id * s * h * q_split_arg + s_id * h * q_split_arg;
        offset_block_dst_k = b_id * s * h * k_split_arg + s_id * h * k_split_arg;
        offset_block_dst_v = b_id * s * h * v_split_arg + s_id * h * v_split_arg;
    }
    int q_limit = q_split_arg;
    int k_limit = q_limit + k_split_arg;
//    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//      printf("q_limit: %d, k_limit: %d, h: %d, d: %d, d2: %d, q_split_arg: %d, k_split_arg: %d, v_split_arg: %d\n", q_limit, k_limit, h, d, d2, q_split_arg, k_split_arg, v_split_arg);
//    }
    fused_qkv_rope_block_backward(
        grad_out_q, q_freqs, qkv_grad,
        s_id, offset_block, offset_block_dst_q,
        h, d, d2, 0, q_split_arg);
    fused_qkv_rope_block_backward(
        grad_out_k, k_freqs, qkv_grad,
        s_id, offset_block, offset_block_dst_k,
        h, d, d2, q_limit, k_split_arg);
    fused_qkv_rope_block_backward(
        grad_out_v, nullptr, qkv_grad,
        s_id, offset_block, offset_block_dst_v,
        h, d, d2, k_limit, v_split_arg);
        
}

template <typename scalar_t>
void fused_rope_forward_launcher(const scalar_t *input, const int *cu_seqlens, const float *freqs,
                                 const int *start_positions, scalar_t *output,
                                 const NVTE_QKV_Format qkv_format, const bool interleaved,
                                 const int cp_size, const int cp_rank, const int s, const int b,
                                 const int h, const int d, const int d2, const int stride_s_or_t,
                                 const int stride_b, const int stride_h, const int stride_d,
                                 cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  int o_stride_s_or_t, o_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    NVTE_CHECK(cu_seqlens != nullptr, "cu_seqlens is required for THD format");
    o_stride_s_or_t = h * d;
    o_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    o_stride_s_or_t = b * h * d;
    o_stride_b = h * d;
  } else {
    o_stride_s_or_t = h * d;
    o_stride_b = s * h * d;
  }
  const int o_stride_h = d;
  const int o_stride_d = 1;

  fused_rope_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, cu_seqlens, freqs, start_positions, output, interleaved, cp_size, cp_rank, s, h, d, d2,
      stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
      o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_backward_launcher(const scalar_t *output_grads, const int *cu_seqlens,
                                  const float *freqs, scalar_t *input_grads,
                                  const NVTE_QKV_Format qkv_format, const bool interleaved,
                                  const int cp_size, const int cp_rank, const int s, const int b,
                                  const int h, const int d, const int d2, const int stride_s_or_t,
                                  const int stride_b, const int stride_h, const int stride_d,
                                  cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  int o_stride_s_or_t, o_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    NVTE_CHECK(cu_seqlens != nullptr, "cu_seqlens is required for THD format");
    o_stride_s_or_t = h * d;
    o_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    o_stride_s_or_t = b * h * d;
    o_stride_b = h * d;
  } else {
    o_stride_s_or_t = h * d;
    o_stride_b = s * h * d;
  }
  const int o_stride_h = d;
  const int o_stride_d = 1;

  fused_rope_backward_kernel<<<blocks, threads, 0, stream>>>(
      output_grads, cu_seqlens, freqs, input_grads, interleaved, cp_size, cp_rank, s, h, d, d2,
      stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
      o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_qkv_rope_forward_launcher(const scalar_t *qkv_input, const float *q_freqs, const float *k_freqs,
                                 scalar_t *q_out, scalar_t *k_out, scalar_t *v_out,
                                 const NVTE_QKV_Format qkv_format, const int s, const int b, const int h, const int d,
                                 const int d2, const int qkv_split_arg_list_0, const int qkv_split_arg_list_1, const int qkv_split_arg_list_2,
                                 cudaStream_t stream) {
  const int shared_mem_size = 2 * d2 * sizeof(float);  // (cos, sin)
  const int THREADS_PER_WARP = 32;
  int warps_per_block = (h <= 8)? h:8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_qkv_rope_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      qkv_input, q_freqs, k_freqs, q_out, k_out, v_out, qkv_format, s, b, h, d, d2,
      qkv_split_arg_list_0, qkv_split_arg_list_1, qkv_split_arg_list_2);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_qkv_rope_backward_launcher(const scalar_t *q_grad_out, const scalar_t *k_grad_out, const scalar_t *v_grad_out,
    const float *q_freqs, const float *k_freqs, scalar_t *qkv_grad_input,
    const NVTE_QKV_Format qkv_format, const int s, const int b, const int h, const int d, const int d2,
    const int qkv_split_arg_list_0, const int qkv_split_arg_list_1, const int qkv_split_arg_list_2,
    cudaStream_t stream) {

  const int shared_mem_size = 2 * d2 * sizeof(float);  // (cos, sin)
  const int THREADS_PER_WARP = 32;
  const int warps_per_block = (h <= 8)? h:8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_qkv_rope_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      q_grad_out, k_grad_out, v_grad_out, q_freqs, k_freqs, qkv_grad_input, qkv_format, s, b, h, d, d2,
      qkv_split_arg_list_0, qkv_split_arg_list_1, qkv_split_arg_list_2);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_rope_forward(const Tensor &input, const Tensor &cu_seqlens, const Tensor &freqs,
                        const Tensor &start_positions, Tensor *output,
                        const NVTE_QKV_Format qkv_format, const bool interleaved, const int cp_size,
                        const int cp_rank, const int s, const int b, const int h, const int d,
                        const int d2, const int stride_s_or_t, const int stride_b,
                        const int stride_h, const int stride_d, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      fused_rope_forward_launcher(reinterpret_cast<const scalar_t *>(input.data.dptr),
                                  reinterpret_cast<const int *>(cu_seqlens.data.dptr),
                                  reinterpret_cast<const float *>(freqs.data.dptr),
                                  reinterpret_cast<const int *>(start_positions.data.dptr),
                                  reinterpret_cast<scalar_t *>(output->data.dptr), qkv_format,
                                  interleaved, cp_size, cp_rank, s, b, h, d, d2, stride_s_or_t,
                                  stride_b, stride_h, stride_d, stream););
}

void fused_rope_backward(const Tensor &output_grads, const Tensor &cu_seqlens, const Tensor &freqs,
                         Tensor *input_grads, const NVTE_QKV_Format qkv_format,
                         const bool interleaved, const int cp_size, const int cp_rank, const int s,
                         const int b, const int h, const int d, const int d2,
                         const int stride_s_or_t, const int stride_b, const int stride_h,
                         const int stride_d, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      output_grads.data.dtype, scalar_t,
      fused_rope_backward_launcher(reinterpret_cast<const scalar_t *>(output_grads.data.dptr),
                                   reinterpret_cast<const int *>(cu_seqlens.data.dptr),
                                   reinterpret_cast<const float *>(freqs.data.dptr),
                                   reinterpret_cast<scalar_t *>(input_grads->data.dptr), qkv_format,
                                   interleaved, cp_size, cp_rank, s, b, h, d, d2, stride_s_or_t,
                                   stride_b, stride_h, stride_d, stream););
}

void fused_qkv_rope_forward(const Tensor &qkv_input, const Tensor &q_freqs, const Tensor &k_freqs,
  Tensor *q_out, Tensor *k_out, Tensor *v_out,
  const NVTE_QKV_Format qkv_format, const int s, const int b, const int h, const int d,
  const int d2, const int qkv_split_arg_list_0, const int qkv_split_arg_list_1, const int qkv_split_arg_list_2,
  cudaStream_t stream) {
TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
qkv_input.data.dtype, scalar_t,
fused_qkv_rope_forward_launcher(reinterpret_cast<const scalar_t *>(qkv_input.data.dptr),
            reinterpret_cast<const float *>(q_freqs.data.dptr),
            reinterpret_cast<const float *>(k_freqs.data.dptr),
            reinterpret_cast<scalar_t *>(q_out->data.dptr),
            reinterpret_cast<scalar_t *>(k_out->data.dptr),
            reinterpret_cast<scalar_t *>(v_out->data.dptr), qkv_format,
            s, b, h, d, d2, qkv_split_arg_list_0, qkv_split_arg_list_1, qkv_split_arg_list_2,
            stream););
}

void fused_qkv_rope_backward(const Tensor &q_grad_out, const Tensor &k_grad_out, const Tensor &v_grad_out,
                             const Tensor &q_freqs, const Tensor &k_freqs, Tensor *qkv_grad_input,
                             const NVTE_QKV_Format qkv_format, const int s, const int b, const int h, const int d, const int d2,
                             const int qkv_split_arg_list_0, const int qkv_split_arg_list_1, const int qkv_split_arg_list_2,
                             cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      q_grad_out.data.dtype, scalar_t,
      fused_qkv_rope_backward_launcher(reinterpret_cast<const scalar_t *>(q_grad_out.data.dptr),
                                       reinterpret_cast<const scalar_t *>(k_grad_out.data.dptr),
                                       reinterpret_cast<const scalar_t *>(v_grad_out.data.dptr),
                                       reinterpret_cast<const float *>(q_freqs.data.dptr),
                                       reinterpret_cast<const float *>(k_freqs.data.dptr),
                                       reinterpret_cast<scalar_t *>(qkv_grad_input->data.dptr), qkv_format, s, b, h, d, d2,
                                       qkv_split_arg_list_0, qkv_split_arg_list_1, qkv_split_arg_list_2, stream););
}
}  // end namespace transformer_engine

void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor cu_seqlens,
                             const NVTETensor freqs, const NVTETensor start_positions,
                             NVTETensor output, const NVTE_QKV_Format qkv_format,
                             const bool interleaved, const int cp_size, const int cp_rank,
                             const int s, const int b, const int h, const int d, const int d2,
                             const int stride_s_or_t, const int stride_b, const int stride_h,
                             const int stride_d, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_forward);
  using namespace transformer_engine;
  fused_rope_forward(*convertNVTETensorCheck(input), *convertNVTETensorCheck(cu_seqlens),
                     *convertNVTETensorCheck(freqs), *convertNVTETensorCheck(start_positions),
                     convertNVTETensorCheck(output), qkv_format, interleaved, cp_size, cp_rank, s,
                     b, h, d, d2, stride_s_or_t, stride_b, stride_h, stride_d, stream);
}

void nvte_fused_rope_backward(const NVTETensor output_grads, const NVTETensor cu_seqlens,
                              const NVTETensor freqs, NVTETensor input_grads,
                              const NVTE_QKV_Format qkv_format, const bool interleaved,
                              const int cp_size, const int cp_rank, const int s, const int b,
                              const int h, const int d, const int d2, const int stride_s_or_t,
                              const int stride_b, const int stride_h, const int stride_d,
                              cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_backward);
  using namespace transformer_engine;
  fused_rope_backward(*convertNVTETensorCheck(output_grads), *convertNVTETensorCheck(cu_seqlens),
                      *convertNVTETensorCheck(freqs), convertNVTETensorCheck(input_grads),
                      qkv_format, interleaved, cp_size, cp_rank, s, b, h, d, d2, stride_s_or_t,
                      stride_b, stride_h, stride_d, stream);
}

void nvte_fused_qkv_rope_forward(const NVTETensor qkv_input, const NVTETensor q_freqs,
                                 const NVTETensor k_freqs, NVTETensor q_out, NVTETensor k_out, NVTETensor v_out, 
                                 const NVTE_QKV_Format qkv_format, const int s, const int b, const int h, const int d, const int d2,
                                 const int qkv_split_arg_list_0, const int qkv_split_arg_list_1, const int qkv_split_arg_list_2,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_qkv_rope_forward);
  using namespace transformer_engine;
  fused_qkv_rope_forward(*convertNVTETensorCheck(qkv_input), *convertNVTETensorCheck(q_freqs),
                         *convertNVTETensorCheck(k_freqs), convertNVTETensorCheck(q_out),
                         convertNVTETensorCheck(k_out), convertNVTETensorCheck(v_out), qkv_format, s, b, h, d, d2,
                         qkv_split_arg_list_0, qkv_split_arg_list_1, qkv_split_arg_list_2, stream);
}

void nvte_fused_qkv_rope_backward(const NVTETensor q_grad_out, const NVTETensor k_grad_out, const NVTETensor v_grad_out,
                                 const NVTETensor q_freqs, const NVTETensor k_freqs, NVTETensor qkv_grad_input,
                                 const NVTE_QKV_Format qkv_format, const int s, const int b, const int h, const int d, const int d2,
                                 const int qkv_split_arg_list_0, const int qkv_split_arg_list_1, const int qkv_split_arg_list_2,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_qkv_rope_backward);
  using namespace transformer_engine;
  fused_qkv_rope_backward(*convertNVTETensorCheck(q_grad_out), *convertNVTETensorCheck(k_grad_out), *convertNVTETensorCheck(v_grad_out),
                         *convertNVTETensorCheck(q_freqs), *convertNVTETensorCheck(k_freqs), convertNVTETensorCheck(qkv_grad_input), qkv_format, s, b, h, d, d2,
                         qkv_split_arg_list_0, qkv_split_arg_list_1, qkv_split_arg_list_2, stream);
}
