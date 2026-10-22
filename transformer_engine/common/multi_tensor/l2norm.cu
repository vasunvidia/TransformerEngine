/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_fp8.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include <limits>
#include <vector>

#include "../utils.cuh"
#include "multi_tensor_apply.cuh"

namespace transformer_engine {
namespace multi_tensor_l2norm {

#define BLOCK_SIZE 512
#define ILP 4

template <typename T>
__device__ __forceinline__ bool is_aligned(T *p) {
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(T *dst, T *src, int dst_offset, int src_offset) {
  typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT *)dst)[dst_offset] = ((LT *)src)[src_offset];  // NOLINT(*)
}

template <typename T>
__device__ __forceinline__ T
reduce_block_into_lanes(T *x, T val, int lanes = 1,
                        bool share_result = false) {  // lanes is intended to be <= 32.
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = x[tid] + x[tid + 32];
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1) final = final + __shfl_down_sync(0xffffffff, final, i);
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
  }
  __syncthreads();
  // Avoid potential write before read race when reduce_block_into_lanes is called back to back

  return final;
}

template <typename T>
__device__ __forceinline__ T
reduce_block_into_lanes_max_op(T *x, T val, int lanes = 1,
                               bool share_result = false) {  // lanes is intended to be <= 32.
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = fmaxf(fabsf(x[tid]), fabsf(x[tid + i]));
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = fmaxf(fabsf(x[tid]), fabsf(x[tid + 32]));
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1)
      final = fmaxf(fabsf(final), fabsf(__shfl_down_sync(0xffffffff, final, i)));
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}

template <typename x_t>
struct L2NormFunctor {
  __device__ __forceinline__ void operator()(int64_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<1> &tl,  // NOLINT(*),
                                             float *output, float *output_per_tensor,
                                             bool per_tensor, int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int64_t n = tl.sizes[tensor_loc];

    x_t *x = reinterpret_cast<x_t *>(tl.addresses[0][tensor_loc]);
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0.f;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] += next * next;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]);
            vals[ii] += next * next;
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val += vals[i];

    float final = reduce_block_into_lanes(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final))
        *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] += final;
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
                          chunk_idx] = final;
    }
  }
};

template <typename x_t>
struct UnscaleL2NormFunctor {
  __device__ __forceinline__ void operator()(int64_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<1> &tl,  // NOLINT(*),
                                             const float *inv_scale, float *output,
                                             float *output_per_tensor, bool per_tensor,
                                             int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int64_t n = tl.sizes[tensor_loc];

    x_t *x = reinterpret_cast<x_t *>(tl.addresses[0][tensor_loc]);
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0.f;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]) * (*inv_scale);
          vals[ii] += next * next;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]) * (*inv_scale);
            vals[ii] += next * next;
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val += vals[i];

    float final = reduce_block_into_lanes(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final))
        *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] += final;
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
                          chunk_idx] = final;
    }
  }
};

// Probably better to template, but since we are not likely to support other norm
template <typename x_t>
struct MaxNormFunctor {
  __device__ __forceinline__ void operator()(int64_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<1> &tl,  // NOLINT(*),
                                             float *output, float *output_per_tensor,
                                             bool per_tensor, int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int64_t n = tl.sizes[tensor_loc];

    x_t *x = reinterpret_cast<x_t *>(tl.addresses[0][tensor_loc]);
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] = fmaxf(fabsf(vals[ii]), fabsf(next));
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]);
            vals[ii] = fmaxf(fabsf(vals[ii]), fabsf(next));
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val = fmaxf(fabsf(val), fabsf(vals[i]));

    float final = reduce_block_into_lanes_max_op(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final))
        *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] = fmaxf(fabsf(output[blockIdx.x]), fabsf(final));
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
                          chunk_idx] = final;
    }
  }
};

__global__ void cleanup(float *output, float *output_per_tensor, float *ret, float *ret_per_tensor,
                        bool per_tensor, int max_chunks_per_tensor) {
  __shared__ float vals[512];

  if (blockIdx.x == 0) {
    float val = 0;
    if (threadIdx.x < 320) val = output[threadIdx.x];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) *ret = sqrt(final);
  }

  if (per_tensor) {
    float *output_this_tensor = output_per_tensor + blockIdx.x * max_chunks_per_tensor;

    float val = 0;
    for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
      val += output_this_tensor[i];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) ret_per_tensor[blockIdx.x] = sqrt(final);
  }
}

__global__ void cleanup_v2(float *output, float *output_per_tensor, float *ret,
                           float *ret_per_tensor, bool per_tensor, int max_chunks_per_tensor,
                           int norm_type, float alpha, float beta) {
  __shared__ float vals[512];

  if (blockIdx.x == 0) {
    float val = 0;
    if (threadIdx.x < 320) val = output[threadIdx.x];

    if (norm_type == 0) {
      float final = reduce_block_into_lanes_max_op(vals, val);
      if (threadIdx.x == 0) *ret = alpha * (*ret) + beta * final;
    } else {
      float final = reduce_block_into_lanes(vals, val);
      if (threadIdx.x == 0) *ret = sqrt(alpha * (*ret) * (*ret) + beta * final);
    }
  }

  if (per_tensor) {
    float *output_this_tensor = output_per_tensor + blockIdx.x * max_chunks_per_tensor;

    if (norm_type == 0) {
      float val = 0;
      for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
        val = fmaxf(fabsf(val), fabsf(output_this_tensor[i]));

      float final = reduce_block_into_lanes_max_op(vals, val);

      if (threadIdx.x == 0)
        ret_per_tensor[blockIdx.x] = alpha * ret_per_tensor[blockIdx.x] + beta * final;
    } else {
      float val = 0;
      for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
        val += output_this_tensor[i];

      float final = reduce_block_into_lanes(vals, val);

      if (threadIdx.x == 0)
        ret_per_tensor[blockIdx.x] =
            sqrt(alpha * ret_per_tensor[blockIdx.x] * ret_per_tensor[blockIdx.x] + beta * final);
    }
  }
}

/* Single-launch L2 norm, bitwise equal to multi_tensor_l2norm_cuda.
 *
 * multi_tensor_l2norm_cuda runs L2NormFunctor through multi_tensor_apply<1>: one block per (tensor,
 * chunk), at most 320 blocks and 110 tensors per launch. Block k of launch l adds its block sum into
 * output[k] (so output[k] sums the launches in launch order), and `cleanup` reduces output[0..319].
 * With 320 blocks of 512 threads per launch, a launch fills only part of the GPU, and the launches
 * run one after the other.
 *
 * The single-launch path runs the blocks of all those launches in one grid (one launch per
 * kSingleLaunchMaxTensors tensors). Every block does L2NormFunctor's arithmetic: same thread ->
 * element assignment, same per-thread accumulation order (it only issues the loads of U loop
 * iterations before accumulating them, in order), same block reduction. Each block stores its sum at
 * output[l * 320 + k], where (l, k) is the launch and block slot multi_tensor_apply<1> would have given
 * it. `cleanup_single_launch` then adds the 320 slots over l in launch order, starting from 0 like the
 * zero-initialized output of the multi-launch path, and runs the same final reduction. The partial
 * sums and every addition are the same, so the result is the same bits; the per-tensor norms and the
 * noop flag are written as by the multi-launch path.
 */
constexpr int kSingleLaunchMaxTensors = 128;  // tensors per launch (kernel parameter space)
constexpr int kSingleLaunchUnroll = 4;        // loop iterations whose loads are issued together
constexpr int kMultiTensorApplyMaxBlocks = depth_to_max_blocks[0];    // 320 (multi_tensor_apply<1>)
constexpr int kMultiTensorApplyMaxTensors = depth_to_max_tensors[0];  // 110 (multi_tensor_apply<1>)

struct L2NormSingleLaunchArgs {
  void *addresses[kSingleLaunchMaxTensors];
  int64_t sizes[kSingleLaunchMaxTensors];
  // Index of the tensor's chunk 0 in the [launches of multi_tensor_apply<1>, 320] block-sum array
  // (launch * 320 + block slot). Chunks of one tensor occupy consecutive entries.
  int64_t output_slot[kSingleLaunchMaxTensors];
  // First block of each tensor in this launch (prefix sums of the chunk counts); num_tensors + 1.
  int block_start[kSingleLaunchMaxTensors + 1];
  int64_t first_tensor;  // index of this launch's tensor 0 in the whole list
  int num_tensors;
};
static_assert(sizeof(L2NormSingleLaunchArgs) <= 4000, "L2NormSingleLaunchArgs exceeds 4 KB");

template <typename x_t, int U>
__global__ void __launch_bounds__(BLOCK_SIZE)
    l2norm_single_launch_kernel(int64_t chunk_size, volatile int *noop_gmem,
                                L2NormSingleLaunchArgs args, float *output,
                                float *output_per_tensor, bool per_tensor,
                                int max_chunks_per_tensor) {
  // The tensor of this block: the last t with block_start[t] <= blockIdx.x.
  const int block = blockIdx.x;
  int lo = 0;
  int hi = args.num_tensors - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (args.block_start[mid] <= block) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int tensor_loc = lo;
  const int chunk_idx = block - args.block_start[tensor_loc];
  int64_t n = args.sizes[tensor_loc];

  x_t *x = reinterpret_cast<x_t *>(args.addresses[tensor_loc]);
  x += chunk_idx * chunk_size;

  n -= chunk_idx * chunk_size;

  __shared__ float s_vals[512];

  float vals[ILP];
  x_t r_x[ILP];
  for (int i = 0; i < ILP; i++) {
    vals[i] = 0.f;
    r_x[i] = 0.f;
  }

  // Same iterations and accumulation order as L2NormFunctor.
  if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
    int i_start = threadIdx.x;
    for (; (i_start + (U - 1) * static_cast<int>(blockDim.x)) * ILP < n &&
           (i_start + (U - 1) * static_cast<int>(blockDim.x)) * ILP < chunk_size;
         i_start += U * blockDim.x) {
      x_t r_u[U][ILP];
#pragma unroll
      for (int u = 0; u < U; u++) {
        load_store(r_u[u], x, 0, i_start + u * blockDim.x);
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_u[u][ii]);
          vals[ii] += next * next;
        }
      }
    }
    for (; i_start * ILP < n && i_start * ILP < chunk_size; i_start += blockDim.x) {
      load_store(r_x, x, 0, i_start);
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        float next = static_cast<float>(r_x[ii]);
        vals[ii] += next * next;
      }
    }
  } else {
    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          float next = static_cast<float>(x[i]);
          vals[ii] += next * next;
        }
      }
    }
  }

  float val = 0.f;
  for (int i = 0; i < ILP; i++) val += vals[i];

  float final = reduce_block_into_lanes(s_vals, val);

  if (threadIdx.x == 0) {
    if (!isfinite(final)) *noop_gmem = 1;  // as L2NormFunctor
    output[args.output_slot[tensor_loc] + chunk_idx] = final;
    if (per_tensor)
      output_per_tensor[(args.first_tensor + tensor_loc) * max_chunks_per_tensor + chunk_idx] =
          final;
  }
}

// `cleanup` for the [num_launches, 320] block sums of l2norm_single_launch_kernel.
__global__ void cleanup_single_launch(const float *output, int64_t num_launches,
                                      float *output_per_tensor, float *ret, float *ret_per_tensor,
                                      bool per_tensor, int max_chunks_per_tensor) {
  __shared__ float vals[512];

  if (blockIdx.x == 0) {
    float val = 0;
    if (threadIdx.x < kMultiTensorApplyMaxBlocks) {
      // output[k] of the multi-launch path: zero-initialized, then `+= block sum` per launch.
      float slot = 0.f;
      for (int64_t l = 0; l < num_launches; l++) {
        slot += output[l * kMultiTensorApplyMaxBlocks + threadIdx.x];
      }
      val = slot;
    }

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) *ret = sqrt(final);
  }

  if (per_tensor) {
    float *output_this_tensor = output_per_tensor + blockIdx.x * max_chunks_per_tensor;

    float val = 0;
    for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
      val += output_this_tensor[i];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) ret_per_tensor[blockIdx.x] = sqrt(final);
  }
}

/* Block layout of multi_tensor_apply<1> (multi_tensor_apply.cuh) for these tensors: returns the
 * number of launches it makes and, if first_slot is given, the launch * 320 + block slot of every
 * tensor's chunk 0. Chunk c of tensor t has slot first_slot[t] + c: inside a tensor the blocks are
 * consecutive, and a launch that fills its 320 blocks mid-tensor continues the tensor at slot 0 of
 * the next launch.
 */
int64_t multi_tensor_apply_1_layout(int64_t chunk_size, const std::vector<Tensor *> &tensors,
                                    int64_t *first_slot) {
  NVTE_CHECK(chunk_size > 0, "chunk_size must be positive.");
  int64_t launch = 0;
  int64_t loc_block = 0;
  int64_t loc_tensor = 0;
  const size_t num_tensors = tensors.size();
  for (size_t t = 0; t < num_tensors; t++) {
    const int64_t chunks =
        (static_cast<int64_t>(tensors[t]->numel()) + chunk_size - 1) / chunk_size;
    NVTE_CHECK(chunks > 0,
               "multi_tensor_l2norm_single_launch expects tensors with at least one chunk; "
               "zero-sized tensors must be filtered before launch.");
    if (first_slot != nullptr) first_slot[t] = launch * kMultiTensorApplyMaxBlocks + loc_block;
    loc_tensor++;
    // Chunks 0 .. chunks - 2: a full launch (320 blocks) splits the tensor; the next launch starts
    // with this tensor alone.
    if (chunks > 1) {
      const int64_t m = loc_block + chunks - 1;
      const int64_t wraps = m / kMultiTensorApplyMaxBlocks;
      if (wraps > 0) {
        launch += wraps;
        loc_block = m % kMultiTensorApplyMaxBlocks;
        loc_tensor = 1;
      } else {
        loc_block = m;
      }
    }
    // Last chunk: the launch ends when its tensors or blocks are full, or at the last tensor.
    loc_block++;
    if (loc_tensor == kMultiTensorApplyMaxTensors || loc_block == kMultiTensorApplyMaxBlocks ||
        t + 1 == num_tensors) {
      launch++;
      loc_block = 0;
      loc_tensor = 0;
    }
  }
  return launch;
}

// Main kernels of the single-launch path: one launch per kSingleLaunchMaxTensors tensors, one block per
// (tensor, chunk).
template <typename x_t>
void launch_l2norm_single_launch(int64_t chunk_size, int *noop_flag,
                                 const std::vector<Tensor *> &tensors,
                                 const std::vector<int64_t> &first_slot, float *output,
                                 float *output_per_tensor, bool per_tensor,
                                 int max_chunks_per_tensor, cudaStream_t stream) {
  const size_t num_tensors = tensors.size();
  L2NormSingleLaunchArgs args{};
  size_t t = 0;
  while (t < num_tensors) {
    args.first_tensor = static_cast<int64_t>(t);
    args.num_tensors = 0;
    int64_t blocks = 0;
    while (t < num_tensors && args.num_tensors < kSingleLaunchMaxTensors) {
      const int64_t chunks =
          (static_cast<int64_t>(tensors[t]->numel()) + chunk_size - 1) / chunk_size;
      if (args.num_tensors > 0 && blocks + chunks > std::numeric_limits<int>::max()) break;
      NVTE_CHECK(blocks + chunks <= std::numeric_limits<int>::max(),
                 "Tensor with too many chunks for multi_tensor_l2norm_single_launch.");
      args.addresses[args.num_tensors] = tensors[t]->data.dptr;
      args.sizes[args.num_tensors] = static_cast<int64_t>(tensors[t]->numel());
      args.output_slot[args.num_tensors] = first_slot[t];
      args.block_start[args.num_tensors] = static_cast<int>(blocks);
      blocks += chunks;
      args.num_tensors++;
      t++;
    }
    args.block_start[args.num_tensors] = static_cast<int>(blocks);
    l2norm_single_launch_kernel<x_t, kSingleLaunchUnroll>
        <<<static_cast<unsigned int>(blocks), BLOCK_SIZE, 0, stream>>>(
            chunk_size, noop_flag, args, output, output_per_tensor, per_tensor,
            max_chunks_per_tensor);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

size_t multi_tensor_l2norm_single_launch_output_size(
    int chunk_size, const std::vector<std::vector<Tensor *>> &tensor_lists) {
  NVTE_CHECK(tensor_lists.size() == 1, "multi_tensor_l2norm_single_launch expects 1 tensor list.");
  return static_cast<size_t>(multi_tensor_apply_1_layout(chunk_size, tensor_lists[0], nullptr)) *
         kMultiTensorApplyMaxBlocks;
}

void multi_tensor_l2norm_single_launch_cuda(int chunk_size, Tensor noop_flag,
                                            std::vector<std::vector<Tensor *>> tensor_lists,
                                            Tensor output, Tensor output_per_tensor, Tensor ret,
                                            Tensor ret_per_tensor, bool per_tensor,
                                            int max_chunks_per_tensor, cudaStream_t stream) {
  NVTE_CHECK(tensor_lists.size() == 1, "multi_tensor_l2norm_single_launch expects 1 tensor list.");
  const std::vector<Tensor *> &tensors = tensor_lists[0];
  const size_t num_tensors = tensors.size();
  NVTE_CHECK(num_tensors > 0, "multi_tensor_l2norm_single_launch expects at least one tensor.");
  const DType dtype = tensors[0]->dtype();
  for (size_t t = 1; t < num_tensors; t++) {
    NVTE_CHECK(tensors[t]->dtype() == dtype,
               "multi_tensor_l2norm_single_launch expects tensors of one dtype.");
  }

  std::vector<int64_t> first_slot(num_tensors);
  const int64_t num_launches = multi_tensor_apply_1_layout(chunk_size, tensors, first_slot.data());
  NVTE_CHECK(output.dtype() == DType::kFloat32, "output must be FP32.");
  NVTE_CHECK(output.numel() >= static_cast<size_t>(num_launches) * kMultiTensorApplyMaxBlocks,
             "output holds ", output.numel(), " elements, needs ",
             num_launches * kMultiTensorApplyMaxBlocks,
             " (multi_tensor_l2norm_single_launch_output_size).");
  float *output_ptr = reinterpret_cast<float *>(output.data.dptr);
  float *output_per_tensor_ptr =
      per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      dtype, x_t,
      launch_l2norm_single_launch<x_t>(chunk_size, reinterpret_cast<int *>(noop_flag.data.dptr),
                                       tensors, first_slot, output_ptr, output_per_tensor_ptr,
                                       per_tensor, max_chunks_per_tensor, stream););

  cleanup_single_launch<<<per_tensor ? num_tensors : 1, 512, 0, stream>>>(
      output_ptr, num_launches, output_per_tensor_ptr, reinterpret_cast<float *>(ret.data.dptr),
      per_tensor ? reinterpret_cast<float *>(ret_per_tensor.data.dptr) : nullptr, per_tensor,
      max_chunks_per_tensor);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_l2norm_cuda(int chunk_size, Tensor noop_flag,
                              std::vector<std::vector<Tensor *>> tensor_lists, Tensor output,
                              Tensor output_per_tensor, Tensor ret, Tensor ret_per_tensor,
                              bool per_tensor, int max_chunks_per_tensor, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      multi_tensor_apply<1>(
          BLOCK_SIZE, chunk_size, noop_flag, tensor_lists, L2NormFunctor<dtype>(), stream,
          reinterpret_cast<float *>(output.data.dptr),
          per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr, per_tensor,
          max_chunks_per_tensor);)

  NVTE_CHECK_CUDA(cudaGetLastError());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  cleanup<<<per_tensor ? tensor_lists[0].size() : 1, 512, 0, stream>>>(
      reinterpret_cast<float *>(output.data.dptr),
      per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr,
      reinterpret_cast<float *>(ret.data.dptr),
      per_tensor ? reinterpret_cast<float *>(ret_per_tensor.data.dptr) : nullptr, per_tensor,
      max_chunks_per_tensor);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_unscale_l2norm_cuda(int chunk_size, Tensor noop_flag,
                                      std::vector<std::vector<Tensor *>> tensor_lists,
                                      Tensor output, Tensor output_per_tensor, Tensor ret,
                                      Tensor ret_per_tensor, Tensor inv_scale, bool per_tensor,
                                      int max_chunks_per_tensor, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      multi_tensor_apply<1>(
          BLOCK_SIZE, chunk_size, noop_flag, tensor_lists, UnscaleL2NormFunctor<dtype>(), stream,
          reinterpret_cast<float *>(inv_scale.data.dptr),
          reinterpret_cast<float *>(output.data.dptr),
          per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr, per_tensor,
          max_chunks_per_tensor);)

  NVTE_CHECK_CUDA(cudaGetLastError());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  cleanup<<<per_tensor ? tensor_lists[0].size() : 1, 512, 0, stream>>>(
      reinterpret_cast<float *>(output.data.dptr),
      per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr,
      reinterpret_cast<float *>(ret.data.dptr),
      per_tensor ? reinterpret_cast<float *>(ret_per_tensor.data.dptr) : nullptr, per_tensor,
      max_chunks_per_tensor);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_l2norm
}  // namespace transformer_engine

void nvte_multi_tensor_l2norm_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                   const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                   NVTETensor output, NVTETensor output_per_tensor, NVTETensor ret,
                                   NVTETensor ret_per_tensor, int per_tensor,
                                   int max_chunks_per_tensor, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_l2norm_cuda);
  using namespace transformer_engine;

  multi_tensor_l2norm::multi_tensor_l2norm_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(output), *convertNVTETensorCheck(output_per_tensor),
      *convertNVTETensorCheck(ret), *convertNVTETensorCheck(ret_per_tensor), per_tensor,
      max_chunks_per_tensor, stream);
}

void nvte_multi_tensor_unscale_l2norm_cuda(int chunk_size, NVTETensor noop_flag,
                                           NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                           const size_t num_tensors_per_list, NVTETensor output,
                                           NVTETensor output_per_tensor, NVTETensor ret,
                                           NVTETensor ret_per_tensor, NVTETensor inv_scale,
                                           int per_tensor, int max_chunks_per_tensor,
                                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_unscale_l2norm_cuda);
  using namespace transformer_engine;

  multi_tensor_l2norm::multi_tensor_unscale_l2norm_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(output), *convertNVTETensorCheck(output_per_tensor),
      *convertNVTETensorCheck(ret), *convertNVTETensorCheck(ret_per_tensor),
      *convertNVTETensorCheck(inv_scale), per_tensor, max_chunks_per_tensor, stream);
}

size_t nvte_multi_tensor_l2norm_single_launch_output_size(int chunk_size, NVTETensor **tensor_lists,
                                                          const size_t num_tensor_lists,
                                                          const size_t num_tensors_per_list) {
  NVTE_API_CALL(nvte_multi_tensor_l2norm_single_launch_output_size);
  using namespace transformer_engine;

  return multi_tensor_l2norm::multi_tensor_l2norm_single_launch_output_size(
      chunk_size, convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list));
}

void nvte_multi_tensor_l2norm_single_launch_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor output, NVTETensor output_per_tensor,
    NVTETensor ret, NVTETensor ret_per_tensor, int per_tensor, int max_chunks_per_tensor,
    cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_l2norm_single_launch_cuda);
  using namespace transformer_engine;

  multi_tensor_l2norm::multi_tensor_l2norm_single_launch_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(output), *convertNVTETensorCheck(output_per_tensor),
      *convertNVTETensorCheck(ret), *convertNVTETensorCheck(ret_per_tensor), per_tensor,
      max_chunks_per_tensor, stream);
}
