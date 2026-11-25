// =============================================================================
// cuda/block_reduce.cuh  —  "Sum / max over all threads of a block".
//
// Several kernels (rmsnorm, matmul, softmax) need many threads to cooperate on
// ONE number: each thread computes a partial result, then the partial results
// are combined. This file does the combining with the classic shared-memory
// tree reduction:
//
//   threads:    0   1   2   3   4   5   6   7      (blockDim.x = 8)
//   start:     a0  a1  a2  a3  a4  a5  a6  a7
//   stride 4:  a0+a4  a1+a5  a2+a6  a3+a7          (threads 0..3 work)
//   stride 2:  (a0+a4)+(a2+a6)  (a1+a5)+(a3+a7)    (threads 0..1 work)
//   stride 1:  total                               (thread 0 works)
//
// __syncthreads() between the steps makes every thread wait until the previous
// step is finished, so nobody reads a value that is still being written.
//
// Rules for callers:
//   * EVERY thread of the block must call the function (it contains
//     __syncthreads; if some threads skip it the kernel hangs).
//   * blockDim.x must be a power of two and at most kMaxBlockSize.
//   * Every thread gets the final result back.
//
// (Faster versions use warp shuffles, e.g. __shfl_down_sync or cub::BlockReduce.
//  The shared-memory version is used here because it is the easiest to follow.)
// =============================================================================
#pragma once

#include <cfloat>

namespace llama3 {
namespace cuda {

constexpr int kMaxBlockSize = 1024;

__device__ inline float block_reduce_sum(float value) {
  __shared__ float partial[kMaxBlockSize];
  const int tid = threadIdx.x;

  partial[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  const float total = partial[0];
  __syncthreads();  // everyone has read partial[0] before it can be reused
  return total;
}

__device__ inline float block_reduce_max(float value) {
  __shared__ float partial[kMaxBlockSize];
  const int tid = threadIdx.x;

  partial[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
    }
    __syncthreads();
  }

  const float result = partial[0];
  __syncthreads();
  return result;
}

}  // namespace cuda
}  // namespace llama3
