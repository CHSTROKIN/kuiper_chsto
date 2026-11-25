// =============================================================================
// cuda/argmax.cu  —  Index of the largest logit, on the GPU.
//
// Thread mapping: ONE block of 1024 threads scans the whole vector.
//
//   1. thread t looks at x[t], x[t + 1024], x[t + 2048], ... and remembers the
//      best (value, index) it has seen.
//   2. a shared-memory tree reduction (same idea as block_reduce.cuh, but on
//      pairs) finds the best pair of the block. On equal values the smaller
//      index wins, so the answer is identical to the CPU version.
//   3. thread 0 stores the index in a __device__ global variable, and the host
//      function copies that single int back to the CPU.
//
// One block is plenty: the vocabulary is ~128k floats (0.5 MB), and the GPU
// time is negligible compared to the matmul that produced the logits.
// =============================================================================
#include <cfloat>

#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

constexpr int kArgmaxThreads = 1024;

// Lives in GPU memory; written by the kernel, read by the host with
// cudaMemcpyFromSymbol. (Simple, but means argmax calls must not overlap.)
__device__ int g_argmax_result;

// Is (value_a, index_a) a better answer than (value_b, index_b)?
__device__ inline bool is_better(float value_a, int index_a, float value_b, int index_b) {
  return value_a > value_b || (value_a == value_b && index_a < index_b);
}

__global__ void argmax_kernel(const float* x, int n) {
  __shared__ float best_values[kArgmaxThreads];
  __shared__ int best_indices[kArgmaxThreads];
  const int tid = threadIdx.x;

  // 1. each thread scans its own elements
  float best_value = -FLT_MAX;
  int best_index = n;  // "nothing found yet"; loses every comparison below
  for (int i = tid; i < n; i += blockDim.x) {
    if (is_better(x[i], i, best_value, best_index)) {
      best_value = x[i];
      best_index = i;
    }
  }
  best_values[tid] = best_value;
  best_indices[tid] = best_index;
  __syncthreads();

  // 2. tree reduction over (value, index) pairs
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      const int other = tid + stride;
      if (is_better(best_values[other], best_indices[other], best_values[tid],
                    best_indices[tid])) {
        best_values[tid] = best_values[other];
        best_indices[tid] = best_indices[other];
      }
    }
    __syncthreads();
  }

  // 3. publish
  if (tid == 0) {
    g_argmax_result = best_indices[0];
  }
}

int argmax(const float* x, int n, cudaStream_t stream) {
  argmax_kernel<<<1, kArgmaxThreads, 0, stream>>>(x, n);
  CUDA_CHECK(cudaGetLastError());

  int result = 0;
  CUDA_CHECK(cudaMemcpyFromSymbolAsync(&result, g_argmax_result, sizeof(int), 0,
                                       cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));  // wait until `result` is filled in
  return result;
}

}  // namespace cuda
}  // namespace llama3
