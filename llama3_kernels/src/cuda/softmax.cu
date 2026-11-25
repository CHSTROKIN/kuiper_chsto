// =============================================================================
// cuda/softmax.cu  —  Row-wise softmax on the GPU, in place.
//
// Thread mapping: one block of 256 threads per row.
//
//    grid = rows blocks,  block = 256 threads
//
// Each block performs the same three steps as the CPU version, but every step
// is shared by the 256 threads (thread t handles columns t, t+256, ...):
//    1. max   : partial max per thread  -> block_reduce_max
//    2. exp   : x = exp(x - max), partial sum per thread -> block_reduce_sum
//    3. divide: x = x / sum
// =============================================================================
#include <cfloat>
#include <cstddef>

#include "block_reduce.cuh"
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

__global__ void softmax_kernel(float* x, int cols) {
  float* row = x + static_cast<size_t>(blockIdx.x) * cols;

  // 1. maximum of the row
  float local_max = -FLT_MAX;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    local_max = fmaxf(local_max, row[i]);
  }
  const float row_max = block_reduce_max(local_max);

  // 2. exponentials and their sum
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float e = expf(row[i] - row_max);
    row[i] = e;
    local_sum += e;
  }
  const float row_sum = block_reduce_sum(local_sum);

  // 3. normalize
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    row[i] /= row_sum;
  }
}

void softmax(float* x, int rows, int cols, cudaStream_t stream) {
  const int threads = 256;
  softmax_kernel<<<rows, threads, 0, stream>>>(x, cols);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace llama3
