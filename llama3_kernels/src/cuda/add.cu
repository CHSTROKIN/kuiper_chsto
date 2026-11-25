// =============================================================================
// cuda/add.cu  —  Residual connection out[i] = a[i] + b[i]  on the GPU.
//
// The "hello world" of CUDA: one thread per element.
//
//    block 0: threads 0..255  -> elements   0..255
//    block 1: threads 0..255  -> elements 256..511
//    ...
//    The last block may have threads past the end of the array, hence `if (i < n)`.
// =============================================================================
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

__global__ void add_kernel(const float* a, const float* b, int n, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

void add(const float* a, const float* b, int n, float* out, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = ceil_div(n, threads);
  add_kernel<<<blocks, threads, 0, stream>>>(a, b, n, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace llama3
