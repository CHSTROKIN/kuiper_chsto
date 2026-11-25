// =============================================================================
// cuda/swiglu.cu  —  out[i] = silu(gate[i]) * up[i]  on the GPU.
//
// Thread mapping: one thread per element (the simplest possible kernel shape).
//    grid = ceil(n / 256), block = 256,   i = blockIdx.x * 256 + threadIdx.x
// =============================================================================
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

__global__ void swiglu_kernel(const float* gate, const float* up, int n, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = gate[i];
    const float silu = g / (1.0f + expf(-g));
    out[i] = silu * up[i];
  }
}

void swiglu(const float* gate, const float* up, int n, float* out, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = ceil_div(n, threads);
  swiglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, n, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace llama3
