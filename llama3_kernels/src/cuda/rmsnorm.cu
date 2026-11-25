// =============================================================================
// cuda/rmsnorm.cu  —  RMSNorm on the GPU.
//
//    out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
//
// Thread mapping: ONE block of 256 threads handles the whole vector, because
// every output needs the same number (the rms of the whole vector).
//
//   step 1  each thread sums the squares of the elements it owns:
//              thread t owns x[t], x[t + 256], x[t + 512], ...
//   step 2  block_reduce_sum adds the 256 partial sums -> sum of squares
//   step 3  each thread writes its own elements, scaled by 1/rms
//
// A vector of a few thousand floats is tiny for a GPU, so this kernel is
// dominated by launch overhead; that is expected.
// =============================================================================
#include "block_reduce.cuh"
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

__global__ void rmsnorm_kernel(const float* x, const float* weight, int dim, float eps,
                               float* out) {
  // Step 1: partial sum of squares (a "block-stride loop").
  float sum_of_squares = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    sum_of_squares += x[i] * x[i];
  }

  // Step 2: combine the partial sums of all threads.
  sum_of_squares = block_reduce_sum(sum_of_squares);
  const float inv_rms = rsqrtf(sum_of_squares / dim + eps);  // rsqrtf(v) = 1 / sqrt(v)

  // Step 3: normalize. Each thread touches the same elements as in step 1,
  // so out == x is safe.
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    out[i] = x[i] * inv_rms * weight[i];
  }
}

void rmsnorm(const float* x, const float* weight, int dim, float eps, float* out,
             cudaStream_t stream) {
  const int threads = 256;
  rmsnorm_kernel<<<1, threads, 0, stream>>>(x, weight, dim, eps, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace llama3
