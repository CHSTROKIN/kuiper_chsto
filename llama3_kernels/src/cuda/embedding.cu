// =============================================================================
// cuda/embedding.cu  —  Token id -> vector, on the GPU.
//
// Thread mapping: one thread copies one float.
//
//    grid  = ceil(dim / 256) blocks,  block = 256 threads
//    thread i  :  out[i] = table[token * dim + i]
//
// Neighbouring threads read neighbouring addresses, so the GPU can serve a
// whole warp (32 threads) with a few wide memory transactions ("coalesced").
// =============================================================================
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

__global__ void embedding_kernel(const float* table, int token, int dim, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    out[i] = table[static_cast<size_t>(token) * dim + i];
  }
}

void embedding(const float* table, int token, int vocab_size, int dim, float* out,
               cudaStream_t stream) {
  // The token id lives on the CPU, so we can validate it before launching.
  if (token < 0 || token >= vocab_size) {
    std::fprintf(stderr, "cuda::embedding: token id %d is outside the vocabulary\n", token);
    std::exit(EXIT_FAILURE);
  }
  const int threads = 256;
  const int blocks = ceil_div(dim, threads);
  embedding_kernel<<<blocks, threads, 0, stream>>>(table, token, dim, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace llama3
