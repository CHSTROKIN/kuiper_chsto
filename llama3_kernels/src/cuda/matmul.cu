// =============================================================================
// cuda/matmul.cu  —  Linear layer on the GPU:  out[m, n] = sum_k x[m, k] * w[n, k]
//
// Thread mapping: ONE block per weight row n (= per output feature).
//
//    grid  = N blocks, block = 256 threads
//    block n computes out[0, n], out[1, n], ..., out[M-1, n]
//
// Inside a block, for every input row m:
//
//    w row n :  [ k0 k1 k2 ... k255 | k256 ... k511 | ... ]
//    thread  :     0  1  2 ...  255 |    0 ...  255 | ...
//
//   1. thread t multiplies the elements k = t, t+256, t+512, ... and keeps a
//      partial sum. Neighbouring threads read neighbouring floats, so the
//      memory accesses are coalesced.
//   2. block_reduce_sum adds the 256 partial sums.
//   3. thread 0 writes the result.
//
// Why this is fast enough for text generation: with M = 1 the GPU must read
// every weight once per token. The GPU reads memory ~5-10x faster than a CPU,
// which is where the speed-up comes from. The arithmetic itself is trivial.
//
// What a production kernel would add: several rows per block, float4 (16-byte)
// loads, warp-shuffle reductions, tiling through shared memory / tensor cores
// for M > 1 (prompt processing), or simply calling cuBLAS.
// =============================================================================
#include <cstddef>

#include "block_reduce.cuh"
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

__global__ void matmul_kernel(const float* x, const float* w, int M, int N, int K, float* out) {
  const int n = blockIdx.x;  // which weight row / output feature this block owns
  const float* w_row = w + static_cast<size_t>(n) * K;

  for (int m = 0; m < M; ++m) {
    const float* x_row = x + static_cast<size_t>(m) * K;

    // 1. partial dot product
    float partial = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      partial += x_row[k] * w_row[k];
    }

    // 2. combine the partial results of all threads in the block
    const float dot = block_reduce_sum(partial);

    // 3. one thread writes the answer
    if (threadIdx.x == 0) {
      out[static_cast<size_t>(m) * N + n] = dot;
    }
  }
}

void matmul(const float* x, const float* w, int M, int N, int K, float* out,
            cudaStream_t stream) {
  const int threads = 256;
  const int blocks = N;
  matmul_kernel<<<blocks, threads, 0, stream>>>(x, w, M, N, K, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace llama3
