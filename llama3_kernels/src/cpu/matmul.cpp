// =============================================================================
// cpu/matmul.cpp  —  Linear layer:  out = x * W^T
//
//    out[m, n] = sum over k of  x[m, k] * w[n, k]
//
// Each output value is the dot product of one input row with one weight row:
//
//      x  (M x K)          w  (N x K)            out (M x N)
//   m [ x x x x ]     n [ w w w w ]   ---->   out[m, n] = x_m . w_n
//
// This is where almost all of Llama's time goes: q, k, v, o projections, the
// three feed-forward matrices, and the final vocabulary projection.
//
// While generating text M = 1 (one token), so every weight is read exactly once
// per token and the speed is limited by memory bandwidth, not by arithmetic.
//
// Parallelism: different output columns n are independent, so OpenMP gives each
// CPU thread a slice of the weight rows. `omp simd` lets the compiler use SIMD
// instructions (AVX) for the dot product. Without OpenMP the pragmas are
// ignored and the code runs on one thread.
// =============================================================================
#include <cstddef>

#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

void matmul(const float* x, const float* w, int M, int N, int K, float* out) {
#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const float* w_row = w + static_cast<size_t>(n) * K;

    // Re-use this weight row for every input row while it is still in cache.
    for (int m = 0; m < M; ++m) {
      const float* x_row = x + static_cast<size_t>(m) * K;

      float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
      for (int k = 0; k < K; ++k) {
        sum += x_row[k] * w_row[k];
      }
      out[static_cast<size_t>(m) * N + n] = sum;
    }
  }
}

}  // namespace cpu
}  // namespace llama3
