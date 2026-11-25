// =============================================================================
// cuda/rope.cu  —  Rotary Position Embedding on the GPU (see cpu/rope.cpp for
// the math).
//
// Thread mapping: one thread rotates one (x0, x1) pair.
//
//   There are n_heads query heads and n_kv_heads key heads, each with
//   half = head_dim / 2 pairs, so we launch (n_heads + n_kv_heads) * half
//   threads and number them like this:
//
//     pair id : 0 ........ n_heads*half-1 | n_heads*half ...... end
//     vector  :            q              |            k
//
//   Inside a vector:  head = id / half,   i = id % half
//   and the thread rotates (head[i], head[i + half]).
// =============================================================================
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

__global__ void rope_kernel(float* q, float* k, int n_heads, int n_kv_heads, int head_dim,
                            int pos, const float* inv_freq) {
  const int half = head_dim / 2;
  const int q_pairs = n_heads * half;
  const int total_pairs = (n_heads + n_kv_heads) * half;

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= total_pairs) return;

  // Is this pair part of q or of k?
  float* vec = q;
  if (id >= q_pairs) {
    vec = k;
    id -= q_pairs;
  }
  const int head = id / half;
  const int i = id % half;
  float* head_ptr = vec + head * head_dim;

  const float angle = static_cast<float>(pos) * inv_freq[i];
  const float c = cosf(angle);
  const float s = sinf(angle);

  const float x0 = head_ptr[i];
  const float x1 = head_ptr[i + half];
  head_ptr[i] = x0 * c - x1 * s;
  head_ptr[i + half] = x0 * s + x1 * c;
}

void rope(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
          const float* inv_freq, cudaStream_t stream) {
  const int total_pairs = (n_heads + n_kv_heads) * (head_dim / 2);
  const int threads = 256;
  const int blocks = ceil_div(total_pairs, threads);
  rope_kernel<<<blocks, threads, 0, stream>>>(q, k, n_heads, n_kv_heads, head_dim, pos,
                                              inv_freq);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace llama3
