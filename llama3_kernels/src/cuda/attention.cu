// =============================================================================
// cuda/attention.cu  —  Causal GQA self-attention for one token, on the GPU.
// (See cpu/attention.cpp for the math and the KV-cache layout.)
//
// The work is split into three small kernels, launched one after another on
// the same stream (the GPU runs them in order):
//
//   1. attention_scores_kernel  : one thread  = one score  (head h, position t)
//         grid  = (ceil(n_tokens / 256), n_heads),  block = 256
//         scores[h][t] = q_h . k_t / sqrt(head_dim)
//
//   2. softmax (softmax.cu)     : one block   = one head's row of scores
//
//   3. attention_output_kernel  : one thread  = one output element (h, i)
//         grid  = n_heads,  block = head_dim
//         out[h][i] = sum_t scores[h][t] * v_t[i]
//         Threads of a block read v_t[0], v_t[1], ... -> coalesced loads.
//
// Splitting is not the fastest approach (production engines fuse everything
// into one "flash attention" kernel) but each piece is easy to understand.
// =============================================================================
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"

namespace llama3 {
namespace cuda {

__global__ void attention_scores_kernel(const float* q, const float* k_cache, int n_kv_heads,
                                        int head_dim, int group_size, int n_tokens,
                                        float scale, float* scores) {
  const int h = blockIdx.y;                               // query head
  const int t = blockIdx.x * blockDim.x + threadIdx.x;    // cached position
  if (t >= n_tokens) return;

  const int kv_dim = n_kv_heads * head_dim;
  const int kv_head = h / group_size;
  const float* q_head = q + h * head_dim;
  const float* k_head = k_cache + static_cast<size_t>(t) * kv_dim + kv_head * head_dim;

  float dot = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    dot += q_head[i] * k_head[i];
  }
  scores[static_cast<size_t>(h) * n_tokens + t] = dot * scale;
}

__global__ void attention_output_kernel(const float* scores, const float* v_cache,
                                        int n_kv_heads, int head_dim, int group_size,
                                        int n_tokens, float* out) {
  const int h = blockIdx.x;   // query head
  const int i = threadIdx.x;  // element inside the head
  if (i >= head_dim) return;

  const int kv_dim = n_kv_heads * head_dim;
  const int kv_head = h / group_size;
  const float* head_scores = scores + static_cast<size_t>(h) * n_tokens;

  float sum = 0.0f;
  for (int t = 0; t < n_tokens; ++t) {
    const float* v_head = v_cache + static_cast<size_t>(t) * kv_dim + kv_head * head_dim;
    sum += head_scores[t] * v_head[i];
  }
  out[h * head_dim + i] = sum;
}

void attention(const float* q, const float* k_cache, const float* v_cache, int n_heads,
               int n_kv_heads, int head_dim, int n_tokens, float* scores, float* out,
               cudaStream_t stream) {
  if (head_dim > 1024) {
    std::fprintf(stderr, "cuda::attention: head_dim %d > 1024 is not supported\n", head_dim);
    std::exit(EXIT_FAILURE);
  }
  const int group_size = n_heads / n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // 1. scores = q . k / sqrt(head_dim)
  const int threads = 256;
  const dim3 score_grid(ceil_div(n_tokens, threads), n_heads);
  attention_scores_kernel<<<score_grid, threads, 0, stream>>>(
      q, k_cache, n_kv_heads, head_dim, group_size, n_tokens, scale, scores);
  CUDA_CHECK(cudaGetLastError());

  // 2. softmax over each head's scores
  softmax(scores, n_heads, n_tokens, stream);

  // 3. out = weights * V
  attention_output_kernel<<<n_heads, head_dim, 0, stream>>>(scores, v_cache, n_kv_heads,
                                                             head_dim, group_size, n_tokens, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace llama3
