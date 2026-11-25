// =============================================================================
// cpu/attention.cpp  —  Causal self-attention for one new token (decode step).
//
// For every query head h:
//   1. scores[t] = (q_h . k_t) / sqrt(head_dim)      for t = 0 .. n_tokens-1
//   2. weights   = softmax(scores)
//   3. out_h     = sum_t weights[t] * v_t
//
// "Causal" comes for free: the cache only contains the current and previous
// positions, so the token can never look at the future.
//
// Grouped Query Attention (GQA): Llama 3 has fewer K/V heads than query heads
// (8B: 32 query heads, 8 KV heads). Consecutive query heads share a KV head:
//
//     query heads : 0 1 2 3 | 4 5 6 7 | ... | 28 29 30 31
//     kv head     :    0    |    1    | ... |     7
//     kv_head = h / (n_heads / n_kv_heads)
//
// KV-cache layout (one layer):  cache[t][kv_head][i]
//     row t = all K (or V) heads of position t, row length kv_dim.
// =============================================================================
#include <cmath>
#include <cstddef>

#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

void attention(const float* q, const float* k_cache, const float* v_cache, int n_heads,
               int n_kv_heads, int head_dim, int n_tokens, float* scores, float* out) {
  const int kv_dim = n_kv_heads * head_dim;
  const int group_size = n_heads / n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Heads are independent -> one head per thread.
#pragma omp parallel for schedule(static)
  for (int h = 0; h < n_heads; ++h) {
    const float* q_head = q + h * head_dim;
    const int kv_head = h / group_size;
    float* head_scores = scores + static_cast<size_t>(h) * n_tokens;

    // ---- 1. scores: how well does this query match the key at each position?
    for (int t = 0; t < n_tokens; ++t) {
      const float* k_head = k_cache + static_cast<size_t>(t) * kv_dim + kv_head * head_dim;
      float dot = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        dot += q_head[i] * k_head[i];
      }
      head_scores[t] = dot * scale;
    }

    // ---- 2. softmax: scores -> weights that sum to 1
    softmax(head_scores, 1, n_tokens);

    // ---- 3. weighted sum of the value vectors
    float* out_head = out + h * head_dim;
    for (int i = 0; i < head_dim; ++i) {
      out_head[i] = 0.0f;
    }
    for (int t = 0; t < n_tokens; ++t) {
      const float* v_head = v_cache + static_cast<size_t>(t) * kv_dim + kv_head * head_dim;
      const float weight = head_scores[t];
      for (int i = 0; i < head_dim; ++i) {
        out_head[i] += weight * v_head[i];
      }
    }
  }
}

}  // namespace cpu
}  // namespace llama3
