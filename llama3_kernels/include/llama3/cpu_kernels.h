// =============================================================================
// cpu_kernels.h  —  Every operation needed to run Llama 3, on the CPU.
//
// Conventions shared by ALL kernels (CPU and CUDA):
//   * Data type is float32.
//   * Tensors are plain pointers to row-major arrays. The comment of every
//     function lists the shape of each argument, e.g. w : [N, K] means
//     w[n * K + k].
//   * Output pointers come last (before the optional CUDA stream).
//   * "In place" is allowed only where the comment says so.
//
// The CUDA versions in cuda_kernels.h have the same names and arguments, so the
// two files can be read side by side.
// =============================================================================
#pragma once

namespace llama3 {
namespace cpu {

// Token embedding: copies row `token` of the embedding table.
//   table : [vocab_size, dim]
//   out   : [dim]
void embedding(const float* table, int token, int vocab_size, int dim, float* out);

// RMSNorm:  out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
//   x, weight, out : [dim]           (out may be the same pointer as x)
void rmsnorm(const float* x, const float* weight, int dim, float eps, float* out);

// Linear layer without bias:  out[m, n] = sum_k x[m, k] * w[n, k]
// (w is stored like PyTorch's nn.Linear.weight: one row per output feature.)
//   x   : [M, K]      M = number of tokens (1 while generating)
//   w   : [N, K]
//   out : [M, N]      (must not overlap x)
void matmul(const float* x, const float* w, int M, int N, int K, float* out);

// Rotary position embedding, applied in place to the query and key of ONE token.
// Uses the "rotate half" layout of HuggingFace checkpoints: inside every head,
// element i is paired with element i + head_dim/2.
//   q        : [n_heads, head_dim]       (modified in place)
//   k        : [n_kv_heads, head_dim]    (modified in place)
//   inv_freq : [head_dim / 2]            (from rope_inv_freq())
void rope(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
          const float* inv_freq);

// Softmax over each row, in place.
//   x : [rows, cols]
void softmax(float* x, int rows, int cols);

// Causal self-attention for ONE new token with Grouped Query Attention.
// The new token's key/value must already be written into the caches.
//   q       : [n_heads, head_dim]
//   k_cache : [>= n_tokens, n_kv_heads, head_dim]   (row t = key of position t)
//   v_cache : [>= n_tokens, n_kv_heads, head_dim]
//   scores  : [n_heads, n_tokens]   scratch space (holds the attention weights afterwards)
//   out     : [n_heads, head_dim]
// n_tokens = pos + 1 = number of positions this token can look at.
void attention(const float* q, const float* k_cache, const float* v_cache, int n_heads,
               int n_kv_heads, int head_dim, int n_tokens, float* scores, float* out);

// SwiGLU activation of the feed-forward block:  out[i] = silu(gate[i]) * up[i]
// with silu(x) = x * sigmoid(x).
//   gate, up, out : [n]              (out may be the same pointer as gate or up)
void swiglu(const float* gate, const float* up, int n, float* out);

// Element-wise add (the residual connection):  out[i] = a[i] + b[i]
//   a, b, out : [n]                  (out may be the same pointer as a or b)
void add(const float* a, const float* b, int n, float* out);

// Index of the largest value (greedy sampling). Ties -> the smallest index.
//   x : [n]
int argmax(const float* x, int n);

}  // namespace cpu
}  // namespace llama3
