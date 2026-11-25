// =============================================================================
// cuda_kernels.h  —  Every operation needed to run Llama 3, on an NVIDIA GPU.
//
// Same names, arguments and shapes as cpu_kernels.h, except:
//   * every float pointer must point to GPU memory (cudaMalloc),
//   * every function takes an optional CUDA stream as its last argument,
//   * the functions only *launch* work; it runs asynchronously on the GPU.
//     Call cudaStreamSynchronize / cudaDeviceSynchronize before reading
//     results on the CPU (argmax does this for you because it returns an int).
//
// This header is plain C++ (no CUDA syntax), so it can be included from
// ordinary .cpp files.
// =============================================================================
#pragma once

#include <cuda_runtime_api.h>

namespace llama3 {
namespace cuda {

// table : [vocab_size, dim] (GPU)   out : [dim] (GPU)   token : CPU integer
void embedding(const float* table, int token, int vocab_size, int dim, float* out,
               cudaStream_t stream = nullptr);

// x, weight, out : [dim]   (out may equal x)
void rmsnorm(const float* x, const float* weight, int dim, float eps, float* out,
             cudaStream_t stream = nullptr);

// x : [M, K]   w : [N, K]   out : [M, N]
void matmul(const float* x, const float* w, int M, int N, int K, float* out,
            cudaStream_t stream = nullptr);

// q : [n_heads, head_dim]   k : [n_kv_heads, head_dim]   inv_freq : [head_dim/2] (GPU)
void rope(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
          const float* inv_freq, cudaStream_t stream = nullptr);

// x : [rows, cols], in place
void softmax(float* x, int rows, int cols, cudaStream_t stream = nullptr);

// See cpu::attention. Internally launches three kernels:
//   1. attention scores  (q . k for every cached position)
//   2. softmax           (turn scores into weights)
//   3. weighted sum of V
void attention(const float* q, const float* k_cache, const float* v_cache, int n_heads,
               int n_kv_heads, int head_dim, int n_tokens, float* scores, float* out,
               cudaStream_t stream = nullptr);

// gate, up, out : [n]
void swiglu(const float* gate, const float* up, int n, float* out, cudaStream_t stream = nullptr);

// a, b, out : [n]
void add(const float* a, const float* b, int n, float* out, cudaStream_t stream = nullptr);

// x : [n] (GPU). Waits for the GPU and returns the index on the CPU.
int argmax(const float* x, int n, cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace llama3
