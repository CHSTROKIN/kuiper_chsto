// =============================================================================
// model.h  —  A complete Llama 3 model built from the kernels, for CPU and CUDA.
//
//   * CpuLlama3  — all math with llama3::cpu::*  on CPU memory
//   * CudaLlama3 — all math with llama3::cuda::* on GPU memory
//
// One decoding step (= one token) of Llama 3:
//
//   x = embedding[token]
//   for each layer:
//       h = rmsnorm(x, attn_norm)
//       q = h * Wq^T,  k = h * Wk^T,  v = h * Wv^T      (k, v go into the KV cache)
//       rope(q, k, pos)
//       a = attention(q, K cache, V cache)
//       x = x + a * Wo^T                                (residual)
//       h = rmsnorm(x, ffn_norm)
//       g = h * Wgate^T,  u = h * Wup^T
//       x = x + swiglu(g, u) * Wdown^T                  (residual)
//   logits = rmsnorm(x, final_norm) * Wlm_head^T
//
// Tokens must be fed in order: position `pos` attends to the cached keys and
// values of positions 0 .. pos, and pos must be < config.max_seq_len.
// Both classes keep a reference to the config; CpuLlama3 also keeps a reference
// to the weights, while CudaLlama3 copies them to the GPU.
// =============================================================================
#pragma once

#include <cstddef>
#include <vector>

#include "llama3/config.h"
#include "llama3/cpu_kernels.h"
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"
#include "llama3/rope_frequencies.h"

namespace llama3 {

// =============================================================================
// Weights (stored on the CPU; the CUDA model copies them to the GPU)
// =============================================================================
struct LayerWeights {
  std::vector<float> attn_norm;  // [dim]
  std::vector<float> wq;         // [dim, dim]
  std::vector<float> wk;         // [kv_dim, dim]
  std::vector<float> wv;         // [kv_dim, dim]
  std::vector<float> wo;         // [dim, dim]
  std::vector<float> ffn_norm;   // [dim]
  std::vector<float> w_gate;     // [hidden_dim, dim]   ("w1" in Meta's code)
  std::vector<float> w_up;       // [hidden_dim, dim]   ("w3")
  std::vector<float> w_down;     // [dim, hidden_dim]   ("w2")
};

struct Weights {
  std::vector<float> embedding;  // [vocab, dim]
  std::vector<LayerWeights> layers;
  std::vector<float> final_norm;  // [dim]
  std::vector<float> lm_head;     // [vocab, dim]; empty when tied to the embedding
};

// Size of all weights in fp32.
inline double weight_bytes(const ModelConfig& c) {
  const double per_layer = 2.0 * c.dim + 2.0 * c.dim * c.dim + 2.0 * c.kv_dim() * c.dim +
                           3.0 * c.hidden_dim * c.dim;
  const double embed = static_cast<double>(c.vocab_size) * c.dim;
  return 4.0 * (embed * (c.tied_embedding ? 1 : 2) + c.n_layers * per_layer + c.dim);
}

// =============================================================================
// CPU model
// =============================================================================
class CpuLlama3 {
 public:
  CpuLlama3(const ModelConfig& c, const Weights& w) : c_(c), w_(w) {
    inv_freq_ = rope_inv_freq(c.head_dim(), c.rope_theta, c.rope_scaling);
    x_.resize(c.dim);
    xb_.resize(c.dim);
    q_.resize(c.dim);
    attn_out_.resize(c.dim);
    gate_.resize(c.hidden_dim);
    up_.resize(c.hidden_dim);
    scores_.resize(static_cast<size_t>(c.n_heads) * c.max_seq_len);
    logits_.resize(c.vocab_size);
    // KV cache: [layer][position][kv_dim]
    k_cache_.resize(static_cast<size_t>(c.n_layers) * c.max_seq_len * c.kv_dim());
    v_cache_.resize(k_cache_.size());
  }

  // Runs one token through the model; returns the logits (vocab_size floats),
  // or nullptr when need_logits is false.
  const float* forward(int token, int pos, bool need_logits = true) {
    namespace k = cpu;
    const int dim = c_.dim, kv_dim = c_.kv_dim(), hidden = c_.hidden_dim;
    const float* lm_head = c_.tied_embedding ? w_.embedding.data() : w_.lm_head.data();

    k::embedding(w_.embedding.data(), token, c_.vocab_size, dim, x_.data());

    for (int l = 0; l < c_.n_layers; ++l) {
      const LayerWeights& L = w_.layers[l];
      // this layer's part of the cache, and the row for the current position
      float* k_cache = k_cache_.data() + static_cast<size_t>(l) * c_.max_seq_len * kv_dim;
      float* v_cache = v_cache_.data() + static_cast<size_t>(l) * c_.max_seq_len * kv_dim;
      float* k_now = k_cache + static_cast<size_t>(pos) * kv_dim;
      float* v_now = v_cache + static_cast<size_t>(pos) * kv_dim;

      // ---- attention block ----
      k::rmsnorm(x_.data(), L.attn_norm.data(), dim, c_.rms_eps, xb_.data());
      k::matmul(xb_.data(), L.wq.data(), 1, dim, dim, q_.data());
      k::matmul(xb_.data(), L.wk.data(), 1, kv_dim, dim, k_now);  // written straight into the cache
      k::matmul(xb_.data(), L.wv.data(), 1, kv_dim, dim, v_now);
      k::rope(q_.data(), k_now, c_.n_heads, c_.n_kv_heads, c_.head_dim(), pos, inv_freq_.data());
      k::attention(q_.data(), k_cache, v_cache, c_.n_heads, c_.n_kv_heads, c_.head_dim(), pos + 1,
                   scores_.data(), attn_out_.data());
      k::matmul(attn_out_.data(), L.wo.data(), 1, dim, dim, xb_.data());
      k::add(x_.data(), xb_.data(), dim, x_.data());

      // ---- feed-forward block ----
      k::rmsnorm(x_.data(), L.ffn_norm.data(), dim, c_.rms_eps, xb_.data());
      k::matmul(xb_.data(), L.w_gate.data(), 1, hidden, dim, gate_.data());
      k::matmul(xb_.data(), L.w_up.data(), 1, hidden, dim, up_.data());
      k::swiglu(gate_.data(), up_.data(), hidden, gate_.data());
      k::matmul(gate_.data(), L.w_down.data(), 1, dim, hidden, xb_.data());
      k::add(x_.data(), xb_.data(), dim, x_.data());
    }

    if (!need_logits) return nullptr;  // prompt tokens only fill the KV cache

    k::rmsnorm(x_.data(), w_.final_norm.data(), dim, c_.rms_eps, x_.data());
    k::matmul(x_.data(), lm_head, 1, c_.vocab_size, dim, logits_.data());
    return logits_.data();
  }

 private:
  const ModelConfig& c_;
  const Weights& w_;
  std::vector<float> inv_freq_;
  std::vector<float> x_, xb_, q_, attn_out_, gate_, up_, scores_, logits_;
  std::vector<float> k_cache_, v_cache_;
};

// =============================================================================
// CUDA model — the same code with cuda:: and GPU buffers
// =============================================================================
struct DeviceLayerWeights {
  DeviceBuffer<float> attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down;
};

class CudaLlama3 {
 public:
  CudaLlama3(const ModelConfig& c, const Weights& w) : c_(c) {
    // copy all weights to the GPU once
    embedding_.upload(w.embedding);
    final_norm_.upload(w.final_norm);
    if (!c.tied_embedding) lm_head_.upload(w.lm_head);
    layers_.resize(c.n_layers);
    for (int l = 0; l < c.n_layers; ++l) {
      const LayerWeights& src = w.layers[l];
      DeviceLayerWeights& dst = layers_[l];
      dst.attn_norm.upload(src.attn_norm);
      dst.wq.upload(src.wq);
      dst.wk.upload(src.wk);
      dst.wv.upload(src.wv);
      dst.wo.upload(src.wo);
      dst.ffn_norm.upload(src.ffn_norm);
      dst.w_gate.upload(src.w_gate);
      dst.w_up.upload(src.w_up);
      dst.w_down.upload(src.w_down);
    }
    inv_freq_.upload(rope_inv_freq(c.head_dim(), c.rope_theta, c.rope_scaling));

    // activations and KV cache live on the GPU too
    x_.allocate(c.dim);
    xb_.allocate(c.dim);
    q_.allocate(c.dim);
    attn_out_.allocate(c.dim);
    gate_.allocate(c.hidden_dim);
    up_.allocate(c.hidden_dim);
    scores_.allocate(static_cast<size_t>(c.n_heads) * c.max_seq_len);
    logits_.allocate(c.vocab_size);
    k_cache_.allocate(static_cast<size_t>(c.n_layers) * c.max_seq_len * c.kv_dim());
    v_cache_.allocate(k_cache_.size());
  }

  // Launches one token through the model; returns the GPU pointer to the logits.
  // Everything runs asynchronously; argmax() below waits for the GPU.
  const float* forward(int token, int pos, bool need_logits = true) {
    namespace k = cuda;
    const int dim = c_.dim, kv_dim = c_.kv_dim(), hidden = c_.hidden_dim;
    const float* lm_head = c_.tied_embedding ? embedding_.get() : lm_head_.get();

    k::embedding(embedding_.get(), token, c_.vocab_size, dim, x_.get());

    for (int l = 0; l < c_.n_layers; ++l) {
      DeviceLayerWeights& L = layers_[l];
      float* k_cache = k_cache_.get() + static_cast<size_t>(l) * c_.max_seq_len * kv_dim;
      float* v_cache = v_cache_.get() + static_cast<size_t>(l) * c_.max_seq_len * kv_dim;
      float* k_now = k_cache + static_cast<size_t>(pos) * kv_dim;
      float* v_now = v_cache + static_cast<size_t>(pos) * kv_dim;

      // ---- attention block ----
      k::rmsnorm(x_.get(), L.attn_norm.get(), dim, c_.rms_eps, xb_.get());
      k::matmul(xb_.get(), L.wq.get(), 1, dim, dim, q_.get());
      k::matmul(xb_.get(), L.wk.get(), 1, kv_dim, dim, k_now);
      k::matmul(xb_.get(), L.wv.get(), 1, kv_dim, dim, v_now);
      k::rope(q_.get(), k_now, c_.n_heads, c_.n_kv_heads, c_.head_dim(), pos, inv_freq_.get());
      k::attention(q_.get(), k_cache, v_cache, c_.n_heads, c_.n_kv_heads, c_.head_dim(), pos + 1,
                   scores_.get(), attn_out_.get());
      k::matmul(attn_out_.get(), L.wo.get(), 1, dim, dim, xb_.get());
      k::add(x_.get(), xb_.get(), dim, x_.get());

      // ---- feed-forward block ----
      k::rmsnorm(x_.get(), L.ffn_norm.get(), dim, c_.rms_eps, xb_.get());
      k::matmul(xb_.get(), L.w_gate.get(), 1, hidden, dim, gate_.get());
      k::matmul(xb_.get(), L.w_up.get(), 1, hidden, dim, up_.get());
      k::swiglu(gate_.get(), up_.get(), hidden, gate_.get());
      k::matmul(gate_.get(), L.w_down.get(), 1, dim, hidden, xb_.get());
      k::add(x_.get(), xb_.get(), dim, x_.get());
    }

    if (!need_logits) return nullptr;  // prompt tokens only fill the KV cache

    k::rmsnorm(x_.get(), final_norm_.get(), dim, c_.rms_eps, x_.get());
    k::matmul(x_.get(), lm_head, 1, c_.vocab_size, dim, logits_.get());
    return logits_.get();
  }

  int argmax() { return cuda::argmax(logits_.get(), c_.vocab_size); }
  std::vector<float> logits_on_cpu() const { return logits_.download(); }

 private:
  const ModelConfig& c_;
  DeviceBuffer<float> embedding_, final_norm_, lm_head_, inv_freq_;
  std::vector<DeviceLayerWeights> layers_;
  DeviceBuffer<float> x_, xb_, q_, attn_out_, gate_, up_, scores_, logits_;
  DeviceBuffer<float> k_cache_, v_cache_;
};

}  // namespace llama3
