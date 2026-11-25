// =============================================================================
// llama3_forward.cpp  —  A complete Llama 3 forward pass built from the kernels.
//
// This program shows how the individual kernels fit together into a model.
// It creates a Llama 3 model with RANDOM weights (real weights would need a
// checkpoint loader and a tokenizer, which are outside the scope of the kernel
// library), generates tokens greedily one at a time, and does it twice:
//
//   * CpuLlama3  — all math with llama3::cpu::*  on CPU memory
//   * CudaLlama3 — all math with llama3::cuda::* on GPU memory
//
// The GPU model is fed exactly the same tokens as the CPU model, so the logits
// of every step can be compared, and the time per token measured.
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
//   next token = argmax(logits)
//
// Usage: llama3_forward [--model tiny|1b|8b] [--layers N] [--steps 32]
//                       [--device both|cpu|cuda] [--seed 1]
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "app_utils.h"
#include "llama3/config.h"
#include "llama3/cpu_kernels.h"
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"
#include "llama3/rope_frequencies.h"

using llama3::DeviceBuffer;
using llama3::ModelConfig;
using std::vector;

// =============================================================================
// Weights (stored on the CPU; the CUDA model copies them to the GPU)
// =============================================================================
struct LayerWeights {
  vector<float> attn_norm;  // [dim]
  vector<float> wq;         // [dim, dim]
  vector<float> wk;         // [kv_dim, dim]
  vector<float> wv;         // [kv_dim, dim]
  vector<float> wo;         // [dim, dim]
  vector<float> ffn_norm;   // [dim]
  vector<float> w_gate;     // [hidden_dim, dim]   ("w1" in Meta's code)
  vector<float> w_up;       // [hidden_dim, dim]   ("w3")
  vector<float> w_down;     // [dim, hidden_dim]   ("w2")
};

struct Weights {
  vector<float> embedding;  // [vocab, dim]
  vector<LayerWeights> layers;
  vector<float> final_norm;  // [dim]
  vector<float> lm_head;     // [vocab, dim]; empty when tied to the embedding
};

static Weights make_random_weights(const ModelConfig& c, uint64_t seed) {
  // Values ~ uniform(-1/sqrt(in_features), +1/sqrt(in_features)) keep the
  // activations at a reasonable size, like a freshly initialised network.
  uint64_t s = seed;
  auto matrix = [&](int rows, int cols) {
    const float a = 1.0f / std::sqrt(static_cast<float>(cols));
    return app::random_vector(static_cast<size_t>(rows) * cols, -a, a, s++);
  };
  auto norm = [&](int n) { return app::random_vector(n, 0.8f, 1.2f, s++); };

  Weights w;
  // Small embedding values (like GPT-2's init). With tied embeddings, large
  // values would make every token predict itself again.
  w.embedding = app::random_vector(static_cast<size_t>(c.vocab_size) * c.dim, -0.05f, 0.05f, s++);
  for (int l = 0; l < c.n_layers; ++l) {
    LayerWeights L;
    L.attn_norm = norm(c.dim);
    L.wq = matrix(c.dim, c.dim);
    L.wk = matrix(c.kv_dim(), c.dim);
    L.wv = matrix(c.kv_dim(), c.dim);
    L.wo = matrix(c.dim, c.dim);
    L.ffn_norm = norm(c.dim);
    L.w_gate = matrix(c.hidden_dim, c.dim);
    L.w_up = matrix(c.hidden_dim, c.dim);
    L.w_down = matrix(c.dim, c.hidden_dim);
    w.layers.push_back(std::move(L));
  }
  w.final_norm = norm(c.dim);
  if (!c.tied_embedding) w.lm_head = matrix(c.vocab_size, c.dim);
  return w;
}

static double weight_bytes(const ModelConfig& c) {
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
    inv_freq_ = llama3::rope_inv_freq(c.head_dim(), c.rope_theta, c.rope_scaling);
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

  // Runs one token through the model; returns the logits (vocab_size floats).
  const float* forward(int token, int pos) {
    namespace k = llama3::cpu;
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

    k::rmsnorm(x_.data(), w_.final_norm.data(), dim, c_.rms_eps, x_.data());
    k::matmul(x_.data(), lm_head, 1, c_.vocab_size, dim, logits_.data());
    return logits_.data();
  }

 private:
  const ModelConfig& c_;
  const Weights& w_;
  vector<float> inv_freq_;
  vector<float> x_, xb_, q_, attn_out_, gate_, up_, scores_, logits_;
  vector<float> k_cache_, v_cache_;
};

// =============================================================================
// CUDA model — the same code with llama3::cuda:: and GPU buffers
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
    inv_freq_.upload(llama3::rope_inv_freq(c.head_dim(), c.rope_theta, c.rope_scaling));

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
  const float* forward(int token, int pos) {
    namespace k = llama3::cuda;
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

    k::rmsnorm(x_.get(), final_norm_.get(), dim, c_.rms_eps, x_.get());
    k::matmul(x_.get(), lm_head, 1, c_.vocab_size, dim, logits_.get());
    return logits_.get();
  }

  int argmax() { return llama3::cuda::argmax(logits_.get(), c_.vocab_size); }
  vector<float> logits_on_cpu() const { return logits_.download(); }

 private:
  const ModelConfig& c_;
  DeviceBuffer<float> embedding_, final_norm_, lm_head_, inv_freq_;
  vector<DeviceLayerWeights> layers_;
  DeviceBuffer<float> x_, xb_, q_, attn_out_, gate_, up_, scores_, logits_;
  DeviceBuffer<float> k_cache_, v_cache_;
};

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
  app::configure_cpu_threads();
  ModelConfig c;
  const std::string model = app::get_arg(argc, argv, "--model", "tiny");
  if (!llama3::config_by_name(model, &c)) {
    std::fprintf(stderr, "unknown --model '%s' (use tiny, 1b or 8b)\n", model.c_str());
    return 1;
  }
  c.n_layers = app::get_int_arg(argc, argv, "--layers", c.n_layers);
  const int steps = app::get_int_arg(argc, argv, "--steps", 32);
  const std::string device = app::get_arg(argc, argv, "--device", "both");
  const uint64_t seed = static_cast<uint64_t>(app::get_int_arg(argc, argv, "--seed", 1));
  c.max_seq_len = std::max(c.max_seq_len, steps);
  const bool run_cpu = device == "both" || device == "cpu";
  const bool run_gpu = device == "both" || device == "cuda";
  if (!run_cpu && !run_gpu) {
    std::fprintf(stderr, "unknown --device '%s' (use both, cpu or cuda)\n", device.c_str());
    return 1;
  }

  const double gb = weight_bytes(c) / 1e9;
  std::printf("Model    : %s shapes, %d layers, dim=%d, hidden=%d, heads=%d/%d, vocab=%d\n",
              c.name.c_str(), c.n_layers, c.dim, c.hidden_dim, c.n_heads, c.n_kv_heads,
              c.vocab_size);
  std::printf("Weights  : %.2f GB (fp32, random values)\n", gb);
  std::printf("Steps    : %d tokens, greedy decoding\n", steps);
  if (run_cpu) std::printf("CPU      : %d threads (change with OMP_NUM_THREADS=n)\n", app::cpu_thread_count());
  if (run_gpu) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("GPU      : %s (%.1f GB free)\n", prop.name, free_bytes / 1e9);
    if (weight_bytes(c) > 0.95 * free_bytes) {
      std::fprintf(stderr, "Not enough GPU memory for these weights. Try fewer layers, "
                           "e.g. --layers 2.\n");
      return 1;
    }
  }

  app::CpuTimer timer;
  timer.start();
  const Weights weights = make_random_weights(c, seed);
  std::printf("Created random weights in %.1f s\n\n", timer.elapsed_ms() / 1000);

  // Every step feeds one token. Step 0 uses token 1; later steps use the
  // token predicted by the previous step (greedy decoding).
  vector<int> input_tokens = {1};
  vector<vector<float>> cpu_logits;  // kept to compare with the GPU
  double cpu_ms_per_token = 0, gpu_ms_per_token = 0;

  if (run_cpu) {
    CpuLlama3 cpu_model(c, weights);
    cpu_model.forward(input_tokens[0], 0);  // warm-up (starts the OpenMP threads)
    timer.start();
    for (int pos = 0; pos < steps; ++pos) {
      const float* logits = cpu_model.forward(input_tokens[pos], pos);
      const int next = llama3::cpu::argmax(logits, c.vocab_size);
      input_tokens.push_back(next);
      if (run_gpu) cpu_logits.emplace_back(logits, logits + c.vocab_size);
    }
    cpu_ms_per_token = timer.elapsed_ms() / steps;
  }

  vector<int> gpu_predictions;
  float worst_error = 0.0f;
  if (run_gpu) {
    timer.start();
    CudaLlama3 gpu_model(c, weights);
    std::printf("Copied weights to the GPU in %.1f s\n", timer.elapsed_ms() / 1000);
    gpu_model.forward(input_tokens[0], 0);  // warm-up (loads the kernels onto the GPU)
    gpu_model.argmax();

    // When comparing with the CPU, measure the time separately from the
    // (slow) logits download used for the comparison.
    double gpu_total_ms = 0;
    for (int pos = 0; pos < steps; ++pos) {
      timer.start();
      gpu_model.forward(input_tokens[pos], pos);
      const int next = gpu_model.argmax();  // waits for the GPU to finish this token
      gpu_total_ms += timer.elapsed_ms();

      gpu_predictions.push_back(next);
      if (run_cpu) {
        worst_error = std::max(worst_error, app::relative_error(cpu_logits[pos],
                                                                gpu_model.logits_on_cpu()));
      } else {
        input_tokens.push_back(next);
      }
    }
    gpu_ms_per_token = gpu_total_ms / steps;
  }

  // ---- report ----
  std::printf("\nGenerated token ids (meaningless, the weights are random): ");
  for (int i = 1; i <= std::min(steps, 16); ++i) std::printf("%d ", input_tokens[i]);
  std::printf("%s\n\n", steps > 16 ? "..." : "");

  if (run_cpu) {
    std::printf("CPU : %9.2f ms/token  (%7.2f tokens/s)\n", cpu_ms_per_token,
                1000 / cpu_ms_per_token);
  }
  if (run_gpu) {
    std::printf("GPU : %9.2f ms/token  (%7.2f tokens/s)\n", gpu_ms_per_token,
                1000 / gpu_ms_per_token);
  }
  if (run_cpu && run_gpu) {
    int agree = 0;
    for (int pos = 0; pos < steps; ++pos) agree += gpu_predictions[pos] == input_tokens[pos + 1];
    std::printf("GPU speed-up         : %.1fx\n", cpu_ms_per_token / gpu_ms_per_token);
    std::printf("Same next token      : %d / %d steps\n", agree, steps);
    std::printf("Worst logits rel.err : %.2e\n", worst_error);
    const bool ok = agree == steps && worst_error < 1e-3f;
    std::printf("%s\n", ok ? "CPU and GPU agree." : "WARNING: CPU and GPU results differ!");
    return ok ? 0 : 1;
  }
  return 0;
}
