// =============================================================================
// llama3_forward.cpp  —  A complete Llama 3 forward pass built from the kernels.
//
// This program shows how the individual kernels fit together into a model.
// It creates a Llama 3 model with RANDOM weights (so it needs no download),
// generates tokens greedily one at a time on both CPU and GPU, and compares
// speed and results. The model itself is in include/llama3/model.h; to run real
// weights, see chat.cpp.
//
// The GPU model is fed exactly the same tokens as the CPU model, so the logits
// of every step can be compared.
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
#include "llama3/cuda_utils.h"
#include "llama3/model.h"

using llama3::CpuLlama3;
using llama3::CudaLlama3;
using llama3::LayerWeights;
using llama3::ModelConfig;
using llama3::weight_bytes;
using llama3::Weights;
using std::vector;

// =============================================================================
// Random weights
// =============================================================================
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
