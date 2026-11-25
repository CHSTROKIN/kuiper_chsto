// =============================================================================
// bench_kernels.cpp  —  CPU vs CUDA speed of every kernel, at real Llama 3 sizes.
//
// For each kernel we
//   1. run the CPU and the CUDA version once on the same input and check that
//      they agree (RelErr column),
//   2. time the CPU version (wall clock) and the CUDA version (CUDA events,
//      kernel time only; inputs are already on the GPU),
//   3. print the time, the effective memory bandwidth and the speed-up.
//
// At the end the per-kernel times are combined into an estimate of how long one
// generated token takes and which kernels dominate.
//
// Usage:  bench_kernels [--model tiny|1b|8b] [--ctx 512] [--prefill 32] [--iters 20]
//         (CPU thread count: OMP_NUM_THREADS=1 bench_kernels ...)
// =============================================================================
#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "app_utils.h"
#include "llama3/config.h"
#include "llama3/cpu_kernels.h"
#include "llama3/cuda_kernels.h"
#include "llama3/cuda_utils.h"
#include "llama3/rope_frequencies.h"

using llama3::DeviceBuffer;
using std::vector;

namespace {

int g_iters = 20;
constexpr double kCpuTimeBudgetMs = 3000.0;  // stop repeating slow CPU kernels after ~3 s

struct Result {
  std::string name;
  double cpu_ms = 0;
  double gpu_ms = 0;
};
std::map<std::string, Result> g_results;

// Average CPU time of fn() in milliseconds.
double time_cpu(const std::function<void()>& fn) {
  fn();  // warm-up: page faults, caches, OpenMP thread start-up
  app::CpuTimer timer;
  timer.start();
  fn();
  const double first = timer.elapsed_ms();
  const int iters = std::max(1, std::min(g_iters, static_cast<int>(kCpuTimeBudgetMs / first)));

  timer.start();
  for (int i = 0; i < iters; ++i) fn();
  return timer.elapsed_ms() / iters;
}

// Average GPU time of fn() in milliseconds.
double time_gpu(const std::function<void()>& fn) {
  fn();  // warm-up: first launch loads the kernel code onto the GPU
  CUDA_CHECK(cudaDeviceSynchronize());
  app::GpuTimer timer;
  timer.start();
  for (int i = 0; i < g_iters; ++i) fn();
  return timer.stop_ms() / g_iters;
}

// One benchmark case.
//   reset()   : restore the inputs (CPU and GPU copies) to their initial values
//   cpu()/gpu(): run the kernel once
//   error()   : relative difference between the CPU and GPU outputs
//   bytes     : minimum number of bytes the kernel must read + write
void run_case(const std::string& name, const std::string& shape, double bytes,
              const std::function<void()>& reset, const std::function<void()>& cpu,
              const std::function<void()>& gpu, const std::function<float()>& error) {
  // correctness first, on identical inputs
  reset();
  cpu();
  gpu();
  CUDA_CHECK(cudaDeviceSynchronize());
  const float err = error();

  const double cpu_ms = time_cpu(cpu);
  const double gpu_ms = time_gpu(gpu);
  const double cpu_gbps = bytes / (cpu_ms * 1e-3) / 1e9;
  const double gpu_gbps = bytes / (gpu_ms * 1e-3) / 1e9;

  std::printf("%-22s %-24s %10.4f %8.1f %10.4f %8.1f %8.1fx %9.1e\n", name.c_str(),
              shape.c_str(), cpu_ms, cpu_gbps, gpu_ms, gpu_gbps, cpu_ms / gpu_ms, err);
  g_results[name] = {name, cpu_ms, gpu_ms};
}

std::string dims(std::initializer_list<long> values) {
  std::string s;
  for (long v : values) s += (s.empty() ? "" : "x") + std::to_string(v);
  return s;
}

// Copies a host vector into a device buffer (used by reset functions).
void to_gpu(const vector<float>& host, DeviceBuffer<float>& device) { device.upload(host); }

// ---- benchmark cases ---------------------------------------------------------

void bench_matmul(const std::string& name, int M, int N, int K) {
  const auto x = app::random_vector(static_cast<size_t>(M) * K, -1, 1, 1);
  const auto w = app::random_vector(static_cast<size_t>(N) * K, -1, 1, 2);
  vector<float> out(static_cast<size_t>(M) * N);
  DeviceBuffer<float> d_x, d_w, d_out(out.size());
  to_gpu(x, d_x);
  to_gpu(w, d_w);

  const double bytes = 4.0 * (static_cast<double>(N) * K + static_cast<double>(M) * K +
                              static_cast<double>(M) * N);
  run_case(
      name, dims({M, N, K}), bytes, [] {},
      [&] { llama3::cpu::matmul(x.data(), w.data(), M, N, K, out.data()); },
      [&] { llama3::cuda::matmul(d_x.get(), d_w.get(), M, N, K, d_out.get()); },
      [&] { return app::relative_error(out, d_out.download()); });
}

void bench_all(const llama3::ModelConfig& c, int ctx, int prefill) {
  const int dim = c.dim, hidden = c.hidden_dim, hd = c.head_dim();

  {  // embedding
    const auto table = app::random_vector(static_cast<size_t>(c.vocab_size) * dim, -1, 1, 3);
    vector<float> out(dim);
    DeviceBuffer<float> d_table, d_out(dim);
    to_gpu(table, d_table);
    const int token = c.vocab_size / 2;
    run_case(
        "embedding", dims({dim}), 8.0 * dim, [] {},
        [&] { llama3::cpu::embedding(table.data(), token, c.vocab_size, dim, out.data()); },
        [&] { llama3::cuda::embedding(d_table.get(), token, c.vocab_size, dim, d_out.get()); },
        [&] { return app::relative_error(out, d_out.download()); });
  }
  {  // rmsnorm
    const auto x = app::random_vector(dim, -1, 1, 4);
    const auto w = app::random_vector(dim, 0.5f, 1.5f, 5);
    vector<float> out(dim);
    DeviceBuffer<float> d_x, d_w, d_out(dim);
    to_gpu(x, d_x);
    to_gpu(w, d_w);
    run_case(
        "rmsnorm", dims({dim}), 12.0 * dim, [] {},
        [&] { llama3::cpu::rmsnorm(x.data(), w.data(), dim, c.rms_eps, out.data()); },
        [&] { llama3::cuda::rmsnorm(d_x.get(), d_w.get(), dim, c.rms_eps, d_out.get()); },
        [&] { return app::relative_error(out, d_out.download()); });
  }

  bench_matmul("matmul_q_o", 1, dim, dim);
  bench_matmul("matmul_k_v", 1, c.kv_dim(), dim);
  bench_matmul("matmul_gate_up", 1, hidden, dim);
  bench_matmul("matmul_down", 1, dim, hidden);
  bench_matmul("matmul_lm_head", 1, c.vocab_size, dim);
  bench_matmul("matmul_gate(prefill)", prefill, hidden, dim);

  {  // rope (in place -> reset() restores the inputs before the correctness check)
    const auto inv_freq = llama3::rope_inv_freq(hd, c.rope_theta, c.rope_scaling);
    const auto q0 = app::random_vector(static_cast<size_t>(dim), -1, 1, 6);
    const auto k0 = app::random_vector(static_cast<size_t>(c.kv_dim()), -1, 1, 7);
    vector<float> q, k;
    DeviceBuffer<float> d_q, d_k, d_freq;
    to_gpu(inv_freq, d_freq);
    const int pos = ctx - 1;
    run_case(
        "rope", dims({c.n_heads + c.n_kv_heads, hd}), 8.0 * (dim + c.kv_dim()),
        [&] {
          q = q0;
          k = k0;
          to_gpu(q0, d_q);
          to_gpu(k0, d_k);
        },
        [&] {
          llama3::cpu::rope(q.data(), k.data(), c.n_heads, c.n_kv_heads, hd, pos,
                            inv_freq.data());
        },
        [&] {
          llama3::cuda::rope(d_q.get(), d_k.get(), c.n_heads, c.n_kv_heads, hd, pos,
                             d_freq.get());
        },
        [&] {
          return std::max(app::relative_error(q, d_q.download()),
                          app::relative_error(k, d_k.download()));
        });
  }
  {  // attention over `ctx` cached tokens
    const auto q = app::random_vector(dim, -1, 1, 8);
    const auto k_cache = app::random_vector(static_cast<size_t>(ctx) * c.kv_dim(), -1, 1, 9);
    const auto v_cache = app::random_vector(static_cast<size_t>(ctx) * c.kv_dim(), -1, 1, 10);
    vector<float> scores(static_cast<size_t>(c.n_heads) * ctx), out(dim);
    DeviceBuffer<float> d_q, d_k, d_v, d_scores(scores.size()), d_out(dim);
    to_gpu(q, d_q);
    to_gpu(k_cache, d_k);
    to_gpu(v_cache, d_v);
    const double bytes = 4.0 * (2.0 * dim + 2.0 * ctx * c.kv_dim() + 2.0 * scores.size());
    run_case(
        "attention", "heads=" + std::to_string(c.n_heads) + " ctx=" + std::to_string(ctx),
        bytes, [] {},
        [&] {
          llama3::cpu::attention(q.data(), k_cache.data(), v_cache.data(), c.n_heads,
                                 c.n_kv_heads, hd, ctx, scores.data(), out.data());
        },
        [&] {
          llama3::cuda::attention(d_q.get(), d_k.get(), d_v.get(), c.n_heads, c.n_kv_heads, hd,
                                  ctx, d_scores.get(), d_out.get());
        },
        [&] { return app::relative_error(out, d_out.download()); });
  }
  {  // softmax on its own (the attention benchmark above already includes it)
    const auto x0 = app::random_vector(static_cast<size_t>(c.n_heads) * ctx, -5, 5, 11);
    vector<float> x;
    DeviceBuffer<float> d_x;
    run_case(
        "softmax", dims({c.n_heads, ctx}), 8.0 * x0.size(),
        [&] {
          x = x0;
          to_gpu(x0, d_x);
        },
        [&] { llama3::cpu::softmax(x.data(), c.n_heads, ctx); },
        [&] { llama3::cuda::softmax(d_x.get(), c.n_heads, ctx); },
        [&] { return app::relative_error(x, d_x.download()); });
  }
  {  // swiglu
    const auto gate = app::random_vector(hidden, -4, 4, 12);
    const auto up = app::random_vector(hidden, -1, 1, 13);
    vector<float> out(hidden);
    DeviceBuffer<float> d_gate, d_up, d_out(hidden);
    to_gpu(gate, d_gate);
    to_gpu(up, d_up);
    run_case(
        "swiglu", dims({hidden}), 12.0 * hidden, [] {},
        [&] { llama3::cpu::swiglu(gate.data(), up.data(), hidden, out.data()); },
        [&] { llama3::cuda::swiglu(d_gate.get(), d_up.get(), hidden, d_out.get()); },
        [&] { return app::relative_error(out, d_out.download()); });
  }
  {  // residual add
    const auto a = app::random_vector(dim, -1, 1, 14);
    const auto b = app::random_vector(dim, -1, 1, 15);
    vector<float> out(dim);
    DeviceBuffer<float> d_a, d_b, d_out(dim);
    to_gpu(a, d_a);
    to_gpu(b, d_b);
    run_case(
        "add", dims({dim}), 12.0 * dim, [] {},
        [&] { llama3::cpu::add(a.data(), b.data(), dim, out.data()); },
        [&] { llama3::cuda::add(d_a.get(), d_b.get(), dim, d_out.get()); },
        [&] { return app::relative_error(out, d_out.download()); });
  }
  {  // argmax (includes copying the answer back to the CPU)
    const auto logits = app::random_vector(c.vocab_size, -10, 10, 16);
    DeviceBuffer<float> d_logits;
    to_gpu(logits, d_logits);
    int cpu_index = -1, gpu_index = -2;
    run_case(
        "argmax", dims({c.vocab_size}), 4.0 * c.vocab_size, [] {},
        [&] { cpu_index = llama3::cpu::argmax(logits.data(), c.vocab_size); },
        [&] { gpu_index = llama3::cuda::argmax(d_logits.get(), c.vocab_size); },
        [&] { return cpu_index == gpu_index ? 0.0f : 1.0f; });
  }
}

// Combines the measured kernel times into the cost of generating ONE token
// with a KV-cache holding `ctx` positions.
void print_token_estimate(const llama3::ModelConfig& c, int ctx) {
  // kernel name -> how many times it runs per generated token
  const int L = c.n_layers;
  const std::vector<std::pair<std::string, int>> calls = {
      {"embedding", 1},          {"rmsnorm", 2 * L + 1},   {"matmul_q_o", 2 * L},
      {"matmul_k_v", 2 * L},     {"rope", L},              {"attention", L},
      {"add", 2 * L},            {"matmul_gate_up", 2 * L}, {"swiglu", L},
      {"matmul_down", L},        {"matmul_lm_head", 1},    {"argmax", 1},
  };

  double cpu_total = 0, gpu_total = 0;
  for (const auto& [name, count] : calls) {
    cpu_total += count * g_results[name].cpu_ms;
    gpu_total += count * g_results[name].gpu_ms;
  }

  std::printf("\nEstimated cost of ONE generated token (%s, %d layers, ctx=%d):\n",
              c.name.c_str(), L, ctx);
  std::printf("%-22s %6s %12s %7s %12s %7s\n", "kernel", "calls", "CPU ms", "CPU %", "GPU ms",
              "GPU %");
  for (const auto& [name, count] : calls) {
    const double cpu_ms = count * g_results[name].cpu_ms;
    const double gpu_ms = count * g_results[name].gpu_ms;
    std::printf("%-22s %6d %12.3f %6.1f%% %12.3f %6.1f%%\n", name.c_str(), count, cpu_ms,
                100 * cpu_ms / cpu_total, gpu_ms, 100 * gpu_ms / gpu_total);
  }
  std::printf("%-22s %6s %12.3f %7s %12.3f\n", "TOTAL", "", cpu_total, "", gpu_total);
  std::printf("=> CPU ~%.1f tokens/s, GPU ~%.1f tokens/s (%.1fx faster)\n", 1000.0 / cpu_total,
              1000.0 / gpu_total, cpu_total / gpu_total);
}

}  // namespace

int main(int argc, char** argv) {
  app::configure_cpu_threads();
  llama3::ModelConfig c;
  const std::string model = app::get_arg(argc, argv, "--model", "1b");
  if (!llama3::config_by_name(model, &c)) {
    std::fprintf(stderr, "unknown --model '%s' (use tiny, 1b or 8b)\n", model.c_str());
    return 1;
  }
  const int ctx = app::get_int_arg(argc, argv, "--ctx", 512);
  const int prefill = app::get_int_arg(argc, argv, "--prefill", 32);
  g_iters = app::get_int_arg(argc, argv, "--iters", 20);

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("Model shapes : %s (dim=%d hidden=%d heads=%d/%d head_dim=%d vocab=%d)\n",
              c.name.c_str(), c.dim, c.hidden_dim, c.n_heads, c.n_kv_heads, c.head_dim(),
              c.vocab_size);
  std::printf("GPU          : %s\n", prop.name);
  std::printf("CPU threads  : %d (change with OMP_NUM_THREADS=n)\n", app::cpu_thread_count());
  std::printf("Iterations   : %d per kernel (CPU stops early after ~%.0f s)\n\n", g_iters,
              kCpuTimeBudgetMs / 1000);

  std::printf("%-22s %-24s %10s %8s %10s %8s %9s %9s\n", "Kernel", "Shape", "CPU ms",
              "CPU GB/s", "GPU ms", "GPU GB/s", "Speedup", "RelErr");
  std::printf("%s\n", std::string(106, '-').c_str());
  bench_all(c, ctx, prefill);
  print_token_estimate(c, ctx);
  return 0;
}
