// =============================================================================
// test_kernels.cpp  —  Correctness tests for every kernel.
//
// Part 1: CPU kernels against answers computed by hand.
// Part 2: CUDA kernels against the CPU kernels on random data, using
//         realistic and awkward sizes (odd lengths, not multiples of 256, ...).
//
// Exit code 0 = all tests passed.
// =============================================================================
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
using std::vector;

static int g_passed = 0;
static int g_failed = 0;

static void report(bool ok, const std::string& name, const std::string& detail = "") {
  std::printf("[%s] %-45s %s\n", ok ? " OK " : "FAIL", name.c_str(), detail.c_str());
  (ok ? g_passed : g_failed)++;
}

static bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }

// Compares a CPU result with a GPU result (relative to the largest value).
static void expect_close(const std::string& name, const vector<float>& cpu,
                         const vector<float>& gpu, float tolerance = 1e-4f) {
  const float err = app::relative_error(cpu, gpu);
  char detail[64];
  std::snprintf(detail, sizeof(detail), "rel. error %.2e", err);
  report(err <= tolerance, name, detail);
}

// =============================================================================
// Part 1: hand-computed answers for the CPU kernels
// =============================================================================
static void test_cpu_known_answers() {
  std::printf("\n--- CPU kernels vs. hand-computed answers ---\n");
  using namespace llama3;

  {  // embedding: row 2 of a 3x2 table
    const vector<float> table = {0, 1, 10, 11, 20, 21};
    vector<float> out(2);
    cpu::embedding(table.data(), 2, 3, 2, out.data());
    report(out[0] == 20 && out[1] == 21, "cpu::embedding picks the right row");
  }
  {  // rmsnorm of [3, 4]: mean of squares = 12.5, rms = 3.5355
    const vector<float> x = {3, 4}, w = {1, 2};
    vector<float> out(2);
    cpu::rmsnorm(x.data(), w.data(), 2, 0.0f, out.data());
    const float rms = std::sqrt(12.5f);
    report(near(out[0], 3 / rms) && near(out[1], 2 * 4 / rms), "cpu::rmsnorm [3,4]");
  }
  {  // matmul: [1 2] x [[1 0] [0 1] [1 1]]^T = [1 2 3]
    const vector<float> x = {1, 2}, w = {1, 0, 0, 1, 1, 1};
    vector<float> out(3);
    cpu::matmul(x.data(), w.data(), 1, 3, 2, out.data());
    report(out[0] == 1 && out[1] == 2 && out[2] == 3, "cpu::matmul 1x2 * (3x2)^T");
  }
  {  // rope at position 0 is the identity (angle 0)
    vector<float> q = {1, 2, 3, 4}, k = {5, 6, 7, 8};
    const vector<float> inv_freq = {1.0f, 0.5f};
    cpu::rope(q.data(), k.data(), 1, 1, 4, 0, inv_freq.data());
    report(q == vector<float>({1, 2, 3, 4}) && k == vector<float>({5, 6, 7, 8}),
           "cpu::rope pos=0 changes nothing");
  }
  {  // rope rotates the pair (1, 0) by angle pos*inv_freq = 2*0.5 = 1 rad
    vector<float> q = {1, 0}, k = {0, 1};
    const vector<float> inv_freq = {0.5f};
    cpu::rope(q.data(), k.data(), 1, 1, 2, 2, inv_freq.data());
    const bool q_ok = near(q[0], std::cos(1.0f)) && near(q[1], std::sin(1.0f));
    const bool k_ok = near(k[0], -std::sin(1.0f)) && near(k[1], std::cos(1.0f));
    report(q_ok && k_ok, "cpu::rope rotates a pair by pos*inv_freq");
  }
  {  // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]; huge values must not overflow
    vector<float> x = {1, 2, 3, 1000, 1000, 1000};
    cpu::softmax(x.data(), 2, 3);
    const bool row0 = near(x[0], 0.09003057f) && near(x[1], 0.24472847f) &&
                      near(x[2], 0.66524096f);
    const bool row1 = near(x[3], 1 / 3.0f) && near(x[4], 1 / 3.0f) && near(x[5], 1 / 3.0f);
    report(row0 && row1, "cpu::softmax (incl. overflow safety)");
  }
  {  // attention with ONE cached token: weight 1 -> output = v
    const vector<float> q = {1, 2}, k = {3, 4}, v = {5, 6};
    vector<float> scores(1), out(2);
    cpu::attention(q.data(), k.data(), v.data(), 1, 1, 2, 1, scores.data(), out.data());
    report(near(scores[0], 1.0f) && out[0] == 5 && out[1] == 6,
           "cpu::attention single token returns v");
  }
  {  // attention, two identical keys -> equal weights -> average of the values
    const vector<float> q = {1, 1}, k = {1, 0, 1, 0}, v = {2, 4, 6, 8};
    vector<float> scores(2), out(2);
    cpu::attention(q.data(), k.data(), v.data(), 1, 1, 2, 2, scores.data(), out.data());
    report(near(out[0], 4) && near(out[1], 6), "cpu::attention averages equal matches");
  }
  {  // swiglu: silu(0) = 0, silu(large) ~ large
    const vector<float> gate = {0, 20}, up = {5, 2};
    vector<float> out(2);
    cpu::swiglu(gate.data(), up.data(), 2, out.data());
    report(out[0] == 0 && near(out[1], 40, 1e-3f), "cpu::swiglu");
  }
  {
    const vector<float> a = {1, 2}, b = {10, 20};
    vector<float> out(2);
    cpu::add(a.data(), b.data(), 2, out.data());
    report(out[0] == 11 && out[1] == 22, "cpu::add");
  }
  {  // argmax returns the FIRST maximum
    const vector<float> x = {1, 7, 3, 7};
    report(cpu::argmax(x.data(), 4) == 1, "cpu::argmax first of equal maxima");
  }
  {  // rope frequencies: first is 1, scaling leaves the highest frequency alone
    RopeScaling none;
    RopeScaling scaled;
    scaled.enabled = true;
    scaled.factor = 8.0f;
    const auto plain = rope_inv_freq(128, 500000.0f, none);
    const auto llama31 = rope_inv_freq(128, 500000.0f, scaled);
    const bool first_ok = near(plain[0], 1.0f) && near(llama31[0], 1.0f);
    const bool last_ok = near(llama31.back(), plain.back() / 8.0f, 1e-9f);
    report(first_ok && last_ok, "rope_inv_freq (+ Llama 3.1 scaling)");
  }
}

// =============================================================================
// Part 2: CUDA vs CPU on random data
// =============================================================================
static void test_embedding(int vocab, int dim) {
  const auto table = app::random_vector(static_cast<size_t>(vocab) * dim, -1, 1, 1);
  const int token = vocab - 1;  // last row: catches indexing mistakes

  vector<float> cpu_out(dim);
  llama3::cpu::embedding(table.data(), token, vocab, dim, cpu_out.data());

  DeviceBuffer<float> d_table, d_out(dim);
  d_table.upload(table);
  llama3::cuda::embedding(d_table.get(), token, vocab, dim, d_out.get());
  expect_close("embedding vocab=" + std::to_string(vocab) + " dim=" + std::to_string(dim),
               cpu_out, d_out.download(), 0.0f);
}

static void test_rmsnorm(int dim) {
  const auto x = app::random_vector(dim, -2, 2, 2);
  const auto w = app::random_vector(dim, 0.5f, 1.5f, 3);
  vector<float> cpu_out(dim);
  llama3::cpu::rmsnorm(x.data(), w.data(), dim, 1e-5f, cpu_out.data());

  DeviceBuffer<float> d_x, d_w, d_out(dim);
  d_x.upload(x);
  d_w.upload(w);
  llama3::cuda::rmsnorm(d_x.get(), d_w.get(), dim, 1e-5f, d_out.get());
  expect_close("rmsnorm dim=" + std::to_string(dim), cpu_out, d_out.download());

  // in place (out == x)
  llama3::cuda::rmsnorm(d_x.get(), d_w.get(), dim, 1e-5f, d_x.get());
  expect_close("rmsnorm in place dim=" + std::to_string(dim), cpu_out, d_x.download());
}

static void test_matmul(int M, int N, int K) {
  const auto x = app::random_vector(static_cast<size_t>(M) * K, -1, 1, 4);
  const auto w = app::random_vector(static_cast<size_t>(N) * K, -1, 1, 5);
  vector<float> cpu_out(static_cast<size_t>(M) * N);
  llama3::cpu::matmul(x.data(), w.data(), M, N, K, cpu_out.data());

  DeviceBuffer<float> d_x, d_w, d_out(cpu_out.size());
  d_x.upload(x);
  d_w.upload(w);
  llama3::cuda::matmul(d_x.get(), d_w.get(), M, N, K, d_out.get());
  expect_close("matmul M=" + std::to_string(M) + " N=" + std::to_string(N) +
                   " K=" + std::to_string(K),
               cpu_out, d_out.download());
}

static void test_rope(int n_heads, int n_kv_heads, int head_dim, int pos) {
  llama3::RopeScaling scaling;
  scaling.enabled = true;
  const auto inv_freq = llama3::rope_inv_freq(head_dim, 500000.0f, scaling);
  auto q = app::random_vector(static_cast<size_t>(n_heads) * head_dim, -1, 1, 6);
  auto k = app::random_vector(static_cast<size_t>(n_kv_heads) * head_dim, -1, 1, 7);

  DeviceBuffer<float> d_q, d_k, d_freq;
  d_q.upload(q);
  d_k.upload(k);
  d_freq.upload(inv_freq);

  llama3::cpu::rope(q.data(), k.data(), n_heads, n_kv_heads, head_dim, pos, inv_freq.data());
  llama3::cuda::rope(d_q.get(), d_k.get(), n_heads, n_kv_heads, head_dim, pos, d_freq.get());
  const std::string shape = " heads=" + std::to_string(n_heads) + "/" +
                            std::to_string(n_kv_heads) + " pos=" + std::to_string(pos);
  expect_close("rope q" + shape, q, d_q.download());
  expect_close("rope k" + shape, k, d_k.download());
}

static void test_softmax(int rows, int cols) {
  auto x = app::random_vector(static_cast<size_t>(rows) * cols, -10, 10, 8);
  DeviceBuffer<float> d_x;
  d_x.upload(x);
  llama3::cpu::softmax(x.data(), rows, cols);
  llama3::cuda::softmax(d_x.get(), rows, cols);
  expect_close("softmax " + std::to_string(rows) + "x" + std::to_string(cols), x,
               d_x.download());
}

static void test_attention(int n_heads, int n_kv_heads, int head_dim, int n_tokens) {
  const int kv_dim = n_kv_heads * head_dim;
  const auto q = app::random_vector(static_cast<size_t>(n_heads) * head_dim, -1, 1, 9);
  const auto k_cache = app::random_vector(static_cast<size_t>(n_tokens) * kv_dim, -1, 1, 10);
  const auto v_cache = app::random_vector(static_cast<size_t>(n_tokens) * kv_dim, -1, 1, 11);

  vector<float> scores(static_cast<size_t>(n_heads) * n_tokens), out(q.size());
  llama3::cpu::attention(q.data(), k_cache.data(), v_cache.data(), n_heads, n_kv_heads,
                         head_dim, n_tokens, scores.data(), out.data());

  DeviceBuffer<float> d_q, d_k, d_v, d_scores(scores.size()), d_out(out.size());
  d_q.upload(q);
  d_k.upload(k_cache);
  d_v.upload(v_cache);
  llama3::cuda::attention(d_q.get(), d_k.get(), d_v.get(), n_heads, n_kv_heads, head_dim,
                          n_tokens, d_scores.get(), d_out.get());
  expect_close("attention heads=" + std::to_string(n_heads) + "/" + std::to_string(n_kv_heads) +
                   " hd=" + std::to_string(head_dim) + " tokens=" + std::to_string(n_tokens),
               out, d_out.download());
}

static void test_swiglu(int n) {
  const auto gate = app::random_vector(n, -8, 8, 12);
  const auto up = app::random_vector(n, -2, 2, 13);
  vector<float> cpu_out(n);
  llama3::cpu::swiglu(gate.data(), up.data(), n, cpu_out.data());

  DeviceBuffer<float> d_gate, d_up, d_out(n);
  d_gate.upload(gate);
  d_up.upload(up);
  llama3::cuda::swiglu(d_gate.get(), d_up.get(), n, d_out.get());
  expect_close("swiglu n=" + std::to_string(n), cpu_out, d_out.download());
}

static void test_add(int n) {
  const auto a = app::random_vector(n, -1, 1, 14);
  const auto b = app::random_vector(n, -1, 1, 15);
  vector<float> cpu_out(n);
  llama3::cpu::add(a.data(), b.data(), n, cpu_out.data());

  DeviceBuffer<float> d_a, d_b, d_out(n);
  d_a.upload(a);
  d_b.upload(b);
  llama3::cuda::add(d_a.get(), d_b.get(), n, d_out.get());
  expect_close("add n=" + std::to_string(n), cpu_out, d_out.download(), 0.0f);
}

static void test_argmax(int n) {
  auto x = app::random_vector(n, -5, 5, 16);
  // plant two equal maxima: both must pick the earlier one
  x[n / 3] = 100.0f;
  x[n - 1] = 100.0f;
  DeviceBuffer<float> d_x;
  d_x.upload(x);
  const int cpu_index = llama3::cpu::argmax(x.data(), n);
  const int gpu_index = llama3::cuda::argmax(d_x.get(), n);
  report(cpu_index == n / 3 && gpu_index == cpu_index, "argmax n=" + std::to_string(n),
         "cpu " + std::to_string(cpu_index) + ", gpu " + std::to_string(gpu_index));
}

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("No CUDA device found; only the CPU tests can run.\n");
    test_cpu_known_answers();
    return g_failed == 0 ? 0 : 1;
  }

  test_cpu_known_answers();

  std::printf("\n--- CUDA kernels vs. CPU kernels (random data) ---\n");
  const llama3::ModelConfig c = llama3::llama3_2_1b_config();

  test_embedding(c.vocab_size, c.dim);
  test_rmsnorm(c.dim);
  test_rmsnorm(1000);  // not a multiple of the block size
  test_matmul(1, c.dim, c.dim);        // q / o projection
  test_matmul(1, c.kv_dim(), c.dim);   // k / v projection
  test_matmul(1, c.hidden_dim, c.dim); // ffn gate / up
  test_matmul(4, 333, 1001);           // several tokens, awkward sizes
  test_rope(c.n_heads, c.n_kv_heads, c.head_dim(), 0);
  test_rope(c.n_heads, c.n_kv_heads, c.head_dim(), 777);
  test_rope(32, 8, 128, 4000);         // Llama 3 8B head size
  test_softmax(1, 1);
  test_softmax(32, 300);
  test_softmax(4, 128256);             // vocabulary-sized row
  test_attention(c.n_heads, c.n_kv_heads, c.head_dim(), 1);
  test_attention(c.n_heads, c.n_kv_heads, c.head_dim(), 300);
  test_attention(32, 8, 128, 1024);    // Llama 3 8B, long context
  test_attention(4, 4, 64, 17);        // plain multi-head attention (no GQA)
  test_swiglu(c.hidden_dim);
  test_swiglu(1001);
  test_add(c.dim);
  test_add(12345);
  test_argmax(c.vocab_size);
  test_argmax(5);

  std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}
