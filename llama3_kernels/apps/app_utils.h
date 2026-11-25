// =============================================================================
// app_utils.h  —  Helpers shared by test_kernels, bench_kernels, llama3_forward:
// random numbers, result comparison, timers and tiny command-line parsing.
// =============================================================================
#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "llama3/cuda_utils.h"

namespace app {

// ---- random numbers ----------------------------------------------------------
// xorshift64: a very small, fast and deterministic generator. Good enough for
// filling test tensors; much faster than std::mt19937 for billions of floats.
class Random {
 public:
  // The seed is scrambled with splitmix64 first: xorshift is linear, so the
  // streams of nearby seeds (1, 2, 3, ...) would otherwise be correlated.
  explicit Random(uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state_ = (z ^ (z >> 31)) | 1;  // xorshift state must not be 0
  }

  uint64_t next_u64() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return state_;
  }
  // uniform in [lo, hi)
  float uniform(float lo, float hi) {
    const float unit = static_cast<float>(next_u64() >> 40) / static_cast<float>(1ULL << 24);
    return lo + (hi - lo) * unit;
  }
  int integer(int lo, int hi) {  // uniform in [lo, hi)
    return lo + static_cast<int>(next_u64() % static_cast<uint64_t>(hi - lo));
  }

 private:
  uint64_t state_;
};

inline void fill_uniform(std::vector<float>& v, float lo, float hi, uint64_t seed) {
  Random rng(seed);
  for (float& x : v) x = rng.uniform(lo, hi);
}

inline std::vector<float> random_vector(size_t n, float lo, float hi, uint64_t seed) {
  std::vector<float> v(n);
  fill_uniform(v, lo, hi, seed);
  return v;
}

// ---- comparing CPU and GPU results ---------------------------------------
// The CPU and GPU add numbers in a different order, and float addition is not
// associative, so results differ in the last few bits. We therefore compare
// with a tolerance instead of exact equality.
inline float max_abs_diff(const float* a, const float* b, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

inline float max_abs_value(const float* a, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i]));
  return m;
}

// Error relative to the size of the reference values (avoids huge "relative
// errors" for values that are almost zero).
inline float relative_error(const std::vector<float>& reference, const std::vector<float>& other) {
  const float scale = std::max(max_abs_value(reference.data(), reference.size()), 1e-6f);
  return max_abs_diff(reference.data(), other.data(), reference.size()) / scale;
}

// ---- timers ------------------------------------------------------------------
class CpuTimer {
 public:
  void start() { begin_ = std::chrono::steady_clock::now(); }
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point begin_;
};

// Measures GPU time with CUDA events. Events are recorded into the GPU's work
// queue, so the measured time covers exactly the kernels launched in between,
// independent of when the CPU-side launch calls return.
class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  double stop_ms() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{}, stop_{};
};

// ---- command line: --name value ------------------------------------------
inline std::string get_arg(int argc, char** argv, const std::string& name,
                           const std::string& default_value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) return argv[i + 1];
  }
  return default_value;
}

inline int get_int_arg(int argc, char** argv, const std::string& name, int default_value) {
  return std::atoi(get_arg(argc, argv, name, std::to_string(default_value)).c_str());
}

inline bool has_flag(int argc, char** argv, const std::string& name) {
  for (int i = 1; i < argc; ++i) {
    if (name == argv[i]) return true;
  }
  return false;
}

// Number of threads the CPU kernels will use.
inline int cpu_thread_count() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

// Picks the CPU thread count. If OMP_NUM_THREADS is set, it is respected.
// Otherwise we use half of the logical CPUs (= the physical cores on machines
// with hyper-threading / SMT): the kernels are limited by memory bandwidth, so
// extra hyper-threads do not help, and with very short parallel loops every
// thread waits at the end of the loop for the slowest one. Measured on an
// 8-core / 16-thread laptop under WSL2, 16 threads were up to 3x SLOWER than 8.
inline void configure_cpu_threads() {
#ifdef _OPENMP
  if (std::getenv("OMP_NUM_THREADS") == nullptr) {
    omp_set_num_threads(std::max(1, omp_get_num_procs() / 2));
  }
#endif
}

}  // namespace app
