// =============================================================================
// cuda_utils.h  —  Small helpers used by the CUDA kernels and the programs.
//
//   CUDA_CHECK(call)   abort with a readable message if a CUDA call fails
//   ceil_div(a, b)     how many blocks of size b are needed to cover a items
//   DeviceBuffer<T>    owns a cudaMalloc'ed array and frees it automatically
// =============================================================================
#pragma once

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t err_ = (call);                                                    \
    if (err_ != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error at %s:%d\n  %s\n  -> %s\n", __FILE__,      \
                   __LINE__, #call, cudaGetErrorString(err_));                    \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

namespace llama3 {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// A GPU array of `count` elements of type T. Not copyable (one owner), movable.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) { allocate(count); }
  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(size_t count) {
    release();
    count_ = count;
    if (count > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
  }

  // CPU -> GPU
  void upload(const T* host, size_t count) {
    if (count != count_) allocate(count);
    CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }
  void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }

  // GPU -> CPU
  std::vector<T> download() const {
    std::vector<T> host(count_);
    CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return count_; }

 private:
  void release() {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }

  T* ptr_ = nullptr;
  size_t count_ = 0;
};

}  // namespace llama3
