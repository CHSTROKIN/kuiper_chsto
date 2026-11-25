# =============================================================================
# llama3_kernels — readable CPU + CUDA kernels for Llama 3 inference.
#
# Stand-alone project: only needs a C++17 compiler and the CUDA toolkit
# (OpenMP is optional but recommended for a fair CPU comparison).
#
#   cmake -S llama3_kernels -B build_kernels -DCMAKE_BUILD_TYPE=Release
#   cmake --build build_kernels -j
#
# See llama3_kernels/README.md for details.
# =============================================================================
cmake_minimum_required(VERSION 3.18)

# GPU architectures to compile for (must be set before project()).
# 75 = Turing (RTX 20xx), 80 = A100, 86 = Ampere (RTX 30xx), 89 = Ada (RTX 40xx).
# Override with e.g. -DCMAKE_CUDA_ARCHITECTURES=86 to build faster.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

project(llama3_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LLAMA3_NATIVE_CPU "Compile CPU code for this machine's instruction set (-march=native)" ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

# ---- the kernel library ------------------------------------------------------
add_library(llama3_kernels STATIC
  src/rope_frequencies.cpp
  # loading real models
  src/json.cpp
  src/checkpoint.cpp
  src/tokenizer.cpp
  # CPU kernels
  src/cpu/embedding.cpp
  src/cpu/rmsnorm.cpp
  src/cpu/matmul.cpp
  src/cpu/rope.cpp
  src/cpu/softmax.cpp
  src/cpu/attention.cpp
  src/cpu/swiglu.cpp
  src/cpu/add.cpp
  src/cpu/argmax.cpp
  # CUDA kernels
  src/cuda/embedding.cu
  src/cuda/rmsnorm.cu
  src/cuda/matmul.cu
  src/cuda/rope.cu
  src/cuda/softmax.cu
  src/cuda/attention.cu
  src/cuda/swiglu.cu
  src/cuda/add.cu
  src/cuda/argmax.cu
)
target_include_directories(llama3_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llama3_kernels PUBLIC CUDA::cudart)

if(OpenMP_CXX_FOUND)
  target_link_libraries(llama3_kernels PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found: CPU kernels will run on a single thread.")
endif()

if(LLAMA3_NATIVE_CPU)
  target_compile_options(llama3_kernels PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

# ---- programs ----------------------------------------------------------------
foreach(app test_kernels bench_kernels llama3_forward chat)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE llama3_kernels)
endforeach()

enable_testing()
add_test(NAME kernel_correctness COMMAND test_kernels)
