//
// Created by 18446 on 2025/9/20.
//
#include <iostream>
#include <cuda_runtime.h>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Block Dimensions: " << prop.maxThreadsDim[0] << "x" << prop.maxThreadsDim[1] << "x" << prop.maxThreadsDim[2] << std::endl;
    std::cout << "  Max Grid Dimensions: " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << "x" << prop.maxGridSize[2] << std::endl;
    std::cout << "  Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "  Registers Per Block: " << prop.regsPerBlock << std::endl;
  }

  return 0;
}