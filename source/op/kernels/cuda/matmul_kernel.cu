#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                     int K) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row = start_row + ROW_PER_BLOCK;

    // Outer loop: Each block computes one or more dot products (rows of the output matrix).
    for (int p = start_row; p < end_row; ++p) {
        if (p >= K) {
            return;
        }

        sdata[tid] = 0.0f;
        int row_offset = p * M;

        constexpr int pack_size = 4;
        const int pack_num = M / pack_size;
        const int pack_off = pack_size * pack_num;

        // Vectorized access using float4 for efficient memory bandwidth utilization.
        const float4* input_float4_ptr = reinterpret_cast<const float4*>(input);
        const float4* weight_float4_ptr = reinterpret_cast<const float4*>(weight + row_offset);

        // Inner loop (Vectorized part): Grid-stride loop pattern for load balancing across threads.
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 input_float4 = input_float4_ptr[i];
            float4 weight_float4 = weight_float4_ptr[i];
            float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                             input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
            sdata[tid] += part_sum;
        }

        // Inner loop (Remainder part): Scalar access for elements not divisible by 4.
        for (int i = pack_off + tid; i < M; i += blockDim.x) {
            sdata[tid] += input[i] * weight[row_offset + i];
        }

        __syncthreads();

        // Block-level Reduction: Uses CUB library for fast parallel sum of partial results.
        using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;
        float final_sum = BlockReduce(temp).Sum(sdata[tid]);
        
        // Final result written by thread 0 after reduction is complete.
        if (tid == 0) {
            output[p] = final_sum;
        }
    }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                         const float* scales, const int32_t group_size,
                                         float* output, int M, int K) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row = start_row + ROW_PER_BLOCK;

    for (int p = start_row; p < end_row; ++p) {
        if (p >= K) {
            return;
        }

        sdata[tid] = 0.0f;
        const int row_offset = p * M;

        // Inner loop: Grid-stride loop for the quantized dot product.
        for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
            const int weight_idx = row_offset + i;
            const int group_idx = weight_idx / group_size;
            
            // De-quantization and accumulation (value = input * (weight_i8 * scale)).
            sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
        }
        __syncthreads();

        // Block-level Reduction: Sums the thread partial results.
        using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;
        float final_sum = BlockReduce(temp).Sum(sdata[tid]);
        
        if (tid == 0) {
            output[p] = final_sum;
        }
    }
}

// --- Host Interface Functions ---

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
    // Parameter validation checks omitted for brevity.
    const int32_t K = weight.get_dim(0); 
    const int32_t M = weight.get_dim(1); 

    constexpr int THREAD_PER_BLOCK = 128;
    constexpr int ROW_PER_BLOCK = 1;

    // Kernel launch: K blocks, 128 threads/block. One block computes one row (dot product).
    if (config && config->stream) {
        matmul_kernel_cu_fp32<THREAD_PER_BLOCK, ROW_PER_BLOCK><<<K, THREAD_PER_BLOCK, 0, config->stream>>>(
            input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    } else {
        matmul_kernel_cu_fp32<THREAD_PER_BLOCK, ROW_PER_BLOCK><<<K, THREAD_PER_BLOCK>>>(
            input.ptr<float>(), weight.ptr<float>(),
            const_cast<float*>(output.ptr<float>()), M, K);
    }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
    // Parameter validation checks omitted for brevity.
    const int32_t K = weight.get_dim(0);
    const int32_t M = weight.get_dim(1);
    
    constexpr int THREAD_PER_BLOCK = 128;
    constexpr int ROW_PER_BLOCK = 1;

    // Kernel launch: K blocks, 128 threads/block for quantized matrix multiplication.
    if (config->stream) {
        matmul_kernel_cu_fp32int8<THREAD_PER_BLOCK, ROW_PER_BLOCK><<<K, THREAD_PER_BLOCK, 0, config->stream>>>(
            input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
            const_cast<float*>(output.ptr<float>()), M, K);
    } else {
        matmul_kernel_cu_fp32int8<THREAD_PER_BLOCK, ROW_PER_BLOCK><<<K, THREAD_PER_BLOCK>>>(
            input.ptr<float>(), weight.ptr<int8_t>(),
            scale.ptr<float>(), group_size,
            const_cast<float*>(output.ptr<float>()), M, K);
    }
}
}  // namespace kernel

#ifndef MATMUL_KERNEL_CU_CUH
#define MATMUL_KERNEL_CU_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config = nullptr);
}  // namespace kernel

#endif  // MATMUL_KERNEL_CU_CUH