#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include <base/tick.h>
#include <cuda_runtime.h> // Include for CUDA types like float4
#include <math.h>         // Include for sqrtf

namespace kernel {

// Use a descriptive name for the common block size
constexpr static int BLOCK_THREADS = 256; 

__device__ void softmax_gpu(float* __restrict__ x, int size) {
    // Optimization: If size is 0 or 1, softmax is trivial, but the loop logic handles size 1.
    if (size <= 0) return;

    int tid = threadIdx.x;
    int step = blockDim.x;

    // --- 1. Find Max Value for Numerical Stability ---
    // Initialize max_val with the first element or -FLT_MAX if out of bounds.
    // This correctly handles the case where tid >= size.
    float max_val = (tid < size) ? x[tid] : -FLT_MAX; 
    
    // Block-strided loop to find the thread's local max
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // CUB BlockReduce to find the max across all threads in the block
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
    // The shared memory usage depends on the block size (thread_num).
    __shared__ BlockReduce::TempStorage temp; 
    __shared__ float shared_val;
    
    // Reduce the local max_val to get the block-wide max
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    
    // Broadcast the block-wide max to all threads via shared memory
    if (tid == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    // --- 2. Calculate Exponential and Sum ---
    float sum = 0.0f;
    // Block-strided loop for element-wise operation
    for (int i = tid; i < size; i += step) {
        // Apply exponential with max subtraction
        float val = x[i];
        x[i] = expf(val - max_val);
        sum += x[i];
    }
    
    // CUB BlockReduce to find the sum across all threads
    sum = BlockReduce(temp).Sum(sum);
    
    // Broadcast the block-wide sum to all threads
    if (tid == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // --- 3. Normalize (Divide by Sum) ---
    // Optimization: Calculate inverse once to replace division with multiplication
    // This is a common micro-optimization on GPUs.
    float inv_sum = 1.0f / sum; 
    
    // Block-strided loop for final normalization
    for (int i = tid; i < size; i += step) {
        x[i] *= inv_sum; // Use multiplication instead of division
    }
}

// ---
// Multi-Head Attention Kernel
// ---

/**
 * @brief CUDA kernel for single-head attention calculation (used across a whole block).
 * * Each block handles one attention head. Calculates $Score = Q \cdot K^T$ 
 * and $Output = Score \cdot V$.
 */
__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
    
    // Block index corresponds to the attention head index
    const int head = blockIdx.x;
    if (head >= head_num) {
        return;
    }

    // Shared memory for the current query head vector
    extern __shared__ float s_query_head[]; 
    
    const float scale = 1.0f / sqrtf(static_cast<float>(head_size));
    // Pointer to the start of the current head's query vector
    float* query_head = query + head * head_size; 

    // --- 1. Preload Query Vector to Shared Memory ---
    // Coalesced access for query
    const int tid = threadIdx.x;
    const int step = blockDim.x;

    for (int i = tid; i < head_size; i += step) {
        s_query_head[i] = query_head[i];
    }
    __syncthreads();

    float* score_head = score_ptr + head * seq_len;
    
    // Index into the Key/Value cache. Handles Grouped-Query Attention (GQA).
    // The key/value head index is head / kv_mul.
    const int head_offset = (head / kv_mul) * head_size;
    
    // --- 2. Compute Attention Scores (Query * Key^T) ---
    // Loop over the sequence history (from token 0 up to 'pos')
    // Loop is strided by blockDim.x to distribute the work among threads.
    for (int t = tid; t <= pos; t += step) {
        // Calculate the address for the key vector at timestep 't'
        float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
        
        float score = 0.0f;
        // Vectorized dot product (unroll by 4 using float4)
        // Assumes head_size is a multiple of 4 and memory is aligned.
        #pragma unroll
        for (int i = 0; i < head_size; i += 4) {
            // Load key from global memory (potential cache miss)
            float4 key_val = *reinterpret_cast<float4*>(key_head + i); 
            // Load query from shared memory (fast access)
            float4 query_val = *reinterpret_cast<float4*>(s_query_head + i); 

            // Fused Multiply-Add (FMA) for dot product contribution
            score += key_val.x * query_val.x + 
                     key_val.y * query_val.y + 
                     key_val.z * query_val.z +
                     key_val.w * query_val.w;
        }

        score *= scale;
        score_head[t] = score; // Write score to global memory
    }
    __syncthreads();

    // --- 3. Softmax on Scores ---
    // Softmax is applied across the sequence dimension (size = pos + 1)
    softmax_gpu(score_head, pos + 1); 
    __syncthreads();

    // --- 4. Compute Weighted Sum (Score * Value) ---
    float* output_head = output + head * head_size;
    
    for (int i = tid; i < head_size; i += step) {
        float value = 0.0f;
        
        for (int t = 0; t <= pos; t++) {
            float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
            float score = score_head[t]; 
            value += score * value_head[i]; 
        }
        output_head[i] = value; // Write final output to global memory
    }
}

// ---
// Host Wrapper Function
// ---

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
    UNUSED(device_type); // Not used, keep for signature

    // Calculate the offset for the current layer's key/value cache data
    const int32_t layer_offset = layer_index * seq_len * kv_dim;
    float* query = const_cast<float*>(query_tensor.ptr<float>());
    float* score = const_cast<float*>(score_tensor.ptr<float>());
    float* output = const_cast<float*>(mha_out.ptr<float>());
    float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
    float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

    // Get CUDA stream from config
    cudaStream_t stream = config->stream;
    multi_head_attention_kernel<<<head_num, BLOCK_THREADS, head_size * sizeof(float), stream>>>(
        pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
        head_size, layer_offset);
}

} // namespace kernel