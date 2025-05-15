#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"
#include <cuda_runtime.h>
#include <math.h>

namespace kernel {

constexpr static int BLOCK_REDUCE_THREADS = 128;
constexpr static int PACK_SIZE = 4;

static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size,
                                             int size, float eps) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_dim_x = blockDim.x;

    if (bid >= dim_size) {
        return;
    }

    float* block_in = in + bid * size;
    float* block_out = out + bid * size;

    const int pack_num = size / PACK_SIZE;
    const int pack_off = PACK_SIZE * pack_num;

    float sum_sq = 0.0f;
    float4* in_pack = reinterpret_cast<float4*>(block_in);

    for (int i = tid; i < pack_num; i += block_dim_x) {
        float4 in_float4 = *(in_pack + i);
        sum_sq += in_float4.x * in_float4.x;
        sum_sq += in_float4.y * in_float4.y;
        sum_sq += in_float4.z * in_float4.z;
        sum_sq += in_float4.w * in_float4.w;
    }

    for (int i = pack_off + tid; i < size; i += block_dim_x) {
        sum_sq += block_in[i] * block_in[i];
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_REDUCE_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    
    sum_sq = BlockReduce(temp).Sum(sum_sq);
    
    if (tid == 0) {
        shared_val = sum_sq;
    }
    __syncthreads();
    sum_sq = shared_val;

    const float scale = rsqrtf(sum_sq / static_cast<float>(size) + eps);

    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(block_out);
    
    for (int i = tid; i < pack_num; i += block_dim_x) {
        float4 in_float4 = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i); 
        
        *(out_pack + i) =
            make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                        scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    }

    for (int i = pack_off + tid; i < size; i += block_dim_x) {
        block_out[i] = block_in[i] * wei[i] * scale;
    }
}

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
    const int tid = threadIdx.x;
    const int block_dim_x = blockDim.x;

    const int pack_num = size / PACK_SIZE;
    const int pack_off = PACK_SIZE * pack_num;

    float sum_sq = 0.0f;
    float4* in_pack = reinterpret_cast<float4*>(in);
    
    for (int i = tid; i < pack_num; i += block_dim_x) {
        float4 in_float4 = *(in_pack + i);
        sum_sq += in_float4.x * in_float4.x;
        sum_sq += in_float4.y * in_float4.y;
        sum_sq += in_float4.z * in_float4.z;
        sum_sq += in_float4.w * in_float4.w;
    }

    for (int i = pack_off + tid; i < size; i += block_dim_x) {
        sum_sq += in[i] * in[i];
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    
    sum_sq = BlockReduce(temp).Sum(sum_sq);
    
    if (tid == 0) {
        shared_val = sum_sq;
    }
    __syncthreads();
    sum_sq = shared_val;

    const float scale = rsqrtf(sum_sq / static_cast<float>(size) + eps);

    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(out);
    
    for (int i = tid; i < pack_num; i += block_dim_x) {
        float4 in_float4 = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i);
        
        *(out_pack + i) =
            make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                        scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    }

    for (int i = pack_off + tid; i < size; i += block_dim_x) {
        out[i] = wei[i] * in[i] * scale;
    }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
          weight.device_type() == base::DeviceType::kDeviceCUDA &&
          output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif
    
    const int32_t size = static_cast<int32_t>(input.size());
    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    
    constexpr int threads_num = BLOCK_REDUCE_THREADS;
    
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32<BLOCK_REDUCE_THREADS><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
        row_rmsnorm_f32<BLOCK_REDUCE_THREADS><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
          weight.device_type() == base::DeviceType::kDeviceCUDA &&
          output.device_type() == base::DeviceType::kDeviceCUDA);

    const float eps = 1e-6f; 
    
    const int32_t total_size = static_cast<int32_t>(input.size());
    const int32_t size = input.get_dim(input.dims_size() - 1);
    const int32_t dim_size = total_size / size; 

    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    
    constexpr int threads_num = BLOCK_REDUCE_THREADS;

    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                                   size, eps);
    } else {
        row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
}
} // namespace kernel