#include "rope_kernel.h"
#include <cmath>
#include <algorithm> // For std::min, if used
#include <cstring>   // For std::memcpy, if used

namespace kernel {

// Define base frequencies for clarity and maintainability
constexpr static float ROPE_DEFAULT_BASE = 10000.0f;
constexpr static float ROPE_QWEN_BASE = 1000000.0f;
constexpr static float ROPE_LLAMA3_BASE = 500000.0f;

// Helper function for the RoPE rotation logic
inline void rotate_pair_cpu(float& v0, float& v1, float fcr, float fci) {
    float temp_v0 = v0;
    v0 = temp_v0 * fcr - v1 * fci;
    v1 = temp_v0 * fci + v1 * fcr;
}


void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int head_dim = 0; head_dim < head_size; ++head_dim) {
            float freq =
                1.0f / std::pow(ROPE_LLAMA3_BASE, static_cast<float>(head_dim) / static_cast<float>(head_size));
            float val = static_cast<float>(pos) * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);
            *(sin_cache + pos * head_size + head_dim) = fci;
            *(cos_cache + pos * head_size + head_dim) = fcr;
        }
    }
}
//rope for llama3
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
    UNUSED(stream);
    const int32_t pos = *input_pos.ptr<int32_t>(0);

    float* q_ptr = const_cast<float*>(input_q.ptr<float>());
    float* k_ptr = const_cast<float*>(input_k.ptr<float>());
    const float* sin_ptr = sin_cache.ptr<float>();
    const float* cos_ptr = cos_cache.ptr<float>();
    
    const int32_t half_head_size = head_size / 2;

    // Loop over all heads (i.e., every head_size block of the vector)
    for (int32_t head_offset = 0; head_offset < dim; head_offset += head_size) {
        // Loop over the first half of the head dimension (head_dim = 0 to head_size/2 - 1)
        for (int32_t head_dim = 0; head_dim < half_head_size; head_dim ++) {
            // RoPE factors are typically looked up at index $2i$ (which is $2 \cdot head\_dim$)
            // The original code uses head_dim * 2 directly for lookup
            const float fci = *(sin_ptr + pos * head_size + head_dim * 2);
            const float fcr = *(cos_ptr + pos * head_size + head_dim * 2);

            // Calculate global indices for $v_0$ and $v_1$
            const int32_t v0_idx = head_offset + head_dim;
            const int32_t v1_idx = head_offset + head_dim + half_head_size;

            // Determine if Key needs rotation (GQA/MQA support)
            const int32_t rotn = head_offset < kv_dim ? 2 : 1; 
            
            // Rotate Query (v=0)
            float v0_q = q_ptr[v0_idx];
            float v1_q = q_ptr[v1_idx];
            rotate_pair_cpu(v0_q, v1_q, fcr, fci);
            q_ptr[v0_idx] = v0_q;
            q_ptr[v1_idx] = v1_q;
            
            // Rotate Key (v=1)
            if (rotn == 2) {
                float v0_k = k_ptr[v0_idx];
                float v1_k = k_ptr[v1_idx];
                rotate_pair_cpu(v0_k, v1_k, fcr, fci);
                k_ptr[v0_idx] = v0_k;
                k_ptr[v1_idx] = v1_k;
            }
        }
    }
}
} // namespace kernel