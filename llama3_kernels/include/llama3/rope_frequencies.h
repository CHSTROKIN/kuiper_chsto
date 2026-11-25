// =============================================================================
// rope_frequencies.h  —  Pre-computes the rotation speed of every RoPE pair.
//
// RoPE rotates pairs of numbers inside each attention head by an angle
//     angle(pos, i) = pos * inv_freq[i]          i = 0 .. head_dim/2 - 1
// inv_freq only depends on the model config, so we compute it once on the CPU
// (and copy it to the GPU for the CUDA kernel).
// =============================================================================
#pragma once

#include <vector>

#include "llama3/config.h"

namespace llama3 {

// Returns head_dim / 2 values:
//     inv_freq[i] = 1 / theta^(2i / head_dim)
// and, when scaling.enabled, applies the Llama 3.1 long-context adjustment.
std::vector<float> rope_inv_freq(int head_dim, float theta, const RopeScaling& scaling);

}  // namespace llama3
