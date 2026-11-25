// =============================================================================
// cpu/swiglu.cpp  —  Activation of the feed-forward block.
//
// Llama's feed-forward block is
//     FFN(x) = W2 * ( silu(W1 * x)  (*)  (W3 * x) )        (*) = element-wise
// W1 produces the "gate", W3 the "up" projection. This kernel computes the
// middle part:
//     out[i] = silu(gate[i]) * up[i],      silu(z) = z * sigmoid(z) = z / (1 + e^-z)
// =============================================================================
#include <cmath>

#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

void swiglu(const float* gate, const float* up, int n, float* out) {
  for (int i = 0; i < n; ++i) {
    const float g = gate[i];
    const float silu = g / (1.0f + std::exp(-g));
    out[i] = silu * up[i];
  }
}

}  // namespace cpu
}  // namespace llama3
