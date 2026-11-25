// =============================================================================
// cpu/add.cpp  —  Residual connection:  out[i] = a[i] + b[i]
//
// Each transformer block adds its result back onto its input:
//     x = x + Attention(RMSNorm(x))
//     x = x + FFN(RMSNorm(x))
// =============================================================================
#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

void add(const float* a, const float* b, int n, float* out) {
  for (int i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

}  // namespace cpu
}  // namespace llama3
