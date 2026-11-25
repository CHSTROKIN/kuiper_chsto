// =============================================================================
// cpu/argmax.cpp  —  Greedy sampling: pick the token with the highest logit.
// =============================================================================
#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

int argmax(const float* x, int n) {
  int best_index = 0;
  float best_value = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > best_value) {  // strict '>' keeps the first (smallest) index on ties
      best_value = x[i];
      best_index = i;
    }
  }
  return best_index;
}

}  // namespace cpu
}  // namespace llama3
