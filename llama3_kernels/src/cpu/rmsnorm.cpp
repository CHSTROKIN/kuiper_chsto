// =============================================================================
// cpu/rmsnorm.cpp  —  Root-Mean-Square normalization.
//
//    rms     = sqrt( (x[0]^2 + x[1]^2 + ... + x[dim-1]^2) / dim  + eps )
//    out[i]  = x[i] / rms * weight[i]
//
// Unlike LayerNorm, RMSNorm does not subtract the mean and has no bias. It only
// rescales the vector to unit "size" and then multiplies each element by a
// learned weight. Llama applies it before attention, before the feed-forward
// block, and once more before the final output layer.
// =============================================================================
#include <cmath>

#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

void rmsnorm(const float* x, const float* weight, int dim, float eps, float* out) {
  // Step 1: mean of the squares.
  float sum_of_squares = 0.0f;
  for (int i = 0; i < dim; ++i) {
    sum_of_squares += x[i] * x[i];
  }
  const float mean_of_squares = sum_of_squares / dim;

  // Step 2: 1 / rms. (eps avoids dividing by zero for an all-zero vector.)
  const float inv_rms = 1.0f / std::sqrt(mean_of_squares + eps);

  // Step 3: normalize and apply the learned per-element weight.
  // Safe when out == x: every x[i] was already used in step 1.
  for (int i = 0; i < dim; ++i) {
    out[i] = x[i] * inv_rms * weight[i];
  }
}

}  // namespace cpu
}  // namespace llama3
