// =============================================================================
// cpu/softmax.cpp  —  Turn arbitrary scores into probabilities (sum = 1).
//
//    softmax(x)[i] = exp(x[i]) / sum_j exp(x[j])
//
// exp() of a large number overflows float (exp(89) = inf), so we first subtract
// the maximum of the row. This does not change the result, because
//    exp(x[i] - max) / sum_j exp(x[j] - max)
//  = exp(x[i]) * exp(-max) / (exp(-max) * sum_j exp(x[j]))
// and now every exponent is <= 0.
// =============================================================================
#include <cmath>

#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

static void softmax_one_row(float* row, int n) {
  // 1. maximum
  float max_value = row[0];
  for (int i = 1; i < n; ++i) {
    if (row[i] > max_value) max_value = row[i];
  }

  // 2. exponentials and their sum.
  // The sum is kept in double: adding 100k+ floats one after another in float
  // precision loses several digits (each addition rounds to ~7 digits).
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    row[i] = std::exp(row[i] - max_value);
    sum += row[i];
  }

  // 3. normalize
  const float inv_sum = static_cast<float>(1.0 / sum);
  for (int i = 0; i < n; ++i) {
    row[i] *= inv_sum;
  }
}

void softmax(float* x, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    softmax_one_row(x + r * cols, cols);
  }
}

}  // namespace cpu
}  // namespace llama3
