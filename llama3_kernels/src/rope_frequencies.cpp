// =============================================================================
// rope_frequencies.cpp  —  inv_freq table for RoPE (shared by CPU and CUDA).
//
// Plain RoPE (Llama 3.0):
//     inv_freq[i] = 1 / theta^(2i / head_dim),   i = 0 .. head_dim/2 - 1
// Pair 0 rotates fastest (inv_freq = 1 rad per position), the last pair
// rotates slowest.
//
// Llama 3.1 / 3.2 scaling (same as HF `_compute_llama3_parameters`):
//   wavelength = 2*pi / inv_freq = how many positions one full turn takes.
//   * short wavelengths (high frequency, "local" information) -> unchanged
//   * long wavelengths (low frequency, "global" information)  -> divided by factor
//   * in between                                              -> smooth blend
// =============================================================================
#include "llama3/rope_frequencies.h"

#include <cmath>

namespace llama3 {

std::vector<float> rope_inv_freq(int head_dim, float theta, const RopeScaling& scaling) {
  const int half = head_dim / 2;
  std::vector<float> inv_freq(half);

  for (int i = 0; i < half; ++i) {
    const double exponent = (2.0 * i) / head_dim;
    double freq = 1.0 / std::pow(static_cast<double>(theta), exponent);

    if (scaling.enabled) {
      const double pi = 3.14159265358979323846;
      const double old_context = scaling.original_max_position;
      const double low_freq_wavelen = old_context / scaling.low_freq_factor;
      const double high_freq_wavelen = old_context / scaling.high_freq_factor;
      const double wavelen = 2.0 * pi / freq;

      if (wavelen < high_freq_wavelen) {
        // high frequency: keep as is
      } else if (wavelen > low_freq_wavelen) {
        freq = freq / scaling.factor;  // low frequency: slow it down
      } else {
        // medium frequency: interpolate between the two cases
        const double smooth = (old_context / wavelen - scaling.low_freq_factor) /
                              (scaling.high_freq_factor - scaling.low_freq_factor);
        freq = (1.0 - smooth) * freq / scaling.factor + smooth * freq;
      }
    }
    inv_freq[i] = static_cast<float>(freq);
  }
  return inv_freq;
}

}  // namespace llama3
