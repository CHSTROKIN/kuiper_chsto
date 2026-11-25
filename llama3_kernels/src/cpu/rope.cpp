// =============================================================================
// cpu/rope.cpp  —  Rotary Position Embedding.
//
// Attention compares a query with keys using a dot product. RoPE makes that
// dot product depend on the *distance* between two tokens by rotating q and k
// according to their position.
//
// Inside each head the numbers are grouped in pairs (x0, x1) and every pair is
// rotated like a 2-D point by   angle = pos * inv_freq[i]:
//
//     x0' = x0 * cos(angle) - x1 * sin(angle)
//     x1' = x0 * sin(angle) + x1 * cos(angle)
//
// Pairing ("rotate half", used by HuggingFace Llama checkpoints):
//     head = [ a0 a1 a2 a3 | b0 b1 b2 b3 ]       head_dim = 8, half = 4
//     pairs: (a0,b0) (a1,b1) (a2,b2) (a3,b3)     i.e. element i with i + half
//
// (Meta's original checkpoints pair neighbours (x0,x1),(x2,x3)... instead; the
//  HF conversion script permutes the q/k weights so both give the same result.)
//
// Only q and k are rotated; v is not. Every query head and every key head uses
// the same angles.
// =============================================================================
#include <cmath>

#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

// Rotates the head_dim/2 pairs of one head vector, in place.
static void rotate_one_head(float* head, int head_dim, int pos, const float* inv_freq) {
  const int half = head_dim / 2;
  for (int i = 0; i < half; ++i) {
    const float angle = static_cast<float>(pos) * inv_freq[i];
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    const float x0 = head[i];
    const float x1 = head[i + half];
    head[i] = x0 * c - x1 * s;
    head[i + half] = x0 * s + x1 * c;
  }
}

void rope(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
          const float* inv_freq) {
  for (int h = 0; h < n_heads; ++h) {
    rotate_one_head(q + h * head_dim, head_dim, pos, inv_freq);
  }
  for (int h = 0; h < n_kv_heads; ++h) {
    rotate_one_head(k + h * head_dim, head_dim, pos, inv_freq);
  }
}

}  // namespace cpu
}  // namespace llama3
