// =============================================================================
// cpu/embedding.cpp  —  Token id -> vector.
//
// The embedding table has one row of `dim` floats per token in the vocabulary.
// Looking up a token is just copying its row; there is no arithmetic.
//
//    table (vocab_size x dim)            out (dim)
//    row 0      [ . . . . ]
//    row 1      [ . . . . ]
//    row token  [ a b c d ]  ------->   [ a b c d ]
//    ...
// =============================================================================
#include <cstddef>
#include <stdexcept>
#include <string>

#include "llama3/cpu_kernels.h"

namespace llama3 {
namespace cpu {

void embedding(const float* table, int token, int vocab_size, int dim, float* out) {
  if (token < 0 || token >= vocab_size) {
    throw std::out_of_range("embedding: token id " + std::to_string(token) +
                            " is outside the vocabulary");
  }
  // size_t: vocab_size * dim can be larger than what an int can hold.
  const float* row = table + static_cast<size_t>(token) * dim;
  for (int i = 0; i < dim; ++i) {
    out[i] = row[i];
  }
}

}  // namespace cpu
}  // namespace llama3
