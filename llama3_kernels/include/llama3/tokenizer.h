// =============================================================================
// tokenizer.h  —  The Llama 3 tokenizer (byte-level BPE), read from the
// tokenizer.json of a Hugging Face model folder.
// =============================================================================
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama3 {

class Tokenizer {
 public:
  // path = .../tokenizer.json. Throws std::runtime_error on problems.
  explicit Tokenizer(const std::string& path);

  // Ordinary text -> token ids. Special tokens such as "<|eot_id|>" that
  // appear in the text are NOT recognised: they are encoded as plain
  // characters, so a user's message cannot fake the chat structure.
  std::vector<int> encode(const std::string& text) const;

  // The raw bytes of one token. A single token may hold only part of a UTF-8
  // character. Special tokens give their name.
  const std::string& token_bytes(int id) const;

  std::string decode(const std::vector<int>& ids) const;

  // Id of a special token, e.g. "<|eot_id|>". Throws if it does not exist.
  int special(const std::string& name) const;

  // Splits text into the pieces that BPE works on (public for testing).
  static std::vector<std::string> split(const std::string& text);

 private:
  void encode_piece(const std::string& piece, std::vector<int>& out) const;

  struct Merge {
    int rank;    // smaller = merged earlier
    int result;  // id of the merged token
  };

  std::vector<std::string> bytes_;                  // id -> raw bytes
  std::array<int, 256> byte_ids_{};                 // single byte -> id
  std::unordered_map<uint64_t, Merge> merges_;      // (left id, right id) -> merge
  std::unordered_map<std::string, int> specials_;  // name -> id
};

}  // namespace llama3
