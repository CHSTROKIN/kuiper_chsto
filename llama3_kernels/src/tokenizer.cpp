// =============================================================================
// tokenizer.cpp  —  Byte-level BPE as used by Llama 3.
//
// Encoding has two steps.
//
// 1. Split the text with the Llama 3 pattern. Tried in this order:
//      (?i:'s|'t|'re|'ve|'m|'ll|'d)         English contractions
//    | [^\r\n\p{L}\p{N}]?\p{L}+             a word, with one leading symbol/space
//    | \p{N}{1,3}                           up to three digits
//    |  ?[^\s\p{L}\p{N}]+[\r\n]*            punctuation
//    | \s*[\r\n]+                           line breaks
//    | \s+(?!\S)                            spaces, leaving one for the next word
//    | \s+
//    std::regex has no \p{...}, so the pattern is written out by hand below.
//
// 2. BPE inside each piece: start with one token per byte, then repeatedly
//    merge the neighbouring pair that comes first in the merge list.
//
// tokenizer.json cannot hold raw bytes, so it spells every byte as a printable
// Unicode character (the GPT-2 "bytes_to_unicode" table). We translate the
// vocabulary back to real bytes while loading.
// =============================================================================
#include "llama3/tokenizer.h"

#include <algorithm>
#include <stdexcept>

#include "llama3/json.h"
#include "unicode_tables.h"

namespace llama3 {
namespace {

// Decodes the UTF-8 character at s[pos] and moves pos past it. Invalid bytes
// are returned one at a time as U+FFFD, so the original bytes are never lost.
uint32_t next_code_point(const std::string& s, size_t& pos) {
  const auto byte = [&](size_t i) { return static_cast<unsigned char>(s[i]); };
  const unsigned char first = byte(pos);
  const int length = first < 0x80          ? 1
                     : (first >> 5) == 0x6  ? 2
                     : (first >> 4) == 0xE  ? 3
                     : (first >> 3) == 0x1E ? 4
                                            : 0;
  if (length == 0 || pos + length > s.size()) {
    ++pos;
    return 0xFFFD;
  }
  uint32_t cp = length == 1 ? first : first & (0x7F >> length);
  for (int i = 1; i < length; ++i) {
    if ((byte(pos + i) & 0xC0) != 0x80) {
      ++pos;
      return 0xFFFD;
    }
    cp = (cp << 6) | (byte(pos + i) & 0x3F);
  }
  pos += length;
  return cp;
}

template <size_t N>
bool in_ranges(uint32_t cp, const unicode::Range (&ranges)[N]) {
  // first range that ends at or after cp
  const auto it = std::lower_bound(
      ranges, ranges + N, cp,
      [](const unicode::Range& r, uint32_t value) { return r.last < value; });
  return it != ranges + N && it->first <= cp;
}

bool is_letter(uint32_t cp) { return in_ranges(cp, unicode::kLetters); }
bool is_number(uint32_t cp) { return in_ranges(cp, unicode::kNumbers); }
bool is_space(uint32_t cp) {  // Unicode "White_Space"
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
         cp == 0x205F || cp == 0x3000;
}
bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }
bool is_other(uint32_t cp) { return !is_space(cp) && !is_letter(cp) && !is_number(cp); }

}  // namespace

std::vector<std::string> Tokenizer::split(const std::string& text) {
  // Decode once; offset[i] is the byte position of character i.
  std::vector<uint32_t> cps;
  std::vector<size_t> offset;
  for (size_t pos = 0; pos < text.size();) {
    offset.push_back(pos);
    cps.push_back(next_code_point(text, pos));
  }
  offset.push_back(text.size());
  const size_t n = cps.size();

  // number of characters starting at `from` that satisfy `pred`
  const auto run = [&](size_t from, bool (*pred)(uint32_t)) {
    size_t end = from;
    while (end < n && pred(cps[end])) ++end;
    return end - from;
  };

  const auto contraction = [&](size_t i) -> size_t {
    if (cps[i] != '\'' || i + 1 >= n) return 0;
    const auto lower = [&](size_t k) -> uint32_t {
      if (k >= n) return 0;
      return cps[k] >= 'A' && cps[k] <= 'Z' ? cps[k] + 32 : cps[k];
    };
    const uint32_t a = lower(i + 1), b = lower(i + 2);
    if (a == 's' || a == 't') return 2;
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e')) return 3;
    if (a == 'm') return 2;
    if (a == 'l' && b == 'l') return 3;
    if (a == 'd') return 2;
    return 0;
  };

  const auto match = [&](size_t i) -> size_t {
    const uint32_t c = cps[i];
    if (const size_t len = contraction(i)) return len;

    // word, optionally with one leading character
    if (is_letter(c)) return run(i, is_letter);
    if (!is_newline(c) && !is_number(c) && i + 1 < n && is_letter(cps[i + 1])) {
      return 1 + run(i + 1, is_letter);
    }

    if (is_number(c)) return std::min<size_t>(run(i, is_number), 3);

    // punctuation, optionally with one leading space, plus trailing newlines
    const size_t start = (c == ' ' && i + 1 < n && is_other(cps[i + 1])) ? i + 1 : i;
    if (is_other(cps[start])) {
      const size_t end = start + run(start, is_other);
      return end + run(end, is_newline) - i;
    }

    // only whitespace is left
    const size_t spaces = run(i, is_space);
    for (size_t k = spaces; k > 0; --k) {
      if (is_newline(cps[i + k - 1])) return k;  // up to the last newline
    }
    if (i + spaces < n && spaces > 1) return spaces - 1;
    return spaces;
  };

  std::vector<std::string> pieces;
  for (size_t i = 0; i < n;) {
    const size_t len = std::max<size_t>(match(i), 1);
    pieces.push_back(text.substr(offset[i], offset[i + len] - offset[i]));
    i += len;
  }
  return pieces;
}

Tokenizer::Tokenizer(const std::string& path) {
  const Json j = Json::parse_file(path);
  const Json& model = j.at("model");

  // The GPT-2 byte <-> character table: printable bytes stand for themselves,
  // all others are moved to U+0100, U+0101, ...
  std::unordered_map<uint32_t, unsigned char> char_to_byte;
  uint32_t moved = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    char_to_byte[printable ? static_cast<uint32_t>(b) : 256 + moved++] =
        static_cast<unsigned char>(b);
  }

  const auto store = [&](int id, std::string bytes) {
    if (id < 0 || id > (1 << 24)) throw std::runtime_error("tokenizer.json: bad token id");
    if (static_cast<size_t>(id) >= bytes_.size()) bytes_.resize(id + 1);
    bytes_[id] = std::move(bytes);
  };

  std::unordered_map<std::string, int> vocab;  // spelling in the file -> id
  for (const auto& entry : model.at("vocab").object) {
    const int id = static_cast<int>(entry.second.as_int());
    std::string raw;
    for (size_t pos = 0; pos < entry.first.size();) {
      const auto it = char_to_byte.find(next_code_point(entry.first, pos));
      if (it == char_to_byte.end()) {
        throw std::runtime_error("tokenizer.json: not a byte-level vocabulary");
      }
      raw += static_cast<char>(it->second);
    }
    vocab.emplace(entry.first, id);
    store(id, std::move(raw));
  }

  byte_ids_.fill(-1);
  for (const auto& entry : vocab) {
    const std::string& raw = bytes_[entry.second];
    if (raw.size() == 1) byte_ids_[static_cast<unsigned char>(raw[0])] = entry.second;
  }
  if (std::count(byte_ids_.begin(), byte_ids_.end(), -1) != 0) {
    throw std::runtime_error("tokenizer.json: vocabulary does not cover all 256 bytes");
  }

  // Each merge is "left right" (older files) or ["left", "right"].
  int rank = 0;
  for (const Json& m : model.at("merges").array) {
    std::string left, right;
    if (m.type == Json::Type::kArray && m.array.size() == 2) {
      left = m.array[0].as_string();
      right = m.array[1].as_string();
    } else {
      const std::string& both = m.as_string();
      const size_t space = both.find(' ');
      if (space == std::string::npos) throw std::runtime_error("tokenizer.json: bad merge");
      left = both.substr(0, space);
      right = both.substr(space + 1);
    }
    const auto l = vocab.find(left), r = vocab.find(right), merged = vocab.find(left + right);
    if (l == vocab.end() || r == vocab.end() || merged == vocab.end()) {
      throw std::runtime_error("tokenizer.json: merge refers to an unknown token");
    }
    const uint64_t key = (static_cast<uint64_t>(l->second) << 32) | r->second;
    merges_.emplace(key, Merge{rank++, merged->second});
  }

  for (const Json& token : j.at("added_tokens").array) {
    const int id = static_cast<int>(token.at("id").as_int());
    specials_.emplace(token.at("content").as_string(), id);
    store(id, token.at("content").as_string());
  }
}

void Tokenizer::encode_piece(const std::string& piece, std::vector<int>& out) const {
  std::vector<int> ids;
  ids.reserve(piece.size());
  for (const char c : piece) ids.push_back(byte_ids_[static_cast<unsigned char>(c)]);

  while (ids.size() > 1) {
    const Merge* best = nullptr;
    size_t best_at = 0;
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
      const auto it = merges_.find((static_cast<uint64_t>(ids[i]) << 32) | ids[i + 1]);
      if (it != merges_.end() && (!best || it->second.rank < best->rank)) {
        best = &it->second;
        best_at = i;
      }
    }
    if (!best) break;
    ids[best_at] = best->result;
    ids.erase(ids.begin() + best_at + 1);
  }
  out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> out;
  for (const std::string& piece : split(text)) encode_piece(piece, out);
  return out;
}

const std::string& Tokenizer::token_bytes(int id) const {
  static const std::string empty;
  // the model's vocabulary can be larger than the tokenizer's
  if (id < 0 || static_cast<size_t>(id) >= bytes_.size()) return empty;
  return bytes_[id];
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  for (const int id : ids) out += token_bytes(id);
  return out;
}

int Tokenizer::special(const std::string& name) const {
  const auto it = specials_.find(name);
  if (it == specials_.end()) throw std::runtime_error("tokenizer has no token " + name);
  return it->second;
}

}  // namespace llama3
