// =============================================================================
// json.cpp  —  Recursive-descent JSON parser (see json.h).
// =============================================================================
#include "llama3/json.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace llama3 {
namespace {

constexpr int kMaxDepth = 64;  // protects the stack against hostile input

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}

  Json parse_document() {
    Json v = parse_value(0);
    skip_space();
    if (pos_ != s_.size()) fail("unexpected text after the value");
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& why) const {
    throw std::runtime_error("JSON error at byte " + std::to_string(pos_) + ": " + why);
  }

  void skip_space() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) {
      ++pos_;
    }
  }

  char peek() {
    skip_space();
    if (pos_ >= s_.size()) fail("unexpected end of input");
    return s_[pos_];
  }

  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }

  bool consume(char c) {
    if (peek() != c) return false;
    ++pos_;
    return true;
  }

  void expect_word(const char* word) {
    for (const char* p = word; *p; ++p, ++pos_) {
      if (pos_ >= s_.size() || s_[pos_] != *p) fail("unknown literal");
    }
  }

  Json parse_value(int depth) {
    if (depth > kMaxDepth) fail("nesting is too deep");
    Json v;
    const char c = peek();
    if (c == '{') {
      ++pos_;
      v.type = Json::Type::kObject;
      if (consume('}')) return v;
      do {
        if (peek() != '"') fail("expected a key");
        std::string key = parse_string();
        expect(':');
        v.object[std::move(key)] = parse_value(depth + 1);
      } while (consume(','));
      expect('}');
    } else if (c == '[') {
      ++pos_;
      v.type = Json::Type::kArray;
      if (consume(']')) return v;
      do {
        v.array.push_back(parse_value(depth + 1));
      } while (consume(','));
      expect(']');
    } else if (c == '"') {
      v.type = Json::Type::kString;
      v.string = parse_string();
    } else if (c == 't') {
      expect_word("true");
      v.type = Json::Type::kBool;
      v.boolean = true;
    } else if (c == 'f') {
      expect_word("false");
      v.type = Json::Type::kBool;
    } else if (c == 'n') {
      expect_word("null");
    } else {
      const char* begin = s_.c_str() + pos_;
      char* end = nullptr;
      v.type = Json::Type::kNumber;
      v.number = std::strtod(begin, &end);
      if (end == begin) fail("unexpected character");
      pos_ += static_cast<size_t>(end - begin);
    }
    return v;
  }

  unsigned parse_hex4() {
    if (pos_ + 4 > s_.size()) fail("short \\u escape");
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[pos_++];
      value <<= 4;
      if (c >= '0' && c <= '9') value |= c - '0';
      else if (c >= 'a' && c <= 'f') value |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') value |= c - 'A' + 10;
      else fail("bad \\u escape");
    }
    return value;
  }

  static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) fail("unterminated string");
      const char c = s_[pos_++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= s_.size()) fail("unterminated escape");
      const char e = s_[pos_++];
      switch (e) {
        case '"': case '\\': case '/': out += e; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned cp = parse_hex4();
          // Two \u escapes (a "surrogate pair") encode characters above U+FFFF.
          if (cp >= 0xD800 && cp <= 0xDBFF && s_.compare(pos_, 2, "\\u") == 0) {
            pos_ += 2;
            const unsigned low = parse_hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("unknown escape");
      }
    }
  }

  const std::string& s_;
  size_t pos_ = 0;
};

}  // namespace

const Json& Json::at(const std::string& key) const {
  const auto it = object.find(key);
  if (it == object.end()) throw std::runtime_error("JSON: missing key \"" + key + "\"");
  return it->second;
}

const std::string& Json::as_string() const {
  if (type != Type::kString) throw std::runtime_error("JSON: value is not a string");
  return string;
}

double Json::as_number() const {
  if (type != Type::kNumber) throw std::runtime_error("JSON: value is not a number");
  return number;
}

Json Json::parse(const std::string& text) { return Parser(text).parse_document(); }

Json Json::parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return parse(text);
}

}  // namespace llama3
