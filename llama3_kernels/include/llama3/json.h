// =============================================================================
// json.h  —  A very small JSON reader.
//
// A Hugging Face model folder describes itself with JSON (config.json,
// tokenizer.json and the header inside *.safetensors). This reader is just big
// enough for those files, so the module needs no third-party library.
// =============================================================================
#pragma once

#include <map>
#include <string>
#include <vector>

namespace llama3 {

struct Json {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type = Type::kNull;
  bool boolean = false;
  double number = 0;
  std::string string;  // UTF-8
  std::vector<Json> array;
  std::map<std::string, Json> object;

  bool is_null() const { return type == Type::kNull; }
  bool has(const std::string& key) const { return object.count(key) != 0; }

  // Member of an object. Throws std::runtime_error if it is missing.
  const Json& at(const std::string& key) const;

  // Typed access. Each throws std::runtime_error if the type is wrong.
  const std::string& as_string() const;
  double as_number() const;
  long long as_int() const { return static_cast<long long>(as_number()); }

  // Throws std::runtime_error on malformed input.
  static Json parse(const std::string& text);
  static Json parse_file(const std::string& path);
};

}  // namespace llama3
