// =============================================================================
// checkpoint.cpp  —  Reading Hugging Face checkpoints (see checkpoint.h).
//
// The safetensors file format is very simple:
//
//     [ 8 bytes: N, little-endian ] [ N bytes: JSON header ] [ raw tensor data ]
//
// The header maps every tensor name to
//     { "dtype": "BF16", "shape": [rows, cols], "data_offsets": [begin, end] }
// with offsets relative to the start of the data section. Matrices are row-major
// and nn.Linear weights are [out_features, in_features], which is exactly the
// layout matmul() expects, so no transposing or reordering is needed.
// =============================================================================
#include "llama3/checkpoint.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "llama3/json.h"

namespace llama3 {
namespace {

class SafetensorsFile {
 public:
  explicit SafetensorsFile(const std::string& path) : path_(path), in_(path, std::ios::binary) {
    if (!in_) throw std::runtime_error("cannot open " + path);
    uint64_t header_size = 0;  // the format is little-endian, like x86
    in_.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!in_ || header_size > (100u << 20)) {
      throw std::runtime_error(path + ": not a safetensors file");
    }
    std::string header(header_size, '\0');
    in_.read(&header[0], static_cast<std::streamsize>(header_size));
    if (!in_) throw std::runtime_error(path + ": truncated header");
    header_ = Json::parse(header);
    data_start_ = sizeof(header_size) + header_size;
  }

  bool contains(const std::string& name) const { return header_.has(name); }

  // Reads one tensor as fp32 and checks that it has the expected shape.
  std::vector<float> read(const std::string& name, const std::vector<long long>& shape) {
    const Json& info = header_.at(name);
    const std::vector<Json>& dims = info.at("shape").array;
    bool same_shape = dims.size() == shape.size();
    size_t count = 1;
    for (size_t i = 0; same_shape && i < shape.size(); ++i) {
      same_shape = dims[i].as_int() == shape[i];
      count *= static_cast<size_t>(shape[i]);
    }
    if (!same_shape) throw std::runtime_error(name + ": shape does not match config.json");

    const std::string& dtype = info.at("dtype").as_string();
    const size_t element_size = dtype == "F32" ? 4 : dtype == "BF16" ? 2 : 0;
    if (element_size == 0) throw std::runtime_error(name + ": unsupported dtype " + dtype);

    const std::vector<Json>& offsets = info.at("data_offsets").array;
    if (offsets.size() != 2 ||
        static_cast<size_t>(offsets[1].as_int() - offsets[0].as_int()) != count * element_size) {
      throw std::runtime_error(name + ": size does not match its shape");
    }
    in_.seekg(static_cast<std::streamoff>(data_start_ + offsets[0].as_int()));

    std::vector<float> out(count);
    if (element_size == 4) {
      in_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * 4));
    } else {
      // BF16 is the upper half of an fp32 number: same sign and exponent bits,
      // fewer mantissa bits. Converting = putting 16 zero bits below it.
      std::vector<uint16_t> chunk(1 << 20);
      for (size_t done = 0; done < count && in_;) {
        const size_t n = std::min(chunk.size(), count - done);
        in_.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(n * 2));
        for (size_t i = 0; i < n; ++i) {
          const uint32_t bits = static_cast<uint32_t>(chunk[i]) << 16;
          std::memcpy(&out[done + i], &bits, sizeof(bits));
        }
        done += n;
      }
    }
    if (!in_) throw std::runtime_error(path_ + ": file is truncated (at " + name + ")");
    return out;
  }

 private:
  std::string path_;
  std::ifstream in_;
  Json header_;
  uint64_t data_start_ = 0;
};

// Finds the file that holds a tensor, for single-file and sharded checkpoints.
class Checkpoint {
 public:
  explicit Checkpoint(const std::string& dir) : dir_(dir) {
    const std::string index_path = dir + "/model.safetensors.index.json";
    if (std::ifstream(index_path)) {
      index_ = Json::parse_file(index_path).at("weight_map");
    } else {
      single_ = &open("model.safetensors");
    }
  }

  bool contains(const std::string& name) const {
    return single_ ? single_->contains(name) : index_.has(name);
  }

  std::vector<float> read(const std::string& name, const std::vector<long long>& shape) {
    if (single_) return single_->read(name, shape);
    return open(index_.at(name).as_string()).read(name, shape);
  }

 private:
  SafetensorsFile& open(const std::string& file_name) {
    // shard names must stay inside the model folder
    if (file_name.find('/') != std::string::npos) {
      throw std::runtime_error("unexpected shard path: " + file_name);
    }
    std::unique_ptr<SafetensorsFile>& slot = files_[file_name];
    if (!slot) slot = std::make_unique<SafetensorsFile>(dir_ + "/" + file_name);
    return *slot;
  }

  std::string dir_;
  Json index_;
  std::map<std::string, std::unique_ptr<SafetensorsFile>> files_;
  SafetensorsFile* single_ = nullptr;
};

}  // namespace

ModelConfig load_config(const std::string& model_dir) {
  const Json j = Json::parse_file(model_dir + "/config.json");
  auto get_int = [&](const char* key) { return static_cast<int>(j.at(key).as_int()); };

  if (j.at("model_type").as_string() != "llama") {
    throw std::runtime_error("config.json: only model_type \"llama\" is supported");
  }
  for (const char* key : {"attention_bias", "mlp_bias"}) {
    if (j.has(key) && j.at(key).boolean) {
      throw std::runtime_error(std::string("config.json: ") + key + " is not supported");
    }
  }

  ModelConfig c;
  c.name = model_dir;
  c.dim = get_int("hidden_size");
  c.hidden_dim = get_int("intermediate_size");
  c.n_layers = get_int("num_hidden_layers");
  c.n_heads = get_int("num_attention_heads");
  c.n_kv_heads = get_int("num_key_value_heads");
  c.vocab_size = get_int("vocab_size");
  c.max_seq_len = get_int("max_position_embeddings");
  c.rms_eps = static_cast<float>(j.at("rms_norm_eps").as_number());
  c.rope_theta = static_cast<float>(j.at("rope_theta").as_number());
  c.tied_embedding = j.has("tie_word_embeddings") && j.at("tie_word_embeddings").boolean;

  if (c.n_heads <= 0 || c.n_kv_heads <= 0 || c.dim % c.n_heads != 0 ||
      c.n_heads % c.n_kv_heads != 0) {
    throw std::runtime_error("config.json: inconsistent head counts");
  }
  if (j.has("head_dim") && j.at("head_dim").as_int() != c.head_dim()) {
    throw std::runtime_error("config.json: head_dim != hidden_size / heads is not supported");
  }

  if (j.has("rope_scaling") && !j.at("rope_scaling").is_null()) {
    const Json& r = j.at("rope_scaling");
    if (r.at("rope_type").as_string() != "llama3") {
      throw std::runtime_error("config.json: unsupported rope_type");
    }
    c.rope_scaling.enabled = true;
    c.rope_scaling.factor = static_cast<float>(r.at("factor").as_number());
    c.rope_scaling.low_freq_factor = static_cast<float>(r.at("low_freq_factor").as_number());
    c.rope_scaling.high_freq_factor = static_cast<float>(r.at("high_freq_factor").as_number());
    c.rope_scaling.original_max_position =
        static_cast<int>(r.at("original_max_position_embeddings").as_int());
  }
  return c;
}

Weights load_weights(const std::string& model_dir, const ModelConfig& c) {
  Checkpoint ckpt(model_dir);
  const long long dim = c.dim, kv_dim = c.kv_dim(), hidden = c.hidden_dim, vocab = c.vocab_size;

  Weights w;
  w.embedding = ckpt.read("model.embed_tokens.weight", {vocab, dim});
  w.layers.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; ++l) {
    LayerWeights& L = w.layers[l];
    const std::string p = "model.layers." + std::to_string(l) + ".";
    L.attn_norm = ckpt.read(p + "input_layernorm.weight", {dim});
    L.wq = ckpt.read(p + "self_attn.q_proj.weight", {dim, dim});
    L.wk = ckpt.read(p + "self_attn.k_proj.weight", {kv_dim, dim});
    L.wv = ckpt.read(p + "self_attn.v_proj.weight", {kv_dim, dim});
    L.wo = ckpt.read(p + "self_attn.o_proj.weight", {dim, dim});
    L.ffn_norm = ckpt.read(p + "post_attention_layernorm.weight", {dim});
    L.w_gate = ckpt.read(p + "mlp.gate_proj.weight", {hidden, dim});
    L.w_up = ckpt.read(p + "mlp.up_proj.weight", {hidden, dim});
    L.w_down = ckpt.read(p + "mlp.down_proj.weight", {dim, hidden});
  }
  w.final_norm = ckpt.read("model.norm.weight", {dim});
  if (!c.tied_embedding) w.lm_head = ckpt.read("lm_head.weight", {vocab, dim});
  return w;
}

}  // namespace llama3
