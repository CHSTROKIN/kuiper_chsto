// =============================================================================
// config.h  —  Shapes / hyper-parameters of the Llama 3 models we care about.
//
// Every kernel in this folder only needs a handful of integers (dim, number of
// heads, ...). This file collects them in one place so the test, benchmark and
// forward-pass programs all agree on the sizes.
// =============================================================================
#pragma once

#include <string>

namespace llama3 {

// Llama 3.1 / 3.2 stretch the RoPE frequencies so the model works on longer
// contexts ("rope_type": "llama3" in the HuggingFace config.json).
// Llama 3.0 does not use this (enabled = false).
struct RopeScaling {
  bool enabled = false;
  float factor = 8.0f;             // how much the low frequencies are slowed down
  float low_freq_factor = 1.0f;
  float high_freq_factor = 4.0f;
  int original_max_position = 8192;
};

struct ModelConfig {
  std::string name;
  int dim = 0;          // width of the residual stream ("hidden_size")
  int hidden_dim = 0;   // width inside the feed-forward block ("intermediate_size")
  int n_layers = 0;     // number of transformer blocks
  int n_heads = 0;      // number of query heads
  int n_kv_heads = 0;   // number of key/value heads (Grouped Query Attention)
  int vocab_size = 0;   // number of tokens in the vocabulary
  int max_seq_len = 0;  // how many positions the KV-cache can hold
  float rms_eps = 1e-5f;
  float rope_theta = 500000.0f;  // Llama 2 used 10000, Llama 3 uses 500000
  RopeScaling rope_scaling;
  bool tied_embedding = false;   // true: the output layer re-uses the embedding table

  int head_dim() const { return dim / n_heads; }          // size of one head
  int kv_dim() const { return n_kv_heads * head_dim(); }  // size of all K (or V) heads
  int kv_group_size() const { return n_heads / n_kv_heads; }  // query heads per KV head
};

// A tiny model: runs instantly, useful for tests and for reading the code.
inline ModelConfig tiny_config() {
  ModelConfig c;
  c.name = "tiny";
  c.dim = 256;
  c.hidden_dim = 768;
  c.n_layers = 4;
  c.n_heads = 8;
  c.n_kv_heads = 2;
  c.vocab_size = 4096;
  c.max_seq_len = 256;
  return c;
}

// Llama 3.2 1B (meta-llama/Llama-3.2-1B).
inline ModelConfig llama3_2_1b_config() {
  ModelConfig c;
  c.name = "llama3.2-1b";
  c.dim = 2048;
  c.hidden_dim = 8192;
  c.n_layers = 16;
  c.n_heads = 32;
  c.n_kv_heads = 8;
  c.vocab_size = 128256;
  c.max_seq_len = 1024;
  c.rope_scaling.enabled = true;
  c.rope_scaling.factor = 32.0f;
  c.tied_embedding = true;
  return c;
}

// Llama 3 8B (meta-llama/Meta-Llama-3-8B). Llama 3.1 8B is identical except
// that it enables rope_scaling with factor 8.
inline ModelConfig llama3_8b_config() {
  ModelConfig c;
  c.name = "llama3-8b";
  c.dim = 4096;
  c.hidden_dim = 14336;
  c.n_layers = 32;
  c.n_heads = 32;
  c.n_kv_heads = 8;
  c.vocab_size = 128256;
  c.max_seq_len = 1024;
  return c;
}

// "tiny", "1b" or "8b" -> config. Returns false for an unknown name.
inline bool config_by_name(const std::string& name, ModelConfig* out) {
  if (name == "tiny") {
    *out = tiny_config();
  } else if (name == "1b") {
    *out = llama3_2_1b_config();
  } else if (name == "8b") {
    *out = llama3_8b_config();
  } else {
    return false;
  }
  return true;
}

}  // namespace llama3
