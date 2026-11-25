// =============================================================================
// checkpoint.h  —  Load a Llama 3.x model folder in Hugging Face format:
//
//     config.json                         shapes and hyper-parameters
//     model.safetensors                   weights (or several shards listed in
//     [model.safetensors.index.json]      this index file)
//
// Weights may be stored as BF16 or F32; the kernels use fp32, so BF16 values
// are widened while loading.
// =============================================================================
#pragma once

#include <string>

#include "llama3/config.h"
#include "llama3/model.h"

namespace llama3 {

// Both throw std::runtime_error with a readable message on any problem.
// max_seq_len is left at the model's own maximum; lower it to save memory.
ModelConfig load_config(const std::string& model_dir);

// Every tensor's shape is checked against the config.
Weights load_weights(const std::string& model_dir, const ModelConfig& config);

}  // namespace llama3
