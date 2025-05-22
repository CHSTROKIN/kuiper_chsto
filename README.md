# Kuiper-Chsto - High-Performance LLM Inference Engine

Kuiper-Chsto is a toy high-performance C++ inference engine for large language models, specifically optimized for LLaMA model architectures. It provides efficient CPU and CUDA acceleration for transformer-based models. This is for educational purpose only. 
Most of the codebase is based on the following two repositories:
https://github.com/zjhellofss/KuiperInfer
https://github.com/zjhellofss/KuiperLLama
## Features

- **Multi-Device Support**: CPU and CUDA acceleration
- **LLaMA Model Support**: Full implementation of LLaMA 2/3 architectures
- **Optimized Kernels**: Custom implementations of transformer operations
- **Memory Efficient**: Smart tensor management and buffer reuse
- **Extensible Architecture**: Easy to add new models and operations
- **Quantization Ready**: Support for quantized model inference
- **Production Ready**: Comprehensive testing and logging

## Quick Start

### Prerequisites

- C++17 compatible compiler
- CMake 3.16+
- CUDA Toolkit (optional, for GPU acceleration)
- Git

### Building from Source

```bash
# Clone the repository
git clone https://github.com/CHSTROKIN/kuiper_chsto.git
cd kuiper_chsto

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DUSE_CPM=ON

# Build the project
make -j$(nproc)
```

### Running the Demo

```bash
# Run the inference demo
./bin/demo /path/to/model.bin /path/to/tokenizer.model
```

## Project Structure

```
kuiper_chsto/
├── include/              # Header files
│   ├── model/           # Model interfaces and implementations
│   ├── op/              # Neural network operations
│   ├── tensor/          # Tensor system
│   ├── sampler/         # Token sampling strategies
│   └── base/            # Foundation utilities
├── source/              # Implementation files
│   ├── model/           # Model implementations
│   ├── op/              # Operation implementations
│   ├── tensor/          # Tensor implementations
│   └── sampler/         # Sampler implementations
├── demo/                # Example usage
├── test/                # Test suite
└── cmake/               # Build system configuration
```

## Core Components

### Model System
- **Base Model**: Abstract interface for all models
- **LLaMA Implementation**: Complete LLaMA 2/3 architecture
- **Tokenization**: SentencePiece and custom tokenizers
- **Inference Pipeline**: Complete text generation workflow

### Operations (op)
- **Multi-Head Attention (MHA)**: Self-attention mechanism
- **Rotary Position Embedding (RoPE)**: Position encoding
- **RMSNorm**: Normalization layer
- **SwiGLU**: Activation function
- **Embedding**: Token embedding layer

### Tensor System
- **Multi-dimensional Tensors**: Flexible tensor operations
- **Device Abstraction**: CPU/CUDA transparent switching
- **Memory Management**: Efficient buffer allocation
- **Type Safety**: Strong typing for data types

## Performance

Kuiper-Chsto is optimized for inference performance:
- Efficient kernel implementations
- Memory reuse and buffer management
- CUDA acceleration for GPU inference
- Minimal overhead in the inference loop

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Technical architecture overview
- [API Reference](docs/API_REFERENCE.md) - Detailed API documentation
- [Build Guide(TODO)](docs/BUILD_GUIDE.md) - Complete build instructions
- [Examples(TODO)](docs/EXAMPLES.md) - Usage examples and tutorials

## Dependencies

- **Google Test**: Unit testing framework
- **Google Logging**: Logging utilities
- **Armadillo**: Linear algebra library
- **SentencePiece**: Tokenization library
- **Abseil**: C++ utility library
- **RE2**: Regular expression library
- **nlohmann/json**: JSON parsing library

## License

This project is licensed under the terms contained in the LICENSE file.


## Acknowledgment

Special thanks to @zjhellofss
 for creating the original repositories KuiperInfer and KuiperLLama, which served as the foundation and inspiration for this project. Your work made Kuiper-Chsto possible. 🙏