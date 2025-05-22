# Kuiper-Chsto Architecture Guide

This document provides a comprehensive overview of the Kuiper-Chsto inference engine architecture, including system design, component relationships, and implementation details.

## System Overview

Kuiper-Chsto is designed as a modular, high-performance inference engine for transformer-based language models. The architecture follows a layered approach with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                     Model Layer                             │
├─────────────────────────────────────────────────────────────┤
│                    Operations Layer                         │
├─────────────────────────────────────────────────────────────┤
│                    Tensor System                            │
├─────────────────────────────────────────────────────────────┤
│                    Device Abstraction                       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Layer

The model layer provides abstract interfaces and concrete implementations for different model architectures.

#### Base Model Interface (`model.h`)

```cpp
class Model {
public:
    virtual base::Status init(base::DeviceType device_type) = 0;
    virtual base::Status predict(const tensor::Tensor& input, 
                                const tensor::Tensor& pos_tensor,
                                bool is_prompt, int& next) const = 0;
    virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;
    // ... other methods
};
```

Key responsibilities:
- Model initialization and configuration
- Inference pipeline management
- Token encoding/decoding
- Memory buffer management

#### LLaMA Implementation (`llama3.h`)

The LLaMA model implementation extends the base model with specific architectural components:

- **Attention Layers**: Multi-head attention with RoPE
- **Feed-Forward Networks**: SwiGLU activation
- **Normalization**: RMSNorm layers
- **Position Encoding**: Rotary Position Embedding (RoPE)

### 2. Operations Layer

The operations layer implements neural network primitives as modular components.

#### Operation Categories

**Attention Operations:**
- `mha.h`: Multi-head attention
- `rope.h`: Rotary position embedding

**Normalization Operations:**
- `rmsnorm.h`: Root mean square normalization

**Activation Operations:**
- `swiglu.h`: Swish-Gated Linear Unit

**Embedding Operations:**
- `embedding.h`: Token embedding
- `encode.h`: Input encoding

#### Operation Interface

Each operation implements the `Layer` interface with forward computation methods and supports both CPU and CUDA backends.

### 3. Tensor System

The tensor system provides multi-dimensional array operations with device abstraction.

#### Tensor Class (`tensor.h`)

```cpp
class Tensor {
public:
    // Construction with various dimensions
    explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, ...);
    
    // Device management
    void to_cpu();
    void to_cuda(cudaStream_t stream = nullptr);
    
    // Memory access
    template <typename T> T* ptr();
    template <typename T> T& index(int64_t offset);
    
    // Properties
    size_t size() const;
    base::DataType data_type() const;
    base::DeviceType device_type() const;
};
```

Key features:
- Multi-dimensional tensor support
- Transparent CPU/CUDA memory management
- Type-safe data access
- Efficient memory allocation and reuse

### 4. Device Abstraction

The device abstraction layer manages computation across different hardware platforms.

#### Device Types

```cpp
enum class DeviceType {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2
};
```

#### Memory Management

- **CPU Allocation**: Standard heap allocation
- **CUDA Allocation**: Unified memory or device memory
- **Buffer Management**: Smart pointer-based memory management

## Data Flow

### Inference Pipeline

1. **Input Processing**
   ```
   Text Input → Tokenization → Token IDs → Embedding Lookup
   ```

2. **Forward Pass**
   ```
   Embedded Input → Multiple Transformer Layers → Output Logits
   ```

3. **Output Generation**
   ```
   Logits → Sampling → Next Token → Decoding → Text Output
   ```

### Memory Management

Kuiper-Chsto employs several memory optimization strategies:

- **Buffer Reuse**: Pre-allocated buffers for intermediate results
- **KV Cache**: Cached key-value pairs for efficient autoregressive generation
- **Memory Pooling**: Efficient allocation for frequently used tensor sizes

## Kernel Implementation

### CPU Kernels

Located in `source/op/kernels/cpu/`:
- `rope_kernel.cpp`: Rotary position embedding
- `swiglu_kernel.cpp`: SwiGLU activation
- `softmax_kernel.cpp`: Softmax computation
- `scale_kernel.cpp`: Element-wise scaling

### CUDA Kernels

Located in `source/op/kernels/cuda/`:
- `rope_kernel.cu`: GPU-optimized RoPE
- `mha_kernel.cu`: Multi-head attention
- `rmsnorm_kernel.cu`: RMS normalization

### Kernel Interface

All kernels implement a common interface defined in `kernels_interface.h`:

```cpp
class KernelInterface {
public:
    virtual base::Status compute(...) = 0;
    virtual bool is_supported(base::DeviceType device) const = 0;
};
```

## Configuration System

### Model Configuration

Model parameters are defined in `config.h`:

```cpp
struct TransformerConfig {
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t num_attention_heads;
    int32_t num_hidden_layers;
    int32_t intermediate_size;
    // ... other parameters
};
```

### Device Configuration

CUDA-specific settings in `cuda_config.h`:

```cpp
struct CudaConfig {
    int device_id;
    cudaStream_t stream;
    // ... CUDA-specific parameters
};
```

## Extension Points

### Adding New Models

1. Extend the base `Model` class
2. Implement required virtual methods
3. Add model-specific layers and operations
4. Register with the model factory

### Adding New Operations

1. Create operation header in `include/op/`
2. Implement operation in `source/op/`
3. Add CPU and CUDA kernels if needed
4. Register with operation registry

### Adding New Devices

1. Extend device abstraction layer
2. Implement memory allocation/deallocation
3. Add kernel implementations for the device
4. Update device detection and selection

## Performance Considerations

### Optimization Strategies

1. **Memory Layout**: Optimized tensor memory layout for cache efficiency
2. **Kernel Fusion**: Combined operations to reduce memory transfers
3. **Asynchronous Execution**: Overlapping computation and memory operations
4. **Batch Processing**: Efficient handling of multiple sequences

### Profiling and Monitoring

- Integrated logging with Google Logging
- Performance counters for key operations
- Memory usage tracking
- CUDA kernel timing

## Testing Architecture

The test framework (`test/`) provides:

- Unit tests for individual components
- Integration tests for model inference
- Performance benchmarks
- CUDA-specific test cases

## Dependencies and Integration

### External Libraries

- **Armadillo**: Linear algebra operations
- **SentencePiece**: Tokenization
- **Google Test**: Testing framework
- **Google Logging**: Logging utilities

### Build System

CMake-based build system with:
- Automatic dependency management via CPM
- CUDA compilation support
- Cross-platform compatibility
- Configurable build options

## Future Architecture Directions

- **Quantization Support**: 8-bit and 4-bit quantization
- **Distributed Inference**: Multi-GPU and multi-node support
- **Plugin System**: Runtime operation loading
- **JIT Compilation**: Runtime kernel optimization
