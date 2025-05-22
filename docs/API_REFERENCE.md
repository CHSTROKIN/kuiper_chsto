# Kuiper-Chsto API Reference

This document provides detailed API documentation for the Kuiper-Chsto inference engine, covering all major classes, methods, and usage patterns.

## Table of Contents

- [Model API](#model-api)
- [Tensor API](#tensor-api)
- [Operations API](#operations-api)
- [Sampler API](#sampler-api)
- [Base Utilities](#base-utilities)
- [Device Management](#device-management)

## Model API

### Base Model Class

The `model::Model` class provides the abstract interface for all model implementations.

#### Constructor

```cpp
explicit Model(base::TokenizerType tokenizer_type, 
               base::ModelType model_type,
               std::string token_path, 
               std::string model_path, 
               bool is_quant_model);
```

**Parameters:**
- `tokenizer_type`: Type of tokenizer (SentencePiece, etc.)
- `model_type`: Type of model (LLaMA, etc.)
- `token_path`: Path to tokenizer model file
- `model_path`: Path to model checkpoint file
- `is_quant_model`: Whether the model uses quantization

#### Core Methods

##### Initialization

```cpp
virtual base::Status init(base::DeviceType device_type) = 0;
```

Initializes the model with the specified device.

**Parameters:**
- `device_type`: Target device (CPU or CUDA)

**Returns:** Status indicating success or failure

##### Inference

```cpp
virtual base::Status predict(const tensor::Tensor& input, 
                            const tensor::Tensor& pos_tensor,
                            bool is_prompt, 
                            int& next) const = 0;
```

Performs inference to generate the next token.

**Parameters:**
- `input`: Input tensor
- `pos_tensor`: Position tensor
- `is_prompt`: Whether this is prompt processing
- `next`: Output parameter for next token ID

**Returns:** Status indicating success or failure

##### Forward Pass

```cpp
virtual base::Status forward(const tensor::Tensor& input, 
                            const tensor::Tensor& pos_tensor,
                            int& next) const = 0;
```

Performs a forward pass through the model.

##### Embedding

```cpp
virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;
```

Computes token embeddings.

**Parameters:**
- `tokens`: Input token IDs

**Returns:** Embedding output structure

#### Utility Methods

##### Tokenization

```cpp
virtual std::vector<int32_t> encode(const std::string& sentence) const;
```

Encodes text into token IDs.

```cpp
virtual std::string decode(int32_t token_idx) const;
virtual std::string decode(std::vector<int32_t> token_idxs) const;
```

Decodes token IDs back to text.

##### Buffer Management

```cpp
virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);
virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;
```

Accesses model buffers for intermediate results.

### LLaMA Model Implementation

The `model::LLama2Model` class implements the LLaMA architecture.

#### Constructor

```cpp
explicit LLama2Model(base::TokenizerType tokenizer_type, 
                     std::string token_path,
                     std::string model_path, 
                     bool is_quant_model);
```

## Tensor API

### Tensor Class

The `tensor::Tensor` class provides multi-dimensional array operations.

#### Constructors

```cpp
// 1D tensor
explicit Tensor(base::DataType data_type, int32_t dim0, 
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr, 
                void* ptr = nullptr);

// 2D tensor  
explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, 
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr);

// N-dimensional tensor
explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, 
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr);
```

#### Device Management

```cpp
void to_cpu();
void to_cuda(cudaStream_t stream = nullptr);
```

Moves tensor between CPU and CUDA devices.

#### Memory Access

```cpp
template <typename T> T* ptr();
template <typename T> const T* ptr() const;

template <typename T> T* ptr(int64_t index);
template <typename T> const T* ptr(int64_t index) const;

template <typename T> T& index(int64_t offset);
template <typename T> const T& index(int64_t offset) const;
```

Accesses tensor data with type safety.

#### Properties

```cpp
size_t size() const;                    // Total number of elements
size_t byte_size() const;              // Total bytes
int32_t dims_size() const;             // Number of dimensions
base::DataType data_type() const;      // Data type
int32_t get_dim(int32_t idx) const;    // Dimension at index
const std::vector<int32_t>& dims() const; // All dimensions
std::vector<size_t> strides() const;   // Memory strides
base::DeviceType device_type() const;  // Current device
```

#### Operations

```cpp
void reshape(const std::vector<int32_t>& dims);
tensor::Tensor clone() const;
void reset(base::DataType data_type, const std::vector<int32_t>& dims);
```

## Operations API

### Base Operation Interface

All operations implement the `op::Layer` interface.

### Multi-Head Attention (MHA)

```cpp
class MHA : public Layer {
public:
    base::Status forward(const tensor::Tensor& input, 
                        tensor::Tensor& output) override;
};
```

### Rotary Position Embedding (RoPE)

```cpp
class RoPE : public Layer {
public:
    base::Status forward(const tensor::Tensor& input, 
                        tensor::Tensor& output) override;
};
```

### RMS Normalization

```cpp
class RMSNorm : public Layer {
public:
    base::Status forward(const tensor::Tensor& input, 
                        tensor::Tensor& output) override;
};
```

### SwiGLU Activation

```cpp
class SwiGLU : public Layer {
public:
    base::Status forward(const tensor::Tensor& input, 
                        tensor::Tensor& output) override;
};
```

### Embedding Layer

```cpp
class Embedding : public Layer {
public:
    base::Status forward(const tensor::Tensor& input, 
                        tensor::Tensor& output) override;
};
```

## Sampler API

### Base Sampler Interface

```cpp
class Sampler {
public:
    virtual int32_t sample(const tensor::Tensor& logits) = 0;
    virtual ~Sampler() = default;
};
```

### ArgMax Sampler

```cpp
class ArgMaxSampler : public Sampler {
public:
    int32_t sample(const tensor::Tensor& logits) override;
};
```

Samples the token with the highest probability.

## Base Utilities

### Status Handling

```cpp
class Status {
public:
    bool ok() const;
    int32_t get_err_code() const;
    const std::string& get_err_msg() const;
    
    static Status OK();
    static Status Error(int32_t code, const std::string& msg);
};
```

### Data Types

```cpp
enum class DataType {
    kDataTypeUnknown = 0,
    kDataTypeFloat32 = 1,
    kDataTypeFloat16 = 2,
    kDataTypeInt32 = 3,
    kDataTypeInt64 = 4,
    // ... other types
};
```

### Device Types

```cpp
enum class DeviceType {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2
};
```

### Model Types

```cpp
enum class ModelType {
    kModelTypeUnknown = 0,
    kModelTypeLLaMA2 = 1,
    kModelTypeLLaMA3 = 2
};
```

### Tokenizer Types

```cpp
enum class TokenizerType {
    kEncodeUnknown = 0,
    kEncodeSpe = 1,      // SentencePiece
    kEncodeTiktoken = 2  // Tiktoken
};
```

## Device Management

### CUDA Configuration

```cpp
struct CudaConfig {
    int device_id;
    cudaStream_t stream;
    // Additional CUDA-specific parameters
};
```

### Memory Allocation

```cpp
class DeviceAllocator {
public:
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual ~DeviceAllocator() = default;
};
```

## Configuration API

### Model Configuration

```cpp
struct TransformerConfig {
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t num_attention_heads;
    int32_t num_hidden_layers;
    int32_t intermediate_size;
    int32_t max_sequence_length;
    float rms_norm_eps;
    float rope_theta;
    // ... other parameters
};
```

### Buffer Types

```cpp
enum class ModelBufferType {
    kInputPos = 0,
    kInputIds = 1,
    kAttentionMask = 2,
    // ... other buffer types
};
```

## Error Codes

Common error codes returned by API methods:

- `0`: Success
- `1001`: Model initialization failed
- `1002`: Tensor allocation failed
- `1003`: CUDA operation failed
- `1004`: Invalid input parameters
- `1005`: File I/O error

## Usage Patterns

### Basic Inference

```cpp
// Create model
model::LLama2Model model(tokenizer_type, token_path, model_path, false);

// Initialize
auto status = model.init(base::DeviceType::kDeviceCUDA);
if (!status.ok()) {
    // Handle error
}

// Encode input
auto tokens = model.encode("Hello, world!");

// Generate text
int next_token;
tensor::Tensor pos_tensor = model.get_buffer(ModelBufferType::kInputPos);
// ... inference loop
```

### Tensor Operations

```cpp
// Create tensor
tensor::Tensor tensor(base::DataType::kDataTypeFloat32, {batch_size, seq_len});

// Access data
float* data = tensor.ptr<float>();
data[0] = 1.0f;

// Move to GPU
tensor.to_cuda();
```

### Custom Operations

```cpp
class CustomLayer : public op::Layer {
public:
    base::Status forward(const tensor::Tensor& input, 
                        tensor::Tensor& output) override {
        // Custom implementation
        return base::Status::OK();
    }
};
```

## Best Practices

1. **Error Handling**: Always check return status from API calls
2. **Memory Management**: Use RAII patterns for tensor management
3. **Device Awareness**: Be mindful of tensor device locations
4. **Performance**: Reuse buffers when possible
5. **Thread Safety**: Most operations are not thread-safe
