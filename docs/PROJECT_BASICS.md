# Kuiper-Chsto 项目基础 / Project Basics

[中文版](#中文版) · [English version](#english-version)

## 中文版

Kuiper-Chsto 是一个用于学习的 C++/CUDA 大模型推理项目。目前能编译和聊天的部分是 `llama3_kernels/`。

### 两套代码

| 目录 | 内容 | 现状 |
| --- | --- | --- |
| `llama3_kernels/` | CPU 和 CUDA 两套 kernel、权重加载、分词、聊天程序 | 可以编译和运行 |
| `source/`、`include/` | 原来的推理框架 | 编译不过：默认设置下缺少 absl 头文件；按 [KERNELS.md](KERNELS.md) 记录，还有很多类没有实现 |

两套代码互相独立。

### 权重保存位置

权重在项目根目录下的 `models/llama-3.2-1b-instruct/`，共约 2.4 GiB。

| 文件 | 作用 | 大小（字节） |
| --- | --- | --- |
| `model.safetensors` | 模型权重 | 2,471,645,608 |
| `tokenizer.json` | 分词器词表和合并规则 | 17,209,920 |
| `config.json` | 层数、维度等模型参数 | 894 |
| `generation_config.json` | 生成参数（`chat` 程序不读取） | 239 |

- 模型是 Llama 3.2 1B Instruct。Llama 3.1 最小为 8B，fp32 权重约 32 GB，8 GB 显存放不下。
- 权重来自公开转载仓库 [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)，使用时需遵守 [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/)。
- `models/` 已加入 `.gitignore`。换一台电脑需要重新下载，命令见 [llama3_kernels/README.md 第 0 节](../llama3_kernels/README.md#0-chat-with-a-real-model)。
- 磁盘上的权重加载时会转成 fp32，在内存或显存中约占 4.9 GB。

### 如何编译

需要 NVIDIA GPU 和驱动、CUDA Toolkit（含 `nvcc`）、CMake 3.18 以上、支持 C++17 的编译器。OpenMP 可选，用于 CPU 多线程。验证过的版本是 CUDA 12.6、CMake 3.22.1、g++ 11.4（WSL2）。

在项目根目录执行：

```bash
# 1. 配置（生成构建文件）
cmake -S llama3_kernels -B build_kernels -DCMAKE_BUILD_TYPE=Release

# 2. 编译
cmake --build build_kernels -j
```

- `-S` 指定源码目录，`-B` 指定输出目录。
- 可选：在第 1 步加 `-DCMAKE_CUDA_ARCHITECTURES=89`，只为 RTX 40 系列编译，速度更快。
- 如果 CMake 找不到 `nvcc`，加 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`。
- 只改了已有的源码时，重新执行第 2 步即可。
- 想从头编译时，换一个新的 `-B` 目录。

### 如何运行

| 命令 | 作用 |
| --- | --- |
| `./build_kernels/chat --stats` | 和模型聊天；`/reset` 开新对话，`/exit` 退出 |
| `./build_kernels/test_kernels` | 测试 kernel，应显示 40 passed, 0 failed |
| `./build_kernels/llama3_forward --model tiny` | 用随机权重比较 CPU 和 GPU 结果 |
| `./build_kernels/bench_kernels --model 1b` | 测量每个 kernel 的速度 |

请在项目根目录运行 `chat`，因为默认的模型路径是相对路径。其他位置可用 `--model-dir` 指定。常用参数有 `--device cpu`、`--temperature 0`（输出更稳定）和 `--max-tokens 1024`。

### 一句话是怎么生成的

1. **分词：** 把文字转成 token ID。
2. **查表（embedding）：** 把每个 token ID 转成一个向量。
3. **Transformer 层（1B 模型有 16 层）：** 每层先做注意力，再做前馈网络（SwiGLU）。两部分之前都做 RMSNorm，之后都加上残差。
4. **输出：** 得到词表中每个 token 的分数（logits）。
5. **采样：** 选出下一个 token，转回文字，然后回到第 2 步，直到模型结束回答、达到长度上限或上下文写满。

注意力部分有两个要点。**RoPE** 把位置信息加到 query 和 key 上。**KV cache** 保存之前 token 的 key 和 value，避免重复计算。

### 代码从哪里读起

以下路径都在 `llama3_kernels/` 下。

| 文件 | 内容 |
| --- | --- |
| `include/llama3/model.h` | 整个模型的计算流程，建议先读 |
| `src/cpu/`、`src/cuda/` | 每个 kernel 的 CPU 版和 GPU 版，函数一一对应 |
| `src/checkpoint.cpp` | 读取配置和权重 |
| `src/tokenizer.cpp` | Llama 3 分词器 |
| `apps/chat.cpp` | 对话格式、采样和聊天循环 |

各 kernel 的数学原理和性能分析见 [KERNELS.md](KERNELS.md)。

### 注意事项

- 1B 模型经常说错事实，中文明显弱于英文。
- `test_kernels` 不测试权重加载、分词器和聊天程序。改动后需要和 Hugging Face 的结果对比。

## English version

Kuiper-Chsto is a C++/CUDA LLM inference project built for learning. The part that currently compiles and chats is `llama3_kernels/`.

### The two codebases

| Directory | Contents | Status |
| --- | --- | --- |
| `llama3_kernels/` | CPU and CUDA kernels, weight loader, tokenizer, chat program | Builds and runs |
| `source/`, `include/` | The original inference framework | Does not build: absl headers are missing with default settings, and [KERNELS.md](KERNELS.md) records many unimplemented classes |

The two codebases are independent.

### Where the weights are stored

The weights are in `models/llama-3.2-1b-instruct/` under the repository root, about 2.4 GiB in total.

| File | Purpose | Size (bytes) |
| --- | --- | --- |
| `model.safetensors` | Model weights | 2,471,645,608 |
| `tokenizer.json` | Tokenizer vocabulary and merge rules | 17,209,920 |
| `config.json` | Layer count, dimensions and other model settings | 894 |
| `generation_config.json` | Generation settings (not read by `chat`) | 239 |

- The model is Llama 3.2 1B Instruct. Llama 3.1 starts at 8B, whose fp32 weights are about 32 GB and do not fit in 8 GB of GPU memory.
- The weights come from the public copy [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct). The [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) applies.
- `models/` is listed in `.gitignore`. On another machine, download the weights again using [section 0 of llama3_kernels/README.md](../llama3_kernels/README.md#0-chat-with-a-real-model).
- The loader converts the weights to fp32, so they take about 4.9 GB in RAM or GPU memory.

### How to compile

You need an NVIDIA GPU and driver, the CUDA Toolkit (with `nvcc`), CMake 3.18 or later, and a C++17 compiler. OpenMP is optional and enables CPU multithreading. Verified versions: CUDA 12.6, CMake 3.22.1 and g++ 11.4 (WSL2).

Run from the repository root:

```bash
# 1. Configure (generate the build files)
cmake -S llama3_kernels -B build_kernels -DCMAKE_BUILD_TYPE=Release

# 2. Compile
cmake --build build_kernels -j
```

- `-S` is the source directory and `-B` is the output directory.
- Optional: add `-DCMAKE_CUDA_ARCHITECTURES=89` in step 1 to build only for RTX 40 series GPUs, which is faster.
- If CMake cannot find `nvcc`, add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
- After editing existing source files, only step 2 is needed.
- For a clean build, use a new `-B` directory.

### How to run

| Command | What it does |
| --- | --- |
| `./build_kernels/chat --stats` | Chat with the model; `/reset` starts a new conversation, `/exit` quits |
| `./build_kernels/test_kernels` | Test the kernels; expect 40 passed, 0 failed |
| `./build_kernels/llama3_forward --model tiny` | Compare CPU and GPU results using random weights |
| `./build_kernels/bench_kernels --model 1b` | Time each kernel |

Run `chat` from the repository root, because the default model path is relative. From elsewhere, pass `--model-dir`. Other useful flags are `--device cpu`, `--temperature 0` (more stable output) and `--max-tokens 1024`.

### How text is generated

1. **Tokenize:** turn the text into token IDs.
2. **Embed:** turn each token ID into a vector.
3. **Transformer layers (16 in the 1B model):** each layer runs attention, then a feed-forward network (SwiGLU). Each part has RMSNorm before it and a residual addition after it.
4. **Output:** produce a score (logit) for every token in the vocabulary.
5. **Sample:** pick the next token, turn it back into text, and return to step 2. This repeats until the model ends its answer, reaches the length limit or fills the context.

Two ideas matter in attention. **RoPE** adds position information to the queries and keys. The **KV cache** stores the keys and values of earlier tokens, so they are not recomputed.

### Where to start reading the code

The paths below are inside `llama3_kernels/`.

| File | Contents |
| --- | --- |
| `include/llama3/model.h` | The whole model computation; read this first |
| `src/cpu/`, `src/cuda/` | CPU and GPU versions of each kernel, with matching functions |
| `src/checkpoint.cpp` | Loads the config and weights |
| `src/tokenizer.cpp` | The Llama 3 tokenizer |
| `apps/chat.cpp` | Chat format, sampling and the chat loop |

See [KERNELS.md](KERNELS.md) for the math behind each kernel and the performance analysis.

### Caveats

- The 1B model often gets facts wrong, and it is clearly weaker in Chinese than in English.
- `test_kernels` does not cover the weight loader, tokenizer or chat program. After changing them, compare the results with Hugging Face.
