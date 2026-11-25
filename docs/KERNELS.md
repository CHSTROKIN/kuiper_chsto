# Llama 3 Kernels: CPU and CUDA, Explained

This guide explains the code in [`llama3_kernels/`](../llama3_kernels). That folder
contains every operation needed to run Llama 3, each written once for the CPU and once
for CUDA. To build and run the code, see [`llama3_kernels/README.md`](../llama3_kernels/README.md).

**Contents**

1. [How to read this code](#1-how-to-read-this-code)
2. [Llama 3 in one page](#2-llama-3-in-one-page)
3. [Conventions: data layout](#3-conventions-data-layout)
4. [Just enough CUDA](#4-just-enough-cuda)
5. [The kernels, one by one](#5-the-kernels-one-by-one)
6. [Putting it together: the forward pass](#6-putting-it-together-the-forward-pass)
7. [How correctness is checked](#7-how-correctness-is-checked)
8. [Performance: CPU vs GPU, and why](#8-performance-cpu-vs-gpu-and-why)
9. [Where to go next (optimizations)](#9-where-to-go-next-optimizations)
10. [Relation to the rest of the repository](#10-relation-to-the-rest-of-the-repository)

---

## 1. How to read this code

Suggested order:

1. `include/llama3/cpu_kernels.h` lists every kernel with the shape of each argument.
2. `apps/llama3_forward.cpp`, class `CpuLlama3::forward`: about 30 lines that show how
   the kernels form a full Llama 3 model.
3. The CPU kernels in `src/cpu/`. They are plain loops: the math with nothing else.
4. The CUDA kernels in `src/cuda/`, each next to its CPU twin. Every file begins with a
   *thread mapping* comment that says which GPU thread computes which output.
5. `apps/test_kernels.cpp` and `apps/bench_kernels.cpp` show how the kernels are checked
   and measured.

Design rules used everywhere:

* **float32 only**, plain pointers, row-major arrays. There is no tensor class, so
  nothing is hidden.
* **CPU and CUDA functions have identical signatures**: `llama3::cpu::matmul(...)` and
  `llama3::cuda::matmul(...)` take the same arguments. The CUDA version also takes an
  optional `cudaStream_t`.
* **One kernel per file**, each with a header comment that explains the math.
* **Simple over fast.** Where a well-known optimization was left out on purpose, the
  file's comment names it. Section 9 collects them.

---

## 2. Llama 3 in one page

Llama 3 generates text **one token at a time**. Each new token runs through the network
once. This is called a *decode step*. The token's vector `x` (length `dim`) flows
through `n_layers` identical blocks:

```
token id
   │
   ▼
embedding ──────────────► x  (dim)
                          │
   ┌──────────────────────┼───────────────────────── repeated n_layers times ─┐
   │                      ├──────────────┐                                    │
   │                      ▼              │ residual                           │
   │                   rmsnorm           │                                    │
   │           ┌──────────┼──────────┐   │                                    │
   │         matmul Wq  matmul Wk  matmul Wv                                  │
   │           │          │          │   │                                    │
   │         rope(q)    rope(k)      │   │      k, v are appended to the      │
   │           │          │          │   │      KV cache of this layer        │
   │           └──── attention(q, K cache, V cache) ◄── softmax inside        │
   │                      │              │                                    │
   │                   matmul Wo         │                                    │
   │                      ▼              │                                    │
   │                     add ◄───────────┘                                    │
   │                      ├──────────────┐                                    │
   │                   rmsnorm           │ residual                           │
   │              ┌───────┴───────┐      │                                    │
   │         matmul Wgate    matmul Wup  │                                    │
   │              └──── swiglu ───┘      │                                    │
   │                      │              │                                    │
   │                  matmul Wdown       │                                    │
   │                      ▼              │                                    │
   │                     add ◄───────────┘                                    │
   └──────────────────────┼───────────────────────────────────────────────────┘
                          ▼
                       rmsnorm
                          ▼
                   matmul (lm head)  ──► logits (vocab_size)
                          ▼
                       argmax  ──► next token id
```

That diagram uses exactly the nine kernels in this folder, and nothing else.

The sizes of the models used in the programs (`include/llama3/config.h`):

| | tiny | Llama 3.2 1B | Llama 3 8B |
|---|---|---|---|
| `dim` (hidden size) | 256 | 2048 | 4096 |
| `hidden_dim` (FFN size) | 768 | 8192 | 14336 |
| `n_layers` | 4 | 16 | 32 |
| `n_heads` (query heads) | 8 | 32 | 32 |
| `n_kv_heads` (key/value heads) | 2 | 8 | 8 |
| `head_dim` = dim / n_heads | 32 | 64 | 128 |
| `vocab_size` | 4096 | 128256 | 128256 |
| `rope_theta` | 500000 | 500000 | 500000 |
| RoPE scaling | no | yes (factor 32) | no (Llama 3.1: factor 8) |
| lm head shares the embedding table | no | yes | no |
| fp32 weights | 0.02 GB | 4.9 GB | 32 GB |

What makes Llama **3** different from Llama 2, as seen by the kernels:

* **Grouped Query Attention** in every model size: several query heads share one
  key/value head (§5.6).
* **RoPE base 500000** instead of 10000, and in 3.1/3.2 a long-context frequency
  scaling (§5.4).
* A **128k vocabulary**, so the final `lm head` matmul is large: 0.5 GB of fp32 weights
  for 1B and 2.1 GB for 8B.

---

## 3. Conventions: data layout

**Row-major arrays.** A tensor of shape `[A, B]` is stored as `A` rows of `B`
consecutive floats, so element `(a, b)` is at `ptr[a * B + b]`. A shape `[A, B, C]`
puts element `(a, b, c)` at `ptr[(a * B + b) * C + c]`.

**Linear-layer weights** are stored like PyTorch's `nn.Linear.weight`: shape
`[out_features, in_features]`, with one row per output. A linear layer is therefore
`out[n] = dot(x, w[n, :])`, a dot product with a contiguous row. This layout is
convenient for both the CPU and the GPU.

**Heads.** `q` has shape `[n_heads, head_dim]`. Head `h` is the slice
`q[h*head_dim .. (h+1)*head_dim)`. `k` and `v` have shape `[n_kv_heads, head_dim]`.

**KV cache.** One per layer, shaped `[max_seq_len, n_kv_heads, head_dim]`:

```
k_cache (one layer)
row 0   [ kvhead0 (head_dim) | kvhead1 | ... | kvhead7 ]   ← key of the token at position 0
row 1   [ ...                                           ]   ← position 1
...
row pos [ written by this decode step                   ]
```

Row `t` holds the keys of all KV heads for position `t`, so a row has
`kv_dim = n_kv_heads * head_dim` floats. During a decode step the `k` and `v`
matmuls write **directly into row `pos`**. Attention then reads rows `0..pos`.

**Kernel arguments.** Inputs come first, then sizes, then outputs, then (CUDA only)
the stream. A comment says when a kernel may work in place (`out == input`).

---

## 4. Just enough CUDA

This section covers only the CUDA features the kernels use.

**Host and device.** The CPU (*host*) and the GPU (*device*) have separate memory.
`cudaMalloc` allocates GPU memory. `cudaMemcpy` copies between the two. In this code,
`DeviceBuffer<T>` (`cuda_utils.h`) wraps both and frees the memory automatically.
Every CUDA kernel expects its float pointers to point to GPU memory.

**Kernels and launches.** A `__global__` function runs on the GPU. You launch it from
the host with

```cpp
my_kernel<<<number_of_blocks, threads_per_block, shared_bytes, stream>>>(args...);
```

The GPU then starts `number_of_blocks × threads_per_block` threads, all running the same
function. Each thread finds out *who it is* from built-in variables:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread number
```

The first decision in every kernel is the **thread mapping**: which thread computes
which part of the output. Each `.cu` file documents its mapping at the top.

| Mapping used | Kernels |
|---|---|
| one thread per output element | `add`, `swiglu`, `embedding`, `rope` (per pair), attention scores and attention output |
| one **block** per output value or row, with threads cooperating on a reduction | `matmul` (one block per weight row), `softmax` (one block per row), `rmsnorm` (one block total), `argmax` (one block total) |

**Bounds check.** The number of threads is rounded up to a whole number of blocks
(`ceil_div`), so the last block may have threads with no work. That is why kernels
contain `if (i < n)`.

**Coalescing.** The GPU serves memory requests for 32 threads (a *warp*) together. If
neighbouring threads read neighbouring addresses, it can do this with a few wide
transactions. That is why loops are written as
`for (k = threadIdx.x; k < K; k += blockDim.x)` (thread 0 reads element 0, thread 1
reads element 1, ...) and not as "thread t reads a contiguous chunk".

**Shared memory and `__syncthreads()`.** Threads of one block can share a small, fast
scratch memory declared with `__shared__`. `__syncthreads()` is a barrier: no thread
continues until all threads of the block reach it. Together they make the **block
reduction** in `src/cuda/block_reduce.cuh` possible. That reduction combines one value
per thread into a sum or maximum in log₂(blockDim) steps:

```
threads:    0   1   2   3   4   5   6   7
start:     a0  a1  a2  a3  a4  a5  a6  a7
stride 4:  a0+a4  a1+a5  a2+a6  a3+a7
stride 2:  (a0+a4)+(a2+a6)  (a1+a5)+(a3+a7)
stride 1:  total
```

Rule: **every thread of the block must call the reduction**. If some threads return
early and skip a `__syncthreads()`, the kernel can hang. Kernels that use reductions
therefore avoid early `return`s.

**Asynchronous execution.** A launch only *queues* work and returns to the CPU
immediately. Work on the same stream runs in order, which is why the forward pass can
queue ~260 kernels without waiting. Before reading a result on the CPU, you must
synchronize. `cudaMemcpy` does this implicitly, and `cuda::argmax` calls
`cudaStreamSynchronize`.

**Launch overhead.** Each launch costs a few microseconds of CPU and driver time, about
10 µs under WSL2. A kernel on a 2048-element vector finishes in far less time than
that, so tiny kernels are limited by launch cost, not by the GPU (see §8).

**Error checking.** `CUDA_CHECK(call)` aborts with file, line and message if a CUDA call
fails. After each launch, `CUDA_CHECK(cudaGetLastError())` catches bad launch
configurations.

---

## 5. The kernels, one by one

Each subsection covers: what the kernel is for, the math, how the CPU code works, how
the CUDA code maps threads, and details worth knowing.

### 5.1 `embedding` — token id → vector

*Files:* `src/cpu/embedding.cpp`, `src/cuda/embedding.cu`

**Role.** This is the first operation of every step. It turns a token id into a vector
of length `dim`.

**Math.** `out = table[token, :]`. It copies one row; there is no arithmetic.

**CPU.** It checks that `0 <= token < vocab_size`, then copies `dim` floats from offset
`token * dim`. The offset is computed in `size_t`, because `vocab_size * dim` can be
larger than an `int` can hold for big models.

**CUDA.** One thread per element: `out[i] = table[token*dim + i]`. The token id is a CPU
`int` passed as a kernel argument. The host wrapper validates it before launching,
because a kernel cannot easily report errors.

**Notes.** With tied embeddings (Llama 3.2 1B/3B) the same table is reused, transposed
in meaning, as the `lm head` weight. `llama3_forward.cpp` handles this with
`tied_embedding`.

### 5.2 `rmsnorm` — normalization

*Files:* `src/cpu/rmsnorm.cpp`, `src/cuda/rmsnorm.cu`

**Role.** It runs before attention, before the feed-forward block, and once at the end:
`2 * n_layers + 1` times per token. It keeps the scale of the activations stable.

**Math.**

```
rms    = sqrt( (x₀² + x₁² + … + x_{dim-1}²) / dim + eps )      eps = 1e-5
out[i] = x[i] / rms * weight[i]
```

Unlike LayerNorm, RMSNorm does not subtract the mean and has no bias.

**CPU.** Three readable steps: sum of squares, `inv_rms = 1/sqrt(mean + eps)`, then
scale. It is safe in place, because all of `x` is read in step 1 before anything is
written in step 3.

**CUDA.** **One block of 256 threads** handles the whole vector, because every output
needs the same number (the rms of the whole vector):

1. thread `t` sums the squares of `x[t], x[t+256], x[t+512], …`,
2. `block_reduce_sum` adds up the 256 partial sums, and every thread receives the total,
3. each thread scales its own elements by `rsqrtf(mean + eps)` (`rsqrtf` = 1/√).

**Notes.** A 2048-float vector is too little work to occupy a GPU. This kernel uses
one block on one of the GPU's 24 SMs and is dominated by launch overhead. That is fine,
because it accounts for less than 4% of a token's time.

### 5.3 `matmul` — every linear layer (where the time goes)

*Files:* `src/cpu/matmul.cpp`, `src/cuda/matmul.cu`

**Role.** The q/k/v/o projections, the three FFN matrices, and the lm head. Together
they account for **over 90% of the time** of a decode step (§8).

**Math.** `out[m, n] = Σ_k x[m, k] · w[n, k]`, with `x: [M, K]`, `w: [N, K]` and
`out: [M, N]`. When generating, `M = 1`: one token, so this is a matrix–vector product.
`M > 1` occurs when a prompt is processed several tokens at once ("prefill").

**CPU.**

```cpp
#pragma omp parallel for            // CPU threads split the weight rows
for (int n = 0; n < N; ++n) {
  const float* w_row = w + n * K;
  for (int m = 0; m < M; ++m) {     // reuse w_row for every token while it is cached
    float sum = 0;
#pragma omp simd reduction(+ : sum)  // let the compiler use AVX for the dot product
    for (int k = 0; k < K; ++k) sum += x[m*K + k] * w_row[k];
    out[m*N + n] = sum;
  }
}
```

Without OpenMP the pragmas are ignored, and the result is the textbook triple loop.

**CUDA.** **One block of 256 threads per weight row `n`.** The grid has `N` blocks:

```
w row n :  [ k0 k1 k2 ... k255 | k256 ... k511 | ... ]
thread  :     0  1  2 ...  255 |    0 ...  255 | ...
```

For each input row `m`, every thread accumulates a partial dot product over its
strided elements, with coalesced reads. `block_reduce_sum` combines the 256 partials,
and thread 0 writes `out[m, n]`.

**Notes.**

* With `M = 1`, each weight is used for exactly **one multiply-add** after being loaded.
  The kernel is therefore limited by **memory bandwidth**, not compute. The GPU version
  reaches ~250 GB/s, essentially the RTX 4060 Laptop's peak. The CPU reaches ~42 GB/s,
  the practical DDR5 limit. That ratio, not the arithmetic, explains the ~5–6× speed-up
  (§8).
* For `M > 1` (prefill) the same weight row is reused `M` times, so the work becomes
  **compute-bound**. This simple kernel then reaches only ~270 GFLOP/s, a few percent
  of the GPU's fp32 peak. Real GEMM kernels tile the matrices through shared memory and
  registers, or use tensor cores. See §9.

### 5.4 `rope` — rotary position embedding

*Files:* `src/rope_frequencies.cpp`, `src/cpu/rope.cpp`, `src/cuda/rope.cu`

**Role.** Attention compares a query with keys using a dot product. RoPE rotates `q` and
`k` by an angle that depends on the token's position, so the dot product of a query at
position `p` with a key at position `s` depends only on the **distance** `p − s`. This
is Llama's only source of position information.

**Math.** Inside every head, numbers are taken in pairs `(x0, x1)`. Pair `i` is rotated
by `angle = pos · inv_freq[i]`:

```
x0' = x0·cos(angle) − x1·sin(angle)
x1' = x0·sin(angle) + x1·cos(angle)
```

**Which elements form a pair.** This code uses the **"rotate half"** layout of
HuggingFace checkpoints, where element `i` pairs with element `i + head_dim/2`:

```
head = [ a0 a1 a2 a3 | b0 b1 b2 b3 ]   pairs: (a0,b0) (a1,b1) (a2,b2) (a3,b3)
```

Meta's original checkpoints pair neighbours instead: `(x0,x1), (x2,x3)…`. The HF
conversion script permutes the rows of `Wq`/`Wk` so both give the same model. If you
load Meta-format weights, change the pairing to neighbours.

**Frequencies** (`rope_frequencies.cpp`, computed once on the CPU):

```
inv_freq[i] = 1 / theta^(2i / head_dim),    i = 0 … head_dim/2 − 1,   theta = 500000
```

Pair 0 turns by 1 radian per position. The last pair turns extremely slowly.

**Llama 3.1/3.2 scaling** (`RopeScaling` in `config.h`, `rope_type: "llama3"` in HF
configs). Each frequency has a *wavelength* `2π / inv_freq`, the number of positions
for one full turn. Short wavelengths, which carry local information, are kept.
Wavelengths longer than the original 8192-token context are divided by `factor`: 8 for
3.1, 32 for 3.2. Wavelengths in between are blended smoothly. This lets the model handle
128k-token contexts. The code follows HF's `_compute_llama3_parameters`.

**CPU.** A loop over heads calls `rotate_one_head` for every query head and every key
head. `v` is not rotated.

**CUDA.** **One thread per pair.** There are `(n_heads + n_kv_heads) · head_dim/2`
pairs. Thread ids below `n_heads·half` belong to `q`; the rest belong to `k`. From the
id, a thread computes `head = id / half` and `i = id % half`.

**Notes.** The angle is computed with `sinf`/`cosf` (accurate versions), not the fast
`__sinf` intrinsics, which lose accuracy at large angles: positions × 1 rad can reach
thousands of radians. CPU and GPU agree to ~1e-7.

### 5.5 `softmax` — scores → probabilities

*Files:* `src/cpu/softmax.cpp`, `src/cuda/softmax.cu`

**Role.** It turns attention scores into weights that sum to 1. Used inside `attention`.
It is also the building block for temperature sampling.

**Math.** `softmax(x)[i] = exp(x[i] − max) / Σ_j exp(x[j] − max)`. Subtracting the row
maximum does not change the result, but it keeps every exponent ≤ 0, so `exp` cannot
overflow. The test feeds `[1000, 1000, 1000]`; without the subtraction the answer would
be `inf/inf = NaN`.

**CPU.** Per row: max, then `exp` and sum, then multiply by `1/sum`. The sum is kept in
`double`. Adding 128k floats one after another in float precision loses several digits.
The GPU's tree reduction adds numbers in pairs and is naturally more accurate. Before
this change the CPU was the *less* accurate of the two.

**CUDA.** **One block of 256 threads per row.** The same three steps, each shared by the
threads, using `block_reduce_max` and `block_reduce_sum`.

### 5.6 `attention` — causal grouped-query attention

*Files:* `src/cpu/attention.cpp`, `src/cuda/attention.cu`

**Role.** This is how the current token reads information from all previous tokens.

**Math.** For each query head `h`, over the `n_tokens = pos + 1` cached positions:

```
scores[t] = (q_h · k_t) / sqrt(head_dim)         t = 0 … n_tokens−1
weights   = softmax(scores)
out_h     = Σ_t weights[t] · v_t
```

*Causal* masking comes for free: the cache only holds positions `≤ pos`, so a token can
never look at the future.

**Grouped Query Attention (GQA).** Llama 3 has 32 query heads but only 8 KV heads.
Consecutive query heads share a KV head:

```
query heads : 0 1 2 3 | 4 5 6 7 | … | 28 29 30 31
kv head     :    0    |    1    | … |     7
kv_head = h / (n_heads / n_kv_heads)
```

This makes the KV cache 4× smaller, which is the reason GQA exists. The test with
`heads=4/4` checks that ordinary multi-head attention, where `n_kv_heads == n_heads`,
also works.

**CPU.** A loop over heads, parallelized with OpenMP because heads are independent. For
each head it runs the three steps above, reusing `cpu::softmax`.

**CUDA.** The work is split into three launches on the same stream:

| Step | Kernel | Mapping |
|---|---|---|
| 1 | `attention_scores_kernel` | one thread per score `(h, t)`; grid `(ceil(n_tokens/256), n_heads)` |
| 2 | `softmax` (from `softmax.cu`) | one block per head |
| 3 | `attention_output_kernel` | one thread per output element `(h, i)`; grid `n_heads`, block `head_dim` |

In step 3, neighbouring threads `i` read neighbouring elements of `v_t`, so the reads
are coalesced.

**Notes.** `scores` is a caller-provided scratch buffer `[n_heads, n_tokens]`, so the
kernel never allocates memory. Production engines fuse all three steps into one
"flash attention" kernel that never writes the scores to memory (§9). Attention's share
of the time grows with context length; at a 512-token context it is ~3–6%.

### 5.7 `swiglu` — feed-forward activation

*Files:* `src/cpu/swiglu.cpp`, `src/cuda/swiglu.cu`

**Role.** Llama's FFN is `W_down · (silu(W_gate·x) ⊙ (W_up·x))`. This kernel computes
the middle part. (Meta's code calls the matrices `w1` = gate, `w3` = up, `w2` = down.)

**Math.** `out[i] = silu(gate[i]) · up[i]`, with `silu(z) = z · sigmoid(z) = z / (1 + e^(−z))`.

**CPU / CUDA.** An element-wise loop, and a one-thread-per-element kernel. It may run in
place (`out == gate`), which the forward pass does to save a buffer.

### 5.8 `add` — residual connection

*Files:* `src/cpu/add.cpp`, `src/cuda/add.cu`

**Math.** `out[i] = a[i] + b[i]`. It is used twice per layer:
`x = x + attention(...)` and `x = x + ffn(...)`.

**CUDA.** The "hello world" of CUDA: one thread per element with a bounds check. This
is the best file to start with if CUDA is new to you.

### 5.9 `argmax` — greedy sampling

*Files:* `src/cpu/argmax.cpp`, `src/cuda/argmax.cu`

**Role.** It picks the token with the largest logit as the next token.

**CPU.** A linear scan with a strict `>`, so on ties the **smallest index** wins.

**CUDA.** **One block of 1024 threads.** Each thread scans `x[t], x[t+1024], …` and
keeps its best `(value, index)` pair. A shared-memory tree reduction on pairs then finds
the block's best pair, using the same tie rule (equal values → smaller index), so the
GPU result is identical to the CPU result. Thread 0 stores the index in a `__device__`
global variable. The host function copies that one `int` back with
`cudaMemcpyFromSymbolAsync`, synchronizes, and returns it.

**Notes.** This is the only kernel that waits for the GPU, because it returns a value
to the CPU. In the forward pass, that one synchronization per token is also what makes
the per-token timing honest.

---

## 6. Putting it together: the forward pass

`apps/llama3_forward.cpp` contains two classes with the same structure:

* `CpuLlama3`: weights in `std::vector<float>`, kernels from `llama3::cpu`
* `CudaLlama3`: weights copied once into `DeviceBuffer<float>`, kernels from `llama3::cuda`

Their `forward(token, pos)` functions are line-for-line identical apart from the
namespace and `.data()` vs `.get()`:

```cpp
embedding(table, token, vocab, dim, x);
for each layer l:
  rmsnorm(x, attn_norm, dim, eps, xb);
  matmul(xb, wq, 1, dim, dim, q);
  matmul(xb, wk, 1, kv_dim, dim, k_cache_l + pos*kv_dim);   // straight into the cache
  matmul(xb, wv, 1, kv_dim, dim, v_cache_l + pos*kv_dim);
  rope(q, k_now, n_heads, n_kv_heads, head_dim, pos, inv_freq);
  attention(q, k_cache_l, v_cache_l, n_heads, n_kv_heads, head_dim, pos+1, scores, att);
  matmul(att, wo, 1, dim, dim, xb);
  add(x, xb, dim, x);
  rmsnorm(x, ffn_norm, dim, eps, xb);
  matmul(xb, w_gate, 1, hidden, dim, gate);
  matmul(xb, w_up,   1, hidden, dim, up);
  swiglu(gate, up, hidden, gate);
  matmul(gate, w_down, 1, dim, hidden, xb);
  add(x, xb, dim, x);
rmsnorm(x, final_norm, dim, eps, x);
matmul(x, lm_head, 1, vocab, dim, logits);
next = argmax(logits, vocab);
```

Things worth noticing:

* **All buffers are allocated once**, in the constructor. A decode step only launches
  kernels and never allocates memory.
* **The KV cache is the model's memory of the past.** Each step writes one row per
  layer. Positions `0..pos` are read by attention.
* **Weights are random** (uniform ±1/√in_features, a typical initialization), so the
  generated ids are meaningless. To run a real model you would need to fill `Weights`
  from a checkpoint (the order is the same as HF's `model.layers.N.self_attn.q_proj`
  etc.) and add a tokenizer. The kernels would not change.
* The program first runs greedy decoding on the CPU. It then feeds **the same tokens**
  to the GPU, so the logits of every step can be compared.

---

## 7. How correctness is checked

`apps/test_kernels.cpp` has two parts:

1. **CPU vs hand-computed answers** on tiny inputs: softmax of `[1, 2, 3]`, RMSNorm of
   `[3, 4]`, RoPE at position 0 (no change), RoPE rotating `(1, 0)` by exactly 1 radian,
   attention with one cached token (must return `v`), argmax ties, and so on.
2. **CUDA vs CPU** on random data at real Llama sizes, plus awkward sizes: lengths that
   are not multiples of 256, 1-element rows, `M = 4` matmuls, 128k-long softmax rows,
   contexts that span several blocks.

**Why compare with a tolerance?** Float addition is not associative: `(a + b) + c` can
differ from `a + (b + c)` in the last bit. The CPU adds a dot product left to right,
while the GPU adds 256 partial sums in a tree. Both are correct but differ by ~1e-7
relative. The tests use

```
relative error = max |cpu − gpu| / max |cpu|   ≤ 1e-4
```

Observed values are ~1e-7. Copy-only kernels (`embedding`, `add`) must match exactly.
The full forward pass through 16 layers stays at ~1e-6, and both devices pick the same
token at every step.

---

## 8. Performance: CPU vs GPU, and why

Measured on an AMD Ryzen 9 7940H (8 cores, DDR5) and an RTX 4060 Laptop GPU (8 GB,
~256 GB/s), WSL2, fp32, 8 CPU threads. Run `bench_kernels` and `llama3_forward` to get
your own numbers.

### 8.1 End-to-end

| | CPU | GPU | Speed-up |
|---|---|---|---|
| Llama 3.2 1B shapes, ms/token | 120.7 | 22.2 | 5.4× |
| Llama 3 8B shapes (2 layers), ms/token | 92.1 | 17.4 | 5.3× |
| tiny model, ms/token | 0.54 | 0.76 | 0.7× (CPU faster!) |

### 8.2 Generation is limited by memory bandwidth

A decode step multiplies every weight by exactly one number: 2 FLOPs (a multiply and an
add) per 4-byte weight, or **0.5 FLOP/byte**. Both processors can compute far faster
than they can read memory:

| | Compute (fp32) | Memory bandwidth | FLOP/byte at balance |
|---|---|---|---|
| RTX 4060 Laptop | ~15 TFLOP/s | ~256 GB/s | ~60 |
| Ryzen 7940H, 8 cores | ~1 TFLOP/s | ~50–60 GB/s practical | ~20 |

At 0.5 FLOP/byte both are far below balance, so **time per token ≈ weight bytes ÷
memory bandwidth**:

| | Prediction | Measured |
|---|---|---|
| 1B on GPU: 4.94 GB ÷ 250 GB/s | 19.8 ms | 22.2 ms |
| 1B on CPU: 4.94 GB ÷ 42 GB/s | 118 ms | 120.7 ms |

So the GPU wins by the **ratio of memory bandwidths** (~5–6×), not by its 15× larger
compute. This also explains why real engines store weights in fp16/bf16 (2× fewer
bytes) or int8/int4 (4–8× fewer bytes): fewer bytes per token means proportionally more
tokens per second.

The benchmark's "estimated cost of one token" table shows where the time goes
(1B, GPU): FFN gate+up 41%, down 21%, lm head 17%, q/k/v/o 6%, attention 6%. Everything
else is a few percent. **Matmul is everything.**

### 8.3 Small kernels: the GPU loses

`add`, `rmsnorm`, `rope` and `embedding` on a 2048-element vector take less than 1 µs of
real GPU work. Each launch, however, costs ~10 µs of fixed overhead, more under WSL2
than on native Linux. The CPU does the same work in ~0.2 µs:

| Kernel (1B sizes) | CPU | GPU |
|---|---|---|
| add, 2048 | 0.0002 ms | 0.011 ms |
| rmsnorm, 2048 | 0.0017 ms | 0.026 ms |

The same effect makes the **tiny model faster on the CPU**. Its weights are only 20 MB,
but every token still needs ~70 kernel launches. On the GPU, launch overhead is the
whole cost. Real engines record the whole step once as a **CUDA Graph** and replay it,
or **fuse** small kernels into the matmuls (for example rmsnorm, then matmul, then add).

### 8.4 Benchmark pitfall: caches

`matmul_q_o` at 1B sizes shows **528 GB/s** on the GPU, which is more than the memory
can deliver. The 2048×2048 weight is 16.8 MB and fits in the GPU's 32 MB L2 cache, so
repeated benchmark iterations read it from cache. In the real model the weights of the
other layers evict it, so the forward pass runs at DRAM speed. Small CPU matrices show
the same effect with the CPU caches. Always check microbenchmarks against end-to-end
numbers.

### 8.5 Prefill (M > 1) is a different problem

With 32 tokens at once, each weight is reused 32 times, which is 16 FLOP/byte. The work
becomes compute-bound:

| 8B gate matmul, 32×14336×4096 = 3.8 GFLOP | time | rate |
|---|---|---|
| CPU | 37.8 ms | ~100 GFLOP/s |
| GPU (this simple kernel) | 13.7 ms | ~275 GFLOP/s |

The GPU only wins by 2.7× here, because this kernel is the wrong tool for compute-bound
work. It does one block reduction per output and gets no data reuse in registers.
cuBLAS or a tiled kernel reaches several TFLOP/s on the same GPU. This is the clearest
example of "readable" costing "fast".

### 8.6 CPU threads

| 1B forward pass | ms/token |
|---|---|
| 1 thread | 162.6 |
| 8 threads (physical cores) | 120.7 |
| 16 threads (all logical CPUs, WSL2) | up to 3× slower than 8 |

Eight threads are only 1.35× faster than one: a few cores already saturate DDR5
bandwidth. Using hyper-threads hurts. Each parallel loop is short, every thread waits
at the loop's end for the slowest one, and in a VM some virtual CPUs are often paused.
For this reason the programs default to half the logical CPUs (see `configure_cpu_threads`
in `apps/app_utils.h`); override with `OMP_NUM_THREADS`.

---

## 9. Where to go next (optimizations)

The code deliberately leaves out these well-known techniques. Each one is a good
exercise. Measure before and after with `bench_kernels` and keep `test_kernels` green.

| Idea | Where | Expected effect |
|---|---|---|
| fp16 / bf16 weights (convert to float while loading) | matmul | ~2× faster decode, half the memory |
| int8 / int4 weights with per-group scales | matmul | ~4–8× fewer bytes |
| `float4` vectorized loads (16 bytes per load) | matmul, rmsnorm | fewer memory instructions |
| warp-shuffle reductions (`__shfl_down_sync`) instead of shared memory | block_reduce.cuh | fewer `__syncthreads`, smaller blocks |
| one warp per weight row, several rows per block | matmul | better for small `K` |
| tiled GEMM in shared memory, or cuBLAS / tensor cores | matmul with M > 1 | 10×+ for prefill |
| fuse q/k/v into one matmul, and gate/up into one | forward pass | fewer launches, x read once |
| fuse rmsnorm into the following matmul, and add into the preceding one | forward pass | fewer launches |
| flash attention (one fused kernel, online softmax) | attention | long contexts, no scores buffer |
| CUDA Graphs (record one decode step, replay it) | forward pass | removes ~260 launches' overhead |
| batch prompt tokens (prefill with M = prompt length) | forward pass | prompt processed much faster |

---

## 10. Relation to the rest of the repository

The main framework (`source/`, `include/`) follows the structure of KuiperLLama: a
`Tensor` class, layer classes (`op::Layer`), and tensor-based kernels in
`source/op/kernels/{cpu,cuda}`. While reading it for this work, these points came up.
They are listed here so they can be fixed separately; none of them affect
`llama3_kernels/`.

**The framework does not currently build or link:**

* `include/op/encode.h` includes `absl/...` headers, which are only available with
  `-DUSE_CPM=ON`.
* Declared but not implemented anywhere: `op::BaseLayer`/`Layer`/`LayerParam`,
  `VecAddLayer`, `EmbeddingLayer`, `MatmulLayer`, `MultiHeadAttention`, `RmsNormLayer`,
  `RoPELayer`, `model::Model`, `sampler::ArgmaxSampler`, and `RawModelData*`.
* `source/model/model.cpp` defines `int main()` inside the shared library.

**Kernel issues noticed in `source/op/kernels/`:**

| File | Issue |
|---|---|
| `cuda/rope_kernel.cu` | `if (idx > total_pairs) return;` should be `>=`; one extra thread can write past the last head. |
| `cuda/emb_kernel.cu` | if the input is already on the GPU, `input_cu` stays empty and a null pointer is passed to the kernel. It also always launches 512 blocks. |
| `cpu/emb_kernel.cpp` | `token > vocab_size` should be `token >= vocab_size`. |
| `cuda/matmul_kernel.cu` | ignores the `scale` argument (the CPU version applies it). The header's declarations are also pasted at the end of the `.cu` file. |
| `kernels_interfaces.cpp` | `get_softmax_kernel`, `get_scale_kernel` and `get_scale_sum_kernel` have no CUDA version (`LOG(FATAL)`). This is fine today, because CUDA MHA is a single fused kernel. |
| `cpu/rope_kernel.cpp`, `cuda/rope_kernel.cu` | no Llama 3.1/3.2 RoPE frequency scaling, so models with `rope_type: llama3` would decode incorrectly at all positions. |
| `source/base/alloc_cu.cpp` | `release()` returns early when no *small* buffer was ever allocated, so *big* buffers are never marked free for reuse. |

`llama3_kernels/` can serve as a reference when fixing these. Its functions take raw
pointers, so the existing tensor-based wrappers could call them, for example
`llama3::cuda::rmsnorm(input.ptr<float>(), weight.ptr<float>(), dim, eps, out.ptr<float>(), stream)`.
