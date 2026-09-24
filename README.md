# samir-vllm

> Building a small LLM inference engine from the ground up with C++ and CUDA.

This project is my hands-on attempt to understand what happens between a prompt and a generated token: loading model weights, moving data to the GPU, and building the transformer execution path piece by piece.

## Build series

| Part 0 | Part 1 |
| :---: | :---: |
| [![Part 0: AI inference explained](https://img.youtube.com/vi/ef0mOukUE6U/hqdefault.jpg)](https://youtu.be/ef0mOukUE6U?si=DwP8ENN0enIp4HGy) | [![Part 1: Loading SafeTensor weights](https://img.youtube.com/vi/39_gWVbYgB4/hqdefault.jpg)](https://youtu.be/39_gWVbYgB4?si=cmMyfK0BywNGfd5o) |
| [AI inference, simply explained](https://youtu.be/ef0mOukUE6U?si=DwP8ENN0enIp4HGy) | [Loading SafeTensor model weights to the GPU](https://youtu.be/39_gWVbYgB4?si=cmMyfK0BywNGfd5o) |

| Part 2 | Part 3 |
| :---: | :---: |
| [![Part 2: Embedding gather kernel](https://img.youtube.com/vi/0V1lVIzGyqs/hqdefault.jpg)](https://youtu.be/0V1lVIzGyqs?si=Uut4WlzZrYMEpS6R) | [![Part 3: RMSNorm kernel](https://img.youtube.com/vi/my8VrOUVWo0/hqdefault.jpg)](https://youtu.be/my8VrOUVWo0?si=cuge6AsVmI3MfYUM) |
| [Writing the embedding-table gather kernel](https://youtu.be/0V1lVIzGyqs?si=Uut4WlzZrYMEpS6R) | [Writing the RMSNorm kernel](https://youtu.be/my8VrOUVWo0?si=cuge6AsVmI3MfYUM) |

### Part 4

[![Part 4: Rotary Position Embedding (RoPE)](https://img.youtube.com/vi/4gL0hd9UEAY/hqdefault.jpg)](https://youtu.be/4gL0hd9UEAY?si=ii5LioF0gTTnEki6)

**[Rotary Position Embedding (RoPE), explained simply](https://youtu.be/4gL0hd9UEAY?si=ii5LioF0gTTnEki6)**

More videos are on the way as the engine develops.

## What works so far

- A C++17 / CUDA project, built with CMake and linked with ICU for Unicode-aware tokenization.
- Direct `model.safetensors` loading: parse metadata, validate tensor offsets, and copy the raw BF16 weights to GPU memory. There is also a memory-mapped loader implementation to avoid an extra CPU-side copy.
- A fixed Llama 3.2 1B Instruct weight layout, with direct pointers to the embedding, normalization, attention, and MLP tensors across all 16 layers.
- A local BPE tokenizer that supports Unicode, special tokens, encode/decode, and the Instruct chat prompt format; its expected output is covered by a dedicated test executable.
- A CUDA embedding-gather kernel and prefill path that copies prompt token IDs to the GPU and allocates activations.
- A CUDA RMSNorm kernel for the 2,048-wide hidden state: it reduces in FP32, applies the learned BF16 weights, and is now run as the first operation of layer 0 during prefill.
- RoPE support for the model’s scaled rotary frequencies: GPU sine/cosine tables are initialized for a 2,048-token context, and a CUDA kernel is ready to rotate Q/K projection buffers.

## What I’m building toward

Next up is Q/K/V projection and connecting RoPE to the query/key buffers. After that come grouped-query attention, output projection and residuals, then the MLP. The larger goal is autoregressive token generation, followed by KV-cache management and batching.

## Build

### Requirements

- A CUDA-capable NVIDIA GPU and CUDA Toolkit
- CMake 3.24+
- A C++17 compiler
- ICU development libraries
- Model weights at `models/llama-3.2-1b-instruct/model.safetensors`
- The matching tokenizer at `models/llama-3.2-1b-instruct/tokenizer.model`

```bash
cmake -S . -B build
cmake --build build -j
./build/samir-vllm
./build/tokenizer-test models/llama-3.2-1b-instruct/tokenizer.model
```

## Project layout

```text
src/        Engine entry point, CUDA kernels, and tokenizer implementation
include/    Public kernel and tokenizer headers
tests/      Tokenizer tests
python/     Tokenizer helper script
notes.md    Working notes from the build
```

This is a learning project in active development; interfaces and assumptions will evolve as the engine grows.
