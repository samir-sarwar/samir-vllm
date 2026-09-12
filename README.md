# samir-vllm

> Building a small LLM inference engine from the ground up with C++ and CUDA.

This project is my hands-on attempt to understand what happens between a prompt and a generated token: loading model weights, moving data to the GPU, and building the transformer execution path piece by piece.

## Build series

| Part 0 | Part 1 | Part 2 |
| :---: | :---: | :---: |
| [![Part 0: AI inference explained](https://img.youtube.com/vi/ef0mOukUE6U/hqdefault.jpg)](https://youtu.be/ef0mOukUE6U?si=DwP8ENN0enIp4HGy) | [![Part 1: Loading SafeTensor weights](https://img.youtube.com/vi/39_gWVbYgB4/hqdefault.jpg)](https://youtu.be/39_gWVbYgB4?si=cmMyfK0BywNGfd5o) | [![Part 2: Embedding gather kernel](https://img.youtube.com/vi/0V1lVIzGyqs/hqdefault.jpg)](https://youtu.be/0V1lVIzGyqs?si=Uut4WlzZrYMEpS6R) |
| [AI inference, simply explained](https://youtu.be/ef0mOukUE6U?si=DwP8ENN0enIp4HGy) | [Loading SafeTensor model weights to the GPU](https://youtu.be/39_gWVbYgB4?si=cmMyfK0BywNGfd5o) | [Writing the embedding-table gather kernel](https://youtu.be/0V1lVIzGyqs?si=Uut4WlzZrYMEpS6R) |

More videos are on the way as the engine develops.

## What works so far

- A C++17 / CUDA project, built with CMake and linked with ICU for Unicode-aware tokenization.
- Direct `model.safetensors` loading: parse metadata, validate tensor offsets, and copy the raw BF16 weights to GPU memory. There is also a memory-mapped loader implementation to avoid an extra CPU-side copy.
- A fixed Llama 3.2 1B Instruct weight layout, with direct pointers to the embedding, normalization, attention, and MLP tensors across all 16 layers.
- A local BPE tokenizer that supports Unicode, special tokens, encode/decode, and the Instruct chat prompt format; its expected output is covered by a dedicated test executable.
- A CUDA embedding-gather kernel and prefill path that copies prompt token IDs to the GPU, allocates activations, and runs the first layer’s RMSNorm.
- Initial RoPE support: precomputed GPU sine/cosine tables for the model’s scaled rotary frequencies, plus a CUDA kernel ready to rotate query and key projections.

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
