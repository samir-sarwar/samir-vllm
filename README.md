# samir-vllm

> Building a small LLM inference engine from the ground up with C++ and CUDA.

This project is my hands-on attempt to understand what happens between a prompt and a generated token: loading a model's weights, moving data to the GPU, and eventually running the transformer efficiently.

[![Watch the first build video](https://img.youtube.com/vi/ef0mOukUE6U/hqdefault.jpg)](https://youtu.be/ef0mOukUE6U?si=DwP8ENN0enIp4HGy)

**▶ [Watch the first video in the series](https://youtu.be/ef0mOukUE6U?si=DwP8ENN0enIp4HGy)** — more build updates are on the way.

## What works so far

- A C++17 / CUDA project scaffold built with CMake.
- CUDA device inspection and a GPU warm-up kernel.
- Direct loading of a `model.safetensors` file: parse its JSON metadata, read tensor offsets, and copy raw weights into GPU memory.
- A typed weight layout for the Llama 3.2 1B Instruct architecture, with direct pointers to embedding, normalization, attention, and MLP tensors for each of its 16 layers.
- A local tokenizer implementation and tokenizer test target.

## What I’m building toward

The next milestones are token embedding lookup, transformer-layer CUDA kernels, and end-to-end autoregressive generation. Along the way, I plan to add the pieces that make inference efficient—attention, KV-cache management, and batching—while documenting what I learn.

## Build

### Requirements

- A CUDA-capable NVIDIA GPU and CUDA Toolkit
- CMake 3.24+
- A C++17 compiler
- Model weights at `models/llama-3.2-1b-instruct/model.safetensors`

```bash
cmake -S . -B build
cmake --build build -j
./build/samir-vllm
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
