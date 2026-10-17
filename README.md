# samir-vllm

> Teaching LLM inference from first principles by building a small engine in C++ and CUDA.

The series follows what happens between a prompt and a generated token, from loading model weights to running each part of the transformer on the GPU.

## Build series

| Part 0 | Part 1 |
| :---: | :---: |
| [![Part 0: AI inference explained](https://img.youtube.com/vi/ef0mOukUE6U/hqdefault.jpg)](https://youtu.be/ef0mOukUE6U?si=DwP8ENN0enIp4HGy) | [![Part 1: Loading SafeTensor weights](https://img.youtube.com/vi/39_gWVbYgB4/hqdefault.jpg)](https://youtu.be/39_gWVbYgB4?si=cmMyfK0BywNGfd5o) |
| [AI inference, simply explained](https://youtu.be/ef0mOukUE6U?si=DwP8ENN0enIp4HGy) | [Loading SafeTensor model weights to the GPU](https://youtu.be/39_gWVbYgB4?si=cmMyfK0BywNGfd5o) |
| Part 2 | Part 3 |
| [![Part 2: Embedding gather kernel](https://img.youtube.com/vi/0V1lVIzGyqs/hqdefault.jpg)](https://youtu.be/0V1lVIzGyqs?si=Uut4WlzZrYMEpS6R) | [![Part 3: RMSNorm kernel](https://img.youtube.com/vi/my8VrOUVWo0/hqdefault.jpg)](https://youtu.be/my8VrOUVWo0?si=cuge6AsVmI3MfYUM) |
| [Writing the embedding-table gather kernel](https://youtu.be/0V1lVIzGyqs?si=Uut4WlzZrYMEpS6R) | [Writing the RMSNorm kernel](https://youtu.be/my8VrOUVWo0?si=cuge6AsVmI3MfYUM) |
| Part 4 | Part 5 |
| [![Part 4: Rotary Position Embedding (RoPE)](https://img.youtube.com/vi/4gL0hd9UEAY/hqdefault.jpg)](https://youtu.be/4gL0hd9UEAY) | [![Part 5: Q/K/V projections with cuBLAS](https://img.youtube.com/vi/8_HJH7pbw78/hqdefault.jpg)](https://youtu.be/8_HJH7pbw78) |
| [Rotary Position Embedding (RoPE), explained simply](https://youtu.be/4gL0hd9UEAY) | [Computing Q, K, and V with cuBLAS](https://youtu.be/8_HJH7pbw78) |
| Part 6 | Part 7 |
| [![Part 6: Multi-head self-attention](https://img.youtube.com/vi/tTULacSPXkc/hqdefault.jpg)](https://youtu.be/tTULacSPXkc) | [![Part 7: Feed-forward network and SwiGLU](https://img.youtube.com/vi/HjllfB1IPrE/hqdefault.jpg)](https://youtu.be/HjllfB1IPrE) |
| [Multi-head self-attention](https://youtu.be/tTULacSPXkc) | [Feed-forward network: SwiGLU and MLP](https://youtu.be/HjllfB1IPrE) |

### Part 8

[![Part 8: Decode and KV cache](https://img.youtube.com/vi/yFwPpH1Hb14/hqdefault.jpg)](https://youtu.be/yFwPpH1Hb14)

[Decode and KV cache, simply explained](https://youtu.be/yFwPpH1Hb14)

## What works so far

- A C++17 / CUDA project, built with CMake and linked with ICU for Unicode-aware tokenization.
- Direct `model.safetensors` loading: parse metadata, validate tensor offsets, and copy the raw BF16 weights to GPU memory. There is also a memory-mapped loader implementation to avoid an extra CPU-side copy.
- A fixed Llama 3.2 1B Instruct weight layout, with direct pointers to the embedding, normalization, attention, and MLP tensors across all 16 layers.
- A local BPE tokenizer that supports Unicode, special tokens, encode/decode, and the Instruct chat prompt format; its expected output is covered by a dedicated test executable.
- A CUDA embedding-gather kernel and prefill path that copies prompt token IDs to the GPU and allocates activations.
- A CUDA RMSNorm kernel for the 2,048-wide hidden state: it reduces in FP32 and applies the learned BF16 weights throughout the forward pass.
- Dense causal prefill now runs all 16 layers: Q/K/V projection, RoPE, grouped-query attention, output projection, the attention residual, post-attention RMSNorm, SwiGLU MLP, and the MLP residual.
- The final RMSNorm and tied embedding/LM-head projection produce FP32 logits; GPU argmax selects each next token.
- A contiguous BF16 KV cache for every layer. Prefill fills it from the whole prompt; decode processes one new token at its absolute RoPE position and attends to the cached prefix.
- Greedy response generation with Instruct chat formatting, end-of-turn/end-of-text stopping, and configurable output length within the 2,048-token context.

Part 8 follows these changes in [Prefill, Decode, and the KV Cache](proper_notes/youtubePT8_prefill_decode_kv_cache.md).

## Next steps

Batching, paged KV storage, and reusing scratch allocations across decode steps come later. This version runs one sequence at a time and prioritizes readable, explicit operations.

## Build

### Requirements

- A CUDA-capable NVIDIA GPU and CUDA Toolkit
- CMake 3.24+
- A C++17 compiler
- ICU development libraries
- The `nlohmann/json` header
- Model weights at `models/llama-3.2-1b-instruct/model.safetensors`
- The matching tokenizer at `models/llama-3.2-1b-instruct/tokenizer.model` (copy it from `models/Llama-3.2-1B-Instruct/original/tokenizer.model`)

```bash
mkdir -p models/llama-3.2-1b-instruct
cp models/Llama-3.2-1B-Instruct/original/tokenizer.model models/llama-3.2-1b-instruct/
cmake -S . -B build
cmake --build build -j
./build/samir-vllm "Explain the KV cache in two sentences." 128
./build/tokenizer-test models/llama-3.2-1b-instruct/tokenizer.model
./build/kv-cache-test
./build/kv-cache-test --model
```

The second argument is `max_new_tokens` (default: 128). The prompt is formatted as an Instruct user message; the executable prints the generated response and its token IDs. Generation stops at a model stop token or the output/context limit.

`kv-cache-test` checks GPU attention against CPU calculations without model weights. With `--model`, it also compares cached decode with dense full-prefix inference, including cache reset and bounds checks. Both modes require a CUDA GPU. An independent Hugging Face comparison is available with `python3 python/reference_generation.py "Say hello in one sentence." --max-new-tokens 32` (requires `torch`, `transformers`, and access to the matching model).

## Project layout

```text
src/           Engine entry point, CUDA kernels, and tokenizer implementation
include/       Public kernel and tokenizer headers
tests/         Tokenizer and GPU attention/KV-cache tests
python/        Tokenizer and generation reference scripts
proper_notes/  Notes for the video series
notes.md       Working notes from the build
```
