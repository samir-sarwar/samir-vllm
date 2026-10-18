# samir-vllm

> Teaching LLM inference from first principles by building a small engine in C++ and CUDA.

The series follows what happens between a prompt and a generated token, from loading model weights to running each part of the transformer on the GPU.

## Build series

<table width="100%">
<tr>
<td align="center" width="50%"><a href="https://youtu.be/ef0mOukUE6U"><img src="https://img.youtube.com/vi/ef0mOukUE6U/maxresdefault.jpg" width="300" alt="Part 0: AI inference explained"><br><strong>Part 0 · AI inference explained</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/39_gWVbYgB4"><img src="https://img.youtube.com/vi/39_gWVbYgB4/maxresdefault.jpg" width="300" alt="Part 1: Loading SafeTensor weights"><br><strong>Part 1 · Loading model weights</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/0V1lVIzGyqs"><img src="https://img.youtube.com/vi/0V1lVIzGyqs/maxresdefault.jpg" width="300" alt="Part 2: Embedding gather kernel"><br><strong>Part 2 · Embedding gather</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/my8VrOUVWo0"><img src="https://img.youtube.com/vi/my8VrOUVWo0/maxresdefault.jpg" width="300" alt="Part 3: RMSNorm kernel"><br><strong>Part 3 · RMSNorm</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/4gL0hd9UEAY"><img src="https://img.youtube.com/vi/4gL0hd9UEAY/maxresdefault.jpg" width="300" alt="Part 4: Rotary position embeddings"><br><strong>Part 4 · RoPE</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/8_HJH7pbw78"><img src="https://img.youtube.com/vi/8_HJH7pbw78/maxresdefault.jpg" width="300" alt="Part 5: Q, K, and V projections with cuBLAS"><br><strong>Part 5 · Q/K/V with cuBLAS</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/tTULacSPXkc"><img src="https://img.youtube.com/vi/tTULacSPXkc/maxresdefault.jpg" width="300" alt="Part 6: Multi-head self-attention"><br><strong>Part 6 · Self-attention</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/HjllfB1IPrE"><img src="https://img.youtube.com/vi/HjllfB1IPrE/maxresdefault.jpg" width="300" alt="Part 7: Feed-forward network and SwiGLU"><br><strong>Part 7 · Feed-forward network</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/yFwPpH1Hb14"><img src="https://img.youtube.com/vi/yFwPpH1Hb14/maxresdefault.jpg" width="300" alt="Part 8: Decode and KV cache"><br><strong>Part 8 · Decode and KV cache</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/ei3M0Ex0qHA"><img src="https://img.youtube.com/vi/ei3M0Ex0qHA/maxresdefault.jpg" width="300" alt="Part 9: Static and continuous batching"><br><strong>Part 9 · Static and continuous batching</strong></a></td>
</tr>
</table>

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
- Persistent scratch buffers and one cuBLAS handle shared by prefill and decode. Q storage is reused for output/MLP projections, and normalization storage is reused for attention output.
- Batched decode for up to two requests, with packed projections and a separate contiguous KV cache and attention length per sequence.
- Static batching and a continuous scheduler that prefills queued prompts into freed slots between decode steps.

Part 8 follows these changes in [Prefill, Decode, and the KV Cache](proper_notes/youtubePT8_prefill_decode_kv_cache.md).

## Next steps

Paged KV storage comes next. This version keeps two fixed sequence slots and contiguous caches, with readable, explicit operations.

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
# Continuous batching (default): two active slots, three queued requests.
./build/samir-vllm "Say hello." 64 "Explain the KV cache." "What is RoPE?"
# Static batching: wait for both requests in a group before admitting more.
./build/samir-vllm --static "Say hello." 64 "Explain the KV cache." "What is RoPE?"
./build/tokenizer-test models/llama-3.2-1b-instruct/tokenizer.model
```

After the optional `--static` or `--continuous` flag, the first argument is a prompt and the second is `max_new_tokens` (default: 128). Any remaining arguments are additional prompts; the token budget applies to each request. Every prompt is formatted as an Instruct user message. Generation stops independently at a model stop token or the output/context limit, and each completed response is printed with its token IDs and request number.

Prompts are prefilled one at a time, pausing decode. Each decode step packs one token from every active sequence into the shared buffers. Attention reads only that sequence's cache; requests can have different prompt and output lengths. Continuous batching immediately reuses finished slots, while static batching admits the next group after the current group finishes.

## Project layout

```text
src/           Engine entry point, CUDA kernels, and tokenizer implementation
include/       Public kernel and tokenizer headers
tests/         Tokenizer tests
python/        Tokenizer reference script
proper_notes/  Notes for the video series
notes.md       Working notes from the build
```
