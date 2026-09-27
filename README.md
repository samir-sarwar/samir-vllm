# samir-vllm

Building a small Llama inference engine in C++ and CUDA, one piece at a time. This is a learning project following the path from a prompt to a generated token, inspired by the [tiny-vllm course](https://github.com/jmaczan/tiny-vllm).

**Current checkpoint:** prompt prefill through the attention block of **layer 0**. The engine does not generate tokens yet.

## Watch the build

Click a thumbnail to watch. The videos are listed newest first, with notes for each step.

| | |
| :---: | :---: |
| [<img src="https://img.youtube.com/vi/8_HJH7pbw78/maxresdefault.jpg" width="320" alt="Part 6: Q, K, V projection with cuBLAS">](https://youtu.be/8_HJH7pbw78)<br>[**Part 6 · Q/K/V with cuBLAS**](https://youtu.be/8_HJH7pbw78) · [Notes](proper_notes/youtubePT6_qkv_projection_notes.md) | [<img src="https://img.youtube.com/vi/4gL0hd9UEAY/maxresdefault.jpg" width="320" alt="Part 4: Rotary position embeddings">](https://youtu.be/4gL0hd9UEAY)<br>[**Part 4 · RoPE**](https://youtu.be/4gL0hd9UEAY) · [Notes](proper_notes/youtubePT4_rope_notes.md) |
| [<img src="https://img.youtube.com/vi/my8VrOUVWo0/maxresdefault.jpg" width="320" alt="Part 3: RMSNorm kernel">](https://youtu.be/my8VrOUVWo0)<br>[**Part 3 · RMSNorm kernel**](https://youtu.be/my8VrOUVWo0) · [Notes](proper_notes/youtubePT3_rmsnorm_notes.md) | [<img src="https://img.youtube.com/vi/0V1lVIzGyqs/maxresdefault.jpg" width="320" alt="Part 2: Embedding gather kernel">](https://youtu.be/0V1lVIzGyqs)<br>[**Part 2 · Embedding gather**](https://youtu.be/0V1lVIzGyqs) · [Notes](proper_notes/youtubePT2_embedding_gather_notes.md) |
| [<img src="https://img.youtube.com/vi/39_gWVbYgB4/maxresdefault.jpg" width="320" alt="Part 1: Loading SafeTensors weights">](https://youtu.be/39_gWVbYgB4)<br>[**Part 1 · Loading SafeTensors weights**](https://youtu.be/39_gWVbYgB4) · [Notes](proper_notes/youtubePT1_notes.md) | [<img src="https://img.youtube.com/vi/ef0mOukUE6U/maxresdefault.jpg" width="320" alt="Part 0: AI inference explained">](https://youtu.be/ef0mOukUE6U)<br>[**Part 0 · AI inference explained**](https://youtu.be/ef0mOukUE6U) · [Notes](proper_notes/youtubePT0_inference_notes.md) |

## Where the engine is now

- Loads Llama 3.2 1B Instruct BF16 weights from `model.safetensors` onto the GPU; a memory-mapped loader is also implemented.
- Tokenizes locally with a Unicode-aware BPE tokenizer and supports the Instruct chat format.
- Runs embedding gather, RMSNorm, cuBLAS Q/K/V projections, RoPE, causal grouped-query attention, stable softmax, value mixing, output projection, and a residual add for layer 0 during prefill.

Next: finish the layer with post-attention normalization and the MLP, then run all 16 layers and produce next-token logits. KV caching and batching come later. The model shape, prompt, and paths are currently hard-coded.

## Build

Requires an NVIDIA GPU, the CUDA Toolkit, CMake 3.24+, a C++17 compiler, ICU development libraries, and the `nlohmann/json` header. Place the model weights at `models/llama-3.2-1b-instruct/model.safetensors` (weights are not included), then copy the checked-in tokenizer to the path expected by the program:

```bash
mkdir -p models/llama-3.2-1b-instruct
cp models/Llama-3.2-1B-Instruct/original/tokenizer.model models/llama-3.2-1b-instruct/
cmake -S . -B build
cmake --build build -j
./build/samir-vllm
./build/tokenizer-test models/Llama-3.2-1B-Instruct/original/tokenizer.model
```

## Project layout

```text
src/           Model loading, prefill, CUDA kernels, and tokenizer
include/       Kernel and tokenizer headers
tests/         Tokenizer tests
proper_notes/  Notes accompanying the videos
notes.md       Working notes
```
