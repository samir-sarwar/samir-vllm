# samir-vllm

> Teaching LLM inference from first principles by building a small engine in C++ and CUDA.

The series follows what happens between a prompt and a generated token, from loading model weights to running each part of the transformer on the GPU.

## Build series

<table width="100%">
<tr>
<td align="center" width="50%"><a href="https://youtu.be/ef0mOukUE6U" title="Follow a prompt through an LLM, from tokenization to predicting the next token."><img src="https://img.youtube.com/vi/ef0mOukUE6U/maxresdefault.jpg" width="300" alt="Part 0: AI inference explained"><br><strong>Part 0 · AI inference explained</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/39_gWVbYgB4" title="Read SafeTensor metadata and load the Llama model weights into GPU memory."><img src="https://img.youtube.com/vi/39_gWVbYgB4/maxresdefault.jpg" width="300" alt="Part 1: Loading SafeTensor weights"><br><strong>Part 1 · Loading model weights</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/0V1lVIzGyqs" title="Write a CUDA gather kernel that turns token IDs into embedding vectors."><img src="https://img.youtube.com/vi/0V1lVIzGyqs/maxresdefault.jpg" width="300" alt="Part 2: Embedding gather kernel"><br><strong>Part 2 · Embedding gather</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/my8VrOUVWo0" title="Normalize hidden states with RMSNorm and implement the CUDA reduction from scratch."><img src="https://img.youtube.com/vi/my8VrOUVWo0/maxresdefault.jpg" width="300" alt="Part 3: RMSNorm kernel"><br><strong>Part 3 · RMSNorm</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/4gL0hd9UEAY" title="Understand rotary position embeddings and write the CUDA kernel for rotating query and key vectors."><img src="https://img.youtube.com/vi/4gL0hd9UEAY/maxresdefault.jpg" width="300" alt="Part 4: Rotary position embeddings"><br><strong>Part 4 · RoPE</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/8_HJH7pbw78" title="Use cuBLAS matrix multiplication to project hidden states into query, key, and value vectors."><img src="https://img.youtube.com/vi/8_HJH7pbw78/maxresdefault.jpg" width="300" alt="Part 5: Q, K, and V projections with cuBLAS"><br><strong>Part 5 · Q/K/V with cuBLAS</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/tTULacSPXkc" title="Build causal self-attention, from query-key scores and softmax to weighted values and the output projection."><img src="https://img.youtube.com/vi/tTULacSPXkc/maxresdefault.jpg" width="300" alt="Part 6: Multi-head self-attention"><br><strong>Part 6 · Self-attention</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/HjllfB1IPrE" title="Implement the SwiGLU feed-forward network, complete all transformer layers, and predict the first next token."><img src="https://img.youtube.com/vi/HjllfB1IPrE/maxresdefault.jpg" width="300" alt="Part 7: Feed-forward network and SwiGLU"><br><strong>Part 7 · Feed-forward network</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/yFwPpH1Hb14" title="Separate prefill from decode and reuse cached keys and values to generate a response one token at a time."><img src="https://img.youtube.com/vi/yFwPpH1Hb14/maxresdefault.jpg" width="300" alt="Part 8: Decode and KV cache"><br><strong>Part 8 · Decode and KV cache</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/ei3M0Ex0qHA" title="Generate for multiple requests together and compare static batching with a continuous scheduler."><img src="https://img.youtube.com/vi/ei3M0Ex0qHA/maxresdefault.jpg" width="300" alt="Part 9: Static and continuous batching"><br><strong>Part 9 · Static and continuous batching</strong></a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://youtu.be/As13goWXIx8" title="Build a paged KV cache with logical-to-physical block tables, allocate pages as requests grow, and reuse them on completion."><img src="https://img.youtube.com/vi/As13goWXIx8/maxresdefault.jpg" width="300" alt="Part 10: Paged KV cache"><br><strong>Part 10 · Paged KV cache</strong></a></td>
<td align="center" width="50%"><a href="https://youtu.be/fle2du_R8DI" title="Implement paged attention with online softmax, then connect the CUDA kernel to decode, batching, and page reclamation."><img src="https://img.youtube.com/vi/fle2du_R8DI/maxresdefault.jpg" width="300" alt="Finale: Paged attention"><br><strong>Finale · Paged attention</strong></a></td>
</tr>
</table>

Companion notes for every video are in the [study-notes index](proper_notes/README.md), with separate [Markdown](proper_notes/markdown/) and [PDF](proper_notes/pdf/) folders. You can also read the [complete PDF collection](proper_notes/pdf/samir_vllm_complete_study_notes.pdf).

## What works so far

- A C++17 / CUDA project, built with CMake and linked with ICU for Unicode-aware tokenization.
- Direct `model.safetensors` loading: parse metadata, validate tensor offsets, and copy the raw BF16 weights to GPU memory. There is also a memory-mapped loader implementation to avoid an extra CPU-side copy.
- A fixed Llama 3.2 1B Instruct weight layout, with direct pointers to the embedding, normalization, attention, and MLP tensors across all 16 layers.
- A local BPE tokenizer that supports Unicode, special tokens, encode/decode, and the Instruct chat prompt format; its expected output is covered by a dedicated test executable.
- A CUDA embedding-gather kernel and prefill path that copies prompt token IDs to the GPU and allocates activations.
- A CUDA RMSNorm kernel for the 2,048-wide hidden state: it reduces in FP32 and applies the learned BF16 weights throughout the forward pass.
- Dense causal prefill now runs all 16 layers: Q/K/V projection, RoPE, grouped-query attention, output projection, the attention residual, post-attention RMSNorm, SwiGLU MLP, and the MLP residual.
- The final RMSNorm and tied embedding/LM-head projection produce FP32 logits; GPU argmax selects each next token.
- A shared BF16 KV page pool with a block table per sequence and layer. Each physical page holds 16 tokens of K and V; prefill fills prompt pages and decode appends one new token at its absolute RoPE position.
- Greedy response generation with Instruct chat formatting, end-of-turn/end-of-text stopping, and configurable output length within the 2,048-token context.
- Persistent scratch buffers and one cuBLAS handle shared by prefill and decode. Q storage is reused for output/MLP projections, and normalization storage is reused for attention output.
- Batched decode with runtime-sized buffers, packed projections, and separate KV page mappings and attention lengths per sequence. Request count does not enlarge the KV pool.
- Static batching and a continuous scheduler that admit queued prompts according to available page budgets, then return their pages and budgets on completion.
- Direct paged decode attention with FP32 online softmax. Each query head streams its sequence's cached pages and accumulates weighted values without gathering a contiguous prefix or storing attention scores.

Part 8 explains [Prefill, Decode, and the KV Cache](proper_notes/markdown/youtubePT8_prefill_decode_kv_cache.md). Part 9 follows with [Static and Continuous Batching](proper_notes/markdown/youtubePT9_static_continuous_batching.md), including the current paged cache and online-softmax decode attention.

## Current scope

The Llama 3.2 1B inference path includes dense causal prefill, paged decode attention, and static or continuous batching. Model dimensions and the 2,048-token context remain fixed for this learning engine. The KV pool defaults to 128 MiB and can be set with `--kv-cache-mib N`; active request capacity follows the workload's page budgets rather than a fixed two-request limit.

Admission guarantees room for each request's maximum processed prefix: `prompt_tokens + allowed_new_tokens - 1`, rounded up to 16-token pages in each of the 16 layers. The allowed output budget is clamped to the context limit, and the final predicted token is returned without being processed again. These admission credits are separate from physical pages, which are assigned lazily. Early completion releases both the assigned pages and the remaining credits.

Requests wait in FIFO order when their budgets cannot fit alongside active requests. Active requests have room to finish, so future page-boundary growth cannot exhaust the pool or deadlock generation. A request whose budget exceeds the entire pool is rejected with a clear error. This conservative policy avoids preemption and recomputation; a large requested output budget can limit concurrency even if the model eventually stops early.

`--stats` reports the pool size, peak active requests, largest decode batch, peak assigned pages, admission waits, and final free-page count. Prefill still processes one prompt at a time between decode iterations.

BF16 computations can round differently at different batch sizes. When vocabulary logits are nearly tied, greedy output tokens can therefore differ between individual and batched runs. Bitwise output equality across batch sizes is not guaranteed.

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
# Continuous batching (default): admit requests whose page budgets fit.
./build/samir-vllm "Say hello." 64 "Explain the KV cache." "What is RoPE?"
# Static batching: wait for the current group before admitting more.
./build/samir-vllm --static "Say hello." 64 "Explain the KV cache." "What is RoPE?"
# More than two requests share the same default 128 MiB pool.
./build/samir-vllm --stats "Explain paging." 64 "Explain GQA." "Explain RoPE." "Explain softmax."
# A smaller pool queues requests when their declared budgets cannot fit together.
./build/samir-vllm --kv-cache-mib 8 --stats "Explain paging." 64 "Explain GQA." "Explain RoPE."
./build/tokenizer-test models/llama-3.2-1b-instruct/tokenizer.model
```

After optional `--static`, `--continuous`, `--kv-cache-mib N`, and `--stats` flags, the first argument is a prompt and the second is `max_new_tokens` (default: 128). Any remaining arguments are additional prompts; the token budget applies to each request. Every prompt is formatted as an Instruct user message. Generation stops independently at EOS, EOT, or EOM, or the output/context limit, and each completed response is printed with its token IDs and request number.

Prompts are prefilled one at a time, pausing decode. Each decode step packs one token from every active sequence into the shared buffers. Attention reads only that sequence's cache; requests can have different prompt and output lengths. Continuous batching admits waiting requests when page credits become available, while static batching admits the next group after the current group finishes.

## Project layout

```text
src/           Engine entry point, CUDA kernels, and tokenizer implementation
include/       Public kernel and tokenizer headers
tests/         Tokenizer tests
python/        Tokenizer reference script
proper_notes/  Study-notes index, markdown/ sources, and pdf/ reading editions
scripts/       Rebuild the study-note PDFs
notes.md       Working notes from the build
```

## License

The engine source code in `src/`, `include/`, `tests/`, and `python/`, along with `CMakeLists.txt`, `.gitignore`, and this README, is licensed under the [MIT License](LICENSE).

The notes in `proper_notes/`, `notes.md`, and `podLibraryInstall.md`, along with handwritten notes, scans, illustrations, photographs of notes, and PDFs provided with this project, are **all rights reserved** and excluded from MIT. Personal reading and study are permitted. Redistribution, republication, rehosting, resale, or distribution of adapted versions requires prior written permission, whether offered for free or for payment. See [LICENSE-NOTES](LICENSE-NOTES) for the full terms and exceptions.

Public repositories can still be viewed and forked within GitHub under [GitHub's Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#5-license-grant-to-other-users).

### Model assets and dependencies

**Built with Llama.** The bundled Llama 3.2 tokenizer and any separately downloaded Llama 3.2 weights retain Meta's [Llama 3.2 Community License](models/LICENSE-LLAMA-3.2) and [Acceptable Use Policy](models/USE_POLICY.md). They are excluded from MIT. Keep the agreement, policy, and [attribution notice](models/NOTICE) with any redistributed Llama materials.

Third-party dependencies, including CUDA, cuBLAS, ICU, and `nlohmann/json`, retain their respective licenses.
