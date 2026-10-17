# Part 8: Prefill, Decode, and the KV Cache

We already run all 16 transformer layers and predict one token. This part turns that into a response by keeping useful work from the prompt and feeding each predicted token back into the model.

Read the code in this order:

1. `KVCache` beside `LLamaWeights` in [`src/main.cpp`](../src/main.cpp).
2. `initializeKvCache`, the existing `prefill`, and the new `decode` in that file.
3. `computeGqaAttentionScores` and `mixGqaValues` near the top of that file.
4. `stableSoftmaxKernel` in [`src/kernels.cu`](../src/kernels.cu).
5. The generation loop in `main`.

## Start with the two paths

**Prefill** processes every prompt token in one forward pass. Each token goes through embeddings, all 16 layers, and causal attention. The last prompt position predicts the first response token.

**Decode** processes just the previously predicted token. That token goes through the same 16 layers, but its query attends to keys and values for the entire processed sequence. Its output predicts the following token.

The math of a transformer layer is the same in both paths. The difference is how many new token rows we compute, how many existing K/V rows attention reads, and which positions RoPE uses.

| Operation | Prefill: prompt length T | Decode: L tokens already cached |
| --- | --- | --- |
| New token rows | T | 1 |
| RoPE positions | 0 through T - 1 | L |
| Q shape | `[T, 2048]` | `[1, 2048]` |
| New K and V shapes, each | `[T, 512]` | `[1, 512]` |
| K and V read by attention, each | `[T, 512]` | `[L + 1, 512]` |
| Attention scores | `[32, T, T]` | `[32, 1, L + 1]` |
| Causal mask | Hide future prompt positions | Every stored position is allowed |
| Cache length after the pass | T | L + 1 |

Decode includes the current token's own K/V. Reading only the previous L positions would drop the attention diagonal and change the model.

## Why old keys and values can be reused

This is a causal model: an earlier token cannot see a later token. Appending text therefore cannot change the earlier token's hidden state or its projected K/V in any layer. We save those projections instead of recomputing the whole prefix for every next-token prediction.

We cache **K and V for every layer**, because every layer uses different weights and different incoming hidden states. We keep K after RoPE and V without RoPE. Old keys must never be rotated again.

We do not cache old Q vectors: a new decode step only needs the current token's query. We also do not need to preserve old attention scores, probabilities, or MLP activations.

Caching avoids repeating the old tokens' projections and layer work. Attention still reads the accumulated K/V, so decode cost grows with context length. Prefill often benefits from large matrix multiplications; decode has small projections and must repeatedly read the model weights and cache. They have different performance characteristics even though they use the same model.

## Our cache is deliberately simple

Two GPU allocations hold the keys and values for a single sequence:

```text
keys   [16 layers, capacity positions, 512 KV features]
values [16 layers, capacity positions, 512 KV features]
```

The capacity is fixed for that request. `main` reserves the formatted prompt length plus the response budget, up to our 2,048-token context. At maximum capacity, K and V together use:

```text
2 × 16 × 2048 × 512 × 2 BF16 bytes = 64 MiB
```

`capacity` determines the distance between layers. `length` says how many positions contain valid K/V. Allocated slots beyond `length` do not need to be zeroed because attention never reads them.

For one layer, the pointers are:

```cpp
layer_offset = layer * cache.capacity * KV_DIM;
layer_keys = cache.keys + layer_offset;
new_key = layer_keys + position * KV_DIM;
```

The existing prefill keeps its temporary `k_gpu` and `v_gpu` buffers. After RoPE, it copies the prompt K/V into the matching layer of the cache and continues its original dense attention calculation. Decode writes the new token's K/V directly into the next cache slot, rotates only the new key, and reads from `layer_keys` and `layer_values` for `key_count` positions.

`cache.length` advances only after all layers and next-token selection succeed. `prefill` resets it for a new prompt. `decode` requires a nonempty cache and refuses an append when capacity is exhausted.

## Keep both paths visible

The existing `prefill` stays in place, with its original buffers and explicit 16-layer loop. It processes the whole prompt at positions starting from zero, saves each layer's K/V, and predicts the first response token.

The new `decode` follows it in the same file. It has its own one-token buffers and layer loop, starts at `cache.length`, and reuses the existing projection and kernel helpers. The two functions make the different attention shapes visible. Scratch buffers are reused across layers, then freed at the end of each call. Keeping scratch and the cuBLAS handle alive between decode steps is a later optimization; the persistent state introduced here is the KV cache.

The attention helpers keep `token_count` for the number of query rows and accept an optional `key_count`. Existing prefill calls stay unchanged; decode supplies the cached length explicitly. For each GQA head:

```text
scores        [query_count, key_count] = Q × cached_Kᵀ / sqrt(64)
probabilities [query_count, key_count] = softmax(scores)
head_output   [query_count, 64]        = probabilities × cached_V
```

Four query heads still share one K/V head. The softmax row width is `key_count`, even when there is only one query. The cuBLAS score matrix leading dimension is also `key_count`. Using query count for either would break decode.

Prefill masks its square score matrices before softmax. Decode has no future K/V entries to mask, because it reads only the valid prefix plus the new token. Softmax itself needs no separate decode kernel: it normalizes any row width with the existing strided reduction.

## Trace the generation loop

Suppose the formatted prompt has three tokens, and the response will be A, B, C:

```text
prefill(prompt positions 0, 1, 2) -> predicts A; cache length = 3
emit A
decode(A at position 3)           -> predicts B; cache length = 4
emit B
decode(B at position 4)           -> predicts C; cache length = 5
emit C
```

Predicting A does not put A in the cache. Processing A on the next decode call does. This distinction prevents an off-by-one bug in both cache writes and RoPE positions.

The loop stops at `<|eot_id|>` (end of assistant turn), `<|end_of_text|>`, or the token budget. It does not run a redundant decode after the last permitted prediction. The tokenizer decodes the complete response token sequence together, allowing tokens that split a UTF-8 character to join correctly.

The executable now formats the input as an Instruct user message with an assistant header. It uses greedy argmax, just as before; there is no temperature or sampling in this part. A response can still be truncated by the configured limits.

```bash
./build/samir-vllm "Explain the KV cache in two sentences." 128
```

## Check correctness on a CUDA GPU

```bash
# Synthetic attention: dense causal rows and one-query decode rows,
# including 257 keys to exercise softmax beyond one 256-thread stride.
./build/kv-cache-test

# Actual model: repeated decode vs. a fresh dense pass on the same prefix,
# cache reset, absolute positions, final-slot append, and input bounds.
./build/kv-cache-test --model

# Independent Hugging Face greedy-generation reference.
# Requires torch, transformers, and access to the matching model.
python3 python/reference_generation.py "Say hello in one sentence." --max-new-tokens 32
```

The model cache test checks every logit with an explicit BF16 tolerance and requires matching greedy token IDs. The tolerance is a starting check to validate on the target GPU, not an established accuracy guarantee. Different GEMM shapes can round differently, and close logits can produce different greedy choices. The Python comparison reports differences rather than silently accepting them. Agreement between cached and dense paths checks caching; the independent reference also helps catch errors shared by both paths.

## Reference and later parts

The baseline is [tiny-vllm's course](https://github.com/jmaczan/tiny-vllm) and its [current forward/decode source](https://github.com/jmaczan/tiny-vllm/blob/main/src/main.cpp), reviewed for this change. Its current engine includes paging and continuous batching. Here we use contiguous per-layer storage for one sequence so the cache and token loop are easy to trace. Paging, batching, persistent scratch buffers, and fused attention can build on this later.
