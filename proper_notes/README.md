# Companion study notes

These notes explain the concepts and code behind the YouTube build series. The [working notes](../notes.md) are the starting point; these chapters organize the explanations, follow tensor shapes through the implementation, and call out mistakes to watch for.

- [Markdown chapters](markdown/) are the editable sources.
- [PDF chapters](pdf/) are the reading and printing editions.
- [Complete study notes (PDF)](pdf/samir_vllm_complete_study_notes.pdf) collects Parts 0-9 and the CUDA primer, with contents and chapter bookmarks.

| Part | Topic | Markdown | PDF |
| --- | --- | --- | --- |
| 0 | Inference and the engine | [Read](markdown/youtubePT0_inference_notes.md) | [Read](pdf/youtubePT0_inference_notes.pdf) |
| 1 | SafeTensors and loading weights | [Read](markdown/youtubePT1_notes.md) | [Read](pdf/youtubePT1_notes.pdf) |
| Primer | CUDA threads, blocks, memory, and launches | [Read](markdown/cuda_kernels_getting_started.md) | [Read](pdf/cuda_kernels_getting_started.pdf) |
| 2 | Embedding gather | [Read](markdown/youtubePT2_embedding_gather_notes.md) | [Read](pdf/youtubePT2_embedding_gather_notes.pdf) |
| 3 | RMSNorm | [Read](markdown/youtubePT3_rmsnorm_notes.md) | [Read](pdf/youtubePT3_rmsnorm_notes.pdf) |
| 4 | Rotary position embeddings | [Read](markdown/youtubePT4_rope_notes.md) | [Read](pdf/youtubePT4_rope_notes.pdf) |
| 5 | Q/K/V projections and cuBLAS | [Read](markdown/youtubePT5_qkv_projection_notes.md) | [Read](pdf/youtubePT5_qkv_projection_notes.pdf) |
| 6 | Causal grouped-query self-attention | [Read](markdown/youtubePT6_self_attention_completion.md) | [Read](pdf/youtubePT6_self_attention_completion.pdf) |
| 7 | Feed-forward network, SwiGLU, and the first token | [Read](markdown/youtubePT7_feed_forward_swiglu_mlp.md) | [Read](pdf/youtubePT7_feed_forward_swiglu_mlp.pdf) |
| 8 | Prefill, decode, and the KV cache | [Read](markdown/youtubePT8_prefill_decode_kv_cache.md) | [Read](pdf/youtubePT8_prefill_decode_kv_cache.pdf) |
| 9 | Static/continuous batching, paging, and online softmax | [Read](markdown/youtubePT9_static_continuous_batching.md) | [Read](pdf/youtubePT9_static_continuous_batching.pdf) |

Read Parts 0 and 1 first, then the CUDA primer before writing kernels. The collected PDF places the primer after Part 9 as an appendix.

## Reading the code alongside the notes

The video milestones are snapshots of a growing engine. Early chapters explain the stage reached in that video; they do not imply that the current checkout still stops at that stage. Parts 7-9 identify their historical commits so you can inspect them with `git show`. Parts 8 and 9 also explain the current cache and scheduler, including the changes after the original batching implementation.

The implementation targets a fixed Llama 3.2 1B Instruct weight layout: 16 layers, hidden width 2,048, intermediate width 8,192, 32 query heads, 8 KV heads, and a 2,048-token context. It is a learning engine. Numerical comparisons and performance measurements require the matching checkpoint, tokenizer, and a suitable CUDA GPU; the notes describe validation steps rather than claiming those checks have been run.

## Rebuild the PDFs

From the repository root, use Python 3.10 or newer:

```bash
python3 -m venv .venv-notes
source .venv-notes/bin/activate
python -m pip install -r scripts/requirements-notes.txt
python scripts/build_notes.py
```

The renderer needs the DejaVu Sans and DejaVu Sans Mono TTF families (including Sans bold and italic). It finds the standard Linux font directory and the Codex bundled fonts automatically. On another system, install DejaVu fonts and set `NOTES_FONT_DIR` to their directory:

```bash
NOTES_FONT_DIR=/path/to/dejavu-fonts python scripts/build_notes.py
```

To regenerate one chapter without rebuilding the collected edition:

```bash
python scripts/build_notes.py --only youtubePT8_prefill_decode_kv_cache
```

Run the full build again before distributing an updated collection. Markdown links between chapters become links to sibling PDFs; repository source links remain relative to the checkout. Keep the folder structure together for those links to work. PDF viewers differ in how they handle local-file links. The `README.md` and `NOTICE.md` at this directory's root are navigation and copyright metadata, not separate study chapters.

Copyright (c) 2026 Samir Sarwar. All rights reserved. Personal reading and study are permitted. See [LICENSE-NOTES](../LICENSE-NOTES) and [NOTICE.md](NOTICE.md) for the full terms. Code excerpts retain the source code's license.
