#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
// GPU entry points called by the host-side program.
// Add inference-kernel declarations here as we implement them.
void warmUpGpu();
cudaError_t launchEmbeddingGather(
    const int *token_id_gpu,
    const __nv_bfloat16 *embed_tokens,
    __nv_bfloat16 *activations_gpu,
    int token_count);

cudaError_t launchRmsNorm(
    activations_gpu,
    normalized_gpu,
    weights.input_layernorm[0],
    token_count);
