#pragma once

// GPU entry points called by the host-side program.
// Add inference-kernel declarations here as we implement them.
void warmUpGpu();
cudaError_t launchEmbeddingGather(
    const int *token_id_gpu,
    const __nv_bfloat16 *embed_tokens,
    __nv_bfloat16 *activations_gpu,
    int token_count);


