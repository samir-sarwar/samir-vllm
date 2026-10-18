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
    const __nv_bfloat16 *input,
    __nv_bfloat16 *output,
    const __nv_bfloat16 *norm_weights,
    int token_count);

cudaError_t initializeRopeTables(
    float **cos_table_gpu,
    float **sin_table_gpu,
    int max_sequence_length);

cudaError_t launchRope(
    __nv_bfloat16 *input,
    const int *position_ids,
    const float *cos_table,
    const float *sin_table,
    int token_count,
    int projection_size,
    int head_size);

// Sets all future-token attention scores to negative infinity in-place.
// scores is laid out as [query_head, query_token, key_token].
cudaError_t launchCausalMask(
    __nv_bfloat16 *scores,
    int token_count,
    int num_query_heads);

// Applies stable softmax over key positions in-place.
// scores is [query_head, query_token, key_token]. Prefill uses T x T rows;
// decode uses one query row over all cached keys for each head.
cudaError_t launchStableSoftmax(
    __nv_bfloat16 *scores,
    int token_count,
    int num_query_heads,
    int key_count = 0);

// Adds update to hidden_state in-place. Both buffers are [token, hidden].
cudaError_t launchResidualAdd(
    __nv_bfloat16 *hidden_state,
    const __nv_bfloat16 *update,
    int token_count,
    int hidden_size);

// gate = SiLU(gate) * up, in-place, over [token, intermediate] values.
cudaError_t launchSwiGlu(
    __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    int token_count,
    int intermediate_size);

// One prediction per row of [token, vocabulary] FP32 logits.
// Ties choose the smallest vocabulary index.
cudaError_t launchArgmax(
    const float *logits,
    int *token_id,
    int vocabulary_size,
    int token_count = 1);
