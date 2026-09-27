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

// Applies a numerically stable softmax over the key-token dimension in-place.
// scores is laid out as [query_head, query_token, key_token].
cudaError_t launchStableSoftmax(
    __nv_bfloat16 *scores,
    int token_count,
    int num_query_heads);

// Adds update to hidden_state in-place. Both buffers are [token, hidden].
cudaError_t launchResidualAdd(
    __nv_bfloat16 *hidden_state,
    const __nv_bfloat16 *update,
    int token_count,
    int hidden_size);
