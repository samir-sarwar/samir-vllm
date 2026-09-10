#include "kernels.cuh"

namespace
{

    __global__ void warmUpKernel()
    {
    }

} // namespace

void warmUpGpu()
{
    warmUpKernel<<<1, 1>>>();
}

/* -------------- Kernels -------------- */

__global__ void embeddingGatherKernel(
    const __nv_bfloat16 *embed_tokens,
    __nv_bfloat16 *activations_gpu,
    const int *token_id_gpu,
    int token_count)
{
    // get thread unique global num, which block x threads per block + thread position
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_values = token_count * 2048;

    if (index >= total_values)
    {
        return;
    }
    // finds out which prompt token this output belongs to
    int token_position = index / 2048;

    const int hidden_index = index % 2048;

    // reads vocab id for that propmt position
    const int token_id = token_id_gpu[token_position];

    // copies the one bf16 value at that position in the embedding table
    activations_gpu[index] = embed_tokens[token_id * 2048 + hidden_index];
}

constexpr int HIDDEN_SIZE = 2048;
constexpr float RMS_EPS = 1.0e-5f;

__global__ void rmsNormKernel(
    const __nv_bfloat16 *input,
    __nv_bfloat16 *output,
    const __nv_bfloat16 *norm_weights)
{
    __shared__ float partial[1024];

    const int tid = threadIdx.x;
    const int base = blockIdx.x * HIDDEN_SIZE;

    const float x0 = __bfloat162float(input[base + tid]);
    const float x1 = __bfloat162float(input[base + tid + 1024]);

    partial[tid] = x0 * x0 + x1 * x1;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (tid < i)
        {
            partial[tid] += partial[tid + i];
        }

        __syncthreads();
    }
    if (tid == 0)
    {
        partial[0] =
            rsqrtf(partial[0] / static_cast<float>(HIDDEN_SIZE) + RMS_EPS);
    }

    __syncthreads();

    const float inverse_rms = partial[0];
    const float weight0 = __bfloat162float(norm_weights[tid]);
    const float weight1 = __bfloat162float(norm_weights[tid + 1024]);

    output[base + tid] =
        __float2bfloat16(x0 * inverse_rms * weight0);

    output[base + tid + 1024] =
        __float2bfloat16(x1 * inverse_rms * weight1);
}

__global__ void ropeKernel(
    __nv_bfloat16 *input,
    const int *position_ids,
    const float *cos_table,
    const float *sin_table,
    int token_count,
    int projection_size,
    int head_size)
{
    const int work_index =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int pairs_per_head = head_size / 2;
    const int pairs_per_token = projection_size / 2;
    const int total_pairs = token_count * pairs_per_token;

    if (work_index >= total_pairs)
    {
        return;
    }

    const int token_index = work_index / pairs_per_token;
    const int pair_in_token = work_index % pairs_per_token;

    const int head_index = pair_in_token / pairs_per_head;
    const int pair_index = pair_in_token % pairs_per_head;

    const int head_base =
        token_index * projection_size +
        head_index * head_size;

    const int first_index = head_base + pair_index;
    const int second_index =
        first_index + pairs_per_head;

    const float x0 =
        __bfloat162float(input[first_index]);

    const float x1 =
        __bfloat162float(input[second_index]);

    const int position = position_ids[token_index];
    const int table_index =
        position * pairs_per_head + pair_index;

    const float cosine = cos_table[table_index];
    const float sine = sin_table[table_index];

    input[first_index] =
        __float2bfloat16(x0 * cosine - x1 * sine);

    input[second_index] =
        __float2bfloat16(x0 * sine + x1 * cosine);
}

/* -------------- Kernel Launchers -------------- */
cudaError_t launchEmbeddingGather(
    const int *token_id_gpu,
    const __nv_bfloat16 *embed_tokens,
    __nv_bfloat16 *activations_gpu,
    int token_count)
{
    const int threads_per_block = 256;           // each CUDA block has 256 gpu threads
    const int total_values = token_count * 2048; // total number of threads

    // How many blocks, rounding upwards
    const int blocks = (total_values + threads_per_block - 1) / threads_per_block;

    embeddingGatherKernel<<<blocks, threads_per_block>>>(
        embed_tokens,
        activations_gpu,
        token_id_gpu,
        token_count);

    return cudaGetLastError();
}

cudaError_t launchRmsNorm(
    const __nv_bfloat16 *input,
    __nv_bfloat16 *output,
    const __nv_bfloat16 *norm_weights,
    int token_count)
{
    if (token_count <= 0)
    {
        return cudaSuccess;
    }

    rmsNormKernel<<<token_count, 1024>>>(
        input,
        output,
        norm_weights);

    return cudaGetLastError();
}

cudaError_t launchRope(
    __nv_bfloat16 *input,
    const int *position_ids,
    const float *cos_table,
    const float *sin_table,
    int token_count,
    int projection_size,
    int head_size)
{
    if (token_count == 0)
    {
        return cudaSuccess;
    }

    if (token_count < 0 ||
        input == nullptr ||
        position_ids == nullptr ||
        cos_table == nullptr ||
        sin_table == nullptr ||
        projection_size <= 0 ||
        head_size <= 0 ||
        head_size % 2 != 0 ||
        projection_size % head_size != 0)
    {
        return cudaErrorInvalidValue;
    }

    constexpr int THREADS_PER_BLOCK = 256;

    const int pairs_per_token = projection_size / 2;
    const int total_pairs = token_count * pairs_per_token;

    const int block_count =
        (total_pairs + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK;

    ropeKernel<<<block_count, THREADS_PER_BLOCK>>>(
        input,
        position_ids,
        cos_table,
        sin_table,
        token_count,
        projection_size,
        head_size);

    return cudaGetLastError();
}