#include "kernels.cuh"

#include <cmath>
#include <vector>

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
constexpr int ROPE_HEAD_SIZE = 64;
constexpr int ROPE_PAIRS_PER_HEAD = ROPE_HEAD_SIZE / 2;
constexpr float ROPE_THETA = 500000.0f;
constexpr float ROPE_FACTOR = 32.0f;
constexpr float ROPE_LOW_FREQUENCY_FACTOR = 1.0f;
constexpr float ROPE_HIGH_FREQUENCY_FACTOR = 4.0f;
constexpr int ROPE_ORIGINAL_MAX_LENGTH = 8192;
constexpr float PI = 3.14159265358979323846f;

// dog bites man
// man bites dog
// the cat sat on the mat
// (xo, x1)
cudaError_t initializeRopeTables(
    float **cos_table_gpu,
    float **sin_table_gpu,
    int max_sequence_length)
{
    if (cos_table_gpu == nullptr ||
        sin_table_gpu == nullptr ||
        max_sequence_length <= 0)
    {
        return cudaErrorInvalidValue;
    }

    *cos_table_gpu = nullptr;
    *sin_table_gpu = nullptr;

    std::vector<float> inverse_frequencies(ROPE_PAIRS_PER_HEAD);
    // early pairs: fast freq
    // later pairs = slow freq
    for (int pair = 0; pair < ROPE_PAIRS_PER_HEAD; ++pair)
    {
        inverse_frequencies[pair] =
            1.0f /
            std::pow(
                ROPE_THETA,
                (2.0f * static_cast<float>(pair)) /
                    static_cast<float>(ROPE_HEAD_SIZE));
    }

    const float low_frequency_wavelength =
        static_cast<float>(ROPE_ORIGINAL_MAX_LENGTH) /
        ROPE_LOW_FREQUENCY_FACTOR;

    const float high_frequency_wavelength =
        static_cast<float>(ROPE_ORIGINAL_MAX_LENGTH) /
        ROPE_HIGH_FREQUENCY_FACTOR;

    for (int pair = 0; pair < ROPE_PAIRS_PER_HEAD; ++pair)
    {
        const float original_frequency = inverse_frequencies[pair];
        const float wavelength = 2.0f * PI / original_frequency;

        if (wavelength > low_frequency_wavelength)
        {
            inverse_frequencies[pair] =
                original_frequency / ROPE_FACTOR;
        }
        else if (wavelength >= high_frequency_wavelength)
        {
            const float smooth =
                (static_cast<float>(ROPE_ORIGINAL_MAX_LENGTH) /
                     wavelength -
                 ROPE_LOW_FREQUENCY_FACTOR) /
                (ROPE_HIGH_FREQUENCY_FACTOR -
                 ROPE_LOW_FREQUENCY_FACTOR);

            inverse_frequencies[pair] =
                (1.0f - smooth) *
                    (original_frequency / ROPE_FACTOR) +
                smooth * original_frequency;
        }
    }

    const size_t table_elements =
        static_cast<size_t>(max_sequence_length) *
        ROPE_PAIRS_PER_HEAD;

    std::vector<float> cos_table_cpu(table_elements);
    std::vector<float> sin_table_cpu(table_elements);

    for (int position = 0;
         position < max_sequence_length;
         ++position)
    {
        for (int pair = 0; pair < ROPE_PAIRS_PER_HEAD; ++pair)
        {
            const float angle =
                static_cast<float>(position) *
                inverse_frequencies[pair];

            const size_t index =
                static_cast<size_t>(position) *
                    ROPE_PAIRS_PER_HEAD +
                pair;

            cos_table_cpu[index] = std::cos(angle);
            sin_table_cpu[index] = std::sin(angle);
        }
    }

    const size_t table_bytes = table_elements * sizeof(float);

    cudaError_t error = cudaMalloc(cos_table_gpu, table_bytes);
    if (error != cudaSuccess)
    {
        return error;
    }

    error = cudaMalloc(sin_table_gpu, table_bytes);
    if (error != cudaSuccess)
    {
        cudaFree(*cos_table_gpu);
        *cos_table_gpu = nullptr;
        return error;
    }

    error = cudaMemcpy(
        *cos_table_gpu,
        cos_table_cpu.data(),
        table_bytes,
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaFree(*cos_table_gpu);
        cudaFree(*sin_table_gpu);
        *cos_table_gpu = nullptr;
        *sin_table_gpu = nullptr;
        return error;
    }

    error = cudaMemcpy(
        *sin_table_gpu,
        sin_table_cpu.data(),
        table_bytes,
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaFree(*cos_table_gpu);
        cudaFree(*sin_table_gpu);
        *cos_table_gpu = nullptr;
        *sin_table_gpu = nullptr;
        return error;
    }

    return cudaSuccess;
}

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
    // 512 -> 256 -> 128 -> 64 -> ... -> 2 -> 1
    // partial [0] += partial[512]
    // partial [511] += partial[1023]
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

// scores is [query_head, query_token, key_token]. A score above the diagonal
// would let a token see the future, so it must never participate in softmax.
__global__ void causalMaskKernel(
    __nv_bfloat16 *scores,
    int token_count,
    int num_query_heads)
{
    const size_t score_index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t score_count =
        static_cast<size_t>(num_query_heads) *
        token_count *
        token_count;

    if (score_index >= score_count)
    {
        return;
    }

    const int key_token = score_index % token_count;
    const int query_token =
        (score_index / token_count) % token_count;

    if (key_token > query_token)
    {
        scores[score_index] = __float2bfloat16(-CUDART_INF_F);
    }
}

// Each block normalizes one [query_head, query_token] row. Threads stride over
// key positions, which keeps this valid for contexts longer than one CUDA block.
__global__ void stableSoftmaxKernel(
    __nv_bfloat16 *scores,
    int token_count)
{
    constexpr int THREADS_PER_BLOCK = 256;
    __shared__ float reductions[THREADS_PER_BLOCK];

    const int thread_index = threadIdx.x;
    const size_t row_start =
        static_cast<size_t>(blockIdx.x) * token_count;

    float local_max = -CUDART_INF_F;
    for (int key_token = thread_index;
         key_token < token_count;
         key_token += THREADS_PER_BLOCK)
    {
        local_max = fmaxf(
            local_max,
            __bfloat162float(scores[row_start + key_token]));
    }

    reductions[thread_index] = local_max;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2;
         stride > 0;
         stride >>= 1)
    {
        if (thread_index < stride)
        {
            reductions[thread_index] = fmaxf(
                reductions[thread_index],
                reductions[thread_index + stride]);
        }
        __syncthreads();
    }

    const float row_max = reductions[0];
    float local_sum = 0.0f;
    for (int key_token = thread_index;
         key_token < token_count;
         key_token += THREADS_PER_BLOCK)
    {
        local_sum += expf(
            __bfloat162float(scores[row_start + key_token]) - row_max);
    }

    reductions[thread_index] = local_sum;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2;
         stride > 0;
         stride >>= 1)
    {
        if (thread_index < stride)
        {
            reductions[thread_index] += reductions[thread_index + stride];
        }
        __syncthreads();
    }

    const float inverse_sum = 1.0f / reductions[0];
    for (int key_token = thread_index;
         key_token < token_count;
         key_token += THREADS_PER_BLOCK)
    {
        const float probability = expf(
            __bfloat162float(scores[row_start + key_token]) - row_max) *
            inverse_sum;
        scores[row_start + key_token] = __float2bfloat16(probability);
    }
}

__global__ void residualAddKernel(
    __nv_bfloat16 *hidden_state,
    const __nv_bfloat16 *update,
    size_t value_count)
{
    const size_t value_index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (value_index >= value_count)
    {
        return;
    }

    hidden_state[value_index] = __float2bfloat16(
        __bfloat162float(hidden_state[value_index]) +
        __bfloat162float(update[value_index]));
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
    // one cuda block = one token
    // 1024 threads = one block
    // one thread = two hidden values

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

cudaError_t launchCausalMask(
    __nv_bfloat16 *scores,
    int token_count,
    int num_query_heads)
{
    if (token_count == 0)
    {
        return cudaSuccess;
    }

    if (scores == nullptr || token_count < 0 || num_query_heads <= 0)
    {
        return cudaErrorInvalidValue;
    }

    constexpr int THREADS_PER_BLOCK = 256;
    const size_t score_count =
        static_cast<size_t>(num_query_heads) *
        token_count *
        token_count;
    const size_t block_count =
        (score_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    causalMaskKernel<<<block_count, THREADS_PER_BLOCK>>>(
        scores,
        token_count,
        num_query_heads);

    return cudaGetLastError();
}

cudaError_t launchStableSoftmax(
    __nv_bfloat16 *scores,
    int token_count,
    int num_query_heads)
{
    if (token_count == 0)
    {
        return cudaSuccess;
    }

    if (scores == nullptr || token_count < 0 || num_query_heads <= 0)
    {
        return cudaErrorInvalidValue;
    }

    constexpr int THREADS_PER_BLOCK = 256;
    const int row_count = num_query_heads * token_count;
    stableSoftmaxKernel<<<row_count, THREADS_PER_BLOCK>>>(
        scores,
        token_count);

    return cudaGetLastError();
}

cudaError_t launchResidualAdd(
    __nv_bfloat16 *hidden_state,
    const __nv_bfloat16 *update,
    int token_count,
    int hidden_size)
{
    if (token_count == 0)
    {
        return cudaSuccess;
    }

    if (hidden_state == nullptr || update == nullptr ||
        token_count < 0 || hidden_size <= 0)
    {
        return cudaErrorInvalidValue;
    }

    constexpr int THREADS_PER_BLOCK = 256;
    const size_t value_count =
        static_cast<size_t>(token_count) * hidden_size;
    const size_t block_count =
        (value_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    residualAddKernel<<<block_count, THREADS_PER_BLOCK>>>(
        hidden_state,
        update,
        value_count);

    return cudaGetLastError();
}
