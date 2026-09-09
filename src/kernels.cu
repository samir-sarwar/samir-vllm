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

__global__ void embeddingGatherKernel(const __nv_bfloat16 *embed_tokens,
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
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            partial[tid] += partial[tid + stride];
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
