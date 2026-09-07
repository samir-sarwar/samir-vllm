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

__global__ void embeddingGatherKernel(__nv_bfloat16 *embed_tokens,
      const __nv_bfloat16 *activations_gpu,
       const int *token_id_gpu,
    int token_count)
{
    // get thread unique global num, which block x threads per block + thread position
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_values = token_count * 2048;

    if(index >= total_values){
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

cudaError_t launchEmbeddingGather(
    const int *token_id_gpu,
    const __nv_bfloat16 *embed_tokens,
    __nv_bfloat16 *activations_gpu,
    int token_count){
        const int threads_per_block = 256; // each CUDA block has 256 gpu threads
        const int total_values = token_count * 2048; // total number of threads 

        // How many blocks, rounding upwards 
        const int blocks = (total_values + threads_per_block -1) / threads_per_block;

        embeddingGatherKernel<<<blocks, threads_per_block>>>(
            embed_tokens,
            activations_gpu,
            token_id_gpu,
            token_count
        );

        return cudaGetLastError();

    }


