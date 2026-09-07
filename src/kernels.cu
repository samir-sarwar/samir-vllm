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
      __nv_bfloat16 *activations_gpu,
       int *token_id_gpu,
    int token_count)
{
    // get thread unique global num, which block x threads per block + thread position
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_values = token_count * 2048;

    if(index > total_values){
        return;
    }
    // finds out which prompt token this output belongs to 
    int token_position = index / 2048

    const int hidden_index = index % 2048;

    // reads vocab id for that propmt position
    const int token_id = token_ids_gpu[token_position];

    // copies the one bf16 value at that position in the embedding table
    activations_gpu[index] = embed_tokens[token_id * 2048 + hidden_index];
}


