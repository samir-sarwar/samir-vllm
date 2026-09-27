// CUDA runtime: allows us to do all our CUDA work.
#include <cuda_runtime.h>
// Allows us to do big matrix multiplications.
#include <cublas_v2.h>
#include <cstdlib>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
// Grabs json header for functions to parse and create json objects in cpp.
#include <nlohmann/json.hpp>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <vector>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include "tokenizer.hpp"

// Local alias.
using json = nlohmann::json;

#include <iostream>

// We can run the functions from kernels.cuh.
// They were implemented in kernels.cu.
#include "kernels.cuh"

int checkGPUStatus()
{
    int device_count = 0;
    // CUDA func to get device count.
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        // cout but for errors.
        std::cerr << "no cuda devices found \n";
        return 1;
    }
    // Creates struct to hold info on gpu.
    cudaDeviceProp prop;
    // Fill struct.
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << (free_mem / (1024 * 1024 * 1024)) << "GB, total memory: " << total_mem / (1024 * 1024 * 1024) << "GB\n";
    return 0;
}
// 16 transformer layers.
constexpr int N_LAYERS = 16;
constexpr int HIDDEN_SIZE = 2048;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr int NUM_QUERY_HEADS = HIDDEN_SIZE / HEAD_DIM;
constexpr int NUM_KV_HEADS = KV_DIM / HEAD_DIM;
constexpr int QUERY_HEADS_PER_KV_HEAD = NUM_QUERY_HEADS / NUM_KV_HEADS;

static_assert(HIDDEN_SIZE % HEAD_DIM == 0);
static_assert(KV_DIM % HEAD_DIM == 0);
static_assert(NUM_QUERY_HEADS % NUM_KV_HEADS == 0);

// Computes:
// output[token, output_feature] =
//     sum(input[token, hidden] * weight[output_feature, hidden])
//
// All buffers are physically row-major. cuBLAS treats them as column-major,
// so the dimensions and transpose flags below produce the desired row-major
// result without explicitly transposing any GPU buffers.

cublasStatus_t projectBf16RowMajor(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    __nv_bfloat16 *output,
    int token_count,
    int output_features)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // C = alpha x A X B + beta x C

    return cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        output_features,
        token_count,
        HIDDEN_SIZE,
        &alpha,
        weight,
        CUDA_R_16BF,
        HIDDEN_SIZE,
        input,
        CUDA_R_16BF,
        HIDDEN_SIZE,
        &beta,
        output,
        CUDA_R_16BF,
        output_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
}
// hidden state
// RMSNORM
// Q,K,V Projection
// RoPE on Q and K
// Scaled QK Attention scores
// causa mask
// stable softmax
// attention weights multiplied by W_v
// output projection
// residual

// score[i,j] = Q[i] . K[j]

// Builds attention scores in [query_head, query_token, key_token] layout.
// Four consecutive Q heads share one K head in this Llama GQA configuration.
cublasStatus_t computeGqaAttentionScores(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    __nv_bfloat16 *scores,
    int token_count)
{
    if (cublas_handle == nullptr || q == nullptr || k == nullptr ||
        scores == nullptr || token_count <= 0)
    {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const float alpha = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    const float beta = 0.0f;
    const size_t scores_per_head =
        static_cast<size_t>(token_count) * token_count;

    for (int query_head_index = 0;
         query_head_index < NUM_QUERY_HEADS;
         ++query_head_index)
    {
        const int kv_head_index =
            query_head_index / QUERY_HEADS_PER_KV_HEAD;

        // q_head and k_head retain their full row strides. cuBLAS's
        // column-major interpretation therefore produces K x Q^T, which is
        // the row-major storage of Q x K^T.
        const __nv_bfloat16 *q_head =
            q + query_head_index * HEAD_DIM;
        const __nv_bfloat16 *k_head =
            k + kv_head_index * HEAD_DIM;
        __nv_bfloat16 *score_head =
            scores + query_head_index * scores_per_head;

        const cublasStatus_t status = cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            token_count,
            token_count,
            HEAD_DIM,
            &alpha,
            k_head,
            CUDA_R_16BF,
            KV_DIM,
            q_head,
            CUDA_R_16BF,
            HIDDEN_SIZE,
            &beta,
            score_head,
            CUDA_R_16BF,
            token_count,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            return status;
        }
    }

    return CUBLAS_STATUS_SUCCESS;
}

// Multiplies each Q head's probability matrix by its shared V head. The
// concatenated result is [token, 32 * 64], or [token, HIDDEN_SIZE].
cublasStatus_t mixGqaValues(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16 *probabilities,
    const __nv_bfloat16 *v,
    __nv_bfloat16 *attention_output,
    int token_count)
{
    if (cublas_handle == nullptr || probabilities == nullptr || v == nullptr ||
        attention_output == nullptr || token_count <= 0)
    {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const size_t scores_per_head =
        static_cast<size_t>(token_count) * token_count;

    for (int query_head_index = 0;
         query_head_index < NUM_QUERY_HEADS;
         ++query_head_index)
    {
        const int kv_head_index =
            query_head_index / QUERY_HEADS_PER_KV_HEAD;
        const __nv_bfloat16 *probability_head =
            probabilities + query_head_index * scores_per_head;
        const __nv_bfloat16 *value_head =
            v + kv_head_index * HEAD_DIM;
        __nv_bfloat16 *output_head =
            attention_output + query_head_index * HEAD_DIM;

        // This produces V^T x probabilities^T in cuBLAS's column-major view,
        // which is the row-major storage of probabilities x V.
        const cublasStatus_t status = cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            HEAD_DIM,
            token_count,
            token_count,
            &alpha,
            value_head,
            CUDA_R_16BF,
            KV_DIM,
            probability_head,
            CUDA_R_16BF,
            token_count,
            &beta,
            output_head,
            CUDA_R_16BF,
            HIDDEN_SIZE,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            return status;
        }
    }

    return CUBLAS_STATUS_SUCCESS;
}
struct LLamaWeights
{
    // Generic pointer to start of gpu alloc.
    void *model_storage = nullptr;

    // Ptr to embedding table, of bf16 vectors.
    __nv_bfloat16 *embed_tokens = nullptr;
    // Final rmsnorm weight vector.
    __nv_bfloat16 *norm = nullptr;

    // Array of 16 GPU pointers init to nullptr: input rmsnorm.
    __nv_bfloat16 *input_layernorm[N_LAYERS]{};
    // Rmsnorm weights after attention.
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS]{};

    // Query: what token is looking for.
    __nv_bfloat16 *w_q[N_LAYERS]{};
    // Key: what each token can be matched on.
    __nv_bfloat16 *w_k[N_LAYERS]{};
    // Value: the information each token contributes.
    __nv_bfloat16 *w_v[N_LAYERS]{};
    // Output: combines attention result.
    __nv_bfloat16 *w_o[N_LAYERS]{};

    // Small feed forward neural network inside each transformer layer.
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS]{};
    __nv_bfloat16 *mlp_up_proj[N_LAYERS]{};
    __nv_bfloat16 *mlp_down_proj[N_LAYERS]{};
};

int loadLlamaModel(LLamaWeights &weights)
{
    // Set local path.
    std::string path = "models/llama-3.2-1b-instruct/model.safetensors";
    // Open file as binary file.
    std::ifstream safetensors_file(path, std::ios::binary);
    if (!safetensors_file)
    {
        std::cerr << "could not open safetensors file";
        return -1;
    }
    uint64_t headersize = 0; // 8X8 = 64
    // Read expects a char buffer to store the extracted data.
    // We cast to tell cpp to treat this variable's memory as an 8-byte
    // destination buffer. We know its size is 8 bytes, but we can also use sizeof.
    safetensors_file.read(reinterpret_cast<char *>(&headersize), 8);
    if (!safetensors_file)
    {
        std::cerr << "could not read safetensors file";
        return -1;
    }
    std::cout << headersize << '\n';
    // The next headersize bytes of the file contain the json header which
    // describes all tensors. We use each tensor's data_offsets values to create
    // pointers later, then allocate max_offset bytes of memory on the GPU.
    // A tensor is not the same thing as a layer: each transformer layer contains
    // multiple tensors.
    // String of headersize init with null char.
    std::string header(headersize, '\0');
    // header.data points to string's actual character buffer.
    safetensors_file.read(header.data(), headersize);

    // Convert this bigass string into a json object.
    json header_json = json::parse(header);

    std::cout << header_json;

    // The way we are going to store tensor -> starting byte offset is by using
    // a hashmap.
    std::unordered_map<std::string, uint64_t> offsets;
    // We need to know the largest tensor end offset in the raw tensor-data
    // bytes, as that tells us how much memory we need to allocate when we copy
    // the ENTIRE raw tensor data from the model file to our gpu.
    uint64_t max_offset = 0;
    // Read-only: access key and value from pair from JSON header.
    for (const auto &[name, tensor_info] : header_json.items())
    {
        if (name == "__metadata__")
        {
            continue;
        }
        // You can read how they structured the json object from when we printed
        // the entire header. It will help us with parsing.
        const auto &data_offsets = tensor_info.at("data_offsets");
        // We want to convert it to a 64 bit integer as it is still a JSON value.
        uint64_t start_offset = data_offsets.at(0).get<uint64_t>();
        uint64_t end_offset = data_offsets.at(1).get<uint64_t>();
        // Add to hashmap; effeciently constructs it in place inside map memory,
        // avoiding temporary memory.
        offsets.emplace(name, start_offset);
        max_offset = std::max(max_offset, end_offset);

        // Test.
        // std::cout << name << " starts at: " << start_offset
        //           << " ends at: " << end_offset << "\n";
        // std::cout << "we have to allocate this many bytes: " << (end_offset / 8) << '\n';
    }
    // Now we have to read the raw tensor data, copy it to memory, and then copy
    // it to the gpu. Our file ptr is now at the start of tensor data: it started
    // at 0, read the header size, then read the json header.
    std::vector<char> model_weights_cpu(max_offset);
    // model_weights_cpu.data() gets the addr of 1st bytes like &array[0].
    safetensors_file.read(model_weights_cpu.data(),
                          static_cast<std::streamsize>(max_offset));

    // Just give cuda a nullptr and the size of buffer, and we can return the
    // starting address of the buffer with size given.
    void *model_weights_gpu = nullptr;
    cudaMalloc(&model_weights_gpu, max_offset);
    // Now we can copy from cpu memory to gpu memory.

    cudaMemcpy(model_weights_gpu, model_weights_cpu.data(), model_weights_cpu.size(), cudaMemcpyHostToDevice);

    // Create a hashmap of names of tensors and pointers to them.

    /*
    std::unordered_map<std::string, __nv_bfloat16*> tensor_pointers;
    // We cast as model weights gpu is a generic pointer to start of whole gpu buffer.
    char* base = static_cast<char*>(model_weights_gpu);
    // Go through hashmap of offsets, so we can make a cpu lookup table to see
    // where things are in the GPU.
    for (const auto& [name, start_offset] : offsets) {
        __nv_bfloat16* tensor_pointer =
            reinterpret_cast<__nv_bfloat16*>(
                base + start_offset
            );

        tensor_pointers.emplace(name, tensor_pointer);
    }
    */
    weights.model_storage = model_weights_gpu;
    char *base = static_cast<char *>(weights.model_storage);

    weights.embed_tokens = reinterpret_cast<__nv_bfloat16 *>(
        base + offsets.at("model.embed_tokens.weight"));

    weights.norm = reinterpret_cast<__nv_bfloat16 *>(base + offsets.at("model.norm.weight"));
    // Must wire pointers to 2 rmsnorm vectors, four attention matrices, and 3
    // mlp matrices, so each of the 16 layers has 9 weight tensors.
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        std::string prefix = "model.layers." + std::to_string(layer);
        weights.w_q[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.q_proj.weight"));
        weights.w_k[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.k_proj.weight"));
        weights.w_v[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.v_proj.weight"));
        weights.w_o[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.o_proj.weight"));

        weights.input_layernorm[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".input_layernorm.weight"));

        weights.post_attn_layernorms[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".post_attention_layernorm.weight"));

        weights.mlp_gate_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.gate_proj.weight"));

        weights.mlp_up_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.up_proj.weight"));

        weights.mlp_down_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.down_proj.weight"));
    }
    return 0;
}

// Spent abount 1hr 30min making the loader except using Mmap this time
int loadModelMmap(LLamaWeights &weights)
{
    std::string path = "models/llama-3.2-1b-instruct/model.safetensors";
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
    {
        std::cerr << "could not open model file";
        return -1;
    }
    // why are we not using read? remember the whole point of using mmap is that we don't copy it to
    // RAM, this avoids that while still allowing us to know the number of bytes in the file.
    struct stat file_info;
    if (fstat(fd, &file_info) == -1)
    {
        std::cerr << "failure to get file info";
        close(fd);
        return -1;
    }
    long file_size = file_info.st_size;
    if (file_size < 8)
    {
        std::cerr << "file size too small";
        close(fd);
        return -1;
    }

    // treat this pointer like the pointer to the first byte of the file
    void *mapped_file = mmap(nullptr, file_info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped_file == MAP_FAILED)
    {
        std::cerr << "memory map failed";
        return -1;
    }
    uint64_t header_size = 0;
    // we must cast our void pointer to one of a char pointer
    const char *file_bytes = reinterpret_cast<char *>(mapped_file);
    std::memcpy(&header_size, file_bytes, sizeof(header_size));
    if (!(header_size <= file_size - 8))
    {
        std::cerr << "header size too large";
        munmap(mapped_file, file_size);
        return -1;
    }

    std::string header(header_size, '\0');

    std::memcpy(header.data(), file_bytes + sizeof(header_size), header_size);
    json header_json = json::parse(header);

    std::unordered_map<std::string, uint64_t> offsets;

    uint64_t max_offset = 0;
    // Read-only: access key and value from pair from JSON header.
    for (const auto &[name, tensor_info] : header_json.items())
    {
        if (name == "__metadata__")
        {
            continue;
        }

        const auto &data_offsets = tensor_info.at("data_offsets");

        uint64_t start_offset = data_offsets.at(0).get<uint64_t>();
        uint64_t end_offset = data_offsets.at(1).get<uint64_t>();
        if (end_offset > file_size - 8 - header_size || !(start_offset <= end_offset))
        {
            std::cerr << "end offset out of bounds";
            munmap(mapped_file, file_size);
            return -1;
        }

        offsets.emplace(name, start_offset);
        max_offset = std::max(max_offset, end_offset);

        // Test.
        std::cout << name << " starts at: " << start_offset
                  << " ends at: " << end_offset << "\n";
        std::cout << "we have to allocate this many bytes: " << (end_offset / 8) << '\n';
    }
    const char *tensor_data = file_bytes + sizeof(header_size) + header_size;

    void *modelweights_gpu = nullptr;

    if (cudaMalloc(&modelweights_gpu, max_offset) != 0)
    {
        std::cerr << "gpu mem allocation failed";
        munmap(mapped_file, file_size);
        return -1;
    }
    if (cudaMemcpy(modelweights_gpu, tensor_data, max_offset, cudaMemcpyHostToDevice) != 0)
    {
        std::cerr << "gpu mem copy failed";
        cudaFree(modelweights_gpu);
        munmap(mapped_file, file_size);
        return -1;
    }
    munmap(mapped_file, file_size);

    weights.model_storage = modelweights_gpu;
    char *base = static_cast<char *>(weights.model_storage);

    weights.embed_tokens = reinterpret_cast<__nv_bfloat16 *>(
        base + offsets.at("model.embed_tokens.weight"));

    weights.norm = reinterpret_cast<__nv_bfloat16 *>(base + offsets.at("model.norm.weight"));
    // Must wire pointers to 2 rmsnorm vectors, four attention matrices, and 3
    // mlp matrices, so each of the 16 layers has 9 weight tensors.
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        std::string prefix = "model.layers." + std::to_string(layer);
        weights.w_q[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.q_proj.weight"));
        weights.w_k[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.k_proj.weight"));
        weights.w_v[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.v_proj.weight"));
        weights.w_o[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.o_proj.weight"));

        weights.input_layernorm[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".input_layernorm.weight"));

        weights.post_attn_layernorms[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".post_attention_layernorm.weight"));

        weights.mlp_gate_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.gate_proj.weight"));

        weights.mlp_up_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.up_proj.weight"));

        weights.mlp_down_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.down_proj.weight"));
    }
    return 0;
}

std::vector<int> tokenize(Tokenizer &tokenizer)
{
    std::string prompt;
    // std::cin >> prompt;
    prompt = "Hello World! "; // temporary hardcoded prompt
    std::vector<int> token_ids = tokenizer.encode(prompt);
    return token_ids;
}

int prefill(
    const std::vector<int> &token_ids,
    LLamaWeights &weights,
    const float *cos_table_gpu,
    const float *sin_table_gpu)
{
    if (token_ids.empty() ||
        cos_table_gpu == nullptr ||
        sin_table_gpu == nullptr)
    {
        std::cerr << "invalid prefill input\n";
        return -1;
    }

    const int token_count = static_cast<int>(token_ids.size());
    int *token_id_gpu = nullptr;
    int *position_ids_gpu = nullptr;
    __nv_bfloat16 *activations_gpu = nullptr;
    __nv_bfloat16 *normalized_gpu = nullptr;
    __nv_bfloat16 *q_gpu = nullptr;
    __nv_bfloat16 *k_gpu = nullptr;
    __nv_bfloat16 *v_gpu = nullptr;
    __nv_bfloat16 *attention_scores_gpu = nullptr;
    __nv_bfloat16 *attention_output_gpu = nullptr;
    __nv_bfloat16 *output_projection_gpu = nullptr;
    cublasHandle_t cublas_handle = nullptr;

    auto free_prefill_buffers = [&]()
    {
        cudaFree(token_id_gpu);
        cudaFree(position_ids_gpu);
        cudaFree(activations_gpu);
        cudaFree(normalized_gpu);
        if (cublas_handle != nullptr)
        {
            cublasDestroy(cublas_handle);
        }

        cudaFree(q_gpu);
        cudaFree(k_gpu);
        cudaFree(v_gpu);
        cudaFree(attention_scores_gpu);
        cudaFree(attention_output_gpu);
        cudaFree(output_projection_gpu);
    };

    if (cudaMalloc(&token_id_gpu, token_ids.size() * sizeof(int)) != 0)
    {
        std::cerr << "gpu token mem allocation failed";
        return -1;
    }
    if (cudaMemcpy(token_id_gpu, token_ids.data(), token_ids.size() * sizeof(int), cudaMemcpyHostToDevice) != 0)
    {
        std::cerr << "gpu token mem copy failed";
        free_prefill_buffers();
        return -1;
    }

    std::vector<int> position_ids_cpu(token_count);
    std::iota(position_ids_cpu.begin(), position_ids_cpu.end(), 0);

    if (cudaMalloc(
            &position_ids_gpu,
            position_ids_cpu.size() * sizeof(int)) != cudaSuccess)
    {
        std::cerr << "gpu position ID allocation failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (cudaMemcpy(
            position_ids_gpu,
            position_ids_cpu.data(),
            position_ids_cpu.size() * sizeof(int),
            cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "gpu position ID copy failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (cudaMalloc(&activations_gpu, token_ids.size() * 2048 * sizeof(__nv_bfloat16)) != 0)
    {
        std::cerr << "gpu activation token mem allocation failed";
        free_prefill_buffers();
        return -1;
    }

    if (launchEmbeddingGather(
            token_id_gpu,
            weights.embed_tokens,
            activations_gpu,
            token_count) != cudaSuccess)
    {
        std::cerr << "embedding kernel launch failed";
        free_prefill_buffers();
        return -1;
    }

    const size_t activation_bytes =
        static_cast<size_t>(token_count) *
        2048 *
        sizeof(__nv_bfloat16);

    if (cudaMalloc(&normalized_gpu, activation_bytes) != cudaSuccess)
    {
        std::cerr << "gpu RMSNorm output allocation failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (launchRmsNorm(
            activations_gpu,
            normalized_gpu,
            weights.input_layernorm[0],
            token_count) != cudaSuccess)
    {
        std::cerr << "RMSNorm kernel launch failed\n";
        free_prefill_buffers();
        return -1;
    }

    cudaError_t execution_error = cudaDeviceSynchronize();

    if (execution_error != cudaSuccess)
    {
        std::cerr << "RMSNorm execution failed: "
                  << cudaGetErrorString(execution_error)
                  << '\n';
        free_prefill_buffers();
        return -1;
    }

    const size_t q_bytes =
        static_cast<size_t>(token_count) *
        HIDDEN_SIZE *
        sizeof(__nv_bfloat16);

    const size_t kv_bytes =
        static_cast<size_t>(token_count) *
        KV_DIM *
        sizeof(__nv_bfloat16);

    if (cudaMalloc(&q_gpu, q_bytes) != cudaSuccess ||
        cudaMalloc(&k_gpu, kv_bytes) != cudaSuccess ||
        cudaMalloc(&v_gpu, kv_bytes) != cudaSuccess)
    {
        std::cerr << "Q/K/V buffer allocation failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS initialization failed\n";
        free_prefill_buffers();
        return -1;
    }

    // This milestone only runs layer 0. Later, this becomes a loop over all layers.
    constexpr int layer = 0;

    const cublasStatus_t q_status = projectBf16RowMajor(
        cublas_handle,
        normalized_gpu,
        weights.w_q[layer],
        q_gpu,
        token_count,
        HIDDEN_SIZE);

    const cublasStatus_t k_status = projectBf16RowMajor(
        cublas_handle,
        normalized_gpu,
        weights.w_k[layer],
        k_gpu,
        token_count,
        KV_DIM);

    const cublasStatus_t v_status = projectBf16RowMajor(
        cublas_handle,
        normalized_gpu,
        weights.w_v[layer],
        v_gpu,
        token_count,
        KV_DIM);

    if (q_status != CUBLAS_STATUS_SUCCESS ||
        k_status != CUBLAS_STATUS_SUCCESS ||
        v_status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Q/K/V projection failed\n";
        free_prefill_buffers();
        return -1;
    }

    // Position belongs in the Q–K comparison, so rotate Q and K only.
    if (launchRope(
            q_gpu, position_ids_gpu, cos_table_gpu, sin_table_gpu,
            token_count, HIDDEN_SIZE, HEAD_DIM) != cudaSuccess ||
        launchRope(
            k_gpu, position_ids_gpu, cos_table_gpu, sin_table_gpu,
            token_count, KV_DIM, HEAD_DIM) != cudaSuccess)
    {
        std::cerr << "RoPE application to Q/K failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        std::cerr << "Q/K/V projection or RoPE execution failed\n";
        free_prefill_buffers();
        return -1;
    }

    // GQA keeps 32 query heads but shares 8 key/value heads. Scores are
    // [query_head, query_token, key_token], so each Q head has one T x T
    // matrix and maps to KV head query_head / 4.
    const size_t attention_score_bytes =
        static_cast<size_t>(NUM_QUERY_HEADS) *
        token_count *
        token_count *
        sizeof(__nv_bfloat16);

    if (cudaMalloc(&attention_scores_gpu, attention_score_bytes) != cudaSuccess ||
        cudaMalloc(&attention_output_gpu, activation_bytes) != cudaSuccess ||
        cudaMalloc(&output_projection_gpu, activation_bytes) != cudaSuccess)
    {
        std::cerr << "attention buffer allocation failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (computeGqaAttentionScores(
            cublas_handle,
            q_gpu,
            k_gpu,
            attention_scores_gpu,
            token_count) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "GQA score projection failed\n";
        free_prefill_buffers();
        return -1;
    }

    // Prevent future tokens from influencing the current token, then turn
    // every score row into a stable probability distribution.
    if (launchCausalMask(
            attention_scores_gpu,
            token_count,
            NUM_QUERY_HEADS) != cudaSuccess ||
        launchStableSoftmax(
            attention_scores_gpu,
            token_count,
            NUM_QUERY_HEADS) != cudaSuccess)
    {
        std::cerr << "causal mask or stable softmax launch failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (mixGqaValues(
            cublas_handle,
            attention_scores_gpu,
            v_gpu,
            attention_output_gpu,
            token_count) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "GQA value mixing failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (projectBf16RowMajor(
            cublas_handle,
            attention_output_gpu,
            weights.w_o[layer],
            output_projection_gpu,
            token_count,
            HIDDEN_SIZE) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "attention output projection failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (launchResidualAdd(
            activations_gpu,
            output_projection_gpu,
            token_count,
            HIDDEN_SIZE) != cudaSuccess)
    {
        std::cerr << "attention residual launch failed\n";
        free_prefill_buffers();
        return -1;
    }

    execution_error = cudaDeviceSynchronize();
    if (execution_error != cudaSuccess)
    {
        std::cerr << "attention execution failed: "
                  << cudaGetErrorString(execution_error)
                  << '\n';
        free_prefill_buffers();
        return -1;
    }

    free_prefill_buffers();
    return 0;
}

int main()
{
    // checkGPUStatus();
    LLamaWeights weights{};
    // temporarily using prompt as hello world, will be modified to take user input
    const std::string prompt = "Hello World!";
    const std::string tokenizer_path = "models/llama-3.2-1b-instruct/tokenizer.model";
    if (loadLlamaModel(weights) != 0)
    {
        return -1;
    }

    constexpr int MAX_SEQUENCE_LENGTH = 2048;
    float *cos_table_gpu = nullptr;
    float *sin_table_gpu = nullptr;

    cudaError_t rope_initialization_error = initializeRopeTables(
        &cos_table_gpu,
        &sin_table_gpu,
        MAX_SEQUENCE_LENGTH);

    if (rope_initialization_error != cudaSuccess)
    {
        std::cerr << "RoPE table initialization failed: "
                  << cudaGetErrorString(rope_initialization_error)
                  << '\n';
        cudaFree(weights.model_storage);
        return -1;
    }

    Tokenizer tokenizer;
    std::string tokenizer_error;

    if (!tokenizer.load("models/llama-3.2-1b-instruct/tokenizer.model",
                        &tokenizer_error))
    {
        std::cerr << tokenizer_error << '\n';
        cudaFree(cos_table_gpu);
        cudaFree(sin_table_gpu);
        cudaFree(weights.model_storage);
        return -1;
    }

    std::vector<int> token_ids = tokenize(tokenizer);
    const int prefill_status = prefill(
        token_ids,
        weights,
        cos_table_gpu,
        sin_table_gpu);

    cudaFree(cos_table_gpu);
    cudaFree(sin_table_gpu);
    cudaFree(weights.model_storage);

    return prefill_status;
}
