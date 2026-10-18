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
#include <stdexcept>
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
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int VOCAB_SIZE = 128256;
constexpr int MAX_SEQUENCE_LENGTH = 2048;
constexpr int BATCH_SIZE = 2;
constexpr int HEAD_DIM = 64;
constexpr int NUM_QUERY_HEADS = HIDDEN_SIZE / HEAD_DIM;
constexpr int NUM_KV_HEADS = KV_DIM / HEAD_DIM;
constexpr int QUERY_HEADS_PER_KV_HEAD = NUM_QUERY_HEADS / NUM_KV_HEADS;

static_assert(HIDDEN_SIZE % HEAD_DIM == 0);
static_assert(KV_DIM % HEAD_DIM == 0);
static_assert(NUM_QUERY_HEADS % NUM_KV_HEADS == 0);

// Computes:
// output[token, output_feature] =
//     sum(input[token, input_feature] * weight[output_feature, input_feature])
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
    int input_features,
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
        input_features,
        &alpha,
        weight,
        CUDA_R_16BF,
        input_features,
        input,
        CUDA_R_16BF,
        input_features,
        &beta,
        output,
        CUDA_R_16BF,
        output_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
}
// Attention path (repeated for each layer):
// hidden state
// RMSNORM
// Q,K,V Projection
// RoPE on Q and K
// Scaled QK Attention scores
// causal mask
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
    int token_count,
    int key_count = 0)
{
    if (key_count == 0)
    {
        key_count = token_count; // Original square prefill matrix.
    }
    if (cublas_handle == nullptr || q == nullptr || k == nullptr ||
        scores == nullptr || token_count <= 0 || key_count <= 0)
    {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const float alpha = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    const float beta = 0.0f;
    const size_t scores_per_head =
        static_cast<size_t>(token_count) * key_count;

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
            key_count,
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
            key_count,
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
    int token_count,
    int key_count = 0)
{
    if (key_count == 0)
    {
        key_count = token_count; // Original square prefill matrix.
    }
    if (cublas_handle == nullptr || probabilities == nullptr || v == nullptr ||
        attention_output == nullptr || token_count <= 0 || key_count <= 0)
    {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const size_t scores_per_head =
        static_cast<size_t>(token_count) * key_count;

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
            key_count,
            &alpha,
            value_head,
            CUDA_R_16BF,
            KV_DIM,
            probability_head,
            CUDA_R_16BF,
            key_count,
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
    // Llama 3.2 1B normally ties this to the embedding table.
    __nv_bfloat16 *lm_head = nullptr;
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

// One sequence, stored as [layer, position, kv_feature] in each allocation.
// Keys are already rotated by RoPE; values are unrotated.
struct KVCache
{
    __nv_bfloat16 *keys = nullptr;
    __nv_bfloat16 *values = nullptr;
    int capacity = 0;
    int length = 0; // Number of tokens actually processed, not predicted.
};

// Shared by prefill and decode, across all layers and requests.
struct InferenceBuffers
{
    cublasHandle_t cublas_handle = nullptr;
    int token_capacity = 0;
    int *token_ids_gpu = nullptr;
    int *position_ids_gpu = nullptr;
    int *next_tokens_gpu = nullptr;
    __nv_bfloat16 *activations_gpu = nullptr;
    __nv_bfloat16 *normalized_gpu = nullptr;
    __nv_bfloat16 *projection_gpu = nullptr;
    __nv_bfloat16 *k_gpu = nullptr;
    __nv_bfloat16 *v_gpu = nullptr;
    __nv_bfloat16 *attention_scores_gpu = nullptr;
    __nv_bfloat16 *gate_gpu = nullptr;
    __nv_bfloat16 *up_gpu = nullptr;
    float *logits_gpu = nullptr;
};

struct Request
{
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    int token_budget = 0;
    int next_token_id = -1;
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
#ifdef DEBUG
    std::cout << headersize << '\n';
#endif
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

#ifdef DEBUG
    std::cout << header_json;
#endif

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
    weights.lm_head = offsets.count("lm_head.weight") != 0
                          ? reinterpret_cast<__nv_bfloat16 *>(base + offsets.at("lm_head.weight"))
                          : weights.embed_tokens;

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
    weights.lm_head = offsets.count("lm_head.weight") != 0
                          ? reinterpret_cast<__nv_bfloat16 *>(base + offsets.at("lm_head.weight"))
                          : weights.embed_tokens;

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

std::vector<int> tokenize(Tokenizer &tokenizer, const std::string &prompt)
{
    std::vector<int> token_ids = tokenizer.encode(prompt);
    return token_ids;
}

void freeKvCache(KVCache &cache)
{
    cudaFree(cache.keys);
    cudaFree(cache.values);
    cache = KVCache{};
}

int initializeKvCache(KVCache &cache, int capacity)
{
    if (capacity <= 0 || capacity > MAX_SEQUENCE_LENGTH ||
        cache.keys != nullptr || cache.values != nullptr)
    {
        std::cerr << "invalid KV cache capacity or cache already allocated\n";
        return -1;
    }

    const size_t bytes = static_cast<size_t>(N_LAYERS) * capacity *
                         KV_DIM * sizeof(__nv_bfloat16);
    if (cudaMalloc(&cache.keys, bytes) != cudaSuccess ||
        cudaMalloc(&cache.values, bytes) != cudaSuccess)
    {
        std::cerr << "KV cache allocation failed\n";
        freeKvCache(cache);
        return -1;
    }
    cache.capacity = capacity;
    cache.length = 0;
    return 0;
}

void freeInferenceBuffers(InferenceBuffers &buffers)
{
    cudaFree(buffers.token_ids_gpu);
    cudaFree(buffers.position_ids_gpu);
    cudaFree(buffers.next_tokens_gpu);
    cudaFree(buffers.activations_gpu);
    cudaFree(buffers.normalized_gpu);
    cudaFree(buffers.projection_gpu);
    cudaFree(buffers.k_gpu);
    cudaFree(buffers.v_gpu);
    cudaFree(buffers.attention_scores_gpu);
    cudaFree(buffers.gate_gpu);
    cudaFree(buffers.up_gpu);
    cudaFree(buffers.logits_gpu);
    if (buffers.cublas_handle != nullptr)
    {
        cublasDestroy(buffers.cublas_handle);
    }
    buffers = InferenceBuffers{};
}

int prefill(
    const std::vector<int> &token_ids,
    LLamaWeights &weights,
    KVCache &cache,
    InferenceBuffers &buffers,
    const float *cos_table_gpu,
    const float *sin_table_gpu,
    int &next_token_id,
    std::vector<float> *logits_cpu = nullptr)
{
    cache.length = 0; // A new prompt replaces the previous sequence.
    if (token_ids.empty() ||
        token_ids.size() > MAX_SEQUENCE_LENGTH ||
        token_ids.size() > static_cast<size_t>(buffers.token_capacity) ||
        cache.keys == nullptr || cache.values == nullptr ||
        cache.capacity <= 0 || cache.capacity > MAX_SEQUENCE_LENGTH ||
        token_ids.size() > static_cast<size_t>(cache.capacity) ||
        std::any_of(token_ids.begin(), token_ids.end(),
                    [](int id) { return id < 0 || id >= VOCAB_SIZE; }) ||
        cos_table_gpu == nullptr ||
        sin_table_gpu == nullptr)
    {
        std::cerr << "invalid prefill input\n";
        return -1;
    }

    const int token_count = static_cast<int>(token_ids.size());
    int *&token_id_gpu = buffers.token_ids_gpu;
    int *&position_ids_gpu = buffers.position_ids_gpu;
    __nv_bfloat16 *&activations_gpu = buffers.activations_gpu;
    __nv_bfloat16 *&normalized_gpu = buffers.normalized_gpu;
    __nv_bfloat16 *&q_gpu = buffers.projection_gpu;
    __nv_bfloat16 *&k_gpu = buffers.k_gpu;
    __nv_bfloat16 *&v_gpu = buffers.v_gpu;
    __nv_bfloat16 *&attention_scores_gpu = buffers.attention_scores_gpu;
    __nv_bfloat16 *attention_output_gpu = nullptr;
    __nv_bfloat16 *output_projection_gpu = nullptr;
    __nv_bfloat16 *&gate_gpu = buffers.gate_gpu;
    __nv_bfloat16 *&up_gpu = buffers.up_gpu;
    float *&logits_gpu = buffers.logits_gpu;
    int *&next_token_gpu = buffers.next_tokens_gpu;
    cublasHandle_t &cublas_handle = buffers.cublas_handle;

    auto free_prefill_buffers = [&]()
    {
        freeInferenceBuffers(buffers);
    };

    // Allocate on the first prefill, then keep these buffers for every request.
    if (token_id_gpu == nullptr &&
        cudaMalloc(&token_id_gpu, buffers.token_capacity * sizeof(int)) != 0)
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

    if (position_ids_gpu == nullptr &&
        cudaMalloc(&position_ids_gpu, buffers.token_capacity * sizeof(int)) != cudaSuccess)
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

    if (activations_gpu == nullptr &&
        cudaMalloc(&activations_gpu, static_cast<size_t>(buffers.token_capacity) * 2048 * sizeof(__nv_bfloat16)) != 0)
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
        static_cast<size_t>(buffers.token_capacity) *
        2048 *
        sizeof(__nv_bfloat16);

    if (normalized_gpu == nullptr &&
        cudaMalloc(&normalized_gpu, activation_bytes) != cudaSuccess)
    {
        std::cerr << "gpu RMSNorm output allocation failed\n";
        free_prefill_buffers();
        return -1;
    }

    // Scratch buffers are reused by every layer.
    const size_t mlp_bytes =
        static_cast<size_t>(buffers.token_capacity) *
        INTERMEDIATE_SIZE *
        sizeof(__nv_bfloat16);
    if ((gate_gpu == nullptr && cudaMalloc(&gate_gpu, mlp_bytes) != cudaSuccess) ||
        (up_gpu == nullptr && cudaMalloc(&up_gpu, mlp_bytes) != cudaSuccess))
    {
        std::cerr << "MLP buffer allocation failed\n";
        free_prefill_buffers();
        return -1;
    }

    // Each layer starts with its own input norm and finishes with its MLP residual.
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
    if (launchRmsNorm(
            activations_gpu,
            normalized_gpu,
            weights.input_layernorm[layer],
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
        static_cast<size_t>(buffers.token_capacity) *
        HIDDEN_SIZE *
        sizeof(__nv_bfloat16);

    const size_t kv_bytes =
        static_cast<size_t>(token_count) *
        KV_DIM *
        sizeof(__nv_bfloat16);

    const size_t kv_buffer_bytes =
        static_cast<size_t>(buffers.token_capacity) * KV_DIM * sizeof(__nv_bfloat16);

    // Create Q/K/V buffers and the cuBLAS handle once, then reuse them.
    if (layer == 0 &&
        ((q_gpu == nullptr && cudaMalloc(&q_gpu, q_bytes) != cudaSuccess) ||
         (k_gpu == nullptr && cudaMalloc(&k_gpu, kv_buffer_bytes) != cudaSuccess) ||
         (v_gpu == nullptr && cudaMalloc(&v_gpu, kv_buffer_bytes) != cudaSuccess)))
    {
        std::cerr << "Q/K/V buffer allocation failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (layer == 0 && cublas_handle == nullptr && cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS initialization failed\n";
        free_prefill_buffers();
        return -1;
    }

    const cublasStatus_t q_status = projectBf16RowMajor(
        cublas_handle,
        normalized_gpu,
        weights.w_q[layer],
        q_gpu,
        token_count,
        HIDDEN_SIZE,
        HIDDEN_SIZE);

    const cublasStatus_t k_status = projectBf16RowMajor(
        cublas_handle,
        normalized_gpu,
        weights.w_k[layer],
        k_gpu,
        token_count,
        HIDDEN_SIZE,
        KV_DIM);

    const cublasStatus_t v_status = projectBf16RowMajor(
        cublas_handle,
        normalized_gpu,
        weights.w_v[layer],
        v_gpu,
        token_count,
        HIDDEN_SIZE,
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

    // Keep this layer's prompt K/V for the later decode steps. Keys have
    // already passed through RoPE; values are stored without rotation.
    const size_t layer_offset = static_cast<size_t>(layer) * cache.capacity * KV_DIM;
    if (cudaMemcpy(cache.keys + layer_offset, k_gpu, kv_bytes,
                   cudaMemcpyDeviceToDevice) != cudaSuccess ||
        cudaMemcpy(cache.values + layer_offset, v_gpu, kv_bytes,
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
    {
        std::cerr << "prefill KV cache copy failed\n";
        free_prefill_buffers();
        return -1;
    }

    // GQA keeps 32 query heads but shares 8 key/value heads. Scores are
    // [query_head, query_token, key_token], so each Q head has one T x T
    // matrix and maps to KV head query_head / 4.
    const size_t attention_score_bytes =
        static_cast<size_t>(NUM_QUERY_HEADS) *
        std::max(static_cast<size_t>(buffers.token_capacity) * buffers.token_capacity,
                 static_cast<size_t>(MAX_SEQUENCE_LENGTH)) *
        sizeof(__nv_bfloat16);

    // Q/K/V have consumed the norm; Q is consumed before the output projection.
    attention_output_gpu = normalized_gpu;
    output_projection_gpu = q_gpu;

    // Attention scratch has the same shape in every layer and covers long decode.
    if (layer == 0 && attention_scores_gpu == nullptr &&
        cudaMalloc(&attention_scores_gpu, attention_score_bytes) != cudaSuccess)
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
            HIDDEN_SIZE,
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

    // The second norm feeds two parallel 8192-wide projections.
    if (launchRmsNorm(
            activations_gpu,
            normalized_gpu,
            weights.post_attn_layernorms[layer],
            token_count) != cudaSuccess)
    {
        std::cerr << "post-attention RMSNorm launch failed\n";
        free_prefill_buffers();
        return -1;
    }

    const cublasStatus_t gate_status = projectBf16RowMajor(
        cublas_handle,
        normalized_gpu,
        weights.mlp_gate_proj[layer],
        gate_gpu,
        token_count,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE);

    const cublasStatus_t up_status = projectBf16RowMajor(
        cublas_handle,
        normalized_gpu,
        weights.mlp_up_proj[layer],
        up_gpu,
        token_count,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE);

    if (gate_status != CUBLAS_STATUS_SUCCESS ||
        up_status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "MLP gate/up projection failed\n";
        free_prefill_buffers();
        return -1;
    }

    // SwiGLU overwrites gate with SiLU(gate) * up. The attention projection
    // buffer is free to reuse for the 2048-wide MLP down projection.
    if (launchSwiGlu(gate_gpu, up_gpu, token_count, INTERMEDIATE_SIZE) != cudaSuccess ||
        projectBf16RowMajor(
            cublas_handle,
            gate_gpu,
            weights.mlp_down_proj[layer],
            output_projection_gpu,
            token_count,
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE) != CUBLAS_STATUS_SUCCESS ||
        launchResidualAdd(
            activations_gpu,
            output_projection_gpu,
            token_count,
            HIDDEN_SIZE) != cudaSuccess)
    {
        std::cerr << "MLP or residual launch failed\n";
        free_prefill_buffers();
        return -1;
    }

    execution_error = cudaDeviceSynchronize();
    if (execution_error != cudaSuccess)
    {
        std::cerr << "MLP execution failed in layer " << layer << ": "
                  << cudaGetErrorString(execution_error) << '\n';
        free_prefill_buffers();
        return -1;
    }
    } // next layer uses this layer's residual output as its hidden state

    // Only the last prompt position contributes the next-token logits.
    const __nv_bfloat16 *last_hidden =
        activations_gpu + static_cast<size_t>(token_count - 1) * HIDDEN_SIZE;
    if (launchRmsNorm(last_hidden, normalized_gpu, weights.norm, 1) != cudaSuccess)
    {
        std::cerr << "final RMSNorm launch failed\n";
        free_prefill_buffers();
        return -1;
    }

    if ((logits_gpu == nullptr &&
         cudaMalloc(&logits_gpu, BATCH_SIZE * VOCAB_SIZE * sizeof(float)) != cudaSuccess) ||
        (next_token_gpu == nullptr &&
         cudaMalloc(&next_token_gpu, BATCH_SIZE * sizeof(int)) != cudaSuccess))
    {
        std::cerr << "LM head buffer allocation failed\n";
        free_prefill_buffers();
        return -1;
    }

    // cuBLAS sees the row-major [vocab, hidden] weight as column-major
    // [hidden, vocab]. Transposing it gives [vocab, 1] FP32 logits.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (cublasGemmEx(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            VOCAB_SIZE, 1, HIDDEN_SIZE,
            &alpha, weights.lm_head, CUDA_R_16BF, HIDDEN_SIZE,
            normalized_gpu, CUDA_R_16BF, HIDDEN_SIZE,
            &beta, logits_gpu, CUDA_R_32F, VOCAB_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS ||
        launchArgmax(logits_gpu, next_token_gpu, VOCAB_SIZE) != cudaSuccess ||
        cudaMemcpy(&next_token_id, next_token_gpu, sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "LM head or argmax failed\n";
        free_prefill_buffers();
        return -1;
    }

    if (logits_cpu != nullptr)
    {
        logits_cpu->resize(VOCAB_SIZE);
        if (cudaMemcpy(logits_cpu->data(), logits_gpu, VOCAB_SIZE * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "logit copy failed\n";
            free_prefill_buffers();
            return -1;
        }
    }
    cache.length = token_count;
    return 0;
}

// Decode processes one token per active sequence, each with its own cached prefix.
int decode(
    const std::vector<int> &token_ids,
    LLamaWeights &weights,
    const std::vector<KVCache *> &caches,
    InferenceBuffers &buffers,
    const float *cos_table_gpu,
    const float *sin_table_gpu,
    std::vector<int> &next_token_ids,
    std::vector<float> *logits_cpu = nullptr)
{
    const int token_count = static_cast<int>(token_ids.size());
    if (token_count <= 0 || token_count > BATCH_SIZE ||
        token_count > buffers.token_capacity || token_ids.size() != caches.size() ||
        buffers.cublas_handle == nullptr ||
        cos_table_gpu == nullptr || sin_table_gpu == nullptr)
    {
        std::cerr << "invalid decode input\n";
        return -1;
    }
    std::vector<int> position_ids_cpu(token_count);
    for (int row = 0; row < token_count; ++row)
    {
        const KVCache *cache = caches[row];
        if (cache == nullptr || cache->keys == nullptr || cache->values == nullptr ||
            cache->capacity <= 0 || cache->capacity > MAX_SEQUENCE_LENGTH ||
            cache->length <= 0 || cache->length >= cache->capacity ||
            token_ids[row] < 0 || token_ids[row] >= VOCAB_SIZE)
        {
            std::cerr << "invalid decode token or KV cache is empty/full\n";
            return -1;
        }
        position_ids_cpu[row] = cache->length;
    }

    int *token_id_gpu = buffers.token_ids_gpu;
    int *position_ids_gpu = buffers.position_ids_gpu;
    __nv_bfloat16 *activations_gpu = buffers.activations_gpu;
    __nv_bfloat16 *normalized_gpu = buffers.normalized_gpu;
    // Q is consumed before the output and MLP down projections reuse its buffer.
    __nv_bfloat16 *q_gpu = buffers.projection_gpu;
    __nv_bfloat16 *k_gpu = buffers.k_gpu;
    __nv_bfloat16 *v_gpu = buffers.v_gpu;
    __nv_bfloat16 *attention_scores_gpu = buffers.attention_scores_gpu;
    // Q/K/V have consumed the norm before attention writes into its buffer.
    __nv_bfloat16 *attention_output_gpu = buffers.normalized_gpu;
    __nv_bfloat16 *output_projection_gpu = buffers.projection_gpu;
    __nv_bfloat16 *gate_gpu = buffers.gate_gpu;
    __nv_bfloat16 *up_gpu = buffers.up_gpu;
    float *logits_gpu = buffers.logits_gpu;
    int *next_token_gpu = buffers.next_tokens_gpu;
    cublasHandle_t cublas_handle = buffers.cublas_handle;

    // Position is the absolute sequence position, not zero for each step.
    if (cudaMemcpy(token_id_gpu, token_ids.data(), token_count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(position_ids_gpu, position_ids_cpu.data(), token_count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        launchEmbeddingGather(token_id_gpu, weights.embed_tokens, activations_gpu, token_count) != cudaSuccess)
    {
        std::cerr << "decode token/position upload or embedding failed\n";
        return -1;
    }

    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        // Project all active tokens together, then scatter their K/V rows.
        if (launchRmsNorm(activations_gpu, normalized_gpu,
                          weights.input_layernorm[layer], token_count) != cudaSuccess ||
            projectBf16RowMajor(cublas_handle, normalized_gpu, weights.w_q[layer],
                               q_gpu, token_count, HIDDEN_SIZE, HIDDEN_SIZE) != CUBLAS_STATUS_SUCCESS ||
            projectBf16RowMajor(cublas_handle, normalized_gpu, weights.w_k[layer],
                               k_gpu, token_count, HIDDEN_SIZE, KV_DIM) != CUBLAS_STATUS_SUCCESS ||
            projectBf16RowMajor(cublas_handle, normalized_gpu, weights.w_v[layer],
                               v_gpu, token_count, HIDDEN_SIZE, KV_DIM) != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "decode RMSNorm or Q/K/V projection failed\n";
            return -1;
        }

        // Rotate only the new Q/K. Old cached keys are already rotated.
        if (launchRope(q_gpu, position_ids_gpu, cos_table_gpu, sin_table_gpu,
                       token_count, HIDDEN_SIZE, HEAD_DIM) != cudaSuccess ||
            launchRope(k_gpu, position_ids_gpu, cos_table_gpu, sin_table_gpu,
                       token_count, KV_DIM, HEAD_DIM) != cudaSuccess)
        {
            std::cerr << "decode RoPE failed\n";
            return -1;
        }

        for (int row = 0; row < token_count; ++row)
        {
            KVCache &cache = *caches[row];
            const int position = cache.length;
            const int key_count = position + 1;
            const size_t layer_offset = static_cast<size_t>(layer) * cache.capacity * KV_DIM;
            __nv_bfloat16 *layer_keys = cache.keys + layer_offset;
            __nv_bfloat16 *layer_values = cache.values + layer_offset;
            __nv_bfloat16 *new_key = layer_keys + static_cast<size_t>(position) * KV_DIM;
            __nv_bfloat16 *new_value = layer_values + static_cast<size_t>(position) * KV_DIM;

            if (cudaMemcpy(new_key, k_gpu + row * KV_DIM, KV_DIM * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice) != cudaSuccess ||
                cudaMemcpy(new_value, v_gpu + row * KV_DIM, KV_DIM * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice) != cudaSuccess)
            {
                std::cerr << "decode KV cache copy failed\n";
                return -1;
            }

            // Each query attends only to its sequence's processed positions.
            // There are no future positions, so decode needs no mask.
            if (computeGqaAttentionScores(cublas_handle, q_gpu + row * HIDDEN_SIZE, layer_keys,
                                          attention_scores_gpu, 1, key_count) != CUBLAS_STATUS_SUCCESS ||
                launchStableSoftmax(attention_scores_gpu, 1, NUM_QUERY_HEADS,
                                    key_count) != cudaSuccess ||
                mixGqaValues(cublas_handle, attention_scores_gpu, layer_values,
                             attention_output_gpu + row * HIDDEN_SIZE, 1, key_count) != CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "decode attention failed\n";
                return -1;
            }
        }

        if (projectBf16RowMajor(cublas_handle, attention_output_gpu, weights.w_o[layer],
                               output_projection_gpu, token_count, HIDDEN_SIZE, HIDDEN_SIZE) != CUBLAS_STATUS_SUCCESS ||
            launchResidualAdd(activations_gpu, output_projection_gpu, token_count, HIDDEN_SIZE) != cudaSuccess ||
            launchRmsNorm(activations_gpu, normalized_gpu,
                          weights.post_attn_layernorms[layer], token_count) != cudaSuccess)
        {
            std::cerr << "decode attention residual or post-attention norm failed\n";
            return -1;
        }

        const cublasStatus_t gate_status = projectBf16RowMajor(
            cublas_handle, normalized_gpu, weights.mlp_gate_proj[layer],
            gate_gpu, token_count, HIDDEN_SIZE, INTERMEDIATE_SIZE);
        const cublasStatus_t up_status = projectBf16RowMajor(
            cublas_handle, normalized_gpu, weights.mlp_up_proj[layer],
            up_gpu, token_count, HIDDEN_SIZE, INTERMEDIATE_SIZE);
        if (gate_status != CUBLAS_STATUS_SUCCESS || up_status != CUBLAS_STATUS_SUCCESS ||
            launchSwiGlu(gate_gpu, up_gpu, token_count, INTERMEDIATE_SIZE) != cudaSuccess ||
            projectBf16RowMajor(cublas_handle, gate_gpu, weights.mlp_down_proj[layer],
                               output_projection_gpu, token_count, INTERMEDIATE_SIZE, HIDDEN_SIZE) != CUBLAS_STATUS_SUCCESS ||
            launchResidualAdd(activations_gpu, output_projection_gpu, token_count, HIDDEN_SIZE) != cudaSuccess)
        {
            std::cerr << "decode MLP or residual failed\n";
            return -1;
        }
        const cudaError_t execution_error = cudaDeviceSynchronize();
        if (execution_error != cudaSuccess)
        {
            std::cerr << "decode execution failed in layer " << layer << ": "
                      << cudaGetErrorString(execution_error) << '\n';
            return -1;
        }
    }

    next_token_ids.resize(token_count);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (launchRmsNorm(activations_gpu, normalized_gpu, weights.norm, token_count) != cudaSuccess ||
        cublasGemmEx(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            VOCAB_SIZE, token_count, HIDDEN_SIZE,
            &alpha, weights.lm_head, CUDA_R_16BF, HIDDEN_SIZE,
            normalized_gpu, CUDA_R_16BF, HIDDEN_SIZE,
            &beta, logits_gpu, CUDA_R_32F, VOCAB_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS ||
        launchArgmax(logits_gpu, next_token_gpu, VOCAB_SIZE, token_count) != cudaSuccess ||
        cudaMemcpy(next_token_ids.data(), next_token_gpu, token_count * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "decode final norm, LM head, or argmax failed\n";
        return -1;
    }
    if (logits_cpu != nullptr)
    {
        logits_cpu->resize(static_cast<size_t>(token_count) * VOCAB_SIZE);
        if (cudaMemcpy(logits_cpu->data(), logits_gpu, logits_cpu->size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "decode logit copy failed\n";
            return -1;
        }
    }
    for (KVCache *cache : caches)
    {
        ++cache->length;
    }
    return 0;
}

int generateRequests(
    std::vector<Request> &requests,
    LLamaWeights &weights,
    InferenceBuffers &buffers,
    Tokenizer &tokenizer,
    const float *cos_table_gpu,
    const float *sin_table_gpu,
    bool continuous_batching)
{
    KVCache caches[BATCH_SIZE]{};
    int slot_requests[BATCH_SIZE];
    std::fill_n(slot_requests, BATCH_SIZE, -1);
    std::queue<int> pending;
    for (int request = 0; request < static_cast<int>(requests.size()); ++request)
    {
        pending.push(request);
    }
    auto free_caches = [&]()
    {
        for (KVCache &cache : caches)
        {
            freeKvCache(cache);
        }
    };
    for (int slot = 0; slot < BATCH_SIZE && slot < static_cast<int>(requests.size()); ++slot)
    {
        if (initializeKvCache(caches[slot], MAX_SEQUENCE_LENGTH) != 0)
        {
            free_caches();
            return -1;
        }
    }

    // Consume predictions from both prefill and decode in exactly the same way.
    auto accept_token = [&](int slot)
    {
        const int request_index = slot_requests[slot];
        Request &request = requests[request_index];
        const bool stop = request.next_token_id == tokenizer.eot_id() ||
                          request.next_token_id == tokenizer.eos_id();
        if (!stop)
        {
            request.generated_tokens.push_back(request.next_token_id);
        }
        if (!stop && static_cast<int>(request.generated_tokens.size()) < request.token_budget)
        {
            return;
        }

        if (requests.size() > 1)
        {
            std::cout << "Request " << request_index + 1 << ":\n";
        }
        // Decode the full byte sequence so split UTF-8 tokens join correctly.
        std::cout << "Response: " << tokenizer.decode(request.generated_tokens) << '\n';
        std::cout << "Generated token IDs:";
        for (int token : request.generated_tokens)
        {
            std::cout << ' ' << token;
        }
        std::cout << '\n';
        if (!stop && request.prompt_tokens.size() + request.generated_tokens.size() == MAX_SEQUENCE_LENGTH)
        {
            std::cout << "Stopped at the context limit.\n";
        }
        slot_requests[slot] = -1;
        caches[slot].length = 0; // Old bytes are ignored and overwritten on reuse.
    };

    std::vector<int> active_slots, active_tokens, next_tokens;
    std::vector<KVCache *> active_caches;
    active_slots.reserve(BATCH_SIZE);
    active_tokens.reserve(BATCH_SIZE);
    next_tokens.reserve(BATCH_SIZE);
    active_caches.reserve(BATCH_SIZE);
    while (true)
    {
        const bool batch_empty = std::all_of(
            slot_requests, slot_requests + BATCH_SIZE, [](int request)
            { return request == -1; });
        // Static batches wait for the entire group. Continuous batches refill now.
        if (continuous_batching || batch_empty)
        {
            for (int slot = 0; slot < BATCH_SIZE; ++slot)
            {
                while (slot_requests[slot] == -1 && !pending.empty())
                {
                    const int request_index = pending.front();
                    pending.pop();
                    slot_requests[slot] = request_index;
                    Request &request = requests[request_index];
                    if (prefill(request.prompt_tokens, weights, caches[slot], buffers,
                                cos_table_gpu, sin_table_gpu, request.next_token_id) != 0)
                    {
                        free_caches();
                        return -1;
                    }
                    accept_token(slot); // The first generated token counts toward the budget.
                    if (!continuous_batching)
                    {
                        break;
                    }
                }
            }
        }

        active_slots.clear();
        active_tokens.clear();
        active_caches.clear();
        for (int slot = 0; slot < BATCH_SIZE; ++slot)
        {
            if (slot_requests[slot] != -1)
            {
                active_slots.push_back(slot);
                active_tokens.push_back(requests[slot_requests[slot]].next_token_id);
                active_caches.push_back(&caches[slot]);
            }
        }
        if (active_slots.empty())
        {
            if (pending.empty())
            {
                break;
            }
            continue;
        }
        if (decode(active_tokens, weights, active_caches, buffers,
                   cos_table_gpu, sin_table_gpu, next_tokens) != 0)
        {
            free_caches();
            return -1;
        }
        for (int row = 0; row < static_cast<int>(active_slots.size()); ++row)
        {
            const int slot = active_slots[row];
            requests[slot_requests[slot]].next_token_id = next_tokens[row];
            accept_token(slot);
        }
    }
    free_caches();
    return 0;
}

#ifndef SAMIR_VLLM_NO_MAIN
int main(int argc, char **argv)
{
    // checkGPUStatus();
    int first_arg = 1;
    bool continuous_batching = true;
    if (argc > 1 && (std::strcmp(argv[1], "--static") == 0 ||
                     std::strcmp(argv[1], "--continuous") == 0))
    {
        continuous_batching = std::strcmp(argv[1], "--static") != 0;
        ++first_arg;
    }
    if (argc > first_arg && std::strcmp(argv[first_arg], "--help") == 0)
    {
        std::cout << "Usage: samir-vllm [--static|--continuous] [prompt] [max_new_tokens] [extra_prompt ...]\n";
        return 0;
    }
    LLamaWeights weights{};
    const std::string prompt = argc > first_arg ? argv[first_arg] : "Hello World!";
    const std::string tokenizer_path = "models/llama-3.2-1b-instruct/tokenizer.model";
    int max_new_tokens = 128;
    if (argc > first_arg + 1)
    {
        try
        {
            size_t parsed = 0;
            max_new_tokens = std::stoi(argv[first_arg + 1], &parsed);
            if (parsed != std::strlen(argv[first_arg + 1]) || max_new_tokens <= 0 ||
                max_new_tokens > MAX_SEQUENCE_LENGTH)
            {
                throw std::invalid_argument("max_new_tokens");
            }
        }
        catch (const std::exception &)
        {
            std::cerr << "max_new_tokens must be an integer from 1 to "
                      << MAX_SEQUENCE_LENGTH << '\n';
            return -1;
        }
    }
    if (loadLlamaModel(weights) != 0)
    {
        return -1;
    }

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

    if (!tokenizer.load(tokenizer_path,
                        &tokenizer_error))
    {
        std::cerr << tokenizer_error << '\n';
        cudaFree(cos_table_gpu);
        cudaFree(sin_table_gpu);
        cudaFree(weights.model_storage);
        return -1;
    }

    std::vector<std::string> prompts{prompt};
    for (int arg = first_arg + 2; arg < argc; ++arg)
    {
        prompts.push_back(argv[arg]);
    }
    std::vector<Request> requests(prompts.size());
    int max_prompt_length = 0;
    for (size_t index = 0; index < prompts.size(); ++index)
    {
        Request &request = requests[index];
        // Instruct weights expect a user message followed by an assistant header.
        request.prompt_tokens = tokenizer.encode_dialog_prompt({{"user", prompts[index]}});
        if (request.prompt_tokens.empty() || request.prompt_tokens.size() >= MAX_SEQUENCE_LENGTH)
        {
            std::cerr << "prompt " << index + 1 << " must leave room for a response\n";
            cudaFree(cos_table_gpu);
            cudaFree(sin_table_gpu);
            cudaFree(weights.model_storage);
            return -1;
        }
        const int prompt_length = static_cast<int>(request.prompt_tokens.size());
        request.token_budget = std::min(max_new_tokens, MAX_SEQUENCE_LENGTH - prompt_length);
        max_prompt_length = std::max(max_prompt_length, prompt_length);
    }

    InferenceBuffers buffers{};
    buffers.token_capacity = std::max(max_prompt_length, BATCH_SIZE);
    const int generation_status = generateRequests(
        requests, weights, buffers, tokenizer,
        cos_table_gpu, sin_table_gpu, continuous_batching);

    freeInferenceBuffers(buffers);
    cudaFree(cos_table_gpu);
    cudaFree(sin_table_gpu);
    cudaFree(weights.model_storage);

    return generation_status;
}
#endif
