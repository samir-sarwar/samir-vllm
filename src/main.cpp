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
#include <unordered_map>
#include <algorithm>
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
    uint64_t headersize = 0;
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
        std::cout << name << " starts at: " << start_offset
                  << " ends at: " << end_offset << "\n";
        std::cout << "we have to allocate this many bytes: " << (end_offset / 8) << '\n';
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
        munmap(file_bytes, file_size);
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
            munmap(file_bytes, file_size);
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
        munmap(file_bytes, file_size);
        return -1;
    }
    if (cudaMemcpy(modelweights_gpu, tensor_data, max_offset, cudaMemcpyHostToDevice) != 0)
    {
        std::cerr << "gpu mem copy failed";
        cudaFree(modelweights_gpu);
        munmap(file_bytes, file_size);
        return -1;
    }
    munmap(file_bytes, file_size);

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

std::vector<int> tokenizePrompt(const std::string &prompt, const std::string &tokenizer_path)
{
    Tokenizer tokenizer;
    std::string error;
    tokenizer.load(tokenizer_path, &error);

    std::vector<int> token_ids = tokenizer.encode(prompt);

    return token_ids;
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

    Tokenizer tokenizer;
    std::string tokenizer_error;

    if (!tokenizer.load("models/llama-3.2-1b-instruct/tokenizer.model",
                        &tokenizer_error))
    {
        std::cerr << tokenizer_error << '\n';
        return -1;
    }

    return 0;
}
