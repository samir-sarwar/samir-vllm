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

// Local alias.
using json = nlohmann::json;

#include <iostream>

// We can run the functions from kernels.cuh.
// They were implemented in kernels.cu.
#include "kernels.cuh"

int checkGPUStatus(){
    int device_count = 0;
    // CUDA func to get device count.
    cudaGetDeviceCount(&device_count);
    if (device_count == 0){
        // cout but for errors.
        std::cerr << "no cuda devices found \n";
        return 1;

    }
    // Creates struct to hold info on gpu.
    cudaDeviceProp prop;
    // Fill struct.
    cudaGetDeviceProperties(&prop,0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << (free_mem / (1024*1024*1024)) << "GB, total memory: " << total_mem / (1024*1024*1024) << "GB\n";
    return 0;
}
// 16 transformer layers.
constexpr int N_LAYERS = 16;

struct LLamaWeights {
    // Generic pointer to start of gpu alloc.
    void* model_storage = nullptr;

    // Ptr to embedding table, of bf16 vectors.
    __nv_bfloat16* embed_tokens = nullptr;
    // Final rmsnorm weight vector.
    __nv_bfloat16* norm = nullptr;

    // Array of 16 GPU pointers init to nullptr: input rmsnorm.
    __nv_bfloat16* input_layernorm[N_LAYERS]{};
    // Rmsnorm weights after attention.
    __nv_bfloat16* post_attn_layernorms[N_LAYERS]{};

    // Query: what token is looking for.
    __nv_bfloat16* w_q[N_LAYERS]{};
    // Key: what each token can be matched on.
    __nv_bfloat16* w_k[N_LAYERS]{};
    // Value: the information each token contributes.
    __nv_bfloat16* w_v[N_LAYERS]{};
    // Output: combines attention result.
    __nv_bfloat16* w_o[N_LAYERS]{};

    // Small feed forward neural network inside each transformer layer.
    __nv_bfloat16* mlp_gate_proj[N_LAYERS]{};
    __nv_bfloat16* mlp_up_proj[N_LAYERS]{};
    __nv_bfloat16* mlp_down_proj[N_LAYERS]{};
};

int loadLlamaModel(LLamaWeights& weights){
    // Set local path.
    std::string path = "models/llama-3.2-1b-instruct/model.safetensors";
    // Open file as binary file.
    std::ifstream safetensors_file(path, std::ios::binary);
    if(!safetensors_file){
        std::cerr << "could not open safetensors file";
        return -1;
    }
    uint64_t headersize = 0;
    // Read expects a char buffer to store the extracted data.
    // We cast to tell cpp to treat this variable's memory as an 8-byte
    // destination buffer. We know its size is 8 bytes, but we can also use sizeof.
    safetensors_file.read(reinterpret_cast<char*>(&headersize),8);
    if(!safetensors_file){
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
    std::string header(headersize, '\0' );
    // header.data points to string's actual character buffer.
    safetensors_file.read(header.data(),headersize);
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
    for(const auto& [name,tensor_info] : header_json.items()){
        if(name == "__metadata__"){
            continue;
        }
        // You can read how they structured the json object from when we printed
        // the entire header. It will help us with parsing.
        const auto& data_offsets = tensor_info.at("data_offsets");
        // We want to convert it to a 64 bit integer as it is still a JSON value.
        uint64_t start_offset = data_offsets.at(0).get<uint64_t>();
        uint64_t end_offset = data_offsets.at(1).get<uint64_t>();
        // Add to hashmap; effeciently constructs it in place inside map memory,
        // avoiding temporary memory.
        offsets.emplace(name, start_offset);
        max_offset = std::max(max_offset,end_offset);

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
    void* model_weights_gpu = nullptr;
    cudaMalloc(&model_weights_gpu,max_offset);
    // Now we can copy from cpu memory to gpu memory.

    cudaMemcpy(model_weights_gpu, model_weights_cpu.data(),model_weights_cpu.size(),cudaMemcpyHostToDevice);
    weights.model_storage = model_weights_gpu;
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

    char* base = static_cast<char*>(weights.model_storage);

    weights.embed_tokens = reinterpret_cast<__nv_bfloat16*>(
        base + offsets.at("model.embed_tokens.weight"));

    weights.norm = reinterpret_cast<__nv_bfloat16*>(base + offsets.at("model.norm.weight"));
    // Must wire pointers to 2 rmsnorm vectors, four attention matrices, and 3
    // mlp matrices, so each of the 16 layers has 9 weight tensors.
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        std::string prefix = "model.layers." + std::to_string(layer);
        weights.w_q[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".self_attn.q_proj.weight"));
        weights.w_k[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".self_attn.k_proj.weight"));
        weights.w_v[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".self_attn.v_proj.weight"));
        weights.w_o[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".self_attn.o_proj.weight"));

        weights.input_layernorm[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".input_layernorm.weight"));

        weights.post_attn_layernorms[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".post_attention_layernorm.weight"));

        weights.mlp_gate_proj[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".mlp.gate_proj.weight"));

        weights.mlp_up_proj[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".mlp.up_proj.weight"));

        weights.mlp_down_proj[layer] = reinterpret_cast<__nv_bfloat16*>(
            base + offsets.at(prefix + ".mlp.down_proj.weight"));

    }
    return 0;

}
int main()
{
    // checkGPUStatus();
    LLamaWeights weights{};
    if(loadLlamaModel(weights) !=0 ){
        return -1;
    }
    


    return 0;
}
