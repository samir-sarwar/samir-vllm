#include <cuda_runtime.h> // Cuda run time, allows us to do all our CUDA work.
#include <cublas_v2.h>    // Allows us to do big matrix multiplictions
#include <cstdlib>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp> // grabs json header for functions to parse and create json objects in cpp
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <vector>

using json = nlohmann::json; // local alias 

#include <iostream>

#include "kernels.cuh" // we can run the functions from kernel.cuh
// they were implemented in kernels.cu 

int checkGPUStatus(){
    int device_count = 0;
    cudaGetDeviceCount(&device_count); // cuda func to get device count
    if (device_count == 0){
        std::cerr << "no cuda devices found \n"; //cout but for errors 
        return 1;

    }
    cudaDeviceProp prop; // creates struct to hold info on gpu
    cudaGetDeviceProperties(&prop,0); // fill struct
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << (free_mem / (1024*1024*1024)) << "GB, total memory: " << total_mem / (1024*1024*1024) << "GB\n";
    return 0;
}
constexpr int N_LAYERS = 16; // 16 transformer layers 

struct LLamaWeights {
    void* model_storage = nullptr;  // generic pointer to start of Gpu alloc

    __nv_bfloat16* embed_tokens = nullptr; // ptr to embedding table, of bf16 vectors
    __nv_bfloat16* norm = nullptr; // final rmsnorm weight vector 

    __nv_bfloat16* input_layernorm[N_LAYERS]{}; // array of 16 GPU pointers init to nullptr: input rmsnorm
    __nv_bfloat16* post_attn_layernorms[N_LAYERS]{}; // rmsnorm weights after attention

    __nv_bfloat16* w_q[N_LAYERS]{}; // query: what token is looking for
    __nv_bfloat16* w_k[N_LAYERS]{}; // key: what each token can be matched on
    __nv_bfloat16* w_v[N_LAYERS]{}; // value: the information each token contributes
    __nv_bfloat16* w_o[N_LAYERS]{}; // output: combines attention result 
    // small feed forward neural network inside each transformer layer 
    __nv_bfloat16* mlp_gate_proj[N_LAYERS]{};
    __nv_bfloat16* mlp_up_proj[N_LAYERS]{};
    __nv_bfloat16* mlp_down_proj[N_LAYERS]{};
};

int loadLlamaModel(LLamaWeights& weights){
    std::string path = "models/llama-3.2-1b-instruct/model.safetensors"; // set local path
    std::ifstream safetensors_file(path, std::ios::binary); // open file as binary file
    if(!safetensors_file){
        std::cerr << "could not open safetensors file";
        return -1;
    }
    uint64_t headersize = 0;
    // so read expects a char buffer to store the extracted data
    // so we cast it to tell cpp to treat this variables memory as an 8-byte 
    // destination buffer, we know its size is 8 bytes, but we can also use sizeof
    safetensors_file.read(reinterpret_cast<char*>(&headersize),8);
    if(!safetensors_file){
        std::cerr << "could not read safetensors file";
        return -1;
    }
    std::cout << headersize << '\n';
    // this means the next headersize amount of bytes of the file contain the json header which
    // describes all tensors. What we want to do now create pointers to each of the tensors, using the offsets from shape value for each tensor 
    // then allocate headersize amount of memory on the GPU 
    // oh shi a tensor is also called a layer makes sense
    std::string header(headersize, '\0' ); // string of headersize init with null char
    // header.data points to strings actual character buffer
    safetensors_file.read(header.data(),headersize);
    // convert this bigass string into a json object
    json header_json = json::parse(header);

    

    // the way we are going to store tensor -> starting byte offset is going to be using
    // a hashmap

    std::unordered_map<std::string, uint64_t> offsets;
    // at some point we need know the largest tensor end offset in the raw tensor data
    // bytes as that will tell us how much memory we need to actually allocate as
    // we end up copying the ENTIRE raw tensordata from model file to our gpu. 
    
    uint64_t max_offset = 0;
    // read only, access key and value from pair from JSON header 
    for(const auto& [name,tensor_info] : header_json.items()){
        if(name == "__metadata__"){
            continue;
        }
        // you can read how they structured the json object from when we printed the entire header
        // will help us with parsing
        const auto& data_offsets = tensor_info.at("data_offsets");
        // we want to convert it to a 64 bit integer as its still a JSON value 
        uint64_t start_offset = data_offsets.at(0).get<uint64_t>();
        uint64_t end_offset = data_offsets.at(1).get<uint64_t>();
        // add to hashmap, effeciently constructs it in place inside map memory 
        // avoiding temporary memory.
        offsets.emplace(name, start_offset);
        max_offset = std::max(max_offset,end_offset);

        // test
        std::cout << name << " starts at: " << start_offset 
        << " ends at: " << end_offset << "\n";
        std::cout << "we have to allocate this many bytes: " << (end_offset / 8) << '\n';
    }
    // now we have to read the raw tensor data copy it to memory so we can then copy it to the gpu
    // remember our file ptr which points to start of where we read from is now at the start
    // of our tensor data as it started from 0, read specifically size of header
    // then read specifically size of json header. Meaning its now at raw data. 
    std::vector<char> model_weights_cpu(max_offset);
    // model_weights_cpu.data() gets the addr of 1st bytes like &array[0]
    safetensors_file.read(model_weights_cpu.data(),
                        static_cast<std::streamsize>(max_offset));

    // just give cuda a nullptr and the size of buffer and we can return the starting address
    // of the buffer with size given
    void* model_weights_gpu = nullptr;
    cudaMalloc(&model_weights_gpu,max_offset);
    // now we can copy from cpu memory to gpu memory. 

    cudaMemcpy(model_weights_gpu, model_weights_cpu.data(),model_weights_cpu.size(),cudaMemcpyHostToDevice);
    weights.model_storage = model_weights_gpu;
    // create a hashmap of names of tensors and pointers to them 

    /* 
    std::unordered_map<std::string, __nv_bfloat16*> tensor_pointers;
    // we cast as model weights gpu is a generic pointer to start of whole gpu buffer 
    char* base = static_cast<char*>(model_weights_gpu);
    // go through hashmap of offsets, so we can make a cpu lookup table to see where things are in the GPU
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
    // must wire pointers to 2 rmsnorm vectors, four attention matricies and 3 mlp matricies
    // so each of the 16 layers has 9 weight tensors
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
        return -1
    }
    


    return 0;
}
