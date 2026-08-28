#include <cuda_runtime.h> // Cuda run time, allows us to do all our CUDA work.
#include <cublas_v2.h>    // Allows us to do big matrix multiplictions
#include <cstdlib>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp> // grabs json header for functions to parse and create json objects in cpp
#include <fstream>
#define B_TO_GB 1024*1024*1024

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

void loadModelHeader(){
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
    // describes all tensors. What we want to do now is allocate this amount of memory on the GPU 
    // then create pointers to each of the tensors, using the offsets from shape value for each tensor
    

}
int main()
{
    checkGPUStatus();
    loadModelHeader();


    return 0;
}
