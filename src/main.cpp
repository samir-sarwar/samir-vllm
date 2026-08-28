#include <cuda_runtime.h> // Cuda run time, allows us to do all our CUDA work.
#include <cublas_v2.h>    // Allows us to do big matrix multiplictions
#include <cstdlib>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp> // grabs json header for functions to parse and create json objects in cpp

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
int main()
{
    checkGPUStatus();

    return 0;
}
