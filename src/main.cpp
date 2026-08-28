#include <cuda_runtime.h> // Cuda run time, allows us to do all our CUDA work.
#include <cublas_v2.h>    // Allows us to do big matrix multiplictions
#include <cstdlib>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

using json = nlohmann::json;

#include <iostream>

#include "kernels.cuh"
namespace
{

    void checkCuda(cudaError_t result, const char *action)
    {
        if (result == cudaSuccess)
        {
            return;
        }

        std::cerr << "CUDA error while " << action << ": "
                  << cudaGetErrorString(result) << '\n';
        std::exit(EXIT_FAILURE);
    }

    void printGpuInfo()
    {
        int deviceCount = 0;
        checkCuda(cudaGetDeviceCount(&deviceCount), "checking for CUDA devices");

        if (deviceCount == 0)
        {
            std::cerr << "No CUDA-capable GPU was found.\n";
            std::exit(EXIT_FAILURE);
        }

        cudaDeviceProp device{};
        checkCuda(cudaGetDeviceProperties(&device, 0), "reading GPU properties");

        std::cout << "GPU: " << device.name << '\n'
                  << "Compute capability: " << device.major << '.' << device.minor << '\n'
                  << "Global memory: " << (device.totalGlobalMem / (1024 * 1024)) << " MiB\n";
    }

} // namespace

int main()
{
    printGpuInfo();

    warmUpGpu();
    checkCuda(cudaGetLastError(), "launching the warm-up kernel");
    checkCuda(cudaDeviceSynchronize(), "waiting for the warm-up kernel");

    std::cout << "CUDA setup is ready.\n";
    return EXIT_SUCCESS;
}
