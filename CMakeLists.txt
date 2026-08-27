cmake_minimum_required(VERSION 3.24)

project(samir_vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(samir-vllm
    src/main.cpp
    src/kernels.cu
)

target_include_directories(samir-vllm PRIVATE include)
target_link_libraries(samir-vllm PRIVATE CUDA::cudart)

set_target_properties(samir-vllm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
