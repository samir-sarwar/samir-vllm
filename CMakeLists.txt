cmake_minimum_required(VERSION 3.24)

project(samir_vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(tokenizer src/tokenizer.cpp)
target_include_directories(tokenizer PUBLIC include)

add_executable(samir-vllm
    src/main.cpp
    src/kernels.cu
)

target_include_directories(samir-vllm PRIVATE include)
target_link_libraries(samir-vllm PRIVATE tokenizer CUDA::cudart)

add_executable(tokenizer-test tests/tokenizer_test.cpp)
target_link_libraries(tokenizer-test PRIVATE tokenizer)

set_target_properties(samir-vllm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
