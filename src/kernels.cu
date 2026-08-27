#include "kernels.cuh"

namespace
{

__global__ void warmUpKernel()
{
}

} // namespace

void warmUpGpu()
{
    warmUpKernel<<<1, 1>>>();
}
