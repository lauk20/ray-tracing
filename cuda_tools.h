#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include <curand.h>
#include <curand_kernel.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const * const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA ERROR " << static_cast<unsigned int>(result) << " at " << file
        << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ int getThreadID() {
    int block_id = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    int thread_id = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x + block_id * (blockDim.x * blockDim.y * blockDim.z);
    return thread_id;
}

__global__ void initRandState(curandState * rand_state, int num_states) {
    int id = getThreadID();
    if (id >= num_states) return;
    curand_init(12345, id, 0, &rand_state[id]);
}

#endif