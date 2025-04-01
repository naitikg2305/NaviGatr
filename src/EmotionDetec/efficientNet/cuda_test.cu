#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
    } else {
        std::cout << "CUDA-capable device count: " << deviceCount << std::endl;

        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
            std::cout << "  Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        }
    }

    return 0;
}
