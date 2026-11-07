// Preserve CUDA error codes so resource errors never depend on message parsing.
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
namespace stitch::harness {
struct CudaError : std::runtime_error {
    cudaError_t code;
    explicit CudaError(cudaError_t value) : std::runtime_error(cudaGetErrorString(value)), code(value) {}
};
}  // namespace stitch::harness
