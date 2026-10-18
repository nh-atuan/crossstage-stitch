// Verifies the shared headers compile with nvcc and cudaStream_t matches the real CUDA type.
#include <cuda_runtime.h>

#include <cstdio>

#include "common/geometry.hpp"

int main() {
    cudaStream_t stream = nullptr;
    stitch::ScaleInfo si;
    if (stitch::make_scale_info(512, 512, 0.5, &si) != stitch::Status::OK || si.work_w != 256) return 1;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    cudaStreamDestroy(stream);
    std::puts("test_common_cuda: passed");
    return 0;
}
