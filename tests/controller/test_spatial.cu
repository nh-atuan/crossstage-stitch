#include <cuda_runtime.h>
#include "controller/spatial.hpp"
#include <cmath>
#include <iostream>
#include <vector>
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)return 77;
    std::vector<std::uint8_t> pixels(64*64*3);for(int i=0;i<int(pixels.size());++i)pixels[i]=(i*17)%251;
    stitch::ImageView host{pixels.data(),64,64,192,stitch::PixelFormat::RGB8},device=host;
    double* output=nullptr;double* pinned=nullptr;
    if(cudaMalloc(reinterpret_cast<void**>(&device.data),pixels.size())!=cudaSuccess || cudaMalloc(reinterpret_cast<void**>(&output),16*sizeof(double))!=cudaSuccess ||
        cudaMallocHost(reinterpret_cast<void**>(&pinned),16*sizeof(double))!=cudaSuccess)return 1;
    if(cudaMemcpy(device.data,pixels.data(),pixels.size(),cudaMemcpyHostToDevice)!=cudaSuccess)return 1;
    stitch::HomographyResult h{};h.status=stitch::Status::OK;h.H[0]=h.H[4]=h.H[8]=1.;
    for(int scenario=0;scenario<3;++scenario)for(unsigned groups=1;groups<=3;++groups) {
        h.H[2]=scenario==0?0:scenario==1?2.25:10000.;double expected[16];
        if(!stitch::controller::spatial_observations(host,host,h.H,expected,groups))return 1;
        if(stitch::controller::extract_spatial(device,device,h,output,pinned,nullptr,groups)!=stitch::Status::OK || cudaDeviceSynchronize()!=cudaSuccess)return 1;
        for(int j=0;j<16;++j)if(!std::isfinite(pinned[j]) || std::abs(expected[j]-pinned[j])>1e-8)return 1;
    }
    cudaFree(device.data);cudaFree(output);cudaFreeHost(pinned);
    std::cout<<"CUDA spatial parity passed (synthetic images, no benchmark claim)\n";
}
