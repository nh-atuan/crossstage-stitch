#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "check.hpp"
#include "common/geometry.hpp"
#include "controller/features.hpp"
using namespace stitch;
int main() {
    int devices=0;if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    std::vector<std::uint8_t> rgb(16*16*3), gray(16*16), mask(16,1);
    for (int y=0;y<16;++y) for (int x=0;x<16;++x) {
        gray[y*16+x]=x*8+y*4;
        for (int c=0;c<3;++c) rgb[(y*16+x)*3+c]=gray[y*16+x];
    }
    ImageView a{nullptr,16,16,48,PixelFormat::RGB8}, g{nullptr,16,16,16,PixelFormat::GRAY8};
    CHECK(cudaMalloc(&a.data,rgb.size())==cudaSuccess);CHECK(cudaMalloc(&g.data,gray.size())==cudaSuccess);
    CHECK(cudaMemcpy(a.data,rgb.data(),rgb.size(),cudaMemcpyHostToDevice)==cudaSuccess);
    CHECK(cudaMemcpy(g.data,gray.data(),gray.size(),cudaMemcpyHostToDevice)==cudaSuccess);
    double *device=nullptr,*host=nullptr;cudaMalloc(&device,controller::gate_feature_count*sizeof(double));
    cudaMallocHost(&host,controller::gate_feature_count*sizeof(double));
    CHECK(controller::extract_scene(g,g,device,host,nullptr)==Status::OK);CHECK(cudaDeviceSynchronize()==cudaSuccess);
    CHECK(std::abs(host[0]-6./255)<1e-12);CHECK(host[0]==host[1]);
    CHECK(host[2]<host[5] && host[3]>0);
    Keypoints kp{};kp.capacity=16;cudaMalloc(&kp.x,16*sizeof(float));cudaMalloc(&kp.y,16*sizeof(float));
    Matches matches{};matches.capacity=16;cudaMalloc(&matches.idx_a,16*sizeof(int));cudaMalloc(&matches.idx_b,16*sizeof(int));
    std::vector<float> x(16),y(16);std::vector<int> indices(16);
    for (int i=0;i<16;++i) {x[i]=(i%4)*4+2;y[i]=(i/4)*4+2;indices[i]=i;}
    cudaMemcpy(kp.x,x.data(),64,cudaMemcpyHostToDevice);cudaMemcpy(kp.y,y.data(),64,cudaMemcpyHostToDevice);
    cudaMemcpy(matches.idx_a,indices.data(),64,cudaMemcpyHostToDevice);cudaMemcpy(matches.idx_b,indices.data(),64,cudaMemcpyHostToDevice);
    std::uint8_t* dm=nullptr;cudaMalloc(&dm,16);cudaMemcpy(dm,mask.data(),16,cudaMemcpyHostToDevice);
    HomographyResult h{};h.status=Status::OK;h.num_matches=h.num_inliers=16;h.inlier_ratio=1;
    h.H[0]=h.H[4]=h.H[8]=1;ScaleInfo scale{};make_scale_info(16,16,1,&scale);
    CHECK(controller::extract_gate(a,a,kp,kp,matches,dm,h,scale,scale,device,host,nullptr)==Status::OK);
    CHECK(cudaDeviceSynchronize()==cudaSuccess);CHECK(host[10]==1 && host[11]<1e-12 && host[13]>.999999);
    CHECK(host[18]==1 && host[19]==1 && std::abs(host[22]-1)<1e-12);
    h.H[8]=0;CHECK(controller::extract_gate(a,a,kp,kp,matches,dm,h,scale,scale,device,host,nullptr)==Status::INVALID_ARGUMENT);
    cudaFree(a.data);cudaFree(g.data);cudaFree(device);cudaFreeHost(host);cudaFree(kp.x);cudaFree(kp.y);
    cudaFree(matches.idx_a);cudaFree(matches.idx_b);cudaFree(dm);
    std::puts("CUDA scene/photometric/inlier features: analytic fixture passed");
}
