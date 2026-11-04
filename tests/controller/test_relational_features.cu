#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "check.hpp"
#include "common/geometry.hpp"
#include "controller/relational_features.hpp"
using namespace stitch;
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    std::vector<std::uint8_t> image(64*64),mask(16,1);
    for(int y=0;y<64;++y)for(int x=0;x<64;++x)image[y*64+x]=(x*17+y*31+x*y*3)%256;
    ImageView a{nullptr,64,64,64,PixelFormat::GRAY8};
    CHECK(cudaMalloc(&a.data,image.size())==cudaSuccess);
    CHECK(cudaMemcpy(a.data,image.data(),image.size(),cudaMemcpyHostToDevice)==cudaSuccess);
    double* device=nullptr;CHECK(cudaMalloc(&device,14*sizeof(double))==cudaSuccess);double host[14];
    CHECK(controller::pair_relations(a,a,device,nullptr)==Status::OK);
    CHECK(cudaMemcpy(host,device,6*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
    CHECK(std::abs(host[0]-1)<1e-12 && host[1]>.01 && host[2]==0 && host[3]==0);
    CHECK(std::abs(host[4]-1)<1e-12 && host[5]==1);
    auto invalid=a;invalid.pitch_bytes=-64;
    CHECK(controller::pair_relations(invalid,a,device,nullptr)==Status::INVALID_ARGUMENT);
    Keypoints kp{};kp.capacity=16;Matches m{};m.capacity=16;
    CHECK(cudaMalloc(&kp.x,64)==cudaSuccess);CHECK(cudaMalloc(&kp.y,64)==cudaSuccess);
    CHECK(cudaMalloc(&m.idx_a,64)==cudaSuccess);CHECK(cudaMalloc(&m.idx_b,64)==cudaSuccess);
    CHECK(cudaMalloc(&m.distance,64)==cudaSuccess);
    std::vector<float> x(16),y(16),distance(16,8);std::vector<int> indices(16);
    std::vector<s4::Top2> forward(16);
    for(int i=0;i<16;++i){x[i]=(i%4)*16+8;y[i]=(i/4)*16+8;indices[i]=i;forward[i]={8,16,i,(i+1)%16};}
    cudaMemcpy(kp.x,x.data(),64,cudaMemcpyHostToDevice);cudaMemcpy(kp.y,y.data(),64,cudaMemcpyHostToDevice);
    cudaMemcpy(m.idx_a,indices.data(),64,cudaMemcpyHostToDevice);cudaMemcpy(m.idx_b,indices.data(),64,cudaMemcpyHostToDevice);
    cudaMemcpy(m.distance,distance.data(),64,cudaMemcpyHostToDevice);
    std::uint8_t* dm=nullptr;s4::Top2* df=nullptr;
    cudaMalloc(&dm,16);cudaMemcpy(dm,mask.data(),16,cudaMemcpyHostToDevice);
    cudaMalloc(&df,forward.size()*sizeof(s4::Top2));cudaMemcpy(df,forward.data(),forward.size()*sizeof(s4::Top2),cudaMemcpyHostToDevice);
    HomographyResult h{};h.status=Status::OK;h.num_matches=h.num_inliers=16;h.inlier_ratio=1;
    h.H[0]=h.H[4]=h.H[8]=1;ScaleInfo scale{};make_scale_info(64,64,1,&scale);
    CHECK(controller::alignment_relations(a,a,kp,kp,m,dm,h,scale,scale,df,device,nullptr)==Status::OK);
    CHECK(cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost)==cudaSuccess);
    for(double v:host)CHECK(std::isfinite(v));
    CHECK(host[0]==0 && host[1]==0 && host[2]<1e-12 && host[3]<1e-12);
    CHECK(host[4]==8./256 && host[5]==8./256 && host[6]==.5 && host[7]==0);
    CHECK(std::abs(host[8]-1)<1e-12 && std::abs(host[9]-1)<1e-12);
    CHECK(host[10]<1e-12 && host[11]>.999999 && host[12]==1 && host[13]<1e-12);
    Keypoints scaled=kp;cudaMalloc(&scaled.x,64);cudaMalloc(&scaled.y,64);
    std::vector<float> sx=x,sy=y;
    for(int i=0;i<16;++i){sx[i]=.5f*(sx[i]+.5f)-.5f;sy[i]=.5f*(sy[i]+.5f)-.5f;}
    cudaMemcpy(scaled.x,sx.data(),64,cudaMemcpyHostToDevice);cudaMemcpy(scaled.y,sy.data(),64,cudaMemcpyHostToDevice);
    ScaleInfo half{};make_scale_info(64,64,.5,&half);
    CHECK(controller::alignment_relations(a,a,scaled,kp,m,dm,h,half,scale,df,device,nullptr)==Status::OK);
    CHECK(cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost)==cudaSuccess);
    CHECK(host[0]==0 && host[1]==0 && host[2]<1e-12);
    cudaFree(scaled.x);cudaFree(scaled.y);
    h.H[8]=0;CHECK(controller::alignment_relations(a,a,kp,kp,m,dm,h,scale,scale,df,device,nullptr)==Status::INVALID_ARGUMENT);
    cudaFree(a.data);cudaFree(device);cudaFree(kp.x);cudaFree(kp.y);cudaFree(m.idx_a);cudaFree(m.idx_b);
    cudaFree(m.distance);cudaFree(dm);cudaFree(df);
    std::puts("Pair correlation, residual, match ambiguity and patch structure analytic fixtures passed");
}
