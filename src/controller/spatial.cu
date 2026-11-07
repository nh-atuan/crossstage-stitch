#include <cuda_runtime.h>
#include "controller/spatial.hpp"
namespace stitch::controller {
namespace {
__global__ void spatial_kernel(ImageView a,ImageView b,HomographyResult h,double* out,unsigned groups) {
    if(!spatial_observations(a,b,h.H,out,groups))for(int i=0;i<16;++i)out[i]=nan("");
}
}
Status extract_spatial(const ImageView& a,const ImageView& b,const HomographyResult& h,double* device,double* pinned,cudaStream_t stream,unsigned groups) {
    if(groups>3 || !device || !pinned || h.status!=Status::OK)return Status::INVALID_ARGUMENT;
    spatial_kernel<<<1,1,0,stream>>>(a,b,h,device,groups);
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(pinned,device,16*sizeof(double),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)return Status::CUDA_ERROR;
    return Status::OK;
}
}
