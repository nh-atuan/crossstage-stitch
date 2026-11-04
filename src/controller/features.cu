// Controller observations only: no changes to S1, S5 or the canonical QoR metric.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "controller/features.hpp"
#include "controller/relational_features.hpp"

namespace stitch::controller {
namespace {
constexpr int threads = 128;
__device__ double gray(const ImageView& im, int x, int y) {
    const auto* p = im.data + y*im.pitch_bytes + x*(im.format == PixelFormat::RGB8 ? 3 : 1);
    return im.format == PixelFormat::GRAY8 ? p[0]/255. :
        (9798.*p[0]+19235.*p[1]+3735.*p[2])/(32768.*255.);
}
__device__ double interpolate(const ImageView& im, double x, double y) {
    const int ix = static_cast<int>(x), iy = static_cast<int>(y);
    const int jx = min(ix+1, im.width-1), jy = min(iy+1, im.height-1);
    const double dx = x-ix, dy = y-iy;
    return (1-dy)*((1-dx)*gray(im,ix,iy)+dx*gray(im,jx,iy)) +
        dy*((1-dx)*gray(im,ix,jy)+dx*gray(im,jx,jy));
}
__device__ double reduce(double value, double* scratch) {
    const int t = threadIdx.x;
    scratch[t] = value; __syncthreads();
    for (int step = threads/2; step; step /= 2) {
        if (t < step) scratch[t] += scratch[t+step];
        __syncthreads();
    }
    const double total = scratch[0]; __syncthreads();
    return total;
}
__global__ void scene_kernel(ImageView a, ImageView b, double* out) {
    __shared__ double scratch[threads];
    const int region = blockIdx.x, image = region < 2 ? region : (region-2)/4;
    const auto im = image ? b : a;
    const int tile = region < 2 ? -1 : (region-2)%4;
    const int x0 = tile < 0 ? 0 : (tile%2)*im.width/2;
    const int y0 = tile < 0 ? 0 : (tile/2)*im.height/2;
    const int x1 = tile < 0 ? im.width : ((tile%2)+1)*im.width/2;
    const int y1 = tile < 0 ? im.height : ((tile/2)+1)*im.height/2;
    const int w = x1-x0, h = y1-y0;
    double sum=0, sq=0, gradient=0;
    for (int i=threadIdx.x; i<w*h; i+=threads) {
        const int x=x0+i%w, y=y0+i/w;
        const double v=gray(im,x,y);
        sum+=v; sq+=v*v;
        if (x+1<x1) gradient+=fabs(v-gray(im,x+1,y));
        if (y+1<y1) gradient+=fabs(v-gray(im,x,y+1));
    }
    sum=reduce(sum,scratch); sq=reduce(sq,scratch); gradient=reduce(gradient,scratch);
    if (!threadIdx.x) {
        const double mean=sum/(w*h), edges=h*(w-1)+w*(h-1);
        if (tile < 0) out[region]=edges ? gradient/edges : 0.;
        else {
            const int offset=2+image*12+tile*3;
            out[offset]=mean; out[offset+1]=sqrt(fmax(0.,sq/(w*h)-mean*mean));
            out[offset+2]=edges ? gradient/edges : 0.;
        }
    }
}
struct Matrix { double v[9]; };
__global__ void photo_kernel(ImageView a, ImageView b, Matrix inverse, double* out) {
    __shared__ double scratch[threads];
    double n=0, sum_a=0, sum_b=0, sq_a=0, sq_b=0, cross=0, abs_error=0, sq_error=0;
    for (int i=threadIdx.x; i<4096; i+=threads) {
        const double x=(i%64)*(a.width-1.)/63., y=(i/64)*(a.height-1.)/63.;
        const auto& h=inverse.v;
        const double z=h[6]*x+h[7]*y+h[8];
        if (fabs(z)<1e-12) continue;
        const double u=(h[0]*x+h[1]*y+h[2])/z, v=(h[3]*x+h[4]*y+h[5])/z;
        if (u<0 || v<0 || u>b.width-1 || v>b.height-1 || !isfinite(u) || !isfinite(v)) continue;
        const double av=interpolate(a,fmax(0.,x),fmax(0.,y)), bv=interpolate(b,u,v), d=av-bv;
        n+=1;sum_a+=av;sum_b+=bv;sq_a+=av*av;sq_b+=bv*bv;cross+=av*bv;
        abs_error+=fabs(d);sq_error+=d*d;
    }
    n=reduce(n,scratch);sum_a=reduce(sum_a,scratch);sum_b=reduce(sum_b,scratch);
    sq_a=reduce(sq_a,scratch);sq_b=reduce(sq_b,scratch);cross=reduce(cross,scratch);
    abs_error=reduce(abs_error,scratch);sq_error=reduce(sq_error,scratch);
    if (!threadIdx.x) {
        const double den=fmax(1.,n), ma=sum_a/den, mb=sum_b/den;
        const double va=fmax(0.,sq_a/den-ma*ma), vb=fmax(0.,sq_b/den-mb*mb);
        const double sd=sqrt(va)+sqrt(vb);
        out[10]=n/4096.;out[11]=abs_error/den;out[12]=sqrt(sq_error/den);
        out[13]=fmin(1.,fmax(-1.,(cross/den-ma*mb)/fmax(1e-12,sqrt(va*vb))));
        out[14]=fabs(ma-mb);out[15]=sqrt(va);out[16]=sqrt(vb);
        out[17]=fmin(1e6,abs_error/den/fmax(1e-6,sd));
    }
}
__global__ void inlier_kernel(Keypoints a, Keypoints b, Matches m, const std::uint8_t* mask,
                             int count, int wa, int ha, int wb, int hb, double* out) {
    __shared__ int hist[32];
    if (threadIdx.x<32) hist[threadIdx.x]=0;
    __syncthreads();
    for (int i=threadIdx.x; i<count; i+=threads) if (mask[i]) {
        const int ia=m.idx_a[i], ib=m.idx_b[i];
        if (ia<0 || ib<0 || ia>=a.capacity || ib>=b.capacity) continue;
        const int xa=max(0,min(3,int(4*a.x[ia]/wa))), ya=max(0,min(3,int(4*a.y[ia]/ha)));
        const int xb=max(0,min(3,int(4*b.x[ib]/wb))), yb=max(0,min(3,int(4*b.y[ib]/hb)));
        atomicAdd(hist+ya*4+xa,1);atomicAdd(hist+16+yb*4+xb,1);
    }
    __syncthreads();
    if (!threadIdx.x) for (int image=0;image<2;++image) {
        int total=0, occupied=0, largest=0;
        for (int j=0;j<16;++j) {int c=hist[image*16+j];total+=c;occupied+=c>0;largest=max(largest,c);}
        double entropy=0;
        if (total) for (int j=0;j<16;++j) if (hist[image*16+j]) {
            const double p=double(hist[image*16+j])/total;entropy-=p*log(p)/log(16.);
        }
        out[18+image]=occupied/16.;out[20+image]=total ? double(largest)/total : 0.;
        out[22+image]=entropy;
    }
}
bool inverse(const double* h, Matrix* out) {
    const double d=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
    if (!std::isfinite(d) || std::abs(d)<1e-18) return false;
    const double adj[9]={h[4]*h[8]-h[5]*h[7],h[2]*h[7]-h[1]*h[8],h[1]*h[5]-h[2]*h[4],
        h[5]*h[6]-h[3]*h[8],h[0]*h[8]-h[2]*h[6],h[2]*h[3]-h[0]*h[5],
        h[3]*h[7]-h[4]*h[6],h[1]*h[6]-h[0]*h[7],h[0]*h[4]-h[1]*h[3]};
    for (int i=0;i<9;++i) {out->v[i]=adj[i]/d;if (!std::isfinite(out->v[i])) return false;}
    return true;
}
}  // namespace
Status extract_scene(const ImageView& a, const ImageView& b, double* device, double* pinned, cudaStream_t stream, bool relations) {
    if (!a.data || !b.data || !device || !pinned || a.width<2 || b.width<2 || a.height<2 || b.height<2 ||
        a.format!=PixelFormat::GRAY8 || b.format!=PixelFormat::GRAY8) return Status::INVALID_ARGUMENT;
    scene_kernel<<<10,threads,0,stream>>>(a,b,device);
    if (cudaGetLastError()!=cudaSuccess) return Status::CUDA_ERROR;
    if (cudaMemsetAsync(device+26,0,pair_relation_count*sizeof(double),stream)!=cudaSuccess) return Status::CUDA_ERROR;
    if (relations && pair_relations(a,b,device+26,device+gate_feature_count,stream)!=Status::OK) return Status::CUDA_ERROR;
    if (cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(pinned,device,scene_feature_count*sizeof(double),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)
        return Status::CUDA_ERROR;
    return Status::OK;
}
Status extract_gate(const ImageView& a, const ImageView& b, const Keypoints& ka, const Keypoints& kb,
                    const Matches& matches, const std::uint8_t* mask, const HomographyResult& h,
                    const ScaleInfo& sa, const ScaleInfo& sb, double* device, double* pinned, cudaStream_t stream,
                    const s4::Top2* forward, unsigned relation_mask) {
    Matrix inv{};
    if (!a.data || !b.data || !device || !pinned || !mask || h.status!=Status::OK ||
        h.num_matches<4 || h.num_matches>matches.capacity || !inverse(h.H,&inv)) return Status::INVALID_ARGUMENT;
    double base[10]={h.num_matches/4000.,h.num_inliers/4000.,h.inlier_ratio,h.inlier_coverage,h.reproj_residual,
        h.reproj_residual/std::max(sa.sx,sa.sy),0,0,0,0};
    double x[4],y[4],zmin=1e300,zmax=0.;
    for (int i=0;i<4;++i) {
        const double u=(i==1 || i==2) ? b.width-1. : 0., v=i>=2 ? b.height-1. : 0.;
        const double z=h.H[6]*u+h.H[7]*v+h.H[8];
        if (!std::isfinite(z) || std::abs(z)<1e-12) return Status::INVALID_ARGUMENT;
        x[i]=(h.H[0]*u+h.H[1]*v+h.H[2])/z;y[i]=(h.H[3]*u+h.H[4]*v+h.H[5])/z;
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) return Status::INVALID_ARGUMENT;
        zmin=std::min(zmin,std::abs(z));zmax=std::max(zmax,std::abs(z));
    }
    double area=0.;for (int i=0;i<4;++i) area+=x[i]*y[(i+1)%4]-y[i]*x[(i+1)%4];
    base[6]=std::min(1e6,std::abs(area)/2./(double(b.width)*b.height));
    base[7]=std::min(1e6,(*std::max_element(x,x+4)-*std::min_element(x,x+4))/a.width);
    base[8]=std::min(1e6,(*std::max_element(y,y+4)-*std::min_element(y,y+4))/a.height);
    base[9]=std::min(1e6,zmax/zmin);
    std::memcpy(pinned,base,sizeof(base));
    if (cudaMemcpyAsync(device,pinned,sizeof(base),cudaMemcpyHostToDevice,stream)!=cudaSuccess) return Status::CUDA_ERROR;
    photo_kernel<<<1,threads,0,stream>>>(a,b,inv,device);
    inlier_kernel<<<1,threads,0,stream>>>(ka,kb,matches,mask,h.num_matches,sa.work_w,sa.work_h,sb.work_w,sb.work_h,device);
    if (cudaGetLastError()!=cudaSuccess || cudaMemsetAsync(device+24,0,alignment_relation_count*sizeof(double),stream)!=cudaSuccess)
        return Status::CUDA_ERROR;
    if (relation_mask && alignment_relations(a,b,ka,kb,matches,mask,h,sa,sb,forward,device+24,device+gate_feature_count,stream,relation_mask)!=Status::OK)
        return Status::CUDA_ERROR;
    if (cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(pinned,device,gate_feature_count*sizeof(double),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)
        return Status::CUDA_ERROR;
    return Status::OK;
}
}  // namespace stitch::controller
