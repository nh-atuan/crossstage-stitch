#include <cuda_runtime.h>
#include <cmath>
#include "controller/relational_features.hpp"

namespace stitch::controller {
namespace {
__device__ double gray(const ImageView& im,int x,int y) {
    const auto* p=im.data+y*im.pitch_bytes+x*(im.format==PixelFormat::RGB8?3:1);
    return im.format==PixelFormat::GRAY8?p[0]/255.:(9798.*p[0]+19235.*p[1]+3735.*p[2])/(32768.*255.);
}
__device__ double sample(const ImageView& im,double x,double y) {
    const int ix=int(x),iy=int(y),jx=min(ix+1,im.width-1),jy=min(iy+1,im.height-1);
    const double dx=x-ix,dy=y-iy;
    return (1-dy)*((1-dx)*gray(im,ix,iy)+dx*gray(im,jx,iy))+
        dy*((1-dx)*gray(im,ix,jy)+dx*gray(im,jx,jy));
}
__device__ double correlation(double a,double b,double aa,double bb,double ab,double n) {
    const double va=fmax(0.,aa-a*a/n),vb=fmax(0.,bb-b*b/n);
    return fmin(1.,fmax(-1.,(ab-a*b/n)/fmax(1e-12,sqrt(va*vb))));
}
template<int threads>
__device__ double reduce(double value,double* scratch) {
    const int t=threadIdx.x;scratch[t]=value;__syncthreads();
    for(int step=threads/2;step;step/=2){if(t<step)scratch[t]+=scratch[t+step];__syncthreads();}
    const double result=scratch[0];__syncthreads();return result;
}
__global__ void pair_kernel(ImageView a,ImageView b,double* workspace) {
    __shared__ double scratch[128];
    const int shift=blockIdx.x;
        const double dx=(shift%7-3)/8.,dy=(shift/7-3)/8.;
        double n=0,sa=0,sb=0,aa=0,bb=0,ab=0;
        for (int i=threadIdx.x;i<1024;i+=128) {
            const double x=(i%32)/31.,y=(i/32)/31.,u=x+dx,v=y+dy;
            if (u<0 || u>1 || v<0 || v>1) continue;
            const double av=sample(a,x*(a.width-1),y*(a.height-1));
            const double bv=sample(b,u*(b.width-1),v*(b.height-1));
            ++n;sa+=av;sb+=bv;aa+=av*av;bb+=bv*bv;ab+=av*bv;
        }
        n=reduce<128>(n,scratch);sa=reduce<128>(sa,scratch);sb=reduce<128>(sb,scratch);
        aa=reduce<128>(aa,scratch);bb=reduce<128>(bb,scratch);ab=reduce<128>(ab,scratch);
        if(!threadIdx.x){workspace[shift]=correlation(sa,sb,aa,bb,ab,n);workspace[49+shift]=n/1024.;}
}
__global__ void pair_finish(const double* scores,double* out) {
        int best=0;double second=-2;
        for (int j=1;j<49;++j) if (scores[j]>scores[best]) best=j;
        for (int j=0;j<49;++j) if (j!=best) second=fmax(second,scores[j]);
        out[0]=scores[best];out[1]=scores[best]-second;
        out[2]=(best%7-3)/8.;out[3]=(best/7-3)/8.;out[4]=scores[24];out[5]=scores[49+best];
}
struct Matrix {double v[9];};
bool invert(const double* h,Matrix* output) {
    const double d=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
    if (!std::isfinite(d) || std::abs(d)<1e-18) return false;
    const double adj[9]={h[4]*h[8]-h[5]*h[7],h[2]*h[7]-h[1]*h[8],h[1]*h[5]-h[2]*h[4],
        h[5]*h[6]-h[3]*h[8],h[0]*h[8]-h[2]*h[6],h[2]*h[3]-h[0]*h[5],
        h[3]*h[7]-h[4]*h[6],h[1]*h[6]-h[0]*h[7],h[0]*h[4]-h[1]*h[3]};
    for (int i=0;i<9;++i) {output->v[i]=adj[i]/d;if(!std::isfinite(output->v[i])) return false;}
    return true;
}
__device__ bool warp(const ImageView& b,Matrix h,double x,double y,double* value) {
    const double z=h.v[6]*x+h.v[7]*y+h.v[8];
    if (!isfinite(z) || fabs(z)<1e-12) return false;
    const double u=(h.v[0]*x+h.v[1]*y+h.v[2])/z,v=(h.v[3]*x+h.v[4]*y+h.v[5])/z;
    if (!isfinite(u) || !isfinite(v) || u<0 || v<0 || u>b.width-1 || v>b.height-1) return false;
    *value=sample(b,u,v);return true;
}
__global__ void patch_kernel(ImageView a,ImageView b,Matrix inverse,double* workspace) {
    __shared__ double scratch[64];
    const int patch=blockIdx.x;
        const double cx=(patch%4+.5)*a.width/4.,cy=(patch/4+.5)*a.height/4.;
        double n=0,sa=0,sb=0,aa=0,bb=0,ab=0,mae=0,edge_error=0,dot=0,ga2=0,gb2=0;
        for (int i=threadIdx.x;i<49;i+=64) {
            const int dx=i%7-3,dy=i/7-3;
            const double x=cx+dx,y=cy+dy;double bv,bx,by;
            if (x<0 || y<0 || x+1>a.width-1 || y+1>a.height-1 ||
                !warp(b,inverse,x,y,&bv) || !warp(b,inverse,x+1,y,&bx) || !warp(b,inverse,x,y+1,&by)) continue;
            const double av=sample(a,x,y),ax=sample(a,x+1,y)-av,ay=sample(a,x,y+1)-av;
            bx-=bv;by-=bv;
            ++n;sa+=av;sb+=bv;aa+=av*av;bb+=bv*bv;ab+=av*bv;mae+=fabs(av-bv);
            edge_error+=fabs(ax-bx)+fabs(ay-by);dot+=ax*bx+ay*by;ga2+=ax*ax+ay*ay;gb2+=bx*bx+by*by;
        }
        n=reduce<64>(n,scratch);sa=reduce<64>(sa,scratch);sb=reduce<64>(sb,scratch);
        aa=reduce<64>(aa,scratch);bb=reduce<64>(bb,scratch);ab=reduce<64>(ab,scratch);
        mae=reduce<64>(mae,scratch);edge_error=reduce<64>(edge_error,scratch);
        dot=reduce<64>(dot,scratch);ga2=reduce<64>(ga2,scratch);gb2=reduce<64>(gb2,scratch);
        if(!threadIdx.x) {
        const double den=fmax(n,1.),ma=sa/den,mb=sb/den,va=fmax(0.,aa/den-ma*ma),vb=fmax(0.,bb/den-mb*mb);
        workspace[patch*5]=((2*ma*mb+.0001)*(2*(ab/den-ma*mb)+.0009))/((ma*ma+mb*mb+.0001)*(va+vb+.0009));
        workspace[patch*5+1]=edge_error/(2*den);workspace[patch*5+2]=dot/fmax(1e-12,sqrt(ga2*gb2));
        workspace[patch*5+3]=fmin(1e6,mae/den/fmax(1e-6,sqrt(va)+sqrt(vb)));
        workspace[patch*5+4]=n==49;
        }
    }
__global__ void patch_finish(const double* workspace,double* out) {
        int n=0;double sum=0,low=1,eg=0,co=0,errors[16];
        for (int j=0;j<16;++j) if (workspace[j*5+4]) {
            errors[n++]=workspace[j*5+3];sum+=workspace[j*5];low=fmin(low,workspace[j*5]);eg+=workspace[j*5+1];co+=workspace[j*5+2];
        }
        for (int j=1;j<n;++j) {double v=errors[j];int k=j;while(k && errors[k-1]>v){errors[k]=errors[k-1];--k;}errors[k]=v;}
        out[8]=n?sum/n:-1;out[9]=n?low:-1;out[10]=n?eg/n:1;out[11]=n?co/n:0;
        out[12]=n/16.;out[13]=n?errors[int(ceil(.9*n))-1]:1e6;
}
__global__ void statistics_kernel(Keypoints a,Keypoints b,Matches m,const std::uint8_t* mask,
    HomographyResult h,ScaleInfo sa,ScaleInfo sb,const s4::Top2* forward,double* out,unsigned groups) {
    // Integer histograms and fixed-order reductions avoid floating atomics.
    __shared__ int residual_hist[257],distance_hist[257],counts[16];
    __shared__ double regional[16][128],scratch[128];
    const int t=threadIdx.x;
    for(int j=t;j<257;j+=128){residual_hist[j]=0;distance_hist[j]=0;}
    if(t<16)counts[t]=0;
    for(int j=0;j<16;++j)regional[j][t]=0;
    __syncthreads();
    int inliers=0;double distance_sum=0;
    for (int i=t;i<h.num_matches;i+=128) {
        const int ia=m.idx_a[i],ib=m.idx_b[i];
        if (ia<0 || ib<0 || ia>=a.capacity || ib>=b.capacity) continue;
        if (groups&matching_relations) {
            const double distance=fmin(256.,fmax(0.,double(m.distance[i])));
            distance_sum+=distance;atomicAdd(distance_hist+int(distance),1);
        }
        if (!(groups&geometry_relations) || !mask[i]) continue;
        const double x=(b.x[ib]+.5)/sb.sx-.5,y=(b.y[ib]+.5)/sb.sy-.5,z=h.H[6]*x+h.H[7]*y+h.H[8];
        if (!isfinite(z) || fabs(z)<1e-12) continue;
        const double u=(h.H[0]*x+h.H[1]*y+h.H[2])/z,v=(h.H[3]*x+h.H[4]*y+h.H[5])/z;
        const double e=hypot(u-((a.x[ia]+.5)/sa.sx-.5),v-((a.y[ia]+.5)/sa.sy-.5));
        if (!isfinite(e)) continue;
        atomicAdd(residual_hist+int(fmin(32.,e)*8),1);++inliers;
        const int tile=max(0,min(3,int(a.x[ia]*4/sa.work_w)))+4*max(0,min(3,int(a.y[ia]*4/sa.work_h)));
        atomicAdd(counts+tile,1);regional[tile][t]+=fmin(1e6,e);
    }
    distance_sum=reduce<128>(distance_sum,scratch);inliers=int(reduce<128>(inliers,scratch));
    double maximum=0,sum=0,sq=0;int regions=0;
    if(groups&geometry_relations)for(int j=0;j<16;++j) {
        const double total=reduce<128>(regional[j][t],scratch);
        if(!t && counts[j]){const double v=total/counts[j];maximum=fmax(maximum,v);sum+=v;sq+=v*v;++regions;}
    }
    double ratio_sum=0;int ratios=0,ambiguous=0;
    if (forward && (groups&matching_relations)) for(int j=t;j<a.capacity;j+=128) {
        const auto p=forward[j];
        if(p.first_index<0 || p.second_index<0 || !isfinite(p.second_distance))continue;
        const double ratio=p.second_distance>0?p.first_distance/p.second_distance:1.;
        ratio_sum+=ratio;++ratios;ambiguous+=ratio>=.8;
    }
    ratio_sum=reduce<128>(ratio_sum,scratch);ratios=int(reduce<128>(ratios,scratch));ambiguous=int(reduce<128>(ambiguous,scratch));
    if(!t) {
    int seen=0,p50=256,p90=256,dp90=256;
    for (int j=0;j<257;++j) {seen+=residual_hist[j];if(seen>=ceil(.5*inliers) && p50==256)p50=j;if(seen>=ceil(.9*inliers)){p90=j;break;}}
    seen=0;for(int j=0;j<257;++j){seen+=distance_hist[j];if(seen>=ceil(.9*h.num_matches)){dp90=j;break;}}
    out[0]=inliers?p50/8.:32;out[1]=inliers?p90/8.:32;out[2]=maximum;
    out[3]=regions?sqrt(fmax(0.,sq/regions-(sum/regions)*(sum/regions))):0;
    out[4]=distance_sum/fmax(1.,double(h.num_matches))/256.;out[5]=dp90/256.;
    out[6]=ratios?ratio_sum/ratios:1;out[7]=ratios?double(ambiguous)/ratios:1;
    }
}
bool valid_image(const ImageView& im) {
    return im.data && im.width>=2 && im.height>=2 && im.pitch_bytes>0 &&
        (im.format==PixelFormat::GRAY8 || im.format==PixelFormat::RGB8) &&
        im.pitch_bytes>=static_cast<std::size_t>(im.width)*(im.format==PixelFormat::RGB8?3:1);
}
}  // namespace
Status pair_relations(const ImageView& a,const ImageView& b,double* output,double* workspace,cudaStream_t stream) {
    if (!valid_image(a) || !valid_image(b) || !output || !workspace) return Status::INVALID_ARGUMENT;
    pair_kernel<<<49,128,0,stream>>>(a,b,workspace);
    if(cudaGetLastError()!=cudaSuccess)return Status::CUDA_ERROR;
    pair_finish<<<1,1,0,stream>>>(workspace,output);
    return cudaGetLastError()==cudaSuccess?Status::OK:Status::CUDA_ERROR;
}
Status alignment_relations(const ImageView& a,const ImageView& b,const Keypoints& ka,const Keypoints& kb,
    const Matches& matches,const std::uint8_t* mask,const HomographyResult& h,const ScaleInfo& sa,
    const ScaleInfo& sb,const s4::Top2* forward,double* output,double* workspace,cudaStream_t stream,unsigned groups) {
    Matrix inverse{};
    if (!valid_image(a) || !valid_image(b) || !output || !mask || !matches.idx_a || !matches.idx_b ||
        !matches.distance || !ka.x || !ka.y || !kb.x || !kb.y || h.status!=Status::OK ||
        h.num_matches<4 || h.num_matches>matches.capacity || ka.capacity<1 || ka.capacity>4000 ||
        kb.capacity<1 || kb.capacity>4000 || !std::isfinite(sa.sx) || !std::isfinite(sa.sy) ||
        !std::isfinite(sb.sx) || !std::isfinite(sb.sy) || sa.sx<=0 || sa.sy<=0 || sb.sx<=0 || sb.sy<=0 ||
        sa.work_w<=0 || sa.work_h<=0 || !invert(h.H,&inverse)) return Status::INVALID_ARGUMENT;
    if (groups>7 || !workspace) return Status::INVALID_ARGUMENT;
    if (cudaMemsetAsync(output,0,alignment_relation_count*sizeof(double),stream)!=cudaSuccess) return Status::CUDA_ERROR;
    if (groups&(matching_relations|geometry_relations))
        statistics_kernel<<<1,128,0,stream>>>(ka,kb,matches,mask,h,sa,sb,forward,output,groups);
    if (cudaGetLastError()!=cudaSuccess) return Status::CUDA_ERROR;
    if (groups&photometric_relations) {
        patch_kernel<<<16,64,0,stream>>>(a,b,inverse,workspace);
        if(cudaGetLastError()!=cudaSuccess)return Status::CUDA_ERROR;
        patch_finish<<<1,1,0,stream>>>(workspace,output);
    }
    return cudaGetLastError()==cudaSuccess?Status::OK:Status::CUDA_ERROR;
}
}  // namespace stitch::controller
