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
__global__ void pair_kernel(ImageView a,ImageView b,double* out) {
    __shared__ double scores[49],overlap[49];
    const int shift=threadIdx.x;
    if (shift<49) {
        const double dx=(shift%7-3)/8.,dy=(shift/7-3)/8.;
        double n=0,sa=0,sb=0,aa=0,bb=0,ab=0;
        for (int i=0;i<1024;++i) {
            const double x=(i%32)/31.,y=(i/32)/31.,u=x+dx,v=y+dy;
            if (u<0 || u>1 || v<0 || v>1) continue;
            const double av=sample(a,x*(a.width-1),y*(a.height-1));
            const double bv=sample(b,u*(b.width-1),v*(b.height-1));
            ++n;sa+=av;sb+=bv;aa+=av*av;bb+=bv*bv;ab+=av*bv;
        }
        scores[shift]=correlation(sa,sb,aa,bb,ab,n);overlap[shift]=n/1024.;
    }
    __syncthreads();
    if (!shift) {
        int best=0;double second=-2;
        for (int j=1;j<49;++j) if (scores[j]>scores[best]) best=j;
        for (int j=0;j<49;++j) if (j!=best) second=fmax(second,scores[j]);
        out[0]=scores[best];out[1]=scores[best]-second;
        out[2]=(best%7-3)/8.;out[3]=(best/7-3)/8.;out[4]=scores[24];out[5]=overlap[best];
    }
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
__global__ void patch_kernel(ImageView a,ImageView b,Matrix inverse,double* out) {
    __shared__ double ssim[16],edge[16],cosine[16],normalized[16];
    __shared__ int valid[16];
    const int patch=threadIdx.x;
    if (patch<16) {
        const double cx=(patch%4+.5)*a.width/4.,cy=(patch/4+.5)*a.height/4.;
        double n=0,sa=0,sb=0,aa=0,bb=0,ab=0,mae=0,edge_error=0,dot=0,ga2=0,gb2=0;
        for (int dy=-3;dy<=3;++dy) for (int dx=-3;dx<=3;++dx) {
            const double x=cx+dx,y=cy+dy;double bv,bx,by;
            if (x<0 || y<0 || x+1>a.width-1 || y+1>a.height-1 ||
                !warp(b,inverse,x,y,&bv) || !warp(b,inverse,x+1,y,&bx) || !warp(b,inverse,x,y+1,&by)) continue;
            const double av=sample(a,x,y),ax=sample(a,x+1,y)-av,ay=sample(a,x,y+1)-av;
            bx-=bv;by-=bv;
            ++n;sa+=av;sb+=bv;aa+=av*av;bb+=bv*bv;ab+=av*bv;mae+=fabs(av-bv);
            edge_error+=fabs(ax-bx)+fabs(ay-by);dot+=ax*bx+ay*by;ga2+=ax*ax+ay*ay;gb2+=bx*bx+by*by;
        }
        valid[patch]=n==49;
        const double den=fmax(n,1.),ma=sa/den,mb=sb/den,va=fmax(0.,aa/den-ma*ma),vb=fmax(0.,bb/den-mb*mb);
        ssim[patch]=((2*ma*mb+.0001)*(2*(ab/den-ma*mb)+.0009))/((ma*ma+mb*mb+.0001)*(va+vb+.0009));
        edge[patch]=edge_error/(2*den);cosine[patch]=dot/fmax(1e-12,sqrt(ga2*gb2));
        normalized[patch]=fmin(1e6,mae/den/fmax(1e-6,sqrt(va)+sqrt(vb)));
    }
    __syncthreads();
    if (!patch) {
        int n=0;double sum=0,low=1,eg=0,co=0,errors[16];
        for (int j=0;j<16;++j) if (valid[j]) {
            errors[n++]=normalized[j];sum+=ssim[j];low=fmin(low,ssim[j]);eg+=edge[j];co+=cosine[j];
        }
        for (int j=1;j<n;++j) {double v=errors[j];int k=j;while(k && errors[k-1]>v){errors[k]=errors[k-1];--k;}errors[k]=v;}
        out[8]=n?sum/n:-1;out[9]=n?low:-1;out[10]=n?eg/n:1;out[11]=n?co/n:0;
        out[12]=n/16.;out[13]=n?errors[int(ceil(.9*n))-1]:1e6;
    }
}
__global__ void statistics_kernel(Keypoints a,Keypoints b,Matches m,const std::uint8_t* mask,
    HomographyResult h,ScaleInfo sa,ScaleInfo sb,const s4::Top2* forward,double* out) {
    // One deterministic collector: at most 4000 matches; no floating atomics.
    int residual_hist[257]={},distance_hist[257]={},counts[16]={};double sums[16]={};
    int inliers=0;double distance_sum=0;
    for (int i=0;i<h.num_matches;++i) {
        const int ia=m.idx_a[i],ib=m.idx_b[i];
        if (ia<0 || ib<0 || ia>=a.capacity || ib>=b.capacity) continue;
        const double distance=fmin(256.,fmax(0.,double(m.distance[i])));
        distance_sum+=distance;++distance_hist[int(distance)];
        if (!mask[i]) continue;
        const double x=b.x[ib]/sb.sx,y=b.y[ib]/sb.sy,z=h.H[6]*x+h.H[7]*y+h.H[8];
        if (!isfinite(z) || fabs(z)<1e-12) continue;
        const double u=(h.H[0]*x+h.H[1]*y+h.H[2])/z,v=(h.H[3]*x+h.H[4]*y+h.H[5])/z;
        const double e=hypot(u-a.x[ia]/sa.sx,v-a.y[ia]/sa.sy);
        if (!isfinite(e)) continue;
        ++residual_hist[min(256,int(e*8))];++inliers;
        const int tile=max(0,min(3,int(a.x[ia]*4/sa.work_w)))+4*max(0,min(3,int(a.y[ia]*4/sa.work_h)));
        ++counts[tile];sums[tile]+=fmin(1e6,e);
    }
    int seen=0,p50=256,p90=256,dp90=256;
    for (int j=0;j<257;++j) {seen+=residual_hist[j];if(seen>=ceil(.5*inliers) && p50==256)p50=j;if(seen>=ceil(.9*inliers)){p90=j;break;}}
    seen=0;for(int j=0;j<257;++j){seen+=distance_hist[j];if(seen>=ceil(.9*h.num_matches)){dp90=j;break;}}
    double maximum=0,sum=0,sq=0;int regions=0;
    for (int j=0;j<16;++j) if(counts[j]){double v=sums[j]/counts[j];maximum=fmax(maximum,v);sum+=v;sq+=v*v;++regions;}
    out[0]=inliers?p50/8.:32;out[1]=inliers?p90/8.:32;out[2]=maximum;
    out[3]=regions?sqrt(fmax(0.,sq/regions-(sum/regions)*(sum/regions))):0;
    out[4]=distance_sum/fmax(1.,double(h.num_matches))/256.;out[5]=dp90/256.;
    double ratio_sum=0;int ratios=0,ambiguous=0;
    if (forward) for(int j=0;j<a.capacity;++j) {
        const auto p=forward[j];
        if(p.first_index<0 || p.second_index<0 || !isfinite(p.second_distance))continue;
        const double ratio=p.second_distance>0?p.first_distance/p.second_distance:1.;
        ratio_sum+=ratio;++ratios;ambiguous+=ratio>=.8;
    }
    out[6]=ratios?ratio_sum/ratios:1;out[7]=ratios?double(ambiguous)/ratios:1;
}
bool valid_image(const ImageView& im) {
    return im.data && im.width>=2 && im.height>=2 && im.pitch_bytes>0 &&
        (im.format==PixelFormat::GRAY8 || im.format==PixelFormat::RGB8) &&
        im.pitch_bytes>=static_cast<std::size_t>(im.width)*(im.format==PixelFormat::RGB8?3:1);
}
}  // namespace
Status pair_relations(const ImageView& a,const ImageView& b,double* output,cudaStream_t stream) {
    if (!valid_image(a) || !valid_image(b) || !output) return Status::INVALID_ARGUMENT;
    pair_kernel<<<1,64,0,stream>>>(a,b,output);
    return cudaGetLastError()==cudaSuccess?Status::OK:Status::CUDA_ERROR;
}
Status alignment_relations(const ImageView& a,const ImageView& b,const Keypoints& ka,const Keypoints& kb,
    const Matches& matches,const std::uint8_t* mask,const HomographyResult& h,const ScaleInfo& sa,
    const ScaleInfo& sb,const s4::Top2* forward,double* output,cudaStream_t stream) {
    Matrix inverse{};
    if (!valid_image(a) || !valid_image(b) || !output || !mask || !matches.idx_a || !matches.idx_b ||
        !matches.distance || !ka.x || !ka.y || !kb.x || !kb.y || h.status!=Status::OK ||
        h.num_matches<4 || h.num_matches>matches.capacity || ka.capacity<1 || ka.capacity>4000 ||
        kb.capacity<1 || kb.capacity>4000 || !std::isfinite(sa.sx) || !std::isfinite(sa.sy) ||
        !std::isfinite(sb.sx) || !std::isfinite(sb.sy) || sa.sx<=0 || sa.sy<=0 || sb.sx<=0 || sb.sy<=0 ||
        sa.work_w<=0 || sa.work_h<=0 || !invert(h.H,&inverse)) return Status::INVALID_ARGUMENT;
    statistics_kernel<<<1,1,0,stream>>>(ka,kb,matches,mask,h,sa,sb,forward,output);
    if (cudaGetLastError()!=cudaSuccess) return Status::CUDA_ERROR;
    patch_kernel<<<1,32,0,stream>>>(a,b,inverse,output);
    return cudaGetLastError()==cudaSuccess?Status::OK:Status::CUDA_ERROR;
}
}  // namespace stitch::controller
