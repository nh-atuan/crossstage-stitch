// Candidate-only stratified overlap evidence; shared CPU/CUDA numerical schema.
#pragma once
#include <cmath>
#include "common/types.hpp"
#ifdef __CUDACC__
#define STITCH_SPATIAL_HD __host__ __device__
#else
#define STITCH_SPATIAL_HD
#endif
namespace stitch::controller {
namespace spatial_detail {
STITCH_SPATIAL_HD inline bool finite(double value) {
#ifdef __CUDA_ARCH__
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}
STITCH_SPATIAL_HD inline double gray(const ImageView& im,int x,int y) {
    const auto* p=im.data+y*im.pitch_bytes+x*(im.format==PixelFormat::RGB8?3:1);
    return im.format==PixelFormat::GRAY8?p[0]/255.:(9798.*p[0]+19235.*p[1]+3735.*p[2])/(32768.*255.);
}
STITCH_SPATIAL_HD inline double sample(const ImageView& im,double x,double y) {
    const int ix=int(x),iy=int(y),jx=ix+1<im.width?ix+1:ix,jy=iy+1<im.height?iy+1:iy;
    const double dx=x-ix,dy=y-iy;
    return (1-dy)*((1-dx)*gray(im,ix,iy)+dx*gray(im,jx,iy))+dy*((1-dx)*gray(im,ix,jy)+dx*gray(im,jx,jy));
}
STITCH_SPATIAL_HD inline void sort(double* x,int n) {
    for(int j=1;j<n;++j){double v=x[j];int k=j;while(k && x[k-1]>v){x[k]=x[k-1];--k;}x[k]=v;}
}
}
// [SSIM min,q25,median,q75; MAE q90; valid fraction; 4 quadrant coverage;
//  Jacobian |det| min,max,spread; invalid projection fraction; mean SSIM; SSIM std].
// 4x4 stratified tile centers, 7x7 patches; only fully observed patches count.
STITCH_SPATIAL_HD inline bool spatial_observations(const ImageView& a,const ImageView& b,const double* h,double* out,unsigned groups=3) {
    for(int i=0;i<16;++i)out[i]=0.;
    if(groups>3 || !a.data || !b.data || a.width<8 || a.height<8 || b.width<8 || b.height<8 ||
        (a.format!=PixelFormat::RGB8 && a.format!=PixelFormat::GRAY8) || (b.format!=PixelFormat::RGB8 && b.format!=PixelFormat::GRAY8) ||
        a.pitch_bytes<a.width*(a.format==PixelFormat::RGB8?3:1) || b.pitch_bytes<b.width*(b.format==PixelFormat::RGB8?3:1)) return false;
    for(int i=0;i<9;++i)if(!spatial_detail::finite(h[i]))return false;
    const double d=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
    if(!spatial_detail::finite(d) || fabs(d)<1e-18)return false;
    double inv[9]={h[4]*h[8]-h[5]*h[7],h[2]*h[7]-h[1]*h[8],h[1]*h[5]-h[2]*h[4],
        h[5]*h[6]-h[3]*h[8],h[0]*h[8]-h[2]*h[6],h[2]*h[3]-h[0]*h[5],
        h[3]*h[7]-h[4]*h[6],h[1]*h[6]-h[0]*h[7],h[0]*h[4]-h[1]*h[3]};
    for(double& v:inv)v/=d;
    double scores[16]{},errors[16]{},sum=0,sq=0,jmin=1e12,jmax=0;int n=0,invalid=0;
    for(int tile=0;tile<16;++tile) {
        const double cx=(tile%4+.5)*(a.width-1)/4.,cy=(tile/4+.5)*(a.height-1)/4.;
        double sa=0,sb=0,aa=0,bb=0,ab=0,mae=0;int valid=0;
        for(int j=0;j<49 && (groups&1);++j) {
            const double x=cx+j%7-3,y=cy+j/7-3,z=inv[6]*x+inv[7]*y+inv[8];
            if(x<0 || y<0 || x>a.width-1 || y>a.height-1 || !spatial_detail::finite(z) || fabs(z)<1e-12) continue;
            const double u=(inv[0]*x+inv[1]*y+inv[2])/z,v=(inv[3]*x+inv[4]*y+inv[5])/z;
            if(!spatial_detail::finite(u) || !spatial_detail::finite(v) || u<0 || v<0 || u>b.width-1 || v>b.height-1)continue;
            const double av=spatial_detail::sample(a,x,y),bv=spatial_detail::sample(b,u,v);
            sa+=av;sb+=bv;aa+=av*av;bb+=bv*bv;ab+=av*bv;mae+=fabs(av-bv);++valid;
        }
        const double z=inv[6]*cx+inv[7]*cy+inv[8];
        const double signed_jac=1./(d*z*z*z),jac=fabs(signed_jac);
        if(!spatial_detail::finite(jac) || signed_jac<=0 || jac>1e6 || fabs(z)<1e-12)++invalid;
        else {if(jac<jmin)jmin=jac;if(jac>jmax)jmax=jac;}
        const int quadrant=(tile%4>=2)+2*(tile/4>=2);out[6+quadrant]+=valid/(49.*4.);
        if(valid!=49)continue;
        const double ma=sa/49.,mb=sb/49.,va=fmax(0.,aa/49.-ma*ma),vb=fmax(0.,bb/49.-mb*mb);
        double score=((2*ma*mb+.0001)*(2*(ab/49.-ma*mb)+.0009))/((ma*ma+mb*mb+.0001)*(va+vb+.0009));
        score=fmax(-1.,fmin(1.,score));scores[n]=score;errors[n]=mae/49.;++n;sum+=score;sq+=score*score;
    }
    spatial_detail::sort(scores,n);spatial_detail::sort(errors,n);
    out[0]=n?scores[0]:-1;out[1]=n?scores[(n-1)/4]:-1;out[2]=n?scores[(n-1)/2]:-1;out[3]=n?scores[(n-1)*3/4]:-1;
    out[4]=n?errors[int(ceil(.9*n))-1]:1;out[5]=n/16.;out[10]=jmin==1e12?0:fmin(jmin,1e6);out[11]=fmin(jmax,1e6);
    out[12]=fmin(1e6,out[11]/fmax(1e-12,out[10]));out[13]=invalid/16.;out[14]=n?sum/n:-1;
    out[15]=n?sqrt(fmax(0.,sq/n-(sum/n)*(sum/n))):0;
    if(!(groups&1)){for(int i=0;i<10;++i)out[i]=0;out[14]=out[15]=0;}
    if(!(groups&2)){for(int i=10;i<14;++i)out[i]=0;}
    return true;
}
Status extract_spatial(const ImageView&,const ImageView&,const HomographyResult&,double* device,double* pinned,cudaStream_t,unsigned groups=3);
} // namespace stitch::controller
#undef STITCH_SPATIAL_HD
