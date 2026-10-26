// S1: Preprocessing kernel implementation.
// Fused integer 15-bit RGB-to-GRAY and Area Resizing box filter.
// Processes image A and image B in a single batched kernel launch (blockIdx.z).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "s1/s1.hpp"

namespace stitch::s1 {
namespace {

struct S1KernelParams {
    const uint8_t* src;
    int src_pitch;
    int orig_w;
    int orig_h;
    uint8_t* dst;
    int dst_pitch;
    int work_w;
    int work_h;
    float inv_sx;
    float inv_sy;
    float scale_area;
    int mode; // 0: 1x1 direct, 1: 2x2 fast path (S=0.5), 2: 4x4 fast path (S=0.25), 3: general box filter
};

__device__ __forceinline__ int rgb_to_gray(int r, int g, int b) {
    // 15-bit integer fixed-point coefficients matching OpenCV 5.0.x bit-exact
    return (9798 * r + 19235 * g + 3735 * b + 16384) >> 15;
}

__global__ void s1_preprocess_kernel(S1KernelParams p0, S1KernelParams p1) {
    const S1KernelParams& p = (blockIdx.z == 0) ? p0 : p1;
    const int xd = blockIdx.x * blockDim.x + threadIdx.x;
    const int yd = blockIdx.y * blockDim.y + threadIdx.y;

    if (xd >= p.work_w || yd >= p.work_h) return;

    if (p.mode == 0) {
        // Direct 1x1 RGB to GRAY (S = 1.0)
        const uint8_t* src_px = p.src + yd * p.src_pitch + 3 * xd;
        const int y = rgb_to_gray(src_px[0], src_px[1], src_px[2]);
        p.dst[yd * p.dst_pitch + xd] = static_cast<uint8_t>(y);
    } else if (p.mode == 1) {
        // 2x2 integer fast path (S = 0.5)
        const uint8_t* r0 = p.src + (2 * yd) * p.src_pitch + 3 * (2 * xd);
        const uint8_t* r1 = p.src + (2 * yd + 1) * p.src_pitch + 3 * (2 * xd);
        const int y00 = rgb_to_gray(r0[0], r0[1], r0[2]);
        const int y01 = rgb_to_gray(r0[3], r0[4], r0[5]);
        const int y10 = rgb_to_gray(r1[0], r1[1], r1[2]);
        const int y11 = rgb_to_gray(r1[3], r1[4], r1[5]);
        p.dst[yd * p.dst_pitch + xd] = static_cast<uint8_t>((y00 + y01 + y10 + y11 + 2) >> 2);
    } else if (p.mode == 2) {
        // 4x4 integer fast path (S = 0.25)
        int sum = 0;
        #pragma unroll
        for (int dy = 0; dy < 4; ++dy) {
            const uint8_t* r = p.src + (4 * yd + dy) * p.src_pitch + 3 * (4 * xd);
            #pragma unroll
            for (int dx = 0; dx < 4; ++dx) {
                sum += rgb_to_gray(r[3 * dx], r[3 * dx + 1], r[3 * dx + 2]);
            }
        }
        // Round to nearest even integer (matching OpenCV generic area downsample)
        p.dst[yd * p.dst_pitch + xd] = static_cast<uint8_t>(__float2int_rn(static_cast<float>(sum) * 0.0625f));
    } else {
        // General area box filter (e.g. S = 0.75)
        const float fsy1 = static_cast<float>(yd) * p.inv_sy;
        const float fsy2 = static_cast<float>(yd + 1) * p.inv_sy;
        const float fsx1 = static_cast<float>(xd) * p.inv_sx;
        const float fsx2 = static_cast<float>(xd + 1) * p.inv_sx;

        const int y_start = static_cast<int>(floorf(fsy1));
        const int y_end = static_cast<int>(ceilf(fsy2));
        const int x_start = static_cast<int>(floorf(fsx1));
        const int x_end = static_cast<int>(ceilf(fsx2));

        float total = 0.0f;
        for (int ys = y_start; ys < y_end && ys < p.orig_h; ++ys) {
            const float y_lo = fmaxf(static_cast<float>(ys), fsy1);
            const float y_hi = fminf(static_cast<float>(ys + 1), fsy2);
            const float wy = fmaxf(0.0f, y_hi - y_lo);
            if (wy <= 0.0f) continue;

            const uint8_t* r_row = p.src + ys * p.src_pitch;
            for (int xs = x_start; xs < x_end && xs < p.orig_w; ++xs) {
                const float x_lo = fmaxf(static_cast<float>(xs), fsx1);
                const float x_hi = fminf(static_cast<float>(xs + 1), fsx2);
                const float wx = fmaxf(0.0f, x_hi - x_lo);
                if (wx <= 0.0f) continue;

                const uint8_t* rgb = r_row + 3 * xs;
                const int y_gray = rgb_to_gray(rgb[0], rgb[1], rgb[2]);
                total += static_cast<float>(y_gray) * (wx * wy);
            }
        }
        p.dst[yd * p.dst_pitch + xd] = static_cast<uint8_t>(__float2int_rn(total * p.scale_area));
    }
}

int determine_mode(int orig_w, int orig_h, int work_w, int work_h) {
    if (orig_w == work_w && orig_h == work_h) return 0;
    if (orig_w == 2 * work_w && orig_h == 2 * work_h) return 1;
    if (orig_w == 4 * work_w && orig_h == 4 * work_h) return 2;
    return 3;
}

Status fill_params(const ImageView& in, const ImageView& out, const ScaleInfo& si, S1KernelParams* p) {
    if (in.data == nullptr || out.data == nullptr ||
        in.format != PixelFormat::RGB8 || out.format != PixelFormat::GRAY8 ||
        in.width <= 0 || in.height <= 0 ||
        out.width != si.work_w || out.height != si.work_h ||
        in.pitch_bytes < in.width * 3 || out.pitch_bytes < out.width) {
        return Status::INVALID_ARGUMENT;
    }

    p->src = in.data;
    p->src_pitch = in.pitch_bytes;
    p->orig_w = in.width;
    p->orig_h = in.height;
    p->dst = out.data;
    p->dst_pitch = out.pitch_bytes;
    p->work_w = out.width;
    p->work_h = out.height;
    p->inv_sx = static_cast<float>(in.width) / static_cast<float>(out.width);
    p->inv_sy = static_cast<float>(in.height) / static_cast<float>(out.height);
    p->scale_area = 1.0f / (p->inv_sx * p->inv_sy);
    p->mode = determine_mode(in.width, in.height, out.width, out.height);
    return Status::OK;
}

}  // namespace

Status s1_preprocess_single(const ImageView& in, double scale,
                            ImageView* out, ScaleInfo* scale_info,
                            cudaStream_t stream) {
    if (out == nullptr || scale_info == nullptr || !(scale > 0.0) || scale > 1.0) {
        return Status::INVALID_ARGUMENT;
    }

    Status s = make_scale_info(in.width, in.height, scale, scale_info);
    if (s != Status::OK) return s;

    S1KernelParams p0{};
    s = fill_params(in, *out, *scale_info, &p0);
    if (s != Status::OK) return s;

    const dim3 block(16, 16);
    const dim3 grid((p0.work_w + block.x - 1) / block.x,
                    (p0.work_h + block.y - 1) / block.y,
                    1);

    s1_preprocess_kernel<<<grid, block, 0, stream>>>(p0, p0);
    if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
    return Status::OK;
}

Status s1_preprocess(const ImageView& img_a, const ImageView& img_b,
                     double scale,
                     ImageView* out_a, ImageView* out_b,
                     ScaleInfo* scale_a, ScaleInfo* scale_b,
                     cudaStream_t stream) {
    if (out_a == nullptr || out_b == nullptr ||
        scale_a == nullptr || scale_b == nullptr ||
        !(scale > 0.0) || scale > 1.0) {
        return Status::INVALID_ARGUMENT;
    }

    Status sa = make_scale_info(img_a.width, img_a.height, scale, scale_a);
    if (sa != Status::OK) return sa;

    Status sb = make_scale_info(img_b.width, img_b.height, scale, scale_b);
    if (sb != Status::OK) return sb;

    S1KernelParams pa{}, pb{};
    Status s = fill_params(img_a, *out_a, *scale_a, &pa);
    if (s != Status::OK) return s;

    s = fill_params(img_b, *out_b, *scale_b, &pb);
    if (s != Status::OK) return s;

    const int max_w = std::max(pa.work_w, pb.work_w);
    const int max_h = std::max(pa.work_h, pb.work_h);

    const dim3 block(16, 16);
    const dim3 grid((max_w + block.x - 1) / block.x,
                    (max_h + block.y - 1) / block.y,
                    2);

    s1_preprocess_kernel<<<grid, block, 0, stream>>>(pa, pb);
    if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
    return Status::OK;
}

}  // namespace stitch::s1
