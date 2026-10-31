// S1: Preprocessing kernel implementation.
// Fused integer 15-bit RGB-to-GRAY and Area Resizing box filter.
// Processes image A and image B in a single batched kernel launch (blockIdx.z).
// Also provides zero-allocation probe gradient reduction (mean absolute gradient).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

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

// ---------------------------------------------------------------------------
// Gradient accumulation and normalization kernels (Probe Scene Signals)
// ---------------------------------------------------------------------------

__global__ void s1_gradient_accum_kernel(
    const uint8_t* src_a, int pitch_a, int w_a, int h_a,
    const uint8_t* src_b, int pitch_b, int w_b, int h_b,
    unsigned long long* d_accum) {
    const int z = blockIdx.z;
    const uint8_t* src = (z == 0) ? src_a : src_b;
    const int pitch = (z == 0) ? pitch_a : pitch_b;
    const int w = (z == 0) ? w_a : w_b;
    const int h = (z == 0) ? h_a : h_b;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int diff = 0;
    if (x < w && y < h && src != nullptr) {
        const uint8_t* row = src + y * pitch;
        const int curr = static_cast<int>(row[x]);
        if (x + 1 < w) {
            diff += std::abs(static_cast<int>(row[x + 1]) - curr);
        }
        if (y + 1 < h) {
            const uint8_t* next_row = src + (y + 1) * pitch;
            diff += std::abs(static_cast<int>(next_row[x]) - curr);
        }
    }

    // Intra-warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        diff += __shfl_down_sync(0xffffffff, diff, offset);
    }

    // Shared memory reduction across warps in block
    __shared__ unsigned long long s_block_sum;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_block_sum = 0;
    }
    __syncthreads();

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if ((tid % 32) == 0 && diff != 0) {
        atomicAdd(&s_block_sum, static_cast<unsigned long long>(diff));
    }
    __syncthreads();

    if (tid == 0 && s_block_sum != 0) {
        atomicAdd(&d_accum[z], s_block_sum);
    }
}

__global__ void s1_gradient_normalize_kernel(
    const unsigned long long* d_accum,
    int w_a, int h_a, int w_b, int h_b,
    double* d_out) {
    const int z = threadIdx.x; // 0 or 1
    if (z > 1) return;
    const int w = (z == 0) ? w_a : w_b;
    const int h = (z == 0) ? h_a : h_b;
    const unsigned long long edges = static_cast<unsigned long long>(h) * (w - 1) +
                                    static_cast<unsigned long long>(w) * (h - 1);
    if (edges == 0) {
        d_out[z] = 0.0;
    } else {
        d_out[z] = static_cast<double>(d_accum[z]) / (255.0 * static_cast<double>(edges));
    }
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

Status s1_compute_gradient(const ImageView& gray_a, const ImageView& gray_b,
                           void* d_workspace,
                           double* pinned_gradient_a, double* pinned_gradient_b,
                           cudaStream_t stream) {
    if (gray_a.data == nullptr || gray_b.data == nullptr ||
        gray_a.format != PixelFormat::GRAY8 || gray_b.format != PixelFormat::GRAY8 ||
        gray_a.width < 2 || gray_a.height < 2 ||
        gray_b.width < 2 || gray_b.height < 2 ||
        gray_a.pitch_bytes < gray_a.width || gray_b.pitch_bytes < gray_b.width ||
        d_workspace == nullptr ||
        pinned_gradient_a == nullptr || pinned_gradient_b == nullptr) {
        return Status::INVALID_ARGUMENT;
    }

    // d_workspace layout:
    // [0..15]  : unsigned long long d_accum[2]
    // [16..31] : double d_out[2]
    auto* d_accum = reinterpret_cast<unsigned long long*>(d_workspace);
    auto* d_out = reinterpret_cast<double*>(reinterpret_cast<uint8_t*>(d_workspace) + 16);

    cudaMemsetAsync(d_workspace, 0, 32, stream);

    const int max_w = std::max(gray_a.width, gray_b.width);
    const int max_h = std::max(gray_a.height, gray_b.height);

    const dim3 block(16, 16);
    const dim3 grid((max_w + block.x - 1) / block.x,
                    (max_h + block.y - 1) / block.y,
                    2);

    s1_gradient_accum_kernel<<<grid, block, 0, stream>>>(
        gray_a.data, gray_a.pitch_bytes, gray_a.width, gray_a.height,
        gray_b.data, gray_b.pitch_bytes, gray_b.width, gray_b.height,
        d_accum);

    s1_gradient_normalize_kernel<<<1, 2, 0, stream>>>(
        d_accum,
        gray_a.width, gray_a.height,
        gray_b.width, gray_b.height,
        d_out);

    cudaMemcpyAsync(pinned_gradient_a, &d_out[0], sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pinned_gradient_b, &d_out[1], sizeof(double), cudaMemcpyDeviceToHost, stream);

    if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
    return Status::OK;
}

Status s1_compute_gradient_single(const ImageView& gray,
                                  void* d_workspace,
                                  double* pinned_gradient,
                                  cudaStream_t stream) {
    if (gray.data == nullptr || gray.format != PixelFormat::GRAY8 ||
        gray.width < 2 || gray.height < 2 ||
        gray.pitch_bytes < gray.width ||
        d_workspace == nullptr || pinned_gradient == nullptr) {
        return Status::INVALID_ARGUMENT;
    }

    auto* d_accum = reinterpret_cast<unsigned long long*>(d_workspace);
    auto* d_out = reinterpret_cast<double*>(reinterpret_cast<uint8_t*>(d_workspace) + 16);

    cudaMemsetAsync(d_workspace, 0, 32, stream);

    const dim3 block(16, 16);
    const dim3 grid((gray.width + block.x - 1) / block.x,
                    (gray.height + block.y - 1) / block.y,
                    1);

    s1_gradient_accum_kernel<<<grid, block, 0, stream>>>(
        gray.data, gray.pitch_bytes, gray.width, gray.height,
        nullptr, 0, 0, 0,
        d_accum);

    s1_gradient_normalize_kernel<<<1, 1, 0, stream>>>(
        d_accum,
        gray.width, gray.height,
        0, 0,
        d_out);

    cudaMemcpyAsync(pinned_gradient, &d_out[0], sizeof(double), cudaMemcpyDeviceToHost, stream);

    if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
    return Status::OK;
}

}  // namespace stitch::s1
