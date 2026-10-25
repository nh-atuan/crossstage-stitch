// S3 CUDA: float32 7x7 Gaussian (row pass, column pass) -> per keypoint, one
// warp computes the intensity-centroid angle and 32 descriptor bytes (one per
// lane). Arithmetic order matches tools/s2_s3/reference.py; fused multiply-add
// is avoided with __fmul_rn/__fadd_rn so results are reproducible bit for bit.
#include <cuda_runtime.h>

#include <cstdint>

#include "s2_s3/orb_pattern.hpp"
#include "s2_s3/s2_s3_internal.hpp"

namespace stitch::s3 {
namespace {

using s2_s3::detail::kAlignment;
using s2_s3::detail::S3Layout;

constexpr int kTile = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kHalfPatch = 15;
constexpr int kEdge = s2::kEdge;

// float32 taps of cv::getGaussianKernel(7, 2, CV_32F), tested in pytest.
__constant__ float kTaps[7] = {0x1.1f5f62p-4f, 0x1.0c70fcp-3f, 0x1.869472p-3f, 0x1.ba95c0p-3f,
                               0x1.869472p-3f, 0x1.0c70fcp-3f, 0x1.1f5f62p-4f};
// Row half-widths of the radius-15 orientation patch (ORB computeKeyPoints).
__constant__ int kUMax[kHalfPatch + 1] = {15, 15, 15, 15, 14, 14, 14, 13,
                                          13, 12, 11, 10, 9, 8, 6, 3};
__constant__ int kPattern[kPatternPairs][4] = {STITCH_S3_ORB_PATTERN_31};

// cv::fastAtan2 polynomial coefficients (degrees), as float32.
constexpr float kAtanP1 = 0x1.ca44dep+5f;
constexpr float kAtanP3 = -0x1.2aaddcp+4f;
constexpr float kAtanP5 = 0x1.1d3f7ep+3f;
constexpr float kAtanP7 = -0x1.4515b2p+1f;
constexpr float kAtanEps = 0x1p-52f;          // (float)DBL_EPSILON
constexpr float kDegToRad = 0x1.1df46ap-6f;  // (float)(CV_PI / 180.f)

__device__ int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
}

__global__ void blur_rows_kernel(ImageView image, float* rows) {
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x >= image.width || y >= image.height) return;
    const std::uint8_t* row = image.data + static_cast<std::size_t>(y) * image.pitch_bytes;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
        const float v = row[reflect101(x + i - 3, image.width)];
        acc = __fadd_rn(acc, __fmul_rn(kTaps[i], v));
    }
    rows[static_cast<std::size_t>(y) * image.width + x] = acc;
}

__global__ void blur_cols_kernel(const float* rows, int width, int height, std::uint8_t* blurred) {
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x >= width || y >= height) return;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
        const float v = rows[static_cast<std::size_t>(reflect101(y + i - 3, height)) * width + x];
        acc = __fadd_rn(acc, __fmul_rn(kTaps[i], v));
    }
    blurred[static_cast<std::size_t>(y) * width + x] =
        static_cast<std::uint8_t>(min(max(__float2int_rn(acc), 0), 255));
}

// cv::fastAtan2 in degrees [0, 360), step by step in float32.
__device__ float fast_atan2(float y, float x) {
    const float ax = fabsf(x), ay = fabsf(y);
    const bool use_x = ax >= ay;
    const float c = use_x ? __fdiv_rn(ay, __fadd_rn(ax, kAtanEps))
                          : __fdiv_rn(ax, __fadd_rn(ay, kAtanEps));
    const float c2 = __fmul_rn(c, c);
    float p = __fadd_rn(__fmul_rn(kAtanP7, c2), kAtanP5);
    p = __fadd_rn(__fmul_rn(p, c2), kAtanP3);
    p = __fadd_rn(__fmul_rn(p, c2), kAtanP1);
    p = __fmul_rn(p, c);
    float a = use_x ? p : __fsub_rn(90.0f, p);
    if (x < 0) a = __fsub_rn(180.0f, a);
    if (y < 0) a = __fsub_rn(360.0f, a);
    return a;
}

__device__ int warp_sum(int v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

__device__ int sample(const std::uint8_t* centre, int width, int px, int py, float a, float b) {
    const float fx = static_cast<float>(px), fy = static_cast<float>(py);
    const int ix = __float2int_rn(__fsub_rn(__fmul_rn(fx, a), __fmul_rn(fy, b)));
    const int iy = __float2int_rn(__fadd_rn(__fmul_rn(fx, b), __fmul_rn(fy, a)));
    return centre[iy * width + ix];
}

__global__ void describe_kernel(ImageView image, const std::uint8_t* blurred, Keypoints keypoints,
                                Descriptors output) {
    const int lane = threadIdx.x & 31;
    const int index = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (index >= keypoints.capacity || index >= *keypoints.d_count) return;
    std::uint8_t* row = static_cast<std::uint8_t*>(output.data) +
                        static_cast<std::size_t>(index) * output.row_stride_bytes;
    const int x = __float2int_rn(keypoints.x[index]);
    const int y = __float2int_rn(keypoints.y[index]);
    if (x < kEdge || x >= image.width - kEdge || y < kEdge || y >= image.height - kEdge) {
        row[lane] = 0;
        return;
    }
    // Intensity-centroid moments on the unblurred image; lane v handles line pair +-v.
    const int pitch = image.pitch_bytes;
    const std::uint8_t* centre = image.data + static_cast<std::size_t>(y) * pitch + x;
    int m10 = 0, m01 = 0;
    if (lane == 0) {
        for (int u = -kHalfPatch; u <= kHalfPatch; ++u) m10 += u * centre[u];
    } else if (lane <= kHalfPatch) {
        const int d = kUMax[lane];
        int v_sum = 0;
        for (int u = -d; u <= d; ++u) {
            const int plus = centre[u + lane * pitch], minus = centre[u - lane * pitch];
            v_sum += plus - minus;
            m10 += u * (plus + minus);
        }
        m01 = lane * v_sum;
    }
    m10 = warp_sum(m10);
    m01 = warp_sum(m01);
    const float angle = __fmul_rn(fast_atan2(static_cast<float>(m01), static_cast<float>(m10)),
                                  kDegToRad);
    const float a = static_cast<float>(cos(static_cast<double>(angle)));
    const float b = static_cast<float>(sin(static_cast<double>(angle)));
    // Lane l packs pattern pairs 8l .. 8l+7 into byte l (bit j = pair 8l+j).
    const std::uint8_t* smooth = blurred + static_cast<std::size_t>(y) * image.width + x;
    int value = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int* p = kPattern[lane * 8 + j];
        const int t0 = sample(smooth, image.width, p[0], p[1], a, b);
        const int t1 = sample(smooth, image.width, p[2], p[3], a, b);
        value |= (t0 < t1) << j;
    }
    row[lane] = static_cast<std::uint8_t>(value);
}

template <class T> T* at(void* base, std::size_t offset) {
    return reinterpret_cast<T*>(static_cast<char*>(base) + offset);
}

}  // namespace

Status s3_describe(const ImageView& image, const Keypoints& keypoints, Descriptors output,
                   void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
    if (!s2_s3::detail::valid_image(image) || keypoints.capacity < 0 || !keypoints.d_count ||
        reinterpret_cast<std::uintptr_t>(keypoints.d_count) % alignof(int) ||
        (keypoints.capacity > 0 && (!keypoints.x || !keypoints.y || !output.data)) ||
        output.type != DescriptorType::BINARY_U8 || output.dim != kDescriptorBytes ||
        output.row_stride_bytes < kDescriptorBytes) {
        return Status::INVALID_ARGUMENT;
    }
    S3Layout layout;
    if (!s2_s3::detail::make_s3_layout(image.width, image.height, &layout) || !workspace ||
        reinterpret_cast<std::uintptr_t>(workspace) % kAlignment) {
        return Status::INVALID_ARGUMENT;
    }
    if (workspace_bytes < layout.total) return Status::INSUFFICIENT_WORKSPACE;
    if (keypoints.capacity == 0) return Status::OK;

    auto* blurred = at<std::uint8_t>(workspace, layout.blurred);
    auto* rows = at<float>(workspace, layout.rows);
    const dim3 tile(kTile, kTile);
    const dim3 grid((image.width + kTile - 1) / kTile, (image.height + kTile - 1) / kTile);
    blur_rows_kernel<<<grid, tile, 0, stream>>>(image, rows);
    blur_cols_kernel<<<grid, tile, 0, stream>>>(rows, image.width, image.height, blurred);
    const int blocks = (keypoints.capacity + kWarpsPerBlock - 1) / kWarpsPerBlock;
    describe_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(image, blurred, keypoints, output);
    return cudaGetLastError() == cudaSuccess ? Status::OK : Status::CUDA_ERROR;
}

}  // namespace stitch::s3
