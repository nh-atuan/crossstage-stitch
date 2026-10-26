// CUDA tests for S1 (Preprocessing: fused 15-bit RGB2GRAY + Area Resizing).
// Verifies argument checks, pitch handling, and accuracy for S in {1.0, 0.5, 0.25, 0.75}.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "check.hpp"
#include "s1/s1.hpp"

using namespace stitch;

#define CUDA_CHECK(call) do {                                                  \
    const cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                                  \
        std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(err));                                 \
        std::exit(1);                                                          \
    }                                                                          \
} while (false)

namespace {

constexpr int kPitchPad = 19; // Odd padding to ensure pitch_bytes != width * channels

template <typename T>
struct DeviceBuffer {
    T* ptr{nullptr};
    std::size_t count{0};

    explicit DeviceBuffer(std::size_t n) : count(n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr),
                              std::max<std::size_t>(n, 1) * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    std::vector<T> to_host() const {
        std::vector<T> host(count);
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return host;
    }
};

struct GpuRgbImage {
    DeviceBuffer<uint8_t> buffer;
    ImageView view;

    GpuRgbImage(int w, int h, const std::vector<uint8_t>& rgb_data)
        : buffer(static_cast<std::size_t>(w * 3 + kPitchPad) * h),
          view{nullptr, w, h, w * 3 + kPitchPad, PixelFormat::RGB8} {
        view.data = buffer.ptr;
        CUDA_CHECK(cudaMemset(buffer.ptr, 0xee, buffer.count));
        CUDA_CHECK(cudaMemcpy2D(buffer.ptr, view.pitch_bytes,
                                rgb_data.data(), w * 3,
                                w * 3, h, cudaMemcpyHostToDevice));
    }
};

struct GpuGrayImage {
    DeviceBuffer<uint8_t> buffer;
    ImageView view;

    GpuGrayImage(int w, int h)
        : buffer(static_cast<std::size_t>(w + kPitchPad) * h),
          view{nullptr, w, h, w + kPitchPad, PixelFormat::GRAY8} {
        view.data = buffer.ptr;
        CUDA_CHECK(cudaMemset(buffer.ptr, 0x55, buffer.count));
    }

    std::vector<uint8_t> download() const {
        std::vector<uint8_t> host(static_cast<std::size_t>(view.width) * view.height);
        CUDA_CHECK(cudaMemcpy2D(host.data(), view.width,
                                buffer.ptr, view.pitch_bytes,
                                view.width, view.height, cudaMemcpyDeviceToHost));
        return host;
    }
};

inline int ref_rgb_to_gray(int r, int g, int b) {
    return (9798 * r + 19235 * g + 3735 * b + 16384) >> 15;
}

std::vector<uint8_t> cpu_reference_s1(const std::vector<uint8_t>& rgb, int w, int h,
                                      int work_w, int work_h, double s) {
    std::vector<uint8_t> gray(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int idx = (y * w + x) * 3;
            gray[y * w + x] = static_cast<uint8_t>(ref_rgb_to_gray(rgb[idx], rgb[idx + 1], rgb[idx + 2]));
        }
    }

    std::vector<uint8_t> out(work_w * work_h);
    if (w == work_w && h == work_h) {
        return gray;
    }

    if (w == 2 * work_w && h == 2 * work_h) {
        for (int yd = 0; yd < work_h; ++yd) {
            for (int xd = 0; xd < work_w; ++xd) {
                const int s_val = gray[(2 * yd) * w + 2 * xd] +
                                  gray[(2 * yd) * w + 2 * xd + 1] +
                                  gray[(2 * yd + 1) * w + 2 * xd] +
                                  gray[(2 * yd + 1) * w + 2 * xd + 1];
                out[yd * work_w + xd] = static_cast<uint8_t>((s_val + 2) >> 2);
            }
        }
        return out;
    }

    if (w == 4 * work_w && h == 4 * work_h) {
        for (int yd = 0; yd < work_h; ++yd) {
            for (int xd = 0; xd < work_w; ++xd) {
                int sum = 0;
                for (int dy = 0; dy < 4; ++dy) {
                    for (int dx = 0; dx < 4; ++dx) {
                        sum += gray[(4 * yd + dy) * w + (4 * xd + dx)];
                    }
                }
                out[yd * work_w + xd] = static_cast<uint8_t>(std::lround(sum / 16.0));
            }
        }
        return out;
    }

    // General area box filter
    const double inv_sx = static_cast<double>(w) / work_w;
    const double inv_sy = static_cast<double>(h) / work_h;
    const double scale_area = 1.0 / (inv_sx * inv_sy);

    for (int yd = 0; yd < work_h; ++yd) {
        const double fsy1 = yd * inv_sy;
        const double fsy2 = (yd + 1) * inv_sy;
        const int y_start = static_cast<int>(std::floor(fsy1));
        const int y_end = static_cast<int>(std::ceil(fsy2));

        for (int xd = 0; xd < work_w; ++xd) {
            const double fsx1 = xd * inv_sx;
            const double fsx2 = (xd + 1) * inv_sx;
            const int x_start = static_cast<int>(std::floor(fsx1));
            const int x_end = static_cast<int>(std::ceil(fsx2));

            double total = 0.0;
            for (int ys = y_start; ys < y_end && ys < h; ++ys) {
                const double y_lo = std::max(static_cast<double>(ys), fsy1);
                const double y_hi = std::min(static_cast<double>(ys + 1), fsy2);
                const double wy = std::max(0.0, y_hi - y_lo);
                if (wy <= 0.0) continue;

                for (int xs = x_start; xs < x_end && xs < w; ++xs) {
                    const double x_lo = std::max(static_cast<double>(xs), fsx1);
                    const double x_hi = std::min(static_cast<double>(xs + 1), fsx2);
                    const double wx = std::max(0.0, x_hi - x_lo);
                    if (wx <= 0.0) continue;

                    total += static_cast<double>(gray[ys * w + xs]) * (wx * wy);
                }
            }
            out[yd * work_w + xd] = static_cast<uint8_t>(std::lround(total * scale_area));
        }
    }
    return out;
}

void test_invalid_arguments() {
    std::size_t ws_bytes = 100;
    CHECK(s1::s1_workspace_bytes(100, 100, 0.5, &ws_bytes) == Status::OK);
    CHECK(ws_bytes == 0);
    CHECK(s1::s1_workspace_bytes(0, 100, 0.5, &ws_bytes) == Status::INVALID_ARGUMENT);
    CHECK(s1::s1_workspace_bytes(100, 100, 0.0, &ws_bytes) == Status::INVALID_ARGUMENT);
    CHECK(s1::s1_workspace_bytes(100, 100, 1.5, &ws_bytes) == Status::INVALID_ARGUMENT);
    CHECK(s1::s1_workspace_bytes(100, 100, 0.5, nullptr) == Status::INVALID_ARGUMENT);

    ImageView invalid_view{};
    ScaleInfo si{};
    CHECK(s1::s1_preprocess_single(invalid_view, 0.5, nullptr, &si, nullptr) == Status::INVALID_ARGUMENT);
    CHECK(s1::s1_preprocess(invalid_view, invalid_view, 0.5, nullptr, nullptr, &si, &si, nullptr) == Status::INVALID_ARGUMENT);
}

void test_accuracy_and_batching(double scale, int w, int h) {
    ScaleInfo si_a{}, si_b{};
    CHECK(make_scale_info(w, h, scale, &si_a) == Status::OK);
    CHECK(make_scale_info(w, h, scale, &si_b) == Status::OK);

    // Create synthetic RGB patterns
    std::vector<uint8_t> rgb_a(w * h * 3);
    std::vector<uint8_t> rgb_b(w * h * 3);
    for (int i = 0; i < w * h * 3; ++i) {
        rgb_a[i] = static_cast<uint8_t>((i * 17 + 23) % 256);
        rgb_b[i] = static_cast<uint8_t>((i * 31 + 97) % 256);
    }

    GpuRgbImage dev_a(w, h, rgb_a);
    GpuRgbImage dev_b(w, h, rgb_b);
    GpuGrayImage out_a(si_a.work_w, si_a.work_h);
    GpuGrayImage out_b(si_b.work_w, si_b.work_h);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Execute batched S1
    CHECK(s1::s1_preprocess(dev_a.view, dev_b.view, scale,
                            &out_a.view, &out_b.view,
                            &si_a, &si_b, stream) == Status::OK);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto host_out_a = out_a.download();
    const auto host_out_b = out_b.download();

    const auto ref_out_a = cpu_reference_s1(rgb_a, w, h, si_a.work_w, si_a.work_h, scale);
    const auto ref_out_b = cpu_reference_s1(rgb_b, w, h, si_b.work_w, si_b.work_h, scale);

    const int max_allowed_diff = (scale == 0.75) ? 1 : 0;

    int max_diff_a = 0;
    for (std::size_t i = 0; i < host_out_a.size(); ++i) {
        max_diff_a = std::max(max_diff_a, std::abs(static_cast<int>(host_out_a[i]) - static_cast<int>(ref_out_a[i])));
    }
    CHECK(max_diff_a <= max_allowed_diff);

    int max_diff_b = 0;
    for (std::size_t i = 0; i < host_out_b.size(); ++i) {
        max_diff_b = std::max(max_diff_b, std::abs(static_cast<int>(host_out_b[i]) - static_cast<int>(ref_out_b[i])));
    }
    CHECK(max_diff_b <= max_allowed_diff);

    // Also verify single-image version matches batch exactly
    GpuGrayImage single_out(si_a.work_w, si_a.work_h);
    CHECK(s1::s1_preprocess_single(dev_a.view, scale, &single_out.view, &si_a, stream) == Status::OK);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto host_single = single_out.download();
    CHECK(host_single == host_out_a);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("No CUDA device found. Skipping test (exit 77).");
        return 77;
    }

    test_invalid_arguments();

    // Test scales {1.0, 0.5, 0.25, 0.75} on different image resolutions
    const double scales[] = {1.0, 0.5, 0.25, 0.75};
    const int resolutions[][2] = {
        {64, 64},
        {128, 96},
        {640, 480}
    };

    for (double s : scales) {
        for (const auto& res : resolutions) {
            test_accuracy_and_batching(s, res[0], res[1]);
        }
    }

    std::puts("test_s1_cuda: passed all tests successfully.");
    return 0;
}
