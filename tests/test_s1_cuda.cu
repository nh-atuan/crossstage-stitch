// CUDA tests for S1 (Preprocessing: fused 15-bit RGB2GRAY + Area Resizing).
// Verifies argument checks, pitch handling, accuracy for S in {1.0, 0.5, 0.25, 0.75},
// and probe gradient computation (mean absolute gradient).
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

    GpuGrayImage(int w, int h, const std::vector<uint8_t>& init_data)
        : buffer(static_cast<std::size_t>(w + kPitchPad) * h),
          view{nullptr, w, h, w + kPitchPad, PixelFormat::GRAY8} {
        view.data = buffer.ptr;
        CUDA_CHECK(cudaMemset(buffer.ptr, 0x55, buffer.count));
        CUDA_CHECK(cudaMemcpy2D(buffer.ptr, view.pitch_bytes,
                                init_data.data(), w,
                                w, h, cudaMemcpyHostToDevice));
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

    if (w == work_w && h == work_h) {
        return gray;
    }

    std::vector<uint8_t> out(work_w * work_h);
    if (w == 2 * work_w && h == 2 * work_h) {
        for (int yd = 0; yd < work_h; ++yd) {
            for (int xd = 0; xd < work_w; ++xd) {
                const int y00 = gray[(2 * yd) * w + (2 * xd)];
                const int y01 = gray[(2 * yd) * w + (2 * xd + 1)];
                const int y10 = gray[(2 * yd + 1) * w + (2 * xd)];
                const int y11 = gray[(2 * yd + 1) * w + (2 * xd + 1)];
                out[yd * work_w + xd] = static_cast<uint8_t>((y00 + y01 + y10 + y11 + 2) >> 2);
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
                out[yd * work_w + xd] = static_cast<uint8_t>(std::lround(static_cast<double>(sum) * 0.0625));
            }
        }
        return out;
    }

    const double inv_sx = static_cast<double>(w) / static_cast<double>(work_w);
    const double inv_sy = static_cast<double>(h) / static_cast<double>(work_h);
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

double cpu_ref_mean_abs_gradient(const std::vector<uint8_t>& gray, int w, int h) {
    uint64_t sum = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w - 1; ++x) {
            sum += std::abs(static_cast<int>(gray[y * w + x + 1]) - static_cast<int>(gray[y * w + x]));
        }
    }
    for (int y = 0; y < h - 1; ++y) {
        for (int x = 0; x < w; ++x) {
            sum += std::abs(static_cast<int>(gray[(y + 1) * w + x]) - static_cast<int>(gray[y * w + x]));
        }
    }
    uint64_t edges = static_cast<uint64_t>(h) * (w - 1) + static_cast<uint64_t>(w) * (h - 1);
    return (edges > 0) ? (static_cast<double>(sum) / (255.0 * static_cast<double>(edges))) : 0.0;
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

    CHECK(s1::s1_gradient_workspace_bytes(nullptr) == Status::INVALID_ARGUMENT);
    std::size_t grad_ws = 0;
    CHECK(s1::s1_gradient_workspace_bytes(&grad_ws) == Status::OK);
    CHECK(grad_ws == 32);

    double dummy_g = 0.0;
    uint8_t dummy_ws[32];
    CHECK(s1::s1_compute_gradient(invalid_view, invalid_view, dummy_ws, &dummy_g, &dummy_g, nullptr) == Status::INVALID_ARGUMENT);
    CHECK(s1::s1_compute_gradient_single(invalid_view, dummy_ws, &dummy_g, nullptr) == Status::INVALID_ARGUMENT);
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

void test_gradient() {
    int w = 64, h = 48;
    DeviceBuffer<uint8_t> d_ws(32);
    double* pinned_ga = nullptr;
    double* pinned_gb = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_ga), sizeof(double)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_gb), sizeof(double)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 1. Constant image test: G must be 0.0
    std::vector<uint8_t> const_a(w * h, 128);
    std::vector<uint8_t> const_b(w * h, 200);
    GpuGrayImage dev_const_a(w, h, const_a);
    GpuGrayImage dev_const_b(w, h, const_b);

    CHECK(s1::s1_compute_gradient(dev_const_a.view, dev_const_b.view,
                                  d_ws.ptr, pinned_ga, pinned_gb, stream) == Status::OK);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(*pinned_ga == 0.0);
    CHECK(*pinned_gb == 0.0);

    // 2. Checkerboard test: G must be 1.0
    std::vector<uint8_t> checker_a(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            checker_a[y * w + x] = ((x + y) % 2) ? 255 : 0;
        }
    }
    GpuGrayImage dev_checker(w, h, checker_a);
    CHECK(s1::s1_compute_gradient_single(dev_checker.view, d_ws.ptr, pinned_ga, stream) == Status::OK);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(std::abs(*pinned_ga - 1.0) < 1e-9);

    // 3. Synthetic texture test against CPU reference
    std::vector<uint8_t> tex_a(w * h);
    std::vector<uint8_t> tex_b(w * h);
    for (int i = 0; i < w * h; ++i) {
        tex_a[i] = static_cast<uint8_t>((i * 37 + 11) % 256);
        tex_b[i] = static_cast<uint8_t>((i * 59 + 83) % 256);
    }
    GpuGrayImage dev_tex_a(w, h, tex_a);
    GpuGrayImage dev_tex_b(w, h, tex_b);

    CHECK(s1::s1_compute_gradient(dev_tex_a.view, dev_tex_b.view,
                                  d_ws.ptr, pinned_ga, pinned_gb, stream) == Status::OK);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double ref_ga = cpu_ref_mean_abs_gradient(tex_a, w, h);
    double ref_gb = cpu_ref_mean_abs_gradient(tex_b, w, h);

    CHECK(std::abs(*pinned_ga - ref_ga) < 1e-9);
    CHECK(std::abs(*pinned_gb - ref_gb) < 1e-9);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(pinned_ga));
    CUDA_CHECK(cudaFreeHost(pinned_gb));
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("No CUDA device found. Skipping test (exit 77).");
        return 77;
    }

    test_invalid_arguments();
    test_gradient();

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
