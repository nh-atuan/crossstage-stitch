// Test-only text bridge: mock S1-S3 -> native CUDA S4/S5 -> mock S6.
// Parameters below match tools/mock/stages.py, not tuned production defaults.
#include <cuda_runtime.h>
#include "check.hpp"
#include "common/geometry.hpp"
#include "s4/s4.hpp"
#include "s5/s5.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace stitch;
#define CUDA_CHECK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (false)

template <class T> struct Device {
    T* p{};
    explicit Device(std::size_t n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(n, 1) * sizeof(T)));
    }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void upload(const std::vector<T>& data) {
        if (!data.empty()) CUDA_CHECK(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download(std::size_t n) {
        std::vector<T> data(n);
        if (n) CUDA_CHECK(cudaMemcpy(data.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return data;
    }
};

template <class T> void read_descriptors(std::size_t size, Device<T>& out) {
    std::vector<T> values(size);
    for (auto& v : values) {
        double value;
        CHECK(static_cast<bool>(std::cin >> value));
        v = static_cast<T>(value);
    }
    out.upload(values);
}

struct Features {
    Device<float> x, y;
    Device<int> count{1};
    Device<std::uint8_t> binary;
    Device<float> floating;
    Keypoints points{};
    Descriptors descriptors{};
    Features(int n, int dim, DescriptorType type)
        : x(n), y(n), binary(type == DescriptorType::BINARY_U8 ? static_cast<std::size_t>(n) * dim : 0),
          floating(type == DescriptorType::FLOAT32 ? static_cast<std::size_t>(n) * dim : 0) {
        std::vector<float> xs(n), ys(n);
        for (int i = 0; i < n; ++i) CHECK(static_cast<bool>(std::cin >> xs[i] >> ys[i]));
        x.upload(xs); y.upload(ys); count.upload({n});
        points = {x.p, y.p, nullptr, n, count.p};
        const bool is_float = type == DescriptorType::FLOAT32;
        descriptors = {type, dim, dim * (is_float ? 4 : 1), is_float ? static_cast<void*>(floating.p) : binary.p};
        if (is_float) read_descriptors(static_cast<std::size_t>(n) * dim, floating);
        else read_descriptors(static_cast<std::size_t>(n) * dim, binary);
    }
};

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    int na, nb, dim, type, wa, ha, wb, hb;
    double scale;
    CHECK(static_cast<bool>(std::cin >> na >> nb >> dim >> type >> wa >> ha >> wb >> hb >> scale));
    CHECK(na >= 0 && nb >= 0 && dim > 0 && (type == 0 || type == 1));
    const auto descriptor_type = static_cast<DescriptorType>(type);
    Features a(na, dim, descriptor_type), b(nb, dim, descriptor_type);
    Device<int32_t> ia(na), ib(na);
    Device<float> distance(na);
    Device<int> count(1);
    Device<std::uint8_t> mask(na);
    Matches matches{ia.p, ib.p, distance.p, na, count.p};
    ScaleInfo sa{}, sb{};
    CHECK(make_scale_info(wa, ha, scale, &sa) == Status::OK);
    CHECK(make_scale_info(wb, hb, scale, &sb) == Status::OK);
    const s4::Config c4{0.8, false};
    const s5::Config c5{2000, 0, 64, 1, 3.0, 0, 1};
    std::size_t bytes4{}, bytes5{};
    CHECK(s4::s4_workspace_bytes(na, nb, c4, &bytes4) == Status::OK);
    CHECK(s5::s5_workspace_bytes(c5, &bytes5) == Status::OK);
    Device<std::uint8_t> work4(bytes4), work5(bytes5);
    HomographyResult* result{};
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&result), sizeof(*result)));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Complete default-stream fixture uploads before using a nonblocking stream.
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(s4::s4_match(a.points, a.descriptors, b.points, b.descriptors, c4, matches,
                      work4.p, bytes4, stream) == Status::OK);
    CHECK(s5::estimate(a.points, b.points, matches, sa, sb, c5, mask.p, result,
                       work5.p, bytes5, stream) == Status::OK);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "GPU=" << prop.name << " seed=" << c5.seed << " R=" << c5.hypotheses
              << " S5=" << status_name(result->status) << '\n';
    const Status status = result->status;
    if (status == Status::OK) {
        const int n = count.download(1)[0];
        const auto ai = ia.download(n), bi = ib.download(n);
        const auto distances = distance.download(n);
        const auto inliers = mask.download(n);
        std::cout << std::setprecision(17) << "{\"H\":[";
        for (int i = 0; i < 9; ++i) std::cout << (i ? "," : "") << result->H[i];
        std::cout << "],\"H_s\":[";
        for (int i = 0; i < 9; ++i) std::cout << (i ? "," : "") << result->H_s[i];
        std::cout << "],\"matches\":[";
        for (int i = 0; i < n; ++i)
            std::cout << (i ? "," : "") << "[" << ai[i] << "," << bi[i] << "," << distances[i] << "]";
        std::cout << "],\"mask\":[";
        for (int i = 0; i < n; ++i) std::cout << (i ? "," : "") << static_cast<int>(inliers[i]);
        std::cout << "],\"num_inliers\":" << result->num_inliers
                  << ",\"inlier_ratio\":" << result->inlier_ratio
                  << ",\"inlier_coverage\":" << result->inlier_coverage
                  << ",\"reproj_residual\":" << result->reproj_residual << "}\n";
    }
    CUDA_CHECK(cudaFreeHost(result));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return status == Status::OK ? 0 : 1;
}
