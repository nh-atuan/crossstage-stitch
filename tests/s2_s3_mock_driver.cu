// Test-only bridge: S1 mock gray images -> native CUDA S2/S3 -> native CUDA S4/S5.
// Usage: s2_s3_mock_driver input.bin   (written by tools/s2_s3/check_mock_pipeline.py)
// Input layout (little endian):
//   char[8] "S2S3PIP1"; int32 orig_wa, orig_ha, orig_wb, orig_hb; float64 scale;
//   int32 fast_threshold, max_keypoints; float32 harris_k;
//   uint8 gray_a[work_ha * work_wa]; uint8 gray_b[work_hb * work_wb]
// Prints one JSON object on stdout. S4/S5 parameters match tests/s4_s5_mock_driver.cu
// and tools/mock/stages.py; they are not tuned production defaults.
#include <cuda_runtime.h>

#include "check.hpp"
#include "common/geometry.hpp"
#include "s2_s3/s2.hpp"
#include "s2_s3/s3.hpp"
#include "s4/s4.hpp"
#include "s5/s5.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace stitch;
#define CUDA_CHECK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (false)

namespace {

template <class T> struct Device {
    T* p{};
    explicit Device(std::size_t n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(n, 1) * sizeof(T)));
    }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

// S2 + S3 buffers for one image.
struct Features {
    int w, h;
    Device<std::uint8_t> image;
    Device<float> x, y, response;
    Device<int> count{1};
    Device<std::uint8_t> descriptors;
    Device<std::uint8_t>* ws2{};
    Device<std::uint8_t>* ws3{};
    std::size_t bytes2 = 0, bytes3 = 0;
    ImageView view{};
    Keypoints points{};
    Descriptors desc{};

    Features(const std::vector<std::uint8_t>& gray, int width, int height, const s2::Config& c)
        : w(width), h(height), image(gray.size()), x(c.max_keypoints), y(c.max_keypoints),
          response(c.max_keypoints),
          descriptors(static_cast<std::size_t>(c.max_keypoints) * s3::kDescriptorBytes) {
        CUDA_CHECK(cudaMemcpy(image.p, gray.data(), gray.size(), cudaMemcpyHostToDevice));
        CHECK(s2::s2_workspace_bytes(w, h, c, &bytes2) == Status::OK);
        CHECK(s3::s3_workspace_bytes(w, h, &bytes3) == Status::OK);
        ws2 = new Device<std::uint8_t>(bytes2);
        ws3 = new Device<std::uint8_t>(bytes3);
        view = {image.p, w, h, w, PixelFormat::GRAY8};
        points = {x.p, y.p, response.p, c.max_keypoints, count.p};
        desc = {DescriptorType::BINARY_U8, s3::kDescriptorBytes, s3::kDescriptorBytes, descriptors.p};
    }
    ~Features() {
        delete ws2;
        delete ws3;
    }
    void run(const s2::Config& c, cudaStream_t stream) {
        CHECK(s2::s2_detect(view, c, points, ws2->p, bytes2, stream) == Status::OK);
        CHECK(s3::s3_describe(view, points, desc, ws3->p, bytes3, stream) == Status::OK);
    }
};

template <class T> void read(std::ifstream& in, T* value, std::size_t n = 1) {
    in.read(reinterpret_cast<char*>(value), static_cast<std::streamsize>(sizeof(T) * n));
    CHECK(in.good());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: s2_s3_mock_driver input.bin\n");
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    std::ifstream in(argv[1], std::ios::binary);
    CHECK(in.good());
    char magic[8];
    read(in, magic, 8);
    CHECK(std::memcmp(magic, "S2S3PIP1", 8) == 0);
    std::int32_t orig[4];
    double scale = 0;
    std::int32_t threshold = 0, k = 0;
    float harris_k = 0;
    read(in, orig, 4);
    read(in, &scale);
    read(in, &threshold);
    read(in, &k);
    read(in, &harris_k);
    ScaleInfo sa{}, sb{};
    CHECK(make_scale_info(orig[0], orig[1], scale, &sa) == Status::OK);
    CHECK(make_scale_info(orig[2], orig[3], scale, &sb) == Status::OK);
    std::vector<std::uint8_t> gray_a(static_cast<std::size_t>(sa.work_w) * sa.work_h);
    std::vector<std::uint8_t> gray_b(static_cast<std::size_t>(sb.work_w) * sb.work_h);
    read(in, gray_a.data(), gray_a.size());
    read(in, gray_b.data(), gray_b.size());
    CHECK(in.peek() == std::char_traits<char>::eof());

    const s2::Config c2{threshold, k, harris_k};
    const s4::Config c4{0.8, false};
    const s5::Config c5{2000, 0, 64, 1, 3.0, 0, 1};
    Features a(gray_a, sa.work_w, sa.work_h, c2), b(gray_b, sb.work_w, sb.work_h, c2);
    Device<int32_t> ia(k), ib(k);
    Device<float> distance(k);
    Device<int> match_count(1);
    Device<std::uint8_t> mask(k);
    Matches matches{ia.p, ib.p, distance.p, k, match_count.p};
    std::size_t bytes4 = 0, bytes5 = 0;
    CHECK(s4::s4_workspace_bytes(k, k, c4, &bytes4) == Status::OK);
    CHECK(s5::s5_workspace_bytes(c5, &bytes5) == Status::OK);
    Device<std::uint8_t> work4(bytes4), work5(bytes5);
    HomographyResult* result{};
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&result), sizeof(*result)));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaDeviceSynchronize());  // uploads on the default stream are complete

    cudaEvent_t e0, e1, e2;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventRecord(e0, stream));
    a.run(c2, stream);
    b.run(c2, stream);
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CHECK(s4::s4_match(a.points, a.desc, b.points, b.desc, c4, matches, work4.p, bytes4, stream) ==
          Status::OK);
    CHECK(s5::estimate(a.points, b.points, matches, sa, sb, c5, mask.p, result, work5.p, bytes5,
                       stream) == Status::OK);
    CUDA_CHECK(cudaEventRecord(e2, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float s2_s3_ms = 0, s4_s5_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&s2_s3_ms, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&s4_s5_ms, e1, e2));
    int na = 0, nb = 0;
    CUDA_CHECK(cudaMemcpy(&na, a.count.p, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&nb, b.count.p, sizeof(int), cudaMemcpyDeviceToHost));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "GPU=" << prop.name << " seed=" << c5.seed << " R=" << c5.hypotheses
              << " K=" << k << " S5=" << status_name(result->status) << '\n';
    std::cout << std::setprecision(17) << "{\"status\":\"" << status_name(result->status)
              << "\",\"gpu\":\"" << prop.name << "\",\"num_kp_a\":" << na << ",\"num_kp_b\":" << nb
              << ",\"num_matches\":" << result->num_matches << ",\"num_inliers\":" << result->num_inliers
              << ",\"inlier_ratio\":" << result->inlier_ratio
              << ",\"inlier_coverage\":" << result->inlier_coverage
              << ",\"s2_s3_gpu_ms\":" << s2_s3_ms << ",\"s4_s5_gpu_ms\":" << s4_s5_ms;
    if (result->status == Status::OK) {
        std::cout << ",\"reproj_residual\":" << result->reproj_residual << ",\"H\":[";
        for (int i = 0; i < 9; ++i) std::cout << (i ? "," : "") << result->H[i];
        std::cout << "]";
    }
    std::cout << "}\n";
    const Status status = result->status;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    CUDA_CHECK(cudaFreeHost(result));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return status == Status::OK ? 0 : 1;
}
