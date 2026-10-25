// Native S2/S3 checks: argument handling, determinism and golden comparison.
// Run: test_s2_s3_cuda [--require-t4] [--timings out.csv] [--golden case.bin ...]
// Golden files come from `python -m tools.s2_s3.export_golden`.
// Exit 77 means no CUDA device (with --require-t4 that is a failure instead).
// Tolerances (docs/decisions/s2-s3-orb.md): keypoints and blur exact, response
// relative error <= 1e-4, >= 99 % descriptor rows exact and the rest <= 2 bits.
#include <cuda_runtime.h>

#include "check.hpp"
#include "s2_s3/s2.hpp"
#include "s2_s3/s2_s3_internal.hpp"
#include "s2_s3/s3.hpp"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace stitch;
#define CUDA_CHECK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (false)

namespace {

constexpr int kPadding = 37;     // pitch padding to exercise pitch_bytes
constexpr int kWarmup = 2;
constexpr int kRepeats = 10;

template <class T> struct Device {
    T* p{};
    std::size_t n{};
    explicit Device(std::size_t count) : n(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T)));
    }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    std::vector<T> download(std::size_t count) const {
        std::vector<T> host(count);
        if (count) CUDA_CHECK(cudaMemcpy(host.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
};

struct Image {
    Device<std::uint8_t> storage;
    ImageView view;
    Image(const std::vector<std::uint8_t>& gray, int w, int h)
        : storage(static_cast<std::size_t>(w + kPadding) * h),
          view{nullptr, w, h, w + kPadding, PixelFormat::GRAY8} {
        view.data = storage.p;
        CUDA_CHECK(cudaMemset(storage.p, 0xa5, storage.n));
        CUDA_CHECK(cudaMemcpy2D(storage.p, view.pitch_bytes, gray.data(), w, w, h,
                                cudaMemcpyHostToDevice));
    }
};

struct Result {
    std::vector<float> x, y, response;
    std::vector<std::uint8_t> descriptors, blurred;
    bool operator==(const Result& o) const {
        return x == o.x && y == o.y && response == o.response &&
               descriptors == o.descriptors && blurred == o.blurred;
    }
};

// Runs S2 then S3 on one stream and downloads everything, including the blur.
struct Runner {
    int w, h, k;
    s2::Config config;
    Image image;
    Device<float> x, y, response;
    Device<int> count{1};
    Device<std::uint8_t> descriptors;
    std::size_t s2_bytes = 0, s3_bytes = 0;
    Device<std::uint8_t>* s2_ws{};
    Device<std::uint8_t>* s3_ws{};
    cudaStream_t stream{};

    Runner(const std::vector<std::uint8_t>& gray, int width, int height, s2::Config c)
        : w(width), h(height), k(c.max_keypoints), config(c), image(gray, width, height),
          x(k), y(k), response(k), descriptors(static_cast<std::size_t>(k) * 48) {
        CHECK(s2::s2_workspace_bytes(w, h, config, &s2_bytes) == Status::OK);
        CHECK(s3::s3_workspace_bytes(w, h, &s3_bytes) == Status::OK);
        s2_ws = new Device<std::uint8_t>(s2_bytes);
        s3_ws = new Device<std::uint8_t>(s3_bytes);
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    ~Runner() {
        delete s2_ws;
        delete s3_ws;
        cudaStreamDestroy(stream);
    }
    Keypoints keypoints() { return {x.p, y.p, response.p, k, count.p}; }
    Descriptors output() { return {DescriptorType::BINARY_U8, 32, 48, descriptors.p}; }

    void launch_s2() {
        CHECK(s2::s2_detect(image.view, config, keypoints(), s2_ws->p, s2_bytes, stream) == Status::OK);
    }
    void launch_s3() {
        CHECK(s3::s3_describe(image.view, keypoints(), output(), s3_ws->p, s3_bytes, stream) ==
              Status::OK);
    }
    Result run() {
        CUDA_CHECK(cudaMemsetAsync(descriptors.p, 0xcd, descriptors.n, stream));
        launch_s2();
        launch_s3();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int n = count.download(1)[0];
        CHECK(n >= 0 && n <= k);
        Result r;
        r.x = x.download(n);
        r.y = y.download(n);
        r.response = response.download(n);
        const auto raw = descriptors.download(static_cast<std::size_t>(n) * 48);
        for (int i = 0; i < n; ++i) {
            r.descriptors.insert(r.descriptors.end(), raw.begin() + i * 48, raw.begin() + i * 48 + 32);
            for (int b = 32; b < 48; ++b) CHECK(raw[i * 48 + b] == 0xcd);  // stride padding untouched
        }
        s2_s3::detail::S3Layout layout;
        CHECK(s2_s3::detail::make_s3_layout(w, h, &layout));
        r.blurred.resize(static_cast<std::size_t>(w) * h);
        CUDA_CHECK(cudaMemcpy(r.blurred.data(), s3_ws->p + layout.blurred, r.blurred.size(),
                              cudaMemcpyDeviceToHost));
        return r;
    }
};

struct Golden {
    std::string name;
    int w, h, threshold, k, count;
    float harris_k;
    std::vector<std::uint8_t> gray, blurred, descriptors;
    std::vector<float> keypoints;
};

Golden load_golden(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    CHECK(in.good());
    char magic[8];
    in.read(magic, 8);
    CHECK(in && std::memcmp(magic, "S2S3GLD1", 8) == 0);
    Golden g;
    g.name = path;
    std::int32_t header[4];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    in.read(reinterpret_cast<char*>(&g.harris_k), sizeof(float));
    std::int32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    CHECK(in.good());
    g.w = header[0]; g.h = header[1]; g.threshold = header[2]; g.k = header[3]; g.count = count;
    const auto pixels = static_cast<std::size_t>(g.w) * g.h;
    g.gray.resize(pixels);
    g.blurred.resize(pixels);
    g.keypoints.resize(static_cast<std::size_t>(count) * 3);
    g.descriptors.resize(static_cast<std::size_t>(count) * 32);
    in.read(reinterpret_cast<char*>(g.gray.data()), pixels);
    in.read(reinterpret_cast<char*>(g.blurred.data()), pixels);
    in.read(reinterpret_cast<char*>(g.keypoints.data()), g.keypoints.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(g.descriptors.data()), g.descriptors.size());
    CHECK(in.good());
    CHECK(in.peek() == std::char_traits<char>::eof());
    return g;
}

void check_golden(const Golden& g, FILE* timings, const std::string& gpu) {
    Runner runner(g.gray, g.w, g.h, {g.threshold, g.k, g.harris_k});
    const Result r = runner.run();
    CHECK(runner.run() == r);  // deterministic across runs

    const auto n = static_cast<int>(r.x.size());
    if (n != g.count) std::fprintf(stderr, "%s: count %d, expected %d\n", g.name.c_str(), n, g.count);
    CHECK(n == g.count);
    int position_errors = 0;
    double max_rel = 0;
    for (int i = 0; i < n; ++i) {
        position_errors += r.x[i] != g.keypoints[3 * i] || r.y[i] != g.keypoints[3 * i + 1];
        const double expected = g.keypoints[3 * i + 2];
        const double rel = std::fabs(r.response[i] - expected) / std::max(1e-30, std::fabs(expected));
        max_rel = std::max(max_rel, expected == 0 ? std::fabs(r.response[i]) : rel);
    }
    std::size_t blur_errors = 0;
    for (std::size_t i = 0; i < r.blurred.size(); ++i) blur_errors += r.blurred[i] != g.blurred[i];
    int exact = 0, max_bits = 0;
    for (int i = 0; i < n; ++i) {
        int bits = 0;
        for (int b = 0; b < 32; ++b) {
            bits += static_cast<int>(std::bitset<8>(r.descriptors[32 * i + b] ^ g.descriptors[32 * i + b]).count());
        }
        exact += bits == 0;
        max_bits = std::max(max_bits, bits);
    }
    std::printf("%s: %d keypoints, position errors %d, max response rel %.3g, blur errors %zu, "
                "descriptors exact %d/%d, max bits %d\n",
                g.name.c_str(), n, position_errors, max_rel, blur_errors, exact, n, max_bits);
    CHECK(position_errors == 0);
    CHECK(max_rel <= 1e-4);
    CHECK(blur_errors == 0);
    CHECK(exact >= 0.99 * n);
    CHECK(max_bits <= 2);

    if (!timings) return;
    cudaEvent_t start, mid, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&mid));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int rep = 0; rep < kWarmup + kRepeats; ++rep) {
        CUDA_CHECK(cudaEventRecord(start, runner.stream));
        runner.launch_s2();
        CUDA_CHECK(cudaEventRecord(mid, runner.stream));
        runner.launch_s3();
        CUDA_CHECK(cudaEventRecord(stop, runner.stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float s2_ms = 0, s3_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&s2_ms, start, mid));
        CUDA_CHECK(cudaEventElapsedTime(&s3_ms, mid, stop));
        std::fprintf(timings, "%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%zu,%zu,%s\n", g.name.c_str(), g.w, g.h,
                     g.threshold, g.k, rep, rep < kWarmup, s2_ms, s3_ms, runner.s2_bytes,
                     runner.s3_bytes, gpu.c_str());
    }
    cudaEventDestroy(start);
    cudaEventDestroy(mid);
    cudaEventDestroy(stop);
}

std::vector<std::uint8_t> pattern_image(int w, int h) {
    std::vector<std::uint8_t> g(static_cast<std::size_t>(w) * h);
    std::uint32_t state = 12345u;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            state = state * 1664525u + 1013904223u;
            const bool square = x % 20 < 11 && y % 16 < 9;  // separated blocks: L corners
            g[static_cast<std::size_t>(y) * w + x] =
                static_cast<std::uint8_t>((square ? 190 : 50) + static_cast<int>(state >> 28));
        }
    }
    return g;
}

void contract_checks() {
    const int w = 160, h = 120;
    const auto gray = pattern_image(w, h);
    Runner runner(gray, w, h, {20, 200, 0.04f});
    const Result first = runner.run();
    CHECK(!first.x.empty());
    CHECK(runner.run() == first);
    for (std::size_t i = 1; i < first.response.size(); ++i) CHECK(first.response[i - 1] >= first.response[i]);
    for (std::size_t i = 0; i < first.x.size(); ++i) {
        CHECK(first.x[i] >= s2::kEdge && first.x[i] < w - s2::kEdge);
        CHECK(first.y[i] >= s2::kEdge && first.y[i] < h - s2::kEdge);
    }

    auto kp = runner.keypoints();
    auto* ws = runner.s2_ws->p;
    const auto bytes = runner.s2_bytes;
    const auto stream = runner.stream;
    CHECK(s2::s2_detect(runner.image.view, runner.config, kp, ws, bytes - 1, stream) ==
          Status::INSUFFICIENT_WORKSPACE);
    CHECK(s2::s2_detect(runner.image.view, runner.config, kp, ws + 1, bytes, stream) ==
          Status::INVALID_ARGUMENT);
    CHECK(s2::s2_detect(runner.image.view, {0, 200, 0.04f}, kp, ws, bytes, stream) ==
          Status::INVALID_ARGUMENT);
    auto small = kp;
    small.capacity = 199;
    CHECK(s2::s2_detect(runner.image.view, runner.config, small, ws, bytes, stream) ==
          Status::INVALID_ARGUMENT);
    auto rgb = runner.image.view;
    rgb.format = PixelFormat::RGB8;
    CHECK(s2::s2_detect(rgb, runner.config, kp, ws, bytes, stream) == Status::INVALID_ARGUMENT);
    auto bad_desc = runner.output();
    bad_desc.dim = 31;
    CHECK(s3::s3_describe(runner.image.view, kp, bad_desc, runner.s3_ws->p, runner.s3_bytes, stream) ==
          Status::INVALID_ARGUMENT);
    bad_desc = runner.output();
    bad_desc.type = DescriptorType::FLOAT32;
    CHECK(s3::s3_describe(runner.image.view, kp, bad_desc, runner.s3_ws->p, runner.s3_bytes, stream) ==
          Status::INVALID_ARGUMENT);
    CHECK(s3::s3_describe(runner.image.view, kp, runner.output(), runner.s3_ws->p,
                          runner.s3_bytes - 1, stream) == Status::INSUFFICIENT_WORKSPACE);

    // K = 0 and images without an interior both report zero keypoints.
    Device<int> count(1);
    CUDA_CHECK(cudaMemset(count.p, 0x7f, sizeof(int)));
    Keypoints none{nullptr, nullptr, nullptr, 0, count.p};
    CHECK(s2::s2_detect(runner.image.view, {20, 0, 0.04f}, none, nullptr, 0, stream) == Status::OK);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(count.download(1)[0] == 0);
    auto thin = runner.image.view;
    thin.height = 2 * s2::kEdge;
    CUDA_CHECK(cudaMemset(count.p, 0x7f, sizeof(int)));
    Keypoints some{runner.x.p, runner.y.p, nullptr, 200, count.p};
    CHECK(s2::s2_detect(thin, runner.config, some, ws, bytes, stream) == Status::OK);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(count.download(1)[0] == 0);
    std::puts("S2/S3 contract checks passed");
}

}  // namespace

int main(int argc, char** argv) {
    bool require_t4 = false;
    std::string timings_path;
    std::vector<std::string> golden;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--require-t4") {
            require_t4 = true;
        } else if (option == "--timings" && i + 1 < argc) {
            timings_path = argv[++i];
        } else if (option == "--golden") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) golden.emplace_back(argv[++i]);
        } else {
            std::fprintf(stderr, "Usage: test_s2_s3_cuda [--require-t4] [--timings out.csv] [--golden case.bin ...]\n");
            return 2;
        }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        if (require_t4) {
            std::fprintf(stderr, "FAIL: --require-t4 needs a CUDA T4 device\n");
            return 1;
        }
        return 77;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU=%s sm_%d%d\n", prop.name, prop.major, prop.minor);
    if (require_t4 && (std::string(prop.name).find("T4") == std::string::npos || prop.major != 7 ||
                       prop.minor != 5)) {
        std::fprintf(stderr, "FAIL: --require-t4 but device is %s\n", prop.name);
        return 1;
    }
    contract_checks();
    FILE* timings = nullptr;
    if (!timings_path.empty()) {
        timings = std::fopen(timings_path.c_str(), "wx");
        CHECK(timings);
        std::fprintf(timings, "case,width,height,fast_threshold,max_keypoints,repeat,is_warmup,"
                              "s2_gpu_ms,s3_gpu_ms,s2_workspace_bytes,s3_workspace_bytes,gpu\n");
    }
    for (const auto& path : golden) check_golden(load_golden(path), timings, prop.name);
    if (timings) std::fclose(timings);
    std::printf("S2/S3 CUDA checks passed (%zu golden cases)\n", golden.size());
}
