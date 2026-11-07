// Device buffers, the S2 -> S3 runner and golden cases shared by
// test_s2_s3_cuda and the A3 budget benchmark (tests/s2_s3/a3_budget_bench.cu).
// Golden files come from `python -m tools.s2_s3.export_golden` and
// `python -m tools.s2_s3.export_timing_golden`.
#pragma once

#include <cuda_runtime.h>

#include "check.hpp"
#include "s2_s3/s2.hpp"
#include "s2_s3/s2_s3_internal.hpp"
#include "s2_s3/s2_s3_marks.hpp"
#include "s2_s3/s3.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (false)

namespace stitch::s2_s3_test {

constexpr int kPadding = 37;  // pitch padding to exercise pitch_bytes

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

// Pitched device image allocated once for max_w x max_h; load() reuses it for
// any smaller image, keeping the pitch.
struct Image {
    Device<std::uint8_t> storage;
    ImageView view;
    Image(int max_w, int max_h)
        : storage(static_cast<std::size_t>(max_w + kPadding) * max_h),
          view{nullptr, max_w, max_h, max_w + kPadding, PixelFormat::GRAY8} {
        view.data = storage.p;
        CUDA_CHECK(cudaMemset(storage.p, 0xa5, storage.n));
    }
    void load(const std::vector<std::uint8_t>& gray, int w, int h) {
        CHECK(w + kPadding <= view.pitch_bytes &&
              static_cast<std::size_t>(view.pitch_bytes) * h <= storage.n);
        view.width = w;
        view.height = h;
        CUDA_CHECK(cudaMemcpy2D(storage.p, view.pitch_bytes, gray.data(), w, w, h,
                                cudaMemcpyHostToDevice));
    }
};

// Allocation bounds of a Runner: image size and keypoint capacity.
struct Sizing {
    int w, h, k;
};

// How S2/S3 see the keypoint buffers (docs/decisions/s2-s3-dynamic-k.md):
// kBounds passes capacity = Sizing::k, kView passes capacity = K of the call
// on the same buffers (the harness fallback).
enum class Capacity { kBounds, kView };

struct Result {
    std::vector<float> x, y, response;
    std::vector<std::uint8_t> descriptors, blurred;
    bool operator==(const Result& o) const {
        return x == o.x && y == o.y && response == o.response &&
               descriptors == o.descriptors && blurred == o.blurred;
    }
};

// Runs S2 then S3 on one stream and downloads everything, including the blur.
// Buffers and workspaces are sized once from `bounds`; load() changes the
// image and K without allocating, as the harness does between attempts.
struct Runner {
    int w, h, k;
    s2::Config config;
    Sizing bounds;
    Capacity capacity;
    Image image;
    Device<float> x, y, response;
    Device<int> count{1};
    Device<std::uint8_t> descriptors;
    std::size_t s2_bytes = 0, s3_bytes = 0;
    Device<std::uint8_t>* s2_ws{};
    Device<std::uint8_t>* s3_ws{};
    cudaStream_t stream{};

    Runner(const std::vector<std::uint8_t>& gray, int width, int height, s2::Config c)
        : Runner(gray, width, height, c, {width, height, c.max_keypoints}) {}
    Runner(const std::vector<std::uint8_t>& gray, int width, int height, s2::Config c,
           Sizing b, Capacity cap = Capacity::kBounds)
        : w(width), h(height), k(c.max_keypoints), config(c), bounds(b), capacity(cap),
          image(b.w, b.h), x(b.k), y(b.k), response(b.k),
          descriptors(static_cast<std::size_t>(b.k) * 48) {
        s2::Config sized = c;
        sized.max_keypoints = b.k;
        CHECK(s2::s2_workspace_bytes(b.w, b.h, sized, &s2_bytes) == Status::OK);
        CHECK(s3::s3_workspace_bytes(b.w, b.h, &s3_bytes) == Status::OK);
        s2_ws = new Device<std::uint8_t>(s2_bytes);
        s3_ws = new Device<std::uint8_t>(s3_bytes);
        CUDA_CHECK(cudaStreamCreate(&stream));
        load(gray, width, height, c.max_keypoints);
    }
    ~Runner() {
        delete s2_ws;
        delete s3_ws;
        cudaStreamDestroy(stream);
    }
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    void load(const std::vector<std::uint8_t>& gray, int width, int height, int max_keypoints) {
        CHECK(width <= bounds.w && height <= bounds.h && max_keypoints <= bounds.k);
        w = width;
        h = height;
        k = max_keypoints;
        config.max_keypoints = max_keypoints;
        image.load(gray, width, height);
    }
    Keypoints keypoints() {
        return {x.p, y.p, response.p, capacity == Capacity::kView ? k : bounds.k, count.p};
    }
    Descriptors output() { return {DescriptorType::BINARY_U8, 32, 48, descriptors.p}; }

    void launch_s2() {
        CHECK(s2::s2_detect(image.view, config, keypoints(), s2_ws->p, s2_bytes, stream) == Status::OK);
    }
    void launch_s3() {
        CHECK(s3::s3_describe(image.view, keypoints(), output(), s3_ws->p, s3_bytes, stream) ==
              Status::OK);
    }
    void launch_s2(const s2_s3::detail::S2Marks& marks) {
        CHECK(s2_s3::detail::s2_detect_marked(image.view, config, keypoints(), s2_ws->p, s2_bytes,
                                              stream, marks) == Status::OK);
    }
    void launch_s3(const s2_s3::detail::S3Marks& marks) {
        CHECK(s2_s3::detail::s3_describe_marked(image.view, keypoints(), output(), s3_ws->p,
                                                s3_bytes, stream, marks) == Status::OK);
    }
    // Downloads what the last launch_s2/launch_s3 pair published.
    Result collect() {
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
        // With capacity 0, S3 returns before blurring: there is no blur to read.
        if (keypoints().capacity == 0) return r;
        s2_s3::detail::S3Layout layout;
        CHECK(s2_s3::detail::make_s3_layout(w, h, &layout));
        r.blurred.resize(static_cast<std::size_t>(w) * h);
        CUDA_CHECK(cudaMemcpy(r.blurred.data(), s3_ws->p + layout.blurred, r.blurred.size(),
                              cudaMemcpyDeviceToHost));
        return r;
    }
    void clear_descriptors() {
        CUDA_CHECK(cudaMemsetAsync(descriptors.p, 0xcd, descriptors.n, stream));
    }
    Result run() {
        clear_descriptors();
        launch_s2();
        launch_s3();
        return collect();
    }
};

struct Golden {
    std::string name;
    int w, h, threshold, k, count;
    float harris_k;
    std::vector<std::uint8_t> gray, blurred, descriptors;
    std::vector<float> keypoints;
};

inline Golden load_golden(const std::string& path) {
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

// Differences between a run and its golden case. Tolerances
// (docs/decisions/s2-s3-orb.md): keypoints and blur exact, response relative
// error <= 1e-4, >= 99 % descriptor rows exact and the rest <= 2 bits.
struct GoldenDiff {
    int count = 0, position_errors = 0, exact = 0, max_bits = 0;
    double max_rel = 0;
    std::size_t blur_errors = 0;
    bool count_matches = false;
    bool passed() const {
        return count_matches && position_errors == 0 && max_rel <= 1e-4 && blur_errors == 0 &&
               exact >= 0.99 * count && max_bits <= 2;
    }
};

inline GoldenDiff golden_diff(const Golden& g, const Result& r) {
    GoldenDiff d;
    d.count = static_cast<int>(r.x.size());
    d.count_matches = d.count == g.count;
    if (!d.count_matches) return d;
    for (int i = 0; i < d.count; ++i) {
        d.position_errors += r.x[i] != g.keypoints[3 * i] || r.y[i] != g.keypoints[3 * i + 1];
        const double expected = g.keypoints[3 * i + 2];
        const double rel = std::fabs(r.response[i] - expected) / std::max(1e-30, std::fabs(expected));
        d.max_rel = std::max(d.max_rel, expected == 0 ? std::fabs(r.response[i]) : rel);
    }
    for (std::size_t i = 0; i < r.blurred.size(); ++i) d.blur_errors += r.blurred[i] != g.blurred[i];
    for (int i = 0; i < d.count; ++i) {
        int bits = 0;
        for (int b = 0; b < 32; ++b) {
            bits += static_cast<int>(std::bitset<8>(r.descriptors[32 * i + b] ^ g.descriptors[32 * i + b]).count());
        }
        d.exact += bits == 0;
        d.max_bits = std::max(d.max_bits, bits);
    }
    return d;
}

inline std::vector<std::uint8_t> pattern_image(int w, int h) {
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

inline std::size_t free_device_bytes() {
    std::size_t free = 0, total = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

}  // namespace stitch::s2_s3_test
