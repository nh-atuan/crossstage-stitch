// Synthetic contract tests and optional independent CPU-reference fixtures.
// Run: test_s6_cuda [--timings results/s6-timings.csv] [--golden fixture.bin ...].
// Options may appear in either order. Exit 77 means no CUDA device.
// Timings are steady-state S6 only, not pipeline runs.csv or end-to-end latency.
#include <cuda_runtime.h>
#include "s6/s6.hpp"
#include "check.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

using namespace stitch;
using namespace stitch::s6;
#define CUDA_CHECK(call) do { const auto error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(1); } } while (false)

namespace {
constexpr std::array<BlendMode, 3> kModes{BlendMode::Feather, BlendMode::Multiband3, BlendMode::Multiband5};
constexpr double kIdentity[9]{1, 0, 0, 0, 1, 0, 0, 0, 1};

struct DeviceImage {
    ImageView view{};
    int channels;
    DeviceImage(int w, int h, int c, int padding = 7) : channels(c) {
        view.width = w; view.height = h; view.pitch_bytes = w * c + padding;
        view.format = c == 3 ? PixelFormat::RGB8 : PixelFormat::GRAY8;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&view.data), static_cast<std::size_t>(view.pitch_bytes) * h));
        CUDA_CHECK(cudaMemset(view.data, 0xa5, static_cast<std::size_t>(view.pitch_bytes) * h));
    }
    ~DeviceImage() { cudaFree(view.data); }
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    void upload(const std::vector<std::uint8_t>& packed) {
        CHECK(packed.size() == static_cast<std::size_t>(view.width) * view.height * channels);
        CUDA_CHECK(cudaMemcpy2D(view.data, view.pitch_bytes, packed.data(), view.width * channels,
                               view.width * channels, view.height, cudaMemcpyHostToDevice));
    }
    std::vector<std::uint8_t> download_and_check_padding() const {
        std::vector<std::uint8_t> storage(static_cast<std::size_t>(view.pitch_bytes) * view.height);
        CUDA_CHECK(cudaMemcpy(storage.data(), view.data, storage.size(), cudaMemcpyDeviceToHost));
        std::vector<std::uint8_t> packed(static_cast<std::size_t>(view.width) * view.height * channels);
        for (int y = 0; y < view.height; ++y) {
            const auto row = storage.data() + static_cast<std::size_t>(y) * view.pitch_bytes;
            std::copy(row, row + view.width * channels, packed.data() + static_cast<std::size_t>(y) * view.width * channels);
            for (int x = view.width * channels; x < view.pitch_bytes; ++x) CHECK(row[x] == 0xa5);
        }
        return packed;
    }
};
struct Workspace {
    void* data = nullptr;
    std::size_t bytes = 0;
    Workspace(const Geometry& g, BlendMode m) {
        CHECK(s6_workspace_bytes(g, m, &bytes) == Status::OK);
        if (bytes) CUDA_CHECK(cudaMalloc(&data, bytes + 256));
    }
    ~Workspace() { if (data) cudaFree(data); }
    Workspace(const Workspace&) = delete;
};
struct Stream {
    cudaStream_t value{};
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
Geometry geometry(int wa, int ha, int wb, int hb, const double* h = kIdentity) {
    Geometry result{};
    CHECK(prepare_geometry(wa, ha, wb, hb, h, GeometryPolicy{}, &result) == Status::OK);
    return result;
}
void execute(const DeviceImage& a, const DeviceImage& b, const Geometry& g, BlendMode mode,
             DeviceImage& out, DeviceImage& mask, Workspace& work, const Stream& stream) {
    CHECK(render(a.view, b.view, g, mode, out.view, mask.view, work.data, work.bytes, stream.value) == Status::OK);
    // Render only reports launch status. The caller must check completion.
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
}

void test_identity() {
    Stream stream;
    for (const auto shape : {std::array<int, 2>{1, 1}, {1, 9}, {11, 1}, {11, 7}, {33, 17}}) {
        const int w = shape[0], h = shape[1];
        const auto g = geometry(w, h, w, h);
        DeviceImage a(w, h, 3), b(w, h, 3), out(w, h, 3), mask(w, h, 1);
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(w) * h * 3);
        for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<std::uint8_t>((i * 37) % 256);
        pixels[0] = pixels[1] = pixels[2] = 0;  // black is valid, not missing.
        a.upload(pixels); b.upload(pixels);
        for (auto mode : kModes) {
            Workspace work(g, mode);
            execute(a, b, g, mode, out, mask, work, stream);
            const auto rendered = out.download_and_check_padding();
            const auto validity = mask.download_and_check_padding();
            for (std::size_t i = 0; i < pixels.size(); ++i) CHECK(std::abs(int(rendered[i]) - int(pixels[i])) <= 1);
            for (auto v : validity) CHECK(v == 1);
        }
    }
}
void test_full_overlap_ownership() {
    Stream stream;
    const auto g = geometry(9, 7, 9, 7);
    DeviceImage a(9, 7, 3), b(9, 7, 3), out(9, 7, 3), mask(9, 7, 1);
    a.upload(std::vector<std::uint8_t>(9 * 7 * 3, 20));
    b.upload(std::vector<std::uint8_t>(9 * 7 * 3, 200));
    for (auto mode : kModes) {
        Workspace work(g, mode);
        execute(a, b, g, mode, out, mask, work, stream);
        const auto rendered = out.download_and_check_padding();
        const int expected = mode == BlendMode::Feather ? 110 : 20;
        for (auto c : rendered) CHECK(std::abs(int(c) - expected) <= 1);
    }
}
void test_negative_origin() {
    Stream stream;
    const double h[9]{1, 0, -2, 0, 1, -1, 0, 0, 1};
    const auto g = geometry(5, 3, 5, 3, h);
    CHECK(g.width == 7 && g.height == 4 && g.origin_x == -2 && g.origin_y == -1);
    DeviceImage a(5, 3, 3), b(5, 3, 3), out(g.width, g.height, 3), mask(g.width, g.height, 1);
    a.upload(std::vector<std::uint8_t>(5 * 3 * 3, 20));
    b.upload(std::vector<std::uint8_t>(5 * 3 * 3, 200));
    for (auto mode : kModes) {
        Workspace work(g, mode);
        execute(a, b, g, mode, out, mask, work, stream);
        const auto rendered = out.download_and_check_padding();
        const auto validity = mask.download_and_check_padding();
        for (int y = 0; y < g.height; ++y) for (int x = 0; x < g.width; ++x) {
            const int xa = x + g.origin_x, ya = y + g.origin_y;
            const bool ma = xa >= 0 && xa < 5 && ya >= 0 && ya < 3;
            const bool mb = xa >= -2 && xa < 3 && ya >= -1 && ya < 2;
            const auto p = static_cast<std::size_t>(y) * g.width + x;
            CHECK(validity[p] == (ma || mb ? 1 : 0));
            if (!(ma && mb)) {
                const int expected = ma ? 20 : (mb ? 200 : 0);
                for (int c = 0; c < 3; ++c) CHECK(rendered[3 * p + c] == expected);
            }
        }
    }
}
void test_validation() {
    const auto g = geometry(5, 3, 5, 3);
    DeviceImage a(5, 3, 3), b(5, 3, 3), out(5, 3, 3), mask(5, 3, 1);
    Stream stream;
    std::size_t bytes = 123;
    CHECK(s6_workspace_bytes(g, BlendMode::Feather, &bytes) == Status::OK && bytes == 0);
    CHECK(s6_workspace_bytes(g, static_cast<BlendMode>(17), &bytes) == Status::INVALID_ARGUMENT);
    CHECK(s6_workspace_bytes(g, BlendMode::Feather, nullptr) == Status::INVALID_ARGUMENT);
    auto broken = g; broken.inverse[0] = std::numeric_limits<double>::quiet_NaN();
    CHECK(s6_workspace_bytes(broken, BlendMode::Feather, &bytes) == Status::INVALID_ARGUMENT);
    for (auto mode : {BlendMode::Multiband3, BlendMode::Multiband5}) {
        Workspace work(g, mode);
        CHECK(work.bytes > 0);
        CHECK(render(a.view, b.view, g, mode, out.view, mask.view, work.data, work.bytes - 1, stream.value) == Status::INSUFFICIENT_WORKSPACE);
        CHECK(render(a.view, b.view, g, mode, out.view, mask.view, nullptr, work.bytes, stream.value) == Status::INSUFFICIENT_WORKSPACE);
        CHECK(render(a.view, b.view, g, mode, out.view, mask.view, static_cast<std::uint8_t*>(work.data) + 1, work.bytes, stream.value) == Status::INVALID_ARGUMENT);
    }
    CHECK(render(a.view, b.view, g, BlendMode::Feather, a.view, mask.view, nullptr, 0, stream.value) == Status::INVALID_ARGUMENT);
    auto bad_out = out.view; bad_out.pitch_bytes = 14;
    CHECK(render(a.view, b.view, g, BlendMode::Feather, bad_out, mask.view, nullptr, 0, stream.value) == Status::INVALID_ARGUMENT);
    bad_out = out.view; bad_out.data = nullptr;
    CHECK(render(a.view, b.view, g, BlendMode::Feather, bad_out, mask.view, nullptr, 0, stream.value) == Status::INVALID_ARGUMENT);
    auto bad_mask = mask.view; bad_mask.data = out.view.data;
    CHECK(render(a.view, b.view, g, BlendMode::Feather, out.view, bad_mask, nullptr, 0, stream.value) == Status::INVALID_ARGUMENT);
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
}

// Events are allocated outside the timed region and destroyed on normal scope exit.
struct Event {
    cudaEvent_t value{};
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
std::string csv_field(const std::string& value) {
    std::string escaped = "\"";
    for (char c : value) { if (c == '"') escaped += '"'; escaped += c; }
    return escaped + '"';
}
const char* mode_name(BlendMode mode) {
    return mode == BlendMode::Feather ? "feather" :
           (mode == BlendMode::Multiband3 ? "multiband_3" : "multiband_5");
}
void profile(const char* fixture, const DeviceImage& a, const DeviceImage& b,
             const Geometry& g, BlendMode mode, DeviceImage& output,
             DeviceImage& mask, Workspace& workspace, const Stream& stream,
             const std::string& gpu_name, std::ostream& csv) {
    Event start, stop;
    // Caller-owned device inputs/output/workspace and prepared geometry persist.
    // All image upload, allocation, geometry setup, golden validation/download,
    // event allocation and CSV I/O are excluded. Every render recomputes the
    // selected branch from scratch; there is no cross-call S6 cache.
    // gpu_ms: CUDA events around all S6 kernels on the supplied stream.
    // host_ms: chrono around render, event recording and checked completion;
    // this includes launch/wait overhead but excludes any pipeline/controller.
    for (int iteration = 0; iteration < 12; ++iteration) {
        const bool warmup = iteration < 2;
        const auto before = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(start.value, stream.value));
        CHECK(render(a.view, b.view, g, mode, output.view, mask.view,
                     workspace.data, workspace.bytes, stream.value) == Status::OK);
        CUDA_CHECK(cudaEventRecord(stop.value, stream.value));
        CUDA_CHECK(cudaEventSynchronize(stop.value));
        const auto after = std::chrono::steady_clock::now();
        float gpu_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start.value, stop.value));
        const double host_ms = std::chrono::duration<double, std::milli>(after - before).count();
        csv << csv_field(fixture) << ',' << mode_name(mode) << ',' << g.width << ',' << g.height
            << ',' << (warmup ? iteration : iteration - 2) << ',' << (warmup ? 1 : 0)
            << ',' << std::setprecision(9) << gpu_ms << ',' << host_ms << ','
            << workspace.bytes << ',' << csv_field(gpu_name) << '\n';
        CHECK(csv.good());
    }
}

// S6GOLD01, int32 LE[9]: wa,ha,wb,hb,wc,hc,origin_x,origin_y,mode;
// double LE[9] H B->A; packedRGB A, packedRGB B, expectedRGB, expected mask.
// This host fixture loader supports little-endian machines with IEEE FP64.
void golden(const char* path, std::ostream* timings, const std::string& gpu_name) {
    const std::uint32_t endian = 1;
    CHECK(*reinterpret_cast<const std::uint8_t*>(&endian) == 1);
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    std::ifstream file(path, std::ios::binary);
    CHECK(file.good());
    auto read = [&](void* data, std::size_t bytes) {
        file.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes)); CHECK(file.good());
    };
    char magic[8]; read(magic, 8); CHECK(std::memcmp(magic, "S6GOLD01", 8) == 0);
    std::int32_t metadata[9]; read(metadata, sizeof(metadata));
    for (int i = 0; i < 6; ++i) CHECK(metadata[i] > 0);
    auto area = [](int w, int h) {
        const auto n = static_cast<std::size_t>(w) * h;
        CHECK(n <= 16777216); return n;
    };
    const auto na = area(metadata[0], metadata[1]);
    const auto nb = area(metadata[2], metadata[3]);
    const auto nc = area(metadata[4], metadata[5]);
    double h[9]; read(h, sizeof(h));
    const auto g = geometry(metadata[0], metadata[1], metadata[2], metadata[3], h);
    CHECK(g.width == metadata[4] && g.height == metadata[5] && g.origin_x == metadata[6] && g.origin_y == metadata[7]);
    CHECK(metadata[8] >= 0 && metadata[8] <= 2);
    const auto mode = static_cast<BlendMode>(metadata[8]);
    std::vector<std::uint8_t> pa(na * 3), pb(nb * 3), expected(nc * 3), expected_mask(nc);
    read(pa.data(), pa.size()); read(pb.data(), pb.size());
    read(expected.data(), expected.size()); read(expected_mask.data(), expected_mask.size());
    CHECK(file.peek() == std::ifstream::traits_type::eof());
    DeviceImage a(g.width_a, g.height_a, 3), b(g.width_b, g.height_b, 3);
    DeviceImage out(g.width, g.height, 3), mask(g.width, g.height, 1);
    a.upload(pa); b.upload(pb);
    Workspace work(g, mode); Stream stream;
    execute(a, b, g, mode, out, mask, work, stream);
    const auto actual = out.download_and_check_padding();
    const auto actual_mask = mask.download_and_check_padding();
    CHECK(actual_mask == expected_mask);
    int worst = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        worst = std::max(worst, std::abs(int(actual[i]) - int(expected[i])));
    std::printf("%s: mode=%d RGB max_abs=%d mask exact\n", path, metadata[8], worst);
    CHECK(worst <= 2);  // preregistered implementation tolerance, not QoR.
    if (timings) profile(path, a, b, g, mode, out, mask, work, stream, gpu_name, *timings);
}
}  // namespace

// Validate paths before opening the output so it can never truncate a fixture.
// Parent directories must already exist; no directory creation is hidden here.
struct Options {
    std::vector<std::string> fixtures;
    std::string timings;
};
Options parse_options(int argc, char** argv) {
    Options options;
    auto clean_path = [](const std::string& value) {
        CHECK(!value.empty());
        for (unsigned char c : value) CHECK(c >= 32 && c != 127);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--timings") {
            CHECK(options.timings.empty() && i + 1 < argc);
            options.timings = argv[++i]; clean_path(options.timings);
            CHECK(options.timings.rfind("--", 0) != 0);
        } else if (option == "--golden") {
            const auto count_before = options.fixtures.size();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                options.fixtures.emplace_back(argv[++i]); clean_path(options.fixtures.back());
            }
            CHECK(options.fixtures.size() > count_before);
        } else {
            std::fprintf(stderr, "Unknown option: %s\nUsage: test_s6_cuda [--timings output.csv] [--golden fixture.bin ...]\n", argv[i]);
            CHECK(false);
        }
    }
    namespace fs = std::filesystem;
    for (const auto& fixture : options.fixtures) CHECK(fs::is_regular_file(fs::path(fixture)));
    if (!options.timings.empty()) {
        CHECK(!options.fixtures.empty());
        const fs::path output = fs::absolute(options.timings);
        CHECK(fs::is_directory(output.parent_path()));
        CHECK(!fs::exists(output) || fs::is_regular_file(output));
        for (const auto& fixture : options.fixtures) {
            const fs::path input = fs::absolute(fixture);
            CHECK(fs::weakly_canonical(output) != fs::weakly_canonical(input));
            if (fs::exists(output)) CHECK(!fs::equivalent(output, input));
        }
    }
    return options;
}

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    int devices = 0;
    const auto device_status = cudaGetDeviceCount(&devices);
    if (device_status != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "SKIP: CUDA device unavailable (%s)\n", cudaGetErrorString(device_status));
        return 77;
    }
    int device = 0;
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device));
    std::ofstream timings;
    if (!options.timings.empty()) {
        timings.open(options.timings, std::ios::out | std::ios::trunc);
        CHECK(timings.good());
        timings << "fixture,blend_mode,width,height,repeat_idx,is_warmup,gpu_ms,host_ms,workspace_bytes,gpu_name\n";
        CHECK(timings.good());
    }
    test_identity(); test_full_overlap_ownership(); test_negative_origin(); test_validation();
    for (const auto& fixture : options.fixtures)
        golden(fixture.c_str(), options.timings.empty() ? nullptr : &timings, device_properties.name);
    if (timings.is_open()) { timings.flush(); CHECK(timings.good()); }
    std::puts("S6 CUDA synthetic/fixture checks passed");
    return 0;
}
