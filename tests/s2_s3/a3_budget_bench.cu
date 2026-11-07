// Ablation A3 benchmark: S2/S3 cost under a per-attempt keypoint budget K
// (docs/experiments/ablation-studies.md, A3; docs/decisions/s2-s3-a3-measurement.md).
//
// Run: s2_s3_a3_bench [--require-t4] [--k-max N] --out a3.csv
//          [--jitter-out jitter.csv] [--jitter-images N] [--jitter-attempts N]
//          --golden <parent>_<stem>_s<S>_k<K>.bin ...
//      s2_s3_a3_bench --self-test
// Golden cases come from `python -m tools.s2_s3.export_timing_golden`; cases
// of one image and scale form a group whose K values are cycled, so K changes
// on every attempt as it does under controller escalation.
//
// Modes, interleaved on one stream inside each attempt:
//   view    buffers sized once for (W_max, H_max, K_max), Keypoints.capacity = K
//   max     the same buffers, capacity = K_max (what the harness passes today)
//   realloc cudaFree + cudaMalloc of buffers sized exactly for (W, H, K)
// Each attempt runs S2+S3 twice: without marks (s2_ms, s3_ms, host time) and
// with the internal phase marks (t_fast ... t_describe). Every mode, with and
// without marks, must match the golden case before anything is timed.
// One image per call: this is not the two-image t_s2/t_s3 boundary of
// docs/conventions.md. Exit 77 means no CUDA device.
#include "s2_s3_cuda_support.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace stitch;
using namespace stitch::s2_s3_test;
using s2_s3::detail::S2Marks;
using s2_s3::detail::S3Marks;

namespace {

constexpr int kWarmup = 2;
constexpr int kRepeats = 10;
constexpr int kDescriptorStride = 32;  // one row per keypoint, as the harness allocates
constexpr int kFastThreshold = 20;
constexpr float kHarrisK = 0.04f;

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Keypoints, descriptors and both workspaces for images up to w x h and K <= k.
struct Buffers {
    Sizing size;
    float *x{}, *y{}, *response{};
    int* count{};
    std::uint8_t* descriptors{};
    std::uint8_t *s2_ws{}, *s3_ws{};
    std::size_t s2_bytes = 0, s3_bytes = 0;

    explicit Buffers(Sizing s) : size(s) {
        CHECK(s2::s2_workspace_bytes(s.w, s.h, {kFastThreshold, s.k, kHarrisK}, &s2_bytes) == Status::OK);
        CHECK(s3::s3_workspace_bytes(s.w, s.h, &s3_bytes) == Status::OK);
        const auto k = static_cast<std::size_t>(std::max(s.k, 1));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&x), k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&y), k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&response), k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&count), sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&descriptors), k * kDescriptorStride));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s2_ws), std::max<std::size_t>(s2_bytes, 1)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s3_ws), std::max<std::size_t>(s3_bytes, 1)));
    }
    ~Buffers() {
        for (void* p : {static_cast<void*>(x), static_cast<void*>(y), static_cast<void*>(response),
                        static_cast<void*>(count), static_cast<void*>(descriptors),
                        static_cast<void*>(s2_ws), static_cast<void*>(s3_ws)}) {
            cudaFree(p);
        }
    }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;

    std::size_t keypoint_bytes() const { return 3 * sizeof(float) * size.k + sizeof(int); }
    std::size_t descriptor_bytes() const { return static_cast<std::size_t>(size.k) * kDescriptorStride; }
    Keypoints keypoints(int capacity) const { return {x, y, response, capacity, count}; }
    Descriptors output() const { return {DescriptorType::BINARY_U8, 32, kDescriptorStride, descriptors}; }
};

enum class Mode { kView, kMax, kRealloc };
constexpr Mode kModes[] = {Mode::kView, Mode::kMax, Mode::kRealloc};

const char* mode_name(Mode m) {
    return m == Mode::kView ? "view" : m == Mode::kMax ? "max" : "realloc";
}

struct Events {
    cudaEvent_t start{}, fast{}, select{}, harris{}, mid{}, blur{}, stop{};
    Events() {
        for (cudaEvent_t* event : {&start, &fast, &select, &harris, &mid, &blur, &stop}) {
            CUDA_CHECK(cudaEventCreate(event));
        }
    }
    ~Events() {
        for (cudaEvent_t event : {start, fast, select, harris, mid, blur, stop}) cudaEventDestroy(event);
    }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
};

float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

struct Timing {
    double s2_ms = 0, s3_ms = 0, host_attempt_ms = 0, alloc_ms = 0;
    double fast = 0, select = 0, harris = 0, rank = 0, blur = 0, describe = 0;
};

// One image on the device and the buffers each mode uses for it.
struct Bench {
    Sizing bounds;
    Image image;
    Buffers shared;                    // view and max
    std::unique_ptr<Buffers> exact;    // realloc, replaced on every attempt
    Events events;
    cudaStream_t stream{};
    int w = 0, h = 0;

    explicit Bench(Sizing b) : bounds(b), image(b.w, b.h), shared(b) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    ~Bench() {
        exact.reset();
        cudaStreamDestroy(stream);
    }

    void load(const std::vector<std::uint8_t>& gray, int width, int height) {
        CHECK(width <= bounds.w && height <= bounds.h);
        w = width;
        h = height;
        image.load(gray, width, height);
    }
    const Buffers& buffers(Mode m) const { return m == Mode::kRealloc ? *exact : shared; }
    int capacity(Mode m, int k) const { return m == Mode::kMax ? bounds.k : k; }

    void launch(Mode m, int k, bool marked) {
        const Buffers& b = buffers(m);
        const s2::Config config{kFastThreshold, k, kHarrisK};
        const Keypoints kp = b.keypoints(capacity(m, k));
        CUDA_CHECK(cudaEventRecord(events.start, stream));
        if (marked) {
            CHECK(s2_s3::detail::s2_detect_marked(image.view, config, kp, b.s2_ws, b.s2_bytes, stream,
                                                  S2Marks{events.fast, events.select, events.harris}) ==
                  Status::OK);
        } else {
            CHECK(s2::s2_detect(image.view, config, kp, b.s2_ws, b.s2_bytes, stream) == Status::OK);
        }
        CUDA_CHECK(cudaEventRecord(events.mid, stream));
        if (marked) {
            CHECK(s2_s3::detail::s3_describe_marked(image.view, kp, b.output(), b.s3_ws, b.s3_bytes,
                                                    stream, S3Marks{events.blur}) == Status::OK);
        } else {
            CHECK(s3::s3_describe(image.view, kp, b.output(), b.s3_ws, b.s3_bytes, stream) == Status::OK);
        }
        CUDA_CHECK(cudaEventRecord(events.stop, stream));
    }

    // One attempt: (re)allocation for realloc, the unmarked run, then the marked run.
    Timing attempt(Mode m, int k) {
        Timing t;
        const auto start = Clock::now();
        if (m == Mode::kRealloc) {
            exact.reset();
            exact = std::make_unique<Buffers>(Sizing{w, h, k});
            t.alloc_ms = ms_since(start);
        }
        launch(m, k, false);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        t.host_attempt_ms = ms_since(start);
        t.s2_ms = elapsed(events.start, events.mid);
        t.s3_ms = elapsed(events.mid, events.stop);

        launch(m, k, true);
        CUDA_CHECK(cudaEventSynchronize(events.stop));
        t.fast = elapsed(events.start, events.fast);
        t.select = elapsed(events.fast, events.select);
        t.harris = elapsed(events.select, events.harris);
        t.rank = elapsed(events.harris, events.mid);
        t.blur = elapsed(events.mid, events.blur);
        t.describe = elapsed(events.blur, events.stop);
        return t;
    }

    // Output of the last launch (descriptor rows packed to 32 bytes, blur included).
    Result collect(Mode m, int k) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const Buffers& b = buffers(m);
        int n = 0;
        CUDA_CHECK(cudaMemcpy(&n, b.count, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(n >= 0 && n <= k);
        Result r;
        r.x.resize(n);
        r.y.resize(n);
        r.response.resize(n);
        r.descriptors.resize(static_cast<std::size_t>(n) * 32);
        if (n) {
            CUDA_CHECK(cudaMemcpy(r.x.data(), b.x, n * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(r.y.data(), b.y, n * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(r.response.data(), b.response, n * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(r.descriptors.data(), b.descriptors, r.descriptors.size(),
                                  cudaMemcpyDeviceToHost));
        }
        s2_s3::detail::S3Layout layout;
        CHECK(s2_s3::detail::make_s3_layout(w, h, &layout));
        r.blurred.resize(static_cast<std::size_t>(w) * h);
        CUDA_CHECK(cudaMemcpy(r.blurred.data(), b.s3_ws + layout.blurred, r.blurred.size(),
                              cudaMemcpyDeviceToHost));
        return r;
    }

    // FAST maxima kept for Harris ranking (min(2K, maxima)) by the last S2 call.
    int maxima_selected(Mode m, int k) {
        s2_s3::detail::S2Layout layout;
        CHECK(s2_s3::detail::make_s2_layout(w, h, k, &layout));
        s2_s3::detail::SelectParams params{};
        CUDA_CHECK(cudaMemcpy(&params, buffers(m).s2_ws + layout.params, sizeof(params),
                              cudaMemcpyDeviceToHost));
        return static_cast<int>(params.selected);
    }

    // Runs every mode with and without marks; all must agree bit for bit.
    // Returns the unmarked view result and the selected candidate count.
    std::pair<Result, int> verify(int k) {
        Result reference;
        int selected = -1;
        for (Mode m : kModes) {
            if (m == Mode::kRealloc) exact = std::make_unique<Buffers>(Sizing{w, h, k});
            for (bool marked : {false, true}) {
                launch(m, k, marked);
                const Result r = collect(m, k);
                const int s = maxima_selected(m, k);
                if (selected < 0) {
                    reference = r;
                    selected = s;
                }
                if (!(r == reference) || s != selected) {
                    std::fprintf(stderr, "A3: %dx%d K=%d mode %s marked %d differs from view\n", w, h, k,
                                 mode_name(m), marked);
                }
                CHECK(r == reference);
                CHECK(s == selected);
            }
        }
        return {reference, selected};
    }
};

struct Case {
    Golden golden;
    std::string stem, image, scale;
    int count = 0, selected = 0;
};

// <parent>_<stem>_s<S>_k<K>.bin -> stem, image and scale; false if the name has another shape.
bool parse_case_name(const std::string& path, std::string* stem, std::string* image, std::string* scale) {
    const auto slash = path.find_last_of("/\\");
    std::string name = path.substr(slash == std::string::npos ? 0 : slash + 1);
    if (name.size() < 4 || name.compare(name.size() - 4, 4, ".bin") != 0) return false;
    name.resize(name.size() - 4);
    const auto k = name.rfind("_k");
    if (k == std::string::npos) return false;
    const auto s = name.rfind("_s", k);
    if (s == std::string::npos || s == 0 || s + 2 >= k) return false;
    *stem = name;
    *image = name.substr(0, s);
    *scale = name.substr(s + 2, k - s - 2);
    return true;
}

const char* kHeader =
    "case,image,scale,width,height,k,k_max,mode,capacity,repeat,is_warmup,count,maxima_selected,"
    "s2_ms,s3_ms,t_fast,t_select,t_harris,t_rank,t_blur,t_describe,host_attempt_ms,alloc_ms,"
    "s2_ws_bytes,s3_ws_bytes,kp_bytes,desc_bytes,gpu\n";

void write_row(FILE* out, const Case& c, const Bench& bench, Mode m, int rep, const Timing& t,
               const std::string& gpu) {
    const Buffers& b = bench.buffers(m);
    const Golden& g = c.golden;
    std::fprintf(out,
                 "%s,%s,%s,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                 "%.6f,%.6f,%zu,%zu,%zu,%zu,%s\n",
                 c.stem.c_str(), c.image.c_str(), c.scale.c_str(), g.w, g.h, g.k, bench.bounds.k,
                 mode_name(m), bench.capacity(m, g.k), rep, rep < kWarmup, c.count, c.selected, t.s2_ms,
                 t.s3_ms, t.fast, t.select, t.harris, t.rank, t.blur, t.describe, t.host_attempt_ms,
                 t.alloc_ms, b.s2_bytes, b.s3_bytes, b.keypoint_bytes(), b.descriptor_bytes(), gpu.c_str());
}

using Groups = std::map<std::pair<std::string, std::string>, std::vector<Case>>;

// Checks every case against its golden file and against all modes, then times
// each group: per repeat, every K of the group in turn, every mode in turn.
void run_groups(Groups& groups, Sizing bounds, FILE* out, const std::string& gpu) {
    Bench bench(bounds);
    for (auto& [key, cases] : groups) {
        bench.load(cases.front().golden.gray, cases.front().golden.w, cases.front().golden.h);
        for (Case& c : cases) {
            const Golden& g = c.golden;
            CHECK(g.w == bench.w && g.h == bench.h && g.gray == cases.front().golden.gray);
            CHECK(g.threshold == kFastThreshold && g.harris_k == kHarrisK);
            auto [result, selected] = bench.verify(g.k);
            const GoldenDiff d = golden_diff(g, result);
            if (!d.passed()) std::fprintf(stderr, "A3: %s does not match its golden case\n", g.name.c_str());
            CHECK(d.passed());
            c.count = d.count;
            c.selected = selected;
        }
        for (int rep = 0; rep < kWarmup + kRepeats; ++rep) {
            for (const Case& c : cases) {
                for (Mode m : kModes) write_row(out, c, bench, m, rep, bench.attempt(m, c.golden.k), gpu);
            }
        }
        std::printf("A3 %s at S=%s: %dx%d, %zu budgets timed\n", key.first.c_str(), key.second.c_str(),
                    bench.w, bench.h, cases.size());
    }
}

// Long attempt sequences on a few images: host-time jitter per mode and a
// free-memory check showing that view and max allocate nothing per attempt.
void run_jitter(Groups& groups, Sizing bounds, FILE* out, int images, int attempts,
                const std::string& gpu) {
    std::fprintf(out, "image,scale,width,height,mode,attempt,k,s2_ms,s3_ms,host_attempt_ms,alloc_ms,gpu\n");
    Bench bench(bounds);
    int done = 0;
    for (auto& [key, cases] : groups) {
        if (done++ == images) break;
        bench.load(cases.front().golden.gray, cases.front().golden.w, cases.front().golden.h);
        for (Mode m : kModes) {
            const std::size_t before = free_device_bytes();
            for (int a = 0; a < attempts; ++a) {
                const int k = cases[static_cast<std::size_t>(a) % cases.size()].golden.k;
                const Timing t = bench.attempt(m, k);
                std::fprintf(out, "%s,%s,%d,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%s\n", key.first.c_str(),
                             key.second.c_str(), bench.w, bench.h, mode_name(m), a, k, t.s2_ms, t.s3_ms,
                             t.host_attempt_ms, t.alloc_ms, gpu.c_str());
            }
            bench.exact.reset();
            const std::size_t after = free_device_bytes();
            std::printf("A3 jitter %s S=%s %s: free bytes %zu -> %zu\n", key.first.c_str(), key.second.c_str(),
                        mode_name(m), before, after);
            if (m != Mode::kRealloc) CHECK(before == after);
        }
    }
}

// CTest smoke check without golden files: on a synthetic image, every mode
// with and without marks agrees, phase times are finite and non-negative, and
// repeated view/max attempts leave free device memory unchanged.
int self_test() {
    const int w = 320, h = 240;
    const auto gray = pattern_image(w, h);
    Bench bench({w, h, 400});
    bench.load(gray, w, h);
    for (int k : {50, 100, 200, 400}) {
        const auto [result, selected] = bench.verify(k);
        CHECK(!result.x.empty() && selected >= static_cast<int>(result.x.size()));
        const Timing t = bench.attempt(Mode::kView, k);
        for (double v : {t.s2_ms, t.s3_ms, t.fast, t.select, t.harris, t.rank, t.blur, t.describe}) {
            CHECK(std::isfinite(v) && v >= 0);
        }
    }
    for (Mode m : {Mode::kView, Mode::kMax}) {
        const std::size_t before = free_device_bytes();
        for (int a = 0; a < 50; ++a) bench.attempt(m, a % 2 ? 100 : 400);
        CHECK(free_device_bytes() == before);
    }
    std::puts("A3 benchmark self-test passed");
    return 0;
}

int usage() {
    std::fprintf(stderr, "Usage: s2_s3_a3_bench [--require-t4] [--k-max N] --out a3.csv [--jitter-out j.csv] "
                         "[--jitter-images N] [--jitter-attempts N] --golden case.bin ...\n"
                         "       s2_s3_a3_bench --self-test\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    bool require_t4 = false, self = false;
    int k_max = 4000, jitter_images = 4, jitter_attempts = 1000;
    std::string out_path, jitter_path;
    std::vector<std::string> golden;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--require-t4") {
            require_t4 = true;
        } else if (option == "--self-test") {
            self = true;
        } else if (option == "--k-max" && i + 1 < argc) {
            k_max = std::atoi(argv[++i]);
        } else if (option == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (option == "--jitter-out" && i + 1 < argc) {
            jitter_path = argv[++i];
        } else if (option == "--jitter-images" && i + 1 < argc) {
            jitter_images = std::atoi(argv[++i]);
        } else if (option == "--jitter-attempts" && i + 1 < argc) {
            jitter_attempts = std::atoi(argv[++i]);
        } else if (option == "--golden") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) golden.emplace_back(argv[++i]);
        } else {
            return usage();
        }
    }
    if (!self && (out_path.empty() || golden.empty() || k_max <= 0 || jitter_images < 0 ||
                  jitter_attempts < 0)) {
        return usage();
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
    if (self) return self_test();

    Groups groups;
    Sizing bounds{0, 0, k_max};
    for (const auto& path : golden) {
        Case c;
        if (!parse_case_name(path, &c.stem, &c.image, &c.scale)) {
            std::fprintf(stderr, "A3: %s is not named <image>_s<S>_k<K>.bin\n", path.c_str());
            return 2;
        }
        c.golden = load_golden(path);
        if (c.golden.k > k_max) {
            std::fprintf(stderr, "A3: %s has K=%d > --k-max %d\n", path.c_str(), c.golden.k, k_max);
            return 2;
        }
        bounds.w = std::max(bounds.w, c.golden.w);
        bounds.h = std::max(bounds.h, c.golden.h);
        groups[{c.image, c.scale}].push_back(std::move(c));
    }
    for (auto& [key, cases] : groups) {
        std::sort(cases.begin(), cases.end(), [](const Case& a, const Case& b) { return a.golden.k < b.golden.k; });
    }
    std::printf("A3: %zu cases in %zu image/scale groups, bounds %dx%d, K_max %d\n", golden.size(),
                groups.size(), bounds.w, bounds.h, bounds.k);

    FILE* out = std::fopen(out_path.c_str(), "wx");
    CHECK(out);
    std::fputs(kHeader, out);
    run_groups(groups, bounds, out, prop.name);
    std::fclose(out);
    if (!jitter_path.empty()) {
        FILE* jitter = std::fopen(jitter_path.c_str(), "wx");
        CHECK(jitter);
        run_jitter(groups, bounds, jitter, jitter_images, jitter_attempts, prop.name);
        std::fclose(jitter);
    }
    std::puts("A3 benchmark finished");
    return 0;
}
