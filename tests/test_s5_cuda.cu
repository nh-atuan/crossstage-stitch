#include <cuda_runtime.h>
#include "check.hpp"
#include "common/geometry.hpp"
#include "s5/s5.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace stitch;
using namespace stitch::s5;

#define CUDA_CHECK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (false)

namespace {
template <class T> struct Device {
    T* p{};
    std::size_t n;
    explicit Device(std::size_t count) : n(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), (n ? n : 1) * sizeof(T)));
    }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& data) {
        CHECK(data.size() <= n);
        if (!data.empty()) CUDA_CHECK(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> data(n);
        if (n) CUDA_CHECK(cudaMemcpy(data.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return data;
    }
};

struct Fixture {
    int n;
    Device<float> ax, ay, bx, by;
    Device<int> na{1}, nb{1}, nm{1};
    Device<int32_t> ia, ib;
    Device<std::uint8_t> mask;
    Device<std::uint8_t> workspace;
    HomographyResult* out{};
    cudaStream_t stream{};
    Keypoints a{}, b{};
    Matches m{};
    ScaleInfo sa{}, sb{};
    Config config{257, 0, 32, 2, 0.5, 20261022, 1};

    static std::size_t workspace_size() {
        const Config c{257, 0, 32, 2, 0.5, 20261022, 1};
        std::size_t bytes{};
        CHECK(s5_workspace_bytes(c, &bytes) == Status::OK);
        return bytes;
    }
    explicit Fixture(int count = 80)
        : n(count), ax(n), ay(n), bx(n), by(n), ia(n), ib(n), mask(n), workspace(workspace_size()) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&out), sizeof(*out)));
        na.upload({n}); nb.upload({n}); nm.upload({n});
        a = {ax.p, ay.p, nullptr, n, na.p};
        b = {bx.p, by.p, nullptr, n, nb.p};
        m = {ia.p, ib.p, nullptr, n, nm.p};
        CHECK(make_scale_info(640, 480, 1, &sa) == Status::OK);
        CHECK(make_scale_info(640, 480, 1, &sb) == Status::OK);
        set_points(false, false);
    }
    ~Fixture() { cudaFreeHost(out); cudaStreamDestroy(stream); }
    Fixture(const Fixture&) = delete;

    void set_points(bool outliers, bool line) {
        std::vector<float> x(n), y(n), u(n), v(n);
        std::vector<int32_t> indices(n);
        for (int i = 0; i < n; ++i) {
            x[i] = 20 + (i % 10) * 50;
            y[i] = line ? 2 * x[i] : 20 + (i / 10) * 45;
            const double z = 0.0002 * x[i] - 0.0001 * y[i] + 1;
            u[i] = static_cast<float>((1.02 * x[i] + 0.03 * y[i] + 11) / z);
            v[i] = static_cast<float>((-0.02 * x[i] + 0.98 * y[i] + 7) / z);
            if (outliers && i % 4 == 0) { u[i] = 600 - i; v[i] = 400 - i * 2; }
            indices[i] = i;
        }
        bx.upload(x); by.upload(y); ax.upload(u); ay.upload(v);
        ia.upload(indices); ib.upload(indices);
    }
    HomographyResult run() {
        // Uploads use the default stream; finish them before our nonblocking stream.
        CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(estimate(a, b, m, sa, sb, config, mask.p, out, workspace.p, workspace.n, stream) == Status::OK);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return *out;
    }
};

void test_known_h_and_outliers() {
    // Given: a projective warp with 25% gross mismatches.
    Fixture f;
    f.set_points(true, false);
    // When: running the CUDA RANSAC and refinement path.
    const auto r = f.run();
    // Then: known H, inliers and telemetry are recovered in the B -> A direction.
    const double truth[9]{1.02, 0.03, 11, -0.02, 0.98, 7, 0.0002, -0.0001, 1};
    CHECK(r.status == Status::OK);
    CHECK(r.num_matches == 80 && r.num_inliers == 60);
    CHECK_NEAR(r.inlier_ratio, 0.75, 1e-12);
    CHECK(r.inlier_coverage > 0 && r.inlier_coverage <= 1);
    CHECK(r.reproj_residual < 1e-3);
    CHECK(homography_max_diff(r.H_s, truth) < 1e-3);
    CHECK(homography_max_diff(r.H, truth) < 1e-3);
    const auto mask = f.mask.download();
    for (int i = 0; i < f.n; ++i) CHECK(mask[i] == (i % 4 != 0));
}

void test_incremental_equals_full() {
    // Given: the same stream, matches and seed, with R crossing batch boundaries.
    Fixture f;
    f.set_points(true, false);
    const auto full = f.run();
    const auto full_mask = f.mask.download();
    f.config.hypotheses = 17;
    CHECK(f.run().status == Status::OK);
    f.config.previous_hypotheses = 17; f.config.hypotheses = 257;
    // When: extending, only hypotheses 17..256 are searched.
    const auto split = f.run();
    // Then: refinement did not replace the persistent raw winner.
    CHECK(full.status == split.status && full.num_inliers == split.num_inliers);
    for (int i = 0; i < 9; ++i) CHECK(full.H[i] == split.H[i] && full.H_s[i] == split.H_s[i]);
    CHECK(full.inlier_ratio == split.inlier_ratio);
    CHECK(full.inlier_coverage == split.inlier_coverage);
    CHECK(full.reproj_residual == split.reproj_residual);
    CHECK(full_mask == f.mask.download());
}

void test_resize_and_repeatability() {
    // Given: H_s corresponds to two different half-pixel scale transforms.
    Fixture f;
    CHECK(make_scale_info(1281, 961, 0.5, &f.sa) == Status::OK);
    CHECK(make_scale_info(1281, 961, 0.75, &f.sb) == Status::OK);
    // When: restarting twice.
    const auto first = f.run(), second = f.run();
    // Then: original-space H uses D_A^-1 H_s D_B and results are repeatable.
    CHECK(first.status == Status::OK && second.status == Status::OK);
    double expected[9]{};
    CHECK(to_original(first.H_s, f.sa.D, f.sb.D, expected) == Status::OK);
    CHECK(homography_max_diff(first.H, expected) < 1e-10);
    for (int i = 0; i < 9; ++i) CHECK(first.H[i] == second.H[i]);
}

void test_failed_data_and_stale_state() {
    // Given/When/Then: too few correspondences are explicit failures with cleared masks.
    Fixture f;
    f.nm.upload({3});
    CHECK(f.run().status == Status::NOT_ENOUGH_POINTS);
    for (auto v : f.mask.download()) CHECK(v == 0);
    f.nm.upload({f.n}); f.set_points(false, true);
    CHECK(f.run().status == Status::DEGENERATE);
    f.set_points(false, false);
    CHECK(f.run().status == Status::OK);
    f.config.previous_hypotheses = f.config.hypotheses;
    ++f.config.generation;
    const auto stale = f.run();
    CHECK(stale.status == Status::INVALID_ARGUMENT && std::isnan(stale.H[0]));
    f.config.previous_hypotheses = 0;
    CHECK(f.run().status == Status::OK);
    f.ia.upload(std::vector<int32_t>(f.n, f.n));
    CHECK(f.run().status == Status::INVALID_ARGUMENT);
    f.set_points(false, false); f.nm.upload({f.n + 1});
    CHECK(f.run().status == Status::INVALID_ARGUMENT);
}

void test_host_boundary_errors() {
    Fixture f;
    // Given: a valid input and caller-owned output/workspace.
    // When/Then: invalid workspace and parameters fail before any launches.
    CHECK(estimate(f.a, f.b, f.m, f.sa, f.sb, f.config, f.mask.p, f.out,
                   f.workspace.p, f.workspace.n - 1, f.stream) == Status::INSUFFICIENT_WORKSPACE);
    CHECK(estimate(f.a, f.b, f.m, f.sa, f.sb, f.config, f.mask.p, f.out,
                   f.workspace.p + 1, f.workspace.n, f.stream) == Status::INVALID_ARGUMENT);
    f.config.threshold = std::numeric_limits<double>::infinity();
    CHECK(estimate(f.a, f.b, f.m, f.sa, f.sb, f.config, f.mask.p, f.out,
                   f.workspace.p, f.workspace.n, f.stream) == Status::INVALID_ARGUMENT);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fputs("S5 CUDA tests require a CUDA device\n", stderr);
        return 77;
    }
    std::puts("S5 CUDA seed=20261022");
    test_known_h_and_outliers();
    test_incremental_equals_full();
    test_resize_and_repeatability();
    test_failed_data_and_stale_state();
    test_host_boundary_errors();
    std::puts("S5 CUDA tests passed");
}
