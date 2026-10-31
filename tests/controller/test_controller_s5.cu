// Actual S5 GPU feedback with synthetic correspondences, not an image benchmark.
#include <cuda_runtime.h>
#include "check.hpp"
#include "common/geometry.hpp"
#include "controller/controller.hpp"
#include "controller/reuse.hpp"
#include "s5/s5.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace stitch;
using namespace stitch::controller;

#define GPU(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (false)

namespace {
template <class T> struct Device {
    T* data{};
    explicit Device(std::size_t count) {
        GPU(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& values) {
        GPU(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};
}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    constexpr int n = 80;
    constexpr std::uint64_t seed = 20261030;
    Device<float> ax(n), ay(n), bx(n), by(n);
    Device<int> na(1), nb(1), nm(1);
    Device<int32_t> ia(n), ib(n);
    Device<std::uint8_t> mask(n);
    std::vector<float> x(n), y(n), u(n), v(n);
    std::vector<int32_t> indices(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 20 + (i % 10) * 50;
        y[i] = 20 + (i / 10) * 45;
        u[i] = x[i] + 11;
        v[i] = y[i] + 7;
        if (i % 4 == 0) { u[i] = 600 - i; v[i] = 400 - 2 * i; }
        indices[i] = i;
    }
    ax.upload(u); ay.upload(v); bx.upload(x); by.upload(y);
    ia.upload(indices); ib.upload(indices);
    na.upload({n}); nb.upload({n}); nm.upload({n});
    Keypoints a{ax.data, ay.data, nullptr, n, na.data};
    Keypoints b{bx.data, by.data, nullptr, n, nb.data};
    Matches matches{ia.data, ib.data, nullptr, n, nm.data};
    ScaleInfo scale{};
    CHECK(make_scale_info(640, 480, 1, &scale) == Status::OK);
    s5::Config config{257, 0, 32, 2, 0.5, seed, 1};
    std::size_t bytes{};
    CHECK(s5::s5_workspace_bytes(config, &bytes) == Status::OK);
    Device<std::uint8_t> workspace(bytes);
    HomographyResult* pinned{};
    GPU(cudaMallocHost(reinterpret_cast<void**>(&pinned), sizeof(*pinned)));
    cudaStream_t stream{};
    GPU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    GPU(cudaDeviceSynchronize());
    const auto run = [&]() {
        CHECK(s5::estimate(a, b, matches, scale, scale, config, mask.data, pinned,
                           workspace.data, bytes, stream) == Status::OK);
        GPU(cudaStreamSynchronize(stream));
        return *pinned;
    };
    const auto full = run();
    CHECK(full.status == Status::OK);
    std::vector<std::uint8_t> full_mask(n), split_mask(n);
    GPU(cudaMemcpy(full_mask.data(), mask.data, n, cudaMemcpyDeviceToHost));

    const auto blend = s6::BlendMode::Feather;
    const Knobs ladder[]{{1, n, 17, blend}, {1, n, 257, blend}};
    // Force a retry on the first model; thresholds are fixture controls only.
    const SelfCheck checks[]{{n + 1, 0, 0, 1}, {4, 0, 0, 1}};
    const SceneRule rules[]{{0, 0}};
    const Policy policy{1, rules, 1, checks, 2};
    Controller controller;
    CHECK(controller.configure(policy) == Status::OK);
    CHECK(controller.initialize(ladder, 2) == Status::OK);
    const SceneSignals scene{{scale, 0.1}, {scale, 0.1}};
    Knobs first{};
    CHECK(controller.begin_pair(scene, &first) == Status::OK);
    config.hypotheses = first.R;
    const auto partial = run();
    CHECK(partial.status == Status::OK);
    Decision decision{};
    CHECK(controller.observe({first, Status::OK, partial}, &decision) == Status::OK);
    CHECK(decision.action == Action::RETRY && decision.reason == Reason::TOO_FEW_INLIERS);
    RetryPlan plan{};
    CHECK(plan_retry(first, decision.next, &plan) == Status::OK);
    CHECK(plan.extend_s5 && plan.previous_hypotheses == 17);
    const auto next = decision.next;
    config.hypotheses = next.R;
    config.previous_hypotheses = plan.previous_hypotheses;
    const auto split = run();
    CHECK(split.status == full.status && split.num_inliers == full.num_inliers);
    CHECK(split.num_matches == full.num_matches);
    for (int i = 0; i < 9; ++i) CHECK(split.H[i] == full.H[i] && split.H_s[i] == full.H_s[i]);
    CHECK(split.inlier_ratio == full.inlier_ratio && split.inlier_coverage == full.inlier_coverage &&
          split.reproj_residual == full.reproj_residual);
    GPU(cudaMemcpy(split_mask.data(), mask.data, n, cudaMemcpyDeviceToHost));
    CHECK(split_mask == full_mask);
    CHECK(controller.observe({next, Status::OK, split}, &decision) == Status::OK);
    CHECK(decision.action == Action::ACCEPT);

    // Actual S5 invalid-reuse payload must terminate, never masquerade as quality failure.
    controller.reset_pair();
    CHECK(controller.begin_pair(scene, &first) == Status::OK);
    config.hypotheses = first.R;
    config.previous_hypotheses = first.R;
    ++config.generation;
    const auto stale = run();
    CHECK(stale.status == Status::INVALID_ARGUMENT);
    CHECK(controller.observe({first, Status::OK, stale}, &decision) == Status::OK);
    CHECK(decision.action == Action::REJECT && decision.reason == Reason::EXECUTION_ERROR);
    GPU(cudaFreeHost(pinned));
    GPU(cudaStreamDestroy(stream));
    std::printf("Controller/S5 GPU integration passed; seed=%llu; synthetic correspondences\n",
                static_cast<unsigned long long>(seed));
}
