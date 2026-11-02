// First probe after each upload must classify the new pair, without a warmup.
#include <cuda_runtime.h>
#include "check.hpp"
#include "harness/pipeline_runner.hpp"

#include <cstdio>

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    using namespace stitch;
    using namespace stitch::controller;
    using namespace stitch::harness;
    constexpr std::uint64_t seed = 20261101;
    constexpr int width = 2048, height = 1024;
    HostImageRGB flat{width, height, std::vector<std::uint8_t>(width * height * 3, 0)};
    HostImageRGB textured = flat;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::uint8_t value = ((x / 8 + y / 8) % 2) ? 255 : 0;
            for (int c = 0; c < 3; ++c) textured.pixels[3 * (y * width + x) + c] = value;
        }
    }
    const Knobs ladder[] = {{.25, 32, 4, s6::BlendMode::Feather},
                            {.25, 32, 8, s6::BlendMode::Feather}};
    const SceneRule rules[] = {{.1, 0}, {0., 1}};
    const SelfCheck checks[] = {{100, 0., 0., 4.}, {100, 0., 0., 4.}};
    const Policy policy{.25, rules, 2, checks, 2};
    Controller controller;
    CHECK(controller.configure(policy) == Status::OK);
    CHECK(controller.initialize(ladder, 2) == Status::OK);
    PipelineRunner runner;
    runner.initialize(width, height, 32, 8);
    for (int iteration = 0; iteration < 96; ++iteration) {
        const bool high = iteration % 2 == 0;
        runner.set_reuse_enabled(iteration % 4 < 2);
        const auto& image = high ? textured : flat;
        runner.upload_pair(image, image);
        RunRecord record;
        runner.execute_adaptive_run(controller, "upload-fixture", "fixture", "synthetic",
                                    "test", "alternating-pair", iteration, 0, seed, &record);
        const auto& trace = runner.attempt_trace();
        CHECK(trace.count > 0);
        CHECK(trace.records[0].knobs.R == (high ? 4 : 8));
    }
    std::printf("First-probe upload consistency passed for 96 alternating pairs; seed=%llu\n",
                static_cast<unsigned long long>(seed));
}
