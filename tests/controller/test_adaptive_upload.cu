// First probe must see the new image pair, and a pre-attempt failure must clear the trace.
#include <cuda_runtime.h>
#include "check.hpp"
#include "harness/pipeline_runner.hpp"

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    using namespace stitch;
    using namespace stitch::controller;
    using namespace stitch::harness;
    constexpr int width = 512, height = 256;
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
    Controller controller;
    CHECK(controller.configure({.25, rules, 2, checks, 2}) == Status::OK);
    CHECK(controller.initialize(ladder, 2) == Status::OK);
    PipelineRunner runner;
    runner.initialize(width, height, 32, 8, width * 2, height * 2, s6::BlendMode::Feather);
    RunRecord record;
    for (int i = 0; i < 4; ++i) {
        const bool high = i % 2 == 0;
        const auto& image = high ? textured : flat;
        runner.upload_pair(image, image);
        runner.execute_adaptive_run(controller, "upload-fixture", "fixture", "synthetic", "test",
                                    "alternating-pair", i, 0, 20261101, &record);
        const auto& trace = runner.attempt_trace();
        CHECK(trace.count > 0 && trace.count == record.attempt_count);
        CHECK(trace.records[0].knobs.R == (high ? 4 : 8));
        CHECK(trace.records[trace.count - 1].knobs.R == record.R);
        CHECK(record.t_e2e > 0 && record.t_probe_gpu > 0 && record.t_stream_wait > 0);
        CHECK(record.status != "OK");
    }

    Controller unconfigured;
    runner.execute_adaptive_run(unconfigured, "upload-fixture", "fixture", "synthetic", "test",
                                "alternating-pair", 4, 0, 20261101, &record);
    CHECK(record.status == "INVALID_PROBE_SCALE" && record.fail_stage == "controller");
    CHECK(runner.attempt_trace().count == 0 && record.attempt_count == 0);
    CHECK(record.t_e2e > 0 && record.t_probe_gpu == 0 && record.num_inliers == 0);
}
