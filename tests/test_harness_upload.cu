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
    {
        HostImageRGB image{64,64,std::vector<std::uint8_t>(64*64*3)};
        std::uint32_t state=static_cast<std::uint32_t>(seed);
        for(auto& value:image.pixels){state^=state<<13;state^=state>>17;state^=state<<5;value=state&255;}
        BoostNode confidence[]={{-1,-1,-1,0.f,10.f}},rejection[]={{-1,-1,-1,0.f,0.f}};
        BoostNode latency[]={{31,1,2,.5f,0.f},{-1,-1,-1,0.f,.69314718f},{-1,-1,-1,0.f,2.30258509f}};
        std::size_t root[]={0};
        PooledModels models{{confidence,1,root,1,39,0.f,1.,0.,false},
            {rejection,1,root,1,77,0.f,1.,0.,false},{latency,3,root,1,39,0.f,1.,0.,true},.5,.95,.5,1,false,0};
        SelfCheck relaxed[]={{4,0.,0.,1e300},{4,0.,0.,1e300}};SceneRule fallback{0.,1};
        Knobs choices[]={{1.,32,128,s6::BlendMode::Feather},{1.,2000,1024,s6::BlendMode::Feather}};
        Policy pooled{.25,&fallback,1,relaxed,2};pooled.pooled=&models;
        Controller adaptive;CHECK(adaptive.configure(pooled)==Status::OK);
        CHECK(adaptive.initialize(choices,2)==Status::OK);
        PipelineRunner small;small.initialize(64,64,2000,1024);small.upload_pair(image,image);
        RunRecord result;
        small.execute_adaptive_run(adaptive,"feature-fallback-fixture","fixture","synthetic","test","identity",0,0,seed,&result);
        CHECK(result.status=="OK" && small.attempt_trace().count==2);
        CHECK(small.attempt_trace().records[0].features[2]>0.);
        CHECK(!result.features_present && result.t_features>0.);
        for(double value:result.gate_features)CHECK(value==0.);
        std::puts("Reference fallback clears candidate feature metadata while retaining total feature cost");
    }
}
