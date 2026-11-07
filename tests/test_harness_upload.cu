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
        CHECK(record.t_probe >= 0.0 && record.t_probe_s1 >= 0.0 && record.t_probe_grad >= 0.0);
    }
    {
        runner.set_probe_enabled(false);
        runner.upload_pair(flat, flat);
        RunRecord blind_rec;
        runner.execute_adaptive_run(controller, "upload-fixture", "fixture", "synthetic",
                                    "test", "blind-pair", 0, 0, seed, &blind_rec);
        const auto& trace = runner.attempt_trace();
        CHECK(trace.count > 0);
        CHECK(trace.records[0].knobs.R == 4);
        CHECK(blind_rec.t_probe == 0.0 && blind_rec.t_probe_s1 == 0.0 && blind_rec.t_probe_grad == 0.0);
        CHECK(blind_rec.probe_gradient_a == 0.0 && blind_rec.probe_gradient_b == 0.0);
        runner.set_probe_enabled(true);
    }
    std::printf("First-probe upload consistency passed for 96 alternating pairs; seed=%llu\n",
                static_cast<unsigned long long>(seed));
    {
        // ORB reserves 31 pixels at each edge; retain enough interior for S5.
        constexpr int fixture_size=256;
        HostImageRGB image{fixture_size,fixture_size,std::vector<std::uint8_t>(fixture_size*fixture_size*3)};
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
        PipelineRunner small;small.initialize(fixture_size,fixture_size,2000,1024);small.upload_pair(image,image);
        RunRecord result;
        small.execute_adaptive_run(adaptive,"feature-fallback-fixture","fixture","synthetic","test","identity",0,0,seed,&result);
        std::printf("Pooled fallback fixture: status=%s stage=%s attempts=%d\n",result.status.c_str(),result.fail_stage.c_str(),small.attempt_trace().count);
        for(int i=0;i<small.attempt_trace().count;++i) {
            const auto& attempt=small.attempt_trace().records[i];
            std::printf("attempt=%d K=%d matches=%d inliers=%d H_status=%d action=%d reason=%d\n",i,attempt.knobs.K,
                attempt.homography.num_matches,attempt.homography.num_inliers,int(attempt.homography.status),int(attempt.decision.action),int(attempt.decision.reason));
        }
        CHECK(result.status=="OK" && small.attempt_trace().count==2);
        CHECK(small.attempt_trace().records[0].features[2]>0.);
        CHECK(!result.features_present && result.t_features>0.);
        for(double value:result.gate_features)CHECK(value==0.);
        std::puts("Reference fallback clears candidate feature metadata while retaining total feature cost");
    }
}
