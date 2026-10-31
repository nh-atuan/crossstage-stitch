// Reproducible host-only integration trace. No fabricated GPU timing or QoR.
#include "common/geometry.hpp"
#include "controller/controller.hpp"
#include "controller/reuse.hpp"

#include <cstdio>

using namespace stitch;
using namespace stitch::controller;

int main() {
    constexpr auto blend = s6::BlendMode::Feather;
    const Knobs ladder[]{{0.5, 500, 64, blend}, {0.5, 500, 128, blend},
                         {0.5, 1000, 128, blend}, {1, 1000, 256, blend}};
    const SelfCheck checks[]{{12, 0.4, 0.1, 1}, {12, 0.4, 0.1, 1},
                             {12, 0.4, 0.1, 1}, {12, 0.4, 0.1, 2}};
    const SceneRule rules[]{{0.2, 0}, {0.1, 2}, {0, 3}};
    const Policy policy{0.25, rules, 3, checks, 4};
    Controller controller;
    if (controller.configure(policy) != Status::OK ||
        controller.initialize(ladder, 4) != Status::OK) return 1;
    SceneSignals scene{};
    if (make_scale_info(640, 480, controller.probe_scale(), &scene.a.scale) != Status::OK) return 1;
    scene.b.scale = scene.a.scale;
    scene.a.mean_abs_gradient = scene.b.mean_abs_gradient = 0.3;
    Knobs current{};
    if (controller.begin_pair(scene, &current) != Status::OK) return 1;
    std::puts("evidence=synthetic_fixture; GPU timing/QoR unavailable; thresholds are test values");
    for (int attempt = 0; attempt < 3; ++attempt) {
        Feedback feedback{current, Status::OK, {}};
        auto& h = feedback.homography;
        if (attempt == 0) {
            h.status = Status::DEGENERATE; h.num_matches = 20;
        } else if (attempt == 1) {
            h.status = Status::NOT_ENOUGH_POINTS; h.num_matches = 3;
        } else {
            h.status = Status::OK; h.num_matches = 20; h.num_inliers = 12;
            h.inlier_ratio = 0.6; h.inlier_coverage = 0.2; h.reproj_residual = 0.5;
            h.H[0] = h.H[4] = h.H[8] = h.H_s[0] = h.H_s[4] = h.H_s[8] = 1;
        }
        Decision d{};
        if (controller.observe(feedback, &d) != Status::OK) return 1;
        std::printf("attempt=%d S=%.2f K=%d R=%d action=%d reason=%d\n",
                    attempt, current.S, current.K, current.R,
                    static_cast<int>(d.action), static_cast<int>(d.reason));
        if (d.action == Action::ACCEPT) return attempt == 2 ? 0 : 1;
        if (d.action != Action::RETRY) return 1;
        RetryPlan plan{};
        if (plan_retry(current, d.next, &plan) != Status::OK) return 1;
        std::printf("restart=S%d extend_s5=%d previous_hypotheses=%d change_generation=%d\n",
                    static_cast<int>(plan.restart_from), plan.extend_s5,
                    plan.previous_hypotheses, plan.change_generation);
        current = d.next;
    }
    return 1;
}
