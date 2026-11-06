// Common types and configurations for the benchmark harness.
#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "common/types.hpp"
#include "s6/s6.hpp"
#include "controller/features.hpp"

namespace stitch::harness {

struct HarnessConfig {
    std::string config_id;
    double S{0.5};
    int K{1000};
    int R{256};
    s6::BlendMode blend_mode{s6::BlendMode::Feather};
    std::string blend_mode_str{"feather"};

    // Algorithm tuning parameters with defaults from Phase 0 decisions
    int fast_threshold{20};
    float harris_k{0.04f};
    double ratio_threshold{0.8};
    bool cross_check{false};
    int msac_batch_size{64};
    int msac_refinements{1};
    double msac_threshold{3.0};
    std::uint64_t seed{0};
};

struct PairEntry {
    std::string pair_id;
    std::string img_a_path;
    std::string img_b_path;
    std::string reference_status{"ok"};
};

struct RunRecord {
    std::string run_id;
    std::string git_rev;
    std::string dataset{"udis_d"};
    std::string split{"dev"};
    std::string pair_id;
    std::string config_id;
    int repeat_idx{0};
    int is_warmup{0};
    std::uint64_t seed{0};

    double S{0.5};
    int K{1000};
    int R{256};
    std::string blend_mode{"feather"};

    // Timings in milliseconds
    double t_h2d{0.0};
    double t_s1{0.0};
    double t_s2{0.0};
    double t_s3{0.0};
    double t_s4{0.0};
    double t_s5{0.0};
    double t_s6{0.0};
    double t_controller{0.0};
    double t_d2h{0.0};
    double t_e2e{0.0};
    double t_probe_s1{0.0};
    double t_probe_grad{0.0};
    double t_probe{0.0};
    double probe_gradient_a{0.0};
    double probe_gradient_b{0.0};
    controller::SceneFeatures scene_features{};
    controller::GateFeatures gate_features{};
    bool features_present{};
    double t_features{};

    // Signals S2-S5
    int num_kp_a{0};
    int num_kp_b{0};
    int num_matches{0};
    int num_inliers{0};
    double inlier_ratio{0.0};
    double inlier_coverage{0.0};
    double reproj_residual{0.0};

    // Homography H (original coordinates, row-major B -> A)
    double H[9]{1, 0, 0, 0, 1, 0, 0, 0, 1};

    // Telemetry S6
    std::uint64_t canvas_pixels{0};
    std::uint64_t overlap_pixels{0};
    std::string blend_mode_effective{"feather"};
    int num_levels{0};
    std::size_t peak_workspace_bytes{0};
    int cache_hit{0};

    // Status
    std::string status{"OK"};
    std::string fail_stage{""};
};

}  // namespace stitch::harness
