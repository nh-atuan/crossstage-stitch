// Versioned v5 research policy. No allocation, labels or Reference H at runtime.
#pragma once
#include <array>
#include "controller/ensemble.hpp"
#include "common/types.hpp"
namespace stitch::controller {
constexpr int v5_spatial_count=16, v5_post_count=93, v5_feedback_count=45;
using SpatialFeatures=std::array<double,v5_spatial_count>;
struct V5Models {
    EnsembleView pre{},attempt{},s6{},reject{},future{};
    std::array<EnsembleView,4> failures{};
    const double* incremental_ms{}; // row-major source/target, attempt only; excludes S6
    std::size_t costs_count{};
    double risk_threshold{},gate_ms{},common_ms{},pre_threshold{.5};
    bool new_selector{},new_gate{},spatial{},staged{},pair_ncc{},margin_heads{};
    unsigned relation_mask{4},spatial_groups{3};
    unsigned max_attempts{4};
};
struct V5Telemetry {
    double pre{},reject{},attempt_ms{},gate_ms{},accept_s6_ms{},fallback_ms{},common_ms{},score_ms{};
    double fail_ssim{},fail_psnr{},fail_coverage{},risk_joint{},uncertainty{};
    int failure_code{};
    bool gate_evaluated{};
};
inline double expected_cost(double common,double attempt,double gate,double reject,
                            double candidate_s6,double fallback_attempt,double reference_s6) noexcept {
    return common+attempt+gate+(1.-reject)*candidate_s6+reject*(fallback_attempt+reference_s6);
}
bool valid_v5(const V5Models&,std::size_t count) noexcept;
bool valid_v5_homography(const HomographyResult&) noexcept;
} // namespace stitch::controller
