// Bounded host trace: collection has no allocation or file I/O in timed runs.
#pragma once
#include <array>
#include <iomanip>
#include <ostream>
#include "controller/controller.hpp"
#include "harness/harness_types.hpp"

namespace stitch::harness {
struct AttemptRecord {
    controller::Knobs knobs{};
    controller::Decision decision{};
    HomographyResult homography{};
    int previous_R{}, eligible_R_old{};
    std::uint64_t generation{};
    double t_s4{}, t_s5{}, t_attempt{};
    controller::GateFeatures features{};
    double t_features{};
    controller::SpatialFeatures spatial{};
    controller::V5Telemetry v5{};
};
struct AttemptTrace {
    std::array<AttemptRecord, 16> records{};
    int count{};
};
inline const char* action_name(controller::Action action) {
    switch (action) {
    case controller::Action::ACCEPT: return "ACCEPT";
    case controller::Action::RETRY: return "RETRY";
    default: return "REJECT";
    }
}
inline const char* reason_name(controller::Reason reason) {
    using controller::Reason;
    switch (reason) {
    case Reason::SELF_CHECK_PASSED: return "SELF_CHECK_PASSED";
    case Reason::TOO_FEW_MATCHES: return "TOO_FEW_MATCHES";
    case Reason::TOO_FEW_INLIERS: return "TOO_FEW_INLIERS";
    case Reason::LOW_INLIER_RATIO: return "LOW_INLIER_RATIO";
    case Reason::LOW_INLIER_COVERAGE: return "LOW_INLIER_COVERAGE";
    case Reason::HIGH_RESIDUAL: return "HIGH_RESIDUAL";
    case Reason::DEGENERATE_MODEL: return "DEGENERATE_MODEL";
    case Reason::EXECUTION_ERROR: return "EXECUTION_ERROR";
    case Reason::QUALITY_PREDICTOR_REJECTED: return "QUALITY_PREDICTOR_REJECTED";
    default: return "BUDGET_EXHAUSTED";
    }
}
inline void write_attempt_header(std::ostream& out) {
    out << "run_id,git_rev,dataset,split,pair_id,repeat_idx,is_warmup,seed,attempt_idx,"
           "S,K,R,previous_R,eligible_R_old,generation,num_matches,num_inliers,inlier_ratio,"
           "inlier_coverage,reproj_residual,homography_status,action,reason,t_s4,t_s5,t_attempt,"
           "h0,h1,h2,h3,h4,h5,h6,h7,h8,t_features";
    for (int i=0;i<controller::gate_feature_count;++i) out << ",gate_" << i;
    out << ",v5_p_pre,v5_p_reject,v5_attempt_ms,v5_gate_ms,v5_accept_s6_ms,v5_fallback_ms,v5_common_ms,v5_score_ms,v5_fail_ssim,v5_fail_psnr,v5_fail_coverage,v5_risk_joint,v5_uncertainty,v5_gate_evaluated,v5_failure_code";
    for(int i=0;i<16;++i)out << ",spatial_" << i;
    out << '\n';
}
inline void write_attempt_rows(std::ostream& out, const RunRecord& run, const AttemptTrace& trace) {
    out << std::setprecision(17);
    for (int i = 0; i < trace.count; ++i) {
        const auto& a = trace.records[i];
        const auto& h = a.homography;
        out << run.run_id << ',' << run.git_rev << ',' << run.dataset << ',' << run.split << ','
            << run.pair_id << ',' << run.repeat_idx << ',' << run.is_warmup << ',' << run.seed << ','
            << i << ',' << a.knobs.S << ',' << a.knobs.K << ',' << a.knobs.R << ','
            << a.previous_R << ',' << a.eligible_R_old << ',' << a.generation << ','
            << h.num_matches << ',' << h.num_inliers << ',' << h.inlier_ratio << ','
            << h.inlier_coverage << ',' << h.reproj_residual << ',' << static_cast<int>(h.status) << ','
            << action_name(a.decision.action) << ',' << reason_name(a.decision.reason) << ','
            << a.t_s4 << ',' << a.t_s5 << ',' << a.t_attempt;
        for (double value : h.H) out << ',' << value;
        out << ',' << a.t_features;
        for (double value:a.features) out << ',' << value;
        const auto& v=a.v5;
        out << ',' << v.pre << ',' << v.reject << ',' << v.attempt_ms << ',' << v.gate_ms << ',' << v.accept_s6_ms
            << ',' << v.fallback_ms << ',' << v.common_ms << ',' << v.score_ms << ',' << v.fail_ssim << ',' << v.fail_psnr
            << ',' << v.fail_coverage << ',' << v.risk_joint << ',' << v.uncertainty << ',' << v.gate_evaluated << ',' << v.failure_code;
        for(double x:a.spatial)out << ',' << x;
        out << '\n';
    }
}
}  // namespace stitch::harness
