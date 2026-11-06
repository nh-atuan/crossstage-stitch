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
    double t_s1{}, t_s2{}, t_s3{}, t_s4{}, t_s5{}, t_attempt{};
    const char* restart_from{"none"};
    const char* fail_stage{""};
    Status execution_status{Status::OK};
    bool has_feedback{};
    bool has_decision{};
    bool has_homography{};
};
struct AttemptTrace {
    static constexpr std::size_t capacity = 16;
    std::array<AttemptRecord, capacity> records{};
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
    default: return "BUDGET_EXHAUSTED";
    }
}
inline const char* attempt_blend_name(s6::BlendMode mode) {
    switch (mode) {
    case s6::BlendMode::Multiband3: return "multiband3";
    case s6::BlendMode::Multiband5: return "multiband5";
    default: return "feather";
    }
}
inline void write_attempt_header(std::ostream& out) {
    out << "run_id,git_rev,dataset,split,pair_id,repeat_idx,is_warmup,seed,attempt_idx,"
           "S,K,R,previous_R,eligible_R_old,generation,num_matches,num_inliers,inlier_ratio,"
           "inlier_coverage,reproj_residual,homography_status,action,reason,t_s4,t_s5,t_attempt,"
           "h0,h1,h2,h3,h4,h5,h6,h7,h8,policy_id,B,restart_from,t_s1,t_s2,t_s3,"
           "execution_status,attempt_fail_stage,terminal_status,terminal_fail_stage\n";
}
inline void write_attempt_rows(std::ostream& out, const RunRecord& run, const AttemptTrace& trace) {
    out << std::setprecision(17);
    for (int i = 0; i < trace.count; ++i) {
        const auto& a = trace.records[i];
        const auto& h = a.homography;
        out << run.run_id << ',' << run.git_rev << ',' << run.dataset << ',' << run.split << ','
            << run.pair_id << ',' << run.repeat_idx << ',' << run.is_warmup << ',' << run.seed << ','
            << i << ',' << a.knobs.S << ',' << a.knobs.K << ',' << a.knobs.R << ','
            << a.previous_R << ',' << a.eligible_R_old << ',' << a.generation << ',';
        if (a.has_feedback) {
            out << h.num_matches << ',' << h.num_inliers << ',' << h.inlier_ratio << ','
                << h.inlier_coverage << ',' << h.reproj_residual << ',' << static_cast<int>(h.status);
        } else {
            out << ",,,,,";
        }
        out << ',';
        if (a.has_decision) out << action_name(a.decision.action) << ',' << reason_name(a.decision.reason);
        else out << ',';
        out << ',' << a.t_s4 << ',' << a.t_s5 << ',' << a.t_attempt;
        for (double value : h.H) {
            out << ',';
            if (a.has_homography) out << value;
        }
        out << ',' << run.policy_id << ',' << attempt_blend_name(a.knobs.B) << ',' << a.restart_from << ','
            << a.t_s1 << ',' << a.t_s2 << ',' << a.t_s3 << ',' << static_cast<int>(a.execution_status)
            << ',' << a.fail_stage << ',' << run.status << ',' << run.fail_stage << '\n';
    }
}
}  // namespace stitch::harness
