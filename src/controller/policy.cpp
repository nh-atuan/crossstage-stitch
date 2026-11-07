#include "controller/policy.hpp"
#include "controller/v5.hpp"

#include <cmath>
#include "controller/features.hpp"

namespace stitch::controller::detail {
namespace {
bool unit_interval(double x) noexcept {
    return std::isfinite(x) && x >= 0.0 && x <= 1.0;
}
}  // namespace

bool valid_policy(const Policy& p) noexcept {
    if (p.v5 && (!p.pooled || !valid_v5(*p.v5,p.self_check_count))) return false;
    if (!std::isfinite(p.probe_scale) || p.probe_scale <= 0.0 || p.probe_scale > 1.0 ||
        !p.scene_rules || p.scene_rule_count == 0 || !p.self_checks || p.self_check_count == 0)
        return false;
    for (std::size_t i = 0; i < p.scene_rule_count; ++i) {
        const auto& r = p.scene_rules[i];
        if (!unit_interval(r.min_pair_gradient) || r.first_index >= p.self_check_count ||
            (i && r.min_pair_gradient >= p.scene_rules[i - 1].min_pair_gradient))
            return false;
    }
    if (p.scene_rules[p.scene_rule_count - 1].min_pair_gradient != 0.0) return false;
    if (p.pooled && (p.selector.count || p.gates || p.gate_count || p.retry_target_count ||
        p.pooled->reference_index+1!=p.self_check_count || p.self_check_count>64 ||
        !valid_ensemble(p.pooled->pre_quality,39,false) ||
        !valid_ensemble(p.pooled->post_quality,77,false) ||
        !valid_ensemble(p.pooled->latency,39,true) ||
        !unit_interval(p.pooled->pre_threshold) || !unit_interval(p.pooled->gate_threshold) ||
        !std::isfinite(p.pooled->extra_ms) || p.pooled->extra_ms<0 || p.pooled->relation_mask>7)) return false;
    if (p.pooled) {
        const auto& m=*p.pooled;
        if (m.fallback_count) {
            if (!m.fallback_fractions || m.fallback_count!=p.self_check_count) return false;
            for (std::size_t i=0;i<m.fallback_count;++i)
                if (!unit_interval(m.fallback_fractions[i])) return false;
            if (m.fallback_fractions[m.reference_index]!=1.) return false;
        } else if (m.fallback_fractions) return false;
    }
    for (std::size_t i = 0; i < p.self_check_count; ++i) {
        const auto& c = p.self_checks[i];
        if (c.min_inliers < 4 || !unit_interval(c.min_inlier_ratio) ||
            !unit_interval(c.min_inlier_coverage) || !std::isfinite(c.max_residual_s1) ||
            c.max_residual_s1 < 0.0)
            return false;
    }
    if (p.selector.count) {
        if (!valid_tree(p.selector, selector_feature_count, static_cast<int>(p.self_check_count)) ||
            !p.gates || p.gate_count != p.self_check_count) return false;
        for (std::size_t i = 0; i < p.gate_count; ++i)
            if (!valid_tree(p.gates[i], 24, 2)) return false;
    } else if (p.gates || p.gate_count) return false;
    if (p.retry_target_count) {
        if (!p.selector.count || !p.retry_targets || p.retry_target_count!=p.self_check_count) return false;
        for (std::size_t i=0;i<p.retry_target_count;++i)
            if (p.retry_targets[i]<=i || p.retry_targets[i]>p.self_check_count) return false;
    } else if (p.retry_targets) return false;
    return true;
}
}  // namespace stitch::controller::detail
