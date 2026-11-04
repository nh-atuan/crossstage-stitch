// Explicit host-side policy. All arrays are borrowed and remain immutable.
#pragma once

#include <cstddef>
#include "controller/tree.hpp"

namespace stitch::controller {

struct SelfCheck {
    int min_inliers;
    double min_inlier_ratio;
    double min_inlier_coverage;
    double max_residual_s1;  // RMS forward error, in S1 pixels at this ladder entry.
};

struct SceneRule {
    double min_pair_gradient;  // Inclusive lower bound on min(G_A, G_B).
    std::size_t first_index;
};

struct Policy {
    double probe_scale;
    const SceneRule* scene_rules;  // Strictly decreasing bounds, ending at zero.
    std::size_t scene_rule_count;
    const SelfCheck* self_checks;  // One entry per configuration in the ladder.
    std::size_t self_check_count;
    TreeView selector{};
    const TreeView* gates{};
    std::size_t gate_count{};
};

namespace detail {
bool valid_policy(const Policy& policy) noexcept;
}  // namespace detail

}  // namespace stitch::controller
