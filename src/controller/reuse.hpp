// Host guidance for harness invalidation. Does not certify GPU cache readiness.
#pragma once

#include "controller/controller.hpp"

namespace stitch::controller {

enum class RestartFrom : int { S1 = 1, S2 = 2, S5 = 5 };

struct RetryPlan {
    RestartFrom restart_from;
    bool extend_s5;
    int previous_hypotheses;
    bool change_generation;
};

// Only for a RETRY in the same run. The caller must additionally prove all S5
// cache conditions (buffers, counts, metadata, seed, stream, completed R, etc.).
// If that proof fails, restart S5 with previous_hypotheses=0 instead of extending.
Status plan_retry(const Knobs& executed, const Knobs& next, RetryPlan* plan) noexcept;

}  // namespace stitch::controller
