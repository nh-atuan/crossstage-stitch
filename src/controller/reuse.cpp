#include "controller/reuse.hpp"

namespace stitch::controller {

Status plan_retry(const Knobs& a, const Knobs& b, RetryPlan* plan) noexcept {
    if (!plan || !detail::valid_knobs(a) || !detail::valid_knobs(b) || a.B != b.B ||
        detail::same_knobs(a, b))
        return Status::INVALID_ARGUMENT;
    RetryPlan result{RestartFrom::S5, false, 0, false};
    if (a.S != b.S) {
        result.restart_from = RestartFrom::S1;
        result.change_generation = true;
    } else if (a.K != b.K) {
        result.restart_from = RestartFrom::S2;
        result.change_generation = true;
    } else if (b.R > a.R) {
        result.extend_s5 = true;
        result.previous_hypotheses = a.R;
    }
    *plan = result;
    return Status::OK;
}

}  // namespace stitch::controller
