#include "controller/controller.hpp"

#include <algorithm>
#include <limits>

namespace stitch::controller {

Status Controller::configure(const Policy& policy) noexcept {
    if (phase_ == Phase::AwaitingFeedback || !detail::valid_policy(policy))
        return Status::INVALID_ARGUMENT;
    policy_ = policy;
    configured_ = true;
    ladder_ = nullptr;
    count_ = ladder_position_ = 0;
    phase_ = Phase::Uninitialized;
    return Status::OK;
}

double Controller::probe_scale() const noexcept {
    return configured_ ? policy_.probe_scale : std::numeric_limits<double>::quiet_NaN();
}

Status Controller::initialize(const Knobs* ladder, std::size_t count) noexcept {
    if (phase_ == Phase::AwaitingFeedback || !configured_ || !ladder || count == 0 ||
        count != policy_.self_check_count)
        return Status::INVALID_ARGUMENT;
    for (std::size_t i = 0; i < count; ++i) {
        if (!detail::valid_knobs(ladder[i]) || ladder[i].B != ladder[0].B)
            return Status::INVALID_ARGUMENT;
        for (std::size_t j = 0; j < i; ++j)
            if (detail::same_knobs(ladder[i], ladder[j])) return Status::INVALID_ARGUMENT;
    }
    ladder_ = ladder;
    count_ = count;
    ladder_position_ = 0;
    phase_ = Phase::Idle;
    return Status::OK;
}

Status Controller::begin_pair(const SceneSignals& scene, Knobs* first) noexcept {
    if (phase_ != Phase::Idle || !first || !detail::valid_scene(scene, policy_.probe_scale))
        return Status::INVALID_ARGUMENT;
    if (policy_.selector.count) {
        if (!scene.features_present) return Status::INVALID_ARGUMENT;
        const int selected = predict_tree(policy_.selector, scene.features.data(), scene_feature_count);
        if (selected < 0 || selected >= static_cast<int>(count_)) return Status::INVALID_ARGUMENT;
        ladder_position_ = static_cast<std::size_t>(selected);
        *first = ladder_[ladder_position_];
        phase_ = Phase::AwaitingFeedback;
        return Status::OK;
    }
    const double gradient = std::min(scene.a.mean_abs_gradient, scene.b.mean_abs_gradient);
    for (std::size_t i = 0; i < policy_.scene_rule_count; ++i) {
        const auto& rule = policy_.scene_rules[i];
        if (gradient >= rule.min_pair_gradient) {
            ladder_position_ = rule.first_index;
            *first = ladder_[ladder_position_];
            phase_ = Phase::AwaitingFeedback;
            return Status::OK;
        }
    }
    return Status::INVALID_ARGUMENT;  // Unreachable for a valid immutable policy.
}

Status Controller::observe(const Feedback& feedback, Decision* decision) noexcept {
    if (phase_ != Phase::AwaitingFeedback || !decision ||
        !detail::same_knobs(feedback.executed, ladder_[ladder_position_]) ||
        !detail::valid_feedback(feedback))
        return Status::INVALID_ARGUMENT;

    Decision result{Action::REJECT, Reason::EXECUTION_ERROR, feedback.executed};
    if (feedback.execution_status != Status::OK ||
        feedback.homography.status == Status::INVALID_ARGUMENT) {
        phase_ = Phase::Finished;
    } else {
        result.reason = detail::self_check_reason(feedback.homography,
                                                 policy_.self_checks[ladder_position_]);
        if (policy_.selector.count && result.reason == Reason::SELF_CHECK_PASSED) {
            if (!feedback.features_present || predict_tree(policy_.gates[ladder_position_],
                    feedback.features.data(), gate_feature_count) != 1)
                result.reason = Reason::QUALITY_PREDICTOR_REJECTED;
        }
        if (result.reason == Reason::SELF_CHECK_PASSED) {
            result.action = Action::ACCEPT;
            phase_ = Phase::Finished;
        } else {
            std::size_t next = ladder_position_ + 1;
            // Changing R cannot add matches. Skip candidates whose inlier
            // requirement already exceeds the available correspondence count.
            while (next < count_ && ladder_[next].S == feedback.executed.S &&
                   ladder_[next].K == feedback.executed.K &&
                   feedback.homography.num_matches < policy_.self_checks[next].min_inliers)
                ++next;
            if (next < count_) {
                result.action = Action::RETRY;
                result.next = ladder_[next];
                ladder_position_ = next;
            } else {
                result.reason = Reason::BUDGET_EXHAUSTED;
                phase_ = Phase::Finished;
            }
        }
    }
    *decision = result;
    return Status::OK;
}

void Controller::reset_pair() noexcept {
    ladder_position_ = 0;
    phase_ = ladder_ ? Phase::Idle : Phase::Uninitialized;
}

}  // namespace stitch::controller
