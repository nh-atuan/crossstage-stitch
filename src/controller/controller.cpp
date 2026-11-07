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
    if (policy_.pooled) {
        if (!scene.features_present) return Status::INVALID_ARGUMENT;
        const auto values=selector_features(scene.features);
        std::copy(values.begin(),values.end(),pooled_context_.begin());
        for(int j=0;j<6;++j)pooled_context_[33+j]=scene.features[26+j];
        const auto& model=*policy_.pooled;
        const auto set_config=[&](std::size_t index) {
            pooled_context_[30]=ladder_[index].S;pooled_context_[31]=ladder_[index].K/2000.;
            pooled_context_[32]=ladder_[index].R/1024.;
        };
        set_config(model.reference_index);
        const double reference_ms=predict_ensemble(model.latency,pooled_context_.data());
        if (!std::isfinite(reference_ms)) return Status::INVALID_ARGUMENT;
        double best=reference_ms;ladder_position_=model.reference_index;
        for(std::size_t i=0;i<count_ && model.pre_threshold<1. && (!policy_.v5 || !policy_.v5->new_selector);++i) if(i!=model.reference_index) {
            set_config(i);
            const double p=predict_ensemble(model.pre_quality,pooled_context_.data());
            const double latency=predict_ensemble(model.latency,pooled_context_.data());
            if(!std::isfinite(p) || !std::isfinite(latency))return Status::INVALID_ARGUMENT;
            const double fallback_ms=reference_ms*(model.fallback_count?model.fallback_fractions[i]:1.);
            const double cost=latency+(1-p)*fallback_ms+model.extra_ms;
            if(p>=model.pre_threshold && cost<best) {best=cost;ladder_position_=i;}
        }
        if(policy_.v5 && policy_.v5->new_selector) ladder_position_=v5_select();
        attempts_=1;
        set_config(ladder_position_);*first=ladder_[ladder_position_];
        phase_=Phase::AwaitingFeedback;return Status::OK;
    }
    if (policy_.selector.count) {
        if (!scene.features_present) return Status::INVALID_ARGUMENT;
        const auto values=selector_features(scene.features);
        const int selected = predict_tree(policy_.selector, values.data(), selector_feature_count);
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

    gate_telemetry_={};
    Decision result{Action::REJECT, Reason::EXECUTION_ERROR, feedback.executed};
    if (feedback.execution_status != Status::OK ||
        feedback.homography.status == Status::INVALID_ARGUMENT) {
        phase_ = Phase::Finished;
    } else {
        result.reason = detail::self_check_reason(feedback.homography,
                                                 policy_.self_checks[ladder_position_]);
        if(policy_.v5 && policy_.v5->new_gate && result.reason==Reason::SELF_CHECK_PASSED && !valid_v5_homography(feedback.homography))
            result.reason=Reason::DEGENERATE_MODEL;
        if(policy_.pooled && ladder_position_!=policy_.pooled->reference_index &&
                result.reason==Reason::SELF_CHECK_PASSED) {
            if(policy_.v5 && policy_.v5->new_gate) {
                if(!v5_gate(feedback)) result.reason=Reason::QUALITY_PREDICTOR_REJECTED;
            } else {
                std::array<double,77> values{};
                std::copy_n(pooled_context_.begin(),33,values.begin());
                std::copy_n(feedback.features.begin(),24,values.begin()+33);
                std::copy_n(pooled_context_.begin()+33,6,values.begin()+57);
                std::copy_n(feedback.features.begin()+24,14,values.begin()+63);
                const double p=feedback.features_present?predict_ensemble(policy_.pooled->post_quality,values.data()):-1.;
                if(!std::isfinite(p) || p<policy_.pooled->gate_threshold) result.reason=Reason::QUALITY_PREDICTOR_REJECTED;
            }
        }
        if (policy_.selector.count && result.reason == Reason::SELF_CHECK_PASSED) {
            if (!feedback.features_present || predict_tree(policy_.gates[ladder_position_],
                    feedback.features.data(), gate_feature_count) != 1)
                result.reason = Reason::QUALITY_PREDICTOR_REJECTED;
        }
        if (result.reason == Reason::SELF_CHECK_PASSED) {
            result.action = Action::ACCEPT;
            phase_ = Phase::Finished;
        } else {
            std::size_t next = policy_.pooled ?
                (ladder_position_==policy_.pooled->reference_index?count_:policy_.pooled->reference_index) :
                policy_.retry_target_count ? policy_.retry_targets[ladder_position_] : ladder_position_+1;
            if(policy_.v5 && policy_.v5->staged && ladder_position_!=policy_.pooled->reference_index)
                next=v5_select(&feedback);
            // Changing R cannot add matches. Skip candidates whose inlier
            // requirement already exceeds the available correspondence count.
            while (next < count_ && ladder_[next].S == feedback.executed.S &&
                   ladder_[next].K == feedback.executed.K &&
                   feedback.homography.num_matches < policy_.self_checks[next].min_inliers)
                next=policy_.retry_target_count ? policy_.retry_targets[next] : next+1;
            if (next < count_) {
                result.action = Action::RETRY;
                result.next = ladder_[next];
                ladder_position_ = next;
                ++attempts_;
                if(policy_.pooled) {
                    pooled_context_[30]=ladder_[next].S;pooled_context_[31]=ladder_[next].K/2000.;
                    pooled_context_[32]=ladder_[next].R/1024.;
                }
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
    attempts_=0;telemetry_={};gate_telemetry_={};
    phase_ = ladder_ ? Phase::Idle : Phase::Uninitialized;
}

}  // namespace stitch::controller
