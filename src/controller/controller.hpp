// Controller Contract v1. No allocation, CUDA calls or I/O in per-pair methods.
#pragma once

#include <cstddef>
#include "common/types.hpp"
#include "controller/policy.hpp"
#include "controller/features.hpp"
#include "s6/s6.hpp"

namespace stitch::controller {

struct Knobs {
    double S;
    int K;
    int R;
    s6::BlendMode B;
};

struct ImageSceneSignals {
    ScaleInfo scale;
    double mean_abs_gradient;
};

struct SceneSignals {
    ImageSceneSignals a;
    ImageSceneSignals b;
    SceneFeatures features{};
    bool features_present{};
};

struct Feedback {
    Knobs executed;
    Status execution_status;
    HomographyResult homography;
    GateFeatures features{};
    bool features_present{};
};

enum class Action : int { ACCEPT = 0, RETRY = 1, REJECT = 2 };

enum class Reason : int {
    SELF_CHECK_PASSED = 0,
    TOO_FEW_MATCHES,
    TOO_FEW_INLIERS,
    LOW_INLIER_RATIO,
    LOW_INLIER_COVERAGE,
    HIGH_RESIDUAL,
    DEGENERATE_MODEL,
    EXECUTION_ERROR,
    BUDGET_EXHAUSTED,
    QUALITY_PREDICTOR_REJECTED
};

struct Decision {
    Action action;
    Reason reason;
    Knobs next;
};

class Controller {
public:
    Controller() noexcept = default;
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    // Setup extension to v1: configure before initialize. No numeric defaults.
    // Borrows rule/check arrays; on success clears the previous ladder binding.
    // Forbidden while awaiting feedback. Invalid input changes nothing.
    Status configure(const Policy& policy) noexcept;
    double probe_scale() const noexcept;
    bool uses_learned_features() const noexcept { return configured_ && (policy_.selector.count != 0 || policy_.pooled); }
    bool uses_pair_relations() const noexcept { return configured_ && policy_.pooled && policy_.pooled->pair_relations; }
    unsigned relation_mask() const noexcept { return configured_ && policy_.pooled ? policy_.pooled->relation_mask : 0; }
    bool needs_gate_features() const noexcept { return uses_learned_features() &&
        (!policy_.pooled || ladder_position_!=policy_.pooled->reference_index); }

    // Borrows the immutable ladder until the next successful configure/initialize.
    Status initialize(const Knobs* ladder, std::size_t count) noexcept;
    Status begin_pair(const SceneSignals& scene, Knobs* first) noexcept;
    Status observe(const Feedback& feedback, Decision* decision) noexcept;
    // Caller finishes GPU work and separately resets S5 cache before reuse.
    void reset_pair() noexcept;

private:
    enum class Phase { Uninitialized, Idle, AwaitingFeedback, Finished };
    Policy policy_{};
    bool configured_ = false;
    const Knobs* ladder_ = nullptr;
    std::size_t count_ = 0;
    std::size_t ladder_position_ = 0;
    Phase phase_ = Phase::Uninitialized;
    std::array<double,39> pooled_context_{};
};

namespace detail {
bool valid_knobs(const Knobs& knobs) noexcept;
bool same_knobs(const Knobs& a, const Knobs& b) noexcept;
bool valid_scene(const SceneSignals& scene, double probe_scale) noexcept;
bool valid_feedback(const Feedback& feedback) noexcept;
Reason self_check_reason(const HomographyResult& h, const SelfCheck& check) noexcept;
}  // namespace detail

}  // namespace stitch::controller
