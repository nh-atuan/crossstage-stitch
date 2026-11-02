#include "check.hpp"
#include "common/geometry.hpp"
#include "controller/controller.hpp"
#include "controller/reuse.hpp"

#include <atomic>
#include <cstdlib>
#include <limits>
#include <new>

using namespace stitch;
using namespace stitch::controller;

namespace {
std::atomic<int> allocations{0};
constexpr auto feather = s6::BlendMode::Feather;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct Fixture {
    // Synthetic test values only; these are not a calibrated pilot policy.
    Knobs ladder[4]{{0.5, 500, 64, feather}, {0.5, 500, 128, feather},
                    {0.5, 1000, 128, feather}, {1, 1000, 256, feather}};
    SelfCheck checks[4]{{12, 0.4, 0.1, 1.0}, {12, 0.4, 0.1, 1.0},
                        {12, 0.4, 0.1, 1.0}, {12, 0.4, 0.1, 2.0}};
    SceneRule rules[3]{{0.2, 0}, {0.1, 2}, {0, 3}};
    Policy policy{0.25, rules, 3, checks, 4};
    Controller controller;
    Fixture() {
        CHECK(controller.configure(policy) == Status::OK);
        CHECK(controller.initialize(ladder, 4) == Status::OK);
    }
    SceneSignals scene(double a = 0.3, double b = 0.3) const {
        SceneSignals s{};
        CHECK(make_scale_info(1281, 961, policy.probe_scale, &s.a.scale) == Status::OK);
        CHECK(make_scale_info(1003, 601, policy.probe_scale, &s.b.scale) == Status::OK);
        s.a.mean_abs_gradient = a;
        s.b.mean_abs_gradient = b;
        return s;
    }
    Knobs begin(double a = 0.3, double b = 0.3) {
        Knobs k{};
        CHECK(controller.begin_pair(scene(a, b), &k) == Status::OK);
        return k;
    }
    Feedback good(const Knobs& k) const {
        Feedback f{k, Status::OK, {}};
        f.homography.status = Status::OK;
        f.homography.num_matches = 20;
        f.homography.num_inliers = 12;
        f.homography.inlier_ratio = 0.6;
        f.homography.inlier_coverage = 0.1;
        f.homography.reproj_residual = 1.0;
        f.homography.H[0] = f.homography.H[4] = f.homography.H[8] = 2.0;
        f.homography.H_s[0] = f.homography.H_s[4] = f.homography.H_s[8] = 2.0;
        return f;
    }
    Feedback failure(const Knobs& k, Status status, int matches) const {
        auto f = good(k);
        f.homography.status = status;
        f.homography.num_matches = matches;
        for (int i = 0; i < 9; ++i) f.homography.H[i] = f.homography.H_s[i] = nan;
        f.homography.inlier_ratio = f.homography.inlier_coverage = f.homography.reproj_residual = nan;
        return f;
    }
};

bool same_decision(const Decision& a, const Decision& b) {
    return a.action == b.action && a.reason == b.reason && detail::same_knobs(a.next, b.next);
}

Decision sentinel() { return {Action::RETRY, Reason::HIGH_RESIDUAL, {0.75, 7, 9, feather}}; }

void expect_invalid(Fixture& f, const Feedback& feedback) {
    auto d = sentinel();
    const auto old = d;
    CHECK(f.controller.observe(feedback, &d) == Status::INVALID_ARGUMENT);
    CHECK(same_decision(d, old));
}

void test_setup_and_ladder() {
    Controller c;
    CHECK(std::isnan(c.probe_scale()));
    CHECK(c.initialize(nullptr, 0) == Status::INVALID_ARGUMENT);
    c.reset_pair();
    Fixture f;
    CHECK(f.controller.probe_scale() == f.policy.probe_scale);
    auto p = f.policy;
    p.probe_scale = nan;
    CHECK(f.controller.configure(p) == Status::INVALID_ARGUMENT);
    p = f.policy; p.self_checks = nullptr;
    CHECK(f.controller.configure(p) == Status::INVALID_ARGUMENT);
    SceneRule gap[1]{{0.1, 0}};
    p = f.policy; p.scene_rules = gap; p.scene_rule_count = 1;
    CHECK(f.controller.configure(p) == Status::INVALID_ARGUMENT);
    SceneRule unordered[3]{{0.1, 0}, {0.2, 1}, {0, 3}};
    p.scene_rules = unordered; p.scene_rule_count = 3;
    CHECK(f.controller.configure(p) == Status::INVALID_ARGUMENT);
    SceneRule outside[1]{{0, 4}};
    p.scene_rules = outside; p.scene_rule_count = 1;
    CHECK(f.controller.configure(p) == Status::INVALID_ARGUMENT);
    SelfCheck bad[1]{{3, 0, 0, 1}};
    SceneRule single[1]{{0, 0}};
    p = {0.25, single, 1, bad, 1};
    CHECK(f.controller.configure(p) == Status::INVALID_ARGUMENT);
    bad[0] = {4, 1.1, 0, 1};
    CHECK(f.controller.configure(p) == Status::INVALID_ARGUMENT);
    bad[0] = {4, 0, 0, -1};
    CHECK(f.controller.configure(p) == Status::INVALID_ARGUMENT);
    CHECK(f.controller.initialize(f.ladder, 3) == Status::INVALID_ARGUMENT);
    Knobs ladder[4];
    for (int i = 0; i < 4; ++i) ladder[i] = f.ladder[i];
    ladder[3].B = s6::BlendMode::Multiband3;
    CHECK(f.controller.initialize(ladder, 4) == Status::INVALID_ARGUMENT);
    ladder[3] = ladder[0];
    CHECK(f.controller.initialize(ladder, 4) == Status::INVALID_ARGUMENT);
    ladder[3] = f.ladder[3]; ladder[3].S = nan;
    CHECK(f.controller.initialize(ladder, 4) == Status::INVALID_ARGUMENT);
    ladder[3] = f.ladder[3]; ladder[3].K = 0;
    CHECK(f.controller.initialize(ladder, 4) == Status::INVALID_ARGUMENT);
    // Failed setup leaves the original binding usable.
    CHECK(detail::same_knobs(f.begin(), f.ladder[0]));
    CHECK(f.controller.initialize(f.ladder, 4) == Status::INVALID_ARGUMENT);
    CHECK(f.controller.configure(f.policy) == Status::INVALID_ARGUMENT);
}

void test_scene_selection_and_validation() {
    Fixture f;
    const double gradients[5][2]{{0.3, 0.3}, {0.2, 0.2}, {0.9, 0.1}, {0, 1}, {1, 1}};
    const int expected[5]{0, 0, 2, 3, 0};
    for (int i = 0; i < 5; ++i) {
        CHECK(detail::same_knobs(f.begin(gradients[i][0], gradients[i][1]), f.ladder[expected[i]]));
        f.controller.reset_pair();
    }
    Knobs first{0.75, 7, 9, feather};
    const auto old = first;
    auto invalid = f.scene(); invalid.a.mean_abs_gradient = nan;
    CHECK(f.controller.begin_pair(invalid, &first) == Status::INVALID_ARGUMENT);
    CHECK(detail::same_knobs(first, old));
    invalid = f.scene(); invalid.b.scale.D[2] += 0.01;
    CHECK(f.controller.begin_pair(invalid, &first) == Status::INVALID_ARGUMENT);
    invalid = f.scene(); invalid.a.scale.work_w = 1;
    CHECK(f.controller.begin_pair(invalid, &first) == Status::INVALID_ARGUMENT);
    invalid = f.scene(); invalid.b.mean_abs_gradient = -0.1;
    CHECK(f.controller.begin_pair(invalid, &first) == Status::INVALID_ARGUMENT);
    invalid = f.scene(); invalid.a.scale.sx = nan;
    CHECK(f.controller.begin_pair(invalid, &first) == Status::INVALID_ARGUMENT);
    invalid = f.scene();
    CHECK(make_scale_info(1281, 961, 0.5, &invalid.a.scale) == Status::OK);
    CHECK(f.controller.begin_pair(invalid, &first) == Status::INVALID_ARGUMENT);
    CHECK(f.controller.begin_pair(f.scene(), nullptr) == Status::INVALID_ARGUMENT);
    CHECK(f.controller.begin_pair(f.scene(), &first) == Status::OK);
    CHECK(f.controller.begin_pair(f.scene(), &first) == Status::INVALID_ARGUMENT);
}

void test_accept_and_reset() {
    Fixture f;
    auto k = f.begin();
    auto feedback = f.good(k);
    Decision d{};
    CHECK(f.controller.observe(feedback, nullptr) == Status::INVALID_ARGUMENT);
    CHECK(f.controller.observe(feedback, &d) == Status::OK);
    CHECK(d.action == Action::ACCEPT && d.reason == Reason::SELF_CHECK_PASSED);
    CHECK(detail::same_knobs(d.next, k));
    expect_invalid(f, feedback);
    f.controller.reset_pair();
    CHECK(detail::same_knobs(f.begin(), k));
    f.controller.reset_pair();
    CHECK(f.controller.configure(f.policy) == Status::OK);
    CHECK(f.controller.begin_pair(f.scene(), &k) == Status::INVALID_ARGUMENT);
    CHECK(f.controller.initialize(f.ladder, 4) == Status::OK);
    CHECK(detail::same_knobs(f.begin(), f.ladder[0]));
}

void test_reasons_and_exhaustion() {
    for (int which = 0; which < 5; ++which) {
        Fixture f;
        auto k = f.begin();
        auto feedback = f.good(k);
        const Reason reasons[]{Reason::TOO_FEW_INLIERS, Reason::LOW_INLIER_RATIO,
                               Reason::LOW_INLIER_COVERAGE, Reason::HIGH_RESIDUAL,
                               Reason::DEGENERATE_MODEL};
        if (which == 0) {
            feedback.homography.num_inliers = 4;
            feedback.homography.inlier_ratio = 0.2;
            feedback.homography.inlier_coverage = 0;
            feedback.homography.reproj_residual = 10;
        } else if (which == 1) {
            feedback.homography.num_matches = 40;
            feedback.homography.inlier_ratio = 0.3;
            feedback.homography.inlier_coverage = 0;
        } else if (which == 2) {
            feedback.homography.inlier_coverage = 0;
            feedback.homography.reproj_residual = 10;
        } else if (which == 3) {
            feedback.homography.reproj_residual = 2;
        } else {
            feedback = f.failure(k, Status::DEGENERATE, 20);
        }
        Decision d{};
        CHECK(f.controller.observe(feedback, &d) == Status::OK);
        CHECK(d.action == Action::RETRY && d.reason == reasons[which]);
        CHECK(detail::same_knobs(d.next, f.ladder[1]));
        expect_invalid(f, feedback);  // Stale feedback after RETRY.
    }
    Fixture f;
    auto k = f.begin();
    Decision d{};
    for (int attempt = 0; attempt < 4; ++attempt) {
        const auto feedback = f.failure(k, Status::DEGENERATE, 20);
        CHECK(f.controller.observe(feedback, &d) == Status::OK);
        if (attempt < 3) {
            CHECK(d.action == Action::RETRY);
            k = d.next;
        } else {
            CHECK(d.action == Action::REJECT && d.reason == Reason::BUDGET_EXHAUSTED);
            CHECK(detail::same_knobs(d.next, k));
        }
    }
}

void test_too_few_matches_skips_r_only() {
    Fixture f;
    auto k = f.begin();
    Decision d{};
    CHECK(f.controller.observe(f.failure(k, Status::NOT_ENOUGH_POINTS, 3), &d) == Status::OK);
    CHECK(d.action == Action::RETRY && d.reason == Reason::TOO_FEW_MATCHES);
    CHECK(detail::same_knobs(d.next, f.ladder[2]));
    f.controller.reset_pair();
    k = f.begin(0, 0);
    CHECK(f.controller.observe(f.failure(k, Status::NOT_ENOUGH_POINTS, 0), &d) == Status::OK);
    CHECK(d.action == Action::REJECT && d.reason == Reason::BUDGET_EXHAUSTED);
}

void test_execution_errors_and_invalid_payloads() {
    const Status errors[]{Status::INVALID_ARGUMENT, Status::INSUFFICIENT_WORKSPACE, Status::CUDA_ERROR};
    for (const auto status : errors) {
        Fixture f;
        auto feedback = f.failure(f.begin(), static_cast<Status>(999), -1);
        feedback.execution_status = status;
        Decision d{};
        CHECK(f.controller.observe(feedback, &d) == Status::OK);
        CHECK(d.action == Action::REJECT && d.reason == Reason::EXECUTION_ERROR);
    }
    Fixture f;
    const auto k = f.begin();
    auto good = f.good(k);
    auto bad = good;
    bad.executed.R++;
    expect_invalid(f, bad);
    for (const auto status : {Status::NOT_ENOUGH_POINTS, Status::DEGENERATE, static_cast<Status>(999)}) {
        bad = good; bad.execution_status = status;
        expect_invalid(f, bad);
    }
    for (const auto status : {Status::INSUFFICIENT_WORKSPACE, Status::CUDA_ERROR, static_cast<Status>(999)}) {
        bad = good; bad.homography.status = status;
        expect_invalid(f, bad);
    }
    bad = f.failure(k, Status::NOT_ENOUGH_POINTS, 4); expect_invalid(f, bad);
    bad = f.failure(k, Status::NOT_ENOUGH_POINTS, -1); expect_invalid(f, bad);
    bad = f.failure(k, Status::DEGENERATE, 3); expect_invalid(f, bad);
    bad = good; bad.homography.num_inliers = 3; expect_invalid(f, bad);
    bad = good; bad.homography.num_inliers = 21; expect_invalid(f, bad);
    bad = good; bad.homography.inlier_ratio = 0.1; expect_invalid(f, bad);
    bad = good; bad.homography.inlier_coverage = nan; expect_invalid(f, bad);
    bad = good; bad.homography.reproj_residual = -1; expect_invalid(f, bad);
    bad = good; bad.homography.H_s[1] = nan; expect_invalid(f, bad);
    bad = good; bad.homography.H[8] = std::numeric_limits<double>::infinity(); expect_invalid(f, bad);
    Decision d{};
    CHECK(f.controller.observe(good, &d) == Status::OK);  // Invalid input did not advance state.
    f.controller.reset_pair();
    bad = f.failure(f.begin(), Status::INVALID_ARGUMENT, -1);
    CHECK(f.controller.observe(bad, &d) == Status::OK);
    CHECK(d.action == Action::REJECT && d.reason == Reason::EXECUTION_ERROR);
}

void test_nonmonotone_ladder_and_blending() {
    Fixture f;
    // Only ladder position is monotone; budgets may decrease.
    Knobs ladder[4]{{1, 1000, 256, feather}, {1, 1000, 64, feather},
                    {1, 500, 64, feather}, {0.5, 500, 64, feather}};
    CHECK(f.controller.initialize(ladder, 4) == Status::OK);
    auto k = f.begin();
    for (int i = 0; i < 3; ++i) {
        Decision d{};
        CHECK(f.controller.observe(f.failure(k, Status::DEGENERATE, 20), &d) == Status::OK);
        CHECK(d.action == Action::RETRY && detail::same_knobs(d.next, ladder[i + 1]));
        k = d.next;
    }
    f.controller.reset_pair();
    for (auto& entry : ladder) entry.B = s6::BlendMode::Multiband5;
    CHECK(f.controller.initialize(ladder, 4) == Status::OK);
    k = f.begin();
    Decision d{};
    CHECK(f.controller.observe(f.good(k), &d) == Status::OK);
    CHECK(d.action == Action::ACCEPT && d.next.B == s6::BlendMode::Multiband5);
}

void test_retry_plan() {
    Knobs a{0.5, 500, 64, feather}, b = a;
    RetryPlan p{};
    b.R = 128;
    CHECK(plan_retry(a, b, &p) == Status::OK);
    CHECK(p.restart_from == RestartFrom::S5 && p.extend_s5 && p.previous_hypotheses == 64 && !p.change_generation);
    CHECK(plan_retry(b, a, &p) == Status::OK);
    CHECK(p.restart_from == RestartFrom::S5 && !p.extend_s5 && p.previous_hypotheses == 0);
    b.K = 1000;
    CHECK(plan_retry(a, b, &p) == Status::OK);
    CHECK(p.restart_from == RestartFrom::S2 && !p.extend_s5 && p.change_generation);
    CHECK(plan_retry(b, a, &p) == Status::OK);
    CHECK(p.restart_from == RestartFrom::S2 && p.change_generation);
    b.S = 1;
    CHECK(plan_retry(a, b, &p) == Status::OK);
    CHECK(p.restart_from == RestartFrom::S1 && !p.extend_s5 && p.change_generation);
    CHECK(plan_retry(b, a, &p) == Status::OK);
    CHECK(p.restart_from == RestartFrom::S1 && p.change_generation);
    const auto old = p;
    b = a; b.B = s6::BlendMode::Multiband3;
    CHECK(plan_retry(a, b, &p) == Status::INVALID_ARGUMENT);
    CHECK(p.restart_from == old.restart_from && p.extend_s5 == old.extend_s5 &&
          p.previous_hypotheses == old.previous_hypotheses && p.change_generation == old.change_generation);
    CHECK(plan_retry(a, a, &p) == Status::INVALID_ARGUMENT);
    CHECK(plan_retry(a, b, nullptr) == Status::INVALID_ARGUMENT);
}

void test_retry_skips_unreachable_inlier_requirements() {
    Fixture f;
    f.checks[1].min_inliers = 25;
    auto k = f.begin();
    auto feedback = f.good(k);
    feedback.homography.num_matches = 23;
    feedback.homography.num_inliers = 6;
    feedback.homography.inlier_ratio = 6.0 / 23;
    Decision d{};
    CHECK(f.controller.observe(feedback, &d) == Status::OK);
    CHECK(d.action == Action::RETRY && detail::same_knobs(d.next, f.ladder[2]));

    // Equality is feasible: R may improve the model to include every match.
    f.controller.reset_pair();
    k = f.begin();
    feedback = f.good(k);
    feedback.homography.num_matches = 25;
    feedback.homography.num_inliers = 6;
    feedback.homography.inlier_ratio = 6.0 / 25;
    CHECK(f.controller.observe(feedback, &d) == Status::OK);
    CHECK(d.action == Action::RETRY && detail::same_knobs(d.next, f.ladder[1]));

    // A degenerate model still reports the usable correspondence count.
    f.controller.reset_pair();
    k = f.begin();
    CHECK(f.controller.observe(f.failure(k, Status::DEGENERATE, 23), &d) == Status::OK);
    CHECK(d.action == Action::RETRY && detail::same_knobs(d.next, f.ladder[2]));

    // Exhaustion is explicit when every remaining candidate is unreachable.
    Knobs ladder[4];
    for (int i = 0; i < 4; ++i) {
        ladder[i] = {0.5, 500, 64 + i, feather};
        f.checks[i].min_inliers = 25;
    }
    f.controller.reset_pair();
    CHECK(f.controller.initialize(ladder, 4) == Status::OK);
    k = f.begin();
    CHECK(f.controller.observe(f.failure(k, Status::DEGENERATE, 23), &d) == Status::OK);
    CHECK(d.action == Action::REJECT && d.reason == Reason::BUDGET_EXHAUSTED);
}

void test_no_per_pair_allocations() {
    Fixture f;
    const auto scene = f.scene();
    const int before = allocations.load();
    for (int repeat = 0; repeat < 1000; ++repeat) {
        f.controller.reset_pair();
        Knobs k{};
        CHECK(f.controller.begin_pair(scene, &k) == Status::OK);
        Decision d{};
        CHECK(f.controller.observe(f.failure(k, Status::DEGENERATE, 20), &d) == Status::OK);
        CHECK(d.action == Action::RETRY);
        CHECK(f.controller.observe(f.good(d.next), &d) == Status::OK);
        CHECK(d.action == Action::ACCEPT);
    }
    CHECK(allocations.load() == before);
}
}  // namespace

void* operator new(std::size_t n) {
    allocations.fetch_add(1);
    if (auto* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main() {
    test_setup_and_ladder();
    test_scene_selection_and_validation();
    test_accept_and_reset();
    test_reasons_and_exhaustion();
    test_too_few_matches_skips_r_only();
    test_execution_errors_and_invalid_payloads();
    test_nonmonotone_ladder_and_blending();
    test_retry_plan();
    test_retry_skips_unreachable_inlier_requirements();
    test_no_per_pair_allocations();
    std::puts("Controller host contract tests passed (synthetic policy; no GPU evidence)");
}
