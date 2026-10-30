#include "controller/controller.hpp"

#include <cmath>
#include <limits>
#include "common/geometry.hpp"

namespace stitch::controller::detail {
namespace {
bool unit_interval(double x) noexcept {
    return std::isfinite(x) && x >= 0.0 && x <= 1.0;
}

bool valid_image_scene(const ImageSceneSignals& image, double probe) noexcept {
    const auto& s = image.scale;
    if (!unit_interval(image.mean_abs_gradient) || s.orig_w <= 0 || s.orig_h <= 0 ||
        s.work_w < 2 || s.work_h < 2 || s.work_w > s.orig_w || s.work_h > s.orig_h)
        return false;
    ScaleInfo expected{};
    if (make_scale_info(s.orig_w, s.orig_h, probe, &expected) != Status::OK ||
        s.work_w != expected.work_w || s.work_h != expected.work_h ||
        s.sx != expected.sx || s.sy != expected.sy)
        return false;
    for (int i = 0; i < 9; ++i)
        if (s.D[i] != expected.D[i]) return false;
    return true;
}
}  // namespace

bool valid_knobs(const Knobs& k) noexcept {
    return std::isfinite(k.S) && k.S > 0.0 && k.S <= 1.0 && k.K > 0 && k.R > 0 &&
           (k.B == s6::BlendMode::Feather || k.B == s6::BlendMode::Multiband3 ||
            k.B == s6::BlendMode::Multiband5);
}

bool same_knobs(const Knobs& a, const Knobs& b) noexcept {
    return a.S == b.S && a.K == b.K && a.R == b.R && a.B == b.B;
}

bool valid_scene(const SceneSignals& s, double probe_scale) noexcept {
    return valid_image_scene(s.a, probe_scale) && valid_image_scene(s.b, probe_scale);
}

bool valid_feedback(const Feedback& f) noexcept {
    switch (f.execution_status) {
        case Status::INVALID_ARGUMENT:
        case Status::INSUFFICIENT_WORKSPACE:
        case Status::CUDA_ERROR:
            return true;  // Payload is not ready; do not inspect it.
        case Status::OK:
            break;
        default:
            return false;
    }
    const auto& h = f.homography;
    switch (h.status) {
        case Status::NOT_ENOUGH_POINTS:
            return h.num_matches >= 0 && h.num_matches < 4;
        case Status::DEGENERATE:
            return h.num_matches >= 4;
        case Status::INVALID_ARGUMENT:
            return true;  // Device input/reuse error; no usable model or counts.
        case Status::OK:
            break;
        default:
            return false;
    }
    if (h.num_inliers < 4 || h.num_matches < h.num_inliers ||
        !unit_interval(h.inlier_ratio) || !unit_interval(h.inlier_coverage) ||
        !std::isfinite(h.reproj_residual) || h.reproj_residual < 0.0)
        return false;
    // Numeric schema check, not a quality tolerance. S5 computes this in FP64.
    const double ratio = static_cast<double>(h.num_inliers) / h.num_matches;
    if (std::fabs(h.inlier_ratio - ratio) > 32 * std::numeric_limits<double>::epsilon())
        return false;
    for (int i = 0; i < 9; ++i)
        if (!std::isfinite(h.H[i]) || !std::isfinite(h.H_s[i])) return false;
    return true;
}

Reason self_check_reason(const HomographyResult& h, const SelfCheck& c) noexcept {
    if (h.status == Status::NOT_ENOUGH_POINTS) return Reason::TOO_FEW_MATCHES;
    if (h.status == Status::DEGENERATE) return Reason::DEGENERATE_MODEL;
    if (h.num_inliers < c.min_inliers) return Reason::TOO_FEW_INLIERS;
    if (h.inlier_ratio < c.min_inlier_ratio) return Reason::LOW_INLIER_RATIO;
    if (h.inlier_coverage < c.min_inlier_coverage) return Reason::LOW_INLIER_COVERAGE;
    if (h.reproj_residual > c.max_residual_s1) return Reason::HIGH_RESIDUAL;
    return Reason::SELF_CHECK_PASSED;
}
}  // namespace stitch::controller::detail
