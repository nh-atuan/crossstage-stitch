#include "s4_s5/s5_internal.hpp"

#include <cmath>
#include <limits>

namespace stitch::s5 {

Status s5_workspace_bytes(const Config& c, std::size_t* bytes) {
    if (!bytes || c.hypotheses <= 0 || c.previous_hypotheses < 0 ||
        c.previous_hypotheses > c.hypotheses || c.batch_size <= 0 ||
        c.batch_size > 65535 || c.refinement_steps < 0 ||
        !std::isfinite(c.threshold) || !(c.threshold > 0.0) ||
        !std::isfinite(c.threshold * c.threshold)) return Status::INVALID_ARGUMENT;
    if (static_cast<std::size_t>(c.batch_size) >
        (std::numeric_limits<std::size_t>::max() - detail::model_offset) / sizeof(detail::Model))
        return Status::INVALID_ARGUMENT;
    *bytes = detail::model_offset + static_cast<std::size_t>(c.batch_size) * sizeof(detail::Model);
    return Status::OK;
}

}  // namespace stitch::s5
