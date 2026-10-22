#include "s4_s5/s4.hpp"

#include <cmath>
#include <limits>

namespace stitch::s4 {

Status s4_workspace_bytes(int capacity_a, int capacity_b, const Config& config,
                          std::size_t* bytes) {
    if (!bytes) return Status::INVALID_ARGUMENT;
    *bytes = 0;
    if (capacity_a < 0 || capacity_b < 0 ||
        !std::isfinite(config.ratio_threshold) ||
        config.ratio_threshold <= 0 || config.ratio_threshold >= 1) {
        return Status::INVALID_ARGUMENT;
    }
    if (capacity_a == 0) return Status::OK;
    const auto a = static_cast<std::size_t>(capacity_a);
    const auto b = config.cross_check ? static_cast<std::size_t>(capacity_b) : 0;
    const auto blocks = (a + 127) / 128;
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (a > limit / sizeof(Top2) || b > limit / sizeof(Top2) - a) {
        return Status::INVALID_ARGUMENT;
    }
    const auto neighbors = (a + b) * sizeof(Top2);
    if (a > limit / sizeof(int) - blocks ||
        (a + blocks) * sizeof(int) > limit - neighbors) {
        return Status::INVALID_ARGUMENT;
    }
    *bytes = neighbors + (a + blocks) * sizeof(int);
    return Status::OK;
}

}  // namespace stitch::s4
