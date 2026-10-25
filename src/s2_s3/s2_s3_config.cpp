#include "s2_s3/s2_s3_internal.hpp"

#include <climits>
#include <cmath>
#include <limits>

namespace stitch::s2_s3::detail {
namespace {

constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();

// Appends `count * element` bytes at the next aligned offset; false on overflow.
bool reserve(std::size_t* cursor, std::size_t count, std::size_t element,
             std::size_t* offset) {
    if (*cursor > kMax - (kAlignment - 1)) return false;
    const std::size_t start = (*cursor + kAlignment - 1) / kAlignment * kAlignment;
    if (element && count > (kMax - start) / element) return false;
    *offset = start;
    *cursor = start + count * element;
    return true;
}

}  // namespace

bool valid_image_shape(int width, int height) {
    return width > 0 && height > 0 && width <= INT_MAX / height;
}

bool valid_image(const ImageView& image) {
    return image.data && image.format == PixelFormat::GRAY8 &&
           valid_image_shape(image.width, image.height) && image.pitch_bytes >= image.width;
}

bool valid_s2_config(const s2::Config& config) {
    return config.fast_threshold >= 1 && config.fast_threshold <= 254 &&
           config.max_keypoints >= 0 &&
           config.max_keypoints <= INT_MAX / s2::kCandidateFactor &&
           std::isfinite(config.harris_k) && config.harris_k >= 0.0f;
}

bool make_s2_layout(int width, int height, int max_keypoints, S2Layout* layout) {
    if (!layout || !valid_image_shape(width, height) || max_keypoints < 0 ||
        max_keypoints > INT_MAX / s2::kCandidateFactor) return false;
    S2Layout l{};
    l.pixels = width * height;
    l.blocks = (l.pixels - 1) / kScanItems + 1;
    l.candidates = s2::kCandidateFactor * max_keypoints;
    const auto pixels = static_cast<std::size_t>(l.pixels);
    const auto blocks = static_cast<std::size_t>(l.blocks);
    std::size_t cursor = 0;
    if (!reserve(&cursor, pixels, 1, &l.scores) ||
        !reserve(&cursor, pixels, 1, &l.maxima) ||
        !reserve(&cursor, 256, sizeof(std::uint32_t), &l.histogram) ||
        !reserve(&cursor, 1, sizeof(SelectParams), &l.params) ||
        !reserve(&cursor, 2 * blocks, sizeof(std::uint32_t), &l.block_counts) ||
        !reserve(&cursor, 2 * blocks, sizeof(std::uint32_t), &l.block_offsets) ||
        !reserve(&cursor, static_cast<std::size_t>(l.candidates), sizeof(std::uint64_t),
                 &l.keys)) return false;
    l.total = cursor;
    *layout = l;
    return true;
}

bool make_s3_layout(int width, int height, S3Layout* layout) {
    if (!layout || !valid_image_shape(width, height)) return false;
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    S3Layout l{};
    std::size_t cursor = 0;
    if (!reserve(&cursor, pixels, 1, &l.blurred) ||
        !reserve(&cursor, pixels, sizeof(float), &l.rows)) return false;
    l.total = cursor;
    *layout = l;
    return true;
}

}  // namespace stitch::s2_s3::detail

namespace stitch::s2 {

Status s2_workspace_bytes(int width, int height, const Config& config,
                          std::size_t* bytes) {
    if (!bytes) return Status::INVALID_ARGUMENT;
    *bytes = 0;
    if (!s2_s3::detail::valid_s2_config(config) ||
        !s2_s3::detail::valid_image_shape(width, height)) return Status::INVALID_ARGUMENT;
    if (config.max_keypoints == 0) return Status::OK;
    s2_s3::detail::S2Layout layout;
    if (!s2_s3::detail::make_s2_layout(width, height, config.max_keypoints, &layout)) {
        return Status::INVALID_ARGUMENT;
    }
    *bytes = layout.total;
    return Status::OK;
}

}  // namespace stitch::s2

namespace stitch::s3 {

Status s3_workspace_bytes(int width, int height, std::size_t* bytes) {
    if (!bytes) return Status::INVALID_ARGUMENT;
    *bytes = 0;
    s2_s3::detail::S3Layout layout;
    if (!s2_s3::detail::make_s3_layout(width, height, &layout)) return Status::INVALID_ARGUMENT;
    *bytes = layout.total;
    return Status::OK;
}

}  // namespace stitch::s3
