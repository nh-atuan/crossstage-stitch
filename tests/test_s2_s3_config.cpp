#include "s2_s3/s2_s3_internal.hpp"
#include "check.hpp"

#include <climits>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <utility>

using namespace stitch;
using stitch::s2_s3::detail::kAlignment;

namespace {

std::size_t align(std::size_t n) { return (n + kAlignment - 1) / kAlignment * kAlignment; }

// Mirrors the documented S2 layout: scores, maxima, histogram, params, two
// block arrays and the Harris keys, each starting on a 256-byte boundary.
std::size_t expected_s2(int w, int h, int k) {
    const std::size_t pixels = static_cast<std::size_t>(w) * h;
    const std::size_t blocks = (pixels + 1023) / 1024;
    std::size_t c = 0;
    c = align(c) + pixels;
    c = align(c) + pixels;
    c = align(c) + 256 * 4;
    c = align(c) + sizeof(s2_s3::detail::SelectParams);
    c = align(c) + 2 * blocks * 4;
    c = align(c) + 2 * blocks * 4;
    c = align(c) + static_cast<std::size_t>(2 * k) * 8;
    return c;
}

}  // namespace

int main() {
    std::size_t bytes = 123;
    const s2::Config base{20, 1000, 0.04f};

    CHECK(s2::s2_workspace_bytes(640, 480, base, &bytes) == Status::OK);
    CHECK(bytes == expected_s2(640, 480, 1000));
    CHECK(s2::s2_workspace_bytes(1, 1, {1, 1, 0.0f}, &bytes) == Status::OK);
    CHECK(bytes == expected_s2(1, 1, 1));
    CHECK(s2::s2_workspace_bytes(640, 480, {20, 0, 0.04f}, &bytes) == Status::OK && bytes == 0);

    // Every offset is aligned and inside the total.
    s2_s3::detail::S2Layout layout{};
    CHECK(s2_s3::detail::make_s2_layout(1921, 1081, 4000, &layout));
    CHECK(layout.blocks == (1921 * 1081 + 1023) / 1024 && layout.candidates == 8000);
    for (std::size_t offset : {layout.scores, layout.maxima, layout.histogram, layout.params,
                               layout.block_counts, layout.block_offsets, layout.keys}) {
        CHECK(offset % kAlignment == 0 && offset < layout.total);
    }
    CHECK(layout.keys + 8000 * sizeof(std::uint64_t) == layout.total);

    const s2::Config bad[] = {{0, 10, 0.04f}, {255, 10, 0.04f}, {20, -1, 0.04f},
                              {20, INT_MAX / 2 + 1, 0.04f}, {20, 10, -0.01f},
                              {20, 10, std::numeric_limits<float>::infinity()},
                              {20, 10, std::numeric_limits<float>::quiet_NaN()}, s2::Config{}};
    for (const auto& config : bad) {
        bytes = 123;
        CHECK(s2::s2_workspace_bytes(64, 64, config, &bytes) == Status::INVALID_ARGUMENT);
        CHECK(bytes == 0);
    }
    for (auto [w, h] : {std::pair{0, 10}, std::pair{10, 0}, std::pair{-5, 10},
                        std::pair{INT_MAX, 2}}) {
        CHECK(s2::s2_workspace_bytes(w, h, base, &bytes) == Status::INVALID_ARGUMENT && bytes == 0);
        CHECK(s3::s3_workspace_bytes(w, h, &bytes) == Status::INVALID_ARGUMENT && bytes == 0);
    }
    CHECK(s2::s2_workspace_bytes(64, 64, base, nullptr) == Status::INVALID_ARGUMENT);

    CHECK(s3::s3_workspace_bytes(640, 480, &bytes) == Status::OK);
    CHECK(bytes == align(640 * 480) + 640 * 480 * sizeof(float));
    CHECK(s3::s3_workspace_bytes(3, 2, nullptr) == Status::INVALID_ARGUMENT);

    // Image validation used by both stages.
    std::uint8_t pixel = 0;
    CHECK(s2_s3::detail::valid_image({&pixel, 1, 1, 1, PixelFormat::GRAY8}));
    CHECK(!s2_s3::detail::valid_image({&pixel, 1, 1, 1, PixelFormat::RGB8}));
    CHECK(!s2_s3::detail::valid_image({nullptr, 1, 1, 1, PixelFormat::GRAY8}));
    CHECK(!s2_s3::detail::valid_image({&pixel, 4, 1, 3, PixelFormat::GRAY8}));
    std::puts("S2/S3 host configuration checks passed");
}
