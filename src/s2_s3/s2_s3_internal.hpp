// Host-side validation and workspace layouts shared by the S2/S3 config
// library and the CUDA kernels. Not part of the stage interface.
#pragma once

#include <cstddef>
#include <cstdint>

#include "s2_s3/s2.hpp"
#include "s2_s3/s3.hpp"

namespace stitch::s2_s3::detail {

constexpr std::size_t kAlignment = 256;
constexpr int kScanItems = 1024;  // pixels per block in the candidate scan

// Valid GRAY8 view with width * height <= INT_MAX; data may be any pointer.
bool valid_image_shape(int width, int height);
bool valid_image(const ImageView& image);

struct S2Layout {
    int pixels;        // width * height
    int blocks;        // ceil(pixels / kScanItems)
    int candidates;    // kCandidateFactor * K
    // Byte offsets from the workspace base, each kAlignment-aligned.
    std::size_t scores;         // uint8 FAST score per pixel
    std::size_t maxima;         // uint8 NMS-surviving score per pixel
    std::size_t histogram;      // uint32[256] over maxima scores
    std::size_t params;         // SelectParams
    std::size_t block_counts;   // uint32[2 * blocks] (above, tie) per block
    std::size_t block_offsets;  // uint32[2 * blocks] exclusive prefix
    std::size_t keys;           // uint64[candidates] Harris sort keys
    std::size_t total;
};

// Selection state computed on the device from the histogram.
struct SelectParams {
    std::uint32_t threshold;   // FAST score at the boundary of the 2K pre-selection
    std::uint32_t above;       // maxima with score > threshold
    std::uint32_t ties;        // maxima with score == threshold that are kept
    std::uint32_t selected;    // above + ties <= candidates
};

bool valid_s2_config(const s2::Config& config);
// False on invalid arguments or size_t overflow.
bool make_s2_layout(int width, int height, int max_keypoints, S2Layout* layout);

struct S3Layout {
    std::size_t blurred;  // uint8 [height][width]
    std::size_t rows;     // float [height][width], horizontal pass
    std::size_t total;
};

bool make_s3_layout(int width, int height, S3Layout* layout);

}  // namespace stitch::s2_s3::detail
