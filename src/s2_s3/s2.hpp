// S2: single-level ORB keypoint detection (FAST-9 + Harris ranking, top-K).
// Method and tolerances: docs/decisions/s2-s3-orb.md. No allocation or
// internal synchronization.
#pragma once

#include <cstddef>

#include "common/types.hpp"

namespace stitch::s2 {

// Keypoints keep this distance from every image border (OpenCV edgeThreshold),
// so S3 can sample its full 31x31 patch without bounds checks.
constexpr int kEdge = 31;
// FAST pre-selection keeps kCandidateFactor * K points before Harris ranking.
constexpr int kCandidateFactor = 2;

// No tuned defaults: zero-initialization is invalid.
struct Config {
    int fast_threshold;  // FAST intensity threshold, 1..254
    int max_keypoints;   // K >= 0; also bounds the Harris and S4 workload
    float harris_k;      // finite, >= 0 (OpenCV ORB uses 0.04)
};

// Host-only. Bytes depend on the image size and K. Zero for K == 0.
Status s2_workspace_bytes(int width, int height, const Config& config,
                          std::size_t* bytes);

// image: device GRAY8 from S1 (sigma = 0), pitch in bytes.
// output: device SoA with capacity >= config.max_keypoints; x and y must be
// non-null when capacity > 0, response may be null; d_count non-null.
// Writes min(found, K) keypoints sorted by Harris response (descending), ties
// by row-major pixel index; *d_count is written on the stream. Coordinates
// are integer pixel centres in S1 coordinates.
// Workspace must be 256-byte aligned with at least the queried bytes and must
// not overlap the image or outputs. Buffers stay alive until stream completion.
// Returns launch errors only; images smaller than 2 * kEdge + 1 give zero points.
Status s2_detect(const ImageView& image, const Config& config, Keypoints output,
                 void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace stitch::s2
