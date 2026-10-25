// S3: oriented rBRIEF descriptors (OpenCV ORB, WTA_K = 2, 31x31 patch).
// Method and tolerances: docs/decisions/s2-s3-orb.md. No allocation or
// internal synchronization.
#pragma once

#include <cstddef>

#include "common/types.hpp"

namespace stitch::s3 {

constexpr int kDescriptorBytes = 32;  // 256 bits, DescriptorType::BINARY_U8

// Host-only. Holds the blurred image and the row-pass buffer.
Status s3_workspace_bytes(int width, int height, std::size_t* bytes);

// image: the same device GRAY8 image given to S2 (unblurred); S3 blurs it
// internally with a 7x7, sigma 2 Gaussian before sampling the pattern.
// keypoints: S2 output on the same stream; rows [0, *d_count) are described.
// output: BINARY_U8, dim 32, row_stride_bytes >= 32, keypoints.capacity rows.
// Keypoints closer than s2::kEdge to a border get an all-zero row.
// Workspace must be 256-byte aligned and must not overlap inputs or output.
// Returns launch errors only; buffers stay alive until stream completion.
Status s3_describe(const ImageView& image, const Keypoints& keypoints,
                   Descriptors output, void* workspace, std::size_t workspace_bytes,
                   cudaStream_t stream);

}  // namespace stitch::s3
