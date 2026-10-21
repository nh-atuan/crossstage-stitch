// Plain-data structs exchanged between pipeline stages (S1..S6).
// Caller allocates every buffer; stages never allocate (see docs/conventions.md).
#pragma once

#include <cstddef>
#include <cstdint>

#include "common/cuda_fwd.hpp"
#include "common/status.hpp"

namespace stitch {

enum class PixelFormat : int { RGB8 = 0, GRAY8 = 1 };

// Device image, interleaved, row pitch in bytes.
struct ImageView {
    uint8_t* data;
    int width;
    int height;
    int pitch_bytes;
    PixelFormat format;
};

// Output of S1 for one image (host). D maps original coordinates to S1 coordinates.
struct ScaleInfo {
    int orig_w, orig_h;
    int work_w, work_h;
    double sx, sy;  // work_w / orig_w, work_h / orig_h
    double D[9];    // 3x3 row-major
};

// Device SoA keypoints in S1 coordinates. Valid entries: [0, *d_count), d_count is a device pointer.
struct Keypoints {
    float* x;
    float* y;
    float* response;
    int capacity;
    int* d_count;
};

enum class DescriptorType : int { BINARY_U8 = 0, FLOAT32 = 1 };

// Device descriptors, one per row. dim = bytes (BINARY_U8) or floats (FLOAT32).
struct Descriptors {
    DescriptorType type;
    int dim;
    int row_stride_bytes;
    void* data;
};

// Device SoA matches (index into image A keypoints, image B keypoints, descriptor distance).
struct Matches {
    int32_t* idx_a;
    int32_t* idx_b;
    float* distance;
    int capacity;
    int* d_count;
};

// Output of S5 (host). H maps B -> A in original RGB coordinates; H_s is in S1 coordinates.
struct HomographyResult {
    double H[9];
    double H_s[9];
    int num_matches;
    int num_inliers;
    double inlier_ratio;
    double inlier_coverage;
    double reproj_residual;
    Status status;
};

// Per-stage times in ms, measured with CUDA events on the pipeline stream.
// t_s1..t_s3 cover both images (event before image A, event after image B).
struct StageTimings {
    double t_h2d, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6;
    double t_controller, t_d2h, t_e2e;
};

}  // namespace stitch
