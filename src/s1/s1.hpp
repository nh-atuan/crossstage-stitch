// S1: preprocessing for cross-stage panorama stitching.
// Fused integer 15-bit RGB-to-GRAY conversion and Area Resizing box filter.
// Processes image A and image B in a single batched kernel launch (blockIdx.z).
// Zero dynamic allocations inside measurement window; no internal synchronization.
#pragma once

#include <cstddef>
#include "common/types.hpp"
#include "common/geometry.hpp"

namespace stitch::s1 {

// Host-only workspace query. Fused S1 kernel does not require device workspace.
inline Status s1_workspace_bytes(int orig_w, int orig_h, double scale, std::size_t* bytes) {
    if (bytes == nullptr || orig_w <= 0 || orig_h <= 0 || !(scale > 0.0) || scale > 1.0) {
        return Status::INVALID_ARGUMENT;
    }
    *bytes = 0;
    return Status::OK;
}

// Preprocesses a single RGB8 image into a GRAY8 image at scale S in (0.0, 1.0].
// scale_info is computed on host via stitch::make_scale_info().
// Caller pre-allocates out->data matching scale_info dimensions.
Status s1_preprocess_single(const ImageView& in, double scale,
                            ImageView* out, ScaleInfo* scale_info,
                            cudaStream_t stream);

// Preprocesses a pair of RGB8 images (A and B) into GRAY8 images at scale S in (0.0, 1.0].
// scale_a and scale_b are computed on host via stitch::make_scale_info().
// Launches a single fused kernel with 3D grid: blockIdx.z == 0 for image A, 1 for image B.
// Caller pre-allocates out_a->data and out_b->data matching scale_a/scale_b dimensions.
Status s1_preprocess(const ImageView& img_a, const ImageView& img_b,
                     double scale,
                     ImageView* out_a, ImageView* out_b,
                     ScaleInfo* scale_a, ScaleInfo* scale_b,
                     cudaStream_t stream);

}  // namespace stitch::s1
