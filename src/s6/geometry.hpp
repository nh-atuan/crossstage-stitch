#pragma once

#include <cstddef>

#include "common/types.hpp"

namespace stitch::s6 {

// Stage-local numerical/resource policy, not the offline evaluator's QoR policy.
struct GeometryPolicy {
    double denominator_tolerance = 1e-12;
    double inverse_tolerance = 1e-12;
    // Absolute pixels, with an additional scale-aware floating-point roundoff floor.
    double snap_tolerance = 1e-10;
    std::size_t max_canvas_pixels = 16777216;
};

struct Geometry {
    int width = 0;
    int height = 0;
    // Original A coordinates at canvas pixel (0, 0), not the A->canvas offset.
    // A source coordinates are (xc + origin_x, yc + origin_y).
    int origin_x = 0;
    int origin_y = 0;
    int width_a = 0;
    int height_a = 0;
    int width_b = 0;
    int height_b = 0;
    double denominator_tolerance = 1e-12;
    double snap_tolerance = 1e-10;
    // A-original -> B-original, row-major, normalized by its largest coefficient.
    // Always perform projective division; inverse[8] need not equal 1.
    double inverse[9] = {};
};

// H is B-original -> A-original. Pixel centers span [0, width-1] x [0, height-1].
// The inclusive canvas is floor(min corner) .. ceil(max corner), after snapping.
// Conditioning is tested after centering and isotropically scaling each image
// by max(width-1, height-1, 1): ||H_normalized||_inf * ||H_normalized^-1||_inf
// must be smaller than 1/inverse_tolerance. This is a numerical guard, not QoR.
// On failure, *out is cleared. No image allocation or CUDA operation is performed;
// the caller must additionally budget image bytes and blender workspace.
Status prepare_geometry(int width_a, int height_a, int width_b, int height_b,
                        const double* h, const GeometryPolicy& policy, Geometry* out);

}  // namespace stitch::s6
