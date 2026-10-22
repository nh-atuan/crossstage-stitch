// S6 renders original RGB images using the host-prepared B -> A geometry.
// No allocations or synchronizations occur in render(). Buffers must stay alive
// until the caller synchronizes the supplied stream and checks completion.
#pragma once

#include <cstddef>
#include "common/types.hpp"
#include "s6/geometry.hpp"

namespace stitch::s6 {

enum class BlendMode : int { Feather = 0, Multiband3 = 1, Multiband5 = 2 };

// Requested reductions, capped at the number needed to reach a 1x1 canvas.
int effective_reductions(const Geometry& geometry, BlendMode mode);

// Feather is fused and needs zero workspace. Multiband workspace is tightly
// packed FP32 pyramids with 256-byte aligned subarrays. Overflow is rejected.
Status s6_workspace_bytes(const Geometry& geometry, BlendMode mode,
                          std::size_t* bytes);

// a, b: device RGB8 images in original coordinates. output: device RGB8 canvas;
// mask: device GRAY8 union mask, encoded as 0 or 1. Each pitch is in bytes.
// geometry must be the unchanged result of prepare_geometry() for a/b shapes.
// output, mask and workspace must not overlap each other or either input;
// a/b may alias each other because they are read-only. Workspace, when needed,
// must be at least the queried size and aligned to 256 bytes (cudaMalloc does so).
// Device-pointer ownership, allocation extent, stream lifetime and completion
// error checks remain the caller's responsibility. No device availability or
// color-accuracy/performance claim follows from successful host-only tests.
Status render(const ImageView& a, const ImageView& b, const Geometry& geometry,
              BlendMode mode, const ImageView& output, const ImageView& mask,
              void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace stitch::s6
