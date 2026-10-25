// S6 renders original RGB images using the host-prepared B -> A geometry.
// No allocations or synchronizations occur in render(). Buffers must stay alive
// until the caller synchronizes the supplied stream and checks completion.
#pragma once

#include <cstddef>
#include <cstdint>
#include "common/types.hpp"
#include "s6/geometry.hpp"

namespace stitch::s6 {

enum class BlendMode : int { Feather = 0, Multiband3 = 1, Multiband5 = 2 };

// Optional caller-owned device counters; validity, not RGB content, defines counts.
struct alignas(8) RenderStatistics {
    std::uint64_t union_pixels;
    std::uint64_t overlap_pixels;
};

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
// statistics, when non-null, points to a separate aligned device RenderStatistics.
// It must not overlap any image or supplied workspace. Render resets it on the
// stream, then uses block-reduced counters from warp validity. Read it only after
// checked stream completion; keep it alive until then. Telemetry adds one async
// memset and block reductions/atomics; callers measuring it must include this cost.
// A null pointer selects the original path without telemetry kernels/reductions.
// Device-pointer ownership, allocation extent, stream lifetime and completion
// error checks remain the caller's responsibility. No device availability or
// color-accuracy/performance claim follows from successful host-only tests.
Status render(const ImageView& a, const ImageView& b, const Geometry& geometry,
              BlendMode mode, const ImageView& output, const ImageView& mask,
              void* workspace, std::size_t workspace_bytes, cudaStream_t stream,
              RenderStatistics* statistics = nullptr);

}  // namespace stitch::s6
