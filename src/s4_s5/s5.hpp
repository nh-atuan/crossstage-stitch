// S5: batched MSAC homography B -> A. No allocation or internal synchronization.
#pragma once

#include <cstddef>
#include <cstdint>
#include "common/types.hpp"

namespace stitch::s5 {

// All operating parameters are explicit; test values are not tuned defaults.
struct Config {
    int hypotheses;             // Total R requested, including previous_hypotheses.
    int previous_hypotheses;    // Zero restarts; otherwise extend the saved raw search.
    int batch_size;
    int refinement_steps;
    double threshold;           // Forward reprojection distance in S1 pixels.
    std::uint64_t seed;
    std::uint64_t generation;    // Change whenever input data or scale metadata changes.
};

// Workspace is persistent across R extensions and must be cudaMalloc-aligned (256).
// Size depends only on batch_size, not R. Host-only callable.
Status s5_workspace_bytes(const Config& config, std::size_t* bytes);

// keypoint/match counts and arrays, mask and workspace are device buffers.
// mask has matches.capacity bytes. Output must be pinned host memory; it is valid
// only after caller synchronizes stream. cudaGetLastError reports launch errors;
// output.status additionally reports data errors, degeneracy and invalid reuse.
// Inputs, scales, batch size, seed, threshold and generation must remain unchanged
// during an extension. Same stream required; no concurrent workspace use.
// Buffers must cover their declared capacities and must not alias writable
// outputs/workspace. All buffers and pinned output stay alive until completion.
// previous_hypotheses must equal the last successfully completed R in workspace.
// Refinement never overwrites the saved raw winner: split/full R agree.
// Caller resets with previous_hypotheses=0 after S/K/input changes or CUDA errors.
Status estimate(const Keypoints& a, const Keypoints& b, const Matches& matches,
                const ScaleInfo& scale_a, const ScaleInfo& scale_b, const Config& config,
                std::uint8_t* inlier_mask, HomographyResult* output,
                void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace stitch::s5
