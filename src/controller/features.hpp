// Feature layout is shared by CUDA collection, exported trees and telemetry.
#pragma once
#include <array>
#include "common/types.hpp"

namespace stitch::controller {
constexpr int scene_feature_count = 26;
constexpr int gate_feature_count = 24;
using SceneFeatures = std::array<double, scene_feature_count>;
using GateFeatures = std::array<double, gate_feature_count>;

// Caller provides 26/24 device doubles and pinned host doubles. No allocation.
Status extract_scene(const ImageView& a, const ImageView& b, double* device,
                     double* pinned, cudaStream_t stream);
Status extract_gate(const ImageView& a, const ImageView& b,
                    const Keypoints& ka, const Keypoints& kb, const Matches& matches,
                    const std::uint8_t* mask, const HomographyResult& h,
                    const ScaleInfo& sa, const ScaleInfo& sb,
                    double* device, double* pinned, cudaStream_t stream);
}  // namespace stitch::controller
