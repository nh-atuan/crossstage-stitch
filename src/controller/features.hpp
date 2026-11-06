// Feature layout is shared by CUDA collection, exported trees and telemetry.
#pragma once
#include <array>
#include <cmath>
#include "common/types.hpp"
#include "s4/s4.hpp"
#include "controller/relational_features.hpp"

namespace stitch::controller {
constexpr int scene_feature_count = 32;
constexpr int selector_feature_count = 30;
constexpr int gate_feature_count = 38;
constexpr int feature_device_count = gate_feature_count+relation_workspace_count;
using SceneFeatures = std::array<double, scene_feature_count>;
using GateFeatures = std::array<double, gate_feature_count>;
inline std::array<double,selector_feature_count> selector_features(const SceneFeatures& scene) noexcept {
    std::array<double,selector_feature_count> values{};
    for (int i=0;i<26;++i) values[i]=scene[i];
    for (int tile=0;tile<4;++tile) values[26+tile]=std::abs(scene[2+tile*3]-scene[14+tile*3]);
    return values;
}

// Caller provides feature_device_count device doubles and gate_feature_count
// pinned doubles. Without relational collection, 38 device doubles suffice.
Status extract_scene(const ImageView& a, const ImageView& b, double* device,
                     double* pinned, cudaStream_t stream, bool relations=false);
Status extract_gate(const ImageView& a, const ImageView& b,
                    const Keypoints& ka, const Keypoints& kb, const Matches& matches,
                    const std::uint8_t* mask, const HomographyResult& h,
                    const ScaleInfo& sa, const ScaleInfo& sb,
                    double* device, double* pinned, cudaStream_t stream,
                    const s4::Top2* forward=nullptr, unsigned relation_mask=0);
}  // namespace stitch::controller
