// Optional candidate observations. Kernels allocate nothing and preserve S1-S6.
#pragma once
#include "common/types.hpp"
#include "s4/s4.hpp"

namespace stitch::controller {
constexpr int pair_relation_count=6;
constexpr int alignment_relation_count=14;
// Output: peak NCC, peak gap, normalized dx/dy, zero-shift NCC, peak overlap.
Status pair_relations(const ImageView& a, const ImageView& b, double* output,
                      cudaStream_t stream);
// Output: residual p50/p90 (0.125 px bins), regional max/std mean residual,
// match-distance mean/p90, forward top2 ratio mean/ambiguity fraction,
// uniform 7x7 patch SSIM mean/min, edge MAE/cosine, valid patch fraction,
// p90 contrast-normalized patch MAE. Distance units assume 256-bit ORB.
// The optional top2 view must cover a.capacity entries, already produced by S4.
Status alignment_relations(const ImageView& a, const ImageView& b,
    const Keypoints& ka, const Keypoints& kb, const Matches& matches,
    const std::uint8_t* mask, const HomographyResult& h, const ScaleInfo& sa,
    const ScaleInfo& sb, const s4::Top2* forward, double* output, cudaStream_t stream);
}  // namespace stitch::controller
