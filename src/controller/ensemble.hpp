// Scalar XGBoost-compatible inference. Strict float32 splits, no allocation.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace stitch::controller {
struct BoostNode {
    int feature{-1}, left{-1}, right{-1};
    float threshold{}, leaf{};
};
struct EnsembleView {
    const BoostNode* nodes{};
    std::size_t node_count{};
    const std::size_t* roots{};
    std::size_t tree_count{};
    int features{};
    float bias{};
    double slope{1.}, intercept{};
    bool exponential{};
};
inline bool valid_ensemble(const EnsembleView& e, int features, bool exponential) noexcept {
    if (!e.nodes || !e.roots || !e.tree_count || e.tree_count>128 ||
        !e.node_count || e.node_count>e.tree_count*127 || e.features!=features ||
        e.exponential!=exponential || !std::isfinite(e.bias) ||
        !std::isfinite(e.slope) || e.slope<0 || !std::isfinite(e.intercept) || e.roots[0]!=0)
        return false;
    for (std::size_t t=0;t<e.tree_count;++t) {
        const std::size_t start=e.roots[t],end=t+1<e.tree_count?e.roots[t+1]:e.node_count;
        if (start>=end || end>e.node_count || end-start>127) return false;
        for (std::size_t i=start;i<end;++i) {
            const auto& n=e.nodes[i];
            if (n.feature==-1) {
                if (n.left!=-1 || n.right!=-1 || !std::isfinite(n.leaf)) return false;
            } else if (n.feature<0 || n.feature>=features || !std::isfinite(n.threshold) ||
                n.left<=static_cast<int>(i) || n.right<=static_cast<int>(i) ||
                n.left>=static_cast<int>(end) || n.right>=static_cast<int>(end)) return false;
        }
    }
    return true;
}
inline double predict_ensemble(const EnsembleView& e, const double* x) noexcept {
    for (int f=0;f<e.features;++f)
        if (!std::isfinite(x[f]) || std::abs(x[f])>std::numeric_limits<float>::max())
            return std::numeric_limits<double>::quiet_NaN();
    float total=e.bias;
    for (std::size_t t=0;t<e.tree_count;++t) {
        auto i=e.roots[t];
        for (int step=0;step<127;++step) {
            const auto& n=e.nodes[i];
            if (n.feature==-1) {total+=n.leaf;break;}
            i=static_cast<float>(x[n.feature])<n.threshold?n.left:n.right;
        }
    }
    if (e.exponential) return std::exp(std::clamp(static_cast<double>(total),-20.,20.));
    const double z=e.slope*total+e.intercept;
    return z>=0 ? 1./(1.+std::exp(-z)) : std::exp(z)/(1.+std::exp(z));
}
struct PooledModels {
    EnsembleView pre_quality{}, post_quality{}, latency{};
    double pre_threshold{}, gate_threshold{}, extra_ms{};
    std::size_t reference_index{};
    bool pair_relations{};
    unsigned relation_mask{};
    // Optional calibration-only stage profile. Empty preserves legacy scoring.
    const double* fallback_fractions{};
    std::size_t fallback_count{};
};
}  // namespace stitch::controller
