// Exported bounded trees have no runtime library dependency or allocation.
#pragma once
#include <cmath>
#include <cstddef>

namespace stitch::controller {
struct TreeNode {
    int feature{-1};
    double threshold{};
    int left{-1}, right{-1}, value{-1};
};
struct TreeView {
    const TreeNode* nodes{};
    std::size_t count{};
};
inline bool valid_tree(TreeView tree, int features, int outputs) noexcept {
    if (!tree.nodes || !tree.count || tree.count > 127) return false;
    for (std::size_t i = 0; i < tree.count; ++i) {
        const auto& n = tree.nodes[i];
        if (n.feature == -1) {
            if (n.value < 0 || n.value >= outputs || n.left != -1 || n.right != -1) return false;
        } else if (n.feature < 0 || n.feature >= features || !std::isfinite(n.threshold) ||
                   n.left <= static_cast<int>(i) || n.right <= static_cast<int>(i) ||
                   n.left >= static_cast<int>(tree.count) || n.right >= static_cast<int>(tree.count) ||
                   n.value != -1) return false;
    }
    return true;
}
inline int predict_tree(TreeView tree, const double* values, int features) noexcept {
    for (int f = 0; f < features; ++f) if (!std::isfinite(values[f])) return -1;
    int index = 0;
    for (std::size_t step = 0; step < tree.count; ++step) {
        if (index < 0 || index >= static_cast<int>(tree.count)) return -1;
        const auto& n = tree.nodes[index];
        if (n.feature == -1) return n.value;
        if (n.feature < 0 || n.feature >= features) return -1;
        index = values[n.feature] <= n.threshold ? n.left : n.right;
    }
    return -1;
}
}  // namespace stitch::controller
