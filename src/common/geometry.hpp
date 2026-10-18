// Host-side geometry helpers shared by all stages and tests.
// Matrices are 3x3 row-major double, column-vector convention.
#pragma once

#include <algorithm>
#include <cmath>

#include "common/types.hpp"

namespace stitch {

// Output size of S1: W_s = max(1, lround(W * S)), computed in double. Requires 0 < S <= 1.
inline Status make_scale_info(int w, int h, double s, ScaleInfo* out) {
    if (out == nullptr || w <= 0 || h <= 0 || !(s > 0.0) || s > 1.0) return Status::INVALID_ARGUMENT;
    out->orig_w = w;
    out->orig_h = h;
    out->work_w = static_cast<int>(std::max(1L, std::lround(static_cast<double>(w) * s)));
    out->work_h = static_cast<int>(std::max(1L, std::lround(static_cast<double>(h) * s)));
    out->sx = static_cast<double>(out->work_w) / w;
    out->sy = static_cast<double>(out->work_h) / h;
    // x_s = sx * (x + 0.5) - 0.5  ->  translation 0.5 * (sx - 1)
    const double d[9] = {out->sx, 0, 0.5 * (out->sx - 1.0), 0, out->sy, 0.5 * (out->sy - 1.0), 0, 0, 1};
    std::copy(d, d + 9, out->D);
    return Status::OK;
}

inline void mat3_mul(const double* a, const double* b, double* out) {
    double r[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    std::copy(r, r + 9, out);
}

// Returns false when the matrix is (numerically) singular.
inline bool mat3_inverse(const double* m, double* out) {
    const double c00 = m[4] * m[8] - m[5] * m[7];
    const double c01 = m[5] * m[6] - m[3] * m[8];
    const double c02 = m[3] * m[7] - m[4] * m[6];
    const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if (std::fabs(det) < 1e-300) return false;
    const double inv = 1.0 / det;
    const double r[9] = {c00 * inv, (m[2] * m[7] - m[1] * m[8]) * inv, (m[1] * m[5] - m[2] * m[4]) * inv,
                         c01 * inv, (m[0] * m[8] - m[2] * m[6]) * inv, (m[2] * m[3] - m[0] * m[5]) * inv,
                         c02 * inv, (m[1] * m[6] - m[0] * m[7]) * inv, (m[0] * m[4] - m[1] * m[3]) * inv};
    std::copy(r, r + 9, out);
    return true;
}

// H = D_A^-1 * H_s * D_B : homography in S1 coordinates -> original coordinates (B -> A).
inline Status to_original(const double* h_s, const double* d_a, const double* d_b, double* h_out) {
    double d_a_inv[9], tmp[9];
    if (!mat3_inverse(d_a, d_a_inv)) return Status::DEGENERATE;
    mat3_mul(d_a_inv, h_s, tmp);
    mat3_mul(tmp, d_b, h_out);
    return Status::OK;
}

// Max |a/a33 - b/b33| over the 9 entries; use to compare homographies in tests.
inline double homography_max_diff(const double* a, const double* b) {
    double worst = 0.0;
    for (int i = 0; i < 9; ++i) worst = std::max(worst, std::fabs(a[i] / a[8] - b[i] / b[8]));
    return worst;
}

}  // namespace stitch
