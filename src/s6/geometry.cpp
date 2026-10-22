#include "s6/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace stitch::s6 {
namespace {

using Real = long double;

bool valid_policy(const GeometryPolicy& p) {
    return std::isfinite(p.denominator_tolerance) && p.denominator_tolerance > 0 &&
           p.denominator_tolerance < 1 && std::isfinite(p.inverse_tolerance) &&
           p.inverse_tolerance > 0 && p.inverse_tolerance < 1 &&
           std::isfinite(p.snap_tolerance) && p.snap_tolerance >= 0 &&
           p.snap_tolerance < 0.5 && p.max_canvas_pixels > 0;
}

void multiply(const Real* a, const Real* b, Real* result) {
    Real tmp[9] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                tmp[3 * r + c] += a[3 * r + k] * b[3 * k + c];
    std::copy(tmp, tmp + 9, result);
}

bool normalize(Real* m) {
    Real largest = 0;
    for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(m[i])) return false;
        largest = std::max(largest, std::abs(m[i]));
    }
    if (largest == 0) return false;
    for (int i = 0; i < 9; ++i) m[i] /= largest;
    return true;
}

Real infinity_norm(const Real* m) {
    Real norm = 0;
    for (int r = 0; r < 3; ++r) {
        Real row_sum = 0;
        for (int c = 0; c < 3; ++c) row_sum += std::abs(m[3 * r + c]);
        norm = std::max(norm, row_sum);
    }
    return norm;
}

// Scaled partial pivoting, not an absolute determinant or h33 threshold.
bool invert(const Real* input, Real tolerance, Real* inverse) {
    Real a[3][6] = {};
    Real scale[3] = {};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            a[r][c] = input[3 * r + c];
            scale[r] = std::max(scale[r], std::abs(a[r][c]));
        }
        if (scale[r] == 0) return false;
        a[r][r + 3] = 1;
    }
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        Real best = 0;
        for (int r = col; r < 3; ++r) {
            const Real score = std::abs(a[r][col]) / scale[r];
            if (score > best) {
                best = score;
                pivot = r;
            }
        }
        if (!std::isfinite(best) || best <= tolerance) return false;
        if (pivot != col) {
            for (int c = 0; c < 6; ++c) std::swap(a[pivot][c], a[col][c]);
            std::swap(scale[pivot], scale[col]);
        }
        const Real divisor = a[col][col];
        for (int c = 0; c < 6; ++c) a[col][c] /= divisor;
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const Real factor = a[r][col];
            for (int c = 0; c < 6; ++c) a[r][c] -= factor * a[col][c];
        }
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            const Real value = a[r][c + 3];
            if (!std::isfinite(value)) return false;
            inverse[3 * r + c] = value;
        }
    const Real condition = infinity_norm(input) * infinity_norm(inverse);
    return std::isfinite(condition) && condition * tolerance < 1;
}

Real snap_integer(Real x, const GeometryPolicy& p) {
    const Real rounded = std::round(x);
    const Real numerical_floor = 16 * std::numeric_limits<double>::epsilon() *
                                 std::max(Real(1), std::abs(x));
    const Real tolerance = std::max(Real(p.snap_tolerance), numerical_floor);
    return std::abs(x - rounded) <= tolerance ? rounded : x;
}

}  // namespace

Status prepare_geometry(int width_a, int height_a, int width_b, int height_b,
                        const double* h, const GeometryPolicy& policy, Geometry* out) {
    if (!out) return Status::INVALID_ARGUMENT;
    // Copy first so even an input H aliasing a previous out->inverse is safe.
    Real matrix[9] = {};
    if (h) std::copy(h, h + 9, matrix);
    *out = Geometry{};
    if (!h || width_a <= 0 || height_a <= 0 || width_b <= 0 || height_b <= 0 ||
        !valid_policy(policy))
        return Status::INVALID_ARGUMENT;
    if (!normalize(matrix)) return Status::DEGENERATE;

    const Real bx = Real(width_b) - 1;
    const Real by = Real(height_b) - 1;
    const Real denominator_scale = std::abs(matrix[6]) * bx +
                                   std::abs(matrix[7]) * by + std::abs(matrix[8]);
    if (!(denominator_scale > 0) || !std::isfinite(denominator_scale))
        return Status::DEGENERATE;
    const Real denominator_floor = policy.denominator_tolerance * denominator_scale;
    Real xmin = 0, ymin = 0;
    Real xmax = Real(width_a) - 1, ymax = Real(height_a) - 1;
    int denominator_sign = 0;
    for (int corner = 0; corner < 4; ++corner) {
        const Real x = (corner & 1) ? bx : 0;
        const Real y = (corner & 2) ? by : 0;
        const Real w = matrix[6] * x + matrix[7] * y + matrix[8];
        if (!std::isfinite(w) || std::abs(w) <= denominator_floor)
            return Status::DEGENERATE;
        const int sign = w > 0 ? 1 : -1;
        if (denominator_sign && sign != denominator_sign) return Status::DEGENERATE;
        denominator_sign = sign;
        const Real px = (matrix[0] * x + matrix[1] * y + matrix[2]) / w;
        const Real py = (matrix[3] * x + matrix[4] * y + matrix[5]) / w;
        if (!std::isfinite(px) || !std::isfinite(py)) return Status::DEGENERATE;
        xmin = std::min(xmin, px);
        ymin = std::min(ymin, py);
        xmax = std::max(xmax, px);
        ymax = std::max(ymax, py);
    }

    xmin = std::floor(snap_integer(xmin, policy));
    ymin = std::floor(snap_integer(ymin, policy));
    xmax = std::ceil(snap_integer(xmax, policy));
    ymax = std::ceil(snap_integer(ymax, policy));
    const Real width = xmax - xmin + 1;
    const Real height = ymax - ymin + 1;
    constexpr int int_max = std::numeric_limits<int>::max();
    constexpr int int_min = std::numeric_limits<int>::min();
    if (xmin < int_min || ymin < int_min || xmin > int_max || ymin > int_max ||
        !std::isfinite(width) || !std::isfinite(height) || width < 1 || height < 1 ||
        width > int_max || height > int_max)
        return Status::INSUFFICIENT_WORKSPACE;
    const auto cw = static_cast<std::size_t>(width);
    const auto ch = static_cast<std::size_t>(height);
    if (cw > std::numeric_limits<std::size_t>::max() / ch ||
        cw > policy.max_canvas_pixels / ch)
        return Status::INSUFFICIENT_WORKSPACE;

    // Coordinate normalization makes the condition check relative to image size,
    // rather than incorrectly declaring an ordinary pixel translation singular.
    const Real sa = std::max({Real(1), Real(width_a) - 1, Real(height_a) - 1});
    const Real sb = std::max({Real(1), bx, by});
    const Real ax_center = (Real(width_a) - 1) / 2;
    const Real ay_center = (Real(height_a) - 1) / 2;
    const Real na[9] = {1 / sa, 0, -ax_center / sa,
                        0, 1 / sa, -ay_center / sa, 0, 0, 1};
    const Real nb_inverse[9] = {sb, 0, bx / 2, 0, sb, by / 2, 0, 0, 1};
    Real temp[9], normalized_h[9], normalized_inverse[9], original_inverse[9];
    multiply(na, matrix, temp);
    multiply(temp, nb_inverse, normalized_h);
    if (!normalize(normalized_h) ||
        !invert(normalized_h, policy.inverse_tolerance, normalized_inverse))
        return Status::DEGENERATE;
    multiply(nb_inverse, normalized_inverse, temp);
    multiply(temp, na, original_inverse);
    if (!normalize(original_inverse)) return Status::DEGENERATE;

    Geometry result{};
    result.width = static_cast<int>(width);
    result.height = static_cast<int>(height);
    result.origin_x = static_cast<int>(xmin);
    result.origin_y = static_cast<int>(ymin);
    result.width_a = width_a;
    result.height_a = height_a;
    result.width_b = width_b;
    result.height_b = height_b;
    result.denominator_tolerance = policy.denominator_tolerance;
    result.snap_tolerance = policy.snap_tolerance;
    for (int i = 0; i < 9; ++i) {
        result.inverse[i] = static_cast<double>(original_inverse[i]);
        if (!std::isfinite(result.inverse[i])) return Status::DEGENERATE;
    }
    *out = result;
    return Status::OK;
}

}  // namespace stitch::s6
