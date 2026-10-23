// Small normalized homography solver shared by CUDA and host numerical tests.
#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define STITCH_S5_HD __host__ __device__
#else
#define STITCH_S5_HD
#endif

namespace stitch::s5::detail {

struct Point {
    double ax, ay, bx, by;
};

STITCH_S5_HD inline bool finite(double x) {
    return x == x && x <= 1.7976931348623157e308 && x >= -1.7976931348623157e308;
}

STITCH_S5_HD inline void multiply(const double* a, const double* b, double* out) {
    double t[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            t[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[j + 3] + a[3 * i + 2] * b[j + 6];
    for (int i = 0; i < 9; ++i) out[i] = t[i];
}

STITCH_S5_HD inline bool normalize(double* h) {
    double scale = 0.0;
    for (int i = 0; i < 9; ++i) {
        if (!finite(h[i])) return false;
        if (fabs(h[i]) > scale) scale = fabs(h[i]);
    }
    if (!(scale > 0.0)) return false;
    for (int i = 0; i < 9; ++i) h[i] /= scale;
    const double det = h[0] * (h[4] * h[8] - h[5] * h[7])
                     - h[1] * (h[3] * h[8] - h[5] * h[6])
                     + h[2] * (h[3] * h[7] - h[4] * h[6]);
    return finite(det) && fabs(det) > 1e-15;
}

STITCH_S5_HD inline double residual_squared(const double* h, Point p) {
    const double z = h[6] * p.bx + h[7] * p.by + h[8];
    if (!finite(z) || fabs(z) < 1e-15) return 1.7976931348623157e308;
    const double dx = (h[0] * p.bx + h[1] * p.by + h[2]) / z - p.ax;
    const double dy = (h[3] * p.bx + h[4] * p.by + h[5]) / z - p.ay;
    const double r = dx * dx + dy * dy;
    return finite(r) ? r : 1.7976931348623157e308;
}

// SplitMix64, Sebastiano Vigna; applied to a counter, not mutable RNG state.
// Source: https://prng.di.unimi.it/splitmix64.c (2026-10-22), public domain.
STITCH_S5_HD inline std::uint64_t random_word(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

// Each hypothesis has its own counter. Splitting R into calls cannot change samples.
STITCH_S5_HD inline void sample_four(int n, std::uint64_t seed, int hypothesis, int* sample) {
    for (int k = 0; k < 4; ++k) {
        int v = static_cast<int>(random_word(seed ^ (static_cast<std::uint64_t>(hypothesis) * 4 + k))
                                 % static_cast<std::uint64_t>(n - k));
        // Map an index in the remaining sorted set back into [0,n).
        for (int j = 0; j < k; ++j) {
            int smallest = n;
            for (int l = 0; l < k; ++l) {
                int rank = 0;
                for (int m = 0; m < k; ++m) rank += sample[m] < sample[l];
                if (rank == j) smallest = sample[l];
            }
            if (v >= smallest) ++v;
        }
        sample[k] = v;
    }
}

STITCH_S5_HD inline bool nondegenerate_four(const Point* p) {
    for (int image = 0; image < 2; ++image) {
        double scale = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < i; ++j) {
                const double dx = image ? p[i].bx - p[j].bx : p[i].ax - p[j].ax;
                const double dy = image ? p[i].by - p[j].by : p[i].ay - p[j].ay;
                const double d = dx * dx + dy * dy;
                if (d > scale) scale = d;
            }
        if (!(scale > 0.0) || !finite(scale)) return false;
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j)
                for (int k = j + 1; k < 4; ++k) {
                    const double x1 = image ? p[j].bx - p[i].bx : p[j].ax - p[i].ax;
                    const double y1 = image ? p[j].by - p[i].by : p[j].ay - p[i].ay;
                    const double x2 = image ? p[k].bx - p[i].bx : p[k].ax - p[i].ax;
                    const double y2 = image ? p[k].by - p[i].by : p[k].ay - p[i].ay;
                    if (fabs(x1 * y2 - y1 * x2) <= 1e-8 * scale) return false;
                }
    }
    return true;
}

struct Normalization {
    double ax, ay, bx, by, sa, sb;
};

template <class Points, class Keep>
STITCH_S5_HD bool fit(int n, const Points& points, const Keep& keep, double* out) {
    Normalization t{};
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (!keep(i)) continue;
        const Point p = points(i);
        t.ax += p.ax; t.ay += p.ay; t.bx += p.bx; t.by += p.by;
        ++count;
    }
    if (count < 4) return false;
    t.ax /= count; t.ay /= count; t.bx /= count; t.by /= count;
    double va = 0.0, vb = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!keep(i)) continue;
        const Point p = points(i);
        va += (p.ax - t.ax) * (p.ax - t.ax) + (p.ay - t.ay) * (p.ay - t.ay);
        vb += (p.bx - t.bx) * (p.bx - t.bx) + (p.by - t.by) * (p.by - t.by);
    }
    if (!(va > 0.0) || !(vb > 0.0) || !finite(va) || !finite(vb)) return false;
    t.sa = sqrt(2.0 * count / va);
    t.sb = sqrt(2.0 * count / vb);
    double a[8][9]{};
    for (int i = 0; i < n; ++i) {
        if (!keep(i)) continue;
        const Point p = points(i);
        const double x = (p.bx - t.bx) * t.sb, y = (p.by - t.by) * t.sb;
        const double u = (p.ax - t.ax) * t.sa, v = (p.ay - t.ay) * t.sa;
        const double rows[2][9]{{x, y, 1, 0, 0, 0, -u * x, -u * y, u},
                                {0, 0, 0, x, y, 1, -v * x, -v * y, v}};
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 9; ++c)
                a[r][c] += rows[0][r] * rows[0][c] + rows[1][r] * rows[1][c];
    }
    double max_entry = 0.0;
    for (int i = 0; i < 8; ++i)
        if (fabs(a[i][i]) > max_entry) max_entry = fabs(a[i][i]);
    for (int c = 0; c < 8; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 8; ++r)
            if (fabs(a[r][c]) > fabs(a[pivot][c])) pivot = r;
        if (!finite(a[pivot][c]) || fabs(a[pivot][c]) <= 1e-12 * max_entry) return false;
        for (int j = c; j < 9; ++j) {
            const double v = a[c][j]; a[c][j] = a[pivot][j]; a[pivot][j] = v;
        }
        const double divisor = a[c][c];
        for (int j = c; j < 9; ++j) a[c][j] /= divisor;
        for (int r = 0; r < 8; ++r) {
            if (r == c) continue;
            const double factor = a[r][c];
            for (int j = c; j < 9; ++j) a[r][j] -= factor * a[c][j];
        }
    }
    double h[9];
    for (int i = 0; i < 8; ++i) h[i] = a[i][8];
    h[8] = 1.0;
    const double inv_a[9]{1.0 / t.sa, 0, t.ax, 0, 1.0 / t.sa, t.ay, 0, 0, 1};
    const double b[9]{t.sb, 0, -t.sb * t.bx, 0, t.sb, -t.sb * t.by, 0, 0, 1};
    double tmp[9];
    multiply(inv_a, h, tmp);
    multiply(tmp, b, out);
    return normalize(out);
}

struct All {
    STITCH_S5_HD bool operator()(int) const { return true; }
};

STITCH_S5_HD inline bool valid_domain(const double* h, int width, int height) {
    const double z[4]{h[8], h[6] * (width - 1) + h[8], h[7] * (height - 1) + h[8],
                       h[6] * (width - 1) + h[7] * (height - 1) + h[8]};
    for (int i = 0; i < 4; ++i)
        if (!finite(z[i]) || fabs(z[i]) < 1e-15 || (z[i] > 0.0) != (z[0] > 0.0)) return false;
    return true;
}

}  // namespace stitch::s5::detail

#undef STITCH_S5_HD
