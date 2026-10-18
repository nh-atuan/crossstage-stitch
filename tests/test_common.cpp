#include <cstring>

#include "check.hpp"
#include "common/geometry.hpp"

using namespace stitch;

static void test_scale_info() {
    ScaleInfo si;
    CHECK(make_scale_info(512, 384, 0.5, &si) == Status::OK);
    CHECK(si.work_w == 256 && si.work_h == 192);
    CHECK_NEAR(si.sx, 0.5, 0);
    CHECK_NEAR(si.D[2], -0.25, 1e-15);
    CHECK_NEAR(si.D[5], -0.25, 1e-15);

    CHECK(make_scale_info(500, 333, 0.75, &si) == Status::OK);
    CHECK(si.work_w == 375 && si.work_h == 250);  // lround(249.75)
    CHECK_NEAR(si.sy, 250.0 / 333.0, 1e-15);

    CHECK(make_scale_info(100, 100, 1.0, &si) == Status::OK);  // S = 1 -> D = I
    const double ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    CHECK(homography_max_diff(si.D, ident) == 0.0);

    CHECK(make_scale_info(1, 1, 0.1, &si) == Status::OK);  // never collapses to 0
    CHECK(si.work_w == 1 && si.work_h == 1);

    CHECK(make_scale_info(100, 100, 0.0, &si) == Status::INVALID_ARGUMENT);
    CHECK(make_scale_info(100, 100, 1.5, &si) == Status::INVALID_ARGUMENT);
    CHECK(make_scale_info(0, 100, 0.5, &si) == Status::INVALID_ARGUMENT);
    CHECK(make_scale_info(100, 100, 0.5, nullptr) == Status::INVALID_ARGUMENT);
}

static void apply(const double* m, const double* p, double* out) {
    for (int i = 0; i < 3; ++i) out[i] = m[3 * i] * p[0] + m[3 * i + 1] * p[1] + m[3 * i + 2] * p[2];
}

// D_A * H * p must equal H_s * D_B * p (projectively) for a point p of B in original coordinates.
static void test_to_original() {
    ScaleInfo a, b;
    CHECK(make_scale_info(640, 480, 0.5, &a) == Status::OK);
    CHECK(make_scale_info(640, 480, 0.25, &b) == Status::OK);
    const double h_s[9] = {1.02, 0.01, 7.0, -0.02, 0.98, -3.0, 1e-5, 2e-5, 1.0};
    double h[9];
    CHECK(to_original(h_s, a.D, b.D, h) == Status::OK);

    const double p[3] = {321.0, 123.0, 1.0};
    double t[3], lhs[3], pb[3], rhs[3];
    apply(h, p, t);
    apply(a.D, t, lhs);
    apply(b.D, p, pb);
    apply(h_s, pb, rhs);
    for (int i = 0; i < 2; ++i) CHECK_NEAR(lhs[i] / lhs[2], rhs[i] / rhs[2], 1e-9);

    const double singular[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    CHECK(to_original(h_s, singular, b.D, h) == Status::DEGENERATE);
}

static void test_inverse_and_status() {
    const double m[9] = {2, 1, 0, 0, 3, 1, 1, 0, 4};
    double inv[9], prod[9];
    CHECK(mat3_inverse(m, inv));
    mat3_mul(m, inv, prod);
    const double ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    CHECK(homography_max_diff(prod, ident) < 1e-12);
    CHECK(std::strcmp(status_name(Status::DEGENERATE), "DEGENERATE") == 0);
    CHECK(std::strcmp(status_name(Status::OK), "OK") == 0);
}

int main() {
    test_scale_info();
    test_to_original();
    test_inverse_and_status();
    std::puts("test_common: all passed");
    return 0;
}
