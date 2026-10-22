#include "s6/geometry.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using stitch::Status;
using stitch::s6::Geometry;
using stitch::s6::GeometryPolicy;
using stitch::s6::prepare_geometry;

namespace {

const double identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

void check_empty(const Geometry& g) {
    CHECK(g.width == 0 && g.height == 0 && g.origin_x == 0 && g.origin_y == 0);
    CHECK(g.width_a == 0 && g.height_a == 0 && g.width_b == 0 && g.height_b == 0);
    for (double value : g.inverse) CHECK(value == 0);
}

void check_inverse_point(const Geometry& g, double ax, double ay,
                         double bx, double by, double tolerance = 1e-10) {
    const double* m = g.inverse;
    const double w = m[6] * ax + m[7] * ay + m[8];
    CHECK(std::isfinite(w) && w != 0);
    CHECK_NEAR((m[0] * ax + m[1] * ay + m[2]) / w, bx, tolerance);
    CHECK_NEAR((m[3] * ax + m[4] * ay + m[5]) / w, by, tolerance);
}

void expect_failure(Status status, int wa, int ha, int wb, int hb,
                    const double* h, const GeometryPolicy& p = {}) {
    Geometry g;
    CHECK(prepare_geometry(2, 2, 2, 2, identity, {}, &g) == Status::OK);
    CHECK(prepare_geometry(wa, ha, wb, hb, h, p, &g) == status);
    check_empty(g);
}

void test_identity_and_dimensions() {
    Geometry g;
    CHECK(prepare_geometry(4, 3, 4, 3, identity, {}, &g) == Status::OK);
    CHECK(g.width == 4 && g.height == 3 && g.origin_x == 0 && g.origin_y == 0);
    CHECK(g.width_a == 4 && g.height_a == 3 && g.width_b == 4 && g.height_b == 3);
    check_inverse_point(g, 3, 2, 3, 2);
    CHECK(prepare_geometry(4, 3, 6, 2, identity, {}, &g) == Status::OK);
    CHECK(g.width == 6 && g.height == 3);
    CHECK(prepare_geometry(1, 1, 1, 1, identity, {}, &g) == Status::OK);
    CHECK(g.width == 1 && g.height == 1);
    check_inverse_point(g, 0, 0, 0, 0);
}

void test_translation_and_scale_equivalence() {
    const double translation[9] = {1, 0, -2, 0, 1, 0, 0, 0, 1};
    Geometry reference;
    CHECK(prepare_geometry(4, 3, 4, 3, translation, {}, &reference) == Status::OK);
    CHECK(reference.width == 6 && reference.height == 3);
    CHECK(reference.origin_x == -2 && reference.origin_y == 0);
    // Expected values are derived from x_A=x_B-2, not a production helper.
    check_inverse_point(reference, -2, 0, 0, 0);
    check_inverse_point(reference, 1, 2, 3, 2);
    for (double factor : {-1.0, 1e-320, 1e-250, -1e250}) {
        double scaled[9];
        for (int i = 0; i < 9; ++i) scaled[i] = translation[i] * factor;
        Geometry g;
        CHECK(prepare_geometry(4, 3, 4, 3, scaled, {}, &g) == Status::OK);
        CHECK(g.width == 6 && g.height == 3 && g.origin_x == -2);
        check_inverse_point(g, -2, 2, 0, 2);
    }
    const double y_translation[9] = {1, 0, 0, 0, 1, -3, 0, 0, 1};
    CHECK(prepare_geometry(2, 3, 2, 3, y_translation, {}, &reference) == Status::OK);
    CHECK(reference.width == 2 && reference.height == 6 && reference.origin_y == -3);
    // Axis interchange requires a row pivot and has a negative determinant.
    const double interchange[9] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
    CHECK(prepare_geometry(2, 2, 4, 3, interchange, {}, &reference) == Status::OK);
    CHECK(reference.width == 3 && reference.height == 4);
    check_inverse_point(reference, 2, 3, 3, 2);
}

void test_projective_and_h33() {
    // x_A=x_B/(1+x_B/4), y_A=y_B/(1+x_B/4).
    const double projective[9] = {1, 0, 0, 0, 1, 0, 0.25, 0, 1};
    Geometry g;
    CHECK(prepare_geometry(2, 2, 4, 3, projective, {}, &g) == Status::OK);
    CHECK(g.width == 3 && g.height == 3);
    check_inverse_point(g, 12.0 / 7, 8.0 / 7, 3, 2);

    // Invertible, but h33=0 puts the included source origin on the horizon.
    const double zero_h33[9] = {1, 0, 1, 0, 1, 0, 1, 0, 0};
    expect_failure(Status::DEGENERATE, 2, 1, 2, 1, zero_h33);
    // A small positive h33 is not rejected by an absolute h33/determinant rule.
    const double small_h33[9] = {1, 0, 1, 0, 1, 0, 1, 0, 1e-5};
    CHECK(prepare_geometry(2, 1, 2, 1, small_h33, {}, &g) == Status::OK);
    CHECK(g.width == 100001 && g.height == 1);
    check_inverse_point(g, 100000, 0, 0, 0, 1e-9);
}

void test_horizon_singular_and_conditioning() {
    const double horizon[9] = {1, 0, 0, 0, 1, 0, 1, 0, -1};
    expect_failure(Status::DEGENERATE, 3, 3, 3, 3, horizon);
    const double corner_horizon[9] = {1, 0, 0, 0, 1, 0, -0.5, 0, 1};
    expect_failure(Status::DEGENERATE, 3, 3, 3, 3, corner_horizon);
    const double singular[9] = {1, 2, 0, 2, 4, 0, 0, 0, 1};
    expect_failure(Status::DEGENERATE, 2, 2, 2, 2, singular);
    const double ill_conditioned[9] = {1e-15, 0, 0, 0, 1, 0, 0, 0, 1};
    expect_failure(Status::DEGENERATE, 4, 3, 4, 3, ill_conditioned);
    double invalid[9];
    std::copy(identity, identity + 9, invalid);
    invalid[2] = std::numeric_limits<double>::infinity();
    expect_failure(Status::DEGENERATE, 2, 2, 2, 2, invalid);
    invalid[2] = std::numeric_limits<double>::quiet_NaN();
    expect_failure(Status::DEGENERATE, 2, 2, 2, 2, invalid);
    const double zero[9] = {};
    expect_failure(Status::DEGENERATE, 2, 2, 2, 2, zero);
    // A large valid in-image translation should not fail a pixel-coordinate condition test.
    const double translation[9] = {1, 0, 10000, 0, 1, 0, 0, 0, 1};
    Geometry g;
    CHECK(prepare_geometry(20001, 1, 10001, 1, translation, {}, &g) == Status::OK);
    CHECK(g.width == 20001 && g.height == 1);
    check_inverse_point(g, 10000, 0, 0, 0, 1e-8);
}

void test_snapping_budget_and_overflow() {
    Geometry g;
    double translation[9] = {1, 0, -2 - 5e-11, 0, 1, 0, 0, 0, 1};
    CHECK(prepare_geometry(4, 3, 4, 3, translation, {}, &g) == Status::OK);
    CHECK(g.width == 6 && g.origin_x == -2);
    translation[2] = -2 - 1e-6;
    CHECK(prepare_geometry(4, 3, 4, 3, translation, {}, &g) == Status::OK);
    CHECK(g.width == 7 && g.origin_x == -3);
    GeometryPolicy p;
    p.max_canvas_pixels = 17;
    translation[2] = -2;
    expect_failure(Status::INSUFFICIENT_WORKSPACE, 4, 3, 4, 3, translation, p);
    p.max_canvas_pixels = 18;
    CHECK(prepare_geometry(4, 3, 4, 3, translation, p, &g) == Status::OK);
    p.max_canvas_pixels = std::numeric_limits<std::size_t>::max();
    translation[2] = 1e30;
    expect_failure(Status::INSUFFICIENT_WORKSPACE, 4, 3, 4, 3, translation, p);
    translation[2] = -1e30;
    expect_failure(Status::INSUFFICIENT_WORKSPACE, 4, 3, 4, 3, translation, p);
    expect_failure(Status::INSUFFICIENT_WORKSPACE,
                   std::numeric_limits<int>::max(), 2, 1, 1, identity);
    // The numerical floor is scale-aware even when absolute snap_tolerance is smaller.
    p = {};
    p.max_canvas_pixels = 2000000;
    p.inverse_tolerance = 1e-15;
    translation[2] = 1000000 + 1e-10;
    CHECK(prepare_geometry(2, 1, 2, 1, translation, p, &g) == Status::OK);
    CHECK(g.width == 1000002 && g.height == 1);
    // This is only metadata: no enormous image is allocated by prepare_geometry.
    p = {};
    p.max_canvas_pixels = std::numeric_limits<int>::max();
    CHECK(prepare_geometry(std::numeric_limits<int>::max(), 1,
                           std::numeric_limits<int>::max(), 1,
                           identity, p, &g) == Status::OK);
    CHECK(g.width == std::numeric_limits<int>::max() && g.height == 1);
}

void test_invalid_arguments() {
    Geometry g;
    CHECK(prepare_geometry(2, 2, 2, 2, identity, {}, nullptr) == Status::INVALID_ARGUMENT);
    expect_failure(Status::INVALID_ARGUMENT, 0, 2, 2, 2, identity);
    expect_failure(Status::INVALID_ARGUMENT, 2, -1, 2, 2, identity);
    expect_failure(Status::INVALID_ARGUMENT, 2, 2, 0, 2, identity);
    expect_failure(Status::INVALID_ARGUMENT, 2, 2, 2, -1, identity);
    expect_failure(Status::INVALID_ARGUMENT, 2, 2, 2, 2, nullptr);
    GeometryPolicy p;
    p.denominator_tolerance = 0;
    expect_failure(Status::INVALID_ARGUMENT, 2, 2, 2, 2, identity, p);
    p = {};
    p.inverse_tolerance = std::numeric_limits<double>::quiet_NaN();
    expect_failure(Status::INVALID_ARGUMENT, 2, 2, 2, 2, identity, p);
    p = {};
    p.inverse_tolerance = 1;
    expect_failure(Status::INVALID_ARGUMENT, 2, 2, 2, 2, identity, p);
    p = {};
    p.snap_tolerance = -1;
    expect_failure(Status::INVALID_ARGUMENT, 2, 2, 2, 2, identity, p);
    p = {};
    p.max_canvas_pixels = 0;
    expect_failure(Status::INVALID_ARGUMENT, 2, 2, 2, 2, identity, p);
}

void test_policy_propagation() {
    GeometryPolicy p;
    p.denominator_tolerance = 1e-9;
    p.snap_tolerance = 1e-7;
    Geometry g;
    CHECK(prepare_geometry(2, 2, 2, 2, identity, p, &g) == Status::OK);
    CHECK(g.denominator_tolerance == p.denominator_tolerance);
    CHECK(g.snap_tolerance == p.snap_tolerance);
}

}  // namespace

int main() {
    test_identity_and_dimensions();
    test_translation_and_scale_equivalence();
    test_projective_and_h33();
    test_horizon_singular_and_conditioning();
    test_snapping_budget_and_overflow();
    test_invalid_arguments();
    test_policy_propagation();
    return 0;
}
