#include "check.hpp"
#include "common/geometry.hpp"
#include "s5/s5.hpp"
#include "s5/s5_math.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <vector>

using namespace stitch;
using namespace stitch::s5;
using namespace stitch::s5::detail;

namespace {
struct Points {
    const Point* p;
    Point operator()(int i) const { return p[i]; }
};

void test_known_projective_model() {
    // Given: B -> A correspondences under a non-affine H.
    const double truth[9]{1.02, 0.03, 11, -0.02, 0.98, 7, 0.0002, -0.0001, 1};
    std::vector<Point> points;
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 8; ++x) {
            const double bx = 11 + x * 80, by = 13 + y * 70;
            const double z = truth[6] * bx + truth[7] * by + truth[8];
            points.push_back({(truth[0] * bx + truth[1] * by + truth[2]) / z,
                              (truth[3] * bx + truth[4] * by + truth[5]) / z, bx, by});
        }
    // When: fitting all observations using the same solver as the CUDA path.
    double h[9]{};
    CHECK(fit(static_cast<int>(points.size()), Points{points.data()}, All{}, h));
    // Then: the recovered model maps B to A, not A to B.
    CHECK(homography_max_diff(h, truth) < 1e-8);
    for (Point p : points) CHECK(residual_squared(h, p) < 1e-16);
    CHECK(valid_domain(h, 640, 480));
}

void test_minimal_and_degenerate_samples() {
    // Given: an affine rectangle and a collinear four-point sample.
    const Point rectangle[4]{{3, 7, 0, 0}, {103, 7, 100, 0}, {103, 107, 100, 100}, {3, 107, 0, 100}};
    const Point line[4]{{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
    double h[9]{};
    // When/Then: only the nondegenerate sample yields a model.
    CHECK(nondegenerate_four(rectangle));
    CHECK(fit(4, Points{rectangle}, All{}, h));
    for (Point p : rectangle) CHECK(residual_squared(h, p) < 1e-20);
    CHECK(!nondegenerate_four(line));
    CHECK(!fit(4, Points{line}, All{}, h));
    auto duplicate = std::array<Point, 4>{rectangle[0], rectangle[1], rectangle[2], rectangle[0]};
    CHECK(!nondegenerate_four(duplicate.data()));
    CHECK(!fit(3, Points{rectangle}, All{}, h));
}

void test_coordinate_scaling() {
    // Given: different resize transforms for two original image dimensions.
    ScaleInfo a{}, b{};
    CHECK(make_scale_info(641, 479, 0.5, &a) == Status::OK);
    CHECK(make_scale_info(803, 601, 0.75, &b) == Status::OK);
    const double hs[9]{1, 0.01, 5, -0.02, 1, 9, 0.0001, 0.0002, 1};
    double h[9]{}, recovered[9]{}, db_inverse[9]{}, tmp[9]{};
    // When: S5 restores original coordinates.
    CHECK(to_original(hs, a.D, b.D, h) == Status::OK);
    CHECK(mat3_inverse(b.D, db_inverse));
    mat3_mul(a.D, h, tmp); mat3_mul(tmp, db_inverse, recovered);
    // Then: the half-pixel transform round-trips.
    CHECK(homography_max_diff(hs, recovered) < 1e-12);
}

void test_counter_samples() {
    // Given: all n >= 4, including the minimal n=4 and an irregular count.
    constexpr std::uint64_t seed = 20261022;
    std::printf("S5 host seed=%llu\n", static_cast<unsigned long long>(seed));
    for (int n : {4, 5, 17, 101})
        for (int hypothesis = 0; hypothesis < 4096; ++hypothesis) {
            int a[4], b[4];
            // When: generating a sample without any previous random state.
            sample_four(n, seed, hypothesis, a);
            sample_four(n, seed, hypothesis, b);
            // Then: every index is distinct, in range and repeatable.
            for (int i = 0; i < 4; ++i) {
                CHECK(a[i] >= 0 && a[i] < n && a[i] == b[i]);
                for (int j = 0; j < i; ++j) CHECK(a[i] != a[j]);
            }
        }
}

void test_domain_and_config_errors() {
    // Given: a horizon crossing B and an explicit pilot-test configuration.
    double crossing[9]{1, 0, 0, 0, 1, 0, -0.1, 0, 1};
    CHECK(!valid_domain(crossing, 100, 100));
    Config c{129, 0, 32, 2, 3.0, 20261022, 1};
    std::size_t bytes = 0, extended = 0;
    // When/Then: workspace depends on the batch, not accumulated R.
    CHECK(s5_workspace_bytes(c, &bytes) == Status::OK);
    CHECK(bytes > 0);
    c.hypotheses = 1000; c.previous_hypotheses = 129;
    CHECK(s5_workspace_bytes(c, &extended) == Status::OK);
    CHECK(bytes == extended);
    c.previous_hypotheses = 1001;
    CHECK(s5_workspace_bytes(c, &bytes) == Status::INVALID_ARGUMENT);
    c.previous_hypotheses = 0; c.threshold = std::numeric_limits<double>::quiet_NaN();
    CHECK(s5_workspace_bytes(c, &bytes) == Status::INVALID_ARGUMENT);
    c.threshold = 0;
    CHECK(s5_workspace_bytes(c, &bytes) == Status::INVALID_ARGUMENT);
    c.threshold = 3; c.batch_size = 65536;
    CHECK(s5_workspace_bytes(c, &bytes) == Status::INVALID_ARGUMENT);
    c.batch_size = 32; c.refinement_steps = -1;
    CHECK(s5_workspace_bytes(c, &bytes) == Status::INVALID_ARGUMENT);
    CHECK(s5_workspace_bytes(c, nullptr) == Status::INVALID_ARGUMENT);
}
}  // namespace

int main() {
    test_known_projective_model();
    test_minimal_and_degenerate_samples();
    test_coordinate_scaling();
    test_counter_samples();
    test_domain_and_config_errors();
    std::puts("S5 host numerical and configuration tests passed");
}
