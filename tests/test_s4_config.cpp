#include "s4_s5/s4.hpp"
#include "check.hpp"

#include <climits>
#include <initializer_list>
#include <limits>

using namespace stitch;
using namespace stitch::s4;

int main() {
    std::size_t bytes = 123;
    CHECK(s4_workspace_bytes(0, 7, {0.8, false}, &bytes) == Status::OK && bytes == 0);
    CHECK(s4_workspace_bytes(129, 17, {0.8, false}, &bytes) == Status::OK);
    CHECK(bytes == 129 * sizeof(Top2) + 131 * sizeof(int));
    const auto forward = bytes;
    CHECK(s4_workspace_bytes(129, 17, {0.8, true}, &bytes) == Status::OK);
    CHECK(bytes == forward + 17 * sizeof(Top2));
    CHECK(s4_workspace_bytes(1, 0, {0.5, true}, &bytes) == Status::OK);
    CHECK(bytes == sizeof(Top2) + 2 * sizeof(int));
    for (double ratio : {0.0, -0.1, 1.0, 1.1,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        bytes = 123;
        CHECK(s4_workspace_bytes(4, 5, {ratio, false}, &bytes) == Status::INVALID_ARGUMENT);
        CHECK(bytes == 0);
    }
    CHECK(s4_workspace_bytes(-1, 4, {0.8, false}, &bytes) == Status::INVALID_ARGUMENT);
    CHECK(s4_workspace_bytes(4, -1, {0.8, false}, &bytes) == Status::INVALID_ARGUMENT);
    CHECK(s4_workspace_bytes(4, 5, {0.8, false}, nullptr) == Status::INVALID_ARGUMENT);
    CHECK(s4_workspace_bytes(4, 5, Config{}, &bytes) == Status::INVALID_ARGUMENT);
    CHECK(s4_workspace_bytes(1, 2, {std::numeric_limits<double>::denorm_min(), true},
                             &bytes) == Status::OK);
    const auto status = s4_workspace_bytes(INT_MAX, INT_MAX, {0.8, true}, &bytes);
    if (sizeof(std::size_t) >= 8) {
        CHECK(status == Status::OK);
        CHECK(bytes == 2 * static_cast<std::size_t>(INT_MAX) * sizeof(Top2) +
                       (static_cast<std::size_t>(INT_MAX) +
                        (static_cast<std::size_t>(INT_MAX) + 127) / 128) * sizeof(int));
    } else {
        CHECK(status == Status::INVALID_ARGUMENT && bytes == 0);
    }
    std::puts("S4 host configuration checks passed");
}
