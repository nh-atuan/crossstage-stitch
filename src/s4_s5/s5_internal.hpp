#pragma once

#include "s4_s5/s5.hpp"

namespace stitch::s5::detail {

struct Model {
    double h[9];
    double loss;
    int hypothesis;
    bool valid;
};

struct State {
    Model best;
    HomographyResult result;
    const float* ax;
    const float* ay;
    const float* bx;
    const float* by;
    const int32_t* idx_a;
    const int32_t* idx_b;
    const int* count_a;
    const int* count_b;
    const int* count_matches;
    int na, nb, nm;
    int capacity_a, capacity_b, capacity_matches;
    int evaluated;
    int batch_size;
    double threshold;
    std::uint64_t seed, generation;
    ScaleInfo scale_a, scale_b;
    Status status;
};

constexpr std::size_t model_offset = (sizeof(State) + 255) / 256 * 256;

}  // namespace stitch::s5::detail
