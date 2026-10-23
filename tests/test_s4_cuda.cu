#include <cuda_runtime.h>

#include "s4/s4.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace stitch;
using namespace stitch::s4;

#define CUDA_CHECK(call) CHECK((call) == cudaSuccess)

namespace {
template <class T>
struct Device {
    T* data = nullptr;
    explicit Device(std::size_t size) {
        if (size) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

template <class T>
struct Input {
    Device<unsigned char> storage;
    Device<int> count{1};
    Keypoints keypoints{};
    Descriptors descriptors{};
    Input(int capacity, int dim, int padding, const std::vector<T>& packed, int active)
        : storage(static_cast<std::size_t>(capacity) * (dim + padding) * sizeof(T)) {
        CHECK(packed.size() == static_cast<std::size_t>(capacity) * dim);
        keypoints.capacity = capacity;
        keypoints.d_count = count.data;
        descriptors = {std::is_same_v<T, float> ? DescriptorType::FLOAT32 : DescriptorType::BINARY_U8,
                       dim, static_cast<int>((dim + padding) * sizeof(T)), storage.data};
        if (capacity) {
            CUDA_CHECK(cudaMemset(storage.data, 0xa5,
                                 static_cast<std::size_t>(capacity) * descriptors.row_stride_bytes));
            CUDA_CHECK(cudaMemcpy2D(storage.data, descriptors.row_stride_bytes,
                                    packed.data(), dim * sizeof(T), dim * sizeof(T),
                                    capacity, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(count.data, &active, sizeof(int), cudaMemcpyHostToDevice));
    }
};

struct Result {
    int a;
    int b;
    float distance;
};

// Independent reference: compute every distance, sort (distance,index), then
// take two. No production insertion/reduction/filter helper is reused.
template <class T>
std::vector<Top2> oracle(const std::vector<T>& a, const std::vector<T>& b,
                         int dim, int count_a, int count_b, int capacity_a) {
    std::vector<Top2> result(capacity_a, {INFINITY, INFINITY, -1, -1});
    if (count_a < 0 || count_a > capacity_a || count_b < 0 ||
        static_cast<std::size_t>(count_b) > b.size() / dim) return result;
    for (int q = 0; q < count_a; ++q) {
        std::vector<std::pair<double, int>> distances;
        for (int t = 0; t < count_b; ++t) {
            double distance = 0;
            for (int d = 0; d < dim; ++d) {
                if constexpr (std::is_same_v<T, float>) {
                    const double delta = static_cast<double>(a[q * dim + d]) - b[t * dim + d];
                    distance += delta * delta;
                } else {
                    unsigned int bits = a[q * dim + d] ^ b[t * dim + d];
                    for (; bits; bits >>= 1) distance += bits & 1;
                }
            }
            distances.emplace_back(distance, t);
        }
        std::sort(distances.begin(), distances.end());
        if (!distances.empty()) {
            result[q].first_distance = distances[0].first;
            result[q].first_index = distances[0].second;
        }
        if (distances.size() > 1) {
            result[q].second_distance = distances[1].first;
            result[q].second_index = distances[1].second;
        }
    }
    return result;
}

void compare_top2(const Top2& actual, const Top2& expected) {
    CHECK(actual.first_index == expected.first_index);
    CHECK(actual.second_index == expected.second_index);
    for (const auto p : {std::make_pair(actual.first_distance, expected.first_distance),
                         std::make_pair(actual.second_distance, expected.second_distance)}) {
        if (std::isinf(p.second)) CHECK(std::isinf(p.first));
        else CHECK_NEAR(p.first, p.second, 1e-12 * std::max(1.0, p.second));
    }
}

template <class T>
std::vector<Result> run(const std::vector<T>& a, const std::vector<T>& b, int dim,
                         int count_a, int count_b, Config config) {
    const int ca = static_cast<int>(a.size() / dim);
    const int cb = static_cast<int>(b.size() / dim);
    Input<T> ia(ca, dim, 3, a, count_a), ib(cb, dim, 7, b, count_b);
    Device<Top2> top(ca);
    Device<int32_t> indices_a(ca), indices_b(ca);
    Device<float> distances(ca);
    Device<int> count(1);
    Matches out{indices_a.data, indices_b.data, distances.data, ca, count.data};
    std::size_t bytes;
    CHECK(s4_workspace_bytes(ca, cb, config, &bytes) == Status::OK);
    // Guard bytes check workspace bounds, including the final partial block.
    Device<unsigned char> work(bytes + 32);
    CUDA_CHECK(cudaMemset(work.data, 0xa5, bytes + 32));
    // Setup uses the default stream; finish it before the nonblocking stream.
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto expected = oracle(a, b, dim, count_a, count_b, ca);
    const auto reverse = oracle(b, a, dim, count_b, count_a, cb);
    std::vector<Result> wanted;
    for (int q = 0; q < ca; ++q) {
        const auto p = expected[q];
        // Reference compares actual Euclidean distances, not squared thresholds.
        const double first = std::is_same_v<T, float> ? std::sqrt(p.first_distance) : p.first_distance;
        const double second = std::is_same_v<T, float> ? std::sqrt(p.second_distance) : p.second_distance;
        if (p.second_index >= 0 && first < config.ratio_threshold * second &&
            (!config.cross_check || reverse[p.first_index].first_index == q)) {
            wanted.push_back({q, p.first_index, first > std::numeric_limits<float>::max() ?
                              std::numeric_limits<float>::infinity() : static_cast<float>(first)});
        }
    }
    std::vector<Result> result;
    std::vector<int32_t> previous_a, previous_b;
    std::vector<float> previous_d;
    for (int repeat = 0; repeat < 3; ++repeat) {
        // Counts are produced on this stream immediately before matching.
        CUDA_CHECK(cudaMemcpyAsync(ia.count.data, &count_a, sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ib.count.data, &count_b, sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(count.data, 0x7f, sizeof(int), stream));
        CHECK(s4_knn_top2(ia.keypoints, ia.descriptors, ib.keypoints, ib.descriptors,
                          top.data, stream) == Status::OK);
        CHECK(s4_match(ia.keypoints, ia.descriptors, ib.keypoints, ib.descriptors,
                       config, out, bytes ? work.data : nullptr, bytes, stream) == Status::OK);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<Top2> actual(ca);
        if (ca) CUDA_CHECK(cudaMemcpy(actual.data(), top.data, ca * sizeof(Top2), cudaMemcpyDeviceToHost));
        for (int q = 0; q < ca; ++q) compare_top2(actual[q], expected[q]);
        int n = -1;
        CUDA_CHECK(cudaMemcpy(&n, count.data, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(n == static_cast<int>(wanted.size()));
        std::vector<int32_t> ha(n), hb(n);
        std::vector<float> hd(n);
        if (n) {
            CUDA_CHECK(cudaMemcpy(ha.data(), indices_a.data, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hb.data(), indices_b.data, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hd.data(), distances.data, n * sizeof(float), cudaMemcpyDeviceToHost));
        }
        result.clear();
        for (int i = 0; i < n; ++i) {
            CHECK(ha[i] == wanted[i].a && hb[i] == wanted[i].b);
            if (std::isinf(wanted[i].distance)) CHECK(std::isinf(hd[i]));
            else CHECK_NEAR(hd[i], wanted[i].distance, 1e-5f * std::max(1.0f, wanted[i].distance));
            result.push_back({ha[i], hb[i], hd[i]});
        }
        if (repeat) {
            CHECK(ha == previous_a && hb == previous_b && hd == previous_d);
        }
        previous_a = ha; previous_b = hb; previous_d = hd;
    }
    unsigned char guard[32];
    CUDA_CHECK(cudaMemcpy(guard, work.data + bytes, sizeof(guard), cudaMemcpyDeviceToHost));
    for (auto value : guard) CHECK(value == 0xa5);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
}

void test_edges() {
    for (bool cross : {false, true}) {
        CHECK(run<unsigned char>({}, {0, 255}, 1, 0, 2, {0.8, cross}).empty());
        CHECK(run<unsigned char>({0, 1}, {}, 1, 2, 0, {0.8, cross}).empty());
        CHECK(run<float>({0, 1}, {0}, 1, 2, 1, {0.8, cross}).empty());
        CHECK(run<float>({0}, {0, 0}, 1, 1, 2, {0.8, cross}).empty());
        CHECK(run<unsigned char>({0}, {0, 0}, 1, 1, 2, {0.8, cross}).empty());
        CHECK(run<unsigned char>({0}, {0}, 1, 1, 1, {0.8, cross}).empty());
        CHECK(run<unsigned char>({0}, {1, 2}, 1, 1, 2, {0.8, cross}).empty());
        CHECK(run<float>({0}, {1, 2}, 1, 1, 2, {0.5, cross}).empty());
        CHECK(run<unsigned char>({0}, {1, 3}, 1, 1, 2, {0.5, cross}).empty());
        CHECK(run<float>({0}, {0, 2}, 1, 1, 2,
                          {std::numeric_limits<double>::denorm_min(), cross}).size() == 1);
        CHECK(run<float>({0, 99}, {1, -1, 9}, 1, 1, 3, {0.9, cross}).empty());
        CHECK(run<unsigned char>({0, 1}, {0, 255}, 1, 0, 2, {0.8, cross}).empty());
        CHECK(run<unsigned char>({0, 1}, {0, 255}, 1, 2, 0, {0.8, cross}).empty());
        CHECK(run<unsigned char>({0, 1}, {0, 255}, 1, 3, 2, {0.8, cross}).empty());
        CHECK(run<unsigned char>({0, 1}, {0, 255}, 1, 2, -1, {0.8, cross}).empty());
    }
    // Equal A descriptors compete for B0: reverse tie must select A0.
    CHECK(run<float>({0, 0}, {0, 10}, 1, 2, 2, {0.8, false}).size() == 2);
    const auto mutual = run<float>({0, 0}, {0, 10}, 1, 2, 2, {0.8, true});
    CHECK(mutual.size() == 1 && mutual[0].a == 0 && mutual[0].b == 0);
    const auto binary_mutual = run<unsigned char>({0, 0}, {0, 255}, 1, 2, 2, {0.8, true});
    CHECK(binary_mutual.size() == 1 && binary_mutual[0].a == 0);
    const auto euclidean = run<float>({0, 0}, {3, 4, 30, 40}, 2, 1, 2, {0.8, false});
    CHECK(euclidean.size() == 1 && euclidean[0].distance == 5);
    const float large = std::numeric_limits<float>::max();
    const auto overflow = run<float>({-large}, {large / 2, large}, 1, 1, 2, {0.9, false});
    CHECK(overflow.size() == 1 && std::isinf(overflow[0].distance));
}

void test_tiles() {
    for (int dim : {1, 31, 32, 33, 65}) {
        std::vector<unsigned char> a(263 * dim), b(259 * dim);
        std::vector<float> fa(a.size()), fb(b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<unsigned char>((i * 73 + i / dim * 19) % 256);
            fa[i] = static_cast<float>(static_cast<int>((i * 13 + i / dim * 11) % 101) - 50) / 8 +
                    static_cast<float>(i / dim) * 0.25f;
        }
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = a[(i + 3 * dim) % a.size()];
            fb[i] = fa[(i + 5 * dim) % fa.size()] + 0.125f;
        }
        for (bool cross : {false, true}) {
            run(a, b, dim, 257, 253, {0.8, cross});
            run(fa, fb, dim, 257, 253, {0.8, cross});
        }
    }
    // First and second ties across training tiles, not just within a warp.
    std::vector<float> train(260, 20);
    train[3] = train[129] = train[258] = 1;
    run<float>({0}, train, 1, 1, 260, {0.8, true});
    // Accepted L2 matches span three compaction blocks, with inactive tail rows.
    std::vector<float> queries(263), rows(259);
    for (int q = 0; q < 263; ++q) queries[q] = q * 4.0f;
    for (int t = 0; t < 259; ++t) rows[t] = t * 4.0f + 0.25f;
    CHECK(run(queries, rows, 1, 257, 253, {0.8, true}).size() == 253);
    CHECK(run(queries, rows, 1, 257, 253, {0.8, false}).size() == 257);
}

void test_validation() {
    Input<float> a(2, 2, 1, {0, 0, 1, 1}, 2), b(2, 2, 2, {0, 0, 9, 9}, 2);
    Device<int32_t> ia(2), ib(2);
    Device<float> dist(2);
    Device<int> count(1);
    Matches out{ia.data, ib.data, dist.data, 2, count.data};
    Config config{0.8, true};
    std::size_t bytes;
    CHECK(s4_workspace_bytes(2, 2, config, &bytes) == Status::OK);
    Device<unsigned char> work(bytes + 8);
    auto call = [&](Keypoints ka, Descriptors da, Keypoints kb, Descriptors db,
                    Config c, Matches output, void* w, std::size_t size) {
        return s4_match(ka, da, kb, db, c, output, w, size, nullptr);
    };
    CHECK(call(a.keypoints, a.descriptors, b.keypoints, b.descriptors, config, out,
               work.data, bytes - 1) == Status::INSUFFICIENT_WORKSPACE);
    CHECK(call(a.keypoints, a.descriptors, b.keypoints, b.descriptors, config, out,
               nullptr, bytes) == Status::INSUFFICIENT_WORKSPACE);
    CHECK(call(a.keypoints, a.descriptors, b.keypoints, b.descriptors, config, out,
               work.data + 1, bytes) == Status::INVALID_ARGUMENT);
    auto invalid = [&](Keypoints ka, Descriptors da, Keypoints kb, Descriptors db,
                       Config c, Matches output) {
        CHECK(call(ka, da, kb, db, c, output, work.data, bytes) == Status::INVALID_ARGUMENT);
    };
    auto da = a.descriptors;
    da.row_stride_bytes = 4;
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    da = a.descriptors; da.row_stride_bytes = 9;
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    da = a.descriptors; da.type = DescriptorType::BINARY_U8;
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    da = a.descriptors; da.dim = std::numeric_limits<int>::max();
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    da = a.descriptors; da.data = nullptr;
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    da = a.descriptors; da.data = a.storage.data + 1;
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    da = a.descriptors;
    da.data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 3);
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    da = a.descriptors; da.type = static_cast<DescriptorType>(9);
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    da = a.descriptors; da.dim = 0;
    invalid(a.keypoints, da, b.keypoints, b.descriptors, config, out);
    CHECK(call(a.keypoints, a.descriptors, b.keypoints, b.descriptors, config, out,
               reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 7),
               bytes) == Status::INVALID_ARGUMENT);
    auto ka = a.keypoints; ka.d_count = nullptr;
    invalid(ka, a.descriptors, b.keypoints, b.descriptors, config, out);
    ka = a.keypoints; ka.capacity = -1;
    invalid(ka, a.descriptors, b.keypoints, b.descriptors, config, out);
    auto output = out; output.capacity = 1;
    invalid(a.keypoints, a.descriptors, b.keypoints, b.descriptors, config, output);
    output = out; output.d_count = nullptr;
    invalid(a.keypoints, a.descriptors, b.keypoints, b.descriptors, config, output);
    output = out; output.idx_b = nullptr;
    invalid(a.keypoints, a.descriptors, b.keypoints, b.descriptors, config, output);
    invalid(a.keypoints, a.descriptors, b.keypoints, b.descriptors, {1, false}, out);
    CHECK(s4_knn_top2(a.keypoints, a.descriptors, b.keypoints, b.descriptors,
                      nullptr, nullptr) == Status::INVALID_ARGUMENT);
    CHECK(s4_knn_top2(a.keypoints, a.descriptors, b.keypoints, b.descriptors,
                      reinterpret_cast<Top2*>(work.data + 1), nullptr) == Status::INVALID_ARGUMENT);
    // Argument failures must not enqueue writes to caller output.
    CUDA_CHECK(cudaMemset(count.data, 0x5a, sizeof(int)));
    invalid(a.keypoints, a.descriptors, b.keypoints, b.descriptors, {0, false}, out);
    int sentinel;
    CUDA_CHECK(cudaMemcpy(&sentinel, count.data, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(sentinel == 0x5a5a5a5a);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("S4 CUDA tests require a CUDA device");
        return 77;
    }
    test_edges();
    test_tiles();
    test_validation();
    std::puts("S4 CUDA oracle and contract checks passed");
}
