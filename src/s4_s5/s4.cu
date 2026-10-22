#include <cuda_runtime.h>
#include <math_constants.h>

#include "s4_s5/s4.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>

namespace stitch::s4 {
namespace {

constexpr int kThreads = 128;
constexpr int kDimensions = 32;

bool valid_input(const Keypoints& k, const Descriptors& d) {
    if (k.capacity < 0 || !k.d_count ||
        reinterpret_cast<std::uintptr_t>(k.d_count) % alignof(int) != 0 ||
        d.dim <= 0 || d.row_stride_bytes <= 0) return false;
    std::size_t element;
    if (d.type == DescriptorType::BINARY_U8) element = 1;
    else if (d.type == DescriptorType::FLOAT32) element = sizeof(float);
    else return false;
    if (static_cast<std::size_t>(d.dim) >
        std::numeric_limits<std::size_t>::max() / element) return false;
    const auto row = static_cast<std::size_t>(d.dim) * element;
    if (row > static_cast<std::size_t>(d.row_stride_bytes) ||
        static_cast<std::size_t>(d.row_stride_bytes) % element != 0 ||
        (k.capacity && !d.data) ||
        reinterpret_cast<std::uintptr_t>(d.data) % element != 0) return false;
    if (k.capacity) {
        const auto last = static_cast<std::size_t>(k.capacity - 1);
        const auto limit = std::numeric_limits<std::size_t>::max();
        if (last > (limit - row) / static_cast<std::size_t>(d.row_stride_bytes)) return false;
        const auto bytes = last * static_cast<std::size_t>(d.row_stride_bytes) + row;
        if (reinterpret_cast<std::uintptr_t>(d.data) >
            std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    }
    return true;
}

bool valid_pair(const Keypoints& a, const Descriptors& da,
                const Keypoints& b, const Descriptors& db) {
    return valid_input(a, da) && valid_input(b, db) &&
           da.type == db.type && da.dim == db.dim;
}

int grid_size(int capacity) {
    return std::min(capacity, 65535);
}

__device__ Top2 empty_top2() {
    return {CUDART_INF, CUDART_INF, -1, -1};
}

__device__ bool before(double distance, int index, double other, int other_index) {
    return index >= 0 && (other_index < 0 || distance < other ||
                         (distance == other && index < other_index));
}

__device__ void insert(Top2& pair, double distance, int index) {
    if (index < 0 || index == pair.first_index || index == pair.second_index) return;
    if (before(distance, index, pair.first_distance, pair.first_index)) {
        pair.second_distance = pair.first_distance;
        pair.second_index = pair.first_index;
        pair.first_distance = distance;
        pair.first_index = index;
    } else if (before(distance, index, pair.second_distance, pair.second_index)) {
        pair.second_distance = distance;
        pair.second_index = index;
    }
}

// One block per query, 128 training rows per tile and 32 descriptor elements
// per dimension tile. Query elements are broadcast from shared memory. Each
// thread retains only its top two; the final tree merges these exact top twos.
template <class Element>
__global__ void top2_kernel(Keypoints a, Descriptors da, Keypoints b,
                            Descriptors db, Top2* output) {
    __shared__ Element query[kDimensions];
    __shared__ Top2 best[kThreads];
    const int tid = threadIdx.x;
    const int count_a = *a.d_count;
    const int count_b = *b.d_count;
    const bool counts_valid = count_a >= 0 && count_a <= a.capacity &&
                              count_b >= 0 && count_b <= b.capacity;
    for (std::size_t q = blockIdx.x; q < static_cast<std::size_t>(a.capacity);
         q += gridDim.x) {
        Top2 local = empty_top2();
        if (counts_valid && q < static_cast<std::size_t>(count_a)) {
            const auto* query_row = reinterpret_cast<const Element*>(
                static_cast<const unsigned char*>(da.data) + q * da.row_stride_bytes);
            for (std::size_t tile = 0; tile < static_cast<std::size_t>(count_b); tile += kThreads) {
                const std::size_t train = tile + tid;
                const Element* train_row = nullptr;
                if (train < static_cast<std::size_t>(count_b)) {
                    train_row = reinterpret_cast<const Element*>(
                        static_cast<const unsigned char*>(db.data) + train * db.row_stride_bytes);
                }
                double distance = 0;
                for (std::size_t dim = 0; dim < static_cast<std::size_t>(da.dim); dim += kDimensions) {
                    if (tid < kDimensions && dim + tid < static_cast<std::size_t>(da.dim))
                        query[tid] = query_row[dim + tid];
                    __syncthreads();
                    const int width = static_cast<int>(
                        static_cast<std::size_t>(da.dim) - dim < kDimensions ?
                        static_cast<std::size_t>(da.dim) - dim : kDimensions);
                    if (train_row) {
                        for (int d = 0; d < width; ++d) {
                            if constexpr (sizeof(Element) == 1) {
                                distance += __popc(static_cast<unsigned int>(query[d] ^ train_row[dim + d]));
                            } else {
                                const double delta = static_cast<double>(query[d]) - train_row[dim + d];
                                // Separate operations avoid FMA-dependent ranking near ties.
                                distance = __dadd_rn(distance, __dmul_rn(delta, delta));
                            }
                        }
                    }
                    __syncthreads();
                }
                if (train_row) insert(local, distance, static_cast<int>(train));
            }
        }
        best[tid] = local;
        __syncthreads();
        for (int offset = kThreads / 2; offset; offset /= 2) {
            if (tid < offset) {
                const auto other = best[tid + offset];
                insert(best[tid], other.first_distance, other.first_index);
                insert(best[tid], other.second_distance, other.second_index);
            }
            __syncthreads();
        }
        if (!tid) output[q] = best[0];
        __syncthreads();
    }
}

void launch_top2(const Keypoints& a, const Descriptors& da,
                 const Keypoints& b, const Descriptors& db, Top2* output,
                 cudaStream_t stream) {
    if (!a.capacity) return;
    if (da.type == DescriptorType::BINARY_U8)
        top2_kernel<unsigned char><<<grid_size(a.capacity), kThreads, 0, stream>>>(a, da, b, db, output);
    else
        top2_kernel<float><<<grid_size(a.capacity), kThreads, 0, stream>>>(a, da, b, db, output);
}

__global__ void rank_kernel(int capacity, const Top2* forward, const Top2* reverse,
                            double ratio, int* ranks, int* totals) {
    __shared__ int prefix[kThreads];
    const int tid = threadIdx.x;
    const auto blocks = (static_cast<std::size_t>(capacity) + kThreads - 1) / kThreads;
    for (std::size_t block = blockIdx.x; block < blocks; block += gridDim.x) {
        const auto q = block * kThreads + tid;
        bool accept = false;
        if (q < static_cast<std::size_t>(capacity)) {
            const auto p = forward[q];
            accept = p.second_index >= 0 &&
                     (p.first_distance == 0 ? p.second_distance > 0 :
                      p.first_distance < ratio * p.second_distance);
            if (accept && reverse) accept = reverse[p.first_index].first_index == static_cast<int>(q);
        }
        prefix[tid] = accept ? 1 : 0;
        __syncthreads();
        for (int offset = 1; offset < kThreads; offset *= 2) {
            const int value = tid >= offset ? prefix[tid - offset] : 0;
            __syncthreads();
            prefix[tid] += value;
            __syncthreads();
        }
        if (q < static_cast<std::size_t>(capacity)) ranks[q] = accept ? prefix[tid] - 1 : -1;
        if (!tid) totals[block] = prefix[kThreads - 1];
        __syncthreads();
    }
}

__global__ void offsets_kernel(std::size_t blocks, int* totals, int* count) {
    int sum = 0;
    for (std::size_t i = 0; i < blocks; ++i) {
        const int value = totals[i];
        totals[i] = sum;
        sum += value;
    }
    *count = sum;
}

__global__ void scatter_kernel(int capacity, const Top2* forward, const int* ranks,
                               const int* offsets, bool l2, Matches output) {
    for (std::size_t q = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         q < static_cast<std::size_t>(capacity); q += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        if (ranks[q] < 0) continue;
        const int index = offsets[q / kThreads] + ranks[q];
        output.idx_a[index] = static_cast<int>(q);
        output.idx_b[index] = forward[q].first_index;
        const double distance = l2 ? sqrt(forward[q].first_distance) : forward[q].first_distance;
        output.distance[index] = distance > FLT_MAX ? CUDART_INF_F : static_cast<float>(distance);
    }
}

Status launch_status() {
    return cudaGetLastError() == cudaSuccess ? Status::OK : Status::CUDA_ERROR;
}
}  // namespace

Status s4_knn_top2(const Keypoints& a, const Descriptors& da,
                   const Keypoints& b, const Descriptors& db, Top2* output,
                   cudaStream_t stream) {
    if (!valid_pair(a, da, b, db) ||
        (a.capacity && (!output || reinterpret_cast<std::uintptr_t>(output) % alignof(Top2))))
        return Status::INVALID_ARGUMENT;
    launch_top2(a, da, b, db, output, stream);
    return a.capacity ? launch_status() : Status::OK;
}

Status s4_match(const Keypoints& a, const Descriptors& da,
                const Keypoints& b, const Descriptors& db, const Config& config,
                Matches output, void* workspace, std::size_t workspace_bytes,
                cudaStream_t stream) {
    std::size_t required = 0;
    if (s4_workspace_bytes(a.capacity, b.capacity, config, &required) != Status::OK ||
        !valid_pair(a, da, b, db) || output.capacity < a.capacity ||
        !output.d_count ||
        reinterpret_cast<std::uintptr_t>(output.d_count) % alignof(int) ||
        (a.capacity && (!output.idx_a || !output.idx_b || !output.distance ||
         reinterpret_cast<std::uintptr_t>(output.idx_a) % alignof(int32_t) ||
         reinterpret_cast<std::uintptr_t>(output.idx_b) % alignof(int32_t) ||
         reinterpret_cast<std::uintptr_t>(output.distance) % alignof(float))))
        return Status::INVALID_ARGUMENT;
    if (workspace_bytes < required || (required && !workspace))
        return Status::INSUFFICIENT_WORKSPACE;
    if (required && (reinterpret_cast<std::uintptr_t>(workspace) % alignof(Top2) ||
        reinterpret_cast<std::uintptr_t>(workspace) >
        std::numeric_limits<std::uintptr_t>::max() - required))
        return Status::INVALID_ARGUMENT;
    if (!a.capacity) {
        return cudaMemsetAsync(output.d_count, 0, sizeof(int), stream) == cudaSuccess ?
               Status::OK : Status::CUDA_ERROR;
    }
    auto* forward = static_cast<Top2*>(workspace);
    auto* reverse = config.cross_check ? forward + a.capacity : nullptr;
    auto* ranks = reinterpret_cast<int*>(forward + static_cast<std::size_t>(a.capacity) +
                  (config.cross_check ? b.capacity : 0));
    auto* totals = ranks + a.capacity;
    launch_top2(a, da, b, db, forward, stream);
    if (launch_status() != Status::OK) return Status::CUDA_ERROR;
    if (config.cross_check && b.capacity) {
        launch_top2(b, db, a, da, reverse, stream);
        if (launch_status() != Status::OK) return Status::CUDA_ERROR;
    }
    const auto blocks = (static_cast<std::size_t>(a.capacity) + kThreads - 1) / kThreads;
    const int grid = static_cast<int>(std::min<std::size_t>(blocks, 65535));
    const double ratio = da.type == DescriptorType::FLOAT32 ?
                         config.ratio_threshold * config.ratio_threshold : config.ratio_threshold;
    rank_kernel<<<grid, kThreads, 0, stream>>>(a.capacity, forward, reverse, ratio, ranks, totals);
    if (launch_status() != Status::OK) return Status::CUDA_ERROR;
    offsets_kernel<<<1, 1, 0, stream>>>(blocks, totals, output.d_count);
    if (launch_status() != Status::OK) return Status::CUDA_ERROR;
    scatter_kernel<<<grid, kThreads, 0, stream>>>(a.capacity, forward, ranks, totals,
                                               da.type == DescriptorType::FLOAT32, output);
    return launch_status();
}

}  // namespace stitch::s4
