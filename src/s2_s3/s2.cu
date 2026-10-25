// S2 CUDA: FAST-9 scores -> strict 3x3 NMS inside the edge margin -> keep the
// 2K highest FAST scores -> Harris response -> keep the K highest responses.
// Every step mirrors tools/s2_s3/reference.py (and OpenCV ORB) exactly; ties
// prefer the smaller row-major pixel index. See docs/decisions/s2-s3-orb.md.
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include "s2_s3/s2_s3_internal.hpp"

namespace stitch::s2 {
namespace {

using s2_s3::detail::kAlignment;
using s2_s3::detail::kScanItems;
using s2_s3::detail::S2Layout;
using s2_s3::detail::SelectParams;

constexpr int kFastRadius = 3;
constexpr int kHarrisBlock = 7;
constexpr int kTile = 16;              // 2D pixel kernels use 16x16 blocks
constexpr int kScanThreads = 256;      // kScanItems / kItemsPerThread
constexpr int kItemsPerThread = kScanItems / kScanThreads;
constexpr int kRankThreads = 256;
// (1.f / (4 * 7 * 255.f))^4 evaluated in float as in OpenCV HarrisResponses.
constexpr float kHarrisScale4 = 0x1.bb9da2p-52f;

static_assert(kScanItems % kScanThreads == 0, "scan block layout");

// FAST circle (dx, dy), same order as offsets16 in OpenCV fast_score.cpp.
__constant__ int kCircleX[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
__constant__ int kCircleY[16] = {3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3};

// cornerScore<16>: max over 9-pixel arcs of the minimum contrast, minus one;
// zero when no arc exceeds the threshold.
__device__ int fast_score(const std::uint8_t* image, int pitch, int x, int y, int threshold) {
    const int v = image[y * pitch + x];
    int d[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) d[k] = v - image[(y + kCircleY[k]) * pitch + x + kCircleX[k]];
    int best = INT_MIN;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        int lo = INT_MAX, hi = INT_MIN;
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            const int e = d[(k + j) & 15];
            lo = min(lo, e);
            hi = max(hi, e);
        }
        best = max(best, max(lo, -hi));
    }
    return best > threshold ? best - 1 : 0;
}

__global__ void fast_kernel(ImageView image, int threshold, std::uint8_t* scores) {
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x >= image.width || y >= image.height) return;
    int score = 0;
    if (x >= kFastRadius && x < image.width - kFastRadius &&
        y >= kFastRadius && y < image.height - kFastRadius) {
        score = fast_score(image.data, image.pitch_bytes, x, y, threshold);
    }
    scores[y * image.width + x] = static_cast<std::uint8_t>(score);
}

// Keeps strict 3x3 maxima inside the edge margin and histograms their scores.
__global__ void nms_kernel(const std::uint8_t* scores, int width, int height,
                           std::uint8_t* maxima, std::uint32_t* histogram) {
    __shared__ std::uint32_t local[256];
    const int t = threadIdx.y * kTile + threadIdx.x;
    local[t] = 0;
    __syncthreads();
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x < width && y < height) {
        const int s = scores[y * width + x];
        bool keep = s > 0 && x >= kEdge && x < width - kEdge && y >= kEdge && y < height - kEdge;
        for (int dy = -1; keep && dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if ((dx || dy) && s <= scores[(y + dy) * width + x + dx]) {
                    keep = false;
                    break;
                }
            }
        }
        maxima[y * width + x] = keep ? static_cast<std::uint8_t>(s) : 0;
        if (keep) atomicAdd(&local[s], 1u);
    }
    __syncthreads();
    if (local[t]) atomicAdd(&histogram[t], local[t]);
}

// Chooses the FAST score threshold of the 2K pre-selection and writes *count.
__global__ void select_kernel(const std::uint32_t* histogram, int candidates,
                              int max_keypoints, SelectParams* params, int* count) {
    const auto cap = static_cast<std::uint32_t>(candidates);
    std::uint32_t above = 0, total = 0;
    for (int s = 1; s < 256; ++s) total += histogram[s];
    SelectParams p{1, 0, 0, 0};
    if (total <= cap) {
        p.threshold = 1;
        p.ties = histogram[1];
        p.above = total - p.ties;
    } else {
        for (int s = 255; s >= 1; --s) {
            if (above + histogram[s] >= cap) {
                p.threshold = static_cast<std::uint32_t>(s);
                p.above = above;
                p.ties = cap - above;
                break;
            }
            above += histogram[s];
        }
    }
    p.selected = p.above + p.ties;
    *params = p;
    *count = static_cast<int>(min(p.selected, static_cast<std::uint32_t>(max_keypoints)));
}

struct Flags {
    std::uint32_t above, tie;
};

__device__ Flags classify(std::uint8_t score, std::uint32_t threshold) {
    return {score > threshold ? 1u : 0u, (score && score == threshold) ? 1u : 0u};
}

// Exclusive block-wide scan of per-thread (above, tie) counts; returns totals.
__device__ Flags block_exclusive_scan(Flags value, Flags* shared) {
    const int t = threadIdx.x;
    shared[t] = value;
    __syncthreads();
    for (int offset = 1; offset < kScanThreads; offset <<= 1) {
        const Flags add = t >= offset ? shared[t - offset] : Flags{0, 0};
        __syncthreads();
        shared[t].above += add.above;
        shared[t].tie += add.tie;
        __syncthreads();
    }
    const Flags inclusive = shared[t];
    const Flags total = shared[kScanThreads - 1];
    __syncthreads();
    shared[t] = {inclusive.above - value.above, inclusive.tie - value.tie};
    __syncthreads();
    return total;
}

__device__ Flags thread_counts(const std::uint8_t* maxima, int pixels, std::uint32_t threshold,
                               int first) {
    Flags c{0, 0};
    for (int i = 0; i < kItemsPerThread; ++i) {
        const int p = first + i;
        if (p >= pixels) break;
        const Flags f = classify(maxima[p], threshold);
        c.above += f.above;
        c.tie += f.tie;
    }
    return c;
}

__global__ void count_kernel(const std::uint8_t* maxima, int pixels, const SelectParams* params,
                             std::uint32_t* block_counts) {
    __shared__ Flags shared[kScanThreads];
    const int first = blockIdx.x * kScanItems + threadIdx.x * kItemsPerThread;
    const Flags total = block_exclusive_scan(
        thread_counts(maxima, pixels, params->threshold, first), shared);
    if (threadIdx.x == 0) {
        block_counts[2 * blockIdx.x] = total.above;
        block_counts[2 * blockIdx.x + 1] = total.tie;
    }
}

// Single block: exclusive prefix over all block totals, in block order.
__global__ void block_scan_kernel(const std::uint32_t* block_counts, int blocks,
                                  std::uint32_t* block_offsets) {
    __shared__ Flags shared[kScanThreads];
    const int per_thread = (blocks + kScanThreads - 1) / kScanThreads;
    const int first = threadIdx.x * per_thread;
    Flags sum{0, 0};
    for (int i = 0; i < per_thread && first + i < blocks; ++i) {
        sum.above += block_counts[2 * (first + i)];
        sum.tie += block_counts[2 * (first + i) + 1];
    }
    block_exclusive_scan(sum, shared);
    Flags running = shared[threadIdx.x];
    for (int i = 0; i < per_thread && first + i < blocks; ++i) {
        block_offsets[2 * (first + i)] = running.above;
        block_offsets[2 * (first + i) + 1] = running.tie;
        running.above += block_counts[2 * (first + i)];
        running.tie += block_counts[2 * (first + i) + 1];
    }
}

// Writes the pixel index of every pre-selected point into its candidate slot:
// all points above the threshold, then the first `ties` points at it.
__global__ void scatter_kernel(const std::uint8_t* maxima, int pixels, const SelectParams* params,
                               const std::uint32_t* block_offsets, std::uint64_t* slots) {
    __shared__ Flags shared[kScanThreads];
    const SelectParams p = *params;
    const int first = blockIdx.x * kScanItems + threadIdx.x * kItemsPerThread;
    block_exclusive_scan(thread_counts(maxima, pixels, p.threshold, first), shared);
    Flags rank = shared[threadIdx.x];
    rank.above += block_offsets[2 * blockIdx.x];
    rank.tie += block_offsets[2 * blockIdx.x + 1];
    for (int i = 0; i < kItemsPerThread; ++i) {
        const int px = first + i;
        if (px >= pixels) break;
        const Flags f = classify(maxima[px], p.threshold);
        if (f.above) slots[rank.above] = static_cast<std::uint64_t>(px);
        if (f.tie && rank.tie < p.ties) slots[p.above + rank.tie] = static_cast<std::uint64_t>(px);
        rank.above += f.above;
        rank.tie += f.tie;
    }
}

// Monotonic map from float order to unsigned order (never zero for finite input).
__device__ std::uint32_t order_bits(float value) {
    const std::uint32_t u = __float_as_uint(value);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ float from_order_bits(std::uint32_t u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}

// OpenCV HarrisResponses: integer gradient sums, float32 formula without FMA.
__device__ float harris_response(const std::uint8_t* image, int pitch, int x0, int y0, float k) {
    constexpr int r = kHarrisBlock / 2;
    int a = 0, b = 0, c = 0;
    for (int i = 0; i < kHarrisBlock; ++i) {
        const std::uint8_t* row = image + (y0 - r + i) * pitch + (x0 - r);
        for (int j = 0; j < kHarrisBlock; ++j) {
            const std::uint8_t* q = row + j;
            const int ix = (q[1] - q[-1]) * 2 + (q[-pitch + 1] - q[-pitch - 1]) +
                           (q[pitch + 1] - q[pitch - 1]);
            const int iy = (q[pitch] - q[-pitch]) * 2 + (q[pitch - 1] - q[-pitch - 1]) +
                           (q[pitch + 1] - q[-pitch + 1]);
            a += ix * ix;
            b += iy * iy;
            c += ix * iy;
        }
    }
    const float fa = static_cast<float>(a), fb = static_cast<float>(b), fc = static_cast<float>(c);
    const float trace = __fadd_rn(fa, fb);
    const float det = __fsub_rn(__fmul_rn(fa, fb), __fmul_rn(fc, fc));
    const float penalty = __fmul_rn(__fmul_rn(k, trace), trace);
    return __fmul_rn(__fsub_rn(det, penalty), kHarrisScale4);
}

// Replaces each candidate pixel index by its sort key (response, -index).
__global__ void harris_kernel(ImageView image, float harris_k, int candidates,
                              const SelectParams* params, std::uint64_t* keys) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= candidates) return;
    if (static_cast<std::uint32_t>(slot) >= params->selected) {
        keys[slot] = 0;
        return;
    }
    const auto px = static_cast<std::uint32_t>(keys[slot]);
    const int x = static_cast<int>(px % static_cast<std::uint32_t>(image.width));
    const int y = static_cast<int>(px / static_cast<std::uint32_t>(image.width));
    const float response = harris_response(image.data, image.pitch_bytes, x, y, harris_k);
    keys[slot] = (static_cast<std::uint64_t>(order_bits(response)) << 32) | (0xffffffffu - px);
}

// Rank selection over the (unique) candidate keys: rank = #keys greater.
__global__ void rank_kernel(const std::uint64_t* keys, int candidates, int max_keypoints,
                            int width, Keypoints output) {
    __shared__ std::uint64_t tile[kRankThreads];
    const int slot = blockIdx.x * kRankThreads + threadIdx.x;
    const std::uint64_t key = slot < candidates ? keys[slot] : 0;
    int rank = 0;
    for (int base = 0; base < candidates; base += kRankThreads) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = j < candidates ? keys[j] : 0;
        __syncthreads();
        const int n = min(kRankThreads, candidates - base);
        for (int i = 0; i < n; ++i) rank += tile[i] > key;
        __syncthreads();
    }
    if (!key || rank >= max_keypoints) return;
    const std::uint32_t px = 0xffffffffu - static_cast<std::uint32_t>(key);
    output.x[rank] = static_cast<float>(px % static_cast<std::uint32_t>(width));
    output.y[rank] = static_cast<float>(px / static_cast<std::uint32_t>(width));
    if (output.response) output.response[rank] = from_order_bits(static_cast<std::uint32_t>(key >> 32));
}

int blocks_for(int n, int threads) { return (n + threads - 1) / threads; }

template <class T> T* at(void* base, std::size_t offset) {
    return reinterpret_cast<T*>(static_cast<char*>(base) + offset);
}

Status launch_status() {
    return cudaGetLastError() == cudaSuccess ? Status::OK : Status::CUDA_ERROR;
}

}  // namespace

Status s2_detect(const ImageView& image, const Config& config, Keypoints output,
                 void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
    if (!s2_s3::detail::valid_image(image) || !s2_s3::detail::valid_s2_config(config) ||
        !output.d_count || reinterpret_cast<std::uintptr_t>(output.d_count) % alignof(int) ||
        output.capacity < config.max_keypoints ||
        (config.max_keypoints > 0 && (!output.x || !output.y))) {
        return Status::INVALID_ARGUMENT;
    }
    if (config.max_keypoints == 0 || image.width <= 2 * kEdge || image.height <= 2 * kEdge) {
        if (cudaMemsetAsync(output.d_count, 0, sizeof(int), stream) != cudaSuccess) {
            return Status::CUDA_ERROR;
        }
        return launch_status();
    }
    S2Layout layout;
    if (!s2_s3::detail::make_s2_layout(image.width, image.height, config.max_keypoints, &layout)) {
        return Status::INVALID_ARGUMENT;
    }
    if (!workspace || reinterpret_cast<std::uintptr_t>(workspace) % kAlignment) {
        return Status::INVALID_ARGUMENT;
    }
    if (workspace_bytes < layout.total) return Status::INSUFFICIENT_WORKSPACE;

    auto* scores = at<std::uint8_t>(workspace, layout.scores);
    auto* maxima = at<std::uint8_t>(workspace, layout.maxima);
    auto* histogram = at<std::uint32_t>(workspace, layout.histogram);
    auto* params = at<SelectParams>(workspace, layout.params);
    auto* block_counts = at<std::uint32_t>(workspace, layout.block_counts);
    auto* block_offsets = at<std::uint32_t>(workspace, layout.block_offsets);
    auto* keys = at<std::uint64_t>(workspace, layout.keys);

    const dim3 tile(kTile, kTile);
    const dim3 grid(blocks_for(image.width, kTile), blocks_for(image.height, kTile));
    if (cudaMemsetAsync(histogram, 0, 256 * sizeof(std::uint32_t), stream) != cudaSuccess) {
        return Status::CUDA_ERROR;
    }
    fast_kernel<<<grid, tile, 0, stream>>>(image, config.fast_threshold, scores);
    nms_kernel<<<grid, tile, 0, stream>>>(scores, image.width, image.height, maxima, histogram);
    select_kernel<<<1, 1, 0, stream>>>(histogram, layout.candidates, config.max_keypoints,
                                       params, output.d_count);
    count_kernel<<<layout.blocks, kScanThreads, 0, stream>>>(maxima, layout.pixels, params,
                                                             block_counts);
    block_scan_kernel<<<1, kScanThreads, 0, stream>>>(block_counts, layout.blocks, block_offsets);
    scatter_kernel<<<layout.blocks, kScanThreads, 0, stream>>>(maxima, layout.pixels, params,
                                                               block_offsets, keys);
    harris_kernel<<<blocks_for(layout.candidates, 128), 128, 0, stream>>>(
        image, config.harris_k, layout.candidates, params, keys);
    rank_kernel<<<blocks_for(layout.candidates, kRankThreads), kRankThreads, 0, stream>>>(
        keys, layout.candidates, config.max_keypoints, image.width, output);
    return launch_status();
}

}  // namespace stitch::s2
