#include "s6/s6.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace stitch::s6 {
namespace {
constexpr int kMaxLevels = 32;
constexpr std::size_t kAlignment = 256;
constexpr unsigned kBlock = 256;
constexpr std::size_t kAbsent = std::numeric_limits<std::size_t>::max();

struct Level {
    int w = 0, h = 0;
    std::size_t a = kAbsent, b = kAbsent, va = kAbsent, vb = kAbsent;
    std::size_t ua = kAbsent, ub = kAbsent, z = kAbsent;
};
struct Layout {
    std::array<Level, kMaxLevels> levels{};
    int top = 0, blend_top = 0;
    std::size_t bytes = 0;
};

bool valid_mode(BlendMode m) {
    return m == BlendMode::Feather || m == BlendMode::Multiband3 ||
           m == BlendMode::Multiband5;
}
bool valid_geometry(const Geometry& g) {
    if (g.width <= 0 || g.height <= 0 ||
        static_cast<std::size_t>(g.width) > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(g.height) ||
        g.width_a <= 0 || g.height_a <= 0 ||
        g.width_b <= 0 || g.height_b <= 0 || g.origin_x > 0 || g.origin_y > 0)
        return false;
    const auto right = static_cast<long long>(g.origin_x) + g.width - 1;
    const auto bottom = static_cast<long long>(g.origin_y) + g.height - 1;
    if (right < g.width_a - 1LL || bottom < g.height_a - 1LL) return false;
    double magnitude = 0;
    for (double v : g.inverse) {
        if (!std::isfinite(v)) return false;
        magnitude = std::max(magnitude, std::fabs(v));
    }
    return magnitude > 0 && std::isfinite(g.denominator_tolerance) && g.denominator_tolerance > 0 && g.denominator_tolerance < 1 &&
        std::isfinite(g.snap_tolerance) && g.snap_tolerance >= 0 && g.snap_tolerance < 0.5;
}
bool add_array(std::size_t count, int channels, std::size_t& cursor,
               std::size_t& offset) {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (cursor > max - (kAlignment - 1)) return false;
    cursor = (cursor + kAlignment - 1) & ~(kAlignment - 1);
    const std::size_t element_bytes = sizeof(float) * channels;
    if (count > (max - cursor) / element_bytes) return false;
    offset = cursor;
    cursor += count * element_bytes;
    return true;
}
Status layout_for(const Geometry& g, BlendMode mode, Layout& layout) {
    if (!valid_geometry(g) || !valid_mode(mode)) return Status::INVALID_ARGUMENT;
    if (mode == BlendMode::Feather) return Status::OK;
    const int requested = mode == BlendMode::Multiband3 ? 3 : 5;
    int w = g.width, h = g.height;
    std::size_t cursor = 0;
    for (int l = 0; l < kMaxLevels; ++l) {
        Level& level = layout.levels[l];
        level.w = w; level.h = h;
        const auto max = std::numeric_limits<std::size_t>::max();
        if (static_cast<std::size_t>(w) > max / static_cast<std::size_t>(h))
            return Status::INVALID_ARGUMENT;
        const std::size_t n = static_cast<std::size_t>(w) * h;
        if (!add_array(n, 3, cursor, level.a) || !add_array(n, 3, cursor, level.b) ||
            !add_array(n, 1, cursor, level.va) || !add_array(n, 1, cursor, level.vb))
            return Status::INVALID_ARGUMENT;
        if (l <= requested &&
            (!add_array(n, 1, cursor, level.ua) || !add_array(n, 1, cursor, level.ub) ||
             !add_array(n, 3, cursor, level.z))) return Status::INVALID_ARGUMENT;
        layout.top = l;
        if (w == 1 && h == 1) break;
        if (l == kMaxLevels - 1) return Status::INVALID_ARGUMENT;
        w = w / 2 + w % 2;
        h = h / 2 + h % 2;
    }
    layout.blend_top = std::min(requested, layout.top);
    layout.bytes = cursor;
    return Status::OK;
}
float* buffer(void* arena, std::size_t offset) {
    return reinterpret_cast<float*>(static_cast<std::uint8_t*>(arena) + offset);
}
struct Range { std::uintptr_t first, last; };  // half-open, checked for overflow
bool image_range(const ImageView& v, PixelFormat format, Range& r) {
    const std::size_t channels = format == PixelFormat::RGB8 ? 3 : 1;
    if (!v.data || v.width <= 0 || v.height <= 0 || v.pitch_bytes <= 0 ||
        v.format != format) return false;
    const std::size_t row = static_cast<std::size_t>(v.width) * channels;
    if (row > static_cast<std::size_t>(v.pitch_bytes)) return false;
    const auto max = std::numeric_limits<std::uintptr_t>::max();
    const auto rows = static_cast<std::size_t>(v.height - 1);
    const auto pitch = static_cast<std::size_t>(v.pitch_bytes);
    if (rows > (max - row) / pitch) return false;
    const auto span = rows * pitch + row;
    r.first = reinterpret_cast<std::uintptr_t>(v.data);
    if (r.first > max - span) return false;
    r.last = r.first + span;
    return true;
}
bool overlaps(const Range& a, const Range& b) {
    return a.first < b.last && b.first < a.last;
}
unsigned blocks_for(std::size_t n) {
    // A bounded grid and a grid-stride loop also cover very large valid views.
    return static_cast<unsigned>(std::min<std::size_t>((n + kBlock - 1) / kBlock, 65535));
}
__device__ std::size_t thread_index() {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}
__device__ std::size_t thread_step() {
    return static_cast<std::size_t>(blockDim.x) * gridDim.x;
}
__device__ int reflect101(long long x, long long n) {
    if (n == 1) return 0;
    const long long period = 2 * n - 2;
    x %= period;
    if (x < 0) x += period;
    return static_cast<int>(x < n ? x : period - x);
}
__device__ float tap(int x) {
    const int t = x < 0 ? -x : x;
    return t == 0 ? 0.375f : (t == 1 ? 0.25f : 0.0625f);
}
__device__ float expanded(const float* source, int cw, int ch, int channels,
                          int x, int y, int c) {
    float sum = 0;
    for (int dy = -2; dy <= 2; ++dy) {
        const int yy = reflect101(static_cast<long long>(y) + dy, 2LL * ch);
        if (yy & 1) continue;
        for (int dx = -2; dx <= 2; ++dx) {
            const int xx = reflect101(static_cast<long long>(x) + dx, 2LL * cw);
            if (xx & 1) continue;
            sum += tap(dx) * tap(dy) * source[(static_cast<std::size_t>(yy / 2) * cw + xx / 2) * channels + c];
        }
    }
    return 4 * sum;
}
// FP64 geometry avoids silently shrinking a footprint due to an FP32 inverse.
__device__ bool center_coordinate(double& x, int n, double tolerance) {
    if (!isfinite(x)) return false;
    const double nearest = nearbyint(x);
    const double roundoff = 16 * 2.2204460492503131e-16 * fmax(1.0, fabs(x));
    if (fabs(x - nearest) <= fmax(tolerance, roundoff)) x = nearest;
    return x >= 0 && x <= n - 1.0;
}
__device__ bool sample(const ImageView& image, double x, double y, float* rgb,
                       double& score, double tolerance) {
    if (!center_coordinate(x, image.width, tolerance) || !center_coordinate(y, image.height, tolerance))
        return false;
    const int x0 = static_cast<int>(floor(x)), y0 = static_cast<int>(floor(y));
    const int x1 = x0 < image.width - 1 ? x0 + 1 : x0;
    const int y1 = y0 < image.height - 1 ? y0 + 1 : y0;
    const double ax = x - x0, ay = y - y0;
    const auto* row0 = image.data + static_cast<std::size_t>(y0) * image.pitch_bytes;
    const auto* row1 = image.data + static_cast<std::size_t>(y1) * image.pitch_bytes;
    for (int c = 0; c < 3; ++c) {
        const double v0 = (1 - ax) * row0[3 * x0 + c] + ax * row0[3 * x1 + c];
        const double v1 = (1 - ax) * row1[3 * x0 + c] + ax * row1[3 * x1 + c];
        rgb[c] = static_cast<float>(((1 - ay) * v0 + ay * v1) / 255.0);
    }
    score = fmin(fmin((x + 0.5) / image.width, (image.width - 0.5 - x) / image.width),
                 fmin((y + 0.5) / image.height, (image.height - 0.5 - y) / image.height));
    return true;
}
__device__ std::uint8_t quantize(float x) {
    return static_cast<std::uint8_t>(floorf(fminf(255.0f, fmaxf(0.0f, 255.0f * x)) + 0.5f));
}
__global__ void warp(const ImageView a, const ImageView b, Geometry g,
                     ImageView output, ImageView mask, float* ca, float* cb,
                     float* va, float* vb, float* ua, float* ub) {
    const std::size_t n = static_cast<std::size_t>(g.width) * g.height;
    for (auto p = thread_index(); p < n; p += thread_step()) {
        const int x = static_cast<int>(p % g.width), y = static_cast<int>(p / g.width);
        const double xa = static_cast<double>(x) + g.origin_x;
        const double ya = static_cast<double>(y) + g.origin_y;
        float ra[3] = {}, rb[3] = {};
        double sa = 0, sb = 0;
        const bool ma = sample(a, xa, ya, ra, sa, g.snap_tolerance);
        const double den = g.inverse[6] * xa + g.inverse[7] * ya + g.inverse[8];
        const double den_scale = fabs(g.inverse[6] * xa) + fabs(g.inverse[7] * ya) + fabs(g.inverse[8]);
        bool mb = false;
        if (isfinite(den) && den_scale > 0 && fabs(den) > g.denominator_tolerance * den_scale) {
            const double xb = (g.inverse[0] * xa + g.inverse[1] * ya + g.inverse[2]) / den;
            const double yb = (g.inverse[3] * xa + g.inverse[4] * ya + g.inverse[5]) / den;
            mb = sample(b, xb, yb, rb, sb, g.snap_tolerance);
        }
        mask.data[static_cast<std::size_t>(y) * mask.pitch_bytes + x] = ma || mb ? 1 : 0;
        if (ca) {
            va[p] = ma ? 1.0f : 0.0f; vb[p] = mb ? 1.0f : 0.0f;
            ua[p] = ma && (!mb || sa >= sb) ? 1.0f : 0.0f;
            ub[p] = mb && (!ma || sb > sa) ? 1.0f : 0.0f;
            for (int c = 0; c < 3; ++c) { ca[3 * p + c] = ra[c]; cb[3 * p + c] = rb[c]; }
        } else {
            auto* out = output.data + static_cast<std::size_t>(y) * output.pitch_bytes + 3 * x;
            for (int c = 0; c < 3; ++c) {
                const float color = ma && mb ? static_cast<float>((sa * ra[c] + sb * rb[c]) / (sa + sb)) : (ma ? ra[c] : rb[c]);
                out[c] = quantize(color);
            }
        }
    }
}
__global__ void reduce(const float* source, int w, int h, float* dest,
                       int dw, int dh, int channels) {
    const std::size_t n = static_cast<std::size_t>(dw) * dh;
    for (auto p = thread_index(); p < n; p += thread_step()) {
        const int x = static_cast<int>(p % dw), y = static_cast<int>(p / dw);
        for (int c = 0; c < channels; ++c) {
            float value = 0;
            for (int dy = -2; dy <= 2; ++dy) {
                const int yy = reflect101(2LL * y + dy, h);
                for (int dx = -2; dx <= 2; ++dx) {
                    const int xx = reflect101(2LL * x + dx, w);
                    value += tap(dx) * tap(dy) * source[(static_cast<std::size_t>(yy) * w + xx) * channels + c];
                }
            }
            dest[p * channels + c] = value;
        }
    }
}
__global__ void normalize_color(float* numerator, const float* validity, std::size_t n) {
    for (auto p = thread_index(); p < n; p += thread_step()) {
        const float weight = validity[p];
        for (int c = 0; c < 3; ++c)
            numerator[3 * p + c] = weight > 0 ? numerator[3 * p + c] / weight : 0;
    }
}
__global__ void push_missing(float* dest, const float* validity, int w, int h,
                             const float* coarse, int cw, int ch) {
    const std::size_t n = static_cast<std::size_t>(w) * h;
    for (auto p = thread_index(); p < n; p += thread_step()) {
        if (validity[p] > 0) continue;
        const int x = static_cast<int>(p % w), y = static_cast<int>(p / w);
        for (int c = 0; c < 3; ++c)
            dest[3 * p + c] = expanded(coarse, cw, ch, 3, x, y, c);
    }
}
__global__ void mix_band(const float* a, const float* b, const float* ua,
                         const float* ub, const float* coarse_a, const float* coarse_b,
                         int w, int h, int cw, int ch, float* dest) {
    const std::size_t n = static_cast<std::size_t>(w) * h;
    for (auto p = thread_index(); p < n; p += thread_step()) {
        const int x = static_cast<int>(p % w), y = static_cast<int>(p / w);
        const float total = ua[p] + ub[p];
        for (int c = 0; c < 3; ++c) {
            const float la = a[3 * p + c] - (coarse_a ? expanded(coarse_a, cw, ch, 3, x, y, c) : 0);
            const float lb = b[3 * p + c] - (coarse_b ? expanded(coarse_b, cw, ch, 3, x, y, c) : 0);
            dest[3 * p + c] = total > 0 ? (ua[p] * la + ub[p] * lb) / total : 0;
        }
    }
}
__global__ void reconstruct(float* fine, int w, int h, const float* coarse,
                            int cw, int ch) {
    const std::size_t n = static_cast<std::size_t>(w) * h;
    for (auto p = thread_index(); p < n; p += thread_step()) {
        const int x = static_cast<int>(p % w), y = static_cast<int>(p / w);
        for (int c = 0; c < 3; ++c)
            fine[3 * p + c] += expanded(coarse, cw, ch, 3, x, y, c);
    }
}
__global__ void finish(const float* a, const float* b, const float* va,
                       const float* vb, const float* result, ImageView output) {
    const std::size_t n = static_cast<std::size_t>(output.width) * output.height;
    for (auto p = thread_index(); p < n; p += thread_step()) {
        const int x = static_cast<int>(p % output.width), y = static_cast<int>(p / output.width);
        auto* out = output.data + static_cast<std::size_t>(y) * output.pitch_bytes + 3 * x;
        for (int c = 0; c < 3; ++c) {
            const float color = va[p] > 0 && vb[p] > 0 ? result[3 * p + c] :
                                (va[p] > 0 ? a[3 * p + c] : (vb[p] > 0 ? b[3 * p + c] : 0));
            out[c] = quantize(color);
        }
    }
}
}  // namespace

int effective_reductions(const Geometry& g, BlendMode mode) {
    if (!valid_mode(mode) || !valid_geometry(g)) return -1;
    if (mode == BlendMode::Feather) return 0;
    int reductions = 0, w = g.width, h = g.height;
    const int requested = mode == BlendMode::Multiband3 ? 3 : 5;
    while ((w > 1 || h > 1) && reductions < requested) {
        w = w / 2 + w % 2; h = h / 2 + h % 2; ++reductions;
    }
    return reductions;
}
Status s6_workspace_bytes(const Geometry& g, BlendMode mode, std::size_t* bytes) {
    if (!bytes) return Status::INVALID_ARGUMENT;
    *bytes = 0;
    Layout layout;
    const Status status = layout_for(g, mode, layout);
    if (status == Status::OK) *bytes = layout.bytes;
    return status;
}

Status render(const ImageView& a, const ImageView& b, const Geometry& g,
              BlendMode mode, const ImageView& output, const ImageView& mask,
              void* workspace, std::size_t available, cudaStream_t stream) {
    Layout layout;
    const auto status = layout_for(g, mode, layout);
    if (status != Status::OK) return status;
    if (a.width != g.width_a || a.height != g.height_a || b.width != g.width_b ||
        b.height != g.height_b || output.width != g.width || output.height != g.height ||
        mask.width != g.width || mask.height != g.height) return Status::INVALID_ARGUMENT;
    Range ranges[4];
    if (!image_range(a, PixelFormat::RGB8, ranges[0]) || !image_range(b, PixelFormat::RGB8, ranges[1]) ||
        !image_range(output, PixelFormat::RGB8, ranges[2]) || !image_range(mask, PixelFormat::GRAY8, ranges[3]))
        return Status::INVALID_ARGUMENT;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (!(i == 0 && j == 1) && overlaps(ranges[i], ranges[j])) return Status::INVALID_ARGUMENT;
    if (available < layout.bytes || (layout.bytes && !workspace)) return Status::INSUFFICIENT_WORKSPACE;
    if (layout.bytes) {
        const auto address = reinterpret_cast<std::uintptr_t>(workspace);
        if (address % kAlignment || address > std::numeric_limits<std::uintptr_t>::max() - available)
            return Status::INVALID_ARGUMENT;
        const Range arena{address, address + available};
        for (const auto& range : ranges) if (overlaps(range, arena)) return Status::INVALID_ARGUMENT;
    }
    const std::size_t n = static_cast<std::size_t>(g.width) * g.height;
    const unsigned grid = blocks_for(n);
#define S6_LAUNCH_CHECK() do { if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR; } while (false)
    if (mode == BlendMode::Feather) {
        warp<<<grid, kBlock, 0, stream>>>(a, b, g, output, mask, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        S6_LAUNCH_CHECK();
        return Status::OK;
    }
    auto ptr = [&](std::size_t offset) { return buffer(workspace, offset); };
    const auto& base = layout.levels[0];
    warp<<<grid, kBlock, 0, stream>>>(a, b, g, output, mask, ptr(base.a), ptr(base.b), ptr(base.va), ptr(base.vb), ptr(base.ua), ptr(base.ub));
    S6_LAUNCH_CHECK();
    // Pull raw numerator/validity all the way to 1x1 before any normalization.
    for (int l = 1; l <= layout.top; ++l) {
        const auto& fine = layout.levels[l - 1]; const auto& coarse = layout.levels[l];
        const auto blocks = blocks_for(static_cast<std::size_t>(coarse.w) * coarse.h);
        for (int source = 0; source < 2; ++source) {
            reduce<<<blocks, kBlock, 0, stream>>>(ptr(source ? fine.b : fine.a), fine.w, fine.h, ptr(source ? coarse.b : coarse.a), coarse.w, coarse.h, 3);
            S6_LAUNCH_CHECK();
            reduce<<<blocks, kBlock, 0, stream>>>(ptr(source ? fine.vb : fine.va), fine.w, fine.h, ptr(source ? coarse.vb : coarse.va), coarse.w, coarse.h, 1);
            S6_LAUNCH_CHECK();
        }
    }
    for (int l = 0; l <= layout.top; ++l) {
        const auto& level = layout.levels[l];
        const auto pixels = static_cast<std::size_t>(level.w) * level.h;
        normalize_color<<<blocks_for(pixels), kBlock, 0, stream>>>(ptr(level.a), ptr(level.va), pixels); S6_LAUNCH_CHECK();
        normalize_color<<<blocks_for(pixels), kBlock, 0, stream>>>(ptr(level.b), ptr(level.vb), pixels); S6_LAUNCH_CHECK();
    }
    for (int l = layout.top - 1; l >= 0; --l) {
        const auto& fine = layout.levels[l]; const auto& coarse = layout.levels[l + 1];
        const auto blocks = blocks_for(static_cast<std::size_t>(fine.w) * fine.h);
        push_missing<<<blocks, kBlock, 0, stream>>>(ptr(fine.a), ptr(fine.va), fine.w, fine.h, ptr(coarse.a), coarse.w, coarse.h); S6_LAUNCH_CHECK();
        push_missing<<<blocks, kBlock, 0, stream>>>(ptr(fine.b), ptr(fine.vb), fine.w, fine.h, ptr(coarse.b), coarse.w, coarse.h); S6_LAUNCH_CHECK();
    }
    // Filled coarser extension levels can now be reused for image Gaussians.
    for (int l = 1; l <= layout.blend_top; ++l) {
        const auto& fine = layout.levels[l - 1]; const auto& coarse = layout.levels[l];
        const auto blocks = blocks_for(static_cast<std::size_t>(coarse.w) * coarse.h);
        reduce<<<blocks, kBlock, 0, stream>>>(ptr(fine.a), fine.w, fine.h, ptr(coarse.a), coarse.w, coarse.h, 3); S6_LAUNCH_CHECK();
        reduce<<<blocks, kBlock, 0, stream>>>(ptr(fine.b), fine.w, fine.h, ptr(coarse.b), coarse.w, coarse.h, 3); S6_LAUNCH_CHECK();
        reduce<<<blocks, kBlock, 0, stream>>>(ptr(fine.ua), fine.w, fine.h, ptr(coarse.ua), coarse.w, coarse.h, 1); S6_LAUNCH_CHECK();
        reduce<<<blocks, kBlock, 0, stream>>>(ptr(fine.ub), fine.w, fine.h, ptr(coarse.ub), coarse.w, coarse.h, 1); S6_LAUNCH_CHECK();
    }
    for (int l = 0; l <= layout.blend_top; ++l) {
        const auto& fine = layout.levels[l];
        const Level* coarse = l < layout.blend_top ? &layout.levels[l + 1] : nullptr;
        mix_band<<<blocks_for(static_cast<std::size_t>(fine.w) * fine.h), kBlock, 0, stream>>>(ptr(fine.a), ptr(fine.b), ptr(fine.ua), ptr(fine.ub), coarse ? ptr(coarse->a) : nullptr, coarse ? ptr(coarse->b) : nullptr, fine.w, fine.h, coarse ? coarse->w : 0, coarse ? coarse->h : 0, ptr(fine.z));
        S6_LAUNCH_CHECK();
    }
    for (int l = layout.blend_top - 1; l >= 0; --l) {
        const auto& fine = layout.levels[l]; const auto& coarse = layout.levels[l + 1];
        reconstruct<<<blocks_for(static_cast<std::size_t>(fine.w) * fine.h), kBlock, 0, stream>>>(ptr(fine.z), fine.w, fine.h, ptr(coarse.z), coarse.w, coarse.h); S6_LAUNCH_CHECK();
    }
    finish<<<grid, kBlock, 0, stream>>>(ptr(base.a), ptr(base.b), ptr(base.va), ptr(base.vb), ptr(base.z), output); S6_LAUNCH_CHECK();
#undef S6_LAUNCH_CHECK
    return Status::OK;
}
}  // namespace stitch::s6
