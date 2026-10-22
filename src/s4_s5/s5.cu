#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include "common/geometry.hpp"
#include "s4_s5/s5_internal.hpp"
#include "s4_s5/s5_math.hpp"

namespace stitch::s5 {
namespace {
using namespace detail;
constexpr int threads = 128;

struct Points {
    Keypoints a, b;
    Matches m;
    __host__ __device__ Point operator()(int i) const {
        return {a.x[m.idx_a[i]], a.y[m.idx_a[i]], b.x[m.idx_b[i]], b.y[m.idx_b[i]]};
    }
};

struct Four {
    Point p[4];
    __host__ __device__ Point operator()(int i) const { return p[i]; }
};

struct Inliers {
    Points points;
    const double* h;
    double threshold2;
    __host__ __device__ bool operator()(int i) const { return residual_squared(h, points(i)) <= threshold2; }
};

__device__ bool same_scale(const ScaleInfo& a, const ScaleInfo& b) {
    if (a.orig_w != b.orig_w || a.orig_h != b.orig_h || a.work_w != b.work_w ||
        a.work_h != b.work_h || a.sx != b.sx || a.sy != b.sy) return false;
    for (int i = 0; i < 9; ++i) if (a.D[i] != b.D[i]) return false;
    return true;
}

__global__ void prepare(Points p, ScaleInfo sa, ScaleInfo sb, Config c, State* s) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int na = *p.a.d_count, nb = *p.b.d_count, nm = *p.m.d_count;
    if (c.previous_hypotheses == 0) {
        s->best = Model{};
        s->ax = p.a.x; s->ay = p.a.y; s->bx = p.b.x; s->by = p.b.y;
        s->idx_a = p.m.idx_a; s->idx_b = p.m.idx_b;
        s->count_a = p.a.d_count; s->count_b = p.b.d_count; s->count_matches = p.m.d_count;
        s->na = na; s->nb = nb; s->nm = nm;
        s->capacity_a = p.a.capacity; s->capacity_b = p.b.capacity; s->capacity_matches = p.m.capacity;
        s->evaluated = 0; s->batch_size = c.batch_size; s->threshold = c.threshold;
        s->seed = c.seed; s->generation = c.generation;
        s->scale_a = sa; s->scale_b = sb;
        s->status = Status::OK;
    } else if (s->evaluated != c.previous_hypotheses || s->generation != c.generation ||
               s->seed != c.seed || s->threshold != c.threshold || s->batch_size != c.batch_size ||
               s->ax != p.a.x || s->ay != p.a.y || s->bx != p.b.x || s->by != p.b.y ||
               s->idx_a != p.m.idx_a || s->idx_b != p.m.idx_b ||
               s->count_a != p.a.d_count || s->count_b != p.b.d_count || s->count_matches != p.m.d_count ||
               s->capacity_a != p.a.capacity || s->capacity_b != p.b.capacity ||
               s->capacity_matches != p.m.capacity || s->na != na || s->nb != nb || s->nm != nm ||
               !same_scale(s->scale_a, sa) || !same_scale(s->scale_b, sb)) {
        s->status = Status::INVALID_ARGUMENT;
    }
    if (s->status != Status::OK) return;
    if (na < 0 || na > p.a.capacity || nb < 0 || nb > p.b.capacity || nm < 0 || nm > p.m.capacity) {
        s->status = Status::INVALID_ARGUMENT;
        return;
    }
    for (int i = 0; i < nm; ++i) {
        if (p.m.idx_a[i] < 0 || p.m.idx_a[i] >= na || p.m.idx_b[i] < 0 || p.m.idx_b[i] >= nb) {
            s->status = Status::INVALID_ARGUMENT;
            return;
        }
        const Point q = p(i);
        if (!detail::finite(q.ax) || !detail::finite(q.ay) || !detail::finite(q.bx) || !detail::finite(q.by)) {
            s->status = Status::INVALID_ARGUMENT;
            return;
        }
    }
    if (nm < 4) s->status = Status::NOT_ENOUGH_POINTS;
}

__global__ void generate(Points p, Config c, int first, int count, State* s, Model* models) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Model model{};
    model.hypothesis = first + i;
    if (s->status == Status::OK) {
        int indices[4];
        sample_four(s->nm, c.seed, model.hypothesis, indices);
        Four f{};
        for (int j = 0; j < 4; ++j) f.p[j] = p(indices[j]);
        model.valid = nondegenerate_four(f.p) && fit(4, f, All{}, model.h) &&
                      valid_domain(model.h, s->scale_b.work_w, s->scale_b.work_h);
    }
    models[i] = model;
}

__global__ void score(Points p, double threshold2, State* s, Model* models) {
    const int id = blockIdx.x, lane = threadIdx.x;
    __shared__ double loss[threads];
    double value = 0.0;
    const Model model = models[id];
    if (s->status == Status::OK && model.valid) {
        for (int i = lane; i < s->nm; i += threads) {
            const double r = residual_squared(model.h, p(i));
            value += r < threshold2 ? r : threshold2;
        }
    }
    loss[lane] = value;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) loss[lane] += loss[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        models[id].loss = loss[0];
        if (!detail::finite(loss[0])) models[id].valid = false;
    }
}

__device__ bool better(const Model& a, const Model& b) {
    return a.valid && (!b.valid || a.loss < b.loss ||
           (a.loss == b.loss && a.hypothesis < b.hypothesis));
}

__global__ void select_best(int count, State* s, const Model* models) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || s->status != Status::OK) return;
    for (int i = 0; i < count; ++i) if (better(models[i], s->best)) s->best = models[i];
}

struct Transform {
    double inverse_a[9];
    double d_b[9];
};

__device__ double serial_loss(int n, Points p, const double* h, double threshold2) {
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
        const double r = residual_squared(h, p(i));
        value += r < threshold2 ? r : threshold2;
    }
    return value;
}

__global__ void finish(Points p, Config c, Transform t, State* s, std::uint8_t* mask) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    HomographyResult result{};
    const double invalid = nan("");
    for (int i = 0; i < 9; ++i) result.H[i] = result.H_s[i] = invalid;
    result.reproj_residual = invalid;
    result.num_matches = s->nm >= 0 && s->nm <= p.m.capacity ? s->nm : 0;
    result.status = s->status;
    for (int i = 0; i < p.m.capacity; ++i) mask[i] = 0;
    if (s->status == Status::OK && !s->best.valid) result.status = Status::DEGENERATE;
    if (result.status == Status::OK) {
        Model model = s->best;
        const double threshold2 = c.threshold * c.threshold;
        // Re-score in the same order for both sides of refinement comparisons.
        model.loss = serial_loss(s->nm, p, model.h, threshold2);
        for (int step = 0; step < c.refinement_steps; ++step) {
            Model candidate{};
            candidate.valid = fit(s->nm, p, Inliers{p, model.h, threshold2}, candidate.h) &&
                              valid_domain(candidate.h, s->scale_b.work_w, s->scale_b.work_h);
            if (!candidate.valid) break;
            candidate.loss = serial_loss(s->nm, p, candidate.h, threshold2);
            if (!detail::finite(candidate.loss) || !(candidate.loss < model.loss)) break;
            model = candidate;
        }
        double min_x = 1.7976931348623157e308, min_y = min_x;
        double max_x = -min_x, max_y = -min_x, sum = 0.0;
        for (int i = 0; i < s->nm; ++i) {
            const Point q = p(i);
            const double r = residual_squared(model.h, q);
            if (r > threshold2) continue;
            mask[i] = 1;
            ++result.num_inliers;
            sum += r;
            min_x = fmin(min_x, q.ax); max_x = fmax(max_x, q.ax);
            min_y = fmin(min_y, q.ay); max_y = fmax(max_y, q.ay);
        }
        if (result.num_inliers < 4) {
            result.status = Status::DEGENERATE;
            for (int i = 0; i < p.m.capacity; ++i) mask[i] = 0;
            result.num_inliers = 0;
        } else {
            result.inlier_ratio = static_cast<double>(result.num_inliers) / s->nm;
            result.reproj_residual = sqrt(sum / result.num_inliers);
            const double width = fmax(1.0, static_cast<double>(s->scale_a.work_w - 1));
            const double height = fmax(1.0, static_cast<double>(s->scale_a.work_h - 1));
            result.inlier_coverage = fmin(1.0, (max_x - min_x) / width) *
                                     fmin(1.0, (max_y - min_y) / height);
            for (int i = 0; i < 9; ++i) result.H_s[i] = model.h[i];
            double tmp[9];
            multiply(t.inverse_a, model.h, tmp);
            multiply(tmp, t.d_b, result.H);
            if (!normalize(result.H) ||
                !valid_domain(result.H, s->scale_b.orig_w, s->scale_b.orig_h)) {
                result.status = Status::DEGENERATE;
                for (int i = 0; i < p.m.capacity; ++i) mask[i] = 0;
                result.num_inliers = 0;
                result.inlier_ratio = result.inlier_coverage = 0.0;
                result.reproj_residual = invalid;
                for (int i = 0; i < 9; ++i) result.H[i] = result.H_s[i] = invalid;
            }
        }
    }
    s->evaluated = c.hypotheses;
    s->result = result;
}

bool valid_keypoints(const Keypoints& p) {
    return p.capacity >= 0 && p.d_count && (p.capacity == 0 || (p.x && p.y));
}

bool valid_scale(const ScaleInfo& s) {
    if (s.orig_w <= 0 || s.orig_h <= 0 || s.work_w <= 0 || s.work_h <= 0 ||
        !(s.sx > 0.0) || !(s.sy > 0.0) || !std::isfinite(s.sx) || !std::isfinite(s.sy)) return false;
    for (double x : s.D) if (!std::isfinite(x)) return false;
    return true;
}
}  // namespace

Status estimate(const Keypoints& a, const Keypoints& b, const Matches& m,
                const ScaleInfo& sa, const ScaleInfo& sb, const Config& c,
                std::uint8_t* mask, HomographyResult* output,
                void* workspace, std::size_t bytes, cudaStream_t stream) {
    std::size_t required = 0;
    const Status query = s5_workspace_bytes(c, &required);
    if (query != Status::OK) return query;
    if (!valid_keypoints(a) || !valid_keypoints(b) || m.capacity < 0 || !m.d_count ||
        (m.capacity > 0 && (!m.idx_a || !m.idx_b || !mask)) || !output ||
        !valid_scale(sa) || !valid_scale(sb)) return Status::INVALID_ARGUMENT;
    if (bytes < required) return Status::INSUFFICIENT_WORKSPACE;
    if (!workspace || reinterpret_cast<std::uintptr_t>(workspace) % 256 != 0)
        return Status::INVALID_ARGUMENT;
    Transform t{};
    if (!mat3_inverse(sa.D, t.inverse_a)) return Status::INVALID_ARGUMENT;
    double inverse_b[9];
    if (!mat3_inverse(sb.D, inverse_b)) return Status::INVALID_ARGUMENT;
    for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(t.inverse_a[i]) || !std::isfinite(inverse_b[i])) return Status::INVALID_ARGUMENT;
        t.d_b[i] = sb.D[i];
    }
    auto* state = static_cast<State*>(workspace);
    auto* models = reinterpret_cast<Model*>(static_cast<char*>(workspace) + model_offset);
    const Points points{a, b, m};
    prepare<<<1, 1, 0, stream>>>(points, sa, sb, c, state);
    if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
    for (int first = c.previous_hypotheses; first < c.hypotheses;) {
        const int count = c.hypotheses - first < c.batch_size ? c.hypotheses - first : c.batch_size;
        generate<<<(count + threads - 1) / threads, threads, 0, stream>>>(points, c, first, count, state, models);
        if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
        score<<<count, threads, 0, stream>>>(points, c.threshold * c.threshold, state, models);
        if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
        select_best<<<1, 1, 0, stream>>>(count, state, models);
        if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
        first += count;
    }
    finish<<<1, 1, 0, stream>>>(points, c, t, state, mask);
    if (cudaGetLastError() != cudaSuccess) return Status::CUDA_ERROR;
    if (cudaMemcpyAsync(output, &state->result, sizeof(*output), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return Status::CUDA_ERROR;
    return Status::OK;
}

}  // namespace stitch::s5
