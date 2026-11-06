// Pipeline runner implementation.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "harness/pipeline_runner.hpp"
#include "harness/cuda_error.hpp"
#include "common/admission_control.hpp"

#define CUDA_CHECK(call) do {                                                  \
    const cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                                  \
        std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(err));                                 \
        throw CudaError(err);                                                  \
    }                                                                          \
} while (false)

namespace stitch::harness {

namespace {

constexpr int kPadding = 64;

const char* status_to_str(Status s) {
    switch (s) {
        case Status::OK: return "OK";
        case Status::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case Status::INSUFFICIENT_WORKSPACE: return "INSUFFICIENT_WORKSPACE";
        case Status::CUDA_ERROR: return "CUDA_ERROR";
        case Status::NOT_ENOUGH_POINTS: return "NOT_ENOUGH_POINTS";
        case Status::DEGENERATE: return "DEGENERATE";
        default: return "UNKNOWN";
    }
}

}  // namespace

PipelineRunner::PipelineRunner() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&ev_start_));
    CUDA_CHECK(cudaEventCreate(&ev_s1_));
    CUDA_CHECK(cudaEventCreate(&ev_s2_));
    CUDA_CHECK(cudaEventCreate(&ev_s3_));
    CUDA_CHECK(cudaEventCreate(&ev_s4_));
    CUDA_CHECK(cudaEventCreate(&ev_s5_));
    CUDA_CHECK(cudaEventCreate(&ev_s6_));
}

PipelineRunner::~PipelineRunner() {
    if (stream_) cudaStreamDestroy(stream_);
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_s1_) cudaEventDestroy(ev_s1_);
    if (ev_s2_) cudaEventDestroy(ev_s2_);
    if (ev_s3_) cudaEventDestroy(ev_s3_);
    if (ev_s4_) cudaEventDestroy(ev_s4_);
    if (ev_s5_) cudaEventDestroy(ev_s5_);
    if (ev_s6_) cudaEventDestroy(ev_s6_);

    if (dev_img_a_.data) cudaFree(dev_img_a_.data);
    if (dev_img_b_.data) cudaFree(dev_img_b_.data);
    if (dev_gray_a_.data) cudaFree(dev_gray_a_.data);
    if (dev_gray_b_.data) cudaFree(dev_gray_b_.data);

    if (kp_a_.x) cudaFree(kp_a_.x);
    if (kp_a_.y) cudaFree(kp_a_.y);
    if (kp_a_.response) cudaFree(kp_a_.response);
    if (d_kp_count_a_) cudaFree(d_kp_count_a_);

    if (kp_b_.x) cudaFree(kp_b_.x);
    if (kp_b_.y) cudaFree(kp_b_.y);
    if (kp_b_.response) cudaFree(kp_b_.response);
    if (d_kp_count_b_) cudaFree(d_kp_count_b_);

    if (desc_a_.data) cudaFree(desc_a_.data);
    if (desc_b_.data) cudaFree(desc_b_.data);

    if (matches_.idx_a) cudaFree(matches_.idx_a);
    if (matches_.idx_b) cudaFree(matches_.idx_b);
    if (matches_.distance) cudaFree(matches_.distance);
    if (d_matches_count_) cudaFree(d_matches_count_);

    if (d_inlier_mask_) cudaFree(d_inlier_mask_);
    if (pinned_h_res_) cudaFreeHost(pinned_h_res_);

    if (dev_canvas_.data) cudaFree(dev_canvas_.data);
    if (dev_s6_mask_.data) cudaFree(dev_s6_mask_.data);
    if (dev_stats_) cudaFree(dev_stats_);

    if (ws_s2_a_) cudaFree(ws_s2_a_);
    if (ws_s2_b_) cudaFree(ws_s2_b_);
    if (ws_s3_) cudaFree(ws_s3_);
    if (ws_s4_) cudaFree(ws_s4_);
    if (ws_s5_) cudaFree(ws_s5_);
    if (ws_s6_) cudaFree(ws_s6_);

    if (d_ws_probe_) cudaFree(d_ws_probe_);
    if (pinned_grad_a_) cudaFreeHost(pinned_grad_a_);
    if (pinned_grad_b_) cudaFreeHost(pinned_grad_b_);
}

PipelineMemoryPlan PipelineRunner::query_memory(int w, int h, int k, int r, int cw, int ch, s6::BlendMode mode) {
    using namespace admission;
    dimensions_fit(w, h, w, h, cw, ch);
    if (k <= 0 || r <= 0 || multiply(cw, ch) > s6::GeometryPolicy{}.max_canvas_pixels) throw Error("ERR_CAPACITY");
    PipelineMemoryPlan result{};
    result.device_bytes = add(multiply(2, image_bytes(w, h, 3)), multiply(2, image_bytes(w, h, 1)));
    result.device_bytes = add(result.device_bytes, add(image_bytes(cw, ch, 3), image_bytes(cw, ch, 1)));
    // Keypoint arrays/counts, descriptors, matches/count, inlier mask, stats/probe.
    result.device_bytes = add(result.device_bytes, add(multiply(k, 6*sizeof(float)+64+3*sizeof(int)+1), 3*sizeof(int)+sizeof(s6::RenderStatistics)+32));
    auto workspace = [&](std::size_t bytes, int count = 1) {
        result.workspace_bytes = add(result.workspace_bytes, multiply(bytes, count));
        if (bytes) result.device_bytes = add(result.device_bytes, multiply(add(bytes, 256), count));
    };
    auto require = [](Status status) { if (status != Status::OK) throw Error("ERR_CAPACITY"); };
    std::size_t bytes = 0;
    require(s2::s2_workspace_bytes(w, h, {20, k, .04f}, &bytes)); workspace(bytes, 2);
    require(s3::s3_workspace_bytes(w, h, &bytes)); workspace(bytes);
    require(s4::s4_workspace_bytes(k, k, {.8, false}, &bytes)); workspace(bytes);
    require(s5::s5_workspace_bytes({r, 0, 64, 1, 3., 0, 1}, &bytes)); workspace(bytes);
    s6::Geometry g{}; g.width = cw; g.height = ch;
    g.width_a = g.width_b = w; g.height_a = g.height_b = h;
    g.inverse[0] = g.inverse[4] = g.inverse[8] = 1;
    require(s6::s6_workspace_bytes(g, mode, &result.s6_workspace_bytes));
    workspace(result.s6_workspace_bytes);
    return result;
}

void PipelineRunner::initialize(int max_w, int max_h, int max_k, int max_r,
                                int canvas_width, int canvas_height, s6::BlendMode workspace_mode,
                                std::size_t device_reserve) {
    if (initialized_) return;
    const auto plan = query_memory(max_w, max_h, max_k, max_r, canvas_width, canvas_height, workspace_mode);
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    admission::fit(plan.device_bytes, free_bytes, device_reserve, "ERR_DEVICE_OOM");
    input_capacity_w_ = max_w; input_capacity_h_ = max_h;
    canvas_capacity_w_ = canvas_width; canvas_capacity_h_ = canvas_height;
    hypothesis_capacity_ = max_r;

    // Allocate input image buffers
    dev_img_a_.width = max_w;
    dev_img_a_.height = max_h;
    dev_img_a_.pitch_bytes = max_w * 3 + kPadding;
    dev_img_a_.format = PixelFormat::RGB8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_img_a_.data),
                          static_cast<std::size_t>(dev_img_a_.pitch_bytes) * max_h));

    dev_img_b_.width = max_w;
    dev_img_b_.height = max_h;
    dev_img_b_.pitch_bytes = max_w * 3 + kPadding;
    dev_img_b_.format = PixelFormat::RGB8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_img_b_.data),
                          static_cast<std::size_t>(dev_img_b_.pitch_bytes) * max_h));

    // S1 preprocessed gray images
    dev_gray_a_.width = max_w;
    dev_gray_a_.height = max_h;
    dev_gray_a_.pitch_bytes = max_w + kPadding;
    dev_gray_a_.format = PixelFormat::GRAY8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_gray_a_.data),
                          static_cast<std::size_t>(dev_gray_a_.pitch_bytes) * max_h));

    dev_gray_b_.width = max_w;
    dev_gray_b_.height = max_h;
    dev_gray_b_.pitch_bytes = max_w + kPadding;
    dev_gray_b_.format = PixelFormat::GRAY8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_gray_b_.data),
                          static_cast<std::size_t>(dev_gray_b_.pitch_bytes) * max_h));

    // S2 Keypoints buffers
    kp_a_.capacity = max_k;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&kp_a_.x), max_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&kp_a_.y), max_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&kp_a_.response), max_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kp_count_a_), sizeof(int)));
    kp_a_.d_count = d_kp_count_a_;

    kp_b_.capacity = max_k;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&kp_b_.x), max_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&kp_b_.y), max_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&kp_b_.response), max_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kp_count_b_), sizeof(int)));
    kp_b_.d_count = d_kp_count_b_;

    // S3 Descriptors buffers (BINARY_U8, 32 bytes)
    desc_a_.type = DescriptorType::BINARY_U8;
    desc_a_.dim = 32;
    desc_a_.row_stride_bytes = 32;
    CUDA_CHECK(cudaMalloc(&desc_a_.data, static_cast<std::size_t>(max_k) * 32));

    desc_b_.type = DescriptorType::BINARY_U8;
    desc_b_.dim = 32;
    desc_b_.row_stride_bytes = 32;
    CUDA_CHECK(cudaMalloc(&desc_b_.data, static_cast<std::size_t>(max_k) * 32));

    // S4 Matches
    matches_.capacity = max_k;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&matches_.idx_a), max_k * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&matches_.idx_b), max_k * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&matches_.distance), max_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_matches_count_), sizeof(int)));
    matches_.d_count = d_matches_count_;

    // S5 Inlier mask and pinned host result
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inlier_mask_), max_k * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_h_res_), sizeof(HomographyResult)));

    // S6 Canvas and Mask
    dev_canvas_.width = canvas_width;
    dev_canvas_.height = canvas_height;
    dev_canvas_.pitch_bytes = canvas_width * 3 + kPadding;
    dev_canvas_.format = PixelFormat::RGB8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_canvas_.data),
                          static_cast<std::size_t>(dev_canvas_.pitch_bytes) * canvas_height));

    dev_s6_mask_.width = canvas_width;
    dev_s6_mask_.height = canvas_height;
    dev_s6_mask_.pitch_bytes = canvas_width + kPadding;
    dev_s6_mask_.format = PixelFormat::GRAY8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_s6_mask_.data),
                          static_cast<std::size_t>(dev_s6_mask_.pitch_bytes) * canvas_height));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_stats_), sizeof(s6::RenderStatistics)));

    // Precalculate maximum workspace sizes
    s2::Config max_s2_cfg{20, max_k, 0.04f};
    CUDA_CHECK(s2::s2_workspace_bytes(max_w, max_h, max_s2_cfg, &ws_bytes_s2_a_) == Status::OK ? cudaSuccess : cudaErrorUnknown);
    ws_bytes_s2_b_ = ws_bytes_s2_a_;
    if (ws_bytes_s2_a_) {
        CUDA_CHECK(cudaMalloc(&ws_s2_a_, ws_bytes_s2_a_ + 256));
        CUDA_CHECK(cudaMalloc(&ws_s2_b_, ws_bytes_s2_b_ + 256));
    }

    CUDA_CHECK(s3::s3_workspace_bytes(max_w, max_h, &ws_bytes_s3_) == Status::OK ? cudaSuccess : cudaErrorUnknown);
    if (ws_bytes_s3_) CUDA_CHECK(cudaMalloc(&ws_s3_, ws_bytes_s3_ + 256));

    s4::Config max_s4_cfg{0.8, false};
    CUDA_CHECK(s4::s4_workspace_bytes(max_k, max_k, max_s4_cfg, &ws_bytes_s4_) == Status::OK ? cudaSuccess : cudaErrorUnknown);
    if (ws_bytes_s4_) CUDA_CHECK(cudaMalloc(&ws_s4_, ws_bytes_s4_ + 256));

    s5::Config max_s5_cfg{max_r, 0, 64, 1, 3.0, 0, 1};
    CUDA_CHECK(s5::s5_workspace_bytes(max_s5_cfg, &ws_bytes_s5_) == Status::OK ? cudaSuccess : cudaErrorUnknown);
    if (ws_bytes_s5_) CUDA_CHECK(cudaMalloc(&ws_s5_, ws_bytes_s5_ + 256));

    // Capacity/mode workspace was queried before any large device allocation.
    ws_bytes_s6_ = plan.s6_workspace_bytes;
    if (ws_bytes_s6_) CUDA_CHECK(cudaMalloc(&ws_s6_, ws_bytes_s6_ + 256));

    // S1 probe gradient reduction workspace and pinned host outputs
    CUDA_CHECK(cudaMalloc(&d_ws_probe_, 32));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_grad_a_), sizeof(double)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_grad_b_), sizeof(double)));

    initialized_ = true;
}

void PipelineRunner::synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

void PipelineRunner::upload_pair(const HostImageRGB& img_a, const HostImageRGB& img_b) {
    if (!initialized_) throw admission::Error("ERR_CAPACITY");
    admission::dimensions_fit(img_a.width, img_a.height, img_b.width, img_b.height, input_capacity_w_, input_capacity_h_);
    if (img_a.pixels.size() != admission::multiply(admission::multiply(img_a.width, img_a.height), 3) ||
        img_b.pixels.size() != admission::multiply(admission::multiply(img_b.width, img_b.height), 3)) throw admission::Error("ERR_INPUT");
    cur_w_a_ = img_a.width;
    cur_h_a_ = img_a.height;
    cur_w_b_ = img_b.width;
    cur_h_b_ = img_b.height;

    dev_img_a_.width = cur_w_a_;
    dev_img_a_.height = cur_h_a_;
    CUDA_CHECK(cudaMemcpy2DAsync(dev_img_a_.data, dev_img_a_.pitch_bytes,
                            img_a.pixels.data(), cur_w_a_ * 3,
                            cur_w_a_ * 3, cur_h_a_, cudaMemcpyHostToDevice, stream_));

    dev_img_b_.width = cur_w_b_;
    dev_img_b_.height = cur_h_b_;
    CUDA_CHECK(cudaMemcpy2DAsync(dev_img_b_.data, dev_img_b_.pitch_bytes,
                            img_b.pixels.data(), cur_w_b_ * 3,
                            cur_w_b_ * 3, cur_h_b_, cudaMemcpyHostToDevice, stream_));
    // Both input images must be device-ready before the measured interval.
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void PipelineRunner::execute_run(const HarnessConfig& cfg,
                                 const std::string& run_id,
                                 const std::string& git_rev,
                                 const std::string& dataset,
                                 const std::string& split,
                                 const std::string& pair_id,
                                 int repeat_idx,
                                 int is_warmup,
                                 RunRecord* record) {
    last_fixed_gpu_span_ms_ = -1;
    record->run_id = run_id;
    record->git_rev = git_rev;
    record->dataset = dataset;
    record->split = split;
    record->pair_id = pair_id;
    record->config_id = cfg.config_id;
    record->repeat_idx = repeat_idx;
    record->is_warmup = is_warmup;
    record->seed = cfg.seed;
    record->S = cfg.S;
    record->K = cfg.K;
    record->R = cfg.R;
    record->blend_mode = cfg.blend_mode_str;
    record->blend_mode_effective = cfg.blend_mode_str;

    // Reset counts on stream
    CUDA_CHECK(cudaMemsetAsync(d_kp_count_a_, 0, sizeof(int), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_kp_count_b_, 0, sizeof(int), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_matches_count_, 0, sizeof(int), stream_));

    ScaleInfo scale_a{}, scale_b{};
    make_scale_info(cur_w_a_, cur_h_a_, cfg.S, &scale_a);
    make_scale_info(cur_w_b_, cur_h_b_, cfg.S, &scale_b);

    dev_gray_a_.width = scale_a.work_w;
    dev_gray_a_.height = scale_a.work_h;
    dev_gray_b_.width = scale_b.work_w;
    dev_gray_b_.height = scale_b.work_h;

    // Start end-to-end host measurement
    const auto t_host_start = std::chrono::steady_clock::now();

    // Begin timing on CUDA stream
    CUDA_CHECK(cudaEventRecord(ev_start_, stream_));

    // Stage 1: Preprocessing
    Status s1_stat = s1::s1_preprocess(dev_img_a_, dev_img_b_, cfg.S,
                                       &dev_gray_a_, &dev_gray_b_,
                                       &scale_a, &scale_b, stream_);
    if (s1_stat != Status::OK) {
        record->status = status_to_str(s1_stat);
        record->fail_stage = "s1";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s1_, stream_));

    // Create per-attempt keypoint views with capacity = K (keep max_k if K > max_k so S2 returns INVALID_ARGUMENT)
    Keypoints kp_view_a = kp_a_;
    Keypoints kp_view_b = kp_b_;
    if (cfg.K >= 0 && cfg.K <= kp_a_.capacity) {
        kp_view_a.capacity = cfg.K;
        kp_view_b.capacity = cfg.K;
    }

    // Stage 2: FAST detection + Harris ranking (núm K)
    s2::Config s2_cfg{cfg.fast_threshold, cfg.K, cfg.harris_k};
    Status s2_stat_a = s2::s2_detect(dev_gray_a_, s2_cfg, kp_view_a, ws_s2_a_, ws_bytes_s2_a_, stream_);
    Status s2_stat_b = s2::s2_detect(dev_gray_b_, s2_cfg, kp_view_b, ws_s2_b_, ws_bytes_s2_b_, stream_);
    if (s2_stat_a != Status::OK || s2_stat_b != Status::OK) {
        record->status = status_to_str(s2_stat_a != Status::OK ? s2_stat_a : s2_stat_b);
        record->fail_stage = "s2";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s2_, stream_));

    // Stage 3: rBRIEF description
    Status s3_stat_a = s3::s3_describe(dev_gray_a_, kp_view_a, desc_a_, ws_s3_, ws_bytes_s3_, stream_);
    Status s3_stat_b = s3::s3_describe(dev_gray_b_, kp_view_b, desc_b_, ws_s3_, ws_bytes_s3_, stream_);
    if (s3_stat_a != Status::OK || s3_stat_b != Status::OK) {
        record->status = status_to_str(s3_stat_a != Status::OK ? s3_stat_a : s3_stat_b);
        record->fail_stage = "s3";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s3_, stream_));

    // Stage 4: Matching
    s4::Config s4_cfg{cfg.ratio_threshold, cfg.cross_check};
    Status s4_stat = s4::s4_match(kp_view_a, desc_a_, kp_view_b, desc_b_, s4_cfg, matches_, ws_s4_, ws_bytes_s4_, stream_);
    if (s4_stat != Status::OK) {
        record->status = status_to_str(s4_stat);
        record->fail_stage = "s4";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s4_, stream_));

    // Stage 5: MSAC Homography (núm R)
    s5::Config s5_cfg{cfg.R, 0, cfg.msac_batch_size, cfg.msac_refinements, cfg.msac_threshold, cfg.seed, 1};
    Status s5_stat = s5::estimate(kp_view_a, kp_view_b, matches_, scale_a, scale_b, s5_cfg,
                                  d_inlier_mask_, pinned_h_res_, ws_s5_, ws_bytes_s5_, stream_);
    if (s5_stat != Status::OK) {
        record->status = status_to_str(s5_stat);
        record->fail_stage = "s5";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s5_, stream_));

    // Host Sync 1: Synchronize stream to inspect homography result
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Download telemetry signals
    CUDA_CHECK(cudaMemcpy(&record->num_kp_a, kp_a_.d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&record->num_kp_b, kp_b_.d_count, sizeof(int), cudaMemcpyDeviceToHost));
    record->num_matches = pinned_h_res_->num_matches;
    record->num_inliers = pinned_h_res_->num_inliers;
    record->inlier_ratio = pinned_h_res_->inlier_ratio;
    record->inlier_coverage = pinned_h_res_->inlier_coverage;
    record->reproj_residual = pinned_h_res_->reproj_residual;
    std::memcpy(record->H, pinned_h_res_->H, 9 * sizeof(double));

    if (pinned_h_res_->status != Status::OK) {
        // S5 failed or degenerate
        const auto t_host_end = std::chrono::steady_clock::now();
        record->t_e2e = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();
        record->status = status_to_str(pinned_h_res_->status);
        record->fail_stage = "s5";

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start_, ev_s1_); record->t_s1 = ms;
        cudaEventElapsedTime(&ms, ev_s1_, ev_s2_); record->t_s2 = ms;
        cudaEventElapsedTime(&ms, ev_s2_, ev_s3_); record->t_s3 = ms;
        cudaEventElapsedTime(&ms, ev_s3_, ev_s4_); record->t_s4 = ms;
        cudaEventElapsedTime(&ms, ev_s4_, ev_s5_); record->t_s5 = ms;
        record->t_s6 = 0.0;
        return;
    }

    // Stage 6: Prepare geometry and render canvas
    s6::Geometry geom{};
    s6::GeometryPolicy geom_policy{};
    Status g_stat = s6::prepare_geometry(cur_w_a_, cur_h_a_, cur_w_b_, cur_h_b_, pinned_h_res_->H, geom_policy, &geom);
    if (g_stat != Status::OK) {
        const auto t_host_end = std::chrono::steady_clock::now();
        record->t_e2e = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();
        record->status = g_stat == Status::INSUFFICIENT_WORKSPACE ? "ERR_CANVAS_OVERFLOW" : status_to_str(g_stat);
        record->fail_stage = "s6";

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start_, ev_s1_); record->t_s1 = ms;
        cudaEventElapsedTime(&ms, ev_s1_, ev_s2_); record->t_s2 = ms;
        cudaEventElapsedTime(&ms, ev_s2_, ev_s3_); record->t_s3 = ms;
        cudaEventElapsedTime(&ms, ev_s3_, ev_s4_); record->t_s4 = ms;
        cudaEventElapsedTime(&ms, ev_s4_, ev_s5_); record->t_s5 = ms;
        record->t_s6 = 0.0;
        return;
    }

    if (geom.width > canvas_capacity_w_ || geom.height > canvas_capacity_h_) {
        record->status = "ERR_CANVAS_OVERFLOW"; record->fail_stage = "s6";
        record->t_e2e = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host_start).count();
        return;
    }
    dev_canvas_.width = geom.width;
    dev_canvas_.height = geom.height;
    dev_s6_mask_.width = geom.width;
    dev_s6_mask_.height = geom.height;

    Status s6_stat = s6::render(dev_img_a_, dev_img_b_, geom, cfg.blend_mode,
                                dev_canvas_, dev_s6_mask_, ws_s6_, ws_bytes_s6_, stream_, dev_stats_);
    if (s6_stat != Status::OK) {
        const auto t_host_end = std::chrono::steady_clock::now();
        record->t_e2e = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();
        record->status = status_to_str(s6_stat);
        record->fail_stage = "s6";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s6_, stream_));

    // Host Sync 2: Wait for S6 stream completion
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto t_host_end = std::chrono::steady_clock::now();
    record->t_e2e = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();

    // Event elapsed times
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start_, ev_s6_)); last_fixed_gpu_span_ms_ = ms;
    cudaEventElapsedTime(&ms, ev_start_, ev_s1_); record->t_s1 = ms;
    cudaEventElapsedTime(&ms, ev_s1_, ev_s2_); record->t_s2 = ms;
    cudaEventElapsedTime(&ms, ev_s2_, ev_s3_); record->t_s3 = ms;
    cudaEventElapsedTime(&ms, ev_s3_, ev_s4_); record->t_s4 = ms;
    cudaEventElapsedTime(&ms, ev_s4_, ev_s5_); record->t_s5 = ms;
    cudaEventElapsedTime(&ms, ev_s5_, ev_s6_); record->t_s6 = ms;

    // S6 telemetry
    s6::RenderStatistics host_stats{};
    CUDA_CHECK(cudaMemcpy(&host_stats, dev_stats_, sizeof(host_stats), cudaMemcpyDeviceToHost));
    record->canvas_pixels = static_cast<std::uint64_t>(geom.width) * geom.height;
    record->overlap_pixels = host_stats.overlap_pixels;
    record->num_levels = s6::effective_reductions(geom, cfg.blend_mode);
    record->peak_workspace_bytes = ws_bytes_s6_;
    record->cache_hit = 0;

    record->status = "OK";
    record->fail_stage = "";
}

void PipelineRunner::execute_adaptive_run(
    controller::Controller& controller,
    const std::string& run_id,
    const std::string& git_rev,
    const std::string& dataset,
    const std::string& split,
    const std::string& pair_id,
    int repeat_idx,
    int is_warmup,
    std::uint64_t seed,
    RunRecord* record) {
    *record = RunRecord{};
    record->run_id = run_id;
    record->git_rev = git_rev;
    record->dataset = dataset;
    record->split = split;
    record->pair_id = pair_id;
    record->repeat_idx = repeat_idx;
    record->is_warmup = is_warmup;
    record->seed = seed;
    record->config_id = "adaptive_unstarted";
    attempt_trace_.count = 0;
    record->attempt_count = 0;
    record->cache_hit = 0;

    using Clock = std::chrono::steady_clock;
    struct Span {
        double& total;
        Clock::time_point start = Clock::now();
        bool finished = false;
        void finish() {
            if (finished) return;
            total += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            finished = true;
        }
        ~Span() { finish(); }
    };

    Span host{record->t_e2e};
    auto timed_gpu = [&](double& elapsed, auto&& submit) {
        CUDA_CHECK(cudaEventRecord(ev_start_, stream_));
        const Status status = submit();
        CUDA_CHECK(cudaEventRecord(ev_s1_, stream_));
        {
            Span wait{record->t_stream_wait};
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start_, ev_s1_));
        elapsed += ms;
        return status;
    };
    auto fail = [&](Status status) { record->status = status_to_str(status); };
    AttemptRecord* active = nullptr;

    try {
        record->fail_stage = "s1";
        controller.reset_pair();
        const double probe_s = controller.probe_scale();
        ScaleInfo probe_a{}, probe_b{};
        if (make_scale_info(cur_w_a_, cur_h_a_, probe_s, &probe_a) != Status::OK ||
            make_scale_info(cur_w_b_, cur_h_b_, probe_s, &probe_b) != Status::OK) {
            record->status = "INVALID_PROBE_SCALE";
            record->fail_stage = "controller";
            return;
        }
        auto set_gray_dimensions = [&](const ScaleInfo& a, const ScaleInfo& b) {
            dev_gray_a_.width = a.work_w;
            dev_gray_a_.height = a.work_h;
            dev_gray_b_.width = b.work_w;
            dev_gray_b_.height = b.work_h;
        };
        set_gray_dimensions(probe_a, probe_b);
        const Status probe = timed_gpu(record->t_probe_gpu, [&] {
            const auto status = s1::s1_preprocess(dev_img_a_, dev_img_b_, probe_s, &dev_gray_a_, &dev_gray_b_,
                                                  &probe_a, &probe_b, stream_);
            if (status != Status::OK) return status;
            return s1::s1_compute_gradient(dev_gray_a_, dev_gray_b_, d_ws_probe_,
                                           pinned_grad_a_, pinned_grad_b_, stream_);
        });
        record->t_s1 = record->t_probe_gpu;
        if (probe != Status::OK) { fail(probe); return; }

        record->fail_stage = "controller";
        const controller::SceneSignals scene{{probe_a, *pinned_grad_a_}, {probe_b, *pinned_grad_b_}};
        controller::Knobs knobs{};
        Status begin;
        {
            Span cpu{record->t_controller};
            begin = controller.begin_pair(scene, &knobs);
        }
        if (begin != Status::OK) { fail(begin); return; }

        controller::Knobs previous{};
        std::uint64_t generation = 1;
        int completed_r = 0;
        while (true) {
            if (attempt_trace_.count >= static_cast<int>(AttemptTrace::capacity)) {
                record->status = "ATTEMPT_CAPACITY_EXCEEDED";
                record->fail_stage = "controller";
                return;
            }
            const bool first = attempt_trace_.count == 0;
            auto& a = attempt_trace_.records[attempt_trace_.count++];
            a = AttemptRecord{};
            active = &a;
            a.knobs = knobs;
            Span attempt{a.t_attempt};
            record->attempt_count = attempt_trace_.count;
            record->S = knobs.S;
            record->K = knobs.K;
            record->R = knobs.R;

            bool rerun_s1 = knobs.S != probe_s;
            bool rerun_features = true;
            if (!first) {
                controller::RetryPlan plan{};
                if (controller::plan_retry(previous, knobs, &plan) != Status::OK) {
                    record->status = "INVALID_RETRY_PLAN";
                    a.fail_stage = "controller";
                    a.execution_status = Status::INVALID_ARGUMENT;
                    record->fail_stage = a.fail_stage;
                    return;
                }
                rerun_s1 = plan.restart_from == controller::RestartFrom::S1;
                rerun_features = rerun_s1 || plan.restart_from == controller::RestartFrom::S2;
                if (plan.change_generation) { ++generation; completed_r = 0; }
                if (plan.extend_s5) a.eligible_R_old = a.previous_R = completed_r;
            }
            a.generation = generation;
            a.restart_from = rerun_s1 ? "s1" : rerun_features ? "s2" : "s5";
            auto stage = [&](const char* name, double& ms, double& total, auto&& submit) {
                record->fail_stage = name;
                a.fail_stage = name;
                const auto status = timed_gpu(ms, submit);
                total += ms;
                a.execution_status = status;
                if (status != Status::OK) fail(status);
                else a.fail_stage = "";
                return status;
            };

            ScaleInfo scale_a{}, scale_b{};
            if (make_scale_info(cur_w_a_, cur_h_a_, knobs.S, &scale_a) != Status::OK ||
                make_scale_info(cur_w_b_, cur_h_b_, knobs.S, &scale_b) != Status::OK ||
                knobs.K > kp_a_.capacity || knobs.K > kp_b_.capacity || knobs.R > hypothesis_capacity_) {
                a.fail_stage = "controller";
                a.execution_status = Status::INVALID_ARGUMENT;
                record->fail_stage = a.fail_stage;
                fail(a.execution_status);
                return;
            }
            if (rerun_s1) {
                set_gray_dimensions(scale_a, scale_b);
                if (stage("s1", a.t_s1, record->t_s1, [&] {
                    return s1::s1_preprocess(dev_img_a_, dev_img_b_, knobs.S, &dev_gray_a_, &dev_gray_b_,
                                             &scale_a, &scale_b, stream_);
                }) != Status::OK) return;
            }

            Keypoints kp_view_a = kp_a_;
            Keypoints kp_view_b = kp_b_;
            kp_view_a.capacity = kp_view_b.capacity = knobs.K;
            if (rerun_features) {
                if (stage("s2", a.t_s2, record->t_s2, [&] {
                    CUDA_CHECK(cudaMemsetAsync(d_kp_count_a_, 0, sizeof(int), stream_));
                    CUDA_CHECK(cudaMemsetAsync(d_kp_count_b_, 0, sizeof(int), stream_));
                    const s2::Config cfg{20, knobs.K, 0.04f};
                    const auto sa = s2::s2_detect(dev_gray_a_, cfg, kp_view_a, ws_s2_a_, ws_bytes_s2_a_, stream_);
                    return sa == Status::OK
                        ? s2::s2_detect(dev_gray_b_, cfg, kp_view_b, ws_s2_b_, ws_bytes_s2_b_, stream_) : sa;
                }) != Status::OK) return;
                if (stage("s3", a.t_s3, record->t_s3, [&] {
                    const auto sa = s3::s3_describe(dev_gray_a_, kp_view_a, desc_a_, ws_s3_, ws_bytes_s3_, stream_);
                    return sa == Status::OK
                        ? s3::s3_describe(dev_gray_b_, kp_view_b, desc_b_, ws_s3_, ws_bytes_s3_, stream_) : sa;
                }) != Status::OK) return;
                if (stage("s4", a.t_s4, record->t_s4, [&] {
                    CUDA_CHECK(cudaMemsetAsync(d_matches_count_, 0, sizeof(int), stream_));
                    const s4::Config cfg{0.8, false};
                    return s4::s4_match(kp_view_a, desc_a_, kp_view_b, desc_b_, cfg, matches_, ws_s4_, ws_bytes_s4_, stream_);
                }) != Status::OK) return;
            }

            *pinned_h_res_ = HomographyResult{};
            pinned_h_res_->status = Status::INVALID_ARGUMENT;
            const auto s5_status = stage("s5", a.t_s5, record->t_s5, [&] {
                if (a.previous_R > 0) record->cache_hit = 1;
                const s5::Config cfg{knobs.R, a.previous_R, 64, 1, 3.0, seed, generation};
                return s5::estimate(kp_view_a, kp_view_b, matches_, scale_a, scale_b, cfg,
                                    d_inlier_mask_, pinned_h_res_, ws_s5_, ws_bytes_s5_, stream_);
            });
            a.has_feedback = s5_status == Status::OK;
            if (a.has_feedback) a.homography = *pinned_h_res_;
            a.has_homography = a.has_feedback && a.homography.status == Status::OK;
            if (s5_status == Status::OK) completed_r = knobs.R;
            record->fail_stage = "controller";
            Status observed;
            {
                Span cpu{record->t_controller};
                observed = controller.observe({knobs, s5_status, *pinned_h_res_}, &a.decision);
            }
            if (observed != Status::OK) {
                if (s5_status == Status::OK) {
                    a.fail_stage = "controller";
                    a.execution_status = observed;
                    fail(observed);
                } else {
                    record->fail_stage = "s5";
                }
                return;
            }
            a.has_decision = true;
            if (a.decision.action == controller::Action::RETRY) {
                previous = knobs;
                knobs = a.decision.next;
                continue;
            }
            if (a.decision.action == controller::Action::REJECT) {
                if (s5_status != Status::OK) {
                    record->fail_stage = "s5";
                    return;
                }
                record->status = a.decision.reason == controller::Reason::BUDGET_EXHAUSTED
                    ? "BUDGET_EXHAUSTED" : a.decision.reason == controller::Reason::EXECUTION_ERROR
                    ? "EXECUTION_ERROR" : "REJECTED";
                return;
            }
            attempt.finish();
            break;
        }

        const auto& h = active->homography;
        active = nullptr;
        record->fail_stage = "s6";
        s6::Geometry geom{};
        const auto geometry = s6::prepare_geometry(cur_w_a_, cur_h_a_, cur_w_b_, cur_h_b_, h.H, {}, &geom);
        if (geometry != Status::OK) {
            record->status = geometry == Status::INSUFFICIENT_WORKSPACE ? "ERR_CANVAS_OVERFLOW" : status_to_str(geometry);
            return;
        }
        if (geom.width > canvas_capacity_w_ || geom.height > canvas_capacity_h_) {
            record->status = "ERR_CANVAS_OVERFLOW";
            return;
        }
        dev_canvas_.width = dev_s6_mask_.width = geom.width;
        dev_canvas_.height = dev_s6_mask_.height = geom.height;
        const auto rendered = timed_gpu(record->t_s6, [&] {
            return s6::render(dev_img_a_, dev_img_b_, geom, knobs.B, dev_canvas_, dev_s6_mask_,
                              ws_s6_, ws_bytes_s6_, stream_, dev_stats_);
        });
        host.finish();
        if (rendered != Status::OK) { fail(rendered); return; }

        record->fail_stage = "telemetry";
        CUDA_CHECK(cudaMemcpy(&record->num_kp_a, kp_a_.d_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&record->num_kp_b, kp_b_.d_count, sizeof(int), cudaMemcpyDeviceToHost));
        record->num_matches = h.num_matches;
        record->num_inliers = h.num_inliers;
        record->inlier_ratio = h.inlier_ratio;
        record->inlier_coverage = h.inlier_coverage;
        record->reproj_residual = h.reproj_residual;
        std::memcpy(record->H, h.H, sizeof(record->H));
        s6::RenderStatistics stats{};
        CUDA_CHECK(cudaMemcpy(&stats, dev_stats_, sizeof(stats), cudaMemcpyDeviceToHost));
        record->canvas_pixels = static_cast<std::uint64_t>(geom.width) * geom.height;
        record->overlap_pixels = stats.overlap_pixels;
        record->num_levels = s6::effective_reductions(geom, knobs.B);
        record->peak_workspace_bytes = ws_bytes_s6_;
        record->status = "OK";
        record->fail_stage.clear();
    } catch (const CudaError& error) {
        record->status = error.code == cudaErrorMemoryAllocation ? "ERR_DEVICE_OOM" : "ERR_CUDA";
        if (active) active->execution_status = Status::CUDA_ERROR;
        throw;
    }
}

}  // namespace stitch::harness
