// Pipeline runner implementation.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "harness/pipeline_runner.hpp"

#define CUDA_CHECK(call) do {                                                  \
    const cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                                  \
        std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(err));                                 \
        throw std::runtime_error(cudaGetErrorString(err));                     \
    }                                                                          \
} while (false)

namespace stitch::harness {

namespace {

constexpr int kMaxCanvasDim = 4096;
constexpr int kMaxKeypoints = 4000;
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
}

void PipelineRunner::initialize(int max_w, int max_h, int max_k, int max_r) {
    if (initialized_) return;

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
    dev_canvas_.width = kMaxCanvasDim;
    dev_canvas_.height = kMaxCanvasDim;
    dev_canvas_.pitch_bytes = kMaxCanvasDim * 3 + kPadding;
    dev_canvas_.format = PixelFormat::RGB8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_canvas_.data),
                          static_cast<std::size_t>(dev_canvas_.pitch_bytes) * kMaxCanvasDim));

    dev_s6_mask_.width = kMaxCanvasDim;
    dev_s6_mask_.height = kMaxCanvasDim;
    dev_s6_mask_.pitch_bytes = kMaxCanvasDim + kPadding;
    dev_s6_mask_.format = PixelFormat::GRAY8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_s6_mask_.data),
                          static_cast<std::size_t>(dev_s6_mask_.pitch_bytes) * kMaxCanvasDim));

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

    // S6 maximum workspace (for multiband5 on max canvas)
    s6::Geometry max_geom{};
    max_geom.width = kMaxCanvasDim;
    max_geom.height = kMaxCanvasDim;
    max_geom.width_a = max_w;
    max_geom.height_a = max_h;
    max_geom.width_b = max_w;
    max_geom.height_b = max_h;
    max_geom.origin_x = 0;
    max_geom.origin_y = 0;
    max_geom.denominator_tolerance = 1e-6;
    max_geom.snap_tolerance = 0.01;
    max_geom.inverse[0] = 1.0;
    max_geom.inverse[4] = 1.0;
    max_geom.inverse[8] = 1.0;
    CUDA_CHECK(s6::s6_workspace_bytes(max_geom, s6::BlendMode::Multiband5, &ws_bytes_s6_) == Status::OK ? cudaSuccess : cudaErrorUnknown);
    if (ws_bytes_s6_) CUDA_CHECK(cudaMalloc(&ws_s6_, ws_bytes_s6_ + 256));

    initialized_ = true;
}

void PipelineRunner::upload_pair(const HostImageRGB& img_a, const HostImageRGB& img_b) {
    cur_w_a_ = img_a.width;
    cur_h_a_ = img_a.height;
    cur_w_b_ = img_b.width;
    cur_h_b_ = img_b.height;

    dev_img_a_.width = cur_w_a_;
    dev_img_a_.height = cur_h_a_;
    CUDA_CHECK(cudaMemcpy2D(dev_img_a_.data, dev_img_a_.pitch_bytes,
                            img_a.pixels.data(), cur_w_a_ * 3,
                            cur_w_a_ * 3, cur_h_a_, cudaMemcpyHostToDevice));

    dev_img_b_.width = cur_w_b_;
    dev_img_b_.height = cur_h_b_;
    CUDA_CHECK(cudaMemcpy2D(dev_img_b_.data, dev_img_b_.pitch_bytes,
                            img_b.pixels.data(), cur_w_b_ * 3,
                            cur_w_b_ * 3, cur_h_b_, cudaMemcpyHostToDevice));
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

    // Stage 2: FAST detection + Harris ranking (núm K)
    s2::Config s2_cfg{cfg.fast_threshold, cfg.K, cfg.harris_k};
    Status s2_stat_a = s2::s2_detect(dev_gray_a_, s2_cfg, kp_a_, ws_s2_a_, ws_bytes_s2_a_, stream_);
    Status s2_stat_b = s2::s2_detect(dev_gray_b_, s2_cfg, kp_b_, ws_s2_b_, ws_bytes_s2_b_, stream_);
    if (s2_stat_a != Status::OK || s2_stat_b != Status::OK) {
        record->status = status_to_str(s2_stat_a != Status::OK ? s2_stat_a : s2_stat_b);
        record->fail_stage = "s2";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s2_, stream_));

    // Stage 3: rBRIEF description
    Status s3_stat_a = s3::s3_describe(dev_gray_a_, kp_a_, desc_a_, ws_s3_, ws_bytes_s3_, stream_);
    Status s3_stat_b = s3::s3_describe(dev_gray_b_, kp_b_, desc_b_, ws_s3_, ws_bytes_s3_, stream_);
    if (s3_stat_a != Status::OK || s3_stat_b != Status::OK) {
        record->status = status_to_str(s3_stat_a != Status::OK ? s3_stat_a : s3_stat_b);
        record->fail_stage = "s3";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s3_, stream_));

    // Stage 4: Matching
    s4::Config s4_cfg{cfg.ratio_threshold, cfg.cross_check};
    Status s4_stat = s4::s4_match(kp_a_, desc_a_, kp_b_, desc_b_, s4_cfg, matches_, ws_s4_, ws_bytes_s4_, stream_);
    if (s4_stat != Status::OK) {
        record->status = status_to_str(s4_stat);
        record->fail_stage = "s4";
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_s4_, stream_));

    // Stage 5: MSAC Homography (núm R)
    s5::Config s5_cfg{cfg.R, 0, cfg.msac_batch_size, cfg.msac_refinements, cfg.msac_threshold, cfg.seed, 1};
    Status s5_stat = s5::estimate(kp_a_, kp_b_, matches_, scale_a, scale_b, s5_cfg,
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
        record->status = status_to_str(g_stat);
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

}  // namespace stitch::harness
