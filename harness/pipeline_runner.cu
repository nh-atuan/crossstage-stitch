// Pipeline runner implementation.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "harness/pipeline_runner.hpp"
#include "controller/spatial.hpp"

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

std::string blend_to_str(s6::BlendMode mode) {
    switch (mode) {
        case s6::BlendMode::Multiband3: return "multiband3";
        case s6::BlendMode::Multiband5: return "multiband5";
        default: return "feather";
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
    if (d_features_) cudaFree(d_features_);
    if (pinned_features_) cudaFreeHost(pinned_features_);
}

void PipelineRunner::initialize(int max_w, int max_h, int max_k, int max_r) {
    if (initialized_) return;
    allocated_width_ = max_w;
    allocated_height_ = max_h;

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

    // S1 probe gradient reduction workspace and pinned host outputs
    CUDA_CHECK(cudaMalloc(&d_ws_probe_, 32));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_grad_a_), sizeof(double)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_grad_b_), sizeof(double)));
    constexpr auto feature_count=std::max(controller::scene_feature_count,controller::gate_feature_count);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_features_), controller::feature_device_count*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_features_), feature_count*sizeof(double)));

    initialized_ = true;
}

void PipelineRunner::upload_pair(const HostImageRGB& img_a, const HostImageRGB& img_b) {
    if (!initialized_ || img_a.width > allocated_width_ || img_b.width > allocated_width_ ||
        img_a.height > allocated_height_ || img_b.height > allocated_height_)
        throw std::runtime_error("input exceeds allocated pipeline dimensions");
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
    // Device-ready is the start boundary of every timed run, including warmups.
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (collect_features_) {
        const auto probe_start=std::chrono::steady_clock::now();
        ScaleInfo a{},b{};
        make_scale_info(cur_w_a_,cur_h_a_,.25,&a);make_scale_info(cur_w_b_,cur_h_b_,.25,&b);
        dev_gray_a_.width=a.work_w;dev_gray_a_.height=a.work_h;
        dev_gray_b_.width=b.work_w;dev_gray_b_.height=b.work_h;
        if (s1::s1_preprocess(dev_img_a_,dev_img_b_,.25,&dev_gray_a_,&dev_gray_b_,&a,&b,stream_)!=Status::OK ||
            controller::extract_scene(dev_gray_a_,dev_gray_b_,d_features_,pinned_features_,stream_,collect_pair_ncc_)!=Status::OK)
            throw std::runtime_error("Scene feature collection failed");
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        std::copy_n(pinned_features_,controller::scene_feature_count,collected_scene_.begin());
        collected_probe_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-probe_start).count();
    }
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
    if (collect_features_) {record->scene_features=collected_scene_;record->t_probe_features=collected_probe_ms_;}

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

    if (collect_features_) {
        const auto begin=std::chrono::steady_clock::now();
        const s4::Top2* forward=nullptr;
        if(s4::s4_forward_neighbors(kp_view_a.capacity,kp_view_b.capacity,s4_cfg,ws_s4_,ws_bytes_s4_,&forward)!=Status::OK)
            throw std::runtime_error("Missing matching feature observations");
        const auto status=controller::extract_gate(dev_img_a_,dev_img_b_,kp_view_a,kp_view_b,matches_,d_inlier_mask_,
            *pinned_h_res_,scale_a,scale_b,d_features_,pinned_features_,stream_,forward,collect_mask_);
        if (status==Status::CUDA_ERROR) throw std::runtime_error("Gate feature CUDA error");
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (status==Status::OK) {
            std::copy_n(pinned_features_,controller::gate_feature_count,record->gate_features.begin());
            record->features_present=true;
            if(collect_spatial_) {
                const auto spatial_start=std::chrono::steady_clock::now();
                const auto status=controller::extract_spatial(dev_img_a_,dev_img_b_,*pinned_h_res_,d_features_,pinned_features_,stream_,collect_spatial_groups_);
                if(status==Status::CUDA_ERROR)throw std::runtime_error("Spatial profile CUDA error");
                CUDA_CHECK(cudaStreamSynchronize(stream_));
                if(status==Status::OK) {
                    std::copy_n(pinned_features_,16,record->spatial.begin());
                    record->spatial_present=std::all_of(record->spatial.begin(),record->spatial.end(),[](double x){return std::isfinite(x);});
                }
                record->t_spatial=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-spatial_start).count();
            }
        }
        record->t_features=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
    // Stage 6: Prepare geometry and render canvas
    s6::Geometry geom{};
    s6::GeometryPolicy geom_policy{};
    Status g_stat = s6::prepare_geometry(cur_w_a_, cur_h_a_, cur_w_b_, cur_h_b_, pinned_h_res_->H, geom_policy, &geom);
    if (g_stat == Status::OK && (geom.width > kMaxCanvasDim || geom.height > kMaxCanvasDim))
        g_stat = Status::INSUFFICIENT_WORKSPACE;
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

    record->run_id = run_id;
    record->git_rev = git_rev;
    record->dataset = dataset;
    record->split = split;
    record->pair_id = pair_id;
    record->repeat_idx = repeat_idx;
    record->is_warmup = is_warmup;
    record->seed = seed;
    record->cache_hit = 0;
    trace_.count = 0;
    *pinned_h_res_ = HomographyResult{};

    const auto t_host_start = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // Step 1: Probe Phase
    // -----------------------------------------------------------------------
    controller.reset_pair();
    const double probe_s = controller.probe_scale();
    if (!(probe_s > 0.0 && probe_s <= 1.0)) {
        record->status = "INVALID_PROBE_SCALE";
        record->fail_stage = "controller";
        return;
    }

    ScaleInfo probe_scale_a{}, probe_scale_b{};
    make_scale_info(cur_w_a_, cur_h_a_, probe_s, &probe_scale_a);
    make_scale_info(cur_w_b_, cur_h_b_, probe_s, &probe_scale_b);

    dev_gray_a_.width = probe_scale_a.work_w;
    dev_gray_a_.height = probe_scale_a.work_h;
    dev_gray_b_.width = probe_scale_b.work_w;
    dev_gray_b_.height = probe_scale_b.work_h;

    const auto t_ctrl_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(ev_start_, stream_));

    Status probe_s1 = s1::s1_preprocess(dev_img_a_, dev_img_b_, probe_s,
                                        &dev_gray_a_, &dev_gray_b_,
                                        &probe_scale_a, &probe_scale_b, stream_);
    if (probe_s1 != Status::OK) {
        record->status = status_to_str(probe_s1);
        record->fail_stage = "s1";
        return;
    }

    const bool learned=controller.uses_learned_features();
    Status probe_grad = learned ? controller::extract_scene(dev_gray_a_,dev_gray_b_,d_features_,pinned_features_,stream_,controller.uses_pair_relations()) :
        s1::s1_compute_gradient(dev_gray_a_, dev_gray_b_, d_ws_probe_,pinned_grad_a_, pinned_grad_b_, stream_);
    if (probe_grad != Status::OK) {
        record->status = status_to_str(probe_grad);
        record->fail_stage = "s1";
        return;
    }

    CUDA_CHECK(cudaEventRecord(ev_s1_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float probe_s1_ms = 0.0f;
    cudaEventElapsedTime(&probe_s1_ms, ev_start_, ev_s1_);

    if (learned) {
        *pinned_grad_a_=pinned_features_[0];*pinned_grad_b_=pinned_features_[1];
        std::copy_n(pinned_features_,controller::scene_feature_count,record->scene_features.begin());
    }
    controller::SceneSignals scene{
        {probe_scale_a, *pinned_grad_a_},
        {probe_scale_b, *pinned_grad_b_},record->scene_features,learned
    };
    record->probe_gradient_a = *pinned_grad_a_;
    record->probe_gradient_b = *pinned_grad_b_;

    controller::Knobs current_knobs{};
    record->t_probe_features=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_ctrl_start).count();
    const auto selector_start=std::chrono::steady_clock::now();
    Status begin_stat = controller.begin_pair(scene, &current_knobs);
    record->t_selector=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-selector_start).count();
    const auto t_ctrl_end = std::chrono::steady_clock::now();
    double ctrl_time_ms = std::chrono::duration<double, std::milli>(t_ctrl_end - t_ctrl_start).count();

    if (begin_stat != Status::OK) {
        record->status = status_to_str(begin_stat);
        record->fail_stage = "controller";
        record->t_controller = ctrl_time_ms;
        return;
    }

    // -----------------------------------------------------------------------
    // Step 2: Escalation Loop (Attempts S1–S5)
    // -----------------------------------------------------------------------
    controller::Knobs executed_knobs = current_knobs;
    controller::Decision decision{};
    bool accepted = false;
    std::uint64_t current_gen = 1;
    int completed_r = 0;
    double prev_s = probe_s; // Gray buffer currently holds probe_s
    int prev_k = -1;
    int attempts = 0;
    double total_s1_ms = probe_s1_ms;
    double total_s2_ms = 0.0, total_s3_ms = 0.0, total_s4_ms = 0.0, total_s5_ms = 0.0;

    while (true) {
        if (trace_.count == static_cast<int>(trace_.records.size())) {
            throw std::runtime_error("adaptive attempt trace capacity exceeded");
        }
        const auto attempt_start = std::chrono::steady_clock::now();
        const double s4_before = total_s4_ms, s5_before = total_s5_ms;
        int eligible_r_old = 0;
        attempts++;
        bool rerun_s1 = (current_knobs.S != prev_s);
        bool rerun_s2 = rerun_s1 || (current_knobs.K != prev_k);
        bool rerun_s4 = rerun_s2;
        int prev_r = 0;

        if (attempts > 1) {
            controller::RetryPlan plan{};
            Status plan_s = controller::plan_retry(executed_knobs, current_knobs, &plan);
            if (plan_s != Status::OK) {
                record->status = "INVALID_RETRY_PLAN";
                record->fail_stage = "controller";
                break;
            }
            if (plan.restart_from == controller::RestartFrom::S1) {
                rerun_s1 = true; rerun_s2 = true; rerun_s4 = true;
            } else if (plan.restart_from == controller::RestartFrom::S2) {
                rerun_s2 = true; rerun_s4 = true;
            }
            if (plan.extend_s5) {
                eligible_r_old = completed_r;
                if (reuse_enabled_) {
                    prev_r = completed_r;
                    record->cache_hit = 1;
                } else {
                    rerun_s4 = true;
                }
            }
            if (plan.change_generation) {
                current_gen++;
                completed_r = 0;
            }
        }

        ScaleInfo scale_a{}, scale_b{};
        make_scale_info(cur_w_a_, cur_h_a_, current_knobs.S, &scale_a);
        make_scale_info(cur_w_b_, cur_h_b_, current_knobs.S, &scale_b);

        // Run S1 Preprocessing if scale changed from current buffer
        if (rerun_s1) {
            dev_gray_a_.width = scale_a.work_w;
            dev_gray_a_.height = scale_a.work_h;
            dev_gray_b_.width = scale_b.work_w;
            dev_gray_b_.height = scale_b.work_h;

            CUDA_CHECK(cudaEventRecord(ev_start_, stream_));
            Status s1_stat = s1::s1_preprocess(dev_img_a_, dev_img_b_, current_knobs.S,
                                               &dev_gray_a_, &dev_gray_b_,
                                               &scale_a, &scale_b, stream_);
            CUDA_CHECK(cudaEventRecord(ev_s1_, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            float ms1 = 0.0f; cudaEventElapsedTime(&ms1, ev_start_, ev_s1_); total_s1_ms += ms1;
            if (s1_stat != Status::OK) {
                record->status = status_to_str(s1_stat);
                record->fail_stage = "s1";
                break;
            }
        }

        Keypoints kp_view_a = kp_a_;
        Keypoints kp_view_b = kp_b_;
        if (current_knobs.K >= 0 && current_knobs.K <= kp_a_.capacity) {
            kp_view_a.capacity = current_knobs.K;
            kp_view_b.capacity = current_knobs.K;
        }

        // Run S2 & S3 if needed
        if (rerun_s2) {
            CUDA_CHECK(cudaMemsetAsync(d_kp_count_a_, 0, sizeof(int), stream_));
            CUDA_CHECK(cudaMemsetAsync(d_kp_count_b_, 0, sizeof(int), stream_));

            CUDA_CHECK(cudaEventRecord(ev_s1_, stream_));
            s2::Config s2_cfg{20, current_knobs.K, 0.04f};
            Status s2_a = s2::s2_detect(dev_gray_a_, s2_cfg, kp_view_a, ws_s2_a_, ws_bytes_s2_a_, stream_);
            Status s2_b = s2::s2_detect(dev_gray_b_, s2_cfg, kp_view_b, ws_s2_b_, ws_bytes_s2_b_, stream_);
            CUDA_CHECK(cudaEventRecord(ev_s2_, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            float ms2 = 0.0f; cudaEventElapsedTime(&ms2, ev_s1_, ev_s2_); total_s2_ms += ms2;
            if (s2_a != Status::OK || s2_b != Status::OK) {
                record->status = status_to_str(s2_a != Status::OK ? s2_a : s2_b);
                record->fail_stage = "s2";
                break;
            }

            CUDA_CHECK(cudaEventRecord(ev_s2_, stream_));
            Status s3_a = s3::s3_describe(dev_gray_a_, kp_view_a, desc_a_, ws_s3_, ws_bytes_s3_, stream_);
            Status s3_b = s3::s3_describe(dev_gray_b_, kp_view_b, desc_b_, ws_s3_, ws_bytes_s3_, stream_);
            CUDA_CHECK(cudaEventRecord(ev_s3_, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            float ms3 = 0.0f; cudaEventElapsedTime(&ms3, ev_s2_, ev_s3_); total_s3_ms += ms3;
            if (s3_a != Status::OK || s3_b != Status::OK) {
                record->status = status_to_str(s3_a != Status::OK ? s3_a : s3_b);
                record->fail_stage = "s3";
                break;
            }
        }

        // Run S4 if needed
        if (rerun_s4) {
            CUDA_CHECK(cudaMemsetAsync(d_matches_count_, 0, sizeof(int), stream_));
            CUDA_CHECK(cudaEventRecord(ev_s3_, stream_));
            s4::Config s4_cfg{0.8, false};
            Status s4_stat = s4::s4_match(kp_view_a, desc_a_, kp_view_b, desc_b_, s4_cfg, matches_, ws_s4_, ws_bytes_s4_, stream_);
            CUDA_CHECK(cudaEventRecord(ev_s4_, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            float ms4 = 0.0f; cudaEventElapsedTime(&ms4, ev_s3_, ev_s4_); total_s4_ms += ms4;
            if (s4_stat != Status::OK) {
                record->status = status_to_str(s4_stat);
                record->fail_stage = "s4";
                break;
            }
        }

        // Run S5 MSAC Homography
        CUDA_CHECK(cudaEventRecord(ev_s4_, stream_));
        s5::Config s5_cfg{current_knobs.R, prev_r, 64, 1, 3.0, seed, current_gen};
        Status s5_stat = s5::estimate(kp_view_a, kp_view_b, matches_, scale_a, scale_b, s5_cfg,
                                      d_inlier_mask_, pinned_h_res_, ws_s5_, ws_bytes_s5_, stream_);
        CUDA_CHECK(cudaEventRecord(ev_s5_, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        float ms5 = 0.0f; cudaEventElapsedTime(&ms5, ev_s4_, ev_s5_); total_s5_ms += ms5;

        completed_r = current_knobs.R;
        prev_s = current_knobs.S;
        prev_k = current_knobs.K;

        // Feedback observation
        const auto t_obs_s = std::chrono::steady_clock::now();
        controller::Feedback feedback{current_knobs, s5_stat, *pinned_h_res_};
        double feature_ms=0.;
        if (controller.needs_gate_features() && s5_stat==Status::OK && pinned_h_res_->status==Status::OK) {
            const auto begin=std::chrono::steady_clock::now();
            const s4::Top2* forward=nullptr;
            if((controller.relation_mask()&1) && s4::s4_forward_neighbors(kp_view_a.capacity,kp_view_b.capacity,
                    s4::Config{.8,false},ws_s4_,ws_bytes_s4_,&forward)!=Status::OK)
                throw std::runtime_error("Missing adaptive matching observations");
            const auto status=controller::extract_gate(dev_img_a_,dev_img_b_,kp_view_a,kp_view_b,matches_,d_inlier_mask_,
                *pinned_h_res_,scale_a,scale_b,d_features_,pinned_features_,stream_,forward,controller.relation_mask());
            if (status==Status::CUDA_ERROR) throw std::runtime_error("Gate feature CUDA error");
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            if (status==Status::OK) {
                std::copy_n(pinned_features_,controller::gate_feature_count,feedback.features.begin());
                feedback.features_present=true;
            }
            feature_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            record->t_features+=feature_ms;
        }
        // A reference fallback intentionally skips the gate. Its final feature
        // row must not retain observations from the rejected cheap candidate.
        if(controller.uses_spatial() && feedback.features_present) {
            const auto spatial_start=std::chrono::steady_clock::now();
            const auto status=controller::extract_spatial(dev_img_a_,dev_img_b_,*pinned_h_res_,d_features_,pinned_features_,stream_,controller.spatial_groups());
            if(status==Status::CUDA_ERROR)throw std::runtime_error("Spatial CUDA error");
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            if(status==Status::OK) {
                std::copy_n(pinned_features_,16,feedback.spatial.begin());
                feedback.spatial_present=std::all_of(feedback.spatial.begin(),feedback.spatial.end(),[](double x){return std::isfinite(x);});
            }
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-spatial_start).count();
            feature_ms+=elapsed;record->t_features+=elapsed;record->t_spatial+=elapsed;
        }
        record->spatial=feedback.spatial;record->spatial_present=feedback.spatial_present;
        record->gate_features=feedback.features;record->features_present=feedback.features_present;
        const auto selected_telemetry=controller.v5_telemetry();
        Status obs_stat = controller.observe(feedback, &decision);
        const auto t_obs_e = std::chrono::steady_clock::now();
        ctrl_time_ms += std::chrono::duration<double, std::milli>(t_obs_e - t_obs_s).count();

        if (obs_stat != Status::OK) {
            record->status = status_to_str(obs_stat);
            record->fail_stage = "controller";
            break;
        }

        auto& trace = trace_.records[trace_.count++];
        trace.knobs = current_knobs;
        trace.decision = decision;
        trace.homography = *pinned_h_res_;
        trace.previous_R = prev_r;
        trace.eligible_R_old = eligible_r_old;
        trace.generation = current_gen;
        trace.features=feedback.features;trace.t_features=feature_ms;trace.spatial=feedback.spatial;
        trace.v5=selected_telemetry;
        const auto gate_telemetry=controller.v5_gate_telemetry();
        {
            trace.v5.fail_ssim=gate_telemetry.fail_ssim;trace.v5.fail_psnr=gate_telemetry.fail_psnr;
            trace.v5.fail_coverage=gate_telemetry.fail_coverage;trace.v5.risk_joint=gate_telemetry.risk_joint;trace.v5.gate_evaluated=gate_telemetry.gate_evaluated;trace.v5.failure_code=gate_telemetry.failure_code;
        }
        trace.t_s4 = total_s4_ms - s4_before;
        trace.t_s5 = total_s5_ms - s5_before;
        trace.t_attempt = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - attempt_start).count();

        if (decision.action == controller::Action::ACCEPT) {
            accepted = true;
            break;
        } else if (decision.action == controller::Action::RETRY) {
            executed_knobs = current_knobs;
            current_knobs = decision.next;
            continue;
        } else { // REJECT
            accepted = false;
            break;
        }
    }

    // Download telemetry from device
    CUDA_CHECK(cudaMemcpy(&record->num_kp_a, kp_a_.d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&record->num_kp_b, kp_b_.d_count, sizeof(int), cudaMemcpyDeviceToHost));
    record->num_matches = pinned_h_res_->num_matches;
    record->num_inliers = pinned_h_res_->num_inliers;
    record->inlier_ratio = pinned_h_res_->inlier_ratio;
    record->inlier_coverage = pinned_h_res_->inlier_coverage;
    record->reproj_residual = pinned_h_res_->reproj_residual;
    std::memcpy(record->H, pinned_h_res_->H, 9 * sizeof(double));

    record->S = current_knobs.S;
    record->K = current_knobs.K;
    record->R = current_knobs.R;
    record->blend_mode = blend_to_str(current_knobs.B);
    record->blend_mode_effective = record->blend_mode;

    std::stringstream cfg_ss;
    cfg_ss << "S" << std::fixed << std::setprecision(2) << current_knobs.S
           << "_K" << current_knobs.K << "_R" << current_knobs.R << "_B" << record->blend_mode;
    record->config_id = cfg_ss.str();

    record->t_s1 = total_s1_ms;
    record->t_s2 = total_s2_ms;
    record->t_s3 = total_s3_ms;
    record->t_s4 = total_s4_ms;
    record->t_s5 = total_s5_ms;
    record->t_controller = ctrl_time_ms;

    // -----------------------------------------------------------------------
    // Step 3: S6 Rendering on ACCEPT
    // -----------------------------------------------------------------------
    if (accepted) {
        s6::Geometry geom{};
        s6::GeometryPolicy geom_policy{};
        Status g_stat = s6::prepare_geometry(cur_w_a_, cur_h_a_, cur_w_b_, cur_h_b_, pinned_h_res_->H, geom_policy, &geom);
        if (g_stat == Status::OK && (geom.width > kMaxCanvasDim || geom.height > kMaxCanvasDim))
            g_stat = Status::INSUFFICIENT_WORKSPACE;
        if (g_stat != Status::OK) {
            const auto t_host_end = std::chrono::steady_clock::now();
            record->t_e2e = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();
            record->status = status_to_str(g_stat);
            record->fail_stage = "s6";
            record->t_s6 = 0.0;
            return;
        }

        dev_canvas_.width = geom.width;
        dev_canvas_.height = geom.height;
        dev_s6_mask_.width = geom.width;
        dev_s6_mask_.height = geom.height;

        CUDA_CHECK(cudaEventRecord(ev_s5_, stream_));
        Status s6_stat = s6::render(dev_img_a_, dev_img_b_, geom, current_knobs.B,
                                    dev_canvas_, dev_s6_mask_, ws_s6_, ws_bytes_s6_, stream_, dev_stats_);
        if (s6_stat != Status::OK) {
            const auto t_host_end = std::chrono::steady_clock::now();
            record->t_e2e = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();
            record->status = status_to_str(s6_stat);
            record->fail_stage = "s6";
            record->t_s6 = 0.0;
            return;
        }
        CUDA_CHECK(cudaEventRecord(ev_s6_, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        float ms6 = 0.0f;
        cudaEventElapsedTime(&ms6, ev_s5_, ev_s6_);
        record->t_s6 = ms6;

        s6::RenderStatistics host_stats{};
        CUDA_CHECK(cudaMemcpy(&host_stats, dev_stats_, sizeof(host_stats), cudaMemcpyDeviceToHost));
        record->canvas_pixels = static_cast<std::uint64_t>(geom.width) * geom.height;
        record->overlap_pixels = host_stats.overlap_pixels;
        record->num_levels = s6::effective_reductions(geom, current_knobs.B);
        record->peak_workspace_bytes = ws_bytes_s6_;

        record->status = "OK";
        record->fail_stage = "";
    } else {
        record->status = (decision.reason == controller::Reason::BUDGET_EXHAUSTED) ? "BUDGET_EXHAUSTED" :
                         (decision.reason == controller::Reason::EXECUTION_ERROR) ? "EXECUTION_ERROR" : "REJECTED";
        record->fail_stage = "controller";
        record->t_s6 = 0.0;
    }

    const auto t_host_end = std::chrono::steady_clock::now();
    record->t_e2e = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();
}

}  // namespace stitch::harness
