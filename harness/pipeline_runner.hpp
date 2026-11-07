// Pipeline runner managing preallocated device memory, CUDA stream and events,
// and zero-allocation execution of stages S1 through S6.
#pragma once

#include "common/cuda_fwd.hpp"
#include <string>
#include <memory>
#include <chrono>

#include "common/types.hpp"
#include "common/geometry.hpp"
#include "controller/controller.hpp"
#include "controller/reuse.hpp"
#include "s1/s1.hpp"
#include "s2_s3/s2.hpp"
#include "s2_s3/s3.hpp"
#include "s4/s4.hpp"
#include "s5/s5.hpp"
#include "s6/s6.hpp"
#include "harness/harness_types.hpp"
#include "harness/attempt_trace.hpp"
#include "harness/image_loader.hpp"

namespace stitch::harness {

struct PipelineMemoryPlan {
    std::size_t device_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::size_t s6_workspace_bytes = 0;
};

class PipelineRunner {
public:
    PipelineRunner();
    ~PipelineRunner();

    // Disable copy
    PipelineRunner(const PipelineRunner&) = delete;
    PipelineRunner& operator=(const PipelineRunner&) = delete;

    // Allocate all device buffers and workspaces once outside the measurement window
    void initialize(int max_width, int max_height, int max_k = 4000, int max_r = 2048,
                    int canvas_width = 4096, int canvas_height = 4096,
                    s6::BlendMode workspace_mode = s6::BlendMode::Multiband5,
                    std::size_t device_reserve = 256u * 1024u * 1024u);
    static PipelineMemoryPlan query_memory(int width, int height, int k, int r,
                                          int canvas_width, int canvas_height, s6::BlendMode mode);
    double last_fixed_gpu_span_ms() const { return last_fixed_gpu_span_ms_; }
    void synchronize();

    // Upload outside the measurement window; return only when device RGB is ready.
    void upload_pair(const HostImageRGB& img_a, const HostImageRGB& img_b);
    void set_reuse_enabled(bool enabled) { reuse_enabled_ = enabled; }
    void set_probe_enabled(bool enabled) { probe_enabled_ = enabled; }
    bool probe_enabled() const { return probe_enabled_; }
    void set_collect_features(bool enabled) { collect_features_ = enabled; }

    // Execute one fixed/oracle run without any cudaMalloc in the measurement window
    void execute_run(const HarnessConfig& config,
                     const std::string& run_id,
                     const std::string& git_rev,
                     const std::string& dataset,
                     const std::string& split,
                     const std::string& pair_id,
                     int repeat_idx,
                     int is_warmup,
                     RunRecord* record);

    // Execute one adaptive run governed by the Controller, including S1 probe,
    // initial selection, bounded escalation loop with S5 raw state reuse, and S6 rendering.
    void execute_adaptive_run(controller::Controller& controller,
                              const std::string& run_id,
                              const std::string& git_rev,
                              const std::string& dataset,
                              const std::string& split,
                              const std::string& pair_id,
                              int repeat_idx,
                              int is_warmup,
                              std::uint64_t seed,
                              RunRecord* record);
    const AttemptTrace& attempt_trace() const noexcept { return attempt_trace_; }

private:
    bool reuse_enabled_{true};
    bool probe_enabled_{true};
    bool collect_features_{};
    controller::SceneFeatures collected_scene_{};
    double* d_features_{};
    double* pinned_features_{};
    cudaStream_t stream_{nullptr};

    // CUDA timing events
    cudaEvent_t ev_start_{nullptr};
    cudaEvent_t ev_probe_prep_{nullptr};
    cudaEvent_t ev_s1_{nullptr};
    cudaEvent_t ev_s2_{nullptr};
    cudaEvent_t ev_s3_{nullptr};
    cudaEvent_t ev_s4_{nullptr};
    cudaEvent_t ev_s5_{nullptr};
    cudaEvent_t ev_s6_{nullptr};

    // Current pair dimensions
    int cur_w_a_{0}, cur_h_a_{0};
    int cur_w_b_{0}, cur_h_b_{0};

    // Input images on device (RGB8)
    ImageView dev_img_a_{};
    ImageView dev_img_b_{};

    // S1 preprocessed outputs on device (GRAY8)
    ImageView dev_gray_a_{};
    ImageView dev_gray_b_{};

    // S2 Keypoints SoA on device
    Keypoints kp_a_{};
    Keypoints kp_b_{};
    int* d_kp_count_a_{nullptr};
    int* d_kp_count_b_{nullptr};

    // S3 Descriptors on device
    Descriptors desc_a_{};
    Descriptors desc_b_{};

    // S4 Matches on device
    Matches matches_{};
    int* d_matches_count_{nullptr};

    // S5 Inlier mask and pinned host output
    std::uint8_t* d_inlier_mask_{nullptr};
    HomographyResult* pinned_h_res_{nullptr};

    // S6 Canvas output and union mask on device
    ImageView dev_canvas_{};
    ImageView dev_s6_mask_{};
    s6::RenderStatistics* dev_stats_{nullptr};

    // Preallocated static device workspaces
    void* ws_s2_a_{nullptr};
    std::size_t ws_bytes_s2_a_{0};
    void* ws_s2_b_{nullptr};
    std::size_t ws_bytes_s2_b_{0};

    void* ws_s3_{nullptr};
    std::size_t ws_bytes_s3_{0};

    void* ws_s4_{nullptr};
    std::size_t ws_bytes_s4_{0};

    void* ws_s5_{nullptr};
    std::size_t ws_bytes_s5_{0};

    void* ws_s6_{nullptr};
    std::size_t ws_bytes_s6_{0};

    // S1 probe gradient reduction workspace (32 bytes) and pinned host outputs
    void* d_ws_probe_{nullptr};
    double* pinned_grad_a_{nullptr};
    double* pinned_grad_b_{nullptr};

    bool initialized_{false};
    int input_capacity_w_ = 0, input_capacity_h_ = 0;
    int canvas_capacity_w_ = 0, canvas_capacity_h_ = 0;
    int hypothesis_capacity_ = 0;
    double last_fixed_gpu_span_ms_ = -1;
    AttemptTrace attempt_trace_{};
};

}  // namespace stitch::harness
