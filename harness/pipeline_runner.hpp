// Pipeline runner managing preallocated device memory, CUDA stream and events,
// and zero-allocation execution of stages S1 through S6.
#pragma once

#include "common/cuda_fwd.hpp"
#include <string>
#include <memory>
#include <chrono>

#include "common/types.hpp"
#include "common/geometry.hpp"
#include "s1/s1.hpp"
#include "s2_s3/s2.hpp"
#include "s2_s3/s3.hpp"
#include "s4/s4.hpp"
#include "s5/s5.hpp"
#include "s6/s6.hpp"
#include "harness/harness_types.hpp"
#include "harness/image_loader.hpp"

namespace stitch::harness {

class PipelineRunner {
public:
    PipelineRunner();
    ~PipelineRunner();

    // Disable copy
    PipelineRunner(const PipelineRunner&) = delete;
    PipelineRunner& operator=(const PipelineRunner&) = delete;

    // Allocate all device buffers and workspaces once outside the measurement window
    void initialize(int max_width, int max_height, int max_k = 4000, int max_r = 2048);

    // Upload image pair A and B to device outside the measurement window
    void upload_pair(const HostImageRGB& img_a, const HostImageRGB& img_b);

    // Execute one run (warmup or measurement) without any cudaMalloc in the measurement window
    void execute_run(const HarnessConfig& config,
                     const std::string& run_id,
                     const std::string& git_rev,
                     const std::string& dataset,
                     const std::string& split,
                     const std::string& pair_id,
                     int repeat_idx,
                     int is_warmup,
                     RunRecord* record);

private:
    cudaStream_t stream_{nullptr};

    // CUDA timing events
    cudaEvent_t ev_start_{nullptr};
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

    bool initialized_{false};
};

}  // namespace stitch::harness
