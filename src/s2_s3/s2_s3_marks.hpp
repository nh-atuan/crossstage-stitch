// Per-phase CUDA event marks for the A3 timing benchmark. Not part of the
// stage interface: s2_detect and s3_describe run the same kernels without
// marks. Each event is recorded on the stream right after its phase; a phase
// that launches nothing (K = 0, image without an interior, capacity 0) still
// records its event, so every mark is valid after an OK return.
#pragma once

#include <cstddef>

#include "common/cuda_fwd.hpp"
#include "s2_s3/s2.hpp"
#include "s2_s3/s3.hpp"

namespace stitch::s2_s3::detail {

struct S2Marks {
    cudaEvent_t after_fast;    // histogram reset and FAST scores
    cudaEvent_t after_select;  // NMS, 2K threshold, block scan and scatter
    cudaEvent_t after_harris;  // Harris responses; the rank kernel follows
};

struct S3Marks {
    cudaEvent_t after_blur;  // both Gaussian passes; the describe kernel follows
};

// Same contract and results as s2::s2_detect, plus the marks above.
Status s2_detect_marked(const ImageView& image, const s2::Config& config, Keypoints output,
                        void* workspace, std::size_t workspace_bytes, cudaStream_t stream,
                        const S2Marks& marks);

// Same contract and results as s3::s3_describe, plus the mark above.
Status s3_describe_marked(const ImageView& image, const Keypoints& keypoints,
                          Descriptors output, void* workspace, std::size_t workspace_bytes,
                          cudaStream_t stream, const S3Marks& marks);

}  // namespace stitch::s2_s3::detail
