#pragma once

#include "common/types.hpp"

namespace stitch::s4 {

// No tuned defaults: callers must supply both fields. Zero-initialization is invalid.
struct Config {
    double ratio_threshold;  // finite, strictly between zero and one
    bool cross_check;        // mutual nearest neighbor, without a reverse ratio test
};

// Missing neighbors have index -1 and distance +infinity. Distances here are
// Hamming or squared L2 (double accumulation); matches report Euclidean L2.
// Ties choose the lower train index, including the second distinct neighbor.
struct Top2 {
    double first_distance;
    double second_distance;
    int first_index;
    int second_index;
};

// Host-only, no CUDA dependency. Returns zero bytes on error or capacity_a == 0.
// Workspace is O(capacity_a + cross_check * capacity_b), not a distance matrix.
Status s4_workspace_bytes(int capacity_a, int capacity_b, const Config& config,
                          std::size_t* bytes);

// Exact tiled search, useful independently of ratio filtering. Caller provides
// capacity_a Top2 entries; every entry is written, including inactive queries.
// All non-null pointers refer to device memory. d_count must be present even
// for zero capacity, with 0 <= *d_count <= capacity, produced earlier on stream.
// Keypoint coordinates are unused. Descriptor rows cover capacity, not count.
// dim/type must agree; strides cover a row; FLOAT32 base/stride are float-aligned.
// FLOAT32 elements must be finite. Buffers must not overlap writable outputs.
// Invalid device counts produce empty results (there is no host count readback).
Status s4_knn_top2(const Keypoints& a, const Descriptors& da,
                   const Keypoints& b, const Descriptors& db,
                   Top2* output, cudaStream_t stream);

// Caller supplies output capacity >= a.capacity and a non-null device d_count.
// Output arrays may be null only for zero a.capacity. No truncation occurs.
// Workspace must be aligned to alignof(Top2), with at least the queried bytes.
// Writable buffers (workspace, output arrays/count) must not overlap each other
// or inputs. Read-only inputs may alias. All buffers must remain alive until
// stream completion. Concurrent calls need disjoint writable buffers.
// Strict ratio: d1 < ratio*d2; squared L2 uses ratio^2. Fewer than two train
// neighbors and two zero distances never pass. Output is compacted in A order.
// L2 output may be +infinity if its Euclidean value exceeds float range.
// Empty inputs succeed and write zero count. Host argument errors enqueue
// nothing; device failures surface at caller synchronization. These functions
// allocate nothing, never synchronize, and report launch errors only.
Status s4_match(const Keypoints& a, const Descriptors& da,
                const Keypoints& b, const Descriptors& db, const Config& config,
                Matches output, void* workspace, std::size_t workspace_bytes,
                cudaStream_t stream);

// Read-only observations already computed by s4_match, valid until workspace
// reuse. The capacities/configuration must match that completed call.
Status s4_forward_neighbors(int capacity_a,int capacity_b,const Config& config,
    const void* workspace,std::size_t workspace_bytes,const Top2** output);

}  // namespace stitch::s4
