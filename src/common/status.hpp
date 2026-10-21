// Status codes returned by every pipeline stage. New codes must go through Tuan.
#pragma once

namespace stitch {

enum class Status : int {
    OK = 0,
    INVALID_ARGUMENT,
    INSUFFICIENT_WORKSPACE,
    CUDA_ERROR,
    NOT_ENOUGH_POINTS,
    DEGENERATE,
};

inline const char* status_name(Status s) {
    switch (s) {
        case Status::OK: return "OK";
        case Status::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case Status::INSUFFICIENT_WORKSPACE: return "INSUFFICIENT_WORKSPACE";
        case Status::CUDA_ERROR: return "CUDA_ERROR";
        case Status::NOT_ENOUGH_POINTS: return "NOT_ENOUGH_POINTS";
        case Status::DEGENERATE: return "DEGENERATE";
    }
    return "UNKNOWN";
}

}  // namespace stitch
