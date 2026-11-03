// Checked byte arithmetic and capacity guards; no CUDA calls or allocations.
#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace stitch::admission {
struct Error : std::runtime_error {
    std::string code;
    explicit Error(const std::string& status) : std::runtime_error(status), code(status) {}
};
inline std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) throw Error("ERR_CANVAS_OVERFLOW");
    return a + b;
}
inline std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw Error("ERR_CANVAS_OVERFLOW");
    return a * b;
}
inline std::size_t image_bytes(int w, int h, int channels, std::size_t padding = 64) {
    if (w <= 0 || h <= 0 || channels <= 0) throw Error("ERR_CAPACITY");
    const auto pitch = add(multiply(static_cast<std::size_t>(w), channels), padding);
    if (pitch > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw Error("ERR_CANVAS_OVERFLOW");
    return multiply(pitch, static_cast<std::size_t>(h));
}
inline void fit(std::size_t requested, std::size_t available, std::size_t reserve, const char* code) {
    if (reserve > available || requested > available - reserve) throw Error(code);
}
inline void dimensions_fit(int wa, int ha, int wb, int hb, int capacity_w, int capacity_h) {
    if (wa <= 0 || ha <= 0 || wb <= 0 || hb <= 0 || wa > capacity_w || wb > capacity_w ||
        ha > capacity_h || hb > capacity_h) throw Error("ERR_CAPACITY");
}
}  // namespace stitch::admission
