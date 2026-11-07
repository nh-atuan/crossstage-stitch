// Bounded PPM input, RAII CUDA resources and atomic native artifact publication.
#pragma once
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include "common/admission_control.hpp"
#include "common/types.hpp"
#include "harness/cuda_error.hpp"
#include "harness/image_loader.hpp"

namespace stitch::stress {
inline void cuda(cudaError_t code) { if (code != cudaSuccess) throw harness::CudaError(code); }
inline std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') out << '\\' << static_cast<char>(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << static_cast<char>(c);
    }
    out << '"'; return out.str();
}
inline void publish(const std::filesystem::path& path, const void* bytes, std::size_t count) {
    const auto temporary = path.string() + ".partial";
    std::FILE* file = std::fopen(temporary.c_str(), "wbx");
    if (!file) throw std::runtime_error("exclusive artifact create failed");
    const auto written = std::fwrite(bytes, 1, count, file);
    const int closed = std::fclose(file);
    if (written != count || closed) throw std::runtime_error("artifact write failed");
    std::filesystem::create_hard_link(temporary, path);
    std::filesystem::remove(temporary);
}
inline void publish(const std::filesystem::path& path, const std::string& text) { publish(path, text.data(), text.size()); }

inline harness::HostImageRGB image(const std::string& path, int width, int height) {
    std::ifstream in(path, std::ios::binary);
    std::string magic, dimensions, depth;
    std::getline(in, magic); std::getline(in, dimensions); std::getline(in, depth);
    if (!in || magic != "P6" || dimensions != std::to_string(width)+" "+std::to_string(height) || depth != "255")
        throw admission::Error("ERR_INPUT");
    harness::HostImageRGB result{width, height, {}};
    const auto bytes = admission::multiply(admission::multiply(width, height), 3);
    const auto offset = in.tellg();
    if (std::filesystem::file_size(path) != static_cast<std::uintmax_t>(offset) + bytes) throw admission::Error("ERR_INPUT");
    result.pixels.resize(bytes);
    in.read(reinterpret_cast<char*>(result.pixels.data()), bytes);
    if (!in) throw admission::Error("ERR_INPUT");
    return result;
}
struct Buffer {
    void* data = nullptr;
    explicit Buffer(std::size_t bytes) { if (bytes) cuda(cudaMalloc(&data, bytes)); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { cuda(cudaEventCreate(&value)); }
    ~Event() { if (value) cudaEventDestroy(value); }
};
struct DeviceImage {
    Buffer memory;
    ImageView view{};
    int channels;
    DeviceImage(int width, int height, int c) : memory(admission::image_bytes(width, height, c)), channels(c) {
        view.data = static_cast<std::uint8_t*>(memory.data); view.width = width; view.height = height;
        view.pitch_bytes = width*c + 64; view.format = c == 3 ? PixelFormat::RGB8 : PixelFormat::GRAY8;
        cuda(cudaMemset(memory.data, 0xa5, admission::image_bytes(width, height, c)));
    }
    void upload(const harness::HostImageRGB& value) {
        cuda(cudaMemcpy2D(view.data, view.pitch_bytes, value.pixels.data(), value.width*3,
                          value.width*3, value.height, cudaMemcpyHostToDevice));
    }
    void padding() const {
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(64)*view.height);
        cuda(cudaMemcpy2D(bytes.data(), 64, view.data+static_cast<std::size_t>(view.width)*channels,
                          view.pitch_bytes, 64, view.height, cudaMemcpyDeviceToHost));
        for (auto byte : bytes) if (byte != 0xa5) throw admission::Error("ERR_CORRECTNESS");
    }
    void save(const std::filesystem::path& path, const std::array<int, 4>& box) const {
        const int x=box[0], y=box[1], w=box[2], h=box[3];
        if (x<0 || y<0 || w<=0 || h<=0 || w>view.width-x || h>view.height-y) throw admission::Error("ERR_CANVAS_OVERFLOW");
        std::vector<std::uint8_t> result(admission::multiply(admission::multiply(w, h), channels));
        cuda(cudaMemcpy2D(result.data(), static_cast<std::size_t>(w)*channels,
                          view.data+static_cast<std::size_t>(y)*view.pitch_bytes+static_cast<std::size_t>(x)*channels,
                          view.pitch_bytes, static_cast<std::size_t>(w)*channels, h, cudaMemcpyDeviceToHost));
        publish(path, result.data(), result.size());
    }
};
}  // namespace stitch::stress
