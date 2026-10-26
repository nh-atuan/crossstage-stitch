// Image loading utilities for the benchmark harness.
// Supports OpenCV when available, with a built-in fallback PPM (P6) reader.
#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef STITCH_HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace stitch::harness {

struct HostImageRGB {
    int width{0};
    int height{0};
    std::vector<std::uint8_t> pixels; // interleaved R, G, B
};

inline HostImageRGB load_ppm(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open PPM file: " + path);
    }
    std::string magic;
    file >> magic;
    if (magic != "P6") {
        throw std::runtime_error("Unsupported PPM format (expected P6): " + path);
    }

    auto skip_comments = [&file]() {
        char ch;
        while (file >> std::ws && file.peek() == '#') {
            std::string line;
            std::getline(file, line);
        }
    };

    skip_comments();
    int width, height, max_val;
    file >> width;
    skip_comments();
    file >> height;
    skip_comments();
    file >> max_val;
    file.get(); // consume trailing newline

    if (width <= 0 || height <= 0 || max_val != 255) {
        throw std::runtime_error("Invalid PPM dimensions or max_val != 255: " + path);
    }

    HostImageRGB img;
    img.width = width;
    img.height = height;
    img.pixels.resize(static_cast<std::size_t>(width) * height * 3);
    file.read(reinterpret_cast<char*>(img.pixels.data()), img.pixels.size());
    if (!file) {
        throw std::runtime_error("Failed to read PPM pixel data: " + path);
    }
    return img;
}

inline HostImageRGB load_image(const std::string& path) {
    // Check if file is PPM
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".ppm") {
        return load_ppm(path);
    }

#ifdef STITCH_HAS_OPENCV
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("OpenCV failed to load image: " + path);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    HostImageRGB img;
    img.width = rgb.cols;
    img.height = rgb.rows;
    img.pixels.resize(static_cast<std::size_t>(img.width) * img.height * 3);
    std::memcpy(img.pixels.data(), rgb.data, img.pixels.size());
    return img;
#else
    // Fallback: try PPM
    try {
        return load_ppm(path);
    } catch (...) {
        throw std::runtime_error("Cannot load non-PPM image without OpenCV compiled in: " + path);
    }
#endif
}

}  // namespace stitch::harness
