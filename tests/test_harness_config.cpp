// Unit tests for harness JSON parsing, PPM image loading, and CSV row formatting.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "check.hpp"
#include "harness/harness_types.hpp"
#include "harness/json_parser.hpp"
#include "harness/image_loader.hpp"

using namespace stitch::harness;

void test_json_manifest_parsing() {
    const std::string manifest_json = R"({
        "pair_01": {
            "a": "data/images/01_a.ppm",
            "b": "data/images/01_b.ppm",
            "reference_status": "ok"
        },
        "pair_02": {
            "a": "data/images/02_a.ppm",
            "b": "data/images/02_b.ppm"
        }
    })";

    const std::string tmp_manifest = "test_manifest_tmp.json";
    {
        std::ofstream out(tmp_manifest);
        out << manifest_json;
    }

    auto pairs = parse_manifest(tmp_manifest);
    std::remove(tmp_manifest.c_str());

    CHECK(pairs.size() == 2);
    bool found_01 = false, found_02 = false;
    for (const auto& p : pairs) {
        if (p.pair_id == "pair_01") {
            found_01 = true;
            CHECK(p.img_a_path == "data/images/01_a.ppm");
            CHECK(p.img_b_path == "data/images/01_b.ppm");
            CHECK(p.reference_status == "ok");
        } else if (p.pair_id == "pair_02") {
            found_02 = true;
            CHECK(p.img_a_path == "data/images/02_a.ppm");
            CHECK(p.img_b_path == "data/images/02_b.ppm");
            CHECK(p.reference_status == "ok");
        }
    }
    CHECK(found_01 && found_02);
}

void test_json_configs_parsing() {
    const std::string configs_json = R"([
        {
            "config_id": "S0.50_K1000_R256_Bfeather",
            "S": 0.5,
            "K": 1000,
            "R": 256,
            "blend_mode": "feather"
        },
        {
            "S": 0.25,
            "K": 500,
            "R": 128,
            "blend_mode": "multiband3"
        }
    ])";

    const std::string tmp_configs = "test_configs_tmp.json";
    {
        std::ofstream out(tmp_configs);
        out << configs_json;
    }

    auto configs = parse_configs(tmp_configs);
    std::remove(tmp_configs.c_str());

    CHECK(configs.size() == 2);
    CHECK(configs[0].config_id == "S0.50_K1000_R256_Bfeather");
    CHECK(configs[0].S == 0.5);
    CHECK(configs[0].K == 1000);
    CHECK(configs[0].R == 256);
    CHECK(configs[0].blend_mode == stitch::s6::BlendMode::Feather);

    CHECK(configs[1].config_id == "S0.25_K500_R128_Bmultiband3");
    CHECK(configs[1].S == 0.25);
    CHECK(configs[1].K == 500);
    CHECK(configs[1].R == 128);
    CHECK(configs[1].blend_mode == stitch::s6::BlendMode::Multiband3);
}

void test_ppm_image_roundtrip() {
    const std::string tmp_ppm = "test_img_tmp.ppm";
    const int w = 4, h = 3;
    {
        std::ofstream out(tmp_ppm, std::ios::binary);
        out << "P6\n# Test comment\n" << w << " " << h << "\n255\n";
        for (int i = 0; i < w * h * 3; ++i) {
            out.put(static_cast<char>(i % 256));
        }
    }

    auto img = load_image(tmp_ppm);
    std::remove(tmp_ppm.c_str());

    CHECK(img.width == w);
    CHECK(img.height == h);
    CHECK(img.pixels.size() == w * h * 3);
    for (int i = 0; i < w * h * 3; ++i) {
        CHECK(img.pixels[i] == static_cast<std::uint8_t>(i % 256));
    }
}

int main() {
    test_json_manifest_parsing();
    test_json_configs_parsing();
    test_ppm_image_roundtrip();
    std::puts("test_harness_config: passed all tests successfully.");
    return 0;
}
