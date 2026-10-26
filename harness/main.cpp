// Benchmark Harness entry point (stitch_harness).
// Supports both batch manifest mode and single-pair debugging/profiling mode.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

#include "harness/harness_types.hpp"
#include "harness/json_parser.hpp"
#include "harness/image_loader.hpp"
#include "harness/pipeline_runner.hpp"

#ifndef STITCH_GIT_REV
#define STITCH_GIT_REV "HEAD"
#endif

namespace stitch::harness {

void write_csv_header(std::ostream& out) {
    out << "run_id,git_rev,dataset,split,pair_id,config_id,repeat_idx,is_warmup,seed,"
        << "S,K,R,blend_mode,"
        << "t_h2d,t_s1,t_s2,t_s3,t_s4,t_s5,t_s6,t_controller,t_d2h,t_e2e,"
        << "num_kp_a,num_kp_b,num_matches,num_inliers,inlier_ratio,inlier_coverage,reproj_residual,"
        << "h0,h1,h2,h3,h4,h5,h6,h7,h8,"
        << "canvas_pixels,overlap_pixels,blend_mode_effective,num_levels,peak_workspace_bytes,cache_hit,"
        << "status,fail_stage\n";
}

void write_csv_row(std::ostream& out, const RunRecord& r) {
    out << r.run_id << "," << r.git_rev << "," << r.dataset << "," << r.split << ","
        << r.pair_id << "," << r.config_id << "," << r.repeat_idx << "," << r.is_warmup << ","
        << r.seed << ","
        << std::fixed << std::setprecision(4) << r.S << ","
        << r.K << "," << r.R << "," << r.blend_mode << ","
        << std::setprecision(3)
        << r.t_h2d << "," << r.t_s1 << "," << r.t_s2 << "," << r.t_s3 << ","
        << r.t_s4 << "," << r.t_s5 << "," << r.t_s6 << "," << r.t_controller << ","
        << r.t_d2h << "," << r.t_e2e << ","
        << r.num_kp_a << "," << r.num_kp_b << "," << r.num_matches << ","
        << r.num_inliers << ","
        << std::setprecision(6)
        << r.inlier_ratio << "," << r.inlier_coverage << "," << r.reproj_residual << ",";

    for (int i = 0; i < 9; ++i) {
        out << std::setprecision(10) << r.H[i] << ",";
    }

    out << r.canvas_pixels << "," << r.overlap_pixels << ","
        << r.blend_mode_effective << "," << r.num_levels << ","
        << r.peak_workspace_bytes << "," << r.cache_hit << ","
        << r.status << "," << r.fail_stage << "\n";
}

std::string generate_run_id() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "run_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

}  // namespace stitch::harness

int main(int argc, char** argv) {
    using namespace stitch::harness;

    std::string manifest_path;
    std::string configs_path;
    std::string out_path = "runs.csv";
    std::string dataset = "udis_d";
    std::string split = "dev";
    std::string run_id = generate_run_id();

    // Single run options
    std::string img_a_path, img_b_path;
    std::string pair_id = "single_pair";
    double single_s = 0.5;
    int single_k = 1000;
    int single_r = 256;
    std::string single_blend = "feather";

    int num_warmup = 2;
    int num_repeats = 10;
    std::uint64_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (arg == "--configs" && i + 1 < argc) configs_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else if (arg == "--split" && i + 1 < argc) split = argv[++i];
        else if (arg == "--run-id" && i + 1 < argc) run_id = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) num_warmup = std::stoi(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc) num_repeats = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (arg == "--img-a" && i + 1 < argc) img_a_path = argv[++i];
        else if (arg == "--img-b" && i + 1 < argc) img_b_path = argv[++i];
        else if (arg == "--pair-id" && i + 1 < argc) pair_id = argv[++i];
        else if (arg == "-S" && i + 1 < argc) single_s = std::stod(argv[++i]);
        else if (arg == "-K" && i + 1 < argc) single_k = std::stoi(argv[++i]);
        else if (arg == "-R" && i + 1 < argc) single_r = std::stoi(argv[++i]);
        else if (arg == "--blend" && i + 1 < argc) single_blend = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: stitch_harness [OPTIONS]\n"
                      << "Batch mode:\n"
                      << "  --manifest <path>   Pairs JSON manifest\n"
                      << "  --configs <path>    Configs JSON file\n"
                      << "  --out <path>        Output CSV file (default: runs.csv)\n"
                      << "Single mode:\n"
                      << "  --img-a <path>      Image A path\n"
                      << "  --img-b <path>      Image B path\n"
                      << "  -S <double>         Scale S in (0, 1] (default: 0.5)\n"
                      << "  -K <int>            Keypoints K (default: 1000)\n"
                      << "  -R <int>            MSAC Hypotheses R (default: 256)\n"
                      << "  --blend <mode>      Blend mode: feather|multiband3|multiband5\n";
            return 0;
        }
    }

    std::vector<PairEntry> pairs;
    std::vector<HarnessConfig> configs;

    if (!manifest_path.empty()) {
        try {
            pairs = parse_manifest(manifest_path);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing manifest: " << e.what() << "\n";
            return 1;
        }
    } else if (!img_a_path.empty() && !img_b_path.empty()) {
        PairEntry pe;
        pe.pair_id = pair_id;
        pe.img_a_path = img_a_path;
        pe.img_b_path = img_b_path;
        pairs.push_back(pe);
    } else {
        std::cerr << "Error: must specify either --manifest or both --img-a and --img-b.\n";
        return 1;
    }

    if (!configs_path.empty()) {
        try {
            configs = parse_configs(configs_path);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing configs: " << e.what() << "\n";
            return 1;
        }
    } else {
        HarnessConfig cfg;
        cfg.S = single_s;
        cfg.K = single_k;
        cfg.R = single_r;
        cfg.blend_mode_str = single_blend;
        if (single_blend == "multiband3") cfg.blend_mode = stitch::s6::BlendMode::Multiband3;
        else if (single_blend == "multiband5") cfg.blend_mode = stitch::s6::BlendMode::Multiband5;
        else cfg.blend_mode = stitch::s6::BlendMode::Feather;

        std::stringstream ss;
        ss << "S" << std::fixed << std::setprecision(2) << cfg.S
           << "_K" << cfg.K << "_R" << cfg.R << "_B" << cfg.blend_mode_str;
        cfg.config_id = ss.str();
        cfg.seed = seed;
        configs.push_back(cfg);
    }

    // Initialize pipeline runner with maximum bounds
    PipelineRunner runner;
    try {
        runner.initialize(2048, 2048, 4000, 2048);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize pipeline runner: " << e.what() << "\n";
        return 1;
    }

    std::ofstream csv_out(out_path);
    if (!csv_out.is_open()) {
        std::cerr << "Error opening output CSV: " << out_path << "\n";
        return 1;
    }
    write_csv_header(csv_out);

    std::cout << "Starting benchmark harness:\n"
              << "  Pairs: " << pairs.size() << "\n"
              << "  Configs: " << configs.size() << "\n"
              << "  Warmup: " << num_warmup << ", Repeats: " << num_repeats << "\n"
              << "  Output: " << out_path << "\n";

    std::filesystem::path manifest_dir = manifest_path.empty() ? "" : std::filesystem::path(manifest_path).parent_path();

    int total_runs = 0;
    for (const auto& pair : pairs) {
        std::string path_a = (manifest_dir / pair.img_a_path).string();
        std::string path_b = (manifest_dir / pair.img_b_path).string();

        HostImageRGB img_a, img_b;
        try {
            img_a = load_image(path_a);
            img_b = load_image(path_b);
        } catch (const std::exception& e) {
            std::cerr << "Warning: skipping pair " << pair.pair_id << ": " << e.what() << "\n";
            continue;
        }

        // Upload images to GPU outside measurement window
        runner.upload_pair(img_a, img_b);

        for (const auto& cfg : configs) {
            // Warmup iterations (is_warmup = 1)
            for (int w = 0; w < num_warmup; ++w) {
                RunRecord rec;
                runner.execute_run(cfg, run_id, STITCH_GIT_REV, dataset, split, pair.pair_id, w, 1, &rec);
                write_csv_row(csv_out, rec);
            }

            // Measurement iterations (is_warmup = 0)
            for (int r = 0; r < num_repeats; ++r) {
                RunRecord rec;
                runner.execute_run(cfg, run_id, STITCH_GIT_REV, dataset, split, pair.pair_id, r, 0, &rec);
                write_csv_row(csv_out, rec);
                total_runs++;
            }
            csv_out.flush();
        }
    }

    std::cout << "Benchmark harness completed successfully. Total measurement runs: " << total_runs << "\n";
    return 0;
}
