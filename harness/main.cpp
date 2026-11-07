// Benchmark Harness entry point (stitch_harness).
// Supports fixed baseline, oracle, and adaptive controller modes.
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "controller/controller.hpp"
#include "controller/policy.hpp"
#include "harness/harness_types.hpp"
#include "harness/json_parser.hpp"
#include "harness/image_loader.hpp"
#include "harness/pipeline_runner.hpp"
#include "harness/controller_policy.hpp"

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
        << "status,fail_stage,probe_gradient_a,probe_gradient_b,t_probe_features,t_selector,t_spatial\n";
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
        out << std::setprecision(17) << r.H[i] << ",";
    }

    out << r.canvas_pixels << "," << r.overlap_pixels << ","
        << r.blend_mode_effective << "," << r.num_levels << ","
        << r.peak_workspace_bytes << "," << r.cache_hit << ","
        << r.status << "," << r.fail_stage << "," << std::setprecision(17)
        << r.probe_gradient_a << "," << r.probe_gradient_b << "," << r.t_probe_features << "," << r.t_selector << "," << r.t_spatial << "\n";
}

std::string generate_run_id() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "run_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

std::string get_parent_dir(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return "";
    return path.substr(0, pos);
}

std::string join_paths(const std::string& dir, const std::string& rel) {
    if (dir.empty()) return rel;
    if (rel.empty()) return dir;
    if (rel.front() == '/' || rel.front() == '\\' || (rel.size() > 1 && rel[1] == ':')) {
        return rel;
    }
    if (dir.back() == '/' || dir.back() == '\\') {
        return dir + rel;
    }
    return dir + "/" + rel;
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
    std::string mode = "fixed";  // "fixed", "oracle", "adaptive"
    std::string trace_path;
    std::string features_path;
    std::string controller_policy_path;
    unsigned feature_profile_mask=7,spatial_groups=3;
    bool spatial_features=false,profile_pair_ncc=true;
    bool reuse_enabled = true;

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
        else if (arg == "--controller-policy" && i + 1 < argc) controller_policy_path = argv[++i];
        else if (arg == "--trace-out" && i + 1 < argc) trace_path = argv[++i];
        else if(arg=="--feature-relation-mask" && i+1<argc) {feature_profile_mask=std::stoul(argv[++i]);if(feature_profile_mask>7)return 2;}
        else if(arg=="--spatial-groups" && i+1<argc){spatial_groups=std::stoul(argv[++i]);if(spatial_groups>3)return 2;}
        else if(arg=="--spatial-features")spatial_features=true;
        else if(arg=="--no-pair-ncc")profile_pair_ncc=false;
        else if (arg == "--features-out" && i + 1 < argc) features_path = argv[++i];
        else if (arg == "--no-reuse") reuse_enabled = false;
        else if (arg == "--configs" && i + 1 < argc) configs_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else if (arg == "--split" && i + 1 < argc) split = argv[++i];
        else if (arg == "--run-id" && i + 1 < argc) run_id = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--adaptive") mode = "adaptive";
        else if (arg == "--oracle") mode = "oracle";
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
                      << "Modes:\n"
                      << "  --mode <fixed|oracle|adaptive>  Execution mode (default: fixed)\n"
                      << "  --adaptive                      Shortcut for --mode adaptive\n"
                      << "  --controller-policy <path>      Required adaptive policy JSON, or legacy-v1\n"
                      << "  --trace-out <path>              Per-attempt Controller trace CSV\n"
                      << "  --features-out <path>           Collect Controller observations (fixed timings diagnostic)\n"
                      << "  --no-reuse                      A2 control: rerun S4/S5 for R-only retries\n"
                      << "  --oracle                        Shortcut for --mode oracle\n"
                      << "Batch options:\n"
                      << "  --manifest <path>               Pairs JSON manifest\n"
                      << "  --configs <path>                Configs JSON file (for fixed/oracle)\n"
                      << "  --out <path>                    Output CSV file (default: runs.csv)\n"
                      << "Single options:\n"
                      << "  --img-a <path>                  Image A path\n"
                      << "  --img-b <path>                  Image B path\n"
                      << "  -S <double>                     Scale S in (0, 1] (default: 0.5)\n"
                      << "  -K <int>                        Keypoints K (default: 1000)\n"
                      << "  -R <int>                        MSAC Hypotheses R (default: 256)\n"
                      << "  --blend <mode>                  Blend mode: feather|multiband3|multiband5\n";
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

    if (mode != "adaptive") {
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
        for (auto& config : configs) config.seed = seed;
    }

    // Configure Controller if in adaptive mode
    stitch::controller::Controller controller;
    ControllerPolicy controller_policy;
    if (mode == "adaptive") {
        try {
            if (controller_policy_path.empty())
                throw std::runtime_error("Adaptive mode requires --controller-policy <path> or explicit legacy-v1");
            controller_policy = controller_policy_path == "legacy-v1" ? legacy_controller_policy()
                : parse_controller_policy(JsonValue::parse_file(controller_policy_path));
            if (controller_policy.configure(controller) != stitch::Status::OK)
                throw std::runtime_error("Failed to initialize Controller");
        } catch (const std::exception& error) {
            std::cerr << "Controller policy error: " << error.what() << "\n";
            return 1;
        }
    }

    // Initialize pipeline runner with maximum bounds
    PipelineRunner runner;
    runner.set_reuse_enabled(reuse_enabled);
    runner.set_collect_features(!features_path.empty() && mode!="adaptive");
    runner.set_feature_profile(feature_profile_mask,spatial_features,profile_pair_ncc,spatial_groups);
    try {
        int max_width = 2048, max_height = 2048;
        const auto directory = manifest_path.empty() ? std::string{} : get_parent_dir(manifest_path);
        for (const auto& pair : pairs) {
            for (const auto& path : {pair.img_a_path, pair.img_b_path}) {
                const auto image = load_image(join_paths(directory, path));
                max_width = std::max(max_width, image.width);
                max_height = std::max(max_height, image.height);
            }
        }
        runner.initialize(max_width, max_height, 4000, 2048);
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
    std::ofstream features_out;
    if (!features_path.empty()) {
        features_out.open(features_path);
        if (!features_out) return 1;
        features_out << "pair_id,config_id,seed,repeat_idx,is_warmup,features_present,t_features";
        for (int i=0;i<stitch::controller::scene_feature_count;++i) features_out << ",scene_" << i;
        for (int i=0;i<stitch::controller::gate_feature_count;++i) features_out << ",gate_" << i;
        features_out << ",spatial_present,t_probe_features,t_spatial";
        for(int i=0;i<16;++i)features_out << ",spatial_" << i;
        features_out << '\n';
    }
    const auto write_features=[&](const RunRecord& r) {
        if (!features_out.is_open()) return;
        features_out << std::setprecision(17) << r.pair_id << ',' << r.config_id << ',' << r.seed << ','
            << r.repeat_idx << ',' << r.is_warmup << ',' << r.features_present << ',' << r.t_features;
        for (double v:r.scene_features) features_out << ',' << v;
        for (double v:r.gate_features) features_out << ',' << v;
        features_out << ',' << r.spatial_present << ',' << r.t_probe_features << ',' << r.t_spatial;
        for(double v:r.spatial)features_out << ',' << v;
        features_out << '\n';
    };
    std::ofstream trace_out;
    if (!trace_path.empty()) {
        if (mode != "adaptive") {
            std::cerr << "Attempt trace requires adaptive mode.\n";
            return 1;
        }
        trace_out.open(trace_path);
        if (!trace_out) return 1;
        write_attempt_header(trace_out);
    }

    std::cout << "Starting benchmark harness:\n"
              << "  Mode: " << mode << "\n"
              << "  Pairs: " << pairs.size() << "\n"
              << "  Configs: " << (mode == "adaptive" ? controller_policy.ladder.size() : configs.size()) << "\n"
              << "  Policy: " << (mode == "adaptive" ? controller_policy.id : "none") << "\n"
              << "  Seed: " << seed << "\n"
              << "  Warmup: " << num_warmup << ", Repeats: " << num_repeats << "\n"
              << "  Output: " << out_path << "\n";

    std::string manifest_dir = manifest_path.empty() ? "" : get_parent_dir(manifest_path);

    int total_runs = 0;
    for (const auto& pair : pairs) {
        std::string path_a = join_paths(manifest_dir, pair.img_a_path);
        std::string path_b = join_paths(manifest_dir, pair.img_b_path);

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

        if (mode == "adaptive") {
            // Adaptive mode: governed by Controller
            for (int w = 0; w < num_warmup; ++w) {
                RunRecord rec;
                runner.execute_adaptive_run(controller, run_id, STITCH_GIT_REV, dataset, split, pair.pair_id, w, 1, seed, &rec);
                write_csv_row(csv_out, rec);
                write_features(rec);
                if (trace_out.is_open()) write_attempt_rows(trace_out, rec, runner.attempt_trace());
            }
            for (int r = 0; r < num_repeats; ++r) {
                RunRecord rec;
                runner.execute_adaptive_run(controller, run_id, STITCH_GIT_REV, dataset, split, pair.pair_id, r, 0, seed, &rec);
                write_csv_row(csv_out, rec);
                write_features(rec);
                if (trace_out.is_open()) write_attempt_rows(trace_out, rec, runner.attempt_trace());
                total_runs++;
            }
            csv_out.flush();
            if (trace_out.is_open()) trace_out.flush();
        } else {
            // Fixed-Best or Oracle mode
            for (const auto& cfg : configs) {
                for (int w = 0; w < num_warmup; ++w) {
                    RunRecord rec;
                    runner.execute_run(cfg, run_id, STITCH_GIT_REV, dataset, split, pair.pair_id, w, 1, &rec);
                    write_csv_row(csv_out, rec);
                    write_features(rec);
                }
                for (int r = 0; r < num_repeats; ++r) {
                    RunRecord rec;
                    runner.execute_run(cfg, run_id, STITCH_GIT_REV, dataset, split, pair.pair_id, r, 0, &rec);
                    write_csv_row(csv_out, rec);
                    write_features(rec);
                    total_runs++;
                }
                csv_out.flush();
            }
        }
    }

    std::cout << "Benchmark harness completed successfully. Total measurement runs: " << total_runs << "\n";
    return 0;
}
