// Benchmark Harness entry point (stitch_harness).
// Supports fixed baseline, oracle, and adaptive controller modes.
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "controller/controller.hpp"
#include "controller/policy.hpp"
#include "common/admission_control.hpp"
#include "harness/cuda_error.hpp"
#include "harness/controller_policy.hpp"
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
        << "status,fail_stage,policy_id,attempt_count,t_probe_gpu,t_stream_wait,probe_gradient_a,probe_gradient_b\n";
}

void write_csv_row(std::ostream& out, const RunRecord& r) {
    out << r.run_id << "," << r.git_rev << "," << r.dataset << "," << r.split << ","
        << r.pair_id << "," << r.config_id << "," << r.repeat_idx << "," << r.is_warmup << ","
        << r.seed << ",";
    if (r.policy_id.empty()) out << std::fixed << std::setprecision(4);
    else out << std::defaultfloat << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << r.S << "," << r.K << "," << r.R << "," << r.blend_mode << ","
        << std::fixed << std::setprecision(r.policy_id.empty() ? 3 : 6)
        << r.t_h2d << "," << r.t_s1 << "," << r.t_s2 << "," << r.t_s3 << ","
        << r.t_s4 << "," << r.t_s5 << "," << r.t_s6 << "," << r.t_controller << ","
        << r.t_d2h << "," << r.t_e2e << ","
        << r.num_kp_a << "," << r.num_kp_b << "," << r.num_matches << ","
        << r.num_inliers << ","
        << std::setprecision(6)
        << r.inlier_ratio << "," << r.inlier_coverage << "," << r.reproj_residual << ",";

    for (int i = 0; i < 9; ++i) {
        if (r.status == "OK" || r.policy_id.empty())
            out << std::defaultfloat << std::setprecision(std::numeric_limits<double>::max_digits10) << r.H[i];
        out << ',';
    }

    out << r.canvas_pixels << "," << r.overlap_pixels << ","
        << r.blend_mode_effective << "," << r.num_levels << ","
        << r.peak_workspace_bytes << "," << r.cache_hit << ","
        << r.status << "," << r.fail_stage << ',';
    if (r.policy_id.empty()) out << ",,,";
    else out << r.policy_id << ',' << r.attempt_count << ',' << std::fixed << std::setprecision(6)
             << r.t_probe_gpu << ',' << r.t_stream_wait;
    out << ',' << std::defaultfloat << std::setprecision(17)
        << r.probe_gradient_a << ',' << r.probe_gradient_b << '\n';
}

void set_adaptive_config(RunRecord& run, const AttemptTrace& trace) {
    if (!trace.count) return;
    const auto& knobs = trace.records[trace.count - 1].knobs;
    run.blend_mode = run.blend_mode_effective = attempt_blend_name(knobs.B);
    std::ostringstream id;
    id << 'S';
    if (std::round(knobs.S * 100) / 100 == knobs.S) id << std::fixed << std::setprecision(2);
    else id << std::setprecision(std::numeric_limits<double>::max_digits10);
    id << knobs.S << "_K" << knobs.K << "_R" << knobs.R << "_B" << run.blend_mode;
    run.config_id = id.str();
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
    bool seed_given = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (arg == "--controller-policy" && i + 1 < argc) controller_policy_path = argv[++i];
        else if (arg == "--trace-out" && i + 1 < argc) trace_path = argv[++i];
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
        else if (arg == "--seed" && i + 1 < argc) {
            const std::string value = argv[++i];
            try {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("seed");
                seed = std::stoull(value);
                seed_given = true;
            } catch (const std::exception&) {
                std::cerr << "Invalid --seed value\n";
                return 1;
            }
        }
        else if (arg == "--controller-policy" && i + 1 < argc) controller_policy_path = argv[++i];
        else if (arg == "--trace-out" && i + 1 < argc) trace_path = argv[++i];
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
        else {
            std::cerr << "Unknown option or missing value: " << arg << "\n";
            return 1;
        }
    }

    if ((mode != "adaptive" && mode != "fixed" && mode != "oracle") ||
        (mode != "adaptive" && (!controller_policy_path.empty() || !trace_path.empty())) ||
        num_warmup < 0 || num_repeats <= 0 || num_warmup > std::numeric_limits<int>::max() - num_repeats) {
        std::cerr << "Invalid benchmark mode, policy option or repeat count\n";
        return 1;
    }
    try {
        const auto output = std::filesystem::weakly_canonical(out_path);
        for (const auto& input : {manifest_path, configs_path, controller_policy_path, img_a_path, img_b_path}) {
            if (input.empty() || input == "legacy-v1") continue;
            std::error_code ec;
            if (std::filesystem::weakly_canonical(input) == output ||
                std::filesystem::equivalent(input, output, ec))
                throw std::runtime_error("--out aliases a benchmark input");
        }
        if (!trace_path.empty()) {
            const auto trace = std::filesystem::weakly_canonical(trace_path);
            if (std::filesystem::exists(trace) || trace == output)
                throw std::runtime_error("Attempt trace requires a fresh path distinct from --out");
        }
    } catch (const std::exception& error) {
        std::cerr << "Output path error: " << error.what() << '\n';
        return 1;
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
                if (seed_given) for (auto& config : configs) config.seed = seed;
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
    }

    // Configure Controller if in adaptive mode
    stitch::controller::Controller controller;
    ControllerPolicy controller_policy;
    if (mode == "adaptive") {
        try {
            if (controller_policy_path.empty())
                throw std::runtime_error("Adaptive mode requires --controller-policy <path> or explicit legacy-v1");
            controller_policy = controller_policy_path == "legacy-v1"
                ? legacy_controller_policy() : parse_controller_policy(JsonValue::parse_file(controller_policy_path));
            if (controller_policy.seed) {
                if (seed_given && seed != *controller_policy.seed)
                    throw std::runtime_error("CLI seed differs from policy provenance.seed");
                seed = *controller_policy.seed;
            }
            if (controller_policy.configure(controller) != stitch::Status::OK)
                throw std::runtime_error("failed to initialize Controller");
        } catch (const std::exception& error) {
            std::cerr << "Controller policy error: " << error.what() << '\n';
            return 1;
        }
    }

    // Initialize pipeline runner with maximum bounds
    PipelineRunner runner;
    runner.set_reuse_enabled(reuse_enabled);
    runner.set_collect_features(!features_path.empty() && mode!="adaptive");
    try {
        int max_width = 2048, max_height = 2048;
        const auto directory = manifest_path.empty() ? std::string{} : get_parent_dir(manifest_path);
        for (const auto& pair : pairs) {
            for (const auto& path : {pair.img_a_path, pair.img_b_path}) {
                try {
                    const auto image = load_image(join_paths(directory, path));
                    max_width = std::max(max_width, image.width);
                    max_height = std::max(max_height, image.height);
                } catch (const std::exception&) {
                    // The execution loop records each input failure in its repeat ledger.
                }
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
        features_out << '\n';
    }
    const auto write_features=[&](const RunRecord& r) {
        if (!features_out.is_open()) return;
        features_out << std::setprecision(17) << r.pair_id << ',' << r.config_id << ',' << r.seed << ','
            << r.repeat_idx << ',' << r.is_warmup << ',' << r.features_present << ',' << r.t_features;
        for (double v:r.scene_features) features_out << ',' << v;
        for (double v:r.gate_features) features_out << ',' << v;
        features_out << '\n';
    };
    std::ofstream trace_out;
    if (!trace_path.empty()) {
        if (mode != "adaptive") {
            std::cerr << "Attempt trace requires adaptive mode.\n";
            return 1;
        }
        trace_out.open(trace_path);
        if (!trace_out) {
            std::cerr << "Error opening attempt trace: " << trace_path << '\n';
            return 1;
        }
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

    std::uint64_t total_runs = 0;
    for (const auto& pair : pairs) {
        std::string path_a = join_paths(manifest_dir, pair.img_a_path);
        std::string path_b = join_paths(manifest_dir, pair.img_b_path);

        std::string input_status, input_stage = "input";
        bool upload_cuda_failure = false;
        try {
            const auto img_a = load_image(path_a);
            const auto img_b = load_image(path_b);
            input_stage = "h2d";
            runner.upload_pair(img_a, img_b);
        } catch (const CudaError& error) {
            input_status = error.code == cudaErrorMemoryAllocation ? "ERR_DEVICE_OOM" : "ERR_CUDA";
            upload_cuda_failure = true;
            std::cerr << "CUDA upload failed for " << pair.pair_id << ": " << error.what() << '\n';
        } catch (const stitch::admission::Error& error) {
            input_status = error.code;
        } catch (const std::exception& error) {
            std::cerr << "Input failed for " << pair.pair_id << ": " << error.what() << '\n';
            input_status = "ERR_INPUT";
        }

        auto emit_repeats = [&](const HarnessConfig* config) {
            const bool adaptive = config == nullptr;
            for (int i = 0; i < num_warmup + num_repeats; ++i) {
                const bool warmup = i < num_warmup;
                const int repeat = warmup ? i : i - num_warmup;
                RunRecord record;
                bool cuda_failed = false;
                const AttemptTrace* trace = nullptr;
                if (!input_status.empty()) {
                    record.run_id = run_id;
                    record.git_rev = STITCH_GIT_REV;
                    record.dataset = dataset;
                    record.split = split;
                    record.pair_id = pair.pair_id;
                    record.repeat_idx = repeat;
                    record.is_warmup = warmup;
                    record.seed = adaptive ? seed : config->seed;
                    record.config_id = adaptive ? "adaptive_unstarted" : config->config_id;
                    if (config) {
                        record.S = config->S; record.K = config->K; record.R = config->R;
                        record.blend_mode = record.blend_mode_effective = config->blend_mode_str;
                    }
                    record.status = input_status;
                    record.fail_stage = input_stage;
                } else if (adaptive) {
                    try {
                        runner.execute_adaptive_run(controller, run_id, STITCH_GIT_REV, dataset, split,
                                                    pair.pair_id, repeat, warmup, seed, &record);
                        cuda_failed = record.status == "CUDA_ERROR" || record.status == "ERR_CUDA";
                    } catch (const CudaError&) {
                        cuda_failed = true;
                    }
                    trace = &runner.attempt_trace();
                    set_adaptive_config(record, *trace);
                } else {
                    runner.execute_run(*config, run_id, STITCH_GIT_REV, dataset, split, pair.pair_id,
                                       repeat, warmup, &record);
                }
                if (adaptive) record.policy_id = controller_policy.id;
                write_csv_row(csv_out, record);
                write_features(record);
                if (trace && trace_out.is_open()) write_attempt_rows(trace_out, record, *trace);
                if (!warmup) ++total_runs;
                if (cuda_failed) {
                    csv_out.flush();
                    if (trace_out.is_open()) trace_out.flush();
                    std::cerr << "CUDA failure stopped the adaptive campaign at " << pair.pair_id << '\n';
                    return false;
                }
            }
            csv_out.flush();
            if (trace_out.is_open()) trace_out.flush();
            if (features_out.is_open()) features_out.flush();
            if (!csv_out || (trace_out.is_open() && !trace_out) || (features_out.is_open() && !features_out)) {
                std::cerr << "Failed to write benchmark output\n";
                return false;
            }
            return true;
        };
        if (mode == "adaptive") {
            if (!emit_repeats(nullptr)) return 2;
        } else {
            for (const auto& config : configs) if (!emit_repeats(&config)) return 2;
        }
        if (upload_cuda_failure) return 2;
    }

    std::cout << "Benchmark harness completed successfully. Total measurement runs: " << total_runs << "\n";
    return 0;
}
