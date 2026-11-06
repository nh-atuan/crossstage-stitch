// Policy loading happens before GPU allocation and outside measured runs.
#pragma once

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "controller/controller.hpp"
#include "harness/json_parser.hpp"

namespace stitch::harness {

struct ControllerPolicy {
    std::string id;
    double probe_scale{};
    std::vector<controller::Knobs> ladder;
    std::vector<controller::SelfCheck> checks;
    std::vector<controller::SceneRule> rules;
    std::optional<std::uint64_t> seed;

    Status configure(controller::Controller& target) const {
        const controller::Policy policy{probe_scale, rules.data(), rules.size(), checks.data(), checks.size()};
        const auto status = target.configure(policy);
        return status == Status::OK ? target.initialize(ladder.data(), ladder.size()) : status;
    }
};

namespace policy_detail {
inline void known_fields(const JsonValue& parent, std::initializer_list<const char*> names) {
    if (parent.type != JsonValue::Object) throw std::runtime_error("Expected controller policy object");
    for (const auto& entry : parent.obj_val) {
        bool known = false;
        for (const char* name : names) known |= entry.first == name;
        if (!known) throw std::runtime_error("Unknown controller policy field: " + entry.first);
    }
}
inline const JsonValue& field(const JsonValue& parent, const char* name, JsonValue::Type type) {
    const auto it = parent.obj_val.find(name);
    if (parent.type != JsonValue::Object || it == parent.obj_val.end() || it->second.type != type)
        throw std::runtime_error(std::string("Invalid controller policy field: ") + name);
    return it->second;
}
inline double number(const JsonValue& parent, const char* name) {
    const auto result = field(parent, name, JsonValue::Number).num_val;
    if (!std::isfinite(result)) throw std::runtime_error("Nonfinite policy value");
    return result;
}
inline int integer(const JsonValue& parent, const char* name, int maximum) {
    const auto result = number(parent, name);
    if (result < 0 || result > maximum || result != std::floor(result))
        throw std::runtime_error(std::string("Invalid policy integer: ") + name);
    return static_cast<int>(result);
}
}  // namespace policy_detail

inline ControllerPolicy parse_controller_policy(const JsonValue& root) {
    using namespace policy_detail;
    known_fields(root, {"schema_version", "policy_id", "probe_scale", "ladder", "self_checks",
                        "scene_rules", "provenance", "residual_unit"});
    if (field(root, "schema_version", JsonValue::String).str_val != "controller-policy-v2")
        throw std::runtime_error("Unsupported controller policy schema");
    ControllerPolicy policy;
    policy.id = field(root, "policy_id", JsonValue::String).str_val;
    policy.probe_scale = number(root, "probe_scale");
    const auto& entries = field(root, "ladder", JsonValue::Array).arr_val;
    if (entries.empty() || entries.size() > 16 || policy.id.empty() ||
        policy.id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos)
        throw std::runtime_error("Policy exceeds attempt-trace capacity or has no identity");
    if (root.obj_val.count("residual_unit") &&
        field(root, "residual_unit", JsonValue::String).str_val != "s1_pixels")
        throw std::runtime_error("max_residual_s1 must use s1_pixels");
    if (root.obj_val.count("provenance")) {
        const auto& provenance = field(root, "provenance", JsonValue::Object);
        if (provenance.obj_val.count("seed")) {
            const double seed = number(provenance, "seed");
            if (seed < 0 || seed > 9007199254740991.0 || seed != std::floor(seed))
                throw std::runtime_error("Policy provenance.seed must be an exactly representable nonnegative integer");
            policy.seed = static_cast<std::uint64_t>(seed);
        }
    }
    for (const auto& entry : entries) {
        known_fields(entry, {"S", "K", "R", "blend_mode", "config_id"});
        if (entry.obj_val.count("config_id")) field(entry, "config_id", JsonValue::String);
        const auto blend = field(entry, "blend_mode", JsonValue::String).str_val;
        if (blend != "feather") throw std::runtime_error("Calibrated policy requires fixed feather blending");
        policy.ladder.push_back({number(entry, "S"), integer(entry, "K", 4000),
                                 integer(entry, "R", 2048), s6::BlendMode::Feather});
    }
    for (const auto& entry : field(root, "self_checks", JsonValue::Array).arr_val) {
        known_fields(entry, {"min_inliers", "min_inlier_ratio", "min_inlier_coverage", "max_residual_s1"});
        policy.checks.push_back({integer(entry, "min_inliers", 4000), number(entry, "min_inlier_ratio"),
                                number(entry, "min_inlier_coverage"), number(entry, "max_residual_s1")});
    }
    for (const auto& entry : field(root, "scene_rules", JsonValue::Array).arr_val) {
        known_fields(entry, {"min_pair_gradient", "first_index"});
        policy.rules.push_back({number(entry, "min_pair_gradient"),
            static_cast<std::size_t>(integer(entry, "first_index", 15))});
    }
    controller::Controller validation;
    if (policy.configure(validation) != Status::OK)
        throw std::runtime_error("Invalid Controller policy or ladder");
    return policy;
}

inline ControllerPolicy legacy_controller_policy() {
    constexpr auto feather = s6::BlendMode::Feather;
    return {"legacy-v1", .25,
            {{.25, 500, 128, feather}, {.5, 1000, 128, feather}, {.5, 1000, 256, feather},
             {.75, 2000, 512, feather}, {1., 2000, 1024, feather}},
            {{15, .15, .05, 4.}, {20, .15, .05, 3.5}, {25, .2, .05, 3.},
             {30, .2, .05, 2.5}, {35, .2, .05, 2.}},
            {{.08, 0}, {.04, 1}, {0., 2}}, std::nullopt};
}

}  // namespace stitch::harness
