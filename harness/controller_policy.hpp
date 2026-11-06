// Policy loading happens before GPU allocation and outside measured runs.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "controller/controller.hpp"
#include "harness/json_parser.hpp"

namespace stitch::harness {

struct BoostedModel {
    std::vector<controller::BoostNode> nodes;
    std::vector<std::size_t> roots;
    int features{};
    float bias{};
    double slope{1.},intercept{};
    bool exponential{};
    controller::EnsembleView view() const {
        return {nodes.data(),nodes.size(),roots.data(),roots.size(),features,bias,slope,intercept,exponential};
    }
};

struct ControllerPolicy {
    std::string id;
    double probe_scale{};
    std::vector<controller::Knobs> ladder;
    std::vector<controller::SelfCheck> checks;
    std::vector<controller::SceneRule> rules;
    std::vector<controller::TreeNode> selector;
    std::vector<std::vector<controller::TreeNode>> gate_nodes;
    mutable std::vector<controller::TreeView> gates;
    std::vector<std::size_t> retry_targets;
    bool pooled_mode{};
    BoostedModel pre_quality,post_quality,latency;
    mutable controller::PooledModels pooled_view{};

    Status configure(controller::Controller& target) const {
        gates.clear();
        for (const auto& nodes : gate_nodes) gates.push_back({nodes.data(), nodes.size()});
        if(pooled_mode) {
            pooled_view.pre_quality=pre_quality.view();pooled_view.post_quality=post_quality.view();
            pooled_view.latency=latency.view();
        }
        const controller::Policy policy{probe_scale, rules.data(), rules.size(), checks.data(), checks.size(),
            {selector.data(), selector.size()}, gates.empty() ? nullptr : gates.data(), gates.size(),
            retry_targets.empty() ? nullptr : retry_targets.data(), retry_targets.size(),pooled_mode?&pooled_view:nullptr};
        const auto status = target.configure(policy);
        return status == Status::OK ? target.initialize(ladder.data(), ladder.size()) : status;
    }
};

namespace policy_detail {
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
inline BoostedModel ensemble(const JsonValue& value) {
    BoostedModel result;
    result.features=integer(value,"features",77);result.bias=static_cast<float>(number(value,"bias"));
    const auto transform=field(value,"transform",JsonValue::String).str_val;
    if(transform!="exp" && transform!="sigmoid")throw std::runtime_error("Unknown ensemble transform");
    result.exponential=transform=="exp";
    const auto& calibration=field(value,"calibration",JsonValue::Object);
    result.slope=number(calibration,"slope");result.intercept=number(calibration,"intercept");
    const auto& trees=field(value,"trees",JsonValue::Array).arr_val;
    if(trees.empty() || trees.size()>128)throw std::runtime_error("Invalid ensemble tree count");
    for(const auto& tree:trees) {
        if(tree.type!=JsonValue::Array || tree.arr_val.empty() || tree.arr_val.size()>127)
            throw std::runtime_error("Invalid ensemble tree size");
        const auto start=result.nodes.size();result.roots.push_back(start);
        for(const auto& node:tree.arr_val) {
            if(node.obj_val.count("leaf"))result.nodes.push_back({-1,-1,-1,0.f,static_cast<float>(number(node,"leaf"))});
            else result.nodes.push_back({integer(node,"feature",76),
                static_cast<int>(start)+integer(node,"left",126),static_cast<int>(start)+integer(node,"right",126),
                static_cast<float>(number(node,"threshold")),0.f});
        }
    }
    if(!controller::valid_ensemble(result.view(),result.features,result.exponential))
        throw std::runtime_error("Malformed ensemble nodes");
    return result;
}
inline std::vector<controller::TreeNode> tree(const JsonValue& array) {
    if (array.type != JsonValue::Array || array.arr_val.empty() || array.arr_val.size() > 127)
        throw std::runtime_error("Invalid bounded tree");
    std::vector<controller::TreeNode> nodes;
    for (const auto& value : array.arr_val) {
        if (value.obj_val.count("value"))
            nodes.push_back({-1, 0., -1, -1, integer(value, "value", 15)});
        else nodes.push_back({integer(value, "feature", controller::selector_feature_count-1),
            number(value, "threshold"), integer(value, "left", 126), integer(value, "right", 126), -1});
    }
    return nodes;
}
}  // namespace policy_detail

inline ControllerPolicy parse_controller_policy(const JsonValue& root) {
    using namespace policy_detail;
    const auto schema = field(root, "schema_version", JsonValue::String).str_val;
    if (schema != "controller-policy-v2" && schema != "controller-policy-v3" && schema!="controller-policy-v4")
        throw std::runtime_error("Unsupported controller policy schema");
    ControllerPolicy policy;
    policy.id = field(root, "policy_id", JsonValue::String).str_val;
    policy.probe_scale = number(root, "probe_scale");
    const auto& entries = field(root, "ladder", JsonValue::Array).arr_val;
    if (entries.empty() || entries.size() > (schema=="controller-policy-v4"?64:16) || policy.id.empty())
        throw std::runtime_error("Policy exceeds attempt-trace capacity or has no identity");
    for (const auto& entry : entries) {
        const auto blend = field(entry, "blend_mode", JsonValue::String).str_val;
        if (blend != "feather") throw std::runtime_error("Calibrated policy requires fixed feather blending");
        policy.ladder.push_back({number(entry, "S"), integer(entry, "K", 4000),
                                 integer(entry, "R", 2048), s6::BlendMode::Feather});
    }
    for (const auto& entry : field(root, "self_checks", JsonValue::Array).arr_val) {
        policy.checks.push_back({integer(entry, "min_inliers", 4000), number(entry, "min_inlier_ratio"),
                                number(entry, "min_inlier_coverage"), number(entry, "max_residual_s1")});
    }
    for (const auto& entry : field(root, "scene_rules", JsonValue::Array).arr_val) {
        policy.rules.push_back({number(entry, "min_pair_gradient"),
            static_cast<std::size_t>(integer(entry, "first_index", schema=="controller-policy-v4"?63:15))});
    }
    if (schema == "controller-policy-v3") {
        if (field(root,"feature_schema",JsonValue::String).str_val != "controller-features-v1")
            throw std::runtime_error("Unknown Controller feature layout");
        policy.selector = tree(field(root, "selector", JsonValue::Array));
        for (const auto& entry : field(root, "gates", JsonValue::Array).arr_val)
            policy.gate_nodes.push_back(tree(entry));
        for (const auto& nodes : policy.gate_nodes) policy.gates.push_back({nodes.data(), nodes.size()});
        if (root.obj_val.count("retry_targets")) {
            for (const auto& target:field(root,"retry_targets",JsonValue::Array).arr_val) {
                if (target.type!=JsonValue::Number || !std::isfinite(target.num_val) || target.num_val<0 ||
                    target.num_val>entries.size() || target.num_val!=std::floor(target.num_val))
                    throw std::runtime_error("Invalid bounded retry target");
                policy.retry_targets.push_back(static_cast<std::size_t>(target.num_val));
            }
        }
    }
    if(schema=="controller-policy-v4") {
        if(field(root,"feature_schema",JsonValue::String).str_val!="controller-features-v2")
            throw std::runtime_error("Unknown pooled feature layout");
        const auto& models=field(root,"models",JsonValue::Object);
        policy.pooled_mode=true;
        policy.pre_quality=ensemble(field(models,"pre_quality",JsonValue::Object));
        policy.post_quality=ensemble(field(models,"post_quality",JsonValue::Object));
        policy.latency=ensemble(field(models,"latency",JsonValue::Object));
        auto& view=policy.pooled_view;
        view.pre_threshold=number(root,"pre_threshold");view.gate_threshold=number(root,"gate_threshold");
        view.extra_ms=number(root,"extra_ms");view.reference_index=integer(root,"reference_index",63);
        view.pair_relations=field(root,"pair_relations",JsonValue::Bool).bool_val;
        view.relation_mask=integer(root,"relation_mask",7);
        if(view.reference_index>=policy.ladder.size())throw std::runtime_error("Invalid pooled fallback");
        const auto& ref=policy.ladder[view.reference_index];
        if(ref.S!=1. || ref.K!=2000 || ref.R!=1024)throw std::runtime_error("Pooled reference differs from QoR baseline");
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
            {{.08, 0}, {.04, 1}, {0., 2}}};
}

}  // namespace stitch::harness
