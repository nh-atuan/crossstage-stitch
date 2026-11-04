#include "check.hpp"
#include "harness/controller_policy.hpp"

using namespace stitch::harness;

namespace {
JsonValue fixture() {
    return JsonValue::parse(R"({"schema_version":"controller-policy-v2","policy_id":"synthetic-test-only",
      "probe_scale":0.25,"ladder":[{"S":0.75,"K":500,"R":128,"blend_mode":"feather"},
      {"S":1.0,"K":2000,"R":1024,"blend_mode":"feather"}],
      "self_checks":[{"min_inliers":4,"min_inlier_ratio":0,"min_inlier_coverage":0,"max_residual_s1":2},
      {"min_inliers":4,"min_inlier_ratio":0,"min_inlier_coverage":0,"max_residual_s1":1e300}],
      "scene_rules":[{"min_pair_gradient":0.05,"first_index":0},{"min_pair_gradient":0,"first_index":1}]})");
}
void invalid(const JsonValue& value) {
    bool failed = false;
    try { parse_controller_policy(value); } catch (const std::exception&) { failed = true; }
    CHECK(failed);
}
}  // namespace

int main() {
    auto policy = parse_controller_policy(fixture());
    stitch::controller::Controller controller;
    CHECK(policy.configure(controller) == stitch::Status::OK);
    CHECK(policy.ladder.size() == 2 && policy.ladder[0].S == .75 && policy.ladder[0].K == 500);
    auto value = fixture(); value.obj_val.erase("scene_rules"); invalid(value);
    value = fixture(); value.obj_val["ladder"].arr_val[0].obj_val["K"].num_val = 500.5; invalid(value);
    value = fixture(); value.obj_val["ladder"].arr_val[0].obj_val["K"].num_val = 4001; invalid(value);
    value = fixture(); value.obj_val["scene_rules"].arr_val[0].obj_val["first_index"].num_val = 2; invalid(value);
    value = fixture(); value.obj_val["scene_rules"].arr_val.pop_back(); invalid(value);
    value = fixture(); value.obj_val["self_checks"].arr_val.pop_back(); invalid(value);
    value = fixture(); value.obj_val["ladder"].arr_val[0].obj_val["blend_mode"].str_val = "unknown"; invalid(value);
    value = fixture(); value.obj_val["ladder"].arr_val.resize(17); invalid(value);
    auto legacy = legacy_controller_policy();
    CHECK(legacy.configure(controller) == stitch::Status::OK && legacy.id == "legacy-v1");
    value=fixture();value.obj_val["schema_version"].str_val="controller-policy-v3";
    value.obj_val["feature_schema"]=JsonValue::parse("\"controller-features-v1\"");
    value.obj_val["selector"]=JsonValue::parse(R"([{"feature":0,"threshold":0.1,"left":1,"right":2},{"value":0},{"value":1}])");
    value.obj_val["gates"]=JsonValue::parse(R"([[{"value":0}],[{"value":1}]])");
    auto learned=parse_controller_policy(value);
    auto copied=learned;
    CHECK(copied.configure(controller)==stitch::Status::OK && controller.uses_learned_features());
    value.obj_val["selector"].arr_val[0].obj_val["left"].num_val=0;invalid(value);
    value.obj_val["selector"].arr_val[0].obj_val["left"].num_val=1;
    value.obj_val["gates"].arr_val[0].arr_val[0].obj_val["value"].num_val=2;invalid(value);
    std::puts("Controller policy loader tests passed (synthetic configuration)");
}
