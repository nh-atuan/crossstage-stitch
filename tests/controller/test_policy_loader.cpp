#include "check.hpp"
#include "harness/controller_policy.hpp"
#include "common/geometry.hpp"

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
    value = fixture(); value.obj_val["unexpected"] = JsonValue{}; invalid(value);
    value = fixture(); value.obj_val["residual_unit"] = JsonValue::parse(R"("original_pixels")"); invalid(value);
    value = fixture(); value.obj_val["provenance"] = JsonValue::parse(R"({"seed":42})");
    CHECK(parse_controller_policy(value).seed == 42);
    for (const auto* malformed : {R"({"S":0.25,"S":0.75})", "{} {}"}) {
        bool failed = false;
        try { JsonValue::parse(malformed); } catch (const std::exception&) { failed = true; }
        CHECK(failed);
    }
    auto legacy = legacy_controller_policy();
    CHECK(legacy.configure(controller) == stitch::Status::OK && legacy.id == "legacy-v1");
    auto blind_s05 = legacy_blind_s05_policy();
    CHECK(blind_s05.configure(controller) == stitch::Status::OK && blind_s05.id == "legacy-v1-blind-s05");
    {
        stitch::ScaleInfo scale{};
        CHECK(stitch::make_scale_info(100, 100, .25, &scale) == stitch::Status::OK);
        stitch::controller::SceneSignals low_grad{{scale, 0.0}, {scale, 0.0}, {}, false};
        stitch::controller::Knobs first{};
        CHECK(controller.begin_pair(low_grad, &first) == stitch::Status::OK);
        CHECK(first.S == 0.5 && first.K == 1000 && first.R == 128);
        controller.reset_pair();
    }
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
    auto pooled=fixture();pooled.obj_val["schema_version"].str_val="controller-policy-v4";
    pooled.obj_val["feature_schema"]=JsonValue::parse("\"controller-features-v2\"");
    pooled.obj_val["reference_index"]=JsonValue::parse("1");
    pooled.obj_val["pre_threshold"]=JsonValue::parse("0.8");
    pooled.obj_val["gate_threshold"]=JsonValue::parse("0.95");
    pooled.obj_val["extra_ms"]=JsonValue::parse("0.5");
    pooled.obj_val["pair_relations"]=JsonValue::parse("false");
    pooled.obj_val["relation_mask"]=JsonValue::parse("0");
    pooled.obj_val["models"]=JsonValue::parse(R"({
      "pre_quality":{"features":39,"bias":0,"transform":"sigmoid","calibration":{"slope":1,"intercept":0},"trees":[[{"leaf":5}]]},
      "post_quality":{"features":77,"bias":0,"transform":"sigmoid","calibration":{"slope":1,"intercept":0},"trees":[[{"leaf":2}]]},
      "latency":{"features":39,"bias":0,"transform":"exp","calibration":{"slope":1,"intercept":0},"trees":[[{"feature":31,"threshold":0.5,"left":1,"right":2},{"leaf":0.69314718},{"leaf":2.30258509}]]}})");
    auto model=parse_controller_policy(pooled);auto copy=model;
    CHECK(copy.configure(controller)==stitch::Status::OK);
    stitch::ScaleInfo scale{};CHECK(stitch::make_scale_info(100,100,.25,&scale)==stitch::Status::OK);
    stitch::controller::SceneSignals scene{{scale,.1},{scale,.1},{},true};
    stitch::controller::Knobs first{};
    CHECK(controller.begin_pair(scene,&first)==stitch::Status::OK && first.K==500);
    stitch::HomographyResult h{};h.status=stitch::Status::OK;h.num_matches=h.num_inliers=10;
    h.inlier_ratio=1;h.inlier_coverage=.5;h.reproj_residual=.1;
    h.H[0]=h.H[4]=h.H[8]=h.H_s[0]=h.H_s[4]=h.H_s[8]=1;
    stitch::controller::Feedback feedback{first,stitch::Status::OK,h,{},true};
    stitch::controller::Decision decision{};
    CHECK(controller.observe(feedback,&decision)==stitch::Status::OK);
    CHECK(decision.action==stitch::controller::Action::RETRY && decision.next.K==2000);
    CHECK(!controller.needs_gate_features());
    feedback.executed=decision.next;feedback.features_present=false;
    CHECK(controller.observe(feedback,&decision)==stitch::Status::OK && decision.action==stitch::controller::Action::ACCEPT);
    pooled.obj_val["models"].obj_val["latency"].obj_val["trees"].arr_val[0].arr_val[0].obj_val["left"].num_val=0;invalid(pooled);
    std::puts("Controller policy loader tests passed (synthetic configuration)");
}
