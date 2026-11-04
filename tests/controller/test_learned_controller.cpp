#include <limits>
#include "check.hpp"
#include "controller/controller.hpp"
#include "common/geometry.hpp"
using namespace stitch;
using namespace stitch::controller;
int main() {
    TreeNode selector[]={{0,.1,1,2,-1},{-1,0,-1,-1,0},{-1,0,-1,-1,1}};
    TreeNode gate[]={{13,.8,1,2,-1},{-1,0,-1,-1,0},{-1,0,-1,-1,1}};
    TreeNode fallback[]={{-1,0,-1,-1,1}};
    TreeView gates[]={{gate,3},{fallback,1}};
    SceneRule rule{0,1};SelfCheck checks[]={{4,0,0,1e300},{4,0,0,1e300}};
    Policy policy{.25,&rule,1,checks,2,{selector,3},gates,2};
    Knobs ladder[]={{.75,500,128,s6::BlendMode::Feather},{1,2000,1024,s6::BlendMode::Feather}};
    Controller c;CHECK(c.configure(policy)==Status::OK);CHECK(c.initialize(ladder,2)==Status::OK);
    SceneSignals scene{};
    make_scale_info(512,512,.25,&scene.a.scale);scene.b.scale=scene.a.scale;
    scene.features_present=true;scene.features[0]=.1;
    Knobs first{};CHECK(c.begin_pair(scene,&first)==Status::OK);CHECK(first.K==500);
    Feedback f{};f.executed=first;f.execution_status=Status::OK;f.homography.status=Status::OK;
    f.homography.num_matches=10;f.homography.num_inliers=8;f.homography.inlier_ratio=.8;
    f.homography.H[0]=f.homography.H[4]=f.homography.H[8]=1;
    f.homography.H_s[0]=f.homography.H_s[4]=f.homography.H_s[8]=1;
    f.features_present=true;f.features[13]=.7;
    Decision d{};CHECK(c.observe(f,&d)==Status::OK);CHECK(d.action==Action::RETRY);
    CHECK(d.reason==Reason::QUALITY_PREDICTOR_REJECTED);
    f.executed=d.next;CHECK(c.observe(f,&d)==Status::OK);CHECK(d.action==Action::ACCEPT);
    c.reset_pair();scene.features[0]=std::numeric_limits<double>::quiet_NaN();
    CHECK(c.begin_pair(scene,&first)==Status::INVALID_ARGUMENT);
    selector[0].left=0;CHECK(c.configure(policy)==Status::INVALID_ARGUMENT);
    selector[0].left=1;gate[0].feature=gate_feature_count;CHECK(c.configure(policy)==Status::INVALID_ARGUMENT);
    gate[0].feature=13;
    std::size_t targets[]={1,2};policy.retry_targets=targets;policy.retry_target_count=2;
    CHECK(c.configure(policy)==Status::OK);
    targets[0]=0;CHECK(c.configure(policy)==Status::INVALID_ARGUMENT);
    SceneFeatures raw{};raw[2]=.2;raw[14]=.7;
    CHECK(std::abs(selector_features(raw)[26]-.5)<1e-12);
    std::puts("Learned Controller boundary, feedback and malformed tree checks passed");
}
