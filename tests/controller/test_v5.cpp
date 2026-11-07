#include "check.hpp"
#include "controller/controller.hpp"
#include "controller/spatial.hpp"
#include "common/geometry.hpp"
#include <cmath>
#include <iostream>
#include <limits>
using namespace stitch;
using namespace stitch::controller;
int main() {
    CHECK(std::abs(expected_cost(1.,2.,.3,.25,4.,5.,6.)-9.05)<1e-12);
    const std::size_t roots[1]{0};BoostNode high{-1,-1,-1,0,8},low{-1,-1,-1,0,-8},zero{-1,-1,-1,0,0};
    BoostNode positive{-1,-1,-1,0,-.84729785f},negative{-1,-1,-1,0,-2};
    BoostNode timing[3]{{30,1,2,.8f,0},{-1,-1,-1,0,1},{-1,-1,-1,0,3}};
    const auto model=[&](const BoostNode* node,int dimensions,bool exp=false,std::size_t count=1) {
        return EnsembleView{node,count,roots,1,dimensions,0,1.,0.,exp};
    };
    Knobs configs[4]{{.5,500,128,s6::BlendMode::Feather},{.5,500,256,s6::BlendMode::Feather},
        {.5,500,512,s6::BlendMode::Feather},{1.,2000,1024,s6::BlendMode::Feather}};
    SelfCheck checks[4];for(auto& c:checks)c={4,0.,0.,10.};SceneRule rule{0.,3};
    PooledModels old{};old.pre_quality=model(&high,39);old.post_quality=model(&high,77);old.latency=model(timing,39,true,3);
    old.pre_threshold=.85;old.gate_threshold=.65;old.extra_ms=.1;old.reference_index=3;old.relation_mask=4;
    double costs[16];for(int i=0;i<4;++i)for(int j=0;j<4;++j)costs[i*4+j]=i==j?0:j==3?5.:.5;
    V5Models v{};v.pre=model(&high,39);v.attempt=model(timing,39,true,3);v.s6=model(&zero,39,true);
    v.reject=model(&negative,39);v.future=model(&high,45);v.failures={model(&positive,93),model(&positive,93),model(&positive,93),model(&negative,93)};
    v.margin_heads=true;
    v.incremental_ms=costs;v.costs_count=16;v.new_selector=v.new_gate=v.staged=true;v.risk_threshold=.8;v.gate_ms=.1;
    Policy policy{.25,&rule,1,checks,4,{},nullptr,0,nullptr,0,&old,&v};Controller controller;
    CHECK(controller.configure(policy)==Status::OK);CHECK(controller.initialize(configs,4)==Status::OK);
    SceneSignals scene{};make_scale_info(512,512,.25,&scene.a.scale);scene.b.scale=scene.a.scale;
    scene.features_present=true;scene.a.mean_abs_gradient=scene.b.mean_abs_gradient=.1;scene.features[0]=scene.features[1]=.1;
    Knobs current{};CHECK(controller.begin_pair(scene,&current)==Status::OK);CHECK(current.R==128);
    for(int iteration=0;iteration<4;++iteration) {
        Feedback feedback{current,Status::OK,{}};feedback.features_present=true;
        auto& h=feedback.homography;h.status=Status::OK;h.num_matches=h.num_inliers=20;h.inlier_ratio=1.;h.inlier_coverage=.5;h.reproj_residual=0.;
        h.H[0]=h.H[4]=h.H[8]=h.H_s[0]=h.H_s[4]=h.H_s[8]=1.;
        Decision decision{};CHECK(controller.observe(feedback,&decision)==Status::OK);
        if(iteration<3){CHECK(controller.v5_gate_telemetry().risk_joint>.89);CHECK(decision.action==Action::RETRY);CHECK(decision.next.R>current.R || decision.next.S>current.S);current=decision.next;}
        else CHECK(decision.action==Action::ACCEPT);
    }
    Decision sentinel{};Feedback late{};CHECK(controller.observe(late,&sentinel)==Status::INVALID_ARGUMENT);
    controller.reset_pair();CHECK(controller.begin_pair(scene,&current)==Status::OK);
    Feedback singular_feedback{current,Status::OK,{}};auto& bad_h=singular_feedback.homography;
    bad_h.status=Status::OK;bad_h.num_matches=bad_h.num_inliers=20;bad_h.inlier_ratio=1.;bad_h.inlier_coverage=1.;
    singular_feedback.features_present=true;Decision rejected{};
    CHECK(controller.observe(singular_feedback,&rejected)==Status::OK);CHECK(rejected.action!=Action::ACCEPT);
    controller.reset_pair();costs[0]=-1;CHECK(controller.configure(policy)==Status::INVALID_ARGUMENT);costs[0]=0;
    std::uint8_t pixels[32*32*3];for(int i=0;i<32*32*3;++i)pixels[i]=(i*19)%255;
    ImageView image{pixels,32,32,96,PixelFormat::RGB8};double identity[9]{1,0,0,0,1,0,0,0,1},features[16];
    CHECK(spatial_observations(image,image,identity,features));CHECK(features[5]==1.);CHECK(std::abs(features[0]-1.)<1e-12);
    CHECK(std::abs(features[10]-1.)<1e-12 && std::abs(features[11]-1.)<1e-12);
    double reflected[9]{-1,0,31,0,1,0,0,0,1};CHECK(spatial_observations(image,image,reflected,features));CHECK(features[13]==1.);
    double singular[9]{};CHECK(!spatial_observations(image,image,singular,features));
    identity[2]=10000;CHECK(spatial_observations(image,image,identity,features));CHECK(features[5]==0.);
    image.pitch_bytes=1;CHECK(!spatial_observations(image,image,identity,features));
    std::cout<<"v5 host lifecycle/cost/spatial tests passed; TEST_FIXTURE, no GPU timing\n";
}
