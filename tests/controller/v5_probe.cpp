// CPU inference parity and actual lifecycle of a loaded v5 policy.
#include <iomanip>
#include <chrono>
#include <iostream>
#include "harness/controller_policy.hpp"
#include "common/geometry.hpp"
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    try {
        auto policy=stitch::harness::parse_controller_policy(stitch::harness::JsonValue::parse_file(argv[1]));
        if(!policy.v5_mode)return 2;
        stitch::controller::Controller controller;
        if(policy.configure(controller)!=stitch::Status::OK)return 2;
        auto rows=stitch::harness::JsonValue::parse_file(argv[2]);
        std::cout<<std::setprecision(17)<<"pre,attempt,s6,reject,future,ssim,psnr,coverage,joint,initial_index,score,t_selector_ms,t_gate_inference_ms,action,gate_risk,gate_evaluated\n";
        for(const auto& row:rows.arr_val) {
            double pre[39],post[93],future[45]{};
            for(int i=0;i<39;++i)pre[i]=future[i]=row.obj_val.at("pre").arr_val.at(i).num_val;
            for(int i=0;i<93;++i)post[i]=row.obj_val.at("post").arr_val.at(i).num_val;
            controller.reset_pair();stitch::controller::SceneSignals scene{};
            stitch::make_scale_info(512,512,policy.probe_scale,&scene.a.scale);scene.b.scale=scene.a.scale;
            scene.features_present=true;scene.a.mean_abs_gradient=pre[0];scene.b.mean_abs_gradient=pre[1];
            std::copy_n(pre,26,scene.features.begin());std::copy_n(pre+33,6,scene.features.begin()+26);
            stitch::controller::Knobs first{};
            const auto begin=std::chrono::steady_clock::now();
            if(controller.begin_pair(scene,&first)!=stitch::Status::OK)return 3;
            const double selector_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            std::size_t index=0;while(index<policy.ladder.size() && !stitch::controller::detail::same_knobs(policy.ladder[index],first))++index;
            const double initial_score=controller.v5_telemetry().score_ms;
            for(int i=0;i<9;++i) {
                const double* input=i==4?future:i>=5?post:pre;
                if(i)std::cout<<',';
                std::cout<<stitch::controller::predict_ensemble(policy.v5_models[i].view(),input);
            }
            const auto gate_begin=std::chrono::steady_clock::now();volatile double observed=0;
            for(int repeat=0;repeat<20;++repeat)for(int k=5;k<9;++k)observed+=stitch::controller::predict_ensemble(policy.v5_models[k].view(),post);
            const double gate_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-gate_begin).count()/20.;
            stitch::controller::Feedback feedback{first,stitch::Status::OK,{}};
            auto& h=feedback.homography;h.status=stitch::Status::OK;h.num_matches=h.num_inliers=first.K;
            h.inlier_ratio=h.inlier_coverage=1.;h.H[0]=h.H[4]=h.H[8]=h.H_s[0]=h.H_s[4]=h.H_s[8]=1.;
            feedback.features_present=feedback.spatial_present=true;
            std::copy_n(post+33,24,feedback.features.begin());std::copy_n(post+63,14,feedback.features.begin()+24);
            std::copy_n(post+77,16,feedback.spatial.begin());stitch::controller::Decision decision{};
            if(controller.observe(feedback,&decision)!=stitch::Status::OK)return 4;
            const auto& gate=controller.v5_gate_telemetry();
            std::cout<<','<<index<<','<<initial_score<<','<<selector_ms<<','<<gate_ms<<','<<int(decision.action)<<','<<gate.risk_joint<<','<<gate.gate_evaluated<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
