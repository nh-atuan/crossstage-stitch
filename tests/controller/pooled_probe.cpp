// Compare exported pooled-model predictions with Python on recorded features.
#include <iomanip>
#include <iostream>
#include <vector>
#include "harness/controller_policy.hpp"
#include "common/geometry.hpp"
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    try {
        auto policy=stitch::harness::parse_controller_policy(stitch::harness::JsonValue::parse_file(argv[1]));
        if(!policy.pooled_mode)return 2;
        stitch::controller::Controller controller;
        if(policy.configure(controller)!=stitch::Status::OK)return 2;
        auto rows=stitch::harness::JsonValue::parse_file(argv[2]);
        if(rows.type!=stitch::harness::JsonValue::Array)return 2;
        std::cout<<std::setprecision(17)<<"pre_probability,latency,post_probability,initial_index\n";
        for(const auto& row:rows.arr_val) {
            std::vector<double> pre,post;
            for(const auto& v:row.obj_val.at("pre").arr_val)pre.push_back(v.num_val);
            for(const auto& v:row.obj_val.at("post").arr_val)post.push_back(v.num_val);
            if(pre.size()!=39 || post.size()!=77)return 2;
            controller.reset_pair();stitch::controller::SceneSignals scene{};
            stitch::make_scale_info(512,512,policy.probe_scale,&scene.a.scale);scene.b.scale=scene.a.scale;
            scene.a.mean_abs_gradient=pre[0];scene.b.mean_abs_gradient=pre[1];scene.features_present=true;
            std::copy_n(pre.begin(),26,scene.features.begin());std::copy_n(pre.begin()+33,6,scene.features.begin()+26);
            stitch::controller::Knobs first{};
            if(controller.begin_pair(scene,&first)!=stitch::Status::OK)return 2;
            std::size_t index=0;
            while(index<policy.ladder.size() && !stitch::controller::detail::same_knobs(policy.ladder[index],first))++index;
            if(index==policy.ladder.size())return 2;
            std::cout<<stitch::controller::predict_ensemble(policy.pre_quality.view(),pre.data())<<','
                <<stitch::controller::predict_ensemble(policy.latency.view(),pre.data())<<','
                <<stitch::controller::predict_ensemble(policy.post_quality.view(),post.data())<<','<<index<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
