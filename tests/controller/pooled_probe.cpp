// Compare exported pooled-model predictions with Python on recorded features.
#include <iomanip>
#include <iostream>
#include <vector>
#include "harness/controller_policy.hpp"
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    try {
        auto policy=stitch::harness::parse_controller_policy(stitch::harness::JsonValue::parse_file(argv[1]));
        if(!policy.pooled_mode)return 2;
        auto rows=stitch::harness::JsonValue::parse_file(argv[2]);
        if(rows.type!=stitch::harness::JsonValue::Array)return 2;
        std::cout<<std::setprecision(17)<<"pre_probability,latency,post_probability\n";
        for(const auto& row:rows.arr_val) {
            std::vector<double> pre,post;
            for(const auto& v:row.obj_val.at("pre").arr_val)pre.push_back(v.num_val);
            for(const auto& v:row.obj_val.at("post").arr_val)post.push_back(v.num_val);
            if(pre.size()!=39 || post.size()!=77)return 2;
            std::cout<<stitch::controller::predict_ensemble(policy.pre_quality.view(),pre.data())<<','
                <<stitch::controller::predict_ensemble(policy.latency.view(),pre.data())<<','
                <<stitch::controller::predict_ensemble(policy.post_quality.view(),post.data())<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
