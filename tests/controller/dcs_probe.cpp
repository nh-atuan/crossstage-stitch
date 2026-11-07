// Actual Controller lifecycle and raw score export; CPU microtiming only.
#include <iostream>
#include <iomanip>
#include <chrono>
#include "harness/controller_policy.hpp"
#include "common/geometry.hpp"
int main(int argc,char** argv){
 if(argc!=3)return 2;
 try{
  auto policy=stitch::harness::parse_controller_policy(stitch::harness::JsonValue::parse_file(argv[1]));
  stitch::controller::Controller controller;if(!policy.dcs_mode || policy.configure(controller)!=stitch::Status::OK)return 3;
  auto rows=stitch::harness::JsonValue::parse_file(argv[2]);
  std::cout<<std::setprecision(17)<<"row,initial,reason,selector_ms,action,next_index,final_action";
  for(int i=0;i<48;++i)std::cout<<",score_"<<i;for(int i=0;i<48;++i)std::cout<<",rank_"<<i;std::cout<<'\n';
  int index=0;
  for(const auto& row:rows.arr_val){stitch::controller::SceneSignals scene{};
   stitch::make_scale_info(512,512,policy.probe_scale,&scene.a.scale);scene.b.scale=scene.a.scale;
   scene.features_present=true;for(int i=0;i<26;++i)scene.features[i]=row.arr_val.at(i).num_val;
   for(int i=0;i<6;++i)scene.features[26+i]=row.arr_val.at(33+i).num_val;
   scene.a.mean_abs_gradient=scene.features[0];scene.b.mean_abs_gradient=scene.features[1];
   auto start=std::chrono::steady_clock::now();
   for(int repeat=0;repeat<20;++repeat){controller.reset_pair();stitch::controller::Knobs first{};if(controller.begin_pair(scene,&first)!=stitch::Status::OK)return 4;}
   double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/20.;
   const auto selection=controller.direct_selection();
   stitch::controller::Feedback feedback{};feedback.executed=policy.ladder[selection.initial];feedback.execution_status=stitch::Status::OK;
   feedback.homography.status=stitch::Status::OK;feedback.homography.num_matches=feedback.homography.num_inliers=feedback.executed.K;
   feedback.homography.inlier_ratio=feedback.homography.inlier_coverage=1.;feedback.homography.reproj_residual=0.;
   for(int i=0;i<9;++i)feedback.homography.H[i]=feedback.homography.H_s[i]=(i%4==0);
   feedback.features_present=true;feedback.spatial_present=true;stitch::controller::Decision decision{};
   if(controller.observe(feedback,&decision)!=stitch::Status::OK)return 6;
   int action=static_cast<int>(decision.action),next=-1,final_action=action;
   if(decision.action==stitch::controller::Action::RETRY){
       for(std::size_t j=0;j<policy.ladder.size();++j)if(stitch::controller::detail::same_knobs(decision.next,policy.ladder[j]))next=j;
       feedback.executed=decision.next;feedback.homography.num_matches=feedback.homography.num_inliers=feedback.executed.K;
       if(controller.observe(feedback,&decision)!=stitch::Status::OK)return 7;final_action=static_cast<int>(decision.action);
   }
   std::cout<<index++<<','<<selection.initial<<','<<selection.reason<<','<<elapsed<<','<<action<<','<<next<<','<<final_action;
   for(double score:selection.scores)std::cout<<','<<score;for(auto j:selection.ranked)std::cout<<','<<j;std::cout<<'\n';
  }
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 5;}return 0;
}
