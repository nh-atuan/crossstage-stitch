#include "check.hpp"
#include "controller/dcs.hpp"
int main(){using namespace stitch::controller;
 BoostNode nodes[]={{32,1,2,.2f,0},{-1,-1,-1,0,2},{-1,-1,-1,0,-1}};std::size_t roots[]={0};
 EnsembleView ensemble{nodes,3,roots,1,48,0,1,0,false};int ties[48];double config[144],scene[39]{};
 for(int j=0;j<48;++j){ties[j]=47-j;config[j*3]=.5;config[j*3+1]=.25;config[j*3+2]=j%4==0?.125:1.;}
 DcsModels model{};model.models=&ensemble;model.count=1;model.tie_order=ties;model.available=true;
 auto ranked=select_direct(model,scene,config,48,47);CHECK(ranked.reason==0 && ranked.initial==44);CHECK(ranked.scores[0]==2 && ranked.scores[1]==-1);
 model.available=false;CHECK(select_direct(model,scene,config,48,47).initial==47);
 model.available=true;scene[0]=std::numeric_limits<double>::quiet_NaN();CHECK(select_direct(model,scene,config,48,47).reason==2);
 scene[0]=0;CHECK(select_direct(model,scene,config,47,46).initial==46);
 ties[1]=ties[0];CHECK(!valid_dcs(model));return 0;
}
