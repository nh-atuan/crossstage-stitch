// Experimental DCS-R1: raw preference scores, never probability thresholds.
#pragma once
#include <array>
#include <algorithm>
#include <limits>
#include "controller/ensemble.hpp"
namespace stitch::controller {
constexpr int dcs_features=48,dcs_candidates=48;
struct DcsModels {
    const EnsembleView* models{};std::size_t count{};
    const int* tie_order{};
    bool available{},pair_ncc{},interactions{},multiclass{},linear{};
    const double* weights{};double bias{};
};
struct DirectSelection {
    std::array<double,48> scores{};
    std::array<std::size_t,48> ranked{};
    std::size_t initial{};int reason{}; // 0 ranked, 1 unavailable/schema/catalog, 2 invalid features/output.
};
inline void dcs_input(const double* pre,double* x) noexcept {
    for(int i=0;i<39;++i)x[i]=static_cast<float>(pre[i]);
    x[39]=x[0]*x[1];x[40]=std::abs(x[0]-x[1]);x[41]=x[33]*x[38];x[42]=x[34]*x[38];
    x[43]=x[30]*x[31];x[44]=x[30]*x[32];x[45]=x[31]*x[32];x[46]=x[0]*x[30];x[47]=x[1]*x[30];
}
inline double dcs_raw(const EnsembleView& e,const double* x) noexcept {
    for(int f=0;f<e.features;++f)if(!std::isfinite(x[f]) || std::abs(x[f])>std::numeric_limits<float>::max())return std::numeric_limits<double>::quiet_NaN();
    float total=e.bias;
    for(std::size_t t=0;t<e.tree_count;++t){auto i=e.roots[t];for(int step=0;step<127;++step){const auto& n=e.nodes[i];
        if(n.feature==-1){total+=n.leaf;break;}i=static_cast<float>(x[n.feature])<n.threshold?n.left:n.right;}}
    return total;
}
inline bool valid_dcs(const DcsModels& m) noexcept {
    if(!m.available || !m.tie_order || (!m.linear && (!m.models || m.count!=(m.multiclass?48u:1u))))return false;
    if(m.linear){if(!m.weights || !std::isfinite(m.bias))return false;for(int i=0;i<48;++i)if(!std::isfinite(m.weights[i]) || (!m.interactions && i>=39 && m.weights[i]!=0) || (!m.pair_ncc && i>=33 && i<39 && m.weights[i]!=0))return false;}
    bool seen[48]{};
    for(int j=0;j<48;++j){if(m.tie_order[j]<0 || m.tie_order[j]>=48 || seen[m.tie_order[j]])return false;seen[m.tie_order[j]]=true;}
    for(std::size_t j=0;j<m.count;++j){if(!valid_ensemble(m.models[j],48,false))return false;
        for(std::size_t i=0;i<m.models[j].node_count;++i){int f=m.models[j].nodes[i].feature;
            if((!m.pair_ncc && f>=33 && f<=42 && f!=39 && f!=40) || (!m.interactions && f>=39))return false;}}
    return true;
}
// Caller supplies exact immutable config descriptors and a scene observed before S2.
inline DirectSelection select_direct(const DcsModels& model,const double* scene,const double* configs,std::size_t count,std::size_t ref) noexcept {
    DirectSelection out;out.initial=ref;out.reason=1;out.scores.fill(std::numeric_limits<double>::quiet_NaN());
    for(int j=0;j<48;++j)out.ranked[j]=j;
    if(count!=48 || ref>=count || !configs || !scene || !valid_dcs(model))return out;
    std::array<double,39> pre{};std::copy_n(scene,39,pre.begin());
    for(int j=0;j<48;++j){for(int k=0;k<3;++k)pre[30+k]=configs[j*3+k];double x[48];dcs_input(pre.data(),x);
        if(model.linear){float total=model.bias;for(int k=0;k<48;++k)total+=static_cast<float>(x[k])*static_cast<float>(model.weights[k]);out.scores[j]=total;}
        else out.scores[j]=dcs_raw(model.models[model.multiclass?j:0],x);
        if(!std::isfinite(out.scores[j])){out.reason=2;return out;}}
    std::sort(out.ranked.begin(),out.ranked.end(),[&](auto a,auto b){return out.scores[a]==out.scores[b]?model.tie_order[a]<model.tie_order[b]:out.scores[a]>out.scores[b];});
    out.initial=out.ranked[0];out.reason=0;return out;
}
} // namespace stitch::controller
