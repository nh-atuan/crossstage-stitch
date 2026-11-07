#include "controller/controller.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace stitch::controller {
bool valid_v5_homography(const HomographyResult& h) noexcept {
    for(const double* matrix:{h.H,h.H_s}) {
        for(int i=0;i<9;++i)if(!std::isfinite(matrix[i]))return false;
        const auto* a=matrix;
        const double d=a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);
        if(!std::isfinite(d) || std::abs(d)<1e-18)return false;
    }
    return true;
}
bool valid_v5(const V5Models& m,std::size_t count) noexcept {
    if(!count || count>64 || !m.incremental_ms || m.costs_count!=count*count || m.max_attempts<2 || m.max_attempts>16 ||
        m.relation_mask>7 || m.spatial_groups>3 || (m.spatial && !m.spatial_groups) || !std::isfinite(m.risk_threshold) || m.risk_threshold<0 || m.risk_threshold>1 ||
        !std::isfinite(m.pre_threshold) || m.pre_threshold<0 || m.pre_threshold>1 ||
        !std::isfinite(m.gate_ms) || m.gate_ms<0 || !std::isfinite(m.common_ms) || m.common_ms<0 ||
        !valid_ensemble(m.pre,39,false) || !valid_ensemble(m.attempt,39,true) ||
        !valid_ensemble(m.s6,39,true) || !valid_ensemble(m.reject,39,false) || !valid_ensemble(m.future,45,false)) return false;
    for(const auto& model:m.failures) if(!valid_ensemble(model,93,false)) return false;
    const auto uses=[](const EnsembleView& e,int begin,int end) {
        for(std::size_t j=0;j<e.node_count;++j)if(e.nodes[j].feature>=begin && e.nodes[j].feature<end)return true;
        return false;
    };
    if(!m.pair_ncc && m.new_selector && (uses(m.pre,33,39) || uses(m.attempt,33,39) || uses(m.s6,33,39) || uses(m.reject,33,39))) return false;
    if(!m.pair_ncc && m.staged && uses(m.future,33,39))return false;
    if(m.new_gate)for(const auto& e:m.failures) {
        if((!m.pair_ncc && uses(e,57,63)) || (!(m.relation_mask&2) && uses(e,63,67)) ||
            (!(m.relation_mask&1) && uses(e,67,71)) || (!(m.relation_mask&4) && uses(e,71,77)))return false;
        if(!m.spatial && uses(e,77,93))return false;
        if(m.spatial && (!(m.spatial_groups&1) && (uses(e,77,87) || uses(e,91,93))))return false;
        if(m.spatial && !(m.spatial_groups&2) && uses(e,87,91))return false;
    }
    for(std::size_t i=0;i<m.costs_count;++i) if(!std::isfinite(m.incremental_ms[i]) || m.incremental_ms[i]<0) return false;
    return true;
}
void Controller::v5_config(std::size_t i) noexcept {
    pooled_context_[30]=ladder_[i].S;pooled_context_[31]=ladder_[i].K/2000.;pooled_context_[32]=ladder_[i].R/1024.;
}
std::size_t Controller::v5_select(const Feedback* feedback) noexcept {
    const auto& m=*policy_.v5;const auto ref=policy_.pooled->reference_index;
    const auto origin=ladder_position_;v5_config(ref);
    const double reference_s6=predict_ensemble(m.s6,pooled_context_.data());
    const double common=feedback?0.:m.common_ms;
    double best=common+(feedback?m.incremental_ms[origin*count_+ref]:predict_ensemble(m.attempt,pooled_context_.data()))+reference_s6;
    std::size_t selected=ref;V5Telemetry chosen{};
    chosen.pre=chosen.reject=chosen.uncertainty=std::numeric_limits<double>::quiet_NaN();
    chosen.score_ms=best;chosen.common_ms=common;chosen.attempt_ms=best-common-reference_s6;chosen.accept_s6_ms=reference_s6;
    if(feedback && attempts_>=m.max_attempts-1) {v5_config(ref);telemetry_=chosen;return ref;}
    for(std::size_t j=0;j<ref;++j) {
        if(feedback && (j<=origin || (ladder_[j].S==feedback->executed.S && ladder_[j].K==feedback->executed.K &&
            (ladder_[j].R<=feedback->executed.R || feedback->homography.num_matches<policy_.self_checks[j].min_inliers)))) continue;
        v5_config(j);const double pre=predict_ensemble(m.pre,pooled_context_.data());
        if(!std::isfinite(pre) || pre<m.pre_threshold) continue;
        double reject=predict_ensemble(m.reject,pooled_context_.data());
        if(feedback) {
            std::array<double,45> x{};std::copy(pooled_context_.begin(),pooled_context_.end(),x.begin());
            const auto& h=feedback->homography;
            x[39]=h.num_matches/2000.;x[40]=h.num_inliers/2000.;x[41]=h.inlier_ratio;
            x[42]=h.inlier_coverage;x[43]=h.reproj_residual/32.;x[44]=feedback->executed.R/1024.;
            const double future=predict_ensemble(m.future,x.data());
            if(std::isfinite(future)) reject=std::max(reject,1.-future);
        }
        const double attempt=feedback?m.incremental_ms[origin*count_+j]:predict_ensemble(m.attempt,pooled_context_.data());
        const double s6=predict_ensemble(m.s6,pooled_context_.data()),fallback=m.incremental_ms[j*count_+ref];
        const double score=expected_cost(common,attempt,m.gate_ms,reject,s6,fallback,reference_s6);
        if(std::isfinite(score) && score<best) {
            best=score;selected=j;chosen={};chosen.pre=pre;chosen.reject=reject;chosen.attempt_ms=attempt;
            chosen.gate_ms=m.gate_ms;chosen.accept_s6_ms=(1.-reject)*s6;chosen.fallback_ms=reject*(fallback+reference_s6);
            chosen.common_ms=common;chosen.score_ms=score;chosen.uncertainty=reject*(1.-reject);
        }
    }
    v5_config(selected);telemetry_=chosen;return selected;
}
bool Controller::v5_gate(const Feedback& feedback) noexcept {
    const auto& m=*policy_.v5;
    if(!feedback.features_present || (m.spatial && !feedback.spatial_present)) return false;
    std::array<double,93> x{};
    std::copy_n(pooled_context_.begin(),33,x.begin());std::copy_n(feedback.features.begin(),24,x.begin()+33);
    std::copy_n(pooled_context_.begin()+33,6,x.begin()+57);std::copy_n(feedback.features.begin()+24,14,x.begin()+63);
    if(m.spatial) std::copy(feedback.spatial.begin(),feedback.spatial.end(),x.begin()+77);
    double p[4]{};for(int i=0;i<4;++i) {p[i]=predict_ensemble(m.failures[i],x.data());if(!std::isfinite(p[i])) return false;}
    telemetry_.fail_ssim=m.margin_heads?p[0]:std::numeric_limits<double>::quiet_NaN();
    telemetry_.fail_psnr=m.margin_heads?p[1]:std::numeric_limits<double>::quiet_NaN();
    telemetry_.fail_coverage=m.margin_heads?p[2]:std::numeric_limits<double>::quiet_NaN();
    // Union bound requires no independence assumption; not a calibrated guarantee.
    telemetry_.risk_joint=m.margin_heads?std::max(p[3],std::min(1.,p[0]+p[1]+p[2])):p[3];
    telemetry_.failure_code=m.margin_heads?1+static_cast<int>(std::max_element(p,p+3)-p):4;
    telemetry_.gate_evaluated=true;
    gate_telemetry_=telemetry_;
    return telemetry_.risk_joint<=m.risk_threshold;
}
} // namespace stitch::controller
