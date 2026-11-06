#include <cmath>
#include <limits>
#include "check.hpp"
#include "controller/ensemble.hpp"
using namespace stitch::controller;
int main() {
    BoostNode nodes[]={{0,1,2,.5f,0.f},{-1,-1,-1,0.f,-2.f},{-1,-1,-1,0.f,2.f}};
    std::size_t roots[]={0};
    EnsembleView e{nodes,3,roots,1,1,0.f,1.,0.,false};
    CHECK(valid_ensemble(e,1,false));
    double x[]={.5};CHECK(std::abs(predict_ensemble(e,x)-1./(1.+std::exp(-2.)))<1e-15);
    x[0]=std::nextafter(.5,0.); // Rounds to .5f: XGBoost follows the right child.
    CHECK(predict_ensemble(e,x)>.8);
    x[0]=.4;CHECK(predict_ensemble(e,x)<.2);
    x[0]=std::numeric_limits<double>::quiet_NaN();CHECK(std::isnan(predict_ensemble(e,x)));
    nodes[0].left=0;CHECK(!valid_ensemble(e,1,false));nodes[0].left=1;
    e.exponential=true;x[0]=.4;CHECK(std::abs(predict_ensemble(e,x)-std::exp(-2.))<1e-15);
    CHECK(!valid_ensemble(e,1,false));
    std::puts("Float32 ensemble splits, transforms and structure validation passed");
}
