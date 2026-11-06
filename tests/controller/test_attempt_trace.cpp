// Trace serialization is outside the timed region and preserves full precision.
#include <sstream>
#include <string>
#include "check.hpp"
#include "harness/attempt_trace.hpp"

int main() {
    using namespace stitch;
    using namespace stitch::harness;
    RunRecord run;
    run.run_id = "campaign";
    run.git_rev = "abc";
    run.pair_id = "pair";
    run.seed = 20261101;
    AttemptTrace trace;
    trace.count = 1;
    auto& a = trace.records[0];
    a.knobs = {0.5, 1000, 256, s6::BlendMode::Feather};
    a.previous_R = 128;
    a.eligible_R_old = 128;
    a.generation = 1;
    a.decision.action = controller::Action::ACCEPT;
    a.decision.reason = controller::Reason::SELF_CHECK_PASSED;
    a.homography.num_inliers = 30;
    a.homography.H[0] = 1.2345678901234567;
    std::ostringstream out;
    write_attempt_header(out);
    write_attempt_rows(out, run, trace);
    CHECK(out.str().find("20261101,0,0.5,1000,256,128,128,1") != std::string::npos);
    CHECK(out.str().find("ACCEPT,SELF_CHECK_PASSED") != std::string::npos);
    CHECK(out.str().find("1.2345678901234567") != std::string::npos);
    CHECK(trace.records.size() == 16);
    CHECK(std::string(reason_name(controller::Reason::BUDGET_EXHAUSTED)) == "BUDGET_EXHAUSTED");
}
