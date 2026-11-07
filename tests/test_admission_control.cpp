// CPU-only overflow/capacity tests, including canvases wider than 4096 pixels.
#include "check.hpp"
#include "common/admission_control.hpp"
#include <limits>
using namespace stitch::admission;
template<class Function> void rejects(Function function, const char* expected) {
    bool rejected=false;
    try { function(); } catch (const Error& e) { rejected=e.code==expected; }
    CHECK(rejected);
}
int main() {
    CHECK(image_bytes(7680,2160,3)==static_cast<std::size_t>(7680*3+64)*2160);
    CHECK(image_bytes(30000,2,3)==180128);  // Byte pitch exceeds uint16 range.
    dimensions_fit(3840,2160,3840,2160,3840,2160);
    rejects([] { dimensions_fit(3840,2160,3840,2160,2048,2048); },"ERR_CAPACITY");
    rejects([] { image_bytes(std::numeric_limits<int>::max(),2,3); },"ERR_CANVAS_OVERFLOW");
    rejects([] { multiply(std::numeric_limits<std::size_t>::max(),2); },"ERR_CANVAS_OVERFLOW");
    rejects([] { add(std::numeric_limits<std::size_t>::max(),1); },"ERR_CANVAS_OVERFLOW");
    rejects([] { fit(901,1000,100,"ERR_DEVICE_OOM"); },"ERR_DEVICE_OOM");
    rejects([] { fit(0,100,101,"ERR_HOST_OOM"); },"ERR_HOST_OOM");
    fit(900,1000,100,"ERR_DEVICE_OOM");
    return 0;
}
