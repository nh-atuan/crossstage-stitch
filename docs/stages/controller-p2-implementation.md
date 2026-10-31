# Bàn giao Controller Phase 2 của Đế

- Người phụ trách: Đế.
- Cập nhật 30/10 (Thứ 6), năm 2026; deadline Phase 2 là 31/10 (Thứ 7).
- Nhánh: `de/p2-controller`.
- Phạm vi đã kiểm chứng: Controller host, retry plan và Controller–S5 trên Colab Tesla T4; chưa nghiệm thu adaptive end-to-end speedup.

## 1. Tiến độ theo kế hoạch

| Công việc | Trạng thái và bằng chứng |
|---|---|
| Contract Controller v1 | Đã có trước task này; giữ nguyên API per-pair và shared stage types |
| Policy và Controller | Đã triển khai rule classifier, kiểm policy/ladder, state machine, threshold checks và bounded escalation; numerical policy chưa khóa |
| Feedback/reuse S5 | Host tests và Controller–S5 CUDA test pass trên T4; split/full H/mask/tín hiệu khớp; stale reuse bị từ chối; sanitizer sạch |
| Ghép S1/harness và pilot | Đã chuẩn bị giao diện và hướng dẫn tích hợp cho Tuấn; chưa chạy Controller adaptive end-to-end |
| Paper | Đã bổ sung S4–S5 và Section 4 theo code đã test; build 9 trang, metadata ẩn danh và audit PASS trên Colab; chưa có adaptive results |

## 2. Module và API setup

| File | Vai trò |
|---|---|
| `src/controller/controller.hpp`, `.cpp` | API/lifecycle, selection và ACCEPT/RETRY/REJECT |
| `src/controller/policy.hpp`, `.cpp` | Schema và validation cấu hình tường minh |
| `src/controller/validation.cpp` | Validity scene/feedback, self-check theo status và đơn vị metric |
| `src/controller/reuse.hpp`, `.cpp` | Phân loại phần pipeline cần chạy lại khi RETRY |
| `tests/controller/test_controller.cpp` | Contract, failure handling và allocation checks |
| `tests/controller/controller_trace.cpp` | Trace host tổng hợp, không ghi số đo GPU/QoR |
| `tests/controller/test_controller_s5.cu` | Controller nhận output S5 CUDA thật; R continuation và invalid reuse |

Xem [quyết định kỹ thuật](../decisions/p2-de-controller.md) về phương án, trade-off, lifetime và calibration. Setup bổ sung của module:

```cpp
stitch::controller::Controller controller;
// policy points to caller-owned, immutable SceneRule and SelfCheck arrays.
// ladder is caller-owned and has one entry per SelfCheck.
auto status = controller.configure(policy);
if (status == stitch::Status::OK)
    status = controller.initialize(ladder, count);
// Stop on any setup error. Use controller.probe_scale() for the S1 probe.
```

`configure` phải chạy trước `initialize`; trước khi configure thành công, `probe_scale()` trả NaN. Configure thành công xóa binding ladder cũ; initialize lại trước pair tiếp theo. Lỗi setup không đổi cấu hình đang có. Không thay policy/ladder trong lúc đang chờ feedback. Policy object được sao chép theo giá trị; các array bên trong và ladder được mượn, phải sống và bất biến tới khi thay setup thành công hoặc hủy Controller.

Khi `observe` trả status khác OK, không đọc Decision. Khi status OK, REJECT/EXECUTION_ERROR khác với API input error. Output không được alias input hoặc array cấu hình. Một Controller chỉ có một pair đang hoạt động.

## 3. Build và test độc lập

Trên môi trường có CMake và C++17:

```bash
cmake -S src/controller -B build/controller -DCMAKE_BUILD_TYPE=Release
cmake --build build/controller -j2
ctest --test-dir build/controller --output-on-failure
./build/controller/controller_trace
```

Kỳ vọng CTest có `test_controller` và `controller_trace`, cả hai PASS. Trace có nhãn `synthetic_fixture`, lần lượt RETRY từ degeneration (ứng viên nối tiếp S5), RETRY do thiếu match (đổi K, chạy lại S2) và ACCEPT. Không có CSV đo GPU hoặc kết luận QoR từ trace này.

Kết quả đã chạy trên Colab Tesla T4 ngày 30/10 (Thứ 6), CUDA 13.0, driver 580.82.07, build architecture 75:

| Kiểm tra | Kết quả |
|---|---|
| Controller host CTest | 2/2 PASS |
| Bộ tích hợp Controller–S5 CUDA | 5/5 PASS: Controller, trace, Controller–S5, S5 math, S5 CUDA; không SKIP |
| Memcheck/racecheck Controller–S5 và S5 CUDA | 0 errors, 0 hazards, 0 warnings |
| Root build và CTest | 12/12 PASS; không SKIP |

Host tests kiểm lifecycle, classifier, self-check, ACCEPT/RETRY/REJECT, exhaustion, invalid feedback và invalidation khi đổi S/K/R. Test 1.000 pair tổng hợp xác nhận không cấp phát heap trong các lời gọi per-pair. Test Controller–S5 dùng correspondences tổng hợp với seed 20261030 và output CUDA thật: ép self-check đầu thất bại, nối tiếp R từ 17 lên 257, kiểm H/H_s/mask/tín hiệu khớp chạy full 257, rồi từ chối stale generation bằng REJECT/EXECUTION_ERROR. Đây chưa phải kiểm chứng pipeline scene-aware từ ảnh.

Lệnh tái kiểm chứng integration trên T4:

```bash
cmake -S src/controller -B build/controller-gpu -DSTITCH_CONTROLLER_GPU_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build/controller-gpu -j2
ctest --test-dir build/controller-gpu --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/controller-gpu/test_controller_s5
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/controller-gpu/test_controller_s5
```

Receipt source và log kiểm chứng được giữ ở `results/de_p2_campaign_20261030/evidence/`, ngoài Git. Host build độc lập trên Windows cũng đã pass CTest 2/2; không thay thế bằng chứng T4 ở trên.

## 4. Điểm nối cho Tuấn

Root CMake và harness vẫn thuộc Tuấn. Để tích hợp library, thêm `add_subdirectory(src/controller)` và link `stitch_controller` vào adaptive harness. Có thể build độc lập module khi chưa chỉnh root.

Harness chuẩn bị policy/ladder/workspace trước vùng đo; kiểm setup; reset pair và cache S5; chạy probe A/B đúng một lần; kiểm launch/copy/completion rồi mới gọi `begin_pair`. Giữ nguyên readiness và ranh giới end-to-end của [contract](controller-contract.md). Probe cost không được bỏ khỏi thời gian adaptive.

Sau mỗi attempt, chuyển đúng executed knobs và feedback S5 hợp lệ cho `observe`. Chỉ ACCEPT mới gọi S6; RETRY chạy phần bị ảnh hưởng; REJECT kết thúc run. `plan_retry` là hướng dẫn dependency, không chứng nhận cache. Để extension, caller giữ nguyên toàn bộ điều kiện ở mục 6.3 của contract và dùng R đã thực sự hoàn tất; trường hợp không chứng minh được thì restart S5. Khi đổi pair/repeat hoặc ghi lại input, đổi generation và reset cache dù địa chỉ buffer giữ nguyên.

Chưa có thay đổi S1/harness trong nhánh này. Không sửa shared types hoặc `Status`.

## 5. Nghiệm thu sau tích hợp

Tuấn phụ trách S1 scene signals và adaptive harness theo [kế hoạch Phase 2](../plan.md). Đế bàn giao module và hỗ trợ nối feedback/reuse; numerical policy cần được cung cấp tường minh sau development review.

Khi harness sẵn sàng, kiểm đường ACCEPT, RETRY chỉ tăng R, RETRY đổi K/S và REJECT; xác nhận probe đúng một lần, output đã ready và cache/generation hợp lệ. Sau đó chạy pilot trên T4 với Fixed-Best, Oracle và Controller cùng timing boundary/QoR để đo overhead và speedup thực tế, gồm probe, synchronization và retry. Các test module hiện tại chưa thay thế lần nghiệm thu này.
