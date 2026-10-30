# Bàn giao Controller Phase 2 của Đế

- Người phụ trách: Đế.
- Cập nhật 30/10 (Thứ 6), năm 2026; deadline Phase 2 là 31/10 (Thứ 7).
- Nhánh: `de/p2-controller`.
- Phạm vi đã kiểm chứng: module Controller host và retry plan; chưa có nghiệm thu adaptive GPU hoặc speedup.

## 1. Tiến độ theo kế hoạch

| Công việc | Trạng thái và bằng chứng |
|---|---|
| Contract Controller v1 | Đã có trước task này; giữ nguyên API per-pair và shared stage types |
| Policy và Controller | Đã triển khai rule classifier, kiểm policy/ladder, state machine, threshold checks và bounded escalation; giá trị policy dùng trên ảnh thật còn cần calibration |
| Feedback/reuse S5 | Đã đối chiếu source/API, thêm helper và test host cho invalidation; test CUDA split/full có sẵn từ Phase 1, chưa chạy lại trong campaign này |
| Ghép S1/harness và pilot | Cần Tuấn tích hợp scene signals và adaptive execution theo contract; chưa có terminal adaptive runs |
| Paper | Draft Section 4 mô tả code host đã test và phần còn chờ; phần S4–S5 mô tả source có bằng chứng T4 từ Phase 1; chưa nghiệm thu PDF mới |
| QoR policy | Giữ trạng thái pilot; chưa khóa threshold bằng dữ liệu development mới, chưa dùng final để tune |

## 2. Module và API setup

| File | Vai trò |
|---|---|
| `src/controller/controller.hpp`, `.cpp` | API/lifecycle, selection và ACCEPT/RETRY/REJECT |
| `src/controller/policy.hpp`, `.cpp` | Schema và validation cấu hình tường minh |
| `src/controller/validation.cpp` | Validity scene/feedback, self-check theo status và đơn vị metric |
| `src/controller/reuse.hpp`, `.cpp` | Phân loại phần pipeline cần chạy lại khi RETRY |
| `tests/controller/test_controller.cpp` | Contract, failure handling và allocation checks |
| `tests/controller/controller_trace.cpp` | Trace host tổng hợp, không ghi số đo GPU/QoR |

Xem [quyết định kỹ thuật](decisions/p2-de-controller.md) về phương án, trade-off, lifetime và calibration. Setup bổ sung của module:

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

Kiểm chứng cục bộ Windows ngày 30/10 (Thứ 6): Zig 0.16.0 cung cấp Clang 21.1.0, CMake 4.4.4, Ninja 1.13.2; build độc lập ở `build/controller-verified`. CMake/CTest: 2/2 PASS. Build trực tiếp với `-std=c++17 -Wall -Wextra -Werror` và chạy `test_controller.exe`: PASS. Bộ test Python phục vụ pilot: 94 PASS bằng lệnh sau:

```bash
python -m pytest tests/python/test_qor_decisions.py tests/python/test_latency_qor.py tests/python/test_sweep_analysis.py tests/python/test_dataset_prep.py -q
```

Tooling chỉ nằm trong `build/`, không thêm dependency runtime hoặc đổi requirements của pipeline. Build đầu tiên của Zig phát cảnh báo nullability từ libc++ đi kèm khi tự tạo runtime; source Controller build với các warning flags trên không báo lỗi. Host tests không thay thế kiểm chứng Colab Tesla T4.

## 4. Điểm nối cho Tuấn

Root CMake và harness vẫn thuộc Tuấn. Để tích hợp library, thêm `add_subdirectory(src/controller)` và link `stitch_controller` vào adaptive harness. Có thể build độc lập module khi chưa chỉnh root.

Harness chuẩn bị policy/ladder/workspace trước vùng đo; kiểm setup; reset pair và cache S5; chạy probe A/B đúng một lần; kiểm launch/copy/completion rồi mới gọi `begin_pair`. Giữ nguyên readiness và ranh giới end-to-end của [contract](controller-contract.md). Probe cost không được bỏ khỏi thời gian adaptive.

Sau mỗi attempt, chuyển đúng executed knobs và feedback S5 hợp lệ cho `observe`. Chỉ ACCEPT mới gọi S6; RETRY chạy phần bị ảnh hưởng; REJECT kết thúc run. `plan_retry` là hướng dẫn dependency, không chứng nhận cache. Để extension, caller giữ nguyên toàn bộ điều kiện ở mục 6.3 của contract và dùng R đã thực sự hoàn tất; trường hợp không chứng minh được thì restart S5. Khi đổi pair/repeat hoặc ghi lại input, đổi generation và reset cache dù địa chỉ buffer giữ nguyên.

Chưa có thay đổi S1/harness trong nhánh này. Không sửa shared types hoặc `Status`.

## 5. Sweep, calibration và nghiệm thu còn lại

Đế yêu cầu chạy lại raw sweep theo [hướng dẫn của Nhóm](sweep-gonogo-guide.md). Lưu raw mới trong một thư mục campaign riêng để không ghi đè báo cáo Phase 1. Dataset preparer ở chế độ auto chỉ quét thư mục raw đã có; nếu không tìm được pairs thì sinh ảnh tổng hợp. Kết quả này phải được ghi đúng là synthetic pilot, không phải UDIS-D hoặc final evaluation.

Các bước còn cần evidence:

1. Kết nối Colab, xác nhận `nvidia-smi` là Tesla T4, ghi revision/toolchain, build và chạy CTest/Compute Sanitizer cho S4/S5.
2. Chuẩn bị 25 pairs, manifest và provenance; chạy đủ 48 cấu hình như guide để có `runs.csv`, `qor.csv`, configs, policy hash và metadata.
3. Trích scene features bằng đúng S1 probe, ghép labels offline từ raw; chọn rule/ladder/self-check trên development và kiểm tra ảnh/failures. Không suy ngưỡng self-check trực tiếp từ QoR tolerance.
4. Tích hợp adaptive harness; chạy Fixed-Best, Oracle và Controller cùng timing boundary và policy; xuất terminal runs cùng retry/reuse telemetry cho Chương.
5. Khóa policy sau development review; hoàn thiện Section 4 và kiểm PDF bằng `python paper/build.py`. Chưa có dữ liệu thì giữ trạng thái chưa nghiệm thu, không đưa số speedup vào paper.

Lệnh kiểm lại S4/S5 trên T4:

```bash
cmake -S tests/s4_s5 -B build/s4-s5 -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build/s4-s5 -j2
ctest --test-dir build/s4-s5 --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/s4-s5/test_s5_cuda
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/s4-s5/test_s5_cuda
```

Build phối hợp đặt runtime output ở `build/s4-s5`. Kết quả mong đợi: CUDA test không SKIP, exit code 0, không có lỗi sanitizer. Chỉ ghi PASS sau khi thực sự chạy.
