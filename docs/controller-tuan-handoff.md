# Bàn giao tích hợp Controller cho Tuấn

- Người chuẩn bị: Đế; người tích hợp S1 và harness: Tuấn.
- Cập nhật 30/10 (Thứ 6), năm 2026; deadline Phase 2: 31/10 (Thứ 7).
- Nguồn chuẩn cho giao diện và điều kiện reuse: [Controller Contract v1](controller-contract.md). Bằng chứng module và test: [bàn giao triển khai](p2-de-controller-implementation.md) và [kiểm chứng T4](p2-de-t4-validation.md).

## Điểm nối trong harness

1. Link target `stitch_controller` từ `src/controller/CMakeLists.txt` vào adaptive harness. Root CMake, S1 và harness thuộc phần tích hợp của Tuấn; nhánh của Đế chưa sửa các file này.
2. Trước vùng đo, tạo policy và ladder tường minh, cấp workspace đủ cho mọi cấu hình có thể được chọn; gọi `configure(policy)` rồi `initialize(ladder, count)`. Kiểm tra `Status::OK` sau từng lời gọi. Mảng rule, self-check và ladder được Controller mượn; giữ chúng sống và bất biến trong run. Lấy `probe_scale()` từ chính policy đã configure, không đặt giá trị mặc định khác trong harness.
3. Với mỗi pair/repeat, kết thúc GPU work của run trước, gọi `reset_pair()` và reset cache S5 riêng. S1 tạo `SceneSignals` cho hai ảnh ở đúng `probe_scale`: `ScaleInfo` theo `make_scale_info`, gradient GRAY8 theo công thức ở mục 4.1 của contract. Chạy probe một lần; chỉ đọc pinned output và gọi `begin_pair` sau khi launch, copy và synchronization thành công. Tính cả probe/copy/synchronization trong end-to-end latency.
4. Thực thi S1–S5 theo `Knobs` từ `begin_pair`. Sau completion S5, điền `Feedback.executed` bằng đúng cấu hình đã chạy, phân biệt execution status với `HomographyResult.status`, rồi gọi `observe`. Nếu API trả lỗi, dừng run và không đọc `Decision`. Với `ACCEPT`, gửi H vừa được chấp nhận và B không đổi sang S6; với `RETRY`, chạy lại theo `Decision.next`; với `REJECT`, kết thúc run. Harness không tự chọn vị trí ladder hoặc fallback.
5. Chỉ nối tiếp R khi S/K, input, match arrays, counts, capacities, ScaleInfo, seed, threshold, batch size, refinement steps, stream, workspace và generation đều còn đúng, và `previous_hypotheses` bằng R đã hoàn thành. Nếu thiếu bằng chứng về một điều kiện, restart S5. R giảm cũng restart S5; K đổi chạy lại S2–S5 và đổi generation; S đổi chạy lại S1–S5 và đổi generation. Pair/repeat mới hoặc ghi lại input phải reset cache, đổi generation ngay cả khi địa chỉ buffer giữ nguyên. `plan_retry` mô tả dependency, không chứng nhận cache GPU hợp lệ.

## Kiểm tra khi Tuấn tích hợp

- Build độc lập: `cmake -S src/controller -B build/controller -DCMAKE_BUILD_TYPE=Release`, `cmake --build build/controller -j2`, `ctest --test-dir build/controller --output-on-failure`. Bằng chứng hiện tại: host 2/2 PASS; Controller–S5 trên Colab Tesla T4 5/5 PASS, sanitizer không báo lỗi hoặc hazard. Những kết quả này chưa chứng nhận adaptive harness của Tuấn.
- Kiểm tra harness bằng một pair có `ACCEPT`, một pair có `RETRY` R-only, một pair đổi K/S và một `REJECT`; xác nhận S1 probe đúng một lần, tín hiệu đã ready trước `begin_pair`, đường S6 chỉ chạy sau `ACCEPT`, và cache/generation theo bảng mục 6.3 của contract.
- Ghi terminal run, từng attempt và lý do retry bằng cùng định danh pair/config/repeat/seed, giữ timing boundary của fixed grid và canonical QoR evaluator. Đối chiếu output với [hướng dẫn Go/No-Go](sweep-gonogo-guide.md) trước khi dùng cho paper.

Numerical policy chưa khóa. Campaign ảnh thật đang kiểm tra gate Oracle trên development; chưa có số đo adaptive harness và chưa yêu cầu benchmark lớn trên synthetic pilot. Tài liệu này là gói bàn giao để Tuấn tích hợp, chưa phải xác nhận Tuấn đã nhận hoặc đã tích hợp.
