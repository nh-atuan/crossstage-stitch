# Quyết định: Controller theo ladder và policy tường minh

- **Task:** Controller, feedback S5 và reuse trong Phase 2.
- **Người yêu cầu:** Đế; triển khai theo yêu cầu thực hiện các task của Đế.
- **Ngày ghi nhận:** 30/10 (Thứ 6), năm 2026.
- **Phase:** Phase 2; deadline chung 31/10 (Thứ 7).
- **Trạng thái:** Đã triển khai logic host; chưa hiệu chỉnh policy trên raw sweep, chưa nghiệm thu integration hoặc latency trên Colab Tesla T4.

## 1. Bối cảnh

[Contract v1](../controller-contract.md) đã khóa API per-pair, thứ tự quyết định, readiness và trách nhiệm cache. Repo chưa có module Controller, scene signals S1 hoặc adaptive harness. Báo cáo Phase 1 là tổng hợp; bản checkout không chứa ảnh, `runs.csv` hoặc `qor.csv` để hiệu chỉnh classifier và self-check.

Controller cần chạy độc lập bằng feedback tổng hợp trước khi Tuấn tích hợp S1/harness. Mọi số đo GPU, policy dùng cho ảnh thật và kết luận speedup vẫn cần campaign riêng.

## 2. Phương án chọn

Module C++17 ở `src/controller/` giữ API `initialize`, `begin_pair`, `observe`, `reset_pair` của v1. Thêm bước setup `configure(Policy)` và accessor `probe_scale()` trước `initialize`; Tuấn cần gọi bước này khi tích hợp. Không đổi `src/common/`, S1, S5 hoặc harness.

Policy chứa scale probe, danh sách rule và một bộ self-check cho mỗi vị trí ladder. Các tham số không có giá trị mặc định. `configure` mượn array bất biến, từ chối policy thiếu rule phủ miền `[0,1]`, ngưỡng không hữu hạn hoặc count không hợp lệ. Setup thành công xóa binding ladder cũ; caller phải initialize lại. Setup thất bại giữ nguyên state. Không được configure/initialize giữa một attempt đang chờ feedback.

Classifier dùng `g = min(G_A, G_B)`; rule xét theo lower bound giảm dần, lấy rule đầu tiên có `g >= min_pair_gradient`. Rule cuối có bound bằng 0. Mỗi rule chỉ định vị trí khởi đầu; không suy ra ngưỡng từ dataset final hoặc từ nhãn QoR online. Dùng minimum giữ thông tin ảnh có gradient thấp hơn, nhưng một scalar không biểu diễn được overlap, parallax hoặc độ lặp texture. Quan hệ gradient với độ khó và các bound phải được kiểm tra trên development data trước khi dùng cho benchmark.

Sau mỗi S5, xét execution status rồi validity của payload. Các field không có hiệu lực ở failure status không được đọc. Với model OK, thứ tự self-check là count, ratio, bounding-box inlier coverage, RMS residual. Ngưỡng residual được khai báo theo pixel S1 cho từng cấu hình. Ratio/count consistency dùng sai số máy FP64; đây là kiểm schema, không phải nới tolerance chất lượng.

RETRY tiến tới vị trí kế tiếp. Nếu có dưới 4 match, bỏ các cấu hình phía sau giữ nguyên S/K. B giữ nguyên trong run. Hết ladder luôn REJECT; cấu hình cuối vẫn phải pass self-check. Execution error kết thúc run. API gọi sai không đổi output hoặc state.

`plan_retry` là helper host thể hiện bảng invalidation của contract: đổi S chạy lại S1–S5; đổi K chạy lại S2–S5; R giảm restart S5; chỉ tăng R mới là ứng viên nối tiếp raw search. Helper không kiểm tra allocation, buffer, seed, stream, completion hoặc generation thật. Harness phải chứng minh toàn bộ điều kiện cache trước khi dùng `previous_hypotheses` được gợi ý; thiếu bằng chứng thì restart S5.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Hạn chế và quyết định |
|---|---|---|
| Rule gradient và ladder tường minh | Ít host work, không allocation per-pair, có thể kiểm chứng bằng input cố định | Cần development data để hiệu chỉnh; chọn làm cấu trúc policy ban đầu |
| Decision tree được học offline | Có thể dùng thêm feature và quan hệ phi tuyến | Chưa có raw/feature labels đủ để huấn luyện và validation; chỉ xét sau khi có dữ liệu |
| Cấu hình đầu cố định | Dễ tích hợp, dùng làm baseline/ablation | Không đáp ứng scene-aware selection; không tự dùng khi thiếu probe |
| Nhúng threshold từ fixture hoặc tổng hợp Go/No-Go | Có thể tạo demo chạy ngay | Không có bằng chứng calibration; chỉ giữ các số này trong test tổng hợp |
| Đưa cache GPU vào Controller | Có thể gom ownership | Trái contract host-only và ownership harness; không chọn |

Khởi tạo ladder kiểm trùng bằng O(n²), ngoài vùng per-pair. Chọn đầu tốn O(số rule); mỗi feedback kiểm số field cố định rồi duyệt phần ladder cần bỏ. Các array do caller sở hữu, nên caller phải giữ lifetime và tính bất biến; module không sao chép cấu hình bằng heap.

## 4. Fallback

Không có policy, probe lỗi hoặc scene không hợp lệ: caller kết thúc lượt tích hợp lỗi. Không thay bằng cấu hình cố định ngầm. Nếu policy heuristic không đạt QoR/latency trên development, điều chỉnh bằng campaign development có provenance hoặc báo kết quả âm; giữ final chưa dùng để tune. Không tự thay quality policy offline.

## 5. Kiểm chứng

Test ở `tests/controller/` kiểm lifecycle, selection ở biên rule, metadata probe, status/payload, thứ tự reason, ladder không đơn điệu về budget, B cố định, budget exhaustion và retry invalidation. Một test chạy 1.000 pair tổng hợp và đếm C++ heap allocations trong các lời gọi per-pair. Trace riêng thể hiện S5 degeneration → R extension → thiếu match → đổi K → ACCEPT. Fixture chỉ có dữ liệu host, không chứng nhận model, cache hoặc QoR GPU.

Hướng dẫn build, bằng chứng và các phần còn thiếu ở [bàn giao Controller](../p2-de-controller-implementation.md). Chạy sweep theo [hướng dẫn Nhóm](../sweep-gonogo-guide.md), lưu raw mới ngoài Git; ảnh tổng hợp phải được ghi đúng loại bằng chứng. Chỉ khóa policy số sau khi có CSV, scene features, ảnh và kiểm tra trực quan.

## 6. Nguồn

Logic dựa trên contract và code S5 của repo. Khi biên tập đã đọc [Wikipedia:Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), ngày 30/10 (Thứ 6), năm 2026; rà lại để bỏ khẳng định chưa có bằng chứng và lời mô tả chung chung.
