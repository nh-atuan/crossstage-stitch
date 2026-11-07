# Quyết định: Controller v5 và unified Run all W1–W5

- **Task:** triển khai và chuẩn bị kiểm chứng Controller v5 theo brief W1–W5 của Đế.
- **Người quyết định:** Đế.
- **Ngày ghi nhận:** 07/11 (Thứ 7), năm 2026.
- **Phase:** Phase 3, deadline chung 07/11 (Thứ 7).
- **Trạng thái:** implementation nghiên cứu, chưa có benchmark T4 mới; default không đổi.

## 1. Bối cảnh

Repo ban đầu có v4 pooled, score ablation P0–P5 và diagnostics A–E, chưa có selector/gate/staged retry v5. Đế cho phép thêm model/feature và yêu cầu ablation, C++/CUDA thật cùng một notebook Run all; Đế tự chạy Colab. Brief cũng ủy quyền push code sau kiểm chứng local để notebook clone được source mới.

Tái sử dụng grid cũ có hash thay vì sweep mới. Final baseline P0 là 64 cây depth 2; archived outer models dùng capacity riêng theo fold, được giữ đúng provenance ở W1. W2–W4 so sánh cùng capacity 64/depth 2. Evaluation 50 pair đã xem không được dùng chọn model/threshold hay gọi là blind test.

## 2. Phương án chọn

- Schema `controller-policy-v5` bổ sung model views và cost matrix; nhánh legacy/v4 không đổi default. C++ scalar tree inference dùng lại exporter repo, không gọi Python trong runtime.
- Separate reject target của đúng frozen gate, expected attempt/gate/S6/fallback cost; profile T4 có giới hạn, shrink khi support ít. S6 không nằm trong incremental matrix.
- Quality Gate có binary controls, three violation heads và joint-risk. Candidate H/photo/spatial có thật trước gate; Reference H chỉ ở offline labels.
- Shared CPU/CUDA spatial definition cho 16 features, group flags và parity tests; lấy CPU spatial observations từ H cũ và ảnh đã kiểm hash để tránh đo lại S1–S5 toàn grid.
- Staged rerank dựa trên feedback thật, catalogue tiến về sau và max 4 attempts; harness quản lý cache identity và continuation như contract cũ.
- CPU grouped CV/nested learning curve, checkpoint model; GPU chỉ build/test/profile/E2E. Model selection có bước development replay tính measured probe costs trước freeze W5; không dùng evaluation tuning.

## 3. Trade-off và phương án phụ

| Lựa chọn | Lợi ích | Giới hạn/phương án phụ |
|---|---|---|
| Reuse fixed grid, CPU spatial | Giảm GPU quota và giữ nhãn/provenance | Timing cũ là prior; W5 phải đo lại cùng source. Không phải replay online achievement |
| Union bound + joint head | Không giả independence | Có thể quá conservative; chỉ benchmark mới cho biết trade-off |
| Matrix shrink theo transition | Có measured telemetry, không dùng một hằng số retry | Không đo đủ mọi source/target. Unsupported class không được nhận discount mặc định |
| Bounded staged retry | Dùng được R/K/S reuse và feedback | Planner một bước, không chứng minh tối ưu toàn chuỗi; extra attempt có thể làm chậm |
| Chưa expansion/fresh holdout | Không dùng sai cohort hoặc tạo dữ liệu giả | Báo NO_NEW_TRAINING_DATA/PARTIAL/NOT_QUALIFIED; chưa tuyên bố generalization |

PRE2 correspondence probe tùy chọn chưa implement, có status rõ. V5_COMBINED hiện bằng STAGED, giữ cả hai như identity control. Không dùng kết quả âm để nới QoR hoặc tự đổi default.

## 4. Fallback

Thiếu input: giữ danh sách path/hash cần bổ sung, tiếp tục các phần độc lập và xuất ZIP PARTIAL. Thiếu/sai T4 hoặc CUDA/test failure: không đo W5. Checkpoint fingerprint gồm code/data/split/QoR/model/package/compiler/GPU; frozen inference costs và policies không đổi khi resume policy batches. Không tự launch GPU sweep bổ sung.

## 5. Kiểm chứng

Local có Python/g++ và đã cài CMake vào virtualenv kiểm chứng. Host CTest, Python unit/smoke, notebook kernel/auth/schema, grouped split, cost/selector/model parity, spatial guards, failure export và policy-freeze resume đều được kiểm tra. W1/W2 và phần W3 không cần ảnh được chạy trên grid thật bằng CPU. W4 và toàn luồng export/inference dùng TEST_FIXTURE; không phải GPU benchmark.

Chưa có nvcc/T4 local và không điều khiển Colab. CUDA spatial parity, S5/harness integration cùng native ablation phải do notebook chạy trên T4. Xem [runbook](../controller-v5-runbook.md), [implementation](../controller-v5-implementation.md), [experiment matrix](../controller-v5-experiment-matrix.md).

## 6. Nguồn

Dùng code và artifact của repo, [Controller contract](../stages/controller-contract.md), [hồ sơ Controller](../reports/controller-investigation.md), các định nghĩa metric/XGBoost đã dẫn nguồn trong modules cũ. Không sao chép số đo ngoài repo hoặc code runtime bên ngoài. Rà văn bản theo [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), đọc ngày 07/11 (Thứ 7), năm 2026.
