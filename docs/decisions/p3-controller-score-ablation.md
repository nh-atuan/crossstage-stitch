# Quyết định: thử scoring có tính reuse cho Controller v4

- **Task:** Cải thiện nhẹ scoring Controller; notebook T4 do Đế tự chạy.
- **Người quyết định:** Đế, theo yêu cầu thực hiện brief và sáu policy P0–P5.
- **Ngày ghi nhận:** 06/11 (Thứ 6), năm 2026.
- **Phase:** Phase 3; deadline chung 07/11 (Thứ 7).
- **Trạng thái:** Code/tooling đã qua kiểm tra local; chưa có benchmark T4 của thay đổi này.

## 1. Bối cảnh

C++ `begin_pair` và Python `choose` dùng fixed E2E của Reference làm penalty
fallback cho mọi candidate. Fixed E2E đã gồm S6. Harness chỉ render S6 khi
ACCEPT; `plan_retry` phân biệt đổi S, đổi K và tăng R. State/cache identity,
generation và kiểm stale reuse thuộc harness/S5, không thuộc scoring.

Brief mô tả 33 pre-feature và 63 post-feature hoạt động của photo. ABI model
export thực tế có 39/77 cột, padding các nhóm không dùng; giữ nguyên ABI này.
Model transition v3 chưa chứng nhận tương thích source/kernel/features v4;
không dùng coefficient cũ hoặc median A2 làm giá retry chung.

## 2. Phương án chọn

Giữ nguyên score cũ khi policy không khai báo mode hoặc có `score_mode=old`.
P0–P2 dùng mode cũ, P3–P5 dùng `reuse_approx`; mỗi nhóm có ngưỡng 0.85/0.70/0.50.
Giữ XGBoost, feature, self-check, post-gate, QoR và fallback tối đa hai attempt.
Không retrain hoặc thay default. Chỉ thêm array tỷ lệ fallback vào policy
thử nghiệm, không thêm inference hoặc kernel vào runtime.

Dùng binary mới đo Reference trên calibration đã khóa, hai warmup/mười repeat.
Mỗi repeat có `fraction=(t_e2e-reusable_stage_ms)/t_e2e`; median theo pair,
rồi p95 theo pair với quantile linear. R-only trừ S1–S4; đổi K trừ S1;
đổi S dùng fraction 1. Giữ full S5, S6, host và sync; không giả định S5
latency tuyến tính theo R. Profile execution fail hoặc stage sum không hợp
lệ dùng fraction 1. Khoá profile/policy trước khi mở campaign đánh giá.

Score mới: `L_hat(c)+(1-p_pre)*L_hat(ref)*fraction+extra_ms`. Giữ chi phí
candidate đầu, `extra_ms` và so sánh với Reference như cũ. `1-p_pre` là
failure proxy, không phải xác suất reject của gate. `L_hat(c)` gồm S1–S6
và host overhead, chưa có probe/gate adaptive; `extra_ms` giữ dự toán gate
cũ, không cộng probe vào ranking vì là chi phí chung. Khi reject, S6 candidate
chưa chạy nhưng score vẫn giữ nó; đây là xấp xỉ bảo thủ, không phải phân rã
expected E2E chính xác. Probe, gate, retry và S6 thật nằm trong phép đo E2E.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Hạn chế | Quyết định |
|---|---|---|---|
| Full Reference fallback | Đối chứng giữ nguyên | Bỏ qua stage reuse | Giữ P0–P2 |
| Trừ stage bằng profile calibration hiện tại | Không retrain; R/K/S khác nhau; ít overhead | Fraction theo cohort, không theo pair; cache invalid có thể mất discount; chưa hiệu chỉnh chi phí host theo transition | Thử P3–P5, ghi uncertainty |
| Dùng model transition v3 hoặc A2 median | Có số đo cũ | Khác source/domain; A2 chỉ R-only | Không dùng |
| Đo/fit model transition mới theo pair | Có thể sát hơn | Mở rộng mô hình và protocol | Ngoài vòng thử này |

## 4. Fallback

Unknown mode, NaN, array sai hoặc scale discount sai bị từ chối trước đo.
Thiếu checkpoint/model/ảnh dừng preflight. Không thay bằng model giả hoặc
CPU timing. Không có Oracle sweep phù hợp thì xuất UNAVAILABLE, không tự
chạy full sweep. Source, binary, input hoặc GPU identity đổi thì không resume
campaign cũ. Kết quả âm vẫn được giữ; không tự chọn policy thắng hoặc đổi gate.

## 5. Kiểm chứng và bàn giao

Tests kiểm score cũ, threshold/tie, transition fractions, policy bất biến,
Python/C++ parity, checkpoint và xác thực Secret. Notebook chứa build/test,
smoke và FULL; giữ raw CSV, attempt/feature traces, canonical QoR, paired
latency và ZIP hash. Holdout v4 đã xem chỉ dùng diagnostic cho cả sáu variant,
không tune trên UDIS-D test hoặc gọi đây là đánh giá độc lập. Agent chỉ chạy
kiểm tra local, không dùng MCP Colab hoặc cloud GPU.

Kiểm chứng local ngày 06/11 (Thứ 6):

- Build Release host bằng CMake: thành công; CTest 12/12 PASS.
- `python -m pytest tests/python -q -rs -p no:cacheprovider`: 486 PASS, 4 SKIP do thiếu OR-Tools/Qiskit ở các test quantum ngoài phạm vi; không tắt test.
- 12 test tập trung cho score/notebook PASS, gồm parity thật qua probe C++ và fixture I/O tổng hợp cho checkpoint/report; fixture không phải số đo GPU.
- Notebook qua kiểm schema nbformat, compile mọi code cell, output trống và thực thi bootstrap trong test cách ly với Secret/network giả; chưa chạy Run all trên Colab.
- `git diff --check`: không có lỗi. Không có nvcc/T4 local; chưa build CUDA, chạy integration GPU hoặc đối chiếu artifact v4 thật vì checkpoint không có trong checkout hiện tại.

## 6. Nguồn

- Yêu cầu trực tiếp của Đế và brief local được cung cấp ngày 06/11 (Thứ 6).
- [Code scoring](../../src/controller/controller.cpp), [reuse](../../src/controller/reuse.cpp), [harness](../../harness/pipeline_runner.cu), [Python scoring](../../tools/controller/pooled_policy.py).
- [Hướng dẫn chạy](../controller-score-colab-run-guide.md).
- [GitHub PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) và [rà văn phong](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), truy cập 06/11 (Thứ 6), năm 2026.
