# Quyết định: chẩn đoán Controller A–E bằng artifact calibration

- **Task:** phân tích Controller và các ca thất bại trước v5, theo yêu cầu trực tiếp của Đế.
- **Người quyết định:** Đế.
- **Ngày ghi nhận:** 07/11 (Thứ 7), năm 2026.
- **Phase:** Phase 3, deadline chung 07/11 (Thứ 7).
- **Trạng thái:** đã triển khai tooling offline; chưa có learning curve đầy đủ hoặc log Colab để kết luận nguyên nhân. Không triển khai v5.

## 1. Bối cảnh

[Hồ sơ Controller](../reports/controller-investigation.md) và ZIP `full-1382a8b5cae5-68c68d47bc6c42ad93cc2974fcd59193.zip` cung cấp pipeline cùng nguồn số đo. Đã kiểm ZIP thực tế: có grid 200 pair × 48 config, mỗi config 2 warmup/10 repeat, 195 pair Reference hợp lệ, model v4 photo dựng lại gồm 64 cây depth 2 và 5 outer-fold model. Không dùng chúng như weights v4 lịch sử 128 cây. Có 8.875 post payload hợp lệ và 485 invalid trên 195 pair; không điền 0 cho invalid.

50 pair evaluation P0–P5 đã được xem. Không có full evaluation sweep, ảnh chỉ được tham chiếu bằng đường dẫn Drive cũ. Các phép thử dùng fixed features, H, QoR và latency có sẵn; không cần ảnh để rerender. Manifest tương đối gốc được dựng lại trong bộ nhớ và xác minh hash, không sửa manifest gốc.

## 2. Phương án chọn

Tạo các module nghiên cứu riêng và một notebook CPU Run all. Giữ nguyên source CUDA, feature, score, post-gate, policy mặc định, Reference và evaluator QoR. Đế đã cho phép retrain XGBoost phục vụ thí nghiệm offline; mọi model mới nằm trong thư mục kết quả E/OOF riêng.

| Source thực tế | Vai trò trong chẩn đoán |
|---|---|
| `src/controller/controller.cpp` | selector, self-check, post decision và planner retry đang chạy |
| `tools/controller/boosted_model.py` | predict/export, raw margin, fit XGBoost và probability calibration |
| `tools/controller/pooled_data.py`, `feature_schema.py` | ABI pre 39/post 77, photo active 33/63, cấu trúc feature |
| `tools/controller/train_pooled.py` | grouped outer folds và provenance trainer cũ |
| `tools/controller/pooled_policy.py` | hàm chọn config với score cũ và reuse fractions |
| `tools/controller/score_profile.py` | định nghĩa transition S/K/R và 6 policy |
| `tools/qor/evaluate.py`, `tools/qor/decision.py` | predicate QoR canonical và Reference unknown/failure |
| `tools/controller/diagnostics/` | audit, A–E, bảng/plot/checkpoint/ZIP riêng |

A/B dùng đúng archived outer models, kiểm booster-fit/probability-calibration/evaluation disjoint theo component ở cả selected và alternatives. Khi thiếu archived models, grouped CV mới dùng capacity cố định và ghi khác historical weights. Final-model prediction trên calibration chỉ là in-sample diagnostic. Gate 0.65 đã benchmark và trainer-selected 0.85 được đối chiếu mô tả, không thay default.

E giữ 64 cây depth 2, photo features, sampling seeds 20261101/20261102/20261103; outer evaluation cố định. Cỡ train tính theo component, gồm booster-fit và probability calibration tách biệt. Cap các cỡ 40/80/120/160/toàn bộ vào mức tối đa chung của outer pools; 195 component cho tối đa 156. CI lấy trung bình seed theo held-out pair rồi resample component, không resample config độc lập. Chỉ retrain calibration; không fit trên UDIS test, HPatches hoặc 50 pair evaluation đã xem. Kiểm image hash giao nhau theo archive; scene annotations cũ chưa đủ để chứng nhận độc lập lịch sử hoàn toàn.

C thay riêng probability hoặc latency rồi giữ post-gate. D giữ initial config và thay post decision bằng truth, vẫn bắt buộc self-check và post payload hợp lệ. Các route thay đổi chưa có thời gian retry/S6 tương ứng được ghi UNKNOWN. Replay full grid dùng hai scenario chi phí: fixed first attempt trừ S6, cộng gate approximation trong policy; fallback cộng stage Reference cần chạy hoặc toàn bộ fixed Reference E2E. R-only giữ S1–S4, K change giữ S1, S change bắt đầu S1; không nội suy S5 tuyến tính theo R. Đây không phải cận timing được chứng minh hoặc measured E2E. Collector cũ tính tất cả relation features nên thời gian đó không thay cho photo feature overhead.

Metrics: ROC-AUC rank với tied ranks, PR-AUC ghi rõ là noninterpolated AP; class không đủ trả về null. Brier/log loss và reliability bins có component-bootstrap CI; ECE dùng 10 bin đều. Regret chỉ trên lựa chọn feasible, denominator và lựa chọn FAIL giữ riêng. Báo paired scenario cost cùng PASS/FAIL/retry/config distribution. Tiêu chí pilot nằm trong `protocol.json`, chỉ dùng OOF/learning curve để đề xuất, không chọn tham số theo evaluation đã xem.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Nhược điểm | Lựa chọn |
|---|---|---|---|
| Tái sử dụng grid và archived folds | Không tốn sweep, kiểm được provenance, tách lỗi model và planner | Chỉ calibration, fixed-run timing không phải retry runtime | Đã chọn cho A–D |
| Learning curve grouped CPU | Kiểm sample support với capacity cố định | Nhiều fit, thời gian phụ thuộc CPU Colab | Đã chọn cho E, checkpoint từng fit |
| Sweep evaluation hoặc đo retry mới | Có timing/nhãn nhánh mới | Tốn GPU và evaluation đã xem | Ngoài notebook này; cần workflow riêng khi có quyết định |

## 4. Fallback

Thiếu dữ liệu hoặc mismatch schema/hash/split thì ghi BLOCKED/PARTIAL/FAILED cho phần liên quan và xuất bằng chứng. Không tự chạy sweep, gán post-feature hoặc nhãn thay thế. Thiếu baseline/self-check gốc thì C/E yêu cầu bổ sung policy; A/B còn có thể train OOF nghiên cứu nếu đủ grid. Không thay policy mặc định bằng model nghiên cứu.

Token chỉ lấy qua Colab Secrets, đưa tạm vào child environment cho GIT_ASKPASS. URL/remote không có token; network output bị giữ kín và token/helper được dọn sau fetch. Notebook khóa full checkout SHA, hash code/requirements, input, package versions và protocol trong run ID; không push hay điều khiển Colab tự động.

## 5. Kiểm chứng

Dùng Python CPU trong virtualenv, các requirement có sẵn của repo cùng nbformat/nbclient/ipykernel để kiểm notebook. Unit tests phủ component leakage, calibration partitions, metrics/ties/class thiếu, invalid Reference/post features, model mismatch, selector parity, oracle gate, archive tampering, checkpoint/export và notebook fixture Run all. Lệnh local:

```bash
python -m pytest tests/python/test_controller_diagnostics.py tests/python/test_diagnostics_notebook.py -q
python -m pytest tests/python -q
```

Đã audit và chạy A–D trên artifact thật bằng CPU. E và toàn luồng notebook được kiểm bằng fixture, có nhãn synthetic test riêng; không xem kết quả đó là benchmark. Chưa chạy Colab, chưa chạy E đầy đủ trên dữ liệu thật và không chạy GPU. Người dùng thực hiện Run all theo [hướng dẫn](../reports/controller-diagnostics-AE/README.md), gửi ZIP để phân tích tiếp.

## 6. Nguồn

- [Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): định nghĩa AP/ties.
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html): reliability/Brier/log loss.
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html): capacity cố định.
- [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing): rà văn bản.

Ngày đọc các nguồn: 07/11 (Thứ 7), năm 2026. Không sao chép code bên ngoài vào inference hoặc runtime.
