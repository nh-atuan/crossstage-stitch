# Mô hình dùng chung: phép thử với feature hiện có

Ngày 04/11 (Thứ 4), năm 2026. Đây là báo cáo tiến độ phần cải tiến bổ sung; chưa nghiệm thu Controller mới.

## Thiết kế và trạng thái

Đã train ba mô hình XGBoost dùng chung: chất lượng trước chạy, latency và chất lượng sau S5. Feature scene kết hợp S/K/R; mô hình sau S5 nhận thêm 24 feature đã có. Giữ toàn bộ 48 cấu hình, gồm K500. Không đổi predicate QoR.

Outer CV chia theo component ảnh. Bên trong mỗi outer train, booster fit trên khoảng hai phần ba component; phần còn lại dùng sigmoid calibration, chọn ngưỡng và hyperparameter. Mọi bước trên loại outer evaluation. Mỗi cặp chỉ xuất hiện ở một outer evaluation.

Chi phí replay dùng latency grid cũ, cộng dự toán probe/inference 0,5 ms và feature đã đo. Fallback tính lại đầy đủ, chưa trừ reuse; đây là mô hình chi phí bảo thủ, chưa phải E2E mới trên T4.

## Kết quả ngoài fold

| Fold | Cặp | Mean reference (ms) | Mean replay (ms) | False accept | Retry | Dừng rẻ |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 39 | 8.4732 | 9.0561 | 3 | 11 | 6 |
| 1 | 38 | 8.5999 | 8.8318 | 0 | 0 | 1 |
| 2 | 38 | 9.2306 | 9.1220 | 1 | 5 | 7 |
| 3 | 38 | 8.7881 | 8.2288 | 2 | 1 | 7 |
| 4 | 38 | 9.2599 | 9.6429 | 0 | 1 | 3 |

Tổng 191 cặp: 6 false accept, 18 retry, 24 lần dừng rẻ. Mean replay 8.9767 ms, reference 8.8683 ms. Chưa có lợi thế dưới cùng QoR.

Fit cuối chọn depth 3, 64 cây mỗi mô hình; ngưỡng trước chạy 0.700, sau S5 0.750. Không có false accept trên phần chọn ngưỡng không bảo đảm zero error ngoài fold.

## Phần tiếp theo

Đã khóa 50 component training mới, loại cả hai cohort validation đã xem và toàn calibration. Chưa chạy đánh giá policy trên tập này. Đã phục hồi ảnh trên Colab; không dùng nhãn holdout để học.

Feature CUDA quan hệ hai ảnh đã có module và fixture riêng. Chưa tích hợp các feature này xong vào phép đo fitter/harness; chưa có ablation lợi ích hoặc chi phí E2E. Loader/inference v4 đã được triển khai và đang kiểm chứng tích hợp trên T4. Chưa thay policy mặc định.

Cần chốt source sau cập nhật main, tích hợp loader/inference/feature, kiểm parity và thu feature calibration, rồi mới so ablation và chạy tập kiểm tra đã khóa. Nếu source mới thay output H/status/count, phải sinh lại nhãn; không trộn ngầm kết quả hai binary.

## Truy xuất dữ liệu

- SHA-256 split đã khóa: `275d7cba34414b29e82938d03b460b16a4bed064ee9a894f5d704b0879b3b85f`.
- SHA-256 summary: `61de873d0115d1c87bf0a2597a6fcb0051d5a04e24d2317b67db9331f0c7ad3d`.
- Artifact local: `results/controller-v4-t4-20261104/existing-features-fit/`.
- Nguồn thiết kế và trade-off: [quyết định mô hình dùng chung](../decisions/p3-controller-pooled-model.md).
