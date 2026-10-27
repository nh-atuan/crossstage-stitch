# Quyết định: predicate QoR và benchmark S6 cùng H trong P2

- **Task:** P2-00 (đóng góp QoR), P2-08, P2-09 (đóng góp phân tích), P2-10 đến P2-12.
- **Người quyết định:** Chương giao agent triển khai trong phạm vi yêu cầu; policy và tích hợp dùng chung vẫn chờ Tuấn/Đế review.
- **Ngày ghi nhận:** 27/10 (Thứ 3), năm 2026.
- **Phase:** 2; deadline chung 31/10 (Thứ 7), review 01/11 (Chủ nhật).
- **Trạng thái:** Đã triển khai predicate và benchmark; kiểm chứng CUDA cục bộ trên GTX 1650, chưa nghiệm thu Colab T4. Không phê duyệt ngưỡng hay chọn lại B.

## 1. Bối cảnh

S6 hiện có feather và multiband 3/5 REDUCE, CPU reference FP64, golden runner CUDA và API caller-owned buffers/stream. Không phát hiện lỗi thuật toán S6 cần thay phương pháp trong đợt này. Các test mới kiểm tra rõ vùng một nguồn, vùng trống, pixel đen hợp lệ và repeat determinism.

Evaluator alignment warp RGB gốc theo H trong hệ ảnh A; panorama S6 không là đầu vào của PSNR/SSIM alignment. Vì thế không dùng các metric này để kết luận multiband tốt hơn feather về seam/ghosting. Chạy S1–S5 riêng cho mỗi B cũng làm thay đổi H và gây confound.

Audit `tools/analysis/analyze_gonogo.py::analyze_gonogo` xác nhận predicate chỉ xét execution, validity và coverage; `max_ssim_drop` chưa được áp dụng, PSNR/reference status cũng chưa tham gia. Nhánh không có candidate pass gán Oracle về Fixed-Best; danh sách `pareto_points` chưa lọc nondominated. Đây là phần Tuấn/Đế cần sửa và chạy lại raw pilot, không được xem summary cũ là corrected QoR result.

Trong `harness/pipeline_runner.cu`, `t_s6` lấy khoảng giữa event sau S5 và sau S6, có host readback/synchronization/geometry ở giữa. `peak_workspace_bytes` ghi workspace cấp theo canvas tối đa, không phải peak đo thực tế cho từng mode. Không dùng các trường đó như thời gian kernel S6 thuần hoặc workspace thực dùng; patch harness thuộc owner khác.

Hiện chỉ có summary pilot 25 cặp; raw runs/QoR, manifest và metadata chưa được đồng đội gửi. Không tái tạo số liệu từ summary, không tính lại Oracle hay công bố speedup adaptive.

## 2. Phương án chọn

1. Dùng `quality_decision` chung, giữ API bool `quality_pass` và schema CSV/policy cũ. Năm trạng thái là PASS, FAIL, UNVERIFIED, REFERENCE_FAILED, EVALUATION_ERROR. Tất cả metric bắt buộc phải có và hợp lệ; PSNR +inf được giữ đúng nghĩa zero MSE. Không đổi `qor-pilot-v1` hoặc trạng thái chưa được Nhóm duyệt.
2. Giữ script Go/No-Go của Tuấn/Đế; cung cấp adapter và hướng dẫn tích hợp cụ thể trong [QoR README](../../tools/qor/README.md). Phân tích P2 mới dùng predicate chuẩn, không tự chọn Fixed-Best/Oracle hoặc định nghĩa controller trace.
3. Tái dùng golden runner S6 để so B với RGB/H/canvas đóng băng, kiểm byte/hash, CPU oracle, mask và timing metadata. Đưa geometry/setup/upload ra ngoài vòng steady-state; ghi host dispatch, host chờ và CUDA event riêng.
4. Nhãn benchmark phải chỉ rõ real/synthetic, split, GPU và seed. Bảng workspace là API query theo canvas/B, không phải peak VRAM toàn runtime. Kết quả simulator/mock/GTX không đổi thành T4 hoặc full CUDA pipeline.
5. Paper dùng lại tài sản OASIcs hiện có, build ẩn danh, chỉ viết phần có source/bằng chứng. Đặt theo dõi phần thiếu của owner trong README biên tập.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Nhược điểm | Lựa chọn |
|---|---|---|---|
| Predicate thuần + wrapper tương thích | Test độc lập, tái dùng, giữ schema | Caller vẫn phải join reference đúng pair | Chọn; có regression tests và adapter |
| Sửa toàn bộ Go/No-Go/Oracle/Pareto | Có thể hợp nhất báo cáo ngay | Vượt ownership và thiếu raw pilot | Giao Tuấn/Đế |
| Golden runner có profiling | Có CPU oracle, giữ H/canvas, API không đổi | Chi phí oracle CPU và fixture I/O ngoài timing | Chọn; memory guard và log tách riêng |
| Viết pipeline benchmark mới | Toàn quyền điều khiển | Trùng hạ tầng, dễ lệch implementation | Không cần |
| Sweep B qua toàn pipeline | Gần end-to-end | H có thể khác; không cô lập B | Để thí nghiệm khác sau khi có contract |
| Dùng PSNR/SSIM alignment để xếp hạng B | Có metric sẵn | Không đo composition | Không sử dụng cho claim blend quality |

## 4. Fallback

Thiếu CUDA thì chỉ chạy CPU/tests và tạo fixture với nhãn `prepared`; thiếu T4 thì ghi preliminary GPU khác. H hoặc input lỗi phải giữ failure ledger, không thay identity hoặc bỏ pair. Khi timing clocks không nhất quán, giữ raw records và cảnh báo, chạy lại trên môi trường chính thức trước kết luận hiệu năng. Nếu policy chưa chốt hoặc reference thiếu, chỉ báo status/denominator, không báo quality-constrained advantage.

## 5. Kiểm chứng

QoR regression: 157 test pass. S6/benchmark Python: 70 test pass. Build toàn dự án bằng NVCC CUDA 12.5, `sm_75` thành công; CTest 12/12 pass trên GTX 1650. Same-H chạy hai cặp Oxford đã có trong máy, mỗi cặp ba B, hai warmup và mười repeat: sáu fixture CUDA pass, mask exact, sai số RGB tối đa 1 so FP64 oracle. H lấy từ ORB mock đã có, không phải S5 CUDA. Một fixture tổng hợp vùng tách rời chạy ba mode với 0 warmup/3 repeat cũng pass và kiểm tra mặc định không giữ ảnh.

Receipt lần chạy cuối có cảnh báo CUDA event lớn hơn host elapsed ở một số repeat; chưa xác định nguyên nhân clock/môi trường WSL. Lệnh `--require-t4` từ chối GTX 1650 đúng như yêu cầu. T4, Compute Sanitizer và corrected pilot chưa được nghiệm thu. Chi tiết và lệnh ở [bàn giao](../p2-chuong-implementation.md).

## 6. Nguồn

Đối chiếu source nội bộ: [S6](../../src/s6/), [evaluator](../../tools/qor/), [runner S6](../../tests/test_s6_cuda.cu), [harness](../../harness/pipeline_runner.cu), [plan](../plan.md), [conventions](../conventions.md), [phương pháp S6](../s6-warp-blend.md). Các lựa chọn trên là sửa hạ tầng kiểm chứng, không đưa phương pháp ngoài vào kernel. Nguồn học thuật và phạm vi đọc cho paper nằm trong [paper README](../../paper/README.md).
