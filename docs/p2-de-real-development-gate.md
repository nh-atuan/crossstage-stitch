# Kiểm lại Go/No-Go trên ảnh panorama thật

- Người thực hiện: Đế. Cập nhật 30/10 (Thứ 6), năm 2026; deadline Phase 2 là 31/10 (Thứ 7).
- Kết luận: **chưa xác lập gate toàn cohort, chưa tuyên bố GO**. Canonical reference thất bại trên 2/25 cặp; báo cáo canonical trả `null` cho cả hai cách tính speedup toàn cohort.
- Phạm vi: pilot development ảnh thật. Numerical policy vẫn ở trạng thái `pilot-not-team-approved`; chưa chạy adaptive harness hoặc dùng final split.

## Dữ liệu và phép đo

[Quyết định chọn cohort](decisions/p2-de-real-development-gate.md) và [manifest đã khóa](cohorts/udis-d-development-v1.json) xác định 25 cặp pilot từ training UDIS-D trước khi xem ảnh hoặc chạy pipeline. Pool development có 250 cặp từ 250 component khác nhau của archive training. Việc gộp component dựa trên `(CRC32, file_size)` của ảnh trùng byte, không phải nhãn scene. Không có bằng chứng scene-disjoint chính thức với final. Không đọc ảnh từ testing archive.

Campaign cục bộ nằm ở `results/udis_train_gate_20261030/` (ngoài Git): 50 JPEG gốc, 50 RGB8 PPM giải mã ở độ phân giải gốc 512×512, `images/pairs.json`, `provenance.json`, `meta.json`, raw `sweep/runs.csv`, `sweep/qor.csv`, canonical reports và `sweep/gate-summary.json`. Mỗi ảnh đã được kiểm ZIP CRC/size, JPEG SHA-256 và PPM SHA-256; provenance Colab khớp máy cục bộ. Manifest có SHA-256 `3e2544aeb8270c17f01140f497e9a7c1a8c3d4f0497b94aef167410799966991`.

Đo trên Colab Tesla T4, driver 580.82.07, CUDA 13.0. Binary đo có SHA-256 `495f4f6cbf2aecfabb9ce4257e38e8c59d269a6a71c9c557973c789c5d5645d8`, build từ revision `2df1118`; policy canonical giữ SHA-256 `579d0815eecd1bbf6c3d7276ec4922aee525b8c280ef756db038e835a5f5f37f`. Grid giữ 48 cấu hình S×K×R, B=`feather`, reference `S1.00_K2000_R1024_Bfeather`, seed RANSAC 0, 2 warm-up và 10 repeat. Timing E2E của harness vẫn tính từ RGB trên device tới panorama trên device, không tính decode/H2D/D2H. Không thay code đo hoặc ngưỡng QoR.

## Kết quả canonical

Các số dưới đây lấy từ `sweep/gate-summary.json`, sinh bởi [real_gate_summary.py](../tools/controller/real_gate_summary.py) từ raw và [canonical audit](../tools/controller/audit_sweep.py); không lấy từ legacy speedup.

| Chỉ số | Kết quả | Cách đọc |
|---|---:|---|
| Raw | 2.400 warm-up; 12.000 measurement; 1.200 QoR rows | Đủ 25 × 48 × 10 lượt đo |
| Fixed-Best theo canonical audit | `S1.00_K2000_R512_Bfeather`; pass 23/25 | Median E2E 7,441 ms theo ranking của audit |
| Oracle có cấu hình khả thi | 23/25 | Hai cặp còn lại có reference thất bại |
| Tỷ trọng S1–S5 của Fixed-Best | Median 87,782% trên 230 run `OK` | Đạt điều kiện 30% trên các run hoàn thành; không áp dụng cho hai cặp lỗi |
| Speedup Fixed-Best / Oracle toàn 25 cặp | `null` | Gate 1,20× **chưa xác lập**, không được coi là đạt hoặc dưới ngưỡng |
| Chẩn đoán riêng trên 23 cặp có QoR khả thi | Tỷ số hai median 2,048× | Mẫu con được xác định sau khi thấy reference failure; không dùng để tuyên bố GO |

Reference `S1.00_K2000_R1024_Bfeather` của `udis_train_009955` trả `NOT_ENOUGH_POINTS` tại S5, 2 matches trong cả 10 repeat. Reference của `udis_train_010327` trả `DEGENERATE` tại S5, 7 matches nhưng 0 inliers trong cả 10 repeat. Predicate chuẩn đánh dấu `REFERENCE_FAILED` cho mọi cấu hình của hai cặp; vì vậy không thể xác nhận QoR của các candidate ấy theo chính sách hiện tại. Ảnh nguồn của hai cặp đã được xem trực tiếp và có vùng ít texture; đây là quan sát hình ảnh, chưa chứng minh nguyên nhân kernel. Không thay reference, loại cặp hay nới threshold sau khi thấy kết quả.

Đã lưu gallery đủ 25 cặp tại `results/udis_train_gate_20261030/gallery/index.html`, ảnh tổng quan ở `gallery/overview.jpg` và bản đóng gói ở `results/udis_train_gate_20261030/udis-25-gallery.zip`. Mỗi cặp có ảnh A/B và preview Reference, Fixed-Best, Oracle khi hợp lệ: 69 panorama PNG, hai reference failures giữ nguyên. Preview được dựng bằng S6 CPU FP64 reference có sẵn từ homography T4 của repeat 0; không phải pixel output GPU hoặc số đo latency mới. `gallery/preview-manifest.json` lưu cấu hình, H và SHA từng panorama. Chưa có ground truth hình học độc lập trên ảnh thật; QoR vẫn là so sánh tương đối với reference hiện có.

Lệnh xuất gallery vào thư mục mới: `python -m tools.controller.export_previews --campaign results/udis_train_gate_20261030 --output results/udis_train_gate_20261030/gallery-review`.

## Kiểm chứng và bước kế tiếp

- Trên T4: 4/4 test chọn cohort pass; 50/50 ảnh xác thực; harness hoàn thành 12.000 measurement; canonical QoR report và Oracle audit chạy xong. Log và receipt giữ trong campaign ngoài Git.
- Archive tải về có SHA-256 `8495843e252a3c66ec000b5802c7dab7a6b17c63f22a0bbd0621f987f59722f4`; SHA của từng raw file khớp Colab. Chạy lại canonical audit cục bộ cho JSON bằng đúng bản T4. Bộ guard xác nhận 38/38 file Controller, S5, paper và synthetic pilot không đổi.
- Để tái tạo phần tóm tắt từ raw: `python -m tools.controller.real_gate_summary --runs results/udis_train_gate_20261030/sweep/runs.csv --qor results/udis_train_gate_20261030/sweep/qor.csv --audit results/udis_train_gate_20261030/sweep/canonical-sweep-audit.json --output build/real-gate-review.json`.

Trước deadline Phase 2, chuyển kết quả này cho Nhóm để review hai reference failures và giữ trạng thái **chưa GO**. Nếu Nhóm không thể thiết lập baseline QoR hợp lệ trên cohort đã khóa, đề xuất nhánh No-Go/Pareto theo [kế hoạch](plan.md), dùng ranking/coverage hiện có và trình bày giới hạn thực nghiệm. Không calibration Controller để đạt speedup, không dùng final để sửa kết luận và không khóa numerical policy từ pilot này. [Gói bàn giao Controller cho Tuấn](controller-tuan-handoff.md) đã sẵn sàng để tích hợp adaptive harness, nhưng chưa có kết quả adaptive E2E.
