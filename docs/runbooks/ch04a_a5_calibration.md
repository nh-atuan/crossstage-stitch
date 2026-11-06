# CH-04a — Hiệu chỉnh và khóa controller A5

Chạy trên development UDIS-D đã khóa. `train-fit` cung cấp nhãn và số liệu để fit self-check cùng scene rules. `train-val` đánh giá controller chạy thật và chọn ứng viên. Hai bước dùng cùng Fixed-High, Feather và lưới dung sai `5×4`. Gói test chỉ cần metadata đã khóa để kiểm split. Các lệnh bên dưới đọc CSV development và phục hồi PPM thuộc training.

Trên Colab, chạy sáu cell của [notebook CH-04a](../../notebooks/phase3_ch04a_calibration.ipynb). Cell 1 mặc định đọc `MyDrive/phase3_ch04a_source.zip`. Cell 3 dùng `MyDrive/p3_protocol_bundle.zip` hoặc đường dẫn `UDIS_BUNDLE` đã chỉnh. Chỉ bật `DOWNLOAD_BUNDLE` khi muốn tải bundle vào workspace Colab. Sau khi chạy, giữ bản export có log, CSV, policy và hash. Ảnh PPM có thể khôi phục từ bundle theo hash trong receipt nên được loại khỏi bản sao Drive.

## 1. Replay và khóa CH-01

Tại repo root, đặt `EVIDENCE` trỏ đến gói CH-01 chứa `development-audit.json`, bốn CSV, `protocol/` và `data/`. Chạy vào thư mục làm việc mới.

```bash
set -e
EVIDENCE=/content/p3_evidence
BUNDLE=/content/drive/MyDrive/p3_protocol_bundle.zip
OUT="$PWD/build/p3-a5-colab"
APPROVAL=tools/data_prep/reference-exceptions-p3.json

python -m tools.data_prep.lock_development \
  --protocol "$EVIDENCE/protocol" --data-root "$EVIDENCE/data" \
  --audit "$EVIDENCE/development-audit.json" --approval "$APPROVAL" \
  --output "$OUT/development-lock.json"
python -m tools.data_prep.lock_development \
  --protocol "$EVIDENCE/protocol" --data-root "$EVIDENCE/data" \
  --audit "$EVIDENCE/development-audit.json" --approval "$APPROVAL" \
  --output "$OUT/development-lock.json" --verify-only
```

Receipt kiểm audit bằng đúng hash đã duyệt và chỉ nhận 12 cặp có mã lỗi lặp lại như ledger. Chương duyệt ngoại lệ ngày 06/11 (Thứ 6). Giữ các cặp `REFERENCE_FAILED` trong mẫu số, tính 0 lượt `PASS` cho mỗi cặp và giữ cấu hình reference chung. Lựa chọn khác là đo lại development cho gate 250/250. Phương án hiện tại giữ được cùng miền tham chiếu và minh bạch về phần QoR chưa chấm được. Nếu có audit hoặc lỗi mới, lệnh dừng để xét lại disposition trước khi chọn Fixed-Best. [Quyết định QoR](../decisions/benchmark-qor-s4-s5.md)

## 2. Chuẩn bị nhãn và fit

```bash
python -m tools.controller.a5_calibration prepare \
  --evidence "$EVIDENCE" --approval "$APPROVAL" \
  --development-lock "$OUT/development-lock.json" --output "$OUT/gates"
python -m tools.controller.a5_calibration fit \
  --evidence "$EVIDENCE" --plan "$OUT/gates" --bundle "$BUNDLE" --output "$OUT/fit"
```

- `prepare` dùng chung predicate QoR với audit. Mỗi policy có Fixed-Best xếp theo số cặp pass trên đủ train-fit, sau đó theo median latency và ID. Cấu hình chọn trên fit được áp nguyên vào validation để lấy mức pass làm đối chứng. Pha này xuất `FIT_GATES_READY` và chạy được chỉ với CSV/metadata.
- `fit` trích đúng 500 PPM của 250 cặp development và kiểm từng SHA-256 với training manifest. Giữ cùng ladder năm nấc, probe 0,25 và self-check fallback của `legacy-v1`. Các nấc trước fallback dùng tối đa 12 bins cho số inlier, inlier ratio và inlier coverage. Nấc chưa có gate an toàn trên fit được giữ trong ladder với yêu cầu inlier lớn hơn K. Ngưỡng SSIM/PSNR chỉ sinh nhãn QoR offline.
- Hai ứng viên mỗi policy dùng tối đa 1 hoặc 4 vùng gradient, `min_leaf=20` và penalty 0,05 ms cho mỗi vùng thêm. Chi phí rollout ước tính ưu tiên số cặp pass, sau đó mới đến latency. Đây là proxy từ fixed sweep. Hành vi retry/reuse và latency dùng để chọn ứng viên lấy từ harness adaptive chạy thật.

Ba hàm fit và probe CPU được lấy nguyên trạng từ commit `7da18e8` của nhóm. [Handoff calibration](https://github.com/nh-atuan/crossstage-stitch/blob/c876b12/docs/reports/controller-policy-calibration.md)

## 3. Chạy validation trên Colab Tesla T4

```bash
cmake -S . -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build "$OUT/build" --target stitch_harness -j2
python -m tools.env.generate_receipt --output "$OUT/environment_receipt.json"
python -m tools.controller.a5_validation \
  --harness "$OUT/build/harness/stitch_harness" --plan "$OUT/gates" \
  --fit "$OUT/fit" --output "$OUT/validation"
python -m tools.controller.a5_validation --output "$OUT/validation" --verify-only
```

Mỗi ứng viên dùng cùng 50 cặp validation, seed của sweep, 2 warmup và 10 lượt đo mỗi cặp. Phép chấm dùng RGB gốc và H của Fixed-High đã khóa. Ledger giữ các reference failure, retry, false accept và ca reject mà H cuối đạt alignment. Cần đủ terminal/attempt và thời gian để xét khóa. Ứng viên khả thi phải có ít nhất một cặp `PASS` và đạt ít nhất số cặp `PASS` của Fixed-Best trên toàn mẫu số validation. Nếu nhiều ứng viên đạt, ưu tiên số cặp `PASS`, median trên các cặp `PASS` rồi ID để phá hòa. Median này phục vụ chọn ứng viên, còn phân tích speedup ở campaign sau cần cohort so sánh chung.

Nếu đủ 20 policy khả thi, `a5-lock.json` khóa controller, quality policy, raw CSV, summary, nguồn fit và hash binary. Nếu có policy thiếu ứng viên đạt, `a5-no-go.json` giữ kết quả từng policy và lệnh trả exit `2`. Đặc biệt, gate fallback hiện tại loại một số cặp có reference hợp lệ trong replay fit, nên khả năng đủ 20 policy còn phải kiểm trên T4. Giữ `validation/`, receipt, gate plan và candidate receipt để bàn giao. [Đặc tả A5](../experiments/ablation-studies.md#25-ablation-a5-độ-nhạy-của-ngưỡng-chất-lượng-overlap)

Với lượt đã xuất `a5-no-go.json`, kiểm độc lập bằng `python -m tools.controller.a5_validation --output "$OUT/validation" --verify-no-go`. Exit `0` ở chế độ này xác nhận đủ evidence và tái tạo được lựa chọn, còn kết luận khoa học vẫn là `NO_GO`. CH-04b là campaign test riêng và cần bộ `LOCKED` đã xác minh.
