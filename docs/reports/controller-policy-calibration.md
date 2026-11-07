# Hiệu chỉnh policy Controller trên development

Ngày 04/11 (Thứ 4), năm 2026. Công việc bổ sung trong Phase 3, deadline
chung 05/11 (Thứ 5), theo yêu cầu của Đế. Đã triển khai và chạy validation
độc lập trên Colab Tesla T4. Policy mới cải thiện số cặp đạt QoR so với
legacy, nhưng chưa tạo lợi thế latency so với Fixed-Best.

## Protocol và bằng chứng

- Khóa trước phép đo: 200 pair calibration và 50 pair validation từ pool
  training UDIS-D đã có. Component được phân theo CRC/size của ảnh; đây là
  kiểm trùng ảnh, chưa chứng nhận scene disjointness theo nhãn chính thức.
- Ảnh gốc 512 × 512; grid 48 cấu hình, blend feather; seed RANSAC 20261101;
  hai warmup và mười repeat cho mỗi pair/config. Grid có 115.200 dòng raw.
- Predicate QoR giữ nguyên SSIM/PSNR/coverage của pilot: SSIM giảm tối đa
  0,02; PSNR giảm tối đa 0,5 dB; coverage tối thiểu 0,95. Đây vẫn là policy
  pilot, chưa thay thế quyết định nghiệm thu QoR của Nhóm.
- Fit chỉ đọc calibration. Freeze policy trước validation; không tune lại
  từ 50 pair validation hoặc các tập test/HPatches.
- Source GPU và công cụ campaign: `e378c53`; build Release `sm_75`.
  Policy SHA-256:
  `f78d5762046e2ed84993d903f012ea0a1c73d8df1e31b293550c0a2fc5805056`.
  Binary SHA-256:
  `78731faafaa476ce819bbdae5fe553876ca797de56374cad6b4e714b77f38de2`.

Checkpoint cục bộ nằm tại
[`results/controller-v2-t4-20261104/final-checkpoint/`](../../results/controller-v2-t4-20261104/final-checkpoint/).
`artifact-extra-hashes.json`, `checkpoint-full-grid-hashes.json` và
`checkpoint-input-hashes.json` đã được đối chiếu từng file sau khi tải.
`policy-evidence.json` tổng hợp số liệu từ raw/audit/trace;
`derive-policy-evidence.py` trong checkpoint tính lại phần đối chiếu đó.
Dataset, policy sinh ra, raw CSV và output không được commit.

## Những đề xuất đã triển khai

| Đề xuất | Kết quả triển khai và kiểm chứng |
|---|---|
| Mở ladder theo development | Loader policy JSON tường minh; fitter chọn 12 cấu hình có gate và reference cuối; có S cao với K/R thấp |
| Hiệu chỉnh lựa chọn ban đầu | Học tối đa bốn khoảng gradient; mọi nấc được phép làm điểm bắt đầu; đối chiếu feature CPU/GPU |
| Bỏ retry không thể đạt ngưỡng | Bỏ ứng viên giữ nguyên S/K khi matches thấp hơn min_inliers của ứng viên; giữ retry đổi S/K và retry R còn khả thi |
| Hiệu chỉnh self-check | Fit bốn ngưỡng S5 theo nhãn QoR, có biên làm tròn CSV; không có false accept đã biết trên calibration và tối thiểu bốn positive cho gate được chọn |
| Validation độc lập | Chạy reference, Fixed-Best, Fixed cũ, legacy và adaptive mới trên cùng 50 pair sau freeze |

Đường `--configs` còn có lỗi seed trước lần đo này: CLI in seed nhưng từng
config dùng seed mặc định 0. Đã sửa harness và thêm regression binary thật
cho fixed/oracle. Toàn bộ grid seed 0 giữ riêng làm diagnostic; grid dùng để
fit đã chạy lại với seed 20261101. Không chỉnh cột seed trong CSV cũ.

## Calibration: cấu hình nhanh vẫn tồn tại

Có 191/200 pair với reference hợp lệ; chín pair reference failure giữ riêng
trong mẫu số. Bảng dưới chỉ xét cùng 191 pair có thể đánh giá QoR.

| Đối chứng trên calibration | Median latency (ms) |
|---|---:|
| Fixed-Best, S1/K2000/R1024 | 8,2390 |
| Oracle trong toàn grid | 3,6380 |
| Oracle giới hạn ladder cũ | 5,2805 |
| Oracle giới hạn ladder mới | 3,9020 |

Fixed-Best/Oracle toàn grid là 2,2647×. Số lựa chọn Oracle nằm trong ladder
tăng từ 23/191 lên 86/191. Ladder mới thu hẹp khoảng cách về không gian cấu
hình, nhưng chưa chứa toàn bộ lựa chọn Oracle. Đây là upper bound offline,
không gồm chi phí probe, chọn cấu hình hoặc retry của Controller.

Kết quả này thuộc cohort/seed mới. Bằng chứng lịch sử 23 pair, tỷ số 2,048×
trong `udis_train_gate_20261030`, vẫn có provenance riêng.

## Validation: QoR tốt hơn legacy, chưa nhanh hơn Fixed-Best

| Mode | QoR PASS | QoR FAIL | Reference failure/unknown | Pair có retry |
|---|---:|---:|---:|---:|
| Fixed-Best | 49 | 0 | 1 | 0 |
| Fixed cũ S0.5/K1000/R256 | 14 | 35 | 1 | 0 |
| Legacy policy | 13 | 36 | 1 | 14 |
| Adaptive mới | 49 | 0 | 1 | 0 |

Fixed-Best được chọn từ calibration và trùng cấu hình reference. Reference
chỉ định nghĩa QoR tương đối; không phải homography ground truth hay bảo
đảm chất lượng tuyệt đối. Pair `udis_train_009502` lỗi workspace tại S6 ở
reference/adaptive; không tính thành QoR PASS. Self-check S5 chấp nhận model
của pair này trước khi S6 thất bại, nên summary ghi một unknown-quality
accept, không phải một false accept đã biết.

Trên 49 pair mà cả Fixed-Best và adaptive đều PASS:

| Mode | Median latency (ms) |
|---|---:|
| Fixed-Best | 7,6260 |
| Adaptive mới | 7,6670 |

Tỷ số hai median là 0,99465×; adaptive cao hơn khoảng 0,54%. Median của tỷ số
paired là 0,99185×. Chưa có phân tích bất định để quy chênh lệch nhỏ này
thành một hiệu ứng hiệu năng có ý nghĩa. Hai mode chạy cùng cấu hình cuối;
adaptive còn có probe/controller và được đo ở lượt khác.

Không dùng median toàn 50 pair làm kết luận speedup vì có pipeline failure.
Không so latency của 13/14 pair PASS ở legacy/Fixed cũ với toàn bộ 49 pair
PASS ở adaptive: đó là những tập khác nhau. Legacy trong phép đo này dùng
implementation đã sửa skip retry; không phải nguyên binary v1 lịch sử.

## Vì sao policy chọn reference và không retry?

Ladder mới có cấu hình rẻ, nhưng self-check bảo thủ chỉ nhận một phần nhỏ
positive. Ví dụ trên 191 pair calibration có reference hợp lệ:

| Cấu hình | Positive đạt QoR | Positive được gate nhận | Negative bị gate nhận |
|---|---:|---:|---:|
| S0.75/K500/R128 | 85 | 9 | 0 |
| S1/K500/R128 | 92 | 3 | 0 |
| S1/K500/R256 | 106 | 4 | 0 |

Gate S1/K500/R128 không đủ bốn positive nên không vào ladder. Các gate được
chọn cũng bỏ lỡ nhiều cấu hình thật sự đạt QoR. Trong mô hình rollout dùng
để học rule, bắt đầu ở nấc S1/K2000/R512 có mean chi phí ước lượng 12,3382 ms,
còn bắt đầu ở reference là 8,8683 ms. Đi từ đầu ladder có mean 37,9766 ms.
Các số này là replay calibration, không phải latency adaptive đo trên T4;
ước lượng chưa mô hình hóa đầy đủ lợi ích reuse giữa các attempt.

Fitter cuối cùng học một rule hằng: `first_index = 12`, tức reference.
Trace validation xác nhận cả 50 pair bắt đầu tại S1/K2000/R1024. Vì đã ở
nấc cuối nên không có retry. Không có thay đổi vô hiệu hóa cơ chế retry;
test vẫn kiểm các nhánh retry khả thi, skip và exhaustion.

Legacy có 25 false accept đã biết; adaptive mới có 0 trên validation này.
Tuy nhiên, giảm false accept bằng cách luôn chọn reference chưa đạt mục
tiêu chọn cấu hình rẻ theo scene. Kết quả chỉ ra giới hạn của tổ hợp feature,
gate và mô hình chi phí hiện tại; chưa chứng minh Controller không thể có
lợi ích với tín hiệu hoặc policy khác. Không đổi ngưỡng để tạo số đẹp và
không học lại từ validation trong lần bàn giao này.

## Kiểm chứng và phạm vi bàn giao

- T4, source `e378c53`: 19/19 CTest và 402 test Python PASS.
- T4, source `02f9f39`: 404 test Python PASS; bổ sung cache cho chấm QoR
  offline. Campaign ở trên đã hoàn thành bằng evaluator gốc, trước cache.
- Chấm lại toàn bộ 250 nhãn pair/mode của validation bằng cache trên T4:
  CSV khớp từng byte với output evaluator gốc đã dùng cho campaign.
- Raw grid và validation có H/status/count nhất quán giữa warmup và repeat;
  trace adaptive/legacy nhất quán qua 12 lượt mỗi pair. Sai số feature
  CPU/GPU lớn nhất là 6,94e-18, dưới điều kiện kiểm 1e-15.
- Cache chỉ dùng lại input trùng bytes/shape/dtype/policy; kiểm CSV từng
  byte và mutable alias. Chi phí chấm QoR offline không được tính vào
  speedup Controller.

Policy để chạy tường minh:

```bash
build/t4/harness/stitch_harness --manifest results/controller-v2-t4-20261104/final-checkpoint/data/validation.json --adaptive --controller-policy results/controller-v2-t4-20261104/final-checkpoint/seed-correct/policy.json --seed 20261101 --out results/controller-v2-replay/adaptive.csv
```

Manifest checkpoint trỏ tới ảnh đã chuẩn bị trên Colab. Khi dùng runtime
mới hoặc máy khác, cần khôi phục ảnh từ split/index bằng
`tools.controller.prepare_calibration` và dùng manifest mới. Không sửa
manifest gốc rồi coi hash vẫn khớp. Wrapper campaign vẫn mặc định
`legacy-v1`; phải truyền file policy để chạy bản mới.

Đã hoàn thành triển khai và phép đo theo các đề xuất được chấp thuận.
Mục tiêu giảm latency online dưới cùng QoR chưa đạt. Chưa chạy lại toàn bộ
UDIS-D test hoặc HPatches với policy mới; không thay kết quả Phase 3 v1
hoặc kết luận paper bằng validation development này. Tín hiệu scene mới
hay mô hình gate/cost khác cần một quyết định tiếp theo và dữ liệu đánh giá
chưa dùng để tune.
