# Công cụ offline cho Controller

## Policy học từ feature nhiều chiều

Theo [quyết định feature/gate](../../docs/decisions/p3-controller-learned-features.md),
`controller-policy-v3` dùng 26 thống kê probe của hai ảnh và bốn chênh lệch
mean theo vùng để chọn điểm bắt đầu. Gate sau S5 dùng 24 feature: các metric
S5, hình học H, photometric trên lưới 64 × 64, occupancy/entropy inlier ở
cả A và B. Feature photometric dùng A/B và H đang có, không đọc reference.
Các kernel Controller có workspace được cấp phát trước vùng đo. Chi phí
feature, synchronization và cây đều nằm trong end-to-end adaptive.

`small_tree` học cây cost-sensitive có giới hạn và xuất mảng node dùng quy
tắc `<=` giống C++. Không thêm dependency inference. Gate chia năm fold
theo pair/component, không chia theo repeat/config. Chọn depth 2 hoặc 3,
ít nhất tám mẫu ở lá; chỉ lá thuần positive trong tập fit được ACCEPT.
Selector depth tối đa ba, ít nhất 20 pair ở lá, có penalty chia nhánh.
Precision ngoài fold là điều kiện chọn gate, chưa phải bảo đảm trên ảnh mới.

`retry_cost` đo các chuỗi đổi R, K và S trên 32 pair calibration chọn bằng
hash. Fit hệ số không âm từ stage cần chạy lại và phần R tăng thêm; giữ
tám pair để kiểm sai số trước khi fit cuối trên 32 pair. Policy học target
retry tiến về phía sau theo chi phí, có thể chuyển thẳng fallback. Loader
từ chối target lùi hoặc vượt giới hạn để không tạo chu trình.

Tập validation v2 đã được xem nên chỉ dùng đối chiếu. `freeze_learned_split`
khóa 50 component training mới ngoài toàn bộ pool cũ bằng index CRC/size;
không dùng kết quả QoR hoặc latency để chọn. Tệp JSON tái tạo dùng CRLF cố
định để khớp bytes của protocol đã khóa. CRC/size chỉ kiểm trùng ảnh, chưa
chứng nhận scene disjointness chính thức.

```bash
python -m tools.controller.freeze_learned_split --old-split results/controller-v2/split.json --index-path build/udis-training-index.json --split results/controller-v3/split.json --restore-split results/controller-v3/restore-split.json --selected-index results/controller-v3/selected-index.json
python -m tools.controller.prepare_calibration --split results/controller-v3/restore-split.json --index results/controller-v3/selected-index.json --output results/controller-v3/holdout-data
python -m tools.controller.learned_measure --harness build/t4/harness/stitch_harness --manifest results/controller-v2/data/calibration.json --configs results/controller-v2/grid/configs.json --original-runs results/controller-v2/grid/runs.csv --output results/controller-v3/observations
```

`learned_measure` kiểm H/status/count/seed khớp grid gốc; NaN của model lỗi
phải khớp NaN tương ứng, không được coi là H hợp lệ. CSV thu feature chỉ
dùng quan sát; latency và nhãn QoR vẫn lấy từ sweep lặp đã kiểm chứng.
Không dùng latency fixed có bật `--features-out` làm baseline.

Sau khi đo `retry_cost.collect` và tạo `cost-model.json` bằng `retry_cost.fit`:

```bash
python -m tools.controller.learned_calibrate --runs results/controller-v2/grid/runs.csv --audit results/controller-v2/grid/audit.json --manifest results/controller-v2/data/calibration.json --split results/controller-v3/split.json --configs results/controller-v2/grid/configs.json --features results/controller-v3/observations/features.csv --cost-model results/controller-v3/transitions/cost-model.json --output results/controller-v3/strict/policy.json
python -m tools.controller.evaluate_learned --harness build/t4/harness/stitch_harness --manifest results/controller-v3/holdout-data/validation.json --split results/controller-v3/split.json --quality-policy results/controller-v2/quality-policy.json --policies results/controller-v3/policies.json --output results/controller-v3/validation
```

`policies.json` ánh xạ tên mode sang đường dẫn policy tính từ thư mục chứa
JSON. Dùng đúng bytes của `quality-policy.json` đã khóa; JSON tương đương
nhưng khác newline vẫn có SHA-256 khác. Freeze tất cả model, input image
hashes, binary, split, seed và QoR
policy trước phép đo holdout; đảo thứ tự mode theo batch. Có thể thu biến
thể diagnostic với `--minimum-precision 0.90` hoặc `0.80`, nhưng không đổi
ngưỡng SSIM/PSNR. Phải báo mọi biến thể đã khóa, gồm pair FAIL và reference
unknown. Chỉ khuyến nghị thay baseline nếu không mất pair QoR PASS trong
phép đo. Latency trên tập cùng PASS là số có điều kiện, không thay chứng
cứ chất lượng toàn cohort.

## Hiệu chỉnh policy từ development

Theo [quyết định calibration](../../docs/decisions/p3-controller-calibration.md),
pool training 250 component được tách theo thứ tự đã khóa: 200 cặp calibration
và 50 cặp validation. Không chia repeat của cùng pair hoặc ảnh chung component
sang hai tập. Các công cụ này từ chối dữ liệu ngoài partition đã khai báo,
kiểm hash đầu vào và giữ các reference failure dưới nhãn chưa xác minh QoR.

`prepare_calibration` chỉ lấy các JPEG training đã chọn qua ZIP byte ranges,
kiểm tên member, size và CRC rồi ghi RGB8 PPM ở độ phân giải gốc. Index và
cohort là metadata đã khóa của nguồn UDIS-D, không được tạo lại từ kết quả
test. Có thể dùng cohort cũ trong artifact `udis_train_gate_20261030`.

```powershell
python -m tools.controller.prepare_calibration --cohort results/udis_train_gate_20261030/documentation-archive/docs/cohorts/udis-d-development-v1.json --split results/controller-v2/split.json --freeze-only
python -m tools.controller.prepare_calibration --split results/controller-v2/split.json --index build/udis-training-index.json --output results/controller-v2/data
python -m tools.controller.sweep_calibration --harness build/t4/harness/stitch_harness --manifest results/controller-v2/data/calibration.json --split results/controller-v2/split.json --quality-policy tools/qor/policy-pilot.json --output results/controller-v2/grid
python -m tools.controller.calibrate --runs results/controller-v2/grid/runs.csv --audit results/controller-v2/grid/audit.json --manifest results/controller-v2/data/calibration.json --split results/controller-v2/split.json --configs results/controller-v2/grid/configs.json --output results/controller-v2/policy.json
python -m tools.controller.validate_policy --harness build/t4/harness/stitch_harness --manifest results/controller-v2/data/validation.json --model results/controller-v2/policy.json --split results/controller-v2/split.json --audit results/controller-v2/grid/audit.json --quality-policy tools/qor/policy-pilot.json --output results/controller-v2/validation
```

Sweep và validation yêu cầu Colab T4, build Release `sm_75`, seed RANSAC
20261101, hai warmup và mười repeat. CLI sweep không tự sinh ảnh synthetic
khi manifest thiếu. Không ghi đè output hoàn chỉnh. Khi runtime bị thu hồi,
giữ checkpoint và kiểm lại nguồn trước khi chạy campaign mới.
Mỗi shard được kiểm đủ pair/config/repeat và seed ngay sau khi harness
trả về; fitter cũng từ chối seed khác protocol, kể cả khi CSV đồng nhất.

Chấm QoR dùng cache có giới hạn ở process đánh giá: kết quả geometry,
valid mask và metric chỉ được dùng lại khi toàn bộ input khớp chính xác.
Cache gọi các hàm chuẩn trong `tools.qor.evaluate` trên mỗi input mới và
trả bản sao để tránh alias dữ liệu mutable. Test đối chiếu output CSV từng
byte, gồm reference failure. Không chạy hai evaluation trong cùng process
đồng thời khi context cache còn mở. `qor-cache.json` ghi số hit/miss, hash
source evaluator/cache, quality policy và output để truy lại phép tính.

`fit_checks` tìm self-check dạng bốn ngưỡng hiện có, không đổi predicate
SSIM/PSNR/coverage offline. Chỉ giữ các gate không có false accept đã biết
trên calibration, có ít nhất bốn positive. Xét tối đa 12 giá trị mỗi trục
và tính cả khoảng sai số làm tròn sáu chữ số của CSV. Điều kiện trên tập học
không phải bảo đảm cho dữ liệu mới. Validation báo false accept, reference
unknown và H bị reject nhưng đạt alignment QoR trước render; chỉ số cuối
không đồng nghĩa đã tạo panorama thành công.

Ladder chọn tối đa 12 cấu hình có lợi ích latency trên các positive, cộng
reference cuối làm fallback. Không buộc tăng S phải tăng K/R. Rule ban đầu
tối ưu chi phí rollout ước lượng theo gradient: tối đa bốn khoảng, ít nhất
20 pair mỗi khoảng, phạt thêm 0,05 ms/pair cho mỗi khoảng bổ sung. Nếu mẫu
nhỏ hoặc gradient không phân biệt được, dùng một rule hằng được học từ
calibration. Replay không thay phép đo latency thật và chưa mô hình hóa đầy
đủ lợi ích cache giữa các attempt.

Harness yêu cầu policy tường minh khi chạy adaptive:

```bash
build/t4/harness/stitch_harness --manifest results/controller-v2/data/validation.json --adaptive --controller-policy results/controller-v2/policy.json --seed 20261101 --out results/controller-v2/adaptive.csv
```

`--controller-policy legacy-v1` giữ policy cũ để đối chiếu. Các wrapper
campaign dùng giá trị này khi không nhận file policy; receipt ghi rõ lựa
chọn, hash policy và cấu hình Fixed. Không tự coi wrapper mặc định đã chạy
policy mới. Chế độ legacy dùng chung implementation đã sửa quy tắc bỏ retry
không thể đạt ngưỡng; đây không phải replay nguyên binary cũ. CSV mới có
`probe_gradient_a/b` ở độ chính xác đầy đủ để đối
chiếu feature GPU với CPU calibration; hai cột bằng zero trong chế độ fixed
vì không chạy probe. Online chỉ dùng scene signals và feedback S5, không
tra cứu nhãn hoặc Oracle của từng pair.

Reference cuối chỉ là baseline của QoR tương đối, không phải homography
ground truth. Nếu validation không đạt mục tiêu, báo nguyên kết quả; không
sửa threshold theo validation hoặc test để có một speedup mong muốn.

## Audit grid

`audit_sweep.py` đọc raw grid của Nhóm và gọi `tools.qor.evaluate.quality_decision` cho từng repeat. Dùng công cụ này để kiểm tra cấu hình, độ phủ Oracle và Pareto trước khi hiệu chỉnh policy của Đế.

```bash
python -m tools.controller.audit_sweep \
  --runs results/campaign/runs.csv \
  --qor results/campaign/qor.csv \
  --policy tools/qor/policy-pilot.json \
  --manifest data/dev_pairs/pairs.json \
  --configs results/campaign/configs.json \
  --repeats 10 \
  --output results/campaign/canonical-sweep-audit.json
python -m pytest tests/python/test_controller_sweep_audit.py -q
```

Manifest kê đủ pairs; configs kê đủ configuration IDs; repeat đo nằm trong `[0, repeats)`, warm-up có `is_warmup=1`. Công cụ từ chối duplicate repeat, dữ liệu ngoài cohort, CSV malformed hoặc nhiều campaign/dataset/split. Một key thiếu repeat vẫn có mặt trong ledger và mẫu số. Median latency chỉ tồn tại khi đủ repeat và mọi latency dương; run failure với latency 0 được giữ là chưa đo đủ latency. QoR failure, reference failure/unreliable và thiếu label không được coi là pass.

Một pair/config chỉ pass khi mọi repeat dự kiến pass predicate chung, gồm SSIM, PSNR, coverage, reference status và pipeline status. Ngưỡng vẫn thuộc quality policy pilot chưa được Nhóm duyệt. Chỉ khi có đầy đủ latency cho toàn cohort mới xếp cấu hình theo số pair pass giảm dần rồi median latency tăng dần, tie-break theo config ID. Cấu hình thiếu latency có danh sách riêng. Đây là chọn Fixed-Best offline, không phải classifier online.

Oracle chọn cấu hình có latency nhỏ nhất trong các cấu hình pass của từng pair, tie-break theo ID. Khi không có cấu hình pass và đủ số đo, Oracle giữ `null`; không fallback thành một cấu hình thất bại. Chỉ xuất speedup thỏa QoR toàn cohort khi Fixed-Best và Oracle đều pass mọi pair. Báo riêng median tỷ số paired và tỷ số hai median; hai phép tính có thể khác nhau. Pareto chỉ giữ điểm không bị trội đồng thời về pass count và median latency trên cùng cohort.

JSON chứa ledger, ranking, các key chưa đủ dữ liệu và SHA-256 của năm file đầu vào. CLI không ghi đè file output có sẵn hoặc input. Công cụ không quyết định GO/NO-GO, không duyệt quality policy, không tạo telemetry GPU và không kết luận Controller online nhanh hơn.

Script `tools.analysis.analyze_gonogo` được guide dẫn chiếu hiện bỏ SSIM/PSNR/reference status, dùng fallback khi Oracle không khả thi và có cách lọc Pareto khác định nghĩa nondominated. Giữ output legacy riêng khi tái chạy guide; không dùng nó để đóng task QoR. Tham số output hiện tại của script đó là `--out-json`; `--out` trong guide bị ambiguous giữa `--out-dir` và `--out-json`.

## Kiểm tra geometry của cohort tổng hợp

```bash
python -m tools.controller.audit_synthetic --campaign results/de_p2_campaign_20261030
```

Layout campaign gồm `sweep/meta.json`, `sweep/runs.csv`, `images/pairs.json` và ảnh gốc cạnh manifest. Công cụ chỉ nhận synthetic generator seed 42 hiện có. Nó phát lại đầy đủ RNG, kiểm từng pixel A/B khớp ảnh campaign, rồi mới phục hồi affine B → A trước nearest-neighbour sampling. Sai ảnh hoặc thứ tự manifest làm lệnh fail; không gán ground truth cho ảnh không tái tạo được.

Output là `sweep/synthetic-reference-audit.json` và preview bốn reference có mean corner error lớn nhất. Sai số tính bằng mean Euclidean distance ở bốn góc B, đơn vị pixel gốc. Đây là diagnostic geometry trên ảnh tổng hợp, không tự sửa reference status hoặc quality policy và không chứng minh khả năng tổng quát trên UDIS-D. Lệnh từ chối ghi đè artifact có sẵn.
