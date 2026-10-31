# Công cụ offline cho Controller

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
