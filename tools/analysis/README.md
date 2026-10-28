# Phân tích latency và QoR cho Phase 2

`latency_qor.py` tổng hợp các lần chạy terminal đã được owner gắn nhãn. Công cụ không chọn Fixed-Best, không tính Oracle từ sweep, không xây Pareto và không chạy controller. `analyze_gonogo.py` vẫn là entry point riêng; các vấn đề QoR/Oracle/Pareto đã ghi nhận cần Tuấn và Đế xử lý trước khi dùng lại số pilot.

## Đầu vào và cách chạy

Tạo manifest JSON cạnh các file CSV/policy. Đường dẫn được giải theo thư mục chứa manifest:

```json
{
  "schema_version": "latency-qor-input-v1",
  "pairs": ["pair_001", "pair_002"],
  "baseline": "fixed",
  "modes": {
    "fixed": {
      "runs": "fixed-runs.csv",
      "qor": "fixed-qor.csv",
      "policy": "policy.json",
      "split": "development",
      "timing_scope": "terminal host elapsed including retries",
      "evidence_type": "synthetic_fixture",
      "expected_repeats": 10
    }
  }
}
```

Ví dụ trên là schema, không chứa kết quả thực nghiệm. `pairs` phải kê đủ cohort dự kiến để cặp bị thiếu vẫn có mặt trong mẫu số. Các mode được hỗ trợ là `fixed`, `oracle`, `adaptive`, `adaptive_no_reuse`, `adaptive_reuse`. Tên mode chỉ là nhãn do caller cung cấp; công cụ không xác nhận thuật toán phía sau tên đó đã được triển khai.

```bash
python -m tools.analysis.latency_qor --manifest results/input.json --output results/summary.json --csv-output results/summary.csv --svg-output results/latency.svg
python -m pytest tests/python/test_latency_qor.py -q
```

Không ghi đè output cũ, manifest, CSV đầu vào hoặc policy. JSON lưu SHA-256 của manifest, CSV và policy. Không cần dependency mới ngoài môi trường QoR hiện có.

`runs.csv` phải có `run_id,pair_id,config_id,repeat_idx,is_warmup,status,fail_stage,split,t_e2e`. Đơn vị latency là ms; thời gian thành công phải dương, hữu hạn. `repeat_idx` là số nguyên từ 0 đến `expected_repeats - 1` cho các lần đo. Warmup được đếm riêng và loại khỏi thống kê. Mỗi mode chỉ nhận một campaign `run_id`, một lần chạy terminal cho mỗi `(pair_id, repeat_idx)`. Grid sweep và các hàng retry trung gian phải được owner chuyển thành terminal executions trước; đưa trực tiếp grid vào sẽ gây lỗi duplicate.

`qor.csv` dùng các cột của [evaluator](../qor/evaluate.py): `pair_id,config_id,ref_config_id,valid_stitch,ssim_overlap,psnr_overlap,overlap_coverage,reference_status`. Mỗi `(pair_id,config_id)` chỉ có một hàng. Phải có hàng cấu hình reference để so delta. QoR này áp dụng cho mọi repeat của cùng pair/config: caller chỉ được dùng adapter khi output/seed và evidence QoR đó thực sự đại diện cho tất cả repeat. Nếu H hoặc QoR thay đổi giữa repeat thì schema hiện tại chưa đủ; cần owner xuất evidence theo repeat trước khi kết luận.

## Semantics và mẫu số

Phân loại gọi trực tiếp `tools.qor.evaluate.quality_decision`: `PASS`, `FAIL`, `UNVERIFIED`, `REFERENCE_FAILED`, `EVALUATION_ERROR`. Không có hàng candidate hoặc reference, hay thiếu metric bắt buộc, không được coi là pass. NaN, infinity không hợp lệ và metric ngoài miền được phân loại bằng predicate chung; riêng PSNR `inf` có nghĩa MSE bằng 0. Reference được đánh dấu lỗi/không tin cậy vẫn giữ phân loại tương ứng. Reference có trạng thái `ok` nhưng không valid hoặc coverage khác 1 là evidence không nhất quán.

Policy hiện tại vẫn là `pilot-not-team-approved`; kết quả `quality_verified` chỉ có nghĩa tất cả repeat đã pass predicate của policy đính kèm. Đây không phải phê duyệt policy của Nhóm, cũng không xác nhận chất lượng composition. Công cụ không đổi threshold.

Thống kê latency lấy median các repeat trong từng pair, sau đó median và p95 tuyến tính của các pair median. Đây không phải p95 của tất cả repeat gộp chung. Các lần fail có thời gian dương vẫn nằm trong latency; fail với `t_e2e=0` thể hiện không có số đo, làm pair thiếu dữ liệu latency. Pair thiếu repeat không có pair median, được giữ trong ledger và mẫu số kỳ vọng. JSON ghi rõ số pair có đủ latency.

`qor_pass_rate_observed` dùng mọi repeat đã quan sát; `qor_pass_rate_expected` dùng mọi repeat dự kiến. `qor_violation_rate_evaluable` chỉ dùng `PASS + FAIL`; các trạng thái chưa xác minh/lỗi reference/lỗi evaluator được báo riêng. `pipeline_failure_rate_observed` độc lập với lỗi evaluator/reference. Thiếu repeat và lỗi không bị đổi thành pass.

So sánh paired yêu cầu cùng split, timing scope, evidence type và SHA-256 policy. `paired_median_speedup` là median tỷ số latency baseline/mode theo pair; chỉ xuất khi đủ toàn bộ cohort. `quality_constrained_paired_median_speedup` còn yêu cầu mọi repeat ở cả hai mode đều `PASS`. Các mode có thể có số repeat dự kiến khác nhau; số lượng được giữ trong report, vì vậy không được diễn giải đây là so sánh cùng ngân sách đo khi chúng khác nhau.

## Retry, fallback và overhead

Nếu có `t_controller` trong runs thì báo median theo pair; nếu thiếu thì để `null`. Để đọc các cột khác, thêm vào spec của mode:

```json
"optional_columns": {
  "retry_count": "retries",
  "fallback_used": "fallback",
  "retry_overhead_ms": "retry_ms"
}
```

`fallback_used` chỉ nhận `0/1`, `retry_count` là số nguyên không âm và overhead là ms không âm. Đây là adapter phân tích tạm thời; không áp đặt trace contract cho controller của Tuấn/Đế. Không cộng lại các thành phần vào `t_e2e` vì chúng có thể đã nằm trong thời gian terminal. Thiếu một giá trị ở repeat bất kỳ thì thống kê tùy chọn của pair là `null`.

JSON còn có `optional_metrics`: histogram retry count, số repeat dùng fallback và tỷ lệ trên repeat có giá trị, median/p95 của overhead controller/retry trên repeat có số đo. Các số này là thống kê các repeat đã quan sát, khác với thống kê latency theo pair median. Mỗi mục ghi `known_repeats`, `observed_repeats`, `expected_repeats`, số repeat thiếu trường và số repeat chưa xuất hiện. Không có evidence tùy chọn thì histogram/rate/median là `null`; không điền retry hoặc fallback bằng 0.

`evidence_type` nhận `real_measurements`, `synthetic_fixture`, `provisional_pilot`. SVG luôn ghi loại evidence, chỉ vẽ median latency của các pair đủ dữ liệu; đọc cùng JSON/CSV để thấy denominator và QoR. Chưa có raw pilot hoặc controller trace thì chỉ kiểm thử bằng fixture tổng hợp, không sinh bảng kết quả adaptive cho paper.

## Phạm vi bàn giao

Task P2-09: Chương cung cấp adapter và regression tests; Tuấn/Đế cung cấp terminal runs, QoR labels, provenance dataset/split và timing contract đã thống nhất. [QoR README](../qor/README.md) mô tả policy và tích hợp predicate cho công cụ khác. Kiểm thử tại đây kiểm chứng phép tổng hợp và failure accounting; không thay thế chạy thực nghiệm Colab T4.
