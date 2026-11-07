# CH-02 — Benchmark chính UDIS-D trên Colab T4

Mẫu số báo cáo là **1.106 cặp danh nghĩa** cho mỗi mode. Manifest đã khóa có **1.105 cặp đủ hai ảnh**. Cặp `000001` thiếu `testing/input1/000001.jpg` trong bản phát hành và có disposition `ERR_LOADER_IO` ở cả ba mode. Runner lấy bằng chứng của cặp này từ `omitted_incomplete_pairs` trong test manifest. Mẫu số latency chỉ gồm các cặp đủ điều kiện ở từng thống kê và được xuất riêng. [Quyết định về dữ liệu và QoR](../decisions/benchmark-qor-s4-s5.md)

## 1. Khóa đầu vào và phục hồi ảnh

Chạy từ repo root trên Colab. Cần gói CH-01 có `development-audit.json`, bốn CSV, `protocol/` và `data/`. `BUNDLE` trỏ đến ZIP PPM đã dùng cho Phase 3. `CONTROLLER` là JSON `controller-policy-v2` đã được chọn cho benchmark chính. Có thể truyền literal `legacy-v1` nếu đó là lựa chọn đã được phê duyệt. Ghi quyết định và seed thật trong lệnh. CH-04a `NO_GO` không cung cấp controller cho CH-02.

```bash
set -euo pipefail
ROOT=/content/ch02
EVIDENCE=/content/p3_evidence
BUNDLE=/content/drive/MyDrive/p3_protocol_bundle.zip
APPROVAL=tools/data_prep/reference-exceptions-p3.json
CONTROLLER=/content/approved-main-controller.json
CONTROLLER_APPROVAL='Replace with the benchmark-main controller approval'
SEED=0
mkdir -p "$ROOT"

python -m tools.data_prep.lock_development \
  --protocol "$EVIDENCE/protocol" --data-root "$EVIDENCE/data" \
  --audit "$EVIDENCE/development-audit.json" --approval "$APPROVAL" \
  --output "$ROOT/development-lock.json"
python -m tools.bench.batch_runner freeze \
  --evidence "$EVIDENCE" --approval "$APPROVAL" \
  --development-lock "$ROOT/development-lock.json" \
  --controller-policy "$CONTROLLER" --controller-approval "$CONTROLLER_APPROVAL" \
  --seed "$SEED" --output "$ROOT/plan.json"
python -m tools.bench.batch_runner restore \
  --plan "$ROOT/plan.json" --bundle "$BUNDLE" --output "$ROOT/test"
```

`freeze --dry-run` chạy replay trên CPU và xác nhận 1.106 ID, 48 config mà không tạo plan. Lệnh `restore` chỉ phục hồi PPM test có hash khớp manifest. Mỗi ảnh thiếu, sai kích thước hoặc sai hash được ghi trong log và runner giữ cặp tương ứng trong ledger. Có thể bỏ bước restore nếu `$ROOT/test/images` đã có PPM được ingest trước đó. Trước khi chạy chính thức, xem lỗi restore và sửa lỗi truyền file nếu có. Disposition đã publish thuộc về campaign của lần đo đó.

## 2. Build và chạy trên Tesla T4

```bash
cmake -S . -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build "$ROOT/build" --target stitch_harness -j2
python -m tools.env.generate_receipt --output "$ROOT/environment.json"
python -m tools.env.generate_receipt --output "$ROOT/environment.json" --verify-only

python -m tools.bench.batch_runner run \
  --plan "$ROOT/plan.json" --image-root "$ROOT/test" \
  --harness "$ROOT/build/harness/stitch_harness" --environment "$ROOT/environment.json" \
  --output "$ROOT/campaign" --batch-size 8 --timeout 900
```

`run --dry-run` xác nhận plan, receipt và hash binary bằng CPU. Lượt chính chạy Fixed-Best, Adaptive và toàn bộ grid, mỗi nhóm có 2 warmup và 10 repeat. `campaign.json` khóa plan, binary, GPU profile, toolchain và tham số batch. `sessions/` giữ receipt và UUID GPU của từng phiên. `invocations/` giữ lệnh, run ID, log và CSV gốc, kể cả phần đã ghi khi tiến trình dừng. `checkpoints/` xuất atomically theo cặp và mode. Lỗi khởi tạo harness dừng campaign để kiểm lệnh và binary.

Khi batch timeout, runner giữ các nhóm đủ repeat và cô lập những cặp còn thiếu. Cặp vẫn timeout ở lần chạy riêng giữ `ERR_TIMEOUT` trong ledger. Khi ngắt notebook, chạy lại chính lệnh `run` với cùng output. Runner đối soát checkpoint và khôi phục nhóm đủ repeat từ CSV còn lại trước khi chạy tiếp. Checkpoint sai hash làm dừng xác minh để điều tra.

Dùng `/content` cho atomic checkpoint. Khi cần giữ tiến độ qua reset Colab, dừng runner rồi sao lưu toàn bộ `campaign/`, `plan.json` và binary đã build lên Drive. Sau reset, khôi phục vào `/content`, dùng lại binary có cùng hash và sinh receipt mới. Phiên mới phải có cùng GPU model, compute capability, driver và toolchain. `dataset/complete.json` là marker cho bộ `runs.csv`, `attempts.csv` và `jobs.json` đã xuất đủ disposition.

## 3. Chấm QoR, kiểm join và xuất báo cáo

```bash
python -m tools.eval.qor_evaluator \
  --campaign "$ROOT/campaign" --image-root "$ROOT/test" \
  --cache "$ROOT/metrics.sqlite" --output "$ROOT/qor"
python -m tools.eval.join_and_validate \
  --campaign "$ROOT/campaign" --qor "$ROOT/qor" --output "$ROOT/joined"
python -m tools.analysis.generate_ch02_report \
  --campaign "$ROOT/campaign" --joined "$ROOT/joined" --output "$ROOT/report"
```

QoR dùng Fixed-High trong grid làm H tham chiếu. Cần đủ 10 H thành công và ổn định để xác định miền overlap cố định. Khi H tham chiếu thiếu hoặc đổi giữa repeat, disposition là `REFERENCE_FAILED`. H ứng viên được giữ theo từng repeat. Cache SQLite dùng hash ảnh A/B theo thứ tự, H ứng viên và H tham chiếu dưới dạng binary64, phiên bản evaluator, thư viện và tham số metric. Các dung sai chỉ dùng khi phân loại. Có thể truyền `--policy` và dùng thư mục QoR, join và report mới để audit offline trên cùng CSV và cache. Bản audit đó giữ Fixed-Best và controller đã đo.

Join yêu cầu trùng khóa `(pair_id, config_id, policy_id, campaign_id)` cùng mode, repeat và run ID. Join kiểm hash của từng dòng gốc và round-trip float64 chính xác qua CSV và JSON. H lệch quá `1e-12` được ghi trong `integrity.json`; mọi H theo lượt vẫn giữ trong dữ liệu. Cặp PASS cần cả 10 lượt đo vượt predicate. Oracle lấy config PASS nhanh nhất theo median của cặp. Oracle không chọn config nếu grid thiếu bằng chứng để xác nhận tối ưu.

`report/` có ledger 1.106 cặp mỗi mode, CDF, Pareto từng cặp và toàn cohort, thống kê từng stage và bảng LaTeX. Pareto từng cặp tối thiểu hóa latency và tối đa hóa riêng SSIM, PSNR, coverage. Pareto cohort so median latency trên tập grid có timing chung với số cặp execution OK và PASS. Báo cáo phân biệt latency trên cặp thực thi thành công, cặp PASS và speedup ghép cặp trên giao PASS. Speedup toàn mẫu số để trống khi thiếu bất kỳ cặp PASS ở hai phía. Cả hai chế độ dùng `t_e2e` native sau upload và tới render completion. Fixed chuẩn bị bộ đếm trước timestamp host, còn Adaptive đo probe, quyết định và retry trong vùng đo. Các khoảng stage, probe và stream wait có thể chồng lấp.
