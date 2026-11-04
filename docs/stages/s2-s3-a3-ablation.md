# S2/S3: kết quả Ablation A3 trên Colab T4

- **Task:** Phase 3, Ablation A3 ([đặc tả ablation](../experiments/ablation-studies.md), mục 2.3). Cách đo theo [quyết định đo A3](../decisions/s2-s3-a3-measurement.md).
- **Người thực hiện:** Thành.
- **Ngày đo:** 04/11 (Thứ 4), năm 2026.
- **Commit:** `7009103` trên nhánh `thanh/p3-ablation-a3` (rebase lên `main` `d13554f`).

## 1. Môi trường và cách đưa code lên

| Mục | Giá trị |
|---|---|
| GPU | Tesla T4, `sm_75`, driver CUDA 13.0 |
| Toolkit | nvcc 13.0.88, host compiler GNU 13.3.0 |
| Python | cv2 4.14, matplotlib 3.10, gdown 5.2.2 (có sẵn trên runtime) |
| Clock SM khi đo | 1590/1590 MHz; nhiệt độ 63–68 °C (bench), 55–59 °C (harness) |

Colab clone `main` `d13554f` từ repo công khai rồi áp diff của nhánh. Cây `src/` sau khi áp (`39ff3336…`) và blob của 5 file mới khớp với máy local.

## 2. Dữ liệu

- Archive `training.zip` tải trên Colab: 1.945.998.335 byte, SHA-256 central directory `1e889aa8a257d1e89f2cf79eb108ee0ba7638279ec87d247a0129b995840cf79`, khớp file cohort của Đế.
- `select_cohort` (seed 20261030) chạy lại trên index của archive cho đúng 10.440 cặp, 1.789 nhóm, 250 cặp dev và 25 `pilot_pair_ids`.
- Chỉ giải nén 50 ảnh pilot (512×512). Receipt có SHA-256 từng ảnh: `data-receipt.json`, SHA-256 `259b57bc1dfec5704551befe3949f622e65f8606e72401d3cdb732630dcd8dde`.
- Golden A3: `export_timing_golden --output data/a3-golden --images <50 ảnh>` với lưới mặc định S ∈ {0.25, 0.5, 0.75, 1.0} × K ∈ {500, 1000, 2000, 4000}, tức 800 case.

Script kiểm tra archive và giải nén là script tạm, không commit.

## 3. Kiểm chứng code trước khi đo

| Kiểm tra | Kết quả |
|---|---|
| Build `src/s2_s3` Release, arch 75, và CTest | 3/3 pass (`test_s2_s3_config`, `test_s2_s3_cuda`, `s2_s3_a3_bench_smoke`) |
| Hồi quy golden sau khi tách launcher | 41/41 case khớp từng bit; "S2/S3 dynamic K checks passed (9 attempts, 1000 repeated launches)" |
| `compute-sanitizer` memcheck `--leak-check full` trên bench (4 case) | 0 lỗi, 0 byte rò rỉ |
| `compute-sanitizer` initcheck trên bench (4 case) | 0 lỗi |
| Build toàn repo và CTest | 16/16 pass |

Hai điểm lệch so với kế hoạch:

- Server Oxford (`thor.robots.ox.ac.uk`) không truy cập được từ Colab. 41 case hồi quy gồm 9 golden tổng hợp và 32 case UDIS-D của cặp `002429` (input1 và input2, 4 S × 4 K), thay cho ảnh `graf`, `wall`.
- 5 trong 9 golden tổng hợp (`scene*`, `odd*`) có hash khác manifest `data/s2s3-golden` trên máy local, vì sinh trong môi trường numpy 2.4.6 và cv2 4.14 của Colab. Output CUDA khớp từng bit với golden sinh lại, nên điểm này không ảnh hưởng tới kết luận hồi quy.

## 4. Lệnh đo

```bash
build/s2_s3/s2_s3_a3_bench --require-t4 --golden data/a3-golden/*.bin \
    --out results/a3/a3-run$r.csv --jitter-out results/a3/a3-jitter-run$r.csv
python tools/s2_s3/a3_report.py results/a3/a3-run1.csv --jitter results/a3/a3-jitter-run1.csv \
    --markdown results/a3/a3-run1-full.md --latex paper/tables/s2-s3-a3.tex
python -m tools.s2_s3.a3_plot results/a3/a3-run1.csv --output results/a3/a3-k-sweep.pdf
```

Bench chạy hai lượt liên tiếp (`r` = 1, 2), mỗi lượt khoảng 14 giây, cả hai exit 0. Log ghi "800 cases in 200 image/scale groups, bounds 512x512, K_max 4000". Mỗi ảnh và mỗi K: 2 warmup, 10 lần đo. Jitter: 4 ảnh × 1000 attempt cho mỗi mode.

SHA-256 của output:

| File | SHA-256 |
|---|---|
| `a3-run1.csv` | `f1e5abdb0cbe7a077b498fe4dfbf6677c0313880fd1e1648c5262198801554aa` |
| `a3-run2.csv` | `2829c4fedebc8489c560dc70b0c297baea75e7e8a48cee37cfa3070e22c283be` |
| `a3-jitter-run1.csv` | `50242e0307de163e3bf5c327f06c1bd958a0cbd84ffd36c10d9aae49e11da6b5` |
| `a3-jitter-run2.csv` | `60a9bcc5c600ad4e8a747aafba0e6f85a710ce38a075ebea091026e973673c4b` |
| `s2-s3-a3.tex` | `bd44382a6ce2fa2b455e80287b09f562ac43ec0147a1c314ba8f2fe8d9219871` |
| `a3-k-sweep.pdf` | `af09176e76f5c42817c4d4b0e54eb0c53af9f7d1da76815ce453c4c97ce95dd4` |

CSV và PDF chỉ nằm trên runtime Colab và trong bản tóm tắt, không có trong repo: `results/` của repo chỉ bỏ qua file ở cấp đầu, nên file trong `results/s2_s3/a3/` sẽ không bị ignore.

## 5. Hai lượt có nhất quán không

Tiêu chí: median của `s2_ms`, `s3_ms` và `host_attempt_ms` ở từng ô (S, K, mode) lệch nhau không quá max(5%, 0,005 ms), và thứ tự giữa các mode không đổi.

- 0/144 giá trị median vượt ngưỡng.
- Có 16 ô mà thứ tự xếp hạng ba mode khác nhau giữa hai lượt. Mọi chỗ đảo đều nằm giữa hai mode có chênh lệch dưới ngưỡng ở cả hai lượt. Không có cặp mode nào chênh lệch quá ngưỡng mà lại đổi chiều.

Vì vậy số dưới đây lấy từ lượt 1.

## 6. Kết quả

Số cho một ảnh, median trên 50 ảnh (p95 trong ngoặc khi cần). "Kept" là median số keypoint thật.

### 6.1. Thời gian S2 và S3 theo K, mode `view`

| S | W×H | K | Kept | Ảnh bão hòa | S2 (ms) | S3 (ms) |
|---|---|---:|---:|---:|---:|---:|
| 0.25 | 128×128 | 500 | 97.5 | 50 | 0.0324 | 0.0320 |
| 0.25 | 128×128 | 4000 | 97.5 | 50 | 0.0321 | 0.0318 |
| 0.5 | 256×256 | 500 | 500 | 18 | 0.0451 | 0.0485 |
| 0.5 | 256×256 | 1000 | 582 | 41 | 0.0450 | 0.0492 |
| 0.5 | 256×256 | 4000 | 582 | 50 | 0.0450 | 0.0493 |
| 0.75 | 384×384 | 500 | 500 | 12 | 0.0599 | 0.0522 |
| 0.75 | 384×384 | 1000 | 1000 | 22 | 0.0614 | 0.0751 |
| 0.75 | 384×384 | 2000 | 1044.5 | 43 | 0.0614 | 0.0820 |
| 1 | 512×512 | 500 | 500 | 8 | 0.0771 | 0.0575 |
| 1 | 512×512 | 1000 | 1000 | 14 | 0.0847 | 0.0802 |
| 1 | 512×512 | 2000 | 1801.5 | 28 | 0.0860 | 0.1225 |
| 1 | 512×512 | 4000 | 1801.5 | 44 | 0.0861 | 0.1227 |

Bảng đầy đủ 48 ô có trong `a3-run1-full.md` và bảng LaTeX `paper/tables/s2-s3-a3.tex`.

Nhận xét:

- S2 gần như chỉ phụ thuộc kích thước ảnh. Ở S = 1, S2 tăng từ 0,077 ms (K = 500) lên 0,086 ms (K ≥ 2000).
- S3 tăng theo số keypoint thật, không theo K danh nghĩa. Ở S = 1, S3 tăng từ 0,058 ms lên 0,123 ms khi số keypoint tăng từ 500 lên 1801,5; K = 4000 không chậm hơn K = 2000 vì ảnh đã bão hòa.
- Ảnh UDIS-D 512×512 bão hòa sớm. Ở S = 0,25, cả 50 ảnh chỉ còn median 97,5 keypoint với mọi K. Ở S = 0,5, K > 1000 không thêm keypoint nào (median 582).

### 6.2. Các pha (chỉ báo dưới dạng tỉ lệ)

Tổng các pha đo bằng marks lệch tổng không marks quá max(5%, 0,005 ms) ở 76/96 ô (hai lượt gộp), phần lớn từ +3% tới +11%. Theo fallback của quyết định đo (mục 4), các pha chỉ được báo dưới dạng tỉ lệ trong S2 hoặc S3.

Ở S = 1, mode `view`, tính từ bảng pha của `a3-run1-full.md` (tỉ lệ trên tổng các pha của cùng stage):

| K | FAST / S2 | Select / S2 | Harris / S2 | Rank / S2 | Blur / S3 | Describe / S3 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 27.0% | 39.9% | 15.0% | 18.1% | 29.6% | 70.4% |
| 1000 | 24.6% | 35.9% | 12.8% | 26.6% | 21.6% | 78.4% |
| 2000 | 24.7% | 35.8% | 13.0% | 26.5% | 14.3% | 85.7% |
| 4000 | 24.4% | 35.8% | 13.0% | 26.8% | 14.2% | 85.8% |

Phần tăng của S2 theo K nằm ở pha rank; phần tăng của S3 nằm ở pha describe. Blur không đổi theo K (0,0177 ms ở mọi K).

### 6.3. `max` so với `view`

Chênh lệch S3 (`max` trừ `view`) chỉ rõ ở K = 1000: +0,007 ms ở S = 0,75 và S = 1 (khoảng 9%), đều nằm ở pha describe. Ở các ô khác, chênh lệch không quá 0,001 ms. Ở K = 4000 hai mode có cùng capacity. Lượt 2 cho cùng kết luận: +0,007 ms ở hai ô trên, các ô khác không quá 0,002 ms.

### 6.4. Cấp phát lại (`realloc`)

| S | `alloc_ms` median | Host thêm so với `view` |
|---|---:|---:|
| 0.25 | 0.031–0.033 | +0.031 tới +0.033 ms |
| 0.5 | 0.032–0.034 | +0.036 tới +0.041 ms |
| 0.75 | 0.034–0.036 | +0.034 tới +0.039 ms |
| 1 | 0.166–0.170 | +0.178 tới +0.189 ms |

Ở S = 1, workspace S3 của `realloc` là 1.310.720 byte, so với 327.680 byte ở S ≤ 0,5. Dữ liệu chưa đủ để nói lý do `cudaMalloc` chậm hơn ở kích thước này.

Jitter của `host_attempt_ms` (4 ảnh × 1000 attempt mỗi mode):

| Lượt | Mode | p50 (ms) | p95 | p99 | Max |
|---|---|---:|---:|---:|---:|
| 1 | view | 0.1195 | 0.2478 | 0.2547 | 0.2892 |
| 1 | max | 0.1205 | 0.2477 | 0.2538 | 0.2687 |
| 1 | realloc | 0.1558 | 0.4064 | 0.4757 | 0.5850 |
| 2 | view | 0.1198 | 0.2505 | 0.2573 | 0.2708 |
| 2 | max | 0.1213 | 0.2507 | 0.2585 | 0.9403 |
| 2 | realloc | 0.1513 | 0.4994 | 1.3460 | 5.4927 |

p50 và p95 nhất quán giữa hai lượt. Đuôi p99 và max của `realloc` thay đổi mạnh giữa hai lượt (0,48 so với 1,35 ms ở p99), nên chỉ kết luận được là đuôi của `realloc` dài hơn, không báo một con số p99 cố định. `cudaMemGetInfo` trước và sau mỗi chuỗi jitter bằng nhau ở cả ba mode.

### 6.5. Bộ nhớ

Cấp một lần theo 512×512 và K_max = 4000: workspace S2 593.664 byte, workspace S3 1.310.720 byte, keypoint 48.004 byte, descriptor 128.000 byte, tổng 2031,6 KiB cho một ảnh. `realloc` vừa đúng ảnh và K cần từ 479,6 KiB (S = 0,5, K = 500) tới 2031,6 KiB.

## 7. Chi phí K trên toàn pipeline

Harness toàn repo build với `-DSTITCH_GIT_REV="7009103"`, harness đã truyền view `capacity = K` (commit `960f1b1` của Tuấn):

```bash
stitch_harness --mode fixed --configs k_sweep_configs.json --manifest pairs.json \
    --warmup 2 --repeats 10 --run-id a3e_k_sweep --out k-sweep-runs.csv
```

Cấu hình: S = 0,5, R = 256, feather, K ∈ {500, 1000, 2000, 4000}, 25 cặp pilot. Exit 0, 1000 lần đo, `git_rev` = `7009103`. CSV có SHA-256 `a05efb11215a7295cccdf37f1c0c8920b19eb877d0a61a36ad1a3573c9d5c7db`. H của mỗi cặp không đổi giữa các repeat.

Hai cặp trả `NOT_ENOUGH_POINTS` ở S5 với mọi K: `009955` (6 và 4 keypoint, 3 match) và `010327` (12 và 32 keypoint, 0 match).

Median trên 25 cặp (p95 trong ngoặc), thời gian tính bằng ms, gộp hai ảnh:

| K | Kept A | Match | S2 | S3 | S4 | S5 | S6 | E2E |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 500 | 143 | 0.086 | 0.098 | 0.186 (0.187) | 1.851 (3.511) | 0.927 | 3.184 (4.916) |
| 1000 | 579 | 157 | 0.086 | 0.101 | 0.260 (0.659) | 1.959 (5.538) | 0.926 | 3.479 (7.488) |
| 2000 | 579 | 157 | 0.086 | 0.101 | 0.260 (1.185) | 1.958 (7.099) | 0.930 | 3.479 (9.612) |
| 4000 | 579 | 157 | 0.086 | 0.101 | 0.261 (1.186) | 1.961 (7.099) | 0.936 | 3.478 (9.611) |

Ở S = 0,5, S2 + S3 chiếm khoảng 0,19 ms trong E2E 3,2–3,5 ms. Tăng K chủ yếu làm tăng S4 và S5 ở các cặp nhiều keypoint (thấy rõ ở p95). Với K ≥ 2000, median không đổi vì ảnh đã bão hòa.

Đối chiếu với `oracle_grid_runs.csv` của Tuấn (`data/udis-d/p3_dev25_results.zip`) ở S0.50, R256, feather, K ∈ {500, 1000, 2000}: S2 median 0,088 ms, S4 0,185–0,358 ms, cùng cỡ với số trên. Median keypoint (723 so với 579) và số cặp lỗi (1 cặp `DEGENERATE` so với 2 cặp `NOT_ENOUGH_POINTS`) khác nhau. Điều này khớp với nhận định trong plan là bộ dev25 của Tuấn có thể là bộ cặp khác; chưa xác nhận được vì chưa có ánh xạ `pair_xx`. Số của Tuấn không dùng trong paper.

## 8. Chưa kiểm chứng

- CSV thô và PDF chưa được lưu ra ngoài runtime Colab. Chỉ có hash, bảng Markdown, log và bảng LaTeX được đem về.
- K sweep trên harness chỉ chạy ở S = 0,5; ở S này K > 1000 bão hòa nên chưa thấy được chi phí S4/S5 khi K lớn thật sự.
- Chưa giải thích được vì sao `max` chậm hơn `view` chỉ ở K = 1000.
