# S2/S3: kết quả Ablation A3 trên Colab T4

- **Task:** Phase 3, Ablation A3 ([đặc tả ablation](../experiments/ablation-studies.md), mục 2.3). Cách đo theo [quyết định đo A3](../decisions/s2-s3-a3-measurement.md).
- **Người thực hiện:** Thành.
- **Ngày đo:** 04/11 (Thứ 4), năm 2026.
- **Commit:** `8c95949` trên nhánh `thanh/p3-ablation-a3` (rebase lên `main` `d13554f`).

Lần đo đầu cùng ngày (commit `7009103`) cho cùng kết luận, nhưng runtime Colab bị thu hồi trước khi lấy được CSV thô về máy. Mọi số dưới đây là của lần đo lại, và CSV thô của lần này đã được lưu lại. Giữa hai commit chỉ có `tools/s2_s3/a3_report.py`, test của nó và tài liệu thay đổi; cây `src/` giống hệt nhau.

## 1. Môi trường và cách đưa code lên

| Mục | Giá trị |
|---|---|
| GPU | Tesla T4, `sm_75`, driver 580.82.07 |
| Toolkit | nvcc 13.0.88, host compiler GNU 13.3.0 |
| Python | cv2 4.14, matplotlib 3.10, gdown 5.2.2 (có sẵn trên runtime) |
| Clock SM khi đo | 1590/1590 MHz; nhiệt độ 49–64 °C (bench), 58–63 °C (harness) |

Colab clone repo công khai rồi checkout nhánh đã push. Hash commit, cây `src/` (`39ff3336…`) và `tools/s2_s3/a3_report.py` khớp với máy local.

## 2. Dữ liệu

- Archive `training.zip` tải trên Colab: 1.945.998.335 byte, SHA-256 central directory `1e889aa8a257d1e89f2cf79eb108ee0ba7638279ec87d247a0129b995840cf79`, khớp file cohort của Đế.
- `select_cohort` (seed 20261030) chạy lại trên index của archive cho đúng 10.440 cặp và 1.789 nhóm. Danh sách 25 `pilot_pair_ids` khớp file cohort, và SHA-256 của danh sách 250 cặp dev khớp giá trị đã ghi (`d2979cd3…`).
- Chỉ giải nén 50 ảnh pilot (512×512). Receipt có SHA-256 từng ảnh: `data-receipt.json`, SHA-256 `259b57bc1dfec5704551befe3949f622e65f8606e72401d3cdb732630dcd8dde`, giống hệt receipt của lần đo đầu.
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

Hồi quy golden và sanitizer chạy ở lần đo đầu (commit `7009103`). Lần đo lại không chạy lại hai bước này vì cây `src/` không đổi.

Hai điểm lệch so với kế hoạch:

- Server Oxford (`thor.robots.ox.ac.uk`) không truy cập được từ Colab. 41 case hồi quy gồm 9 golden tổng hợp và 32 case UDIS-D của cặp `002429` (input1 và input2, 4 S × 4 K), thay cho ảnh `graf`, `wall`.
- 5 trong 9 golden tổng hợp (`scene*`, `odd*`) có hash khác manifest `data/s2s3-golden` trên máy local, vì sinh trong môi trường numpy 2.4.6 và cv2 4.14 của Colab. Output CUDA khớp từng bit với golden sinh lại, nên điểm này không ảnh hưởng tới kết luận hồi quy.

## 4. Lệnh đo

```bash
build/s2_s3/s2_s3_a3_bench --require-t4 --golden data/a3-golden/*.bin \
    --out results/a3/a3-run$r.csv --jitter-out results/a3/a3-jitter-run$r.csv
python tools/s2_s3/a3_report.py results/a3/a3-run2.csv --jitter results/a3/a3-jitter-run2.csv \
    --markdown results/a3/a3-run2-full.md --latex paper/tables/s2-s3-a3.tex --latex-scales 0.5 1
python -m tools.s2_s3.a3_plot results/a3/a3-run1.csv --output results/a3/a3-k-sweep.pdf
```

Bench chạy ba lượt (`r` = 1, 2, 3), mỗi lượt khoảng 14 giây, cả ba exit 0. Log ghi "800 cases in 200 image/scale groups, bounds 512x512, K_max 4000". Mỗi ảnh và mỗi K: 2 warmup, 10 lần đo. Jitter: 4 ảnh × 1000 attempt cho mỗi mode.

SHA-256 của output:

| File | SHA-256 |
|---|---|
| `a3-run1.csv` | `c9e22986ab50fd8a3b85d07f80fe31867331630755c873b38e3320f720f288ce` |
| `a3-run2.csv` | `affbea78d87cb2f23822af7601f20838fdb6d55d4b3ec0ec1f3f37955ad94de0` |
| `a3-run3.csv` | `baf0fc11383944fb07be8e0794252c7328ce495c727177f3c7bfa6f8595d3be7` |
| `a3-jitter-run1.csv` | `8315d76f4e3996784553964abd0e2e6940fae75c5a4b06201ab4cbbde4a4a7ea` |
| `a3-jitter-run2.csv` | `b7644fe71283e5128dfa5357f810eb63488176b9fd168bef64547aa76bd93d54` |
| `a3-jitter-run3.csv` | `246377d1c8b8dff5ed1465b2149cbe16c59123d50149fed2595d8d57eca92241` |
| `paper/tables/s2-s3-a3.tex` | `07c6e30f7da8acc984dffc7d910fc8dd35f5210b481a2e8bcf4360b46c40c4c3` |

CSV thô của lượt 1 và 2 cùng log và bảng Markdown của cả ba lượt được lưu ngoài repo, trong thư mục kết quả của Thành. CSV của lượt 3 chỉ còn trên runtime Colab. Không commit vào `results/`: `.gitignore` chỉ bỏ qua file nằm trực tiếp trong `results/`, nên file trong `results/s2_s3/a3/` sẽ không bị ignore.

## 5. Các lượt có nhất quán không

Tiêu chí: median của `s2_ms`, `s3_ms` và `host_attempt_ms` ở từng ô (S, K, mode) lệch nhau không quá max(5%, 0,005 ms), và thứ tự giữa các mode không đổi.

| So sánh | Median vượt ngưỡng (trên 144) |
|---|---:|
| Lượt 1 với lượt 2 | 2 |
| Lượt 1 với lượt 3 | 2 |
| Lượt 2 với lượt 3 | 0 |

Hai ô vượt ngưỡng ở cả hai phép so với lượt 1 là `host_attempt_ms` của `realloc` ở S = 0,5: K = 1000 (0,1355 ms ở lượt 1, 0,1478 ms ở lượt 2, 0,1457 ms ở lượt 3) và K = 2000 (0,1340, 0,1458, 0,1448 ms). Lượt 1 không đạt tiêu chí, nên theo plan đã chạy thêm lượt 3. Lượt 2 và 3 khớp nhau ở mọi ô. Vì vậy số trong mục 6 lấy từ lượt 2, còn lượt 1 dùng làm lượt so sánh khi cần.

Giữa lượt 1 và 2 có 28 ô mà ba mode xếp thứ tự khác nhau. Mọi chỗ đảo đều nằm giữa hai mode chênh nhau dưới ngưỡng ở cả hai lượt.

## 6. Kết quả

Số cho một ảnh, median trên 50 ảnh, lấy từ lượt 2. "Kept" là median số keypoint thật.

### 6.1. Thời gian S2 và S3 theo K, mode `view`

| S | W×H | K | Kept | Ảnh bão hòa | S2 (ms) | S3 (ms) |
|---|---|---:|---:|---:|---:|---:|
| 0.25 | 128×128 | 500 | 97.5 | 50 | 0.0324 | 0.0317 |
| 0.25 | 128×128 | 4000 | 97.5 | 50 | 0.0323 | 0.0319 |
| 0.5 | 256×256 | 500 | 500 | 18 | 0.0450 | 0.0483 |
| 0.5 | 256×256 | 1000 | 582 | 41 | 0.0449 | 0.0492 |
| 0.5 | 256×256 | 4000 | 582 | 50 | 0.0450 | 0.0492 |
| 0.75 | 384×384 | 500 | 500 | 12 | 0.0612 | 0.0525 |
| 0.75 | 384×384 | 1000 | 1000 | 22 | 0.0617 | 0.0748 |
| 0.75 | 384×384 | 2000 | 1044.5 | 43 | 0.0617 | 0.0821 |
| 1 | 512×512 | 500 | 500 | 8 | 0.0769 | 0.0574 |
| 1 | 512×512 | 1000 | 1000 | 14 | 0.0858 | 0.0800 |
| 1 | 512×512 | 2000 | 1801.5 | 28 | 0.0868 | 0.1225 |
| 1 | 512×512 | 4000 | 1801.5 | 44 | 0.0864 | 0.1224 |

Bảng đầy đủ 48 ô có trong `a3-run2-full.md`. Bảng LaTeX `paper/tables/s2-s3-a3.tex` chỉ giữ S ∈ {0.5, 1} để vừa ngân sách trang.

Nhận xét:

- S2 gần như chỉ phụ thuộc kích thước ảnh. Ở S = 1, S2 tăng từ 0,077 ms (K = 500) lên 0,086–0,087 ms (K ≥ 1000).
- S3 tăng theo số keypoint thật, không theo K danh nghĩa. Ở S = 1, S3 tăng từ 0,057 ms lên 0,122 ms khi số keypoint tăng từ 500 lên 1801,5; K = 4000 không chậm hơn K = 2000 vì ảnh đã bão hòa.
- Ảnh UDIS-D 512×512 bão hòa sớm. Ở S = 0,25, cả 50 ảnh chỉ còn median 97,5 keypoint với mọi K. Ở S = 0,5, median giữ ở 582 keypoint với mọi K từ 1000 trở lên.

### 6.2. Các pha (chỉ báo dưới dạng tỉ lệ)

Tổng các pha đo bằng marks lệch tổng không marks quá max(5%, 0,005 ms) ở 72/96 ô (lượt 1 và 2 gộp), trong khoảng từ −2,7% tới +11,2%. Theo fallback của quyết định đo (mục 4), các pha chỉ được báo dưới dạng tỉ lệ trong S2 hoặc S3.

Ở S = 1, mode `view`, lượt 2 (tỉ lệ trên tổng các pha của cùng stage):

| K | FAST / S2 | Select / S2 | Harris / S2 | Rank / S2 | Blur / S3 | Describe / S3 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 27.3% | 39.8% | 14.9% | 18.0% | 29.8% | 70.2% |
| 1000 | 24.5% | 36.0% | 12.9% | 26.6% | 21.5% | 78.5% |
| 2000 | 24.5% | 35.7% | 13.0% | 26.8% | 14.2% | 85.8% |
| 4000 | 24.5% | 35.7% | 13.0% | 26.8% | 14.1% | 85.9% |

Phần tăng của S2 theo K nằm ở pha rank; phần tăng của S3 nằm ở pha describe. Blur không đổi theo K (0,0177 ms ở mọi K).

### 6.3. `max` so với `view`

Chênh lệch S3 (`max` trừ `view`) chỉ rõ ở K = 1000: +0,0070 ms ở S = 0,75 và +0,0072 ms ở S = 1 (khoảng 9%), đều nằm ở pha describe. Ở các ô khác, chênh lệch không quá 0,0013 ms. Ở K = 4000 hai mode có cùng capacity. Lượt 1 cho cùng kết luận (+0,0072 và +0,0071 ms ở hai ô trên).

### 6.4. Cấp phát lại (`realloc`)

| S | Workspace S3 (byte) | `alloc_ms` median | Host thêm so với `view` (ms) |
|---|---:|---:|---:|
| 0.25 | 81.920 | 0.030–0.032 | +0.030 tới +0.033 |
| 0.5 | 327.680 | 0.031–0.033 | +0.035 tới +0.047 |
| 0.75 | 737.280 | 0.033–0.034 | +0.036 tới +0.039 |
| 1 | 1.310.720 | 0.163–0.164 | +0.173 tới +0.193 |

Dữ liệu chưa đủ để giải thích vì sao `alloc_ms` tăng mạnh ở S = 1 mà không tăng dần theo kích thước.

Jitter của `host_attempt_ms` (4 ảnh × 1000 attempt mỗi mode):

| Lượt | Mode | p50 (ms) | p95 | p99 | Max |
|---|---|---:|---:|---:|---:|
| 1 | view | 0.1198 | 0.2437 | 0.2491 | 0.3679 |
| 1 | max | 0.1214 | 0.2442 | 0.2499 | 0.4271 |
| 1 | realloc | 0.1498 | 0.3975 | 0.4351 | 0.5228 |
| 2 | view | 0.1196 | 0.2434 | 0.2489 | 0.2714 |
| 2 | max | 0.1208 | 0.2435 | 0.2485 | 1.0558 |
| 2 | realloc | 0.1506 | 0.4039 | 0.4742 | 0.6751 |
| 3 | view | 0.1193 | 0.2436 | 0.2483 | 0.5052 |
| 3 | max | 0.1205 | 0.2437 | 0.2498 | 0.4441 |
| 3 | realloc | 0.1525 | 0.4065 | 0.4874 | 0.6165 |

p50, p95 và p99 nhất quán giữa ba lượt: `realloc` có p95 khoảng 0,40 ms so với 0,24 ms của `view` và `max`. Giá trị max dao động mạnh giữa các lượt và giữa các mode (ví dụ 1,06 ms của `max` ở lượt 2), nên không dùng để so sánh. Ở lần đo đầu, p99 của `realloc` từng lên 1,35 ms trong một lượt; lần đo lại không lặp lại hiện tượng này. `cudaMemGetInfo` trước và sau mỗi chuỗi jitter bằng nhau ở cả ba mode và cả ba lượt.

### 6.5. Bộ nhớ

Cấp một lần theo 512×512 và K_max = 4000: workspace S2 593.664 byte, workspace S3 1.310.720 byte, keypoint 48.004 byte, descriptor 128.000 byte, tổng 2031,6 KiB cho một ảnh. `realloc` vừa đúng ảnh và K cần từ 479,6 KiB (S = 0,5, K = 500) tới 2031,6 KiB (S = 1, K = 4000).

## 7. Chi phí K trên toàn pipeline

Harness toàn repo build với `-DSTITCH_GIT_REV="8c95949"`, harness đã truyền view `capacity = K` (commit `960f1b1` của Tuấn):

```bash
stitch_harness --mode fixed --configs k_sweep_configs.json --manifest pairs.json \
    --warmup 2 --repeats 10 --run-id a3e_k_sweep --out k-sweep-runs.csv
```

Cấu hình: S = 0,5, R = 256, feather, K ∈ {500, 1000, 2000, 4000}, 25 cặp pilot. Exit 0, 1000 lần đo, `git_rev` = `8c95949`. CSV có SHA-256 `79466a007b0e2e2a406bc62e4272f901c35735d058607b2e66a17a11c075fb07`. H của mỗi cặp không đổi giữa các repeat.

Hai cặp trả `NOT_ENOUGH_POINTS` ở S5 với mọi K: `009955` (6 và 4 keypoint, 3 match) và `010327` (12 và 32 keypoint, 0 match).

Median trên 25 cặp (p95 trong ngoặc), thời gian tính bằng ms, gộp hai ảnh:

| K | Kept A | Match | S2 | S3 | S4 | S5 | S6 | E2E |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 500 | 143 | 0.086 | 0.099 | 0.186 (0.187) | 1.835 (3.482) | 0.925 | 3.165 (4.903) |
| 1000 | 579 | 157 | 0.086 | 0.101 | 0.259 (0.658) | 1.944 (5.740) | 0.926 | 3.502 (7.979) |
| 2000 | 579 | 157 | 0.086 | 0.101 | 0.260 (1.184) | 1.944 (7.038) | 0.927 | 3.500 (9.542) |
| 4000 | 579 | 157 | 0.086 | 0.101 | 0.261 (1.184) | 1.945 (7.038) | 0.931 | 3.490 (9.545) |

Ở S = 0,5, S2 + S3 chiếm khoảng 0,19 ms trong E2E 3,2–3,5 ms. Tăng K chủ yếu làm tăng S4 và S5 ở các cặp nhiều keypoint (thấy rõ ở p95). Với K ≥ 2000, median không đổi vì ảnh đã bão hòa.

Đối chiếu với `oracle_grid_runs.csv` của Tuấn (`data/udis-d/p3_dev25_results.zip`) ở S0.50, R256, feather, K ∈ {500, 1000, 2000}: S2 median 0,088 ms, S4 0,185–0,358 ms, cùng cỡ với số trên. Median keypoint (723 so với 579) và số cặp lỗi (1 cặp `DEGENERATE` so với 2 cặp `NOT_ENOUGH_POINTS`) khác nhau, nên bộ dev25 của Tuấn nhiều khả năng là một bộ cặp khác. Chưa xác nhận được điều này vì chưa có ánh xạ `pair_xx` sang pilot id. Số của Tuấn không dùng trong paper.

## 8. Chưa kiểm chứng

- CSV thô của lượt 3 chỉ còn trên runtime Colab (trình duyệt chặn lần tải thứ hai). Bảng Markdown, log và hash của lượt 3 đã được lưu.
- Đồ thị `a3-k-sweep.pdf` vẽ từ lượt 1 và không dùng trong paper.
- K sweep trên harness chỉ chạy ở S = 0,5; ở S này K > 1000 bão hòa nên chưa thấy được chi phí S4/S5 khi K lớn thật sự.
- Chưa giải thích được vì sao `max` chậm hơn `view` chỉ ở K = 1000, và vì sao `alloc_ms` tăng mạnh ở S = 1.
