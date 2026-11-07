# Quyết định: cách đo Ablation A1 (S1 Probe Signal) trên UDIS-D

- **Task:** Phase 3, task của Tuấn trong [kế hoạch](../plan.md), mục 8: Batch runner UDIS-D, Ablation A1 theo [đặc tả ablation](../experiments/ablation-studies.md), mục 2.1, và viết Section 5.1 cùng mục Ablation A1 của paper.
- **Người quyết định:** Tuấn.
- **Ngày ghi nhận:** 06/11 (Thứ 6), năm 2026.
- **Phase:** Phase 3 (deadline 07/11 (Thứ 7)).
- **Trạng thái:** Đã chọn và đã kiểm chứng, đo đạc đầy đủ trên Google Colab Tesla T4 ngày 06/11 (Thứ 6). Chi tiết số liệu và phân tích ghi tại [báo cáo thực nghiệm A1](../stages/s1-a1-ablation.md).

## 1. Bối cảnh

Theo [đặc tả ablation](../experiments/ablation-studies.md) mục 2.1, Ablation A1 đánh giá hai câu hỏi trên tập test UDIS-D:

1. Tín hiệu probe độ dốc Sobel đầu vào (`s1_compute_gradient` tại `probe_scale = 0.25`) giúp Controller chọn mức thang (rung) khởi điểm hiệu quả ra sao so với việc đoán mù (*Blind Escalation*) hoặc luôn cố định cấu hình cao nhất (*Conservative High*: `S = 1.0, K = 2000, R = 1024`)?
2. Chi phí thời gian nội tại của bước probe trên GPU (`t_probe`) chiếm bao nhiêu phần trăm tổng thời gian toàn trình (`t_e2e`), và có đủ nhỏ để biện minh cho việc chạy probe trước mỗi cặp ảnh không?

Qua khảo sát mã nguồn hiện tại trên nhánh `main` (sau PR `#16` và `#17`), có ba điểm kỹ thuật cần giải quyết để thực hiện phép đo A1 khách quan, tách biệt đúng hiệu ứng của tín hiệu probe:

- Trong `harness/pipeline_runner.cu`, thời gian chạy `s1::s1_preprocess` ở tỷ lệ `probe_s = 0.25` và kernel `s1::s1_compute_gradient` đang được gộp chung vào biến `acc_s1` (`t_s1`), còn thời gian đồng bộ stream để đọc hai số vô hướng `g_avg_a`, `g_avg_b` về host được tính vào `t_controller`. Nếu không tách riêng CUDA event cho bước probe, báo cáo không thể tách bạch giữa chi phí tiền xử lý ở tỷ lệ `0.25` (`t_probe_s1`) và chi phí tính chuẩn L1 Sobel (`t_probe_grad`).
- Trong policy mặc định `legacy-v1` (`harness/controller_policy.hpp`), bậc thang 5 cấu hình gồm: Rung 0 (`S = 0.25, K = 500, R = 128`), Rung 1 (`S = 0.50, K = 1000, R = 128`), Rung 2 (`S = 0.50, K = 1000, R = 256`), Rung 3 (`S = 0.75, K = 2000, R = 512`), Rung 4 (`S = 1.00, K = 2000, R = 1024`). Khi tắt probe (`g_avg_a = 0, g_avg_b = 0`), hàm `Controller::select_initial_knobs` của `legacy-v1` sẽ mặc định chọn Rung 0 (`S_0 = 0.25`). Trong khi đó, văn bản đặc tả mục 2.1 ghi ví dụ khởi điểm của *Blind Escalation* là `S_0 = 0.5, K = 1000, R = 128` (tức Rung 1).
- Sau PR `#16`, `stitch_harness --mode adaptive` bắt buộc nhận cờ `--controller-policy <path|legacy-v1>`, nhưng `tools/colab_runner.py` chưa truyền cờ này khi gọi chế độ `adaptive`.

## 2. Phương án chọn

### 2.1. Tách riêng CUDA event đo `t_probe_s1`, `t_probe_grad` và `t_probe`

Trong `harness/pipeline_runner.hpp` và `harness/pipeline_runner.cu`, bổ sung một CUDA event `ev_probe_prep_` nằm giữa `ev_probe_start_` và `ev_probe_end_`:

- `t_probe_s1`: thời gian GPU của hai lời gọi `s1::s1_preprocess` (ảnh A và ảnh B) ở tỷ lệ `probe_s = 0.25`.
- `t_probe_grad`: thời gian GPU của hai lời gọi `s1::s1_compute_gradient` (Sobel 3×3 + block/atomic reduction trên ảnh xám `probe_s = 0.25`).
- `t_probe`: tổng thời gian GPU của pha probe (`ev_probe_start_` tới `ev_probe_end_`).

Ba trường này được lưu trong `RunRecord` (`harness/harness_types.hpp`) và xuất vào cuối dòng CSV của `stitch_harness` (`t_probe_s1,t_probe_grad,t_probe`). Các giá trị `t_s1`, `t_controller` và `t_e2e` hiện có được giữ nguyên ngữ nghĩa theo [quy ước chung](../conventions.md) để không phá vỡ bất kỳ công cụ phân tích nào của các thành viên khác.

### 2.2. Cờ `--no-probe` và bốn biến thể thực nghiệm trong Ablation A1

Bổ sung cờ `--no-probe` (`PipelineRunner::set_probe_enabled(false)`) vào `stitch_harness`. Khi bật `--no-probe`:

- Harness bỏ qua hoàn toàn các kernel GPU của pha probe (`t_probe_s1 = t_probe_grad = t_probe = 0.0`, `g_avg_a = g_avg_b = 0.0`) và gán `prev_s = -1.0` để lần thử đầu tiên (`attempt 0`) luôn chạy `s1::s1_preprocess` ở đúng tỷ lệ `S` của bậc thang khởi điểm.
- Để bao phủ trọn vẹn cả hành vi mặc định của code (`Rung 0`) lẫn ví dụ trong đặc tả (`Rung 1`), chúng tôi đo cả bốn biến thể trên toàn bộ 1.105 cặp ảnh hợp lệ của tập test UDIS-D (`testing.zip` chính thức):
  1. `adaptive` (*Adaptive with Probe*): `--mode adaptive --controller-policy legacy-v1` (có chạy probe S1 ở `probe_scale = 0.25`).
  2. `blind_s025` (*Blind Escalation, $S_0 = 0.25$*): `--mode adaptive --controller-policy legacy-v1 --no-probe` (bắt đầu 100% cặp ảnh tại Rung 0: `S = 0.25, K = 500, R = 128`).
  3. `blind_s050` (*Blind Escalation, $S_0 = 0.50$*): `--mode adaptive --controller-policy legacy-v1-blind-s05 --no-probe` (bắt đầu 100% cặp ảnh tại Rung 1: `S = 0.50, K = 1000, R = 128`, khớp chữ trong mục 2.1 của `docs/experiments/ablation-studies.md`).
  4. `conservative_high` (*Conservative High, $S = 1.0$*): `--mode fixed` với cấu hình `S = 1.00, K = 2000, R = 1024, blend = feather`.

### 2.3. Tích hợp batch runner và module tổng hợp tự động `tools/s1/a1_report.py`

- Cập nhật `tools/colab_runner.py` để truyền `--controller-policy` (mặc định `legacy-v1`) và `--trace-out` cho chế độ `adaptive`, đồng thời thêm cờ `--ablation-a1` chạy tự động cả 4 biến thể trên cùng manifest.
- Viết module `tools/s1/a1_report.py` đọc trực tiếp các file CSV và attempt trace từ kết quả đo, kiểm tra tính toàn vẹn và tính tất định giữa các repeat, tính median theo từng cặp ảnh rồi xuất đồng thời JSON (`a1_summary.json`), báo cáo Markdown và bảng LaTeX (`paper/tables/s1-a1.tex`).

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Nhược điểm | Vì sao không chọn |
|---|---|---|---|
| Cờ `--no-probe` trong harness + đo cả `blind_s025` và `blind_s050` (chọn) | Bỏ thực sự kernel probe trên GPU; bao phủ cả hành vi mặc định của `legacy-v1` (`S_0=0.25`) và chữ trong đặc tả (`S_0=0.50`); giữ nguyên các quy tắc self-check | Cần thêm 1 built-in policy nhỏ `legacy-v1-blind-s05` trong `harness/controller_policy.hpp` | |
| Chỉ giả lập Blind Escalation bằng policy JSON mà vẫn chạy kernel probe trên GPU | Không cần thêm cờ `--no-probe` trong C++ | Kernel probe và đồng bộ D2H vẫn chạy trong vùng đo `t_e2e`, làm sai lệch phép đo chi phí khi tắt probe | Không phản ánh đúng việc loại bỏ bước probe trên GPU |
| Chỉ đo 1 biến thể Blind Escalation (`S_0=0.25` hoặc `S_0=0.50`) | Bảng kết quả gọn hơn 1 dòng | Nếu chỉ đo `S_0=0.25` thì lệch ví dụ `S_0=0.5` trong đặc tả; nếu chỉ đo `S_0=0.50` thì không thấy được trường hợp tái sử dụng kết quả tiền xử lý `S=0.25` của probe | Thiếu góc nhìn đầy đủ về tương tác giữa ngưỡng độ dốc của `legacy-v1` và dữ liệu UDIS-D |

## 4. Fallback

- Nếu phiên Colab T4 bị ngắt giữa chừng khi đang chạy batch `test1105`, `tools/colab_runner.py` tái sử dụng kho lưu trữ `data/raw_udis/testing.zip` và manifest `data/test_pairs/pairs.json` đã giải nén để chạy lại từng mode mà không cần tải lại dữ liệu.
- Nếu phát hiện tín hiệu probe kết hợp với bộ ngưỡng tĩnh của `legacy-v1` chưa đem lại lợi thế latency toàn cục trên UDIS-D, báo cáo trung thực hiện tượng thực nghiệm kèm số liệu định lượng theo đúng quy tắc "không bịa đặt hoặc chọn lọc kết quả" của Nhóm, đồng thời phân tích rõ nguyên nhân gốc từ phân phối độ dốc gradient thực tế trên tập UDIS-D.

## 5. Kiểm chứng

Ngày 06/11 (Thứ 6), trên Google Colab Tesla T4 (`sm_75`, Driver `580.82.07`, CUDA `13.0`, `nvcc 13.0.88`, GNU `13.3.0`) và trên máy local Windows:

| Kiểm tra | Lệnh thực thi | Kết quả |
|---|---|---|
| Build C++/CUDA Release (`sm_75`) | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build -j4` | Thành công (0 lỗi) |
| Toàn bộ CTest trên Colab Tesla T4 | `ctest --test-dir build --output-on-failure` | `24/24 tests passed (100%)` trong `5.49s` |
| Kiểm tra bộ nhớ CUDA S1 với `compute-sanitizer` | `compute-sanitizer --tool memcheck ./build/src/s1/test_s1_cuda` | `ERROR SUMMARY: 0 errors` |
| Kiểm tra bộ nhớ Harness (`--no-probe` & `t_probe`) với `compute-sanitizer` | `compute-sanitizer --tool memcheck ./build/tests/test_harness_upload` | `ERROR SUMMARY: 0 errors` |
| Toàn bộ bộ kiểm thử Python trên Colab T4 | `python3 -m pytest tests/python -q` | `477 passed, 4 skipped` trong `14.32s` |
| Kiểm thử đơn vị cho `a1_report.py` và `colab_runner.py` trên máy local | `python -m pytest tests/python/test_s1_a1_report.py -v` | `3 passed` |
| Chiến dịch đo Ablation A1 trên 1.105 cặp UDIS-D (`test1105`) | `python3 -m tools.colab_runner --subset test1105 --ablation-a1 --seed 20261101 --warmup 2 --repeats 10 --out-dir results/p3_a1_test1105` | Hoàn tất 53.040 dòng telemetry (4 biến thể × 1.105 cặp × 12 lượt) |

## 6. Nguồn

- https://github.com/nie-lang/UnsupervisedDeepImageStitching — Kho lưu trữ chính thức của tập dữ liệu UDIS-D (`testing.zip`, Google Drive ID `1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0`); truy cập ngày 06/11 (Thứ 6), năm 2026.
