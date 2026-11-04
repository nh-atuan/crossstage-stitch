# Quyết định: cách đo Ablation A3 (K động ở S2–S3)

- **Task:** Phase 3, task 1 của Thành trong [kế hoạch](../plan.md), mục 8: Ablation A3 theo [đặc tả ablation](../experiments/ablation-studies.md), mục 2.3.
- **Người quyết định:** Thành.
- **Ngày ghi nhận:** 03/11 (Thứ 3), năm 2026. Cập nhật 04/11 (Thứ 4): lưới đo theo đặc tả mới (mục 2.6), harness đã truyền view (mục 1).
- **Phase:** Phase 3.
- **Trạng thái:** Đã chọn và đã đo trên Colab T4 ngày 04/11 (Thứ 4). Kết quả và các điểm lệch so với kế hoạch ghi ở [kết quả A3](../stages/s2-s3-a3-ablation.md). Marks overhead vượt ngưỡng ở 76/96 ô, nên fallback ở mục 4 được áp dụng: các pha chỉ báo dưới dạng tỉ lệ.

## 1. Bối cảnh

Bản đặc tả A3 trước commit `7e5e9d7` hỏi thời gian của FAST và ORB co giãn thế nào theo K ∈ {500, 1000, 2000}, và việc cấp phát một lần giúp tránh jitter ra sao. Đặc tả ghi K_max = 2000 cho buffer cấp phát sẵn, yêu cầu đo `t_fast`, `t_harris` (S2) và `t_orb` (S3), so với cấp phát lại (`cudaMalloc`/`cudaFree` trong vòng lặp đo).

Ba điểm trong code không khớp với đặc tả:

- Harness cấp phát theo `max_k = 4000` (`runner.initialize(2048, 2048, 4000, 2048)` trong `harness/main.cpp`), không phải 2000.
- [Quyết định K động](s2-s3-dynamic-k.md) mục 4 bắt buộc caller truyền view `Keypoints.capacity = K`, vì với `capacity = K_max`, S3 chậm vượt ngưỡng ở K = 1000 trên T4. Khi ghi quyết định này (03/11), harness trên `main` vẫn truyền `capacity = max_k` và đặc tả không nhắc tới view. Ngày 03/11, Tuấn đã sửa cả hai: harness truyền view từ commit `960f1b1`, đặc tả A3 được cập nhật ở commit `7e5e9d7`.
- `s2_detect` và `s3_describe` là hai lời gọi, mỗi lời gọi launch nhiều kernel. Caller chỉ đo được tổng của từng stage.

Ràng buộc sẵn có: stage không cấp phát và không đồng bộ ([quy ước chung](../conventions.md), mục 4); thời gian đo bằng CUDA event, 2 warmup và 10 lần đo, median theo cặp rồi median và p95 trên các cặp (mục 7).

## 2. Phương án chọn

### 2.1. Ba biến thể, K_max = 4000

Mỗi attempt chạy S2 rồi S3 trên một ảnh, theo ba mode xen kẽ nhau trên cùng một stream:

| Mode | Buffer và workspace | `Keypoints.capacity` |
|---|---|---|
| `view` | Cấp một lần theo ảnh lớn nhất và K_max = 4000 | K |
| `max` | Như `view` | 4000, như harness hiện tại |
| `realloc` | `cudaFree` rồi `cudaMalloc` vừa đúng ảnh và K ở mỗi attempt | K |

K của mỗi ảnh được quay vòng 500 → 1000 → 2000 → 4000, nên K đổi ở mọi attempt và `realloc` phải cấp lại mỗi lần. Đây là trường hợp bất lợi nhất cho `realloc`. Khi Controller chỉ tăng R, K giữ nguyên, nên một cài đặt `realloc` hợp lý sẽ không cấp lại; số đo vì vậy là cận trên của chi phí cấp phát lại.

### 2.2. Đo từng pha bằng event marks nội bộ

`src/s2_s3/s2_s3_marks.hpp` khai báo `s2_detect_marked` và `s3_describe_marked`. Hai hàm này chạy đúng chuỗi kernel của API công khai và ghi một CUDA event sau mỗi pha. API công khai không đổi; `s2_detect` và `s3_describe` gọi cùng launcher và không truyền marks. Các pha, kèm cách ánh xạ sang tên trong đặc tả:

| Pha | Kernel | Tên trong đặc tả |
|---|---|---|
| `t_fast` | reset histogram, FAST | `t_fast` |
| `t_select` | NMS, chọn ngưỡng 2K, scan, scatter | (phần còn lại của S2) |
| `t_harris` | Harris response | `t_harris` |
| `t_rank` | xếp hạng và ghi K điểm | (phần còn lại của S2) |
| `t_blur` | hai pass Gaussian | `t_orb` = `t_blur` + `t_describe` |
| `t_describe` | orientation và rBRIEF | |

Mỗi attempt chạy hai lần: lần không marks cho `s2_ms`, `s3_ms` và thời gian host; lần có marks cho các pha. Event chèn giữa các kernel có thể làm tổng thời gian lệch đi, nên số đo chính của S2 và S3 luôn lấy từ lần không marks. Phần lệch giữa tổng các pha và tổng không marks được báo riêng (cột "Marks overhead" của `a3_report.py`).

### 2.3. Thời gian host và cấp phát

`host_attempt_ms` đo bằng `steady_clock`, từ đầu attempt (gồm `cudaFree`/`cudaMalloc` của `realloc`) tới sau `cudaStreamSynchronize`. `alloc_ms` là riêng phần cấp phát. Jitter đo trên 1000 attempt liên tiếp của 4 ảnh đầu cho từng mode. `cudaMemGetInfo` trước và sau chuỗi đó phải bằng nhau với `view` và `max`.

### 2.4. Ranh giới đo

Bench đo một ảnh mỗi lần gọi, không phải `t_s2`, `t_s3` của quy ước chung (gộp ảnh A và B). Số trong bảng A3 vì vậy là cho một ảnh.

### 2.5. Kiểm tra trước khi đo

Với mỗi ảnh và mỗi K, cả ba mode, có và không marks, phải cho cùng count, keypoint, response, descriptor và ảnh blur, và khớp golden do bản tham chiếu NumPy sinh ra theo sai số của [decision ORB](s2-s3-orb.md). Có sai lệch thì bench dừng.

### 2.6. Dữ liệu

25 cặp pilot UDIS-D lấy từ tập train do Đế freeze (`tools/controller/real_cohort.py`, seed 20261030), tức 50 ảnh. Ảnh được lấy từ `training.zip` gốc. Trước khi dùng, kiểm tra lại cohort trên archive: kích thước file, SHA-256 của central directory, và `select_cohort` phải trả về đúng `pilot_pair_ids`.

Lưới đo là K ∈ {500, 1000, 2000, 4000} × S ∈ {0.25, 0.5, 0.75, 1.0}, theo đặc tả A3 sau commit `7e5e9d7`. Thành chốt lưới này ngày 04/11 (Thứ 4), thay cho lưới 3 K × S ∈ {0.5, 1.0} ghi ngày 03/11. Golden sinh bằng `tools/s2_s3/export_timing_golden.py` với lưới mặc định, đúng bằng lưới trên.

Có hai điểm cần lưu ý:

- Ở K = 4000, `view` và `max` có cùng capacity.
- Ảnh UDIS-D khoảng 512×512, nên ở S nhỏ nhiều ảnh không đủ 2K maxima và K lớn bị bão hòa. Bench ghi số keypoint thật (`count`) và số candidate (`maxima_selected`) để đo tình trạng này.

Đồ thị K theo thời gian vẽ bằng `tools/s2_s3/a3_plot.py`. Script import matplotlib bên trong hàm vẽ, chỉ chạy trên Colab và không thêm vào `tools/requirements.txt`.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Nhược điểm | Vì sao không chọn |
|---|---|---|---|
| Event marks nội bộ (chọn) | Cùng cơ chế CUDA event với quy ước chung; chạy được trên Colab không cần thêm công cụ; không đổi API | Thêm một launcher nội bộ trong code stage; event chèn giữa kernel có overhead | |
| Profiler `nsys` theo kernel | Không sửa code, có thời gian từng kernel | Chưa kiểm chứng `nsys` có sẵn trên Colab T4; overhead trace khác event; phải xử lý file report | Không chắc chạy được trong phase này |
| Chỉ đo `t_s2`, `t_s3` | Không sửa code | Không biết pha nào co giãn theo K; không đạt đặc tả | Thiếu số liệu chính của A3 |
| K_max = 4000 (chọn) | Khớp harness thật; footprint báo ra đúng với pipeline | Lệch giá trị 2000 trong đặc tả | |
| K_max = 2000 như đặc tả | Khớp chữ trong đặc tả | Không phản ánh harness; `max` khi đó ít block thừa hơn thực tế | Kết quả không áp dụng được cho pipeline |
| Cohort UDIS-D train (chọn) | Ảnh thật, không đụng tập test, cùng nguồn với pilot Phase 2 | Cần archive train hoặc file cohort của Đế | |
| Ảnh Oxford như Phase 2 | Có sẵn tool, độ phân giải lớn hơn | Không phải dữ liệu benchmark của paper | Chỉ dùng cho test hồi quy |
| Ảnh tổng hợp | Không cần tải dữ liệu | Phân bố corner khác ảnh thật | Không dùng cho số trong paper |

## 4. Fallback

- Nếu tổng các pha lệch tổng không marks quá max(5%, 0.005 ms) ở một ô (cùng ngưỡng với [quyết định K động](s2-s3-dynamic-k.md) mục 5), các pha chỉ được báo dưới dạng tỉ lệ trong S2 hoặc S3, không báo dưới dạng thời gian tuyệt đối.
- Nếu không lấy được cohort train (cả file của Đế lẫn archive `training.zip`), dừng phần đo và hỏi Thành. Không thay bằng ảnh tổng hợp.
- Nếu `realloc` làm hỏng phép đo do đồng bộ ngầm của `cudaFree` lan sang mode khác (thấy qua `host_attempt_ms` của `view` hoặc `max` tăng so với lượt chạy riêng), chạy riêng từng mode và ghi lại điều này.

## 5. Kiểm chứng

Ngày 03/11 (Thứ 3), trên máy Windows của Thành (không có nvcc):

| Kiểm tra | Kết quả |
|---|---|
| Build host-only `src/s2_s3` và CTest host | Pass (1/1) |
| Kiểm cú pháp `g++ -fsyntax-only` của `tests/test_s2_s3_cuda.cu` và `tests/s2_s3/a3_budget_bench.cu` với một `cuda_runtime.h` giả chỉ khai báo các hàm được dùng | Pass; mới chỉ kiểm phần host, chưa phải build CUDA |
| `pytest` của S2/S3 (`test_s2_s3_a3_report.py`, `test_s2_s3_dynamic_timings.py`, `test_s2_s3_reference.py`) | 33 passed (chạy lại sau rebase ngày 04/11) |
| `test_s2_s3_a3_plot.py` (dữ liệu của đồ thị) và chạy thử `a3_plot.py` trên CSV giả | 2 passed; PDF vẽ đúng |

Ngày 04/11 (Thứ 4), trên Colab T4, commit `7009103` (chi tiết ở [kết quả A3](../stages/s2-s3-a3-ablation.md), mục 3 và 5):

| Kiểm tra | Kết quả |
|---|---|
| Build CUDA `src/s2_s3` và CTest | 3/3 pass, gồm `s2_s3_a3_bench_smoke` |
| Hồi quy golden sau refactor | 41/41 khớp từng bit (ảnh Oxford thay bằng cặp UDIS-D `002429`), dynamic K checks pass |
| compute-sanitizer memcheck và initcheck trên bench | 0 lỗi, 0 byte rò rỉ |
| Hai lượt bench | 0/144 median lệch quá max(5%, 0,005 ms); không có đảo thứ tự mode vượt ngưỡng |
| Build toàn repo và CTest | 16/16 pass |

## 6. Nguồn

Không dùng nguồn ngoài.
