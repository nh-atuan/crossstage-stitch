# S2/S3: bằng chứng trên Colab T4

- Người chạy: Thành, AI hỗ trợ chạy qua Colab MCP theo yêu cầu.
- Ngày chạy: 25/10 (Chủ nhật), năm 2026.
- Mã nguồn: nhánh `thanh/p1-s2-s3`. Golden và CUDA ở commit `a423bab`, pipeline mock ở `85fe756`, tối ưu ở `6ff852a`. Lượt chạy cuối dùng mã của `6ff852a`.
- Môi trường: Tesla T4 sm_75, driver 580.82.07, CUDA 13.0 (V13.0.88), CMake 3.31.10, Python 3.13.16, NumPy 2.1.3, `opencv-python` 5.0.0.93.

Đây là bằng chứng cho correctness của S2/S3 và cho lần nối đầu tiên với S4/S5. Chưa phải benchmark UDIS-D, chưa có latency end-to-end, và chưa có Go/No-Go.

## 1. Cách đưa code lên Colab

Repo là private nên Colab không clone được. Các file cần thiết được ghi vào `/content/cs` qua cell, và mỗi file đều được so SHA-256 với bản trong git. Có hai ngoại lệ:
- `orb_pattern.hpp` được sinh lại trên Colab từ `orb.cpp` của OpenCV 5.0.0 bằng cùng script, hash khớp bản trong repo;
- bốn file test riêng của S4/S5 chỉ được thay bằng file giữ chỗ để CMake configure được, và không được build. Driver build bằng `--target s2_s3_mock_driver`.

Colab cài sẵn `opencv-contrib-python` 4.14, gói này che bản pin. Gói đó đã được gỡ khỏi runtime và cài lại đúng `opencv-python==5.0.0.93` theo `tools/requirements.txt`.

## 2. Correctness

```bash
python -m tools.s2_s3.export_golden --output data/s2s3-golden
cmake -S src/s2_s3 -B build/s2_s3 -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build/s2_s3 -j2
ctest --test-dir build/s2_s3 --output-on-failure
build/s2_s3/test_s2_s3_cuda --require-t4 --golden data/s2s3-golden/*.bin data/s2s3-timing/*.bin
compute-sanitizer --tool memcheck --error-exitcode 1 build/s2_s3/test_s2_s3_cuda --require-t4 --golden ...
```

| Kiểm tra | Kết quả |
|---|---|
| CTest (host + contract CUDA) | 2/2 pass |
| Golden tổng hợp (9 case: K = 1 tới 20000, ảnh phẳng, ảnh thấp hơn 63 px, `harris_k = 0`, ảnh lẻ 333×257) | 9/9 pass |
| Golden từ ảnh Oxford `graf`, `wall` ở S ∈ {1, 0.75, 0.5, 0.25}, K ∈ {500, 1000, 2000, 4000} | 32/32 pass |
| Mức khớp với bản tham chiếu ở cả 41 case | Vị trí keypoint 0 lỗi, response sai lệch 0, blur 0 pixel lệch, descriptor khớp bit 100% |
| Chạy hai lần liên tiếp | Output giống từng byte |
| compute-sanitizer memcheck, racecheck, initcheck, synccheck (4 case, gồm K = 20000 và 1000×700 ở K = 4000) | 0 lỗi, 0 hazard |

Ngưỡng nghiệm thu trong decision (≥ 99% descriptor khớp bit) có dư địa cho chênh lệch `cos`/`sin` giữa GPU và CPU. Thực tế không cần dùng tới dư địa này: tất cả đều khớp hoàn toàn.

Bản tham chiếu đã được so với `cv2.ORB(nlevels=1)` trong `tests/python/test_s2_s3_reference.py`. Test này chạy được cả ở máy Windows của Thành lẫn trên Colab.

## 3. Nối với S4/S5 CUDA trên ảnh thật

Máy chủ HPatches (`icvl.ee.ic.ac.uk`) không truy cập được từ Colab trong ngày chạy. Thay vào đó dùng 8 chuỗi [Oxford affine của VGG](https://www.robots.ox.ac.uk/~vgg/research/affine/), truy cập 25/10 (Chủ nhật), năm 2026. Đây là nguồn gốc của các chuỗi `v_graffiti`, `v_wall`, `v_boat`, `v_bark` trong HPatches. Các chuỗi được đổi sang cấu trúc thư mục kiểu HPatches (`1.ppm`, `H_1_2`, …) ngay trong runtime, rồi chạy `tools/mock/gen_mock.py` gốc với cặp (1, 2) và (1, 3). Tổng cộng 16 cặp. Dữ liệu không được commit.

```bash
cmake -S tests/s2_s3 -B build/pipe -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/pipe --target s2_s3_mock_driver -j2
python -m tools.s2_s3.check_mock_pipeline build/pipe/s2_s3_mock_driver results/s2s3-oxford data/mock/*
```

Cấu hình:
- S = 0.5, K = 1000, `fast_threshold = 20`;
- S4: ratio 0.8, không cross-check;
- S5: R = 2000, batch 64, ngưỡng 3 px, 1 lần refine, seed 0.

Các tham số S4/S5 giống `tests/s4_s5_mock_driver.cu` của Đế. Sai số là trung bình khoảng cách 4 góc ảnh so với H đúng, tính bằng pixel ảnh gốc. Ba cột sai số:

| Cột | Cách chạy |
|---|---|
| `CUDA` | Chuỗi của Nhóm |
| `mock ORB` | Kết quả lưu sẵn của `cv2.ORB` 8 level trên ảnh S1 σ = 1 |
| `ORB 1 level` | `cv2.ORB(nlevels=1)` trên cùng ảnh σ = 0, ghép bằng S4/S5 mock của OpenCV |

| Cặp | Keypoint A/B | Match | Inlier | CUDA (px) | mock ORB (px) | ORB 1 level (px) | Kết quả |
|---|---:|---:|---:|---:|---:|---:|---|
| bark 1–2 | 1000/1000 | 75 | 58 | 6.365 | 8.550 | 6.404 | PASS |
| bark 1–3 | 1000/1000 | 18 | 4 | 33580.333 | 56.524 | 933.531 | FAIL |
| bikes 1–2 | 1000/796 | 641 | 616 | 0.718 | 1.278 | 1.173 | PASS |
| bikes 1–3 | 1000/667 | 547 | 531 | 1.175 | 2.643 | 3.010 | PASS |
| boat 1–2 | 1000/1000 | 359 | 343 | 1.541 | 3.349 | 5.821 | PASS |
| boat 1–3 | 1000/1000 | 44 | 19 | 40.190 | 8.667 | 30.345 | FAIL |
| graf 1–2 | 1000/1000 | 249 | 224 | 1.995 | 2.885 | 9.031 | PASS |
| graf 1–3 | 1000/1000 | 50 | 24 | 32.428 | 9.454 | 16.012 | FAIL |
| leuven 1–2 | 1000/978 | 639 | 610 | 0.552 | 0.798 | 0.635 | PASS |
| leuven 1–3 | 1000/782 | 537 | 511 | 0.780 | 2.228 | 0.971 | PASS |
| trees 1–2 | 1000/1000 | 345 | 340 | 1.554 | 2.970 | 2.604 | PASS |
| trees 1–3 | 1000/1000 | 270 | 259 | 2.400 | 2.217 | 2.743 | PASS |
| ubc 1–2 | 1000/1000 | 900 | 896 | 0.146 | 0.112 | 0.104 | PASS |
| ubc 1–3 | 1000/1000 | 836 | 831 | 0.120 | 0.164 | 0.120 | PASS |
| wall 1–2 | 1000/1000 | 358 | 350 | 3.227 | 4.173 | 3.149 | PASS |
| wall 1–3 | 1000/1000 | 239 | 230 | 2.564 | 6.263 | 3.564 | PASS |

Kết quả: 13/16 cặp đạt tiêu chí "không tệ hơn mock ORB quá 1 px". Ở 11 trong 13 cặp này, CUDA còn có sai số thấp hơn mock ORB.

Ba cặp không đạt là bark 1–3 (zoom và xoay), boat 1–3 (zoom và xoay) và graf 1–3 (góc nhìn lệch khoảng 40°). Trên cùng ba cặp đó, `cv2.ORB` một level của OpenCV cũng sai 933, 30 và 16 px. Từ đó suy ra nguyên nhân là phương pháp một level không có pyramid scale, không phải lỗi kernel CUDA. Đây là suy luận từ ba cặp, chưa được kiểm tra trên dữ liệu khác.

Ngưỡng không được nới, và ba cặp fail được giữ nguyên trong `report.json`. Ở bark 1–3, H của CUDA sai tới mức canvas preview vượt 5,7·10⁹ pixel, nên preview bị bỏ qua và lý do được ghi vào report.

Các cặp ghép panorama, như UDIS-D là benchmark chính, thường có scale gần nhau. Nhưng điều này **chưa được kiểm chứng** trên UDIS-D. Cách xử lý ba cặp này cần Nhóm quyết định (xem decision, mục 4 và 6).

Sau khi tối ưu (`6ff852a`), H của cả 16 cặp giống hệt trước tối ưu.

## 4. Thời gian S2 và S3 trên T4

Đo bằng `test_s2_s3_cuda --timings`: 2 lần warmup và 10 lần đo, lấy median của 10 lần đo. CUDA event bao quanh `s2_detect` và `s3_describe` của **một ảnh**, trên buffer và workspace đã cấp phát sẵn. Số đo không gồm H2D, cấp phát, S1 hay đồng bộ host. Ảnh là ảnh 1 của Oxford `wall` (1000×700 ở S = 1), S1 với σ = 0. Xung SM lúc đo là 1515/1590 MHz.

Thời gian S2 (ms):

| S (kích thước) | K = 500 | K = 1000 | K = 2000 | K = 4000 |
|---|---:|---:|---:|---:|
| 1.00 (1000×700) | 0.173 | 0.183 | 0.213 | 0.274 |
| 0.75 (750×525) | 0.164 | 0.184 | 0.224 | 0.311 |
| 0.50 (500×350) | 0.109 | 0.122 | 0.158 | 0.199 |
| 0.25 (250×175) | 0.070 | 0.067 | 0.067 | 0.067 |

Thời gian S3 (ms):

| S | K = 500 | K = 1000 | K = 2000 | K = 4000 |
|---|---:|---:|---:|---:|
| 1.00 | 0.087 | 0.111 | 0.166 | 0.265 |
| 0.75 | 0.098 | 0.135 | 0.212 | 0.356 |
| 0.50 | 0.083 | 0.118 | 0.195 | 0.342 |
| 0.25 | 0.073 | 0.108 | 0.119 | 0.121 |

Ở S = 0.25, ảnh `wall` chỉ có 1006 maxima. Vì vậy K ≥ 2000 không làm tăng chi phí: rank và describe chỉ chạy trên số điểm thực có. Trước tối ưu, tổng S2 + S3 cho ô S = 1, K = 4000 là 0.949 ms. Số đo đầy đủ của cả `graf` và `wall` trước và sau tối ưu nằm trong `results/s2s3-timings-before.csv` và `results/s2s3-timings.csv` trên runtime Colab. Các file này không được commit.

Đối chiếu một lượt chạy đơn của driver trên 16 cặp Oxford (S = 0.5, K = 1000, một lần chạy, có cả overhead lần đầu):
- S2 + S3 cho cả hai ảnh: 0.88–1.38 ms;
- S4 + S5: 5.6–16.5 ms.

Đây chưa phải số đo theo ranh giới `t_e2e` của conventions. Nhưng số này cho thấy với cấu hình thử, phần lớn thời gian nằm ở S4/S5, không ở S2/S3.

Các số trên chỉ dùng cho kỹ thuật, chưa dùng trong paper: chưa đo theo harness của Tuấn, chưa trên UDIS-D, và chưa lặp trong cùng phiên với các stage khác.

## 5. Chưa kiểm chứng

- Chạy trên UDIS-D và tập development của Nhóm.
- Ghép vào root build và harness, đo `t_s2`, `t_s3` theo ranh giới chung.
- Ảnh độ phân giải cao (1080p trở lên) và K > 4000.
- Ảnh hưởng của K lên chất lượng H và lên thời gian S4 trên dữ liệu benchmark.
- Build bằng CUDA 12.x. Lần chạy này dùng CUDA 13.0 có sẵn trên Colab.
