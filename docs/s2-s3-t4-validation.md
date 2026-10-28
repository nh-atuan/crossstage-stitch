# S2/S3: bằng chứng trên Colab T4

- Người chạy: Thành, AI hỗ trợ chạy qua Colab MCP theo yêu cầu.
- Ngày chạy: 25/10 (Chủ nhật), năm 2026.
- Mã nguồn: nhánh `thanh/p1-s2-s3`. Golden và CUDA ở commit `a423bab`, pipeline mock ở `85fe756`, tối ưu ở `6ff852a`. Lượt chạy cuối dùng mã của `6ff852a`.
- Môi trường: Tesla T4 sm_75, driver 580.82.07, CUDA 13.0 (V13.0.88), CMake 3.31.10, Python 3.13.16, NumPy 2.1.3, `opencv-python` 5.0.0.93.

Đây là bằng chứng cho correctness của S2/S3 và cho lần nối đầu tiên với S4/S5. Chưa phải benchmark UDIS-D, chưa có latency end-to-end, và chưa có Go/No-Go.

Mục 1 tới 5 là lượt chạy Phase 1. Mục 6 là lượt chạy Phase 2 cho K động, có môi trường và commit riêng.

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

## 6. Phase 2: K động

- Ngày chạy: 28/10 (Thứ 4), năm 2026. Thành yêu cầu, AI hỗ trợ chạy qua Colab MCP.
- Mã nguồn: nhánh `thanh/p2-dynamic-k`, commit `16fbfe7`.
- Môi trường: Tesla T4 sm_75, driver 580.82.07, CUDA 13.0 (V13.0.88), CMake 3.31.10, Python 3.13.16, NumPy 2.1.3, `opencv-python` 5.0.0.93. Xung SM 1590/1590 MHz trước lượt đo đầu, 1575 MHz sau lượt đo thứ hai, nhiệt độ GPU 48–49 °C.
- Quyết định liên quan: [K động](decisions/s2-s3-dynamic-k.md).

### 6.1. Cách đưa code lên Colab

Code được đóng gói bằng `git archive` từ commit trên rồi upload lên runtime. Vì máy Windows bật `core.autocrlf`, gói chứa file xuống dòng CRLF. Các file text được đổi về LF ngay trên runtime. Sau đó hash tổng của 153 file (SHA-256 của danh sách đường dẫn kèm SHA-256 từng file) khớp với cây của commit trong git. Dataset Oxford và golden được sinh lại trên runtime như Phase 1.

Một lượt chạy trước đó trong cùng ngày, trên runtime khác, cho cùng kết luận. Runtime đó bị reset nên file kết quả bị mất. Mọi số trong mục này chỉ lấy từ lượt chạy sau.

### 6.2. Lệnh

```bash
python -m tools.s2_s3.export_golden --output data/s2s3-golden
python -m tools.s2_s3.export_timing_golden --output data/s2s3-timing \
    --images /content/oxford/graf/img1.ppm /content/oxford/wall/img1.ppm
cmake -S src/s2_s3 -B build/s2_s3 -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build/s2_s3 -j2 && ctest --test-dir build/s2_s3 --output-on-failure
build/s2_s3/test_s2_s3_cuda --require-t4 --golden data/s2s3-golden/*.bin data/s2s3-timing/*.bin
build/s2_s3/test_s2_s3_cuda --require-t4 --leak-iterations 0 \
    --dynamic-timings results/s2s3-dynamic-k.csv --golden data/s2s3-timing/*.bin
build/s2_s3/test_s2_s3_cuda --require-t4 --leak-iterations 0 \
    --dynamic-timings results/s2s3-dynamic-k-run2.csv --golden data/s2s3-timing/*.bin
python -m tools.s2_s3.compare_dynamic_timings results/s2s3-dynamic-k.csv --gate max view
python -m tools.s2_s3.compare_dynamic_timings results/s2s3-dynamic-k-run2.csv --gate max view
python -m tools.s2_s3.check_mock_pipeline build/pipe/s2_s3_mock_driver results/s2s3-oxford data/mock/*
CASES="data/s2s3-golden/odd_k20000_all.bin data/s2s3-golden/tiny_k100.bin \
       data/s2s3-timing/wall_img1_s1_k4000.bin data/s2s3-timing/graf_img1_s0.5_k500.bin"
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
    build/s2_s3/test_s2_s3_cuda --require-t4 --leak-iterations 50 --golden $CASES
# racecheck, initcheck, synccheck: cùng lệnh, không có --leak-check
```

Cách dựng `data/mock` và `build/pipe` giống mục 3.

### 6.3. Correctness và bộ nhớ

| Kiểm tra | Kết quả |
|---|---|
| Test Python của S2/S3 (bản tham chiếu, script so timing) | 24 passed |
| CTest (host + contract CUDA) | 2/2 pass |
| 41 golden case của Phase 1 | 41/41 pass: vị trí keypoint 0 lỗi, response sai lệch 0, blur 0 pixel lệch, descriptor khớp bit 100% |
| `dynamic_k_checks`: 9 attempt trên một bộ buffer cấp cho 1000×700 và K = 4000, đổi cả K (0 tới 4000) lẫn kích thước ảnh (40×30 tới 1000×700), chạy với cả `capacity = 4000` và `capacity = K`, so với lần chạy cấp vừa đúng | Giống từng bit ở mọi attempt |
| `cudaMemGetInfo` trước và sau 1000 lần gọi S2 + S3 với K luân phiên | Bộ nhớ free bằng nhau |
| Ca biên: ảnh lớn hơn kích thước đã cấp; K = capacity + 1 | `INSUFFICIENT_WORKSPACE` ở cả S2 và S3; `INVALID_ARGUMENT`; `d_count` không bị ghi |
| compute-sanitizer trên 4 case, 50 lần lặp K động | memcheck: 0 lỗi, leak 0 byte; racecheck: 0 hazard; initcheck: 0 lỗi; synccheck: 0 lỗi |
| 16 cặp Oxford, S = 0.5, K = 1000 | Keypoint, match, inlier và sai số giống bảng ở mục 3 tới 3 chữ số thập phân; 13/16 PASS, cùng ba cặp fail |

Lần chạy initcheck đầu tiên trong ngày tìm ra lỗi ở code test mới, không phải ở S2/S3. Với `capacity = 0`, S3 trả về trước khi blur, nhưng test vẫn copy vùng blur chưa ghi về host. Commit `16fbfe7` sửa test để bỏ bước copy đó khi capacity bằng 0. Không có điều kiện kiểm tra nào bị nới.

Leak check của sanitizer dùng 50 lần lặp cho đỡ tốn thời gian. Phép đo 1000 lần lặp bằng `cudaMemGetInfo` chạy không có sanitizer.

Runtime chạy các phép đo trên bị ngắt kết nối khi racecheck đang chạy, nên cả 4 sanitizer được chạy lại trên một runtime Tesla T4 khác trong cùng ngày, với cùng commit và cùng hash tổng 153 file. Golden được sinh lại trên runtime đó. Racecheck mất khoảng 19 phút, ba tool còn lại mỗi tool dưới 10 giây.

### 6.4. Thời gian khi cấp theo K_max

Mỗi ô S×K được đo theo ba cách cấp phát, xen kẽ nhau trong cùng một vòng lặp: 2 lần warmup, 10 lần đo, lấy median.

| Cách cấp | Buffer và workspace | `Keypoints.capacity` |
|---|---|---|
| `exact` | Vừa đúng ảnh và K của ô, như Phase 1 | K |
| `max` | Theo ảnh lớn nhất và K = 4000 | 4000 |
| `view` | Theo ảnh lớn nhất và K = 4000 | K |

Ngưỡng đạt: median của `max` hoặc `view` không chậm hơn `exact` quá max(5%, 0.005 ms), xét riêng S2 và S3 ở từng ô. Có hai lượt đo liên tiếp, SHA-256 của file kết quả:

- `results/s2s3-dynamic-k.csv`: `10e0bf554c3e15421cdd5cb4df2c408fd57a04402cb8cab12aa78b91b4837208`;
- `results/s2s3-dynamic-k-run2.csv`: `479c45a00903caa02bde3acf6c3d854f1ef399151a704c2b798f2679b036e877`.

Hai file nằm trên runtime Colab và không được commit.

Tổng hợp từ output của `compare_dynamic_timings.py`:

| Cách cấp | S2 | S3 lượt 1 | S3 lượt 2 |
|---|---|---|---|
| `max` | Đạt 32/32 ô ở cả hai lượt | Không đạt 6/32 ô | Không đạt 7/32 ô |
| `view` | Đạt 32/32 ô ở cả hai lượt | Đạt 32/32 ô | Đạt 32/32 ô |

Mọi ô không đạt đều ở K = 1000. Các dòng K = 1000 của S3, chép từ output của script (ms, Δ so với `exact`):

| Ảnh | W×H | Lượt | S3 `exact` | Δ `max` | Kết quả `max` | Δ `view` | Kết quả `view` |
|---|---|---|---:|---:|---|---:|---|
| graf | 200×160 | 1 | 0.0473 | +0.0006 | PASS | +0.0010 | PASS |
| graf | 200×160 | 2 | 0.0478 | +0.0011 | PASS | -0.0001 | PASS |
| graf | 400×320 | 1 | 0.0741 | +0.0078 | FAIL | +0.0011 | PASS |
| graf | 400×320 | 2 | 0.0744 | +0.0065 | FAIL | -0.0006 | PASS |
| graf | 600×480 | 1 | 0.0841 | +0.0049 | PASS | -0.0021 | PASS |
| graf | 600×480 | 2 | 0.0840 | +0.0060 | FAIL | -0.0014 | PASS |
| graf | 800×640 | 1 | 0.0951 | +0.0075 | FAIL | +0.0012 | PASS |
| graf | 800×640 | 2 | 0.0947 | +0.0076 | FAIL | +0.0016 | PASS |
| wall | 250×175 | 1 | 0.0709 | +0.0070 | FAIL | +0.0008 | PASS |
| wall | 250×175 | 2 | 0.0707 | +0.0070 | FAIL | +0.0008 | PASS |
| wall | 500×350 | 1 | 0.0770 | +0.0070 | FAIL | +0.0008 | PASS |
| wall | 500×350 | 2 | 0.0776 | +0.0065 | FAIL | +0.0001 | PASS |
| wall | 750×525 | 1 | 0.0873 | +0.0088 | FAIL | +0.0007 | PASS |
| wall | 750×525 | 2 | 0.0869 | +0.0093 | FAIL | +0.0012 | PASS |
| wall | 1000×700 | 1 | 0.1062 | +0.0078 | FAIL | -0.0008 | PASS |
| wall | 1000×700 | 2 | 0.1062 | +0.0070 | FAIL | -0.0016 | PASS |

Với `capacity = 4000`, S3 luôn launch 1000 block (4 warp mỗi block), dù K nhỏ. Các block thừa chỉ đọc `d_count` rồi thoát. Đây là nguyên nhân hợp lý nhất của phần chậm thêm, vì `view` bỏ các block này và đạt ở mọi ô. Nhưng giả thuyết này chưa được profile, và chưa giải thích được vì sao K = 500 (nhiều block thừa hơn) vẫn đạt ở cả hai lượt.

Kết luận: S2 và S3 nhận K động trên bộ nhớ cấp một lần mà không cấp phát lại, không rò rỉ và không đổi output. Để không mất hiệu năng ở S3, harness phải truyền view `capacity = K`. Hướng dẫn cho harness nằm trong [triển khai S2/S3](s2-s3-p1-implementation.md), mục 7.

### 6.5. Chưa kiểm chứng ở Phase 2

- Harness thật của Tuấn với Controller đổi K giữa các attempt. Các test trên dùng `Runner` riêng của test CUDA.
- Ảnh 1080p trở lên, K > 4000 và UDIS-D.
- Nguyên nhân phần chậm thêm ở K = 1000 khi `capacity = 4000` (chưa chạy nsys hay ncu).
