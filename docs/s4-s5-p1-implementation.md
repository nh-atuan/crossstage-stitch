# S4/S5: triển khai P1 của Đế

- Người phụ trách: Đế.
- Ngày ghi nhận: 22/10 (Thứ 5), năm 2026.
- Cập nhật cấu trúc thư mục: 23/10 (Thứ 6), năm 2026, theo yêu cầu của Đế.
- Phase 1, deadline chung: 25/10 (Chủ nhật).
- Trạng thái: đã build và chạy test S4/S5 trên Colab Tesla T4 ngày 23/10 (Thứ 6). CTest, memcheck/racecheck và mock tổng hợp ORB/SIFT pass; xem [bằng chứng T4](s4-s5-t4-validation.md). Chưa nghiệm thu toàn bộ Phase 1 trên benchmark thật.

Phạm vi theo [kế hoạch, mục 8](plan.md): S4 matching; S5 homography có núm R, tín hiệu tự kiểm tra và tái sử dụng khi chỉ tăng R. Controller thuộc Phase 2. Ghép đủ sáu stage và Go/No-Go còn phụ thuộc kernel của các thành viên, harness của Tuấn và kết quả trên development data.

## 1. Code và build

| Đường dẫn | Nội dung |
|---|---|
| `src/s4/s4.hpp`, `s4.cu`, `s4_config.cpp` | Top-2 matching, ratio test, cross-check tùy chọn, query workspace |
| `src/s5/s5.hpp`, `s5.cu`, `s5_config.cpp` | Batched MSAC, refinement, R extension, query workspace |
| `src/s5/s5_math.hpp`, `s5_internal.hpp` | Solver số học dùng chung host/device và layout workspace riêng S5 |
| `src/s4/CMakeLists.txt`, `src/s5/CMakeLists.txt` | Build độc lập của từng stage, không sửa CMake hoặc interface chung |
| `tests/s4_s5/CMakeLists.txt` | Build kiểm thử cả hai stage và mock driver |
| `tests/test_s4_config.cpp`, `test_s5_math.cpp` | Kiểm tra host, không cần CUDA |
| `tests/test_s4_cuda.cu`, `test_s5_cuda.cu` | Kiểm tra GPU; trả 77 nếu thiếu device |
| `tests/s4_s5_mock_driver.cu` | Chạy S4 → S5 trên cùng stream, không đọc count về host giữa hai stage |
| `tools/s4_s5/check_mock_pipeline.py` | OpenCV mock S1–S3 → CUDA S4/S5 → OpenCV mock S6 |

Root build chưa đưa các target này vào pipeline. Tuấn có thể dùng `add_subdirectory(src/s4)` và `add_subdirectory(src/s5)` rồi link `stitch_s4`, `stitch_s5`. Test host của mỗi stage chỉ link library config tương ứng; stage không phụ thuộc OpenCV hoặc thư viện của stage kia.

Đế yêu cầu tách source để mở rộng từng stage riêng. Phương án chọn là mỗi stage giữ header, implementation và CMake trong thư mục của mình; build phối hợp nằm trong `tests/s4_s5`, cùng nhóm với mock driver. Đổi lại, phần khai báo CMake của hai stage có một số dòng giống nhau. Không thêm lớp build dùng chung cho phần nhỏ này. Có thể dùng build kiểm thử cả hai stage khi cần chạy cùng một lệnh; API, thuật toán, tham số và cơ chế reuse R giữ nguyên.

Build riêng từng stage:

```bash
cmake -S src/s4 -B build/s4 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/s4 -j 2
ctest --test-dir build/s4 --output-on-failure
cmake -S src/s5 -B build/s5 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/s5 -j 2
ctest --test-dir build/s5 --output-on-failure
```

## 2. S4

Giữ phương pháp trong [quyết định Phase 0](benchmark-qor-s4-s5-decisions.md): exact brute-force top-2 theo tile. Mỗi block xử lý một query, mỗi thread giữ top-2 riêng rồi reduction. Descriptor binary dùng Hamming thông thường; descriptor float tích lũy squared L2 bằng double, xuất distance L2. Đây không phải Hamming2 cho ORB WTA_K 3/4.

Tie chọn index B nhỏ hơn. Ratio test dùng bất đẳng thức nghiêm ngặt; khi tính squared L2, bình phương ratio. Thiếu hai train descriptors hoặc hai distance cùng bằng 0 thì loại. `cross_check` kiểm tra nearest theo chiều ngược, không thêm reverse ratio test. API yêu cầu caller truyền ratio và cờ cross-check; không có tham số đã tune ngầm.

Compaction dùng prefix trong block, prefix của block totals, rồi scatter theo thứ tự A. Không dùng thứ tự atomic để sắp match. Workspace tuyến tính theo capacity A và, nếu bật cross-check, capacity B; không lưu ma trận distance A×B. Prefix của block totals hiện chạy tuần tự trên GPU, chưa có bằng chứng đây là điểm nghẽn.

`s4_knn_top2` cho phép đối chiếu cả nearest và second-nearest. `Top2` lưu squared L2; `Matches.distance` lưu L2. Header quy định stride/alignment, finite float descriptors, count trên device và các buffer không được chồng nhau.

## 3. S5

Giữ batched RANSAC với MSAC-style score trong quyết định Phase 0:

1. Kiểm tra count, match index và tọa độ hữu hạn trên device.
2. Với mỗi hypothesis index, lấy bốn index khác nhau từ counter hash có seed cố định. Loại sample có điểm trùng hoặc bộ ba gần thẳng hàng ở A hoặc B.
3. Chuẩn hóa tọa độ về centroid và RMS distance √2. Giải hệ tám ẩn bằng normal equations FP64 và partial pivoting, rồi đưa H về hệ S1.
4. Chấm `sum(min(forward_reprojection_error², threshold²))` bằng reduction cố định. Loss bằng nhau chọn hypothesis index nhỏ hơn.
5. Refit trên inliers, chỉ nhận model có loss giảm và miền projective hợp lệ; lặp tối đa `refinement_steps`.
6. Xuất mask, telemetry, `H_s` và `H = D_A⁻¹ H_s D_B`.

Solver này không phải SVD-DLT, symmetric-transfer fitting hoặc MAGSAC++. Normal equations có thể kém ổn định hơn SVD khi dữ liệu gần suy biến. Pivot nhỏ hơn `1e-12` lần diagonal lớn nhất bị loại; kiểm tra sample dùng relative area `1e-8`. Đây là guard số học, không phải QoR hay tham số đã học trên development data. Cần kiểm tra thêm trên matches thực trước nghiệm thu.

Các tham số `hypotheses`, `batch_size`, `refinement_steps`, `threshold`, `seed` đều do caller truyền. Threshold tính bằng pixel trong hệ S1, không tự đổi thành pixel gốc. `batch_size` không quá 65535 để mỗi hypothesis dùng một block chấm điểm trong một launch. Không có adaptive early stopping; R là số sample attempts, kể cả sample suy biến.

Telemetry:

- `num_matches`: số match đầu vào.
- `num_inliers`, `inlier_ratio`: theo forward residual không vượt threshold.
- `reproj_residual`: RMS forward residual của inliers, đơn vị pixel S1.
- `inlier_coverage`: diện tích bounding box của inlier keypoints A, chia cho `(work_w−1)(work_h−1)` và chặn trong [0,1]. Đây không phải overlap coverage của evaluator, cũng không phải tỷ lệ grid cells có inlier.

H suy biến, không đủ bốn inliers hoặc horizon cắt miền B bị báo lỗi. H lỗi được ghi NaN, mask được xóa. Không thay H lỗi bằng identity.

### Hợp đồng bất đồng bộ

Hai stage không cấp phát, không đồng bộ và nhận stream của caller. Counts vẫn nằm trên device từ S4 sang S5. Output `HomographyResult` của S5 phải là pinned host memory do caller cấp phát bằng `cudaMallocHost`; S5 enqueue D2H trên stream đó. Chỉ đọc output sau khi caller kiểm tra `cudaStreamSynchronize`.

Phân biệt hai loại status:

- Giá trị trả về từ `estimate`: lỗi tham số host, workspace, launch hoặc enqueue copy.
- `output.status` sau khi stream hoàn tất: lỗi count/index/tọa độ, stale reuse, không đủ điểm hoặc model suy biến.

Workspace S5 phải căn 256 byte và tồn tại qua các lần tăng R. Không dùng đồng thời workspace trên nhiều stream. Mọi input/output phải sống tới khi hoàn tất; không chồng writable buffers.

### Tăng R và reset

Lần đầu đặt `previous_hypotheses=0`. Ví dụ sau khi R=17 hoàn tất, đặt `previous_hypotheses=17`, `hypotheses=257` để chỉ sinh và chấm các hypothesis 17…256.

Workspace giữ best model thô, score và số attempts đã chạy. Refinement chỉ tạo output, không ghi đè best model thô. Counter hash phụ thuộc seed và hypothesis index, nên chia R thành nhiều lần không đổi sample sequence hoặc tie-break. GPU test so H, mask và toàn bộ telemetry của R=257 một lần với R=17 rồi tăng lên 257, qua biên batch.

Giữ nguyên tọa độ, match ordering/counts, buffers, scale metadata, batch size, threshold và seed khi extend. `generation` là token do caller quản lý: đổi khi ghi lại dữ liệu vào cùng buffer. S5 kiểm tra token và metadata, không hash toàn bộ nội dung device buffer. Khi thay S/K/matches, đặt `previous_hypotheses=0`; sau lỗi CUDA cũng phải reset. Không khởi đầu bằng extension trên workspace chưa được khởi tạo.

## 4. Kiểm chứng local

Máy phát triển: Linux x64, g++ 15.2.0. Không có `nvcc`, GPU, CMake, `clangd`, `basedpyright` hoặc module Python `cv2`. Không thêm dependency hoặc sửa cấu hình chung.

Đã chạy:

```bash
mkdir -p build/s4_s5-host
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -Isrc -Itests \
  tests/test_s4_config.cpp src/s4/s4_config.cpp \
  -o build/s4_s5-host/test_s4_config
build/s4_s5-host/test_s4_config
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -Isrc -Itests \
  tests/test_s5_math.cpp src/s5/s5_config.cpp \
  -o build/s4_s5-host/test_s5_math
build/s4_s5-host/test_s5_math
python3 -m py_compile tools/s4_s5/check_mock_pipeline.py
```

Hai executable host pass; script Python qua kiểm tra cú pháp. `test_common` hiện có cũng pass. Solver host còn được build/chạy với `-fsanitize=address,undefined -fno-omit-frame-pointer`, không có báo lỗi.

Ở lần kiểm tra local ban đầu, lệnh `python3 -m pytest tests/python/test_qor.py -q` dừng ở collection vì `ModuleNotFoundError: No module named 'cv2'`. Diagnostics LSP không chạy được vì thiếu `clangd` và `basedpyright`. Sau đó bộ test Python đã được chạy trên Colab, kết quả ở [bằng chứng T4](s4-s5-t4-validation.md).

Host tests kiểm tra workspace/config, homography projective đã biết, sample bốn điểm, suy biến, sample counter xác định và chuyển hệ half-pixel. Chúng không chứng minh CUDA biên dịch, GPU correctness, reuse trên GPU hoặc hiệu năng.

## 5. Quy trình kiểm thử trên Colab T4

Quy trình này đã được chạy trên revision `aaa937a`; kết quả và môi trường Python riêng được ghi tại [bằng chứng T4](s4-s5-t4-validation.md).

Từ root repo, kiểm tra `nvidia-smi` hiển thị Tesla T4, ghi version CUDA và commit SHA vào log. Cài các dependency đã khai báo của Nhóm nếu phiên Colab chưa có:

```bash
python -m pip install -r tools/requirements.txt
cmake -S tests/s4_s5 -B build/s4_s5 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/s4_s5 -j 2
test -x build/s4_s5/test_s4_cuda
test -x build/s4_s5/test_s5_cuda
ctest --test-dir build/s4_s5 --output-on-failure
build/s4_s5/test_s4_cuda
build/s4_s5/test_s5_cuda
compute-sanitizer --tool memcheck --error-exitcode 1 build/s4_s5/test_s4_cuda
compute-sanitizer --tool memcheck --error-exitcode 1 build/s4_s5/test_s5_cuda
compute-sanitizer --tool racecheck --error-exitcode 1 build/s4_s5/test_s4_cuda
compute-sanitizer --tool racecheck --error-exitcode 1 build/s4_s5/test_s5_cuda
python -m pytest tests/python -q
```

Kỳ vọng: CTest có bốn test pass, không skip; executable GPU in `S4 CUDA oracle and contract checks passed` và `S5 CUDA tests passed`; sanitizer không báo lỗi. Test S5 in seed 20261022. Exit 77, chỉ có hai test host, hoặc thiếu executable GPU đều không phải nghiệm thu CUDA.

Nếu không dùng CMake, build riêng S5:

```bash
nvcc -std=c++17 -arch=sm_75 -Isrc -Itests \
  src/s5/s5.cu src/s5/s5_config.cpp tests/test_s5_cuda.cu \
  -o build/test_s5_cuda
build/test_s5_cuda
```

### Chạy với mock của Nhóm

Sau khi có HPatches tại `data/hpatches-sequences-release`:

```bash
python tools/mock/gen_mock.py \
  --hpatches data/hpatches-sequences-release --out data/mock \
  --manifest data/mock/manifest.sha256
python -m tools.s4_s5.check_mock_pipeline \
  build/s4_s5/s4_s5_mock_driver data/mock/v_graffiti_1_2 orb results/s4_s5-orb
python -m tools.s4_s5.check_mock_pipeline \
  build/s4_s5/s4_s5_mock_driver data/mock/v_graffiti_1_2 sift results/s4_s5-sift
```

Runner tính lại mock S1–S3 từ RGB, chuyển tọa độ/descriptors sang driver CUDA, so S4 với OpenCV BFMatcher, kiểm tra H gốc, rồi warp/blend bằng mock S6. Các giá trị S=0.5, ratio=0.8, R=2000, threshold=3 pixel S1 và seed=0 ở đây theo fixture mock; không phải cấu hình chính thức của stage.

Match indices phải khớp; distance tolerance `rtol=1e-5`, `atol=1e-4`. Với S5, không yêu cầu mask hoặc ma trận H giống bitwise OpenCV vì sampling/scoring khác nhau. Runner yêu cầu mean corner difference của CUDA H và OpenCV H không quá 3 pixel gốc, ghi thêm known-H error nếu fixture có ground truth. Tiêu chí này đã pass trên fixture tổng hợp có cấu trúc, chưa được kiểm chứng trên HPatches/UDIS-D thật và không thay quality policy của paper. Nếu fail phải lưu pair, điều tra trước khi đổi tolerance.

Kỳ vọng: runner in thông báo pass cho cả ORB và SIFT; `results/` chứa H, panorama, mask và report. Stage/reference thất bại làm lệnh fail, không tự bỏ pair. Driver ghi tên GPU và seed; chưa đo latency. Cần xem panorama/crop sau khi chạy, không chỉ đọc thông báo pass.

## 6. Review QoR của Chương

Đã đọc `tools/qor/evaluate.py`, test QoR và README, đối chiếu với quyết định Phase 0 và quy ước chung. Bộ test `tests/python` đã pass trên Colab trong môi trường OpenCV đúng bản pin; không sửa code của Chương.

Không thấy sai lệch chặn việc thử pilot qua review tĩnh: miền Ω lấy từ reference; coverage chia cho Ω; PSNR dùng RGB; SSIM chỉ dùng cửa sổ hoàn toàn hợp lệ; reference/pipeline failure giữ lại trong CSV. Reference unreliable không thể pass. Đây chưa phải phê duyệt quality policy để dùng trong paper.

Các điểm cần Đế, Chương và Nhóm xác nhận:

- Candidate dùng Ω∩valid, còn reference metric dùng toàn Ω. Mất tối đa 5% overlap vẫn có thể đổi tập pixel được chấm; README đã ghi selection bias này. Pilot cần xem các trường hợp sát ngưỡng coverage trước khi khóa protocol.
- Reference PSNR bằng infinity chỉ cho candidate PSNR infinity pass với tolerance hữu hạn. Đây là hệ quả của policy, không được tự thay infinity bằng số hữu hạn.
- Giá trị coverage 0.95, SSIM-drop 0.02 và PSNR-drop 0.5 dB vẫn là ngưỡng pilot. Chưa có số liệu để khóa.
- Bounding-box inlier coverage của S5 không dùng thay overlap coverage. Hai tín hiệu có ý nghĩa và miền tính khác nhau.

## 7. Trade-off, fallback và phần chưa nghiệm thu

Triển khai ban đầu dùng FP64 cho fitting/scoring. Generate hypotheses chạy song song; chấm residual chạy song song theo match. Validation dữ liệu, chọn winner giữa các batch và local refinement còn phần tuần tự trên GPU. Cần đo trên T4 trước khi quyết định tối ưu, không suy ra speedup từ cấu trúc kernel.

Nếu solver/refinement không ổn định, giữ phương án phụ đã ghi ở Phase 0: vanilla RANSAC với inlier-count score và một lần least-squares refinement, sau khi Đế xác nhận. Không tự thay method hoặc nới tolerance để test pass.

Đã kiểm chứng build CUDA, test và sanitizer GPU, mock end-to-end tổng hợp và ảnh preview của mock. Chưa nghiệm thu ảnh HPatches/UDIS-D thực, latency, ghép đủ sáu stage GPU, quality policy trên development data và Go/No-Go. Chưa có số đo hiệu năng cho paper.

## 8. Nguồn

Matching và solver được triển khai theo quyết định của Đế tại [Phase 0](benchmark-qor-s4-s5-decisions.md) và hợp đồng [quy ước chung](conventions.md). Nguồn RANSAC, local refinement và ratio test đã ghi trong tài liệu Phase 0.

Counter hash dùng công thức SplitMix64 của Sebastiano Vigna: [mã nguồn tác giả](https://prng.di.unimi.it/splitmix64.c), public domain, truy cập 22/10 (Thứ 5), năm 2026. Dùng phần add/xorshift/multiply để sinh sample theo counter, không dùng biến RNG toàn cục của nguồn. Phép modulo có bias nhỏ; chưa tuyên bố sample phân phối đều tuyệt đối. Seed/counter phục vụ tái lập mẫu.
