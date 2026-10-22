# S6 và QoR — bàn giao triển khai P1 của Chương

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-22
- Verification Status: UNVERIFIED
- Version Label: s6_p1_implementation_v1
- Phạm vi bằng chứng: test Python và geometry C++ đã chạy; CUDA chưa biên dịch/chạy, chưa có kết quả T4. Nhãn trên áp dụng cho toàn bộ bàn giao, không phủ nhận các kiểm tra cục bộ bên dưới.

Ngày cập nhật: 22/10 (Thứ 5), năm 2026. Người phụ trách: Chương; AI hỗ trợ triển khai theo yêu cầu. Đây là bản P1 ban đầu, chưa phải nghiệm thu P1 hay kết quả thực nghiệm cho paper.

## 1. Việc có thể làm trước và phạm vi đã triển khai

Theo [plan](plan.md), [stages](stages.md) và [phương pháp S6](s6-warp-blend.md), geometry, sampling, blender và evaluator có thể phát triển bằng dữ liệu tổng hợp, độc lập với tiến độ kernel S1–S5. Khi tích hợp, S5 phải trả H chiều B → A trong hệ RGB gốc; S6 không tự suy đoán hoặc đổi scale lần nữa.

| Thứ tự | Sản phẩm hiện có | Phần còn phải xác nhận |
|---|---|---|
| 1 | `src/s6/geometry.*`: kiểm tra H, canvas/origin, policy; không cấp phát ảnh | Chạy cùng H thực do S5 trả, thống nhất resource budget toàn pipeline |
| 2 | `src/s6/s6.cu`: inverse warp, bilinear, mask riêng, stride và single-source | Đối chiếu CUDA–FP64, kiểm tra memory trên T4 |
| 3 | Feather fuse warp/blend, workspace bằng 0 | Chi phí và chất lượng trên dữ liệu thật |
| 4 | Multiband 3/5, pull–push extension, ownership, pyramid/reconstruction | Sai số màu và peak memory thực; tối ưu filter sau profiling |
| 5 | Reference FP64, fixture generator, test CUDA và runner timing riêng S6 | Chạy `nvcc`, GPU golden tests, sanitizer và profiling |
| Song song | `tools/qor`: PSNR/SSIM overlap, coverage, CSV theo schema hiện tại | Đế review policy, miền SSIM và threshold trước khi đóng băng |

Không sửa `src/common/`, root `CMakeLists.txt`, plan hoặc CSV chung. S6 dùng `ImageView`, `Status`, `cudaStream_t` hiện có; header riêng của stage nằm trong `src/s6`. Root pipeline chưa gọi S6 mới: Tuấn cần tích hợp target `stitch_s6` sau khi review. Chưa triển khai cache giữa các cấu hình B, controller hoặc tối ưu hiệu năng cuối cùng.

## 2. Contract để tích hợp

API ở [geometry.hpp](../src/s6/geometry.hpp) và [s6.hpp](../src/s6/s6.hpp):

```cpp
stitch::s6::Geometry geometry;
stitch::s6::GeometryPolicy policy;
auto status = stitch::s6::prepare_geometry(
    a.width, a.height, b.width, b.height, H_original_B_to_A, policy, &geometry);
// Check status; choose the mode requested by the caller/controller.
std::size_t bytes = 0;
status = stitch::s6::s6_workspace_bytes(geometry, mode, &bytes);
// Check status, budget all buffers, then allocate/reuse them in the caller.
// output: RGB8 [geometry.height, geometry.width], pitch in bytes.
// mask: GRAY8 with the same shape; values are 0 or 1.
// workspace: nullptr for zero bytes; otherwise at least bytes, aligned to 256.
status = stitch::s6::render(a, b, geometry, mode, output, mask,
                          workspace, bytes, stream);
// Check launch status, then check stream completion before consuming/releasing.
```

Đây là đoạn minh họa chuỗi gọi, không phải hàm hoàn chỉnh: caller phải xử lý mọi `Status`, lỗi cấp phát và lỗi hoàn tất CUDA. Test [test_s6_cuda.cu](../tests/test_s6_cuda.cu) có ví dụ quản lý device buffer, stream và event thực tế.

`a/b/output/mask` là **device views**; pitch tính bằng byte, cho phép padding. A/B chỉ đọc và có thể trùng nhau. Output/mask/workspace không được chồng nhau hoặc chồng input. `render()` không `cudaMalloc`, không đồng bộ và không copy ảnh về host; mọi buffer phải sống đến khi stream hoàn tất. `Status::OK` khi trả về chỉ xác nhận validation/launch đã qua, không xác nhận hoàn tất GPU. Các lỗi async được caller kiểm tra theo [CUDA Programming Guide 12.5](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution).

`origin_x/origin_y` là tọa độ trong A của pixel canvas `(0,0)`. Offset A → canvas là **`(-origin_x,-origin_y)`**. Ví dụ A/B đều 4×3, B dịch x = −2: canvas 6×3, origin `(−2,0)`, offset `(2,0)`. Không nhầm hai đại lượng khi ghi metadata.

## 3. Quy tắc số học và blender

- H 3×3 row-major; normalize theo hệ số có trị tuyệt đối lớn nhất, không chia riêng `h33`. H và −H tương đương. Loại non-finite, singular, horizon cắt ảnh B và canvas vượt budget.
- Conditioning được kiểm tra sau khi đưa mỗi ảnh về tâm và scale đẳng hướng `max(W−1,H−1,1)`: `||Hn||inf * ||Hn^-1||inf < 1/inverse_tolerance`. Mặc định inverse/denominator tolerance `1e-12`; snap tuyệt đối `1e-10` pixel có floor `16*epsilon*max(1,abs(x))`. Đây là guard số học, không phải Qmin.
- Canvas tính trên miền tâm pixel `[0,W−1]×[0,H−1]`, floor/ceil có snap. Budget mặc định `16,777,216` pixel chỉ giới hạn canvas; caller còn phải cộng RGB/mask/workspace và các stage khác trước cấp phát.
- **CUDA dùng FP64 cho tọa độ warp, FP32 cho màu/filter/pyramid**. Bản đầu ưu tiên correctness; chưa dùng texture sampling hoặc fast-math. Reference độc lập dùng NumPy FP64 và bilinear tự tính.
- Validity tách khỏi màu: RGB đen vẫn là pixel hợp lệ. Mask union uint8 **0/1**. Một nguồn thì giữ warped color của nguồn đó; ngoài union RGB/mask bằng 0.
- Feather dùng khoảng cách tương đối tới biên nguồn theo §5.3 tài liệu phương pháp, không phải distance transform của mask canvas. Multiband chọn ownership hình học, tie chọn A.
- `Multiband3/5` là **3/5 lần REDUCE**, tức 4/6 mức lưu trữ khi canvas đủ lớn. `effective_reductions()` cap khi tới 1×1. Pull–push extension của cả hai mode vẫn đi tới 1×1, không dừng theo B.
- REDUCE: kernel `[1,4,6,4,1]/16`, ceil-half, lấy mẫu chẵn; EXPAND: zero insertion, filter ×4 trong 2D rồi crop về shape cha. Border reflect-101, xử lý chiều dài 1. Bản CUDA đầu dùng vòng lọc 5×5 trực tiếp; tách pass/fusion là tối ưu sau khi có số đo.
- Extension chỉ phục vụ filter, không tăng coverage. Chưa gamma/gain correction. Chỉ clamp/round output bằng `floor(clamp(255*C,0,255)+0.5)`, không clamp Laplacian band.

Gaussian/Laplacian blending dựa trên Burt & Adelson và Brown & Lowe; công thức weight, ownership, extension và ngưỡng số học là **policy của dự án**, không tuyên bố tái lập nguyên bản hoặc bit-identical OpenCV. Nguồn và mức đọc nằm tại [§10 tài liệu phương pháp](s6-warp-blend.md#102-tài-liệu-tham-khảo-và-mức-đọc).

## 4. Bằng chứng đã chạy tại máy phát triển

Môi trường kiểm tra: Windows, Python 3.14, NumPy 2.5.4, OpenCV 5.0.0, pytest 9.1.1; host C++ qua WSL Ubuntu/g++ 13.3, C++17. Dependency Python lấy từ `tools/requirements.txt` trong venv riêng, không thay requirements chung.

```powershell
python -m pytest tests/python -q
python -m tools.s6_reference.generate_fixtures --output data/s6-fixtures --include-mock
```

Lần chạy cuối: **95 test Python pass** (32 reference S6, 55 QoR, 8 mock cũ). Reference kiểm tra geometry/projective, H scale/sign, mẫu bilinear giải tích độc lập, black pixel, nguồn đơn, khoảng trống, shape lẻ/1D, reduce/expand, extension, stride và fixture reproducibility. Test QoR bao gồm policy/CSV/SSIM/coverage/reference failure và ngăn ghi đè input. Không lấy số assertion từ đồ án cũ.

Geometry C++ đã build và chạy với `-Wall -Wextra -Wpedantic -Werror`. CMake standalone ở chế độ không CUDA cũng build và CTest host **1/1 pass**. Một probe host tạm còn đối chiếu canvas/origin C++ với cả 27 header fixture FP64: khớp toàn bộ. Máy có GTX 1650 nhưng **chưa có `nvcc`**; các kết quả này không xác nhận `s6.cu` biên dịch, không xác nhận tolerance màu hay hiệu năng GPU.

Generator tạo **27 fixture**: 8 trường hợp tổng hợp × 3 B, cộng 1 cặp texture tổng hợp đi qua OpenCV mock S1–S5 rồi dùng H ước lượng × 3 B. `meta.json` ghi version/source hashes; `manifest.sha256` ghi hash từng fixture. Test sinh lặp kiểm tra hash giống nhau trong cùng môi trường. Đây là tích hợp với **mock CPU**, chưa phải CUDA end-to-end hoặc HPatches benchmark. Fixture nằm trong `data/` được gitignore.

## 5. Các lệnh bắt buộc trên Colab T4

Chạy từ root repo trong môi trường Linux có T4, toolkit CUDA và CMake ≥3.18. Ghi `nvidia-smi`, `nvcc --version`, commit SHA và dependency versions vào log lần chạy.

```bash
python -m pip install -r tools/requirements.txt
python -m pytest tests/python -q
cmake -S src/s6 -B build/s6 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/s6 -j 2
test -x build/s6/test_s6_cuda
ctest --test-dir build/s6 --output-on-failure
python -m tools.s6_reference.generate_fixtures --output data/s6-fixtures --include-mock
(cd data/s6-fixtures && sha256sum -c manifest.sha256)
build/s6/test_s6_cuda --golden data/s6-fixtures/*.bin
compute-sanitizer --tool memcheck --error-exitcode 1 build/s6/test_s6_cuda --golden data/s6-fixtures/*.bin
compute-sanitizer --tool racecheck --error-exitcode 1 build/s6/test_s6_cuda --golden data/s6-fixtures/*.bin
```

Kỳ vọng: build có cả `stitch_s6` và `test_s6_cuda`; host/native CUDA/golden tests pass; sanitizer không báo lỗi. CUDA test trả 77 khi không có device và CTest đánh dấu skip: **skip hoặc chỉ host pass không phải nghiệm thu GPU**. Geometry/origin/mask phải khớp tuyệt đối; RGB sai lệch tối đa **2 mức uint8** là ngưỡng chấp nhận ban đầu đã đặt trong harness, **chưa xác nhận trên GPU**. Nếu không đạt, điều tra precision/filter trước khi đổi ngưỡng; không tự tăng tolerance để tạo pass.

Tiếp theo mới kiểm tra ảnh thật: H từ S5, RGB có padding, exposure khác, overlap hẹp, texture thấp, đường thẳng, moving objects/parallax. Lưu panorama/crop/mask và thông tin failure. S6 không sửa được mọi lỗi parallax hoặc H sai chỉ bằng tăng B.

## 6. Đo chi phí từng B

```bash
mkdir -p results
build/s6/test_s6_cuda --timings results/s6-timings.csv --golden data/s6-fixtures/*.bin
```

Runner kiểm tra golden trước khi đo; dùng buffer/stream/geometry/workspace đã chuẩn bị, **2 warmup + 10 repeat** cho mỗi fixture. `gpu_ms` dùng CUDA event bao quanh toàn bộ render; `host_ms` bao gồm gọi render, ghi event và chờ hoàn tất. CSV còn có fixture, mode, shape, repeat, warmup, workspace bytes và GPU name. Loại warmup khi tổng hợp; ảnh tổng hợp nhỏ chỉ kiểm tra harness, chưa đủ cho kết luận hiệu năng.

Ranh giới này đo **S6 steady-state**, không gồm geometry, cấp phát, truyền H2D/D2H hoặc validation download. CUDA-event interval có thể gồm khoảng trống giữa các lần host dispatch, không phải tổng thời gian từng kernel riêng lẻ. Không đưa trực tiếp vào `t_e2e`, không thay `runs.csv` chung. Benchmark pipeline cần đo lại toàn ranh giới theo conventions, bao gồm chi phí policy/cache thực tế. Hiện chưa có số ms, speedup, VRAM peak hoặc Pareto frontier để báo cáo.

## 7. QoR độc lập và việc cần Đế review

Hướng dẫn đầy đủ và schema ở [tools/qor/README.md](../tools/qor/README.md). Chạy:

```bash
python -m tools.qor.evaluate --manifest data/pairs.json --runs results/runs.csv --policy tools/qor/policy-pilot.json --ref-config S1_K2000_R1024_Bfeather --output results/qor.csv
```

Evaluator đọc RGB gốc và H từ CSV; không dùng panorama sau blend làm input. Miền chuẩn Ω cố định theo H reference trong ảnh A; miền ứng viên V = Ω ∩ validity(Hcandidate). Coverage = `|V|/|Ω|`. PSNR dùng mọi sample RGB trên V. SSIM dùng Gaussian 11×11, sigma 1.5, chỉ nhận tâm có **toàn bộ cửa sổ nằm trong V**; không đủ cửa sổ trả failure rõ ràng. Cách cửa sổ này là quy tắc pilot của dự án; không đồng nhất với mọi thư viện SSIM. Định nghĩa SSIM nền tảng xem [Wang et al., 2004, bản tác giả](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf).

`policy-pilot.json` ghi `pilot-not-team-approved`: coverage ≥0.95, SSIM drop ≤0.02, PSNR drop ≤0.5 dB. Không gọi đây là ngưỡng chính thức UDIS-D hoặc của Nhóm. Đế cần review miền so sánh, trường hợp reference PSNR = +∞, overlap quá nhỏ, reference unreliable và thiên lệch khi candidate chỉ giữ phần dễ. QoR hiện tại kiểm tra alignment; chưa chứng nhận seam/blending quality. Khi so S/K/R phải giữ B cố định cho đến khi có phép đo render được duyệt.

## 8. Điều kiện đóng task

| Điều kiện | Trạng thái bàn giao |
|---|---|
| Geometry/reference/evaluator và regression Python/host | Có mã và bằng chứng cục bộ |
| CUDA compile/native/golden/sanitizer trên T4 | Chưa chạy |
| Đế duyệt QoR policy và Tuấn review tích hợp interface/target | Chờ review; chưa sửa common thay thành viên khác |
| S6 với dữ liệu thật/H thực từ S5, kiểm tra crop/seam/failure | Chưa có bằng chứng |
| Timing từng B, peak memory và benchmark toàn pipeline | Có runner riêng S6; chưa có số đo |

Ưu tiên tiếp theo là chạy chuỗi T4 ở §5, sửa lỗi nếu phát hiện, rồi tích hợp root pipeline và đo trên dữ liệu development. Chỉ khi đủ các bằng chứng này mới đánh dấu hai sản phẩm P1 hoàn tất; không suy ra P1 đã xong từ số test CPU.
