# S6 và QoR — bàn giao triển khai P1 của Chương

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-23
- Verification Status: UNVERIFIED
- Version Label: s6_p1_implementation_v2
- Phạm vi bằng chứng: Python/host và CUDA sm_75 đã chạy trên GTX 1650; chưa nghiệm thu T4 hoặc sanitizer. Nhãn trên áp dụng cho nghiệm thu chính thức toàn bộ P1, không phủ nhận các kiểm tra cục bộ bên dưới.

Ngày cập nhật: 23/10 (Thứ 6), năm 2026. Người phụ trách: Chương; AI hỗ trợ triển khai theo yêu cầu. Mã bàn giao đã có kiểm chứng GPU cục bộ; nghiệm thu chính thức P1 và số liệu paper vẫn chờ T4/review của Nhóm.

## 1. Việc có thể làm trước và phạm vi đã triển khai

Theo [plan](plan.md), [stages](stages.md) và [phương pháp S6](s6-warp-blend.md), geometry, sampling, blender và evaluator có thể phát triển bằng dữ liệu tổng hợp, độc lập với tiến độ kernel S1–S5. Khi tích hợp, S5 phải trả H chiều B → A trong hệ RGB gốc; S6 không tự suy đoán hoặc đổi scale lần nữa.

| Thứ tự | Sản phẩm hiện có | Phần còn phải xác nhận |
|---|---|---|
| 1 | `src/s6/geometry.*`: kiểm tra H, canvas/origin, policy; không cấp phát ảnh | Chạy cùng H thực do S5 trả, thống nhất resource budget toàn pipeline |
| 2 | Inverse warp, bilinear, mask riêng, stride và single-source; CUDA–FP64 đã đối chiếu | Kiểm tra memory trên T4 |
| 3 | Feather fuse warp/blend, workspace bằng 0; đã so cả mock OpenCV trên hai cặp ảnh thật | Đánh giá trên tập development của Nhóm |
| 4 | Multiband 3/5, pull–push extension, ownership, pyramid/reconstruction; GPU goldens pass | Peak memory thực và profiling T4; tối ưu sau số đo |
| 5 | Reference, importer mock, artifact export, telemetry và runner kiểm tra có log/hash | Chạy cùng quy trình trên T4 với sanitizer |
| Song song | `tools/qor`: evaluator + báo cáo đạt/không đạt và failure accounting | Đế review policy, miền SSIM và threshold trước khi đóng băng |

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

Tham số cuối tùy chọn `RenderStatistics* statistics=nullptr` là **device buffer 16 byte do caller cấp**, chứa `union_pixels` và `overlap_pixels`. Khi bật, render reset trên cùng stream rồi đếm từ validity bằng block reduction và tối đa hai atomic/block. Kiểm tra alignment/alias trước mọi device write; chỉ đọc counter sau checked completion. Caller có thể dùng counts cùng canvas, effective reductions và workspace query để điền telemetry chung. `nullptr` giữ đường không telemetry; chi phí counter phải được tính nếu bật trong benchmark, không cộng lẫn hai cách đo. Không thay struct common của Tuấn.

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

Môi trường kiểm tra: Windows/Python 3.14 và WSL/Python 3.12.3; NumPy 2.5.4, OpenCV 5.0.0, pytest 9.1.1; g++ 13.3, NVCC 12.5.82, CMake 4.4.4, `sm_75`. GTX 1650 4 GiB, driver Windows 592.00, CUDA runtime 12050/driver API 13010. Dependency Python lấy từ `tools/requirements.txt` trong thư mục riêng. Toolchain cục bộ lấy từ [NVIDIA redistrib 12.5.1](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.5.1.json), kiểm tra SHA256, đặt trong `build/`; không thay requirements/common.

```powershell
python -m pytest tests/python -q
python -m tools.s6_reference.generate_fixtures --output data/s6-fixtures --include-mock
```

Lần chạy cuối: **181 test Python pass**: 32 reference S6, 55 evaluator QoR, 51 report QoR, 17 importer mock, 18 validation runner, 8 mock cũ. Test bao gồm sampling giải tích, shape/stride/extension, failure, chống ghi đè input, metadata/fixture completeness và từ chối coi exit 77 là pass.

Ngày 22/10 đã kiểm tra geometry host với warning nghiêm ngặt và 27 header fixture. Ngày 23/10 build CUDA thực tế phát hiện lỗi NVCC không suy diễn được initializer-list shape trong test; đã sửa thành `std::array` có kiểu tường minh. Build NVCC hiện pass, CTest **2/2 pass** gồm native CUDA. Native test còn kiểm tra counter reset, alignment/alias, giữ counter khi input lỗi và aggregation nhiều block.

Generator tạo **27 fixture**: 8 trường hợp tổng hợp × 3 B, cộng 1 cặp texture tổng hợp đi qua OpenCV mock S1–S5 rồi dùng H ước lượng × 3 B. `meta.json` ghi version/source hashes; `manifest.sha256` ghi hash từng fixture. Test sinh lặp kiểm tra hash giống nhau trong cùng môi trường. S1–S5 ở đây là mock CPU; phép đối chiếu dưới đây kiểm chứng CUDA S6, chưa phải pipeline toàn CUDA hoặc benchmark HPatches. Fixture nằm trong `data/` được gitignore.

**27/27 CUDA goldens pass**, geometry/mask khớp tuyệt đối, sai lệch RGB lớn nhất quan sát được **1 mức uint8** (tolerance đã đặt vẫn ≤2). Đã xuất/đối chiếu 27 bộ PPM/PGM/JSON và 324 dòng timing (2 warmup + 10 repeat mỗi fixture).

Bổ sung ảnh thật từ [Oxford affine graffiti, nguồn VGG](https://www.robots.ox.ac.uk/~vgg/research/affine/), hai cặp ảnh 1–2 và 1–3, RGB 800×640. Đây **không phải** một lượt chạy HPatches hoặc UDIS-D. Gọi `tools.mock.gen_mock.gen_pair` trên ảnh/homography gốc để sinh mock S1–S5, sau đó importer nhận nguyên `orb_H.npy`/`sift_H.npy`. Bốn handoff × ba B tạo **12/12 GPU goldens pass**, mask exact, RGB max_abs = 1; không có handoff bị bỏ. Canvas ORB lần lượt 1253×922 và 1686×965. Feather CUDA so trực tiếp `s6_panorama.npy`/`s6_mask.npy` của mock OpenCV trên hai cặp này cũng có max_abs = 1, mask mismatch = 0. Đây là kết quả trên hai cặp cụ thể, chưa là bảo đảm sai số cho toàn dataset.

Evidence cục bộ trong `results/s6-validation-20261023-local/`, `results/s6-validation-20261023-oxford/` và `results/s6-validation-20261023-final/`: log, `validation.json`, ảnh thực CUDA, timing CSV. Dữ liệu/kết quả không commit. Receipt có commit, trạng thái working tree và source hashes, không giả định mã chưa commit giống một revision sạch.

Compute Sanitizer đã được thử và **không khởi tạo được WDDM debugger interface** trong môi trường hiện tại (`ERROR SUMMARY: 2 errors`, exit 1). Đây không phải sanitizer pass hoặc bằng chứng kernel sạch memory/race. Chưa chạy T4, chưa có nghiệm thu sanitizer.

## 5. Các lệnh bắt buộc trên Colab T4

Chạy từ root repo trong môi trường Linux có T4, toolkit CUDA và CMake ≥3.18. Ghi `nvidia-smi`, `nvcc --version`, commit SHA và dependency versions vào log lần chạy.

Cách ưu tiên, sau khi dependency đã sẵn sàng:

```bash
python -m tools.s6_reference.validate_cuda --output results/s6-t4-validation
```

Runner mặc định bắt buộc CUDA, sm_75, GPU T4 thực, golden tests và bốn tool sanitizer (memcheck/racecheck/initcheck/synccheck). Không cài package hoặc driver. Output phải mới/rỗng; mỗi command có log và status, có timeout; khi fail vẫn ghi `validation.json`. Hash/manifest/metadata phải khớp và đủ fixture; pixel PPM/PGM được đối chiếu lại golden, CSV/JSON phải khớp identity/shape/mode/GPU/workspace. Chỉ đủ các bước này mới ghi `acceptance=t4_complete`.

Để chạy mock data đã có hoặc fixture đã chuẩn bị, dùng một trong `--mock-root data/mock --max-reference-mib 2048` hoặc `--fixtures-dir data/s6-mock-fixtures`. Importer không thay H thiếu bằng H thật/identity; failure ledger còn handoff bị bỏ thì runner không nghiệm thu toàn tập. Trên máy phát triển khác T4 có thể chỉ định rõ `--allow-non-t4 --skip-sanitizer`; receipt khi đó luôn là `preliminary`, không biến bước bị bỏ thành pass.

Các lệnh riêng để debug:

```bash
python -m pip install -r tools/requirements.txt
python -m pytest tests/python -q
cmake -S src/s6 -B build/s6 -DCMAKE_CUDA_ARCHITECTURES=75 -DS6_REQUIRE_CUDA=ON
cmake --build build/s6 -j 2
test -x build/s6/test_s6_cuda
ctest --test-dir build/s6 --output-on-failure
python -m tools.s6_reference.generate_fixtures --output data/s6-fixtures --include-mock
(cd data/s6-fixtures && sha256sum -c manifest.sha256)
build/s6/test_s6_cuda --require-t4 --golden data/s6-fixtures/*.bin
compute-sanitizer --tool memcheck --error-exitcode 1 build/s6/test_s6_cuda --golden data/s6-fixtures/*.bin
compute-sanitizer --tool racecheck --error-exitcode 1 build/s6/test_s6_cuda --golden data/s6-fixtures/*.bin
```

Kỳ vọng: build có cả `stitch_s6` và `test_s6_cuda`; host/native CUDA/golden tests pass; sanitizer không báo lỗi. CUDA test mặc định trả 77 khi không có device; chế độ strict từ chối cả không device và device khác T4: **skip hoặc chỉ host pass không phải nghiệm thu GPU**. Geometry/origin/mask phải khớp tuyệt đối; tolerance RGB vẫn tối đa **2 mức uint8**, đã qua các ca GTX 1650 nêu trên, còn phải xác nhận T4. Nếu không đạt, điều tra precision/filter trước khi đổi ngưỡng.

Sau hai cặp Oxford, mở rộng kiểm tra ảnh development: H từ S5 CUDA, RGB có padding, exposure khác, overlap hẹp, texture thấp, đường thẳng, moving objects/parallax. Lưu panorama/crop/mask và thông tin failure. S6 không sửa được mọi lỗi parallax hoặc H sai chỉ bằng tăng B.

## 6. Đo chi phí từng B

```bash
mkdir -p results
build/s6/test_s6_cuda --timings results/s6-timings.csv --golden data/s6-fixtures/*.bin
```

Runner kiểm tra golden trước khi đo; dùng buffer/stream/geometry/workspace đã chuẩn bị, **2 warmup + 10 repeat** cho mỗi fixture. `gpu_ms` dùng CUDA event bao quanh toàn bộ render; `host_ms` bao gồm gọi render, ghi event và chờ hoàn tất. CSV còn có fixture, mode, shape, repeat, warmup, workspace bytes và GPU name. Loại warmup khi tổng hợp; ảnh tổng hợp nhỏ chỉ kiểm tra harness, chưa đủ cho kết luận hiệu năng.

Ranh giới này đo **S6 steady-state**, không gồm geometry, cấp phát, truyền H2D/D2H hoặc validation download. CUDA-event interval có thể gồm khoảng trống giữa các lần host dispatch, không phải tổng thời gian từng kernel riêng lẻ. Không đưa trực tiếp vào `t_e2e`, không thay `runs.csv` chung. Benchmark pipeline cần đo lại toàn ranh giới theo conventions, bao gồm chi phí policy/cache thực tế. Chưa có timing T4, speedup, VRAM peak hoặc Pareto frontier để báo cáo.

Đã có CSV timing cục bộ: 324 dòng cho synthetic và 144 dòng cho Oxford. `validation.json` mới ghi median của **10 lần đo** riêng từng fixture, loại hai warmup; không gộp repeat thành phân phối các pair. Counter chỉ bật ở correctness/artifact render, tắt trong profiling, metadata ghi rõ khác biệt. Chưa dùng số đo GTX này làm số liệu paper, speedup hoặc kết luận B nào tốt nhất trên T4.

```bash
mkdir -p results/s6-artifacts
build/s6/test_s6_cuda --artifacts results/s6-artifacts --golden data/s6-fixtures/*.bin
```

Mỗi fixture xuất RGB PPM, mask PGM và JSON geometry/mode/levels/workspace/counts/GPU/runtime. PGM có maxval 1, payload giữ nguyên 0/1; thư viện ảnh có thể scale thành 0/255 khi decode. Công cụ từ chối ghi đè artifact, tên fixture trùng và alias input. Output là ảnh **CUDA thực**, không sao chép golden làm panorama.

## 7. QoR độc lập và việc cần Đế review

Hướng dẫn đầy đủ và schema ở [tools/qor/README.md](../tools/qor/README.md). Chạy:

```bash
python -m tools.qor.evaluate --manifest data/pairs.json --runs results/runs.csv --policy tools/qor/policy-pilot.json --ref-config S1_K2000_R1024_Bfeather --output results/qor.csv
python -m tools.qor.report --qor results/qor.csv --policy tools/qor/policy-pilot.json --output results/qor-report.json
```

Evaluator đọc RGB gốc và H từ CSV; không dùng panorama sau blend làm input. Miền chuẩn Ω cố định theo H reference trong ảnh A; miền ứng viên V = Ω ∩ validity(Hcandidate). Coverage = `|V|/|Ω|`. PSNR dùng mọi sample RGB trên V. SSIM dùng Gaussian 11×11, sigma 1.5, chỉ nhận tâm có **toàn bộ cửa sổ nằm trong V**; không đủ cửa sổ trả failure rõ ràng. Cách cửa sổ này là quy tắc pilot của dự án; không đồng nhất với mọi thư viện SSIM. Định nghĩa SSIM nền tảng xem [Wang et al., 2004, bản tác giả](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf).

`policy-pilot.json` ghi `pilot-not-team-approved`: coverage ≥0.95, SSIM drop ≤0.02, PSNR drop ≤0.5 dB. Không gọi đây là ngưỡng chính thức UDIS-D hoặc của Nhóm. Đế cần review miền so sánh, trường hợp reference PSNR = +∞, overlap quá nhỏ, reference unreliable và thiên lệch khi candidate chỉ giữ phần dễ. QoR hiện tại kiểm tra alignment; chưa chứng nhận seam/blending quality. Khi so S/K/R phải giữ B cố định cho đến khi có phép đo render được duyệt.

Report ghi pass/fail cùng lý do, reference failure/unreliable và unachievable theo **configuration space được chọn rõ**. Mặc định gồm mọi config CSV, kể cả reference; reference hợp lệ tự pass nên không suy diễn có nhiều cấu hình rẻ đạt QoR từ achievable. `--candidate-config` có thể lặp để chỉ định tập hữu hạn khác; thiếu config ở một pair sẽ báo lỗi. PSNR +∞ được ghi chuỗi `"inf"` trong strict JSON; không phát `Infinity`/`NaN`. Không triển khai lại Pareto/Oracle latency thuộc task Tuấn.

Đã chạy hai public CLI evaluator/report trên hai cặp Oxford, H mock SIFT làm reference và H mock ORB làm candidate, B cố định feather: bốn dòng QoR và báo cáo JSON hợp lệ, cả hai lệnh exit 0. Hai candidate ORB đều không đạt policy pilot vì vượt mức giảm SSIM/PSNR cho phép; coverage lần lượt 1.0 và khoảng 0.9944. Nhãn unachievable chỉ áp dụng cho tập candidate ORB được chọn trong lần chạy này. Đây là smoke test I/O với dữ liệu thật, không phải ablation S/K/R của pipeline GPU hoặc policy đã được Đế duyệt. Evidence nằm tại `results/s6-qor-smoke/`.

## 8. Điều kiện đóng task

| Điều kiện | Trạng thái bàn giao |
|---|---|
| Geometry/reference/evaluator và regression Python/host | 181 Python tests và CTest 2/2 pass |
| CUDA compile/native/golden trên GTX 1650 | Pass 27 fixture synthetic/mock + 12 fixture ảnh thật từ bốn handoff; mask exact, RGB max_abs 1 |
| CUDA compile/native/golden/sanitizer trên T4 | Chưa chạy; sanitizer WDDM cục bộ bị chặn |
| Đế duyệt QoR policy và Tuấn review tích hợp interface/target | Chờ review; chưa sửa common thay thành viên khác |
| S6 với H từ mock S5 trên ảnh thật | Đã kiểm tra hai cặp Oxford; còn development HPatches/UDIS-D và S5 CUDA của Nhóm |
| Timing từng B, peak memory và benchmark toàn pipeline | Có CSV cục bộ và runner riêng; còn số đo T4/VRAM peak/toàn pipeline |

Ưu tiên tiếp theo là chạy chuỗi T4 ở §5, sửa lỗi nếu phát hiện, rồi tích hợp root pipeline và đo trên dữ liệu development. Chỉ khi đủ các bằng chứng này mới đánh dấu hai sản phẩm P1 hoàn tất; không suy ra P1 đã xong từ số test CPU.
