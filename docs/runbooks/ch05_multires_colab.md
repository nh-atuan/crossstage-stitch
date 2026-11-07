# CH-05: Synthetic multi-resolution stress-test

Ngày lập: 03/11 (Thứ 3), năm 2026. Deadline chung Phase 3: 05/11 (Thứ 5). Đây là runbook tooling; chưa có số đo 720p/1080p/4K trên GPU. Các lệnh GPU dưới đây dành cho người dùng chạy trên Colab. Nguồn contract: [plan](../plan.md), [S6](../stages/s6-warp-blend.md), [controller](../stages/controller-contract.md), [CH-03](ch03_same_h_colab.md).

## 1. Hai track và giới hạn kết luận

| Track | Input H của native | Đánh giá |
|---|---|---|
| `s6_only` | H_gt đã khóa, B→A trong RGB gốc | S6 CUDA event interval, checked host invocation, workspace query/allocation; full golden hoặc ROI theo admission |
| `end_to_end` | Không được nhận H, ROI theo H_gt hoặc golden | Cùng `PipelineRunner` chạy S1→S6 với cấu hình fixed S/K/R; Ĥ từ S5 được đổi về RGB gốc trong S5; Python tính corner error so H_gt sau chạy |

Native từ chối `--h-file`, `--rois`, `--export-full` ở end-to-end. Capacity của track này chỉ phụ thuộc input dimensions và budget, không dùng H_gt để sửa estimator hoặc render. Đây chưa phải thí nghiệm Adaptive Controller, A1/A2/A3 hay quantum; không dùng số fixed pipeline này để kết luận speedup controller.

`SUCCESS` là thực thi hoàn tất. Không là QoR PASS, không chứng minh homography chính xác, không tự chứng minh toàn bộ multiband output đã được golden-verify. Các ngưỡng QoR hiện có giữ nguyên; synthetic geometric overlap 30/50/70% khác `overlap_coverage≥0.95` trên reference domain của evaluator.

## 2. Quyết định và trade-off

Các quyết định ngày 03/11 (Thứ 3) được Chương/agent thực hiện theo phạm vi người dùng giao:

| Quyết định | Lý do, giới hạn và fallback |
|---|---|
| Cùng continuous scene theo tọa độ chuẩn hóa | Giữ chu kỳ texture và feature density theo field of view; wavelength theo pixel tăng cùng resolution. Không thể đồng thời giữ nguyên cycles/pixel và số feature/FoV khi đổi pixel count. Không tự retune texture theo kết quả |
| H' = D_A H D_B⁻¹ với half-pixel D | `x_s=s_x(x+0.5)-0.5`; dùng scale thực từ width/height. Không copy H pixel giữa tiers hoặc chỉ nhân translation |
| Sinh trực tiếp A/B từ cùng world texture | B(y)=F(H_base D_B⁻¹ y), A(x)=F(D_A⁻¹ x). Không warp A hữu hạn rồi coi padding đen là dữ liệu cảnh thật. Pixel sampling có thể tạo sai khác interpolation; H_gt là quan hệ tọa độ của scene liên tục |
| Calibrate translation trước render để khóa overlap | Dùng clipping polygon FP64 và support biên −0.5…W−0.5, −0.5…H−0.5; bisection 60 bước. Đây không phải QoR/scene label hoặc pixel-count overlap trong S6 |
| Generator streaming rows, output PPM | Bounded working set, không materialize full 4K reference. PPM lớn hơn PNG nhưng không cần codec hay dependency mới; byte hashes và geometry truy vết |
| Capacity tường minh thay canvas 4096 cố định | Checked size_t arithmetic, int32 pitch bounds; input và estimated canvas vượt capacity bị từ chối trước upload/render. Guard không phải chứng nhận zero OOB trên GPU chưa chạy |
| Query API trước allocation + free-memory reserve | Không dùng số ước lượng Python làm workspace CUDA đã đo. Native kiểm lại free VRAM; malloc vẫn có thể fail do cạnh tranh, trả typed `ERR_DEVICE_OOM` và giữ ledger |
| Full golden khi đủ RAM, ROI nếu không đủ | Chọn scope trước native execution; RAM thay đổi sau chạy làm full oracle fail, không hạ scope để lấy pass. Multiband có global pull-push/pyramid nên không dùng crop nhỏ giả làm oracle global |

Texture là tổng sin/cos có phase từ seed, cùng hàm và contrast ở mọi resolution. Recipe ghi mean/std RGB thực tế để kiểm tra, không tuyên bố histogram/entropy bằng nhau tuyệt đối. Pattern có tính tuần hoàn, không đại diện phân bố UDIS-D; ambiguity hoặc thất bại S5 phải được báo như kết quả synthetic. Không dùng H_gt để cứu những ca đó.

## 3. Module

- `tools/data_prep/generate_synthetic_multires.py`: recipe, source/H hashes, generator và verifier.
- `tools/bench/multires_geometry.py`: half-pixel transforms, polygon support overlap và corner error.
- `tools/bench/multires_admission.py`: host-memory snapshot/budget và capacity input-only.
- `tools/bench/multires_oracle.py`: mmap PPM, analytic/chunked bilinear ROI, full/ROI verification.
- `tools/bench/stress_multires.py`: khóa campaign, launch từng case cách ly, watchdog và complete repeat ledger.
- `harness/stress_multires.cu`, `stress_io.hpp`: native dual-track runner, typed errors, guarded allocation, raw telemetry và ROI output.
- `src/common/admission_control.hpp`: overflow/pitch/capacity arithmetic thuần host.
- `tools/analysis/generate_multires_report.py`: summary JSON/CSV, LaTeX và workspace SVG vector, không thêm Matplotlib dependency.

## 4. Chuẩn bị và kiểm CPU

Chạy từ root repo, Python 3.10+ cùng NumPy/OpenCV/pytest hiện có. Không thêm thư viện mới.

```bash
python -m tools.data_prep.generate_synthetic_multires --output data/ch05 --seed 20261030 --scenes 1 --overlaps 0.3 0.5 0.7 --warps affine projective --dry-run
python -m tools.data_prep.generate_synthetic_multires --output data/ch05-fixture --fixture --seed 20261030 --scenes 1 --overlaps 0.5 --warps projective
python -m tools.data_prep.generate_synthetic_multires --output data/ch05-fixture --verify-only
python -m tools.bench.stress_multires --manifest data/ch05-fixture/data_manifest.json --track s6_only --output results/ch05-dry --dry-run
python -m tools.bench.stress_multires --manifest data/ch05-fixture/data_manifest.json --track end_to_end --output results/ch05-plan --plan-only
python -m tools.bench.stress_multires --output results/ch05-plan --verify-only
```

Generator `--dry-run` chỉ tính geometry/recipe, không sinh ảnh chuẩn hoặc ghi file. Fixture là 64×36, 96×54, 192×108 với nhãn `fixture_*`; không được đổi nhãn thành 720p/1080p/4K. Benchmark dry-run không gọi native/GPU và không ghi output; plan-only giữ manifest/CSV với status `PLANNED`, không phải pass.

Output directories phải mới; không ghi đè frozen data/campaign. File được publish atomic/no-clobber. Failure có thể để lại output chưa hoàn chỉnh hoặc `.partial`; không xóa để chạy lại trên cùng evidence, dùng thư mục mới. Không có fallback từ thiếu runner/data sang mock.

## 5. Sinh bộ chuẩn — người dùng chạy trên Colab

```bash
python -m tools.data_prep.generate_synthetic_multires --output data/ch05 --seed 20261030 --scenes 1 --overlaps 0.3 0.5 0.7 --warps affine projective --host-budget-mib 256 --chunk-rows 64
python -m tools.data_prep.generate_synthetic_multires --output data/ch05 --verify-only
```

Ba tiers chính xác: 1280×720, 1920×1080, 3840×2160. Default tạo 18 pair, 6 recipe instances mỗi tier. Đây là cohort nhỏ cho feasibility/scaling, không đủ để tự tuyên bố generalization. Nếu tăng `--scenes`, khóa giá trị trước mọi GPU result; không thay pair lỗi sau khi xem kết quả. Mỗi pair ghi H_base, H_gt, D_A/D_B, input size, canvas/origin, target/measured support overlap, PPM/source pixel hashes, H text/hash và seed.

Working set generator chặn theo bound bảo thủ 512 bytes/pixel của tile. Chunk size có thể giảm để giữ RAM nhưng không đổi pixel/hash. Lấy mean/std là diagnostics của nguồn, không là criterion để chọn bỏ ảnh. Dữ liệu được sinh trên ổ local Colab, sau đó copy nguyên bundle; không commit dataset/output hoặc notebook outputs.

## 6. Build native — chỉ compile tại local, GPU chạy do người dùng

T4 mặc định sm_75; L4 dùng 89, A100 dùng 80. Root CMake giữ default 75 nhưng tôn trọng target chỉ rõ, không ép lại 75.

```bash
cmake -S . -B build/ch05-t4 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build/ch05-t4 --target stress_multires_cuda -j2
python -m tools.env.generate_receipt --output results/ch05-t4-env/environment_receipt.json
```

Binary ở `build/ch05-t4/harness/stress_multires_cuda`. Với L4/A100, dùng build directory riêng, đổi architecture tương ứng và `--expected-sm` cùng giá trị. Native kiểm cả model và SM: sm75 phải là T4, sm89 phải là L4, sm80 phải là A100. Không dùng GTX/RTX cùng compute capability làm bằng chứng Colab target. Receipt CPU không tự quan sát GPU; native receipts mới ghi model/SM/free/total VRAM thực tế.

Sinh một dataset bundle duy nhất rồi chuyển nguyên hash sang T4/L4/A100, không regenerate hoặc retune theo từng GPU. Archive build log, `CMakeCache.txt` và effective compile flags cùng environment receipt; runtime model/SM không tự chứng minh binary đã được compile native cho target đó. Native receipt ghi CUDA runtime/driver version, campaign khóa revision/dirty/source hashes và binary hash trước execution.

Native stress runner dùng shared `PipelineRunner` với capacity explicit. `stitch_harness` cũ vẫn có default input 2048 trong CLI; CH-05 không dùng CLI đó cho 4K. Không tự lấy bản benchmark legacy rồi đổi nhãn thành stress run. Các thuật toán S1–S6, RANSAC seed behavior và quality thresholds không đổi.

## 7. Chạy hai campaign độc lập trên T4

Các lệnh sau chưa được agent thực thi trên GPU:

```bash
python -m tools.bench.stress_multires --manifest data/ch05/data_manifest.json --track s6_only --runner build/ch05-t4/harness/stress_multires_cuda --output results/ch05-s6-t4 --expected-sm 75 --warmups 2 --repeats 10 --seed 20261030 --host-budget-mib 512 --device-reserve-mib 256 --timeout 300
python -m tools.bench.stress_multires --manifest data/ch05/data_manifest.json --track end_to_end --runner build/ch05-t4/harness/stress_multires_cuda --output results/ch05-e2e-t4 --expected-sm 75 --scale 0.5 --keypoints 1000 --ransac 256 --warmups 2 --repeats 10 --seed 20261030 --host-budget-mib 512 --device-reserve-mib 256 --timeout 300
python -m tools.bench.stress_multires --output results/ch05-s6-t4 --verify-only
python -m tools.bench.stress_multires --output results/ch05-e2e-t4 --verify-only
```

S/K/R được khóa cùng giá trị ở mọi resolution/B trong end-to-end; không tune để 4K pass. Ground truth chỉ dùng ở S6-only hoặc analysis sau native. Thứ tự B được shuffle có seed và lưu trong plan. Input RGB/H và native binary được hash trước/sau execution; source thay đổi làm verify thất bại. Default 18 pair × 3 B × (2+10) = 648 ledger rows mỗi track, kể cả failure. Warmups giữ riêng, không gộp vào latency statistics.

Mỗi case chạy trong process riêng, giữ command/stdout/return code hoặc timeout. Có `stress_plan.json` trước launch, từng `case_*/case_receipt.json`/`rows.json`, native CSV/receipt/ROIs, `multires_scaling_metrics.csv` và `report.json` cuối cùng. Case receipt là checkpoint điều tra; chưa có auto-resume đổi tên/thay dữ liệu hoặc chọn lại repeats. Process exit 77 không phải pass. CLI trả 2 nếu có case fail, nhưng vẫn có ledger để sinh báo cáo âm; không bỏ case fail để làm CLI xanh.

## 8. Memory và timing

Native query stage workspace trước buffer allocation, cộng padded input/output/mask, feature/match/inlier/probe buffers và workspace guards. Kiểm free VRAM với reserve, sau đó vẫn bắt typed cudaErrorMemoryAllocation nếu có cạnh tranh. `workspace_bytes` là **S6** query cho geometry thực tế, `workspace_reserved_bytes` là S6 capacity reservation của native, `workspace_allocated_bytes` chỉ có giá trị xác định khi allocation/completion thành công. `pipeline_workspace_capacity_bytes` tách riêng tổng query capacity của các stage, còn `declared_device_bytes` gồm cả buffers/guards theo kế hoạch allocations. Không trường nào là peak VRAM. Stream/events/runtime overhead nằm ngoài tổng này; reserve không phải số đo overhead. CSV/JSON summary có cả pipeline capacity và declared device allocation; bảng/SVG ghi rõ S6 WS.

End-to-end dùng capacity input-only: width tối đa khoảng 1.9W, height tối đa 2H và tổng không vượt geometry policy 16.777.216 pixel; capacity ít nhất chứa input. Ĥ cần canvas lớn hơn capacity sẽ nhận `ERR_CANVAS_OVERFLOW`, không allocate lại trong timing hoặc cắt output. Đây là resource ceiling, không thay quality policy. S6-only query theo canvas H_gt thật. Pitch được kiểm trong int32 và byte arithmetic trong size_t; không giả input resolution chính là canvas resolution.

Host RAM budget được cap bởi snapshot MemAvailable/GlobalMemoryStatusEx và cgroup v2 nếu có; đây không phải reservation. Native check host budget trước load hai PPM; input PPM phải đúng dimensions/payload, không resize/crop ngầm. Generator/reference dùng bounded tiles hoặc mmap. RAM thực thay đổi có thể vẫn gây `ERR_HOST_OOM`, phải giữ trong ledger.

S6-only: CUDA events trên dedicated stream quanh render; host wall quanh dispatch và checked completion. Allocation/upload/download/oracle/CSV nằm ngoài repeat. Canary/padding kiểm sau timed repeats. End-to-end: `t_s1`…`t_s6` dùng stage events của shared pipeline; `t_gpu_steady` là CUDA event span toàn pipeline, có thể chứa stream-idle/host gaps. `t_host_e2e` là wrapper invocation đến checked completion, gồm telemetry overhead của pipeline. Legacy `t_s6` có thể chứa khoảng host inspect sau S5, nên không gọi nó là active S6-kernel time hoặc cộng GPU span vào host wall. Failure chưa chạy stage không được dùng làm latency 0 để tính speedup.

## 9. Oracle và error

- Full golden chỉ được chọn trước execution nếu bound reference (192 bytes/canvas pixel cho feather, 768 cho multiband, cộng input/reserve) nằm trong effective host budget. Native export full RGB/mask ngoài timing; Python so mask exact và RGB≤2 theo chunk.
- Nếu full golden không được admission, native chỉ export ROIs: canvas corners, center, overlap polygon corners/center và một vị trí gần geometric ownership boundary tìm trên grid nhỏ từ H_gt. Python so full-canvas coordinates và bilinear samples, không resize ROI. Feather overlap RGB được chấm theo analytic weights; mọi B đều kiểm mask/source-only RGB ở ROI.
- **Multiband overlap RGB không được golden-verify bằng ROI local**, vì global pull-push và pyramid có support ngoài crop. Kết quả ghi số sample overlap chưa verify và số sample màu nguồn valid thực sự kiểm. Đây không phải full-image QoR hoặc composition score. Không gọi ROI pass là full golden pass.
- End-to-end tính mean/max L2 của bốn B corners sau projective division, đơn vị pixel RGB gốc. Ĥ không hợp lệ hoặc native canvas không khớp Ĥ bị từ chối; không dùng identity/H_gt thay thế. S6-only để trống registration error, không ghi 0 rồi gọi đó là estimator thành công.

Status ledger gồm `SUCCESS`, `PLANNED`, `ERR_DEVICE_OOM`, `ERR_HOST_OOM`, `ERR_CANVAS_OVERFLOW`, `ERR_CAPACITY`, `ERR_TIMEOUT`, `ERR_NOT_ENOUGH_POINTS`, `ERR_DEGENERATE_GEOM`, `ERR_VERIFY`, `ERR_CUDA`, `ERR_INPUT`, `ERR_DEVICE_MISMATCH`, `ERR_RUNNER`. Native raw status vẫn được lưu. Fatal/missing-repeat case giữ đủ expected repeat rows với status lỗi và timing trống; raw prefix nếu có vẫn ở native artifacts để điều tra. Không đổi QoR thresholds hoặc dùng failed-row timings để tạo distribution có vẻ hoàn chỉnh.

## 10. Báo cáo và nghiệm thu

```bash
python -m tools.analysis.generate_multires_report --campaign results/ch05-s6-t4 --output results/ch05-s6-t4-report
python -m tools.analysis.generate_multires_report --campaign results/ch05-e2e-t4 --output results/ch05-e2e-t4-report
python -m tools.analysis.generate_multires_report --campaign results/ch05-s6-t4 --output results/ch05-s6-t4-report --verify-only
```

Output: summary JSON/CSV, `scaling_table.tex` dùng LaTeX chuẩn và `workspace_scaling.svg` vector. Median/p95 tính qua pair medians; chỉ case có đủ warmup/measured repeats hợp lệ được dùng cho latency, nhưng mọi case vẫn ở N, completion/OOM/timeout/failure counts và ledger. Đây là latency trên complete subset, không là distribution thành công toàn workload. End-to-end có geometry-error summary riêng. Không trộn GPU khác nhau, hai track hoặc CPU/mock timings vào một bảng hardware. SVG ngắt đường khi tier thiếu/fail, không nối để che dữ liệu thiếu.

```bash
python -m pytest tests/python/test_multires_geometry_generation.py tests/python/test_multires_stress.py -q
cmake -S . -B build/ch05-host -DCMAKE_CUDA_COMPILER=NOTFOUND
cmake --build build/ch05-host -j2
ctest --test-dir build/ch05-host --output-on-failure
```

Môi trường phát triển chỉ kiểm CPU/mock, geometry 4K bằng số học, fixture ảnh nhỏ và compile/link CUDA; không có số đo performance hoặc sanitizer GPU từ task local này. Tests mock gắn sm_00 và bị publication CLI từ chối. Sau khi mở Colab, người dùng cần chạy correctness/sanitizer trước campaign, ví dụ thêm `compute-sanitizer --tool memcheck --error-exitcode 1` trước lệnh native nhỏ đã ghi trong case command; giữ profiler/sanitizer runs riêng, không dùng latency có instrumentation làm bảng chính. GPU stress 4K và OOB/sanitizer acceptance vẫn chưa hoàn thành cho tới khi có artifact thật.
