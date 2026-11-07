# CH-03: A4 same-H trên Colab T4

Ngày lập: 03/11 (Thứ 3), năm 2026. Phase 3 có deadline chung 05/11 (Thứ 5). Runbook này bàn giao tooling và lệnh để người dùng chạy; chưa có kết quả A4 trên GPU hoặc UDIS-D. Nguồn contract: [plan](../plan.md), [A4](../experiments/ablation-studies.md), [S6](../stages/s6-warp-blend.md) và [runner nền](../../tools/bench/README.md).

## 1. Phạm vi và quyết định kỹ thuật

Theo yêu cầu của người dùng, A4 cô lập B sau khi giữ RGB/H/canvas cố định. Không sửa kernel S6, Gaussian/Laplacian, pull-push extension hoặc feather weights. Không chạy S1–S5 trong benchmark này. CH-01 được bàn giao tooling; dữ liệu thật/reference vẫn chờ kiểm chứng, không coi là đã đạt QoR chỉ vì có notebook.

| Quyết định ngày 03/11 (Thứ 3) | Lý do, trade-off và fallback |
|---|---|
| Khóa toàn cohort trước lần render đầu | Chặn chọn H/pair sau xem output. Tốn một lượt CPU warp cho provenance/footprints. Input lỗi làm cả registration invalid; giữ ledger, không thay ảnh hay H |
| Alignment đo một lần trên ảnh gốc và H | Dùng evaluator hiện có, không so SSIM giữa panorama. Miền SSIM quá nhỏ/no overlap ghi unverified; không nới window để lấy số |
| Giữ ba ranh giới timing hiện có | Geometry/setup ngoài repeat; CUDA events quanh render; host wall quanh invocation/completion. Event interval có thể chứa host dispatch gap, không đổi tên thành tổng active-kernel time |
| Giữ nguyên API/kernel S6 | Native xuất union mask theo pixel và overlap count. Intersection theo pixel được kiểm bằng geometry oracle, không giả là CUDA intersection mask. Nếu cần đo trực tiếp từng pixel intersection của production kernel, đó là instrumentation/API riêng cần thiết kế tiếp |
| Lưu queried và allocated workspace riêng | Runner cấp thêm 256 byte guard ngoài query cho multiband. Không có phép đo peak VRAM; trường peak để trống |
| Crop chọn từ nguồn trước render | Seeded pair ranking; vị trí có gradient lớn nhất trên ownership boundary trong overlap, fallback sang overlap nếu không có boundary. Đây là rule hình minh họa, không phải metric composition |
| Hash và atomic no-clobber publication | Không ghi đè evidence. Native file `.partial` được publish bằng hard link khi hoàn tất; crash giữ partial để điều tra. Filesystem không hỗ trợ hard link phải dùng ổ local Colab trước |

Các quyết định trên do Chương/agent triển khai theo ủy quyền của người dùng. Ngưỡng RGB tối đa 2 và policy evaluator giữ nguyên. Inverse FP64 của native/Python được đối chiếu với `rtol=1e-12`, `atol=1e-14`; canvas/origin/T, union mask và source/H bytes vẫn phải exact. Đây là dung sai setup số học, không phải thay quality policy.

## 2. Module và artifact

| Module | Trách nhiệm |
|---|---|
| `tools/bench/same_h.py` | Điều phối prepare/native runner, failure ledger, timing và raw CSV |
| `tools/bench/same_h_protocol.py` | Khóa RGB/H/canvas/T/inverse, footprints, alignment và visual plan trước render |
| `tools/bench/a4_artifacts.py` | Verify hash, fixture identity, native RGB/mask/counters và timing; hỗ trợ chuyển nguyên campaign sang đường dẫn khác |
| `tools/bench/a4_telemetry.py` | Schema và xuất `a4_same_h_metrics.csv`; giữ cả FAILED/PREPARED |
| `tools/analysis/extract_seam_crops.py` | Xuất PNG RGB lossless, cùng bbox/zoom, caption trung lập |
| `tools/analysis/generate_a4_table.py` | Pair medians, p95 qua pair medians, CSV/JSON và bảng LaTeX chuẩn |

Mỗi campaign có `registration.json`, `report.json`, `a4_same_h_metrics.csv` và thư mục từng pair. Mỗi pair chứa golden `.bin` cho ba B, `timings.csv`, `runner.log`, metadata native, panorama PPM và mask PGM nếu giữ ảnh. Report ghi hash input/decoded RGB/H float64/binary/fixture/artifact/timings, revision, dirty state, seed, order và disposition. Dữ liệu sinh ra đặt dưới `data/`, `results/` hoặc `build/`, không commit; không commit tài liệu cá nhân.

## 3. Chuẩn bị manifest và kiểm CPU

Manifest giữ format cũ, paths tương đối từ thư mục chứa manifest:

```json
{
  "schema_version": "s6-same-h-v1",
  "evidence_type": "real_images",
  "split": "development",
  "pairs": {
    "pair_001": {"a": "a.ppm", "b": "b.ppm", "H": "H.json"}
  }
}
```

H là B→A trong tọa độ RGB gốc, 3×3 row-major FP64; có thể truyền inline JSON, JSON file hoặc `.npy`. Không dùng H ở ảnh resize nếu chưa đổi tọa độ đúng. `.npy` ảnh phải RGB uint8; PPM từ CH-01 giữ decoded RGB và geometry. Các sample tổng hợp phải ghi `evidence_type=synthetic_fixture`, `split=synthetic`; không đổi nhãn thành UDIS-D. Thiếu dữ liệu thật thì dừng nhánh real, không tự sinh ảnh thay thế.

```bash
python -m tools.bench.same_h --manifest data/a4/input.json --output results/a4-dry --dry-run --warmups 2 --repeats 10 --seed 20261030
python -m tools.bench.same_h --manifest data/a4/input.json --output results/a4-lock --prepare-only --warmups 2 --repeats 10 --seed 20261030
python -m tools.bench.same_h --output results/a4-lock --verify-only
```

`--dry-run` tạo fixture trong thư mục tạm rồi xóa, không tạo output được chỉ định và không chạy native/GPU. `--prepare-only` giữ registration và fixture để review. Thành công chỉ mang nhãn `prepared`, GPU `not_executed`. Verify-only kiểm được prepared fixture nhưng không biến nó thành CUDA pass. Thư mục output mới hoặc rỗng; không ghi đè campaign cũ.

Registration được publish trước bất kỳ output blend nào và khóa toàn bộ pair IDs, input file hashes, decoded RGB hashes/shapes, H nguyên bản cùng encoding, canvas/origin, T, inverse, footprints, alignment, mode order và visual plan. Nếu reuse registration bằng `--registration`, input/controls phải khớp. Sau chạy, tool kiểm lại input và manifest; sửa nguồn trong lúc chạy làm campaign thất bại.

## 4. Lệnh dành cho người dùng trên Colab T4

Các lệnh sau **chưa được agent thực thi trên GPU**. Chạy trong root repo đã checkout đúng commit và môi trường đủ NumPy/OpenCV; không cài dependency mới từ runbook này.

```bash
python -m tools.env.generate_receipt --output results/ch03-env/environment_receipt.json
cmake -S src/s6 -B build/ch03-t4 -DS6_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build/ch03-t4 --target test_s6_cuda -j2
python -m tools.bench.same_h --manifest data/a4/input.json --output results/a4-lock-t4 --prepare-only --warmups 2 --repeats 10 --seed 20261030
python -m tools.bench.same_h --manifest data/a4/input.json --registration results/a4-lock-t4/registration.json --output results/a4-t4 --runner build/ch03-t4/test_s6_cuda --warmups 2 --repeats 10 --seed 20261030 --save-images
python -m tools.bench.same_h --output results/a4-t4 --verify-only
```

Kiểm log configure/compiler xác nhận target `sm_75`. Benchmark mặc định thêm `--require-t4`; runner phải quan sát T4 thực sm_75. Exit 77, timeout hoặc exit khác 0 đều là failure. B được shuffle theo seed; `--mode-order sequential` là lựa chọn tường minh và phải giống registration nếu dùng nó. Mỗi mode dùng hai warmup và mười repeats, allocation/upload/statistics không nằm trong timed repeats. Statistics dùng ở correctness render riêng, timed render truyền null statistics.

Native runner kiểm padding, mask exact và RGB≤2 so FP64 golden. Python đọc lại pixel để xác minh, không tin riêng cờ metadata. Native metadata thêm T/inverse và allocated workspace, nên phải rebuild runner; binary cũ thiếu trường sẽ bị từ chối. CSV native được buffer/publish khi hoàn tất; runner giới hạn 100.000 timing rows mỗi invocation. Crash có thể không có timing cho repeat chưa hoàn tất; raw A4 giữ failure rows với timing trống và process `runner_wall_ms`, không bịa thời gian từng repeat.

Local hoặc GPU khác chỉ dùng `--allow-non-t4` một cách tường minh, giữ nhãn `preliminary_non_t4`. Task hiện tại không chạy các lệnh GPU này. Không coi compile thành công là runtime/sanitizer pass. Correctness/sanitizer T4 theo [workflow S6](../../tools/s6_reference/README.md) vẫn cần người dùng thực hiện trước kết luận chính thức.

## 5. Telemetry và bảng

`a4_same_h_metrics.csv` có một row cho từng warmup/repeat của mỗi pair/B pass. Pair lỗi hoặc chưa chạy có một row disposition cho mỗi B, timing để trống. Nhóm cột:

- Identity: schema, pair, B, evidence type, registration/H/RGB hashes, revision/binary và order index.
- Geometry: width/height/origin; T/inverse và mask hashes nằm trong registration/native JSON.
- Time: `t_setup_geom_ms`, `t_device_setup_ms`, `t_h2d_ms` là số đo một lần, lặp lại như metadata; `t_gpu_steady_ms`, `t_host_e2e_ms`, enqueue/wait là từng repeat. Không cộng CUDA event vào host wall, không gộp setup một lần vào phân phối end-to-end bằng tổng median.
- Memory: `workspace_bytes` từ API query, `workspace_allocated_bytes` gồm guard, `peak_vram_bytes` trống vì chưa đo. Không gồm toàn bộ input/output/pitch/runtime allocations.
- Verification: RGB max error, union-mask flag, overlap/union counts và `intersection_check=locked_geometry_oracle_and_native_count`.

```bash
python -m tools.analysis.generate_a4_table --campaign results/a4-t4 --output results/a4-tables --dry-run
python -m tools.analysis.generate_a4_table --campaign results/a4-t4 --output results/a4-tables
python -m tools.analysis.generate_a4_table --campaign results/a4-t4 --output results/a4-tables --verify-only
```

Output gồm `summary.json`, `summary.csv`, `a4_table.tex`, hash manifest. Bảng dùng median repeat theo pair, rồi median/p95 tuyến tính qua pair medians; p95 repeat trong native report chỉ là diagnostic khác. Throughput = full-canvas pixels / median GPU interval, đơn vị MPix/s, không phải valid-only pixels/s. Workspace trình bày maximum qua workload; CSV/JSON có cả host median và allocated workspace. `n/N` giữ failed/missing pair; không bỏ mẫu số để che failure.

Table generator từ chối prepare-only và CPU/mock timings. Preliminary GPU cần `--allow-preliminary`; không trộn GPU architectures trong một bảng. Workload synthetic được đánh dấu trong JSON và comment LaTeX, không tự gọi là UDIS-D. Không có pair đã verify trên GPU thì chưa có bảng latency để công bố; vẫn giữ failure ledger. LaTeX dùng `table/tabular` chuẩn, không cần package mới.

## 6. Crop và quan sát định tính

```bash
python -m tools.analysis.extract_seam_crops --campaign results/a4-t4 --output results/a4-visuals --dry-run
python -m tools.analysis.extract_seam_crops --campaign results/a4-t4 --output results/a4-visuals
python -m tools.analysis.extract_seam_crops --campaign results/a4-t4 --output results/a4-visuals --verify-only
```

Visual plan chọn tối đa ba pair full panorama và hai pair crop từ seeded hash ranking **trước render**. Bbox lấy từ gradient của warped RGB nguồn, tại boundary ownership trong overlap; không chọn từ output của một B. Crop không có gradient nguồn dương không được gọi là high-contrast. Không đủ hai case phù hợp thì báo thiếu qua plan/exports, không chọn case mới sau xem output. Rule không chứng minh tập này đại diện thống kê toàn dataset.

PNG giữ nguyên RGB uint8, cùng bbox và zoom 1 cho ba mode; không exposure correction, auto-contrast, retouch hay nội suy phóng đại. Có PNG riêng từng mode, full-size panorama, `visual_manifest.json` với hashes và `captions.txt` tiếng Anh. Encoder được kiểm round-trip pixel; verify-only tái tạo nội dung để phát hiện đổi crop/ảnh. Case fail có omission, không thay bằng case thuận lợi khác.

Caption mặc định chỉ mô tả thiết kế và yêu cầu xem duplicate contours, edge sharpness, transition falloff; chưa ghi nhận xét thắng/thua cho case chưa xem. Người viết phải bổ sung quan sát thực tế sau review ảnh, giữ qualitative. PSNR/SSIM/MSE alignment trong registration chỉ phụ thuộc RGB/H và không được dùng để kết luận blender nào có seam tốt hơn.

## 7. Kiểm tra và bàn giao

```bash
python -m pytest tests/python/test_a4_campaign.py tests/python/test_same_h_bench.py tests/python/test_s6_validation.py -q
python -m pytest tests/python -q
```

Các tests mới dùng ảnh nhỏ và subprocess mock có `sm_00` cùng nhãn CPU; số timing trong fixture chỉ kiểm công thức/parser, không là số đo hardware. Đã có compile-only runner `sm_75` trong môi trường phát triển; chưa chạy CUDA binary. Copy nguyên campaign để giữ report, registration, fixture, raw timing và RGB/mask; verifier dùng aliases có hash để đối chiếu các native fixture paths sau khi chuyển thư mục. Không sửa JSON/CSV để sửa path thủ công. Giữ `--save-images` cho A4: thiếu ảnh đã xóa thì không tái kiểm pixel hoặc xuất crop được.

CH-03 tooling chỉ đủ điều kiện bàn giao khi CPU tests/compile pass. CH-03 thực nghiệm chỉ nghiệm thu sau khi có cohort/H được khóa, correctness và timing T4 thật, hai crop sets đủ dữ liệu, ledger đầy đủ và phân tích không vượt bằng chứng. Không đánh dấu A4 thực nghiệm hoàn tất bằng dry-run/mock.
