# Chuẩn bị dữ liệu và protocol Phase 3

Runbook chung cho CH-00/CH-01, lập 02/11 (Thứ 2), năm 2026. Tooling này chỉ chạy CPU: kiểm môi trường, archive, ảnh, split và CSV; không gọi CUDA harness, sweep hay profiler. Dữ liệu, receipts và kết quả sinh ra nằm trong `data/` hoặc `build/`, không commit. Kế hoạch cá nhân không thuộc gói tài liệu này.

## 1. Điều kiện và giới hạn nghiệm thu

- Python 3.10+, NumPy, OpenCV và pytest từ môi trường hiện có; không thêm dependency. Receipt ghi phiên bản thực tế. Khi đổi decoder hoặc phiên bản, chuẩn bị bundle mới và đối chiếu hash pixel; không ghi đè bundle cũ.
- Chạy lệnh từ root repo bằng Python của môi trường đã chọn. Trên Windows có thể thay `python` bằng `build/p1-venv/Scripts/python.exe`. Trên Colab dùng môi trường Python của phiên đó.
- Mọi CLI có `--dry-run`, `--verify-only`, `--verbose`. Hai chế độ đầu không ghi file. Mỗi lần sinh artifact phải dùng output mới. Lỗi trả exit code 2 và JSON log trên stderr; không chuyển sang synthetic.
- Atomic publication dùng temporary file cùng thư mục, `fsync` rồi hard link không ghi đè. NTFS/ext4 được hỗ trợ; filesystem không hỗ trợ hard link sẽ báo lỗi. Trên Colab nên sinh bundle tại ổ local `/content`, xác minh rồi mới copy ra Drive.
- Đây là tooling đã kiểm bằng fixture nhỏ. Nghiệm thu dataset thật, reference và GPU vẫn cần artifact thật. Không dùng kết quả Phase 2 để chứng nhận revision mới.

## 2. CH-00: receipt và baseline

```bash
python -m tools.env.generate_receipt --output build/p3-local/environment_receipt.json --verbose
python -m tools.env.generate_receipt --output build/p3-local/environment_receipt.json --verify-only
python -m pytest tests/python -q --junitxml=build/p3-local/python-tests.xml
```

Receipt ghi SHA/branch/dirty, hash từng file tracked, hash inventory, phiên bản Python/package và trạng thái `git`, `nvcc`, `g++`, `clang++`, CMake, CTest, Compute Sanitizer. File untracked chỉ được liệt kê bởi Git, chưa nằm trong source inventory. Tool thiếu được ghi `unavailable`; đó không phải kết quả test fail hoặc GPU pass. Receipt không chạy test hay kernel. Ma trận T4/L4/A100 chứa target dự kiến 75/89/80, không giả làm GPU đã quan sát.

Có thể chỉ rõ executable bằng `--tool cmake=/path/to/cmake` và gắn log bằng nhiều `--evidence path/to/log`. Receipt chỉ hash log, không suy diễn log thành pass cho revision hiện tại. Log phải ghi command, revision, scope, exit code và số test thực tế. Receipt Windows không đại diện cho toolchain WSL; tạo receipt riêng trong WSL nếu build host ở đó.

Trên môi trường có CMake/C++ compiler, baseline CPU được cấu hình tách biệt:

```bash
cmake -S . -B build/p3-host -DCMAKE_CUDA_COMPILER=NOTFOUND -DCMAKE_BUILD_TYPE=Release
cmake --build build/p3-host -j2
ctest --test-dir build/p3-host --show-only=json-v1
ctest --test-dir build/p3-host --output-on-failure
```

Kiểm log configure xác nhận host-only trước CTest. Với contract hiện tại có 8 host tests; CUDA tests không được đăng ký ở build này. Không gọi build GPU hoặc campaign từ runbook này. GPU acceptance, sanitizer và latency T4/L4/A100 do người dùng thực hiện riêng trên Colab.

## 3. Nguồn UDIS-D và checksum

Dùng đường dẫn Dataset từ [repository tác giả UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching#dataset-udis-d), truy cập 02/11 (Thứ 2), năm 2026. README dẫn Google Drive/Baidu và phát hành cả kết quả tham khảo. Tool không tải mạng và không xác thực danh tính tác giả bằng chữ ký số. Người chuẩn bị dữ liệu phải kiểm đường dẫn archive với nguồn này trước khi lập receipt.

Tạo `data/p3/training-source.json` và `data/p3/testing-source.json` theo schema sau; thay các chuỗi `REPLACE_*` bằng thông tin đã kiểm, không dùng nguyên ví dụ để chạy:

```json
{
  "schema_version": "udis-source-v1",
  "split": "training",
  "author_source_url": "https://github.com/nie-lang/UnsupervisedDeepImageStitching",
  "download_url": "REPLACE_WITH_AUTHOR_LINKED_HTTPS_DRIVE_OR_BAIDU_URL",
  "sha256": "REPLACE_WITH_64_LOWERCASE_HEX_DIGITS",
  "checksum_origin": "user-pinned",
  "checksum_reference": "REPLACE_WITH_TRUST_SOURCE_OR_TRANSFER_RECEIPT"
}
```

Receipt test dùng `split=testing`. `publisher` chỉ dùng khi có checksum do nhà phát hành cung cấp và ghi nguồn kiểm chứng. Chưa có checksum tác giả được xác minh trong task này. `user-pinned` chỉ chứng minh archive đang đọc khớp bản đã pin, không chứng minh độc lập tính xác thực của lần tải đầu. Có thể tính SHA-256 bằng `Get-FileHash -Algorithm SHA256` hoặc `sha256sum`; hạ chữ hex thành lowercase khi lập receipt. Hash tự tính không được gắn nhãn publisher.

## 4. Ingestion strict và chuyển PPM

Giữ archive tải về tại `data/raw/`. Bố trí output thành các thư mục con của cùng bundle root để có thể copy nguyên cây sang Colab:

```text
data/p3/
  training-source.json
  testing-source.json
  training/
  testing/
  protocol/
  train-fit-runs.csv
  train-fit-qor.csv
  train-val-runs.csv
  train-val-qor.csv
```

```bash
python -m tools.data_prep.udis_ingest --archive data/raw/training.zip --source-receipt data/p3/training-source.json --split training --output data/p3/training --dry-run
python -m tools.data_prep.udis_ingest --archive data/raw/training.zip --source-receipt data/p3/training-source.json --split training --output data/p3/training --verbose
python -m tools.data_prep.udis_ingest --archive data/raw/testing.zip --source-receipt data/p3/testing-source.json --split testing --output data/p3/testing --verbose
python -m tools.data_prep.udis_ingest --output data/p3/training --verify-only
python -m tools.data_prep.udis_ingest --output data/p3/testing --verify-only
```

`--dry-run` vẫn đọc/kiểm dữ liệu CPU đầy đủ, chỉ không ghi file; không phải kiểm tên file sơ bộ. Mode mặc định `real`. `--mode fixture` chỉ dành cho test, receipt phải mang `synthetic-fixture`, mọi output mang `synthetic_fixture`; verify ở mode real từ chối output fixture.

Archive phải có `<training|testing>/input1/<pair_id>.<ext>` và `input2` đối ứng. Pair ID giữ nguyên stem, phân biệt chữ hoa/thường; tên không portable bị từ chối, không tự đổi tên. Tool kiểm SHA-256 archive, CRC từng member, duplicate/case collision, traversal, symlink, file đặc biệt, encryption và giới hạn admission. Các member ngoài input được kiểm CRC và ghi `not_pair_input` trong index, không dùng làm ảnh benchmark. Chỉ hỗ trợ ZIP stored/deflated, tối đa 100.000 entries, 64 MiB/member, 32 GiB uncompressed và 40 triệu pixel/ảnh. Vượt giới hạn phải điều tra nguồn; không nới guard chỉ để chạy qua.

PNG RGB8, JPEG 8-bit ba thành phần và P6 uint8 PPM được decode; grayscale, palette, alpha, 16-bit và JPEG thiếu end marker bị từ chối. OpenCV đọc raw geometry, bỏ EXIF orientation. Không resize/crop, không sửa exposure/ICC. PPM lưu đúng mẫu RGB uint8 đã decode; JPEG gốc đã lossy nên không gọi đây là khôi phục dữ liệu trước nén. Manifest ghi phiên bản decoder, source byte hash, pixel hash có geometry, PPM hash, width/height và bit depth. Ảnh decode được vẫn có thể không phù hợp về thị giác; reference audit và ledger QoR chịu trách nhiệm phần đó.

Test thật bắt buộc đủ 1.106 pair. Mỗi ID quan sát được có disposition, kể cả ảnh lỗi/thiếu một phía. Nếu thiếu cả hai phía và không có danh sách ID chính thức riêng, tool chỉ biết thiếu số lượng, ghi `count_error` và dừng; không bịa ID còn thiếu. Khi thất bại, giữ `ingestion_report.json` và các file đã kiểm tại output để điều tra, không có `data_manifest.json` thành công. Chạy lại bằng thư mục mới sau khi xử lý nguyên nhân, không thay pair lỗi bằng pair khác. Thiếu archive/receipt thì dừng trước ingestion.

## 5. Freeze cohort, grid và quality policy

```bash
python -m tools.data_prep.freeze_protocol --training data/p3/training/data_manifest.json --testing data/p3/testing/data_manifest.json --seed 20261030 --development-count 250 --validation-count 50 --output data/p3/protocol --dry-run
python -m tools.data_prep.freeze_protocol --training data/p3/training/data_manifest.json --testing data/p3/testing/data_manifest.json --seed 20261030 --development-count 250 --validation-count 50 --output data/p3/protocol
python -m tools.data_prep.freeze_protocol --output data/p3/protocol --verify-only
```

Protocol tái dùng `tools.controller.real_cohort.select_cohort` chỉ trên training. CRC32+size chỉ phục vụ selector cũ; guard leakage dùng connected components của source SHA-256 **hoặc** decoded RGB SHA-256, hợp nhất bắc cầu. Fit/validation giữ nguyên component. Validation có thể vượt 50 để không cắt component; requested và actual count đều được ghi. Không đủ components thì dừng, không lấy test bổ sung. Các pair training không được chọn có nhãn `train-unused`; toàn bộ test giữ nguyên IDs và mẫu số. Shared image giữa train/test làm freeze thất bại, không tự loại test pair.

Image-disjoint không đủ chứng minh scene-disjoint khi cùng cảnh có nhiều góc nhìn khác nhau. Mặc định ghi `scene_disjointness=not_certified`. Nếu có metadata scene đã được xác minh, truyền `--scenes data/p3/scene-groups.json --require-scene-groups`. Metadata phải có `schema_version=udis-scene-groups-v1`, `verified=true`, `source_url` HTTPS và `labels` cho toàn population, khóa dạng `training:<pair_id>`/`testing:<pair_id>`, giá trị là scene ID được xác minh. Tool hợp nhất thêm các scene này và từ chối scene chung giữa train/test. Cờ `verified` là khai báo provenance của người cung cấp, không phải tool tự xác thực nguồn. Không có metadata thì không dùng scene claim trong paper.

Output:

| Artifact | Contract |
|---|---|
| `training/data_manifest.json`, `testing/data_manifest.json` | Mọi pair và ảnh/hash/geometry/ledger; không chứa chất lượng homography đã giả định |
| `protocol/split_manifest.json` | fit/val/unused/test, components, seed và giới hạn scene claim |
| `protocol/config_grid.json` | 48 cấu hình từ `run_sweep.generate_grid_configs`, B=feather |
| `protocol/quality_policy.json` | Coverage ≥0.95, SSIM candidate−reference ≥−0.02, PSNR candidate−reference ≥−0.5 dB |
| `protocol/pairs_train-fit.json`, `pairs_train-val.json`, `pairs_test.json` | Pair mapping tương thích harness/evaluator; paths tương đối, RGB gốc qua PPM |
| `protocol/protocol_manifest.json` | Hash input/output, timing, reference candidate và trạng thái lock; ghi cuối cùng |

Policy giữ wire schema `qor-pilot-v1`/`pilot-not-team-approved` vì evaluator hiện từ chối phiên bản khác. Protocol ghi riêng thẩm quyền ngưỡng do người dùng giao cho Phase 3. Không sửa nhãn parser để giả nghiệm thu. Predicate vẫn dùng miền overlap reference của evaluator, không dùng metric composition để chấm seam. Warmup/repeats là 2/10; median theo pair rồi median/p95 trên pair medians.

Input manifests và metadata scene phải nằm dưới bundle root là thư mục cha của output protocol. Paths trong pair mappings có `../` để truy cập các thư mục training/testing cùng root; tool tự sinh và verify bằng reconstruction, không chấp nhận path tùy ý từ bên ngoài bundle. Copy nguyên `data/p3/` sẽ giữ paths. `--verify-only` kiểm hash PPM/pixel/geometry, input manifests, các output và tái dựng split/protocol; sửa file đóng băng làm verify thất bại. Để phát hiện manifest bị thay đổi trước lúc chuyển máy, lưu hash `protocol_manifest.json` ngoài bundle trong campaign receipt.

## 6. Reference và Fixed-Best: chỉ development

Sau khi **người dùng** chạy development trên Colab và có CSV theo conventions, đưa các file vào bundle rồi chạy CPU audit:

```bash
python -m tools.data_prep.audit_development --protocol data/p3/protocol --fit-runs data/p3/train-fit-runs.csv --fit-qor data/p3/train-fit-qor.csv --val-runs data/p3/train-val-runs.csv --val-qor data/p3/train-val-qor.csv --output data/p3/development-audit.json --dry-run
python -m tools.data_prep.audit_development --protocol data/p3/protocol --fit-runs data/p3/train-fit-runs.csv --fit-qor data/p3/train-fit-qor.csv --val-runs data/p3/train-val-runs.csv --val-qor data/p3/train-val-qor.csv --output data/p3/development-audit.json
python -m tools.data_prep.audit_development --protocol data/p3/protocol --output data/p3/development-audit.json --verify-only
```

Runs phải khai báo dataset `UDIS-D`, split `train-fit`/`train-val` tương ứng; cả warmup cũng được kiểm provenance. Audit từ chối test/foreign pair/split/dataset và ref ID khác protocol. `reference_failure`, `reference_unreliable`, missing và incomplete không biến thành pass; mọi pair vẫn trong denominator. Review thủ công reliability nếu thiếu ground truth; không dùng việc reference tự đạt SSIM/PSNR tương đối để chứng minh homography chính xác.

Reference candidate là `S1.00_K2000_R1024_Bfeather`, chưa được chứng nhận trước audit. Fixed-Best khác Fixed-High: tái dùng ranking của `audit_sweep`, ưu tiên số pair pass trên fit, rồi median latency và config ID. Validation kiểm chính ứng viên đã chọn, không chọn lại theo validation/test để làm đẹp. Grid/repeats thiếu hoặc reference không sẵn sàng thì `selected_fixed_best=null`. `selected_fixed_best` có thể vẫn có pair fail; phải đọc `passing_pairs` và kết quả validation, không tự gọi đó là QoR guarantee. Audit không tự mở test campaign hoặc approve controller. Protocol vẫn `inputs_frozen_reference_pending`; bước lock controller/reference sau audit thuộc công việc tiếp theo.

## 7. Bàn giao sang Colab

1. Checkout đúng commit; không copy binary local. Copy nguyên bundle root cùng các receipt/checksum đã pin, lưu external hash của protocol.
2. Tạo receipt riêng cho môi trường Colab; chạy `udis_ingest --verify-only` cho hai split rồi `freeze_protocol --verify-only`. Kỳ vọng exit 0, JSON event xác nhận và đúng 1.106 test pairs ở real mode. Bất kỳ hash/missing/mode mismatch nào phải dừng.
3. Chỉ sau khi reference/controller/timing được nghiệm thu mới chạy GPU bằng runbook campaign riêng. Các lệnh GPU không được thực thi bởi task CH-00/CH-01 này.
4. Chạy `audit_development` sau development, giữ raw CSV, logs, seeds, repeats, receipt và mọi failure. Không sửa split hoặc đổi pair sau khi thấy kết quả.

CPU regression cho tooling mới:

```bash
python -m pytest tests/python/test_repro_environment.py tests/python/test_udis_ingest.py tests/python/test_udis_protocol.py -q
```

Fixture test dùng ZIP/ảnh nhỏ sinh trong thư mục tạm và gắn nhãn synthetic; không phải dữ liệu hoặc kết quả benchmark UDIS-D. Nguồn contract: [plan](../plan.md), [conventions](../conventions.md), [controller contract](../stages/controller-contract.md).
