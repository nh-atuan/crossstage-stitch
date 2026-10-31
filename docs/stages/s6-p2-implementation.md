# Bàn giao P2 của Chương: QoR, S6 và working paper

Cập nhật 27/10 (Thứ 3), năm 2026. Deadline Phase 2: 31/10 (Thứ 7); review Nhóm: 01/11 (Chủ nhật). Nhánh `feat/p2-chuong-qor-paper`, dựa trên `origin/main` tại `ccc7786`. Chỉ commit cục bộ; không push hoặc tạo PR trong task này.

## 1. Phạm vi và trạng thái

| Task | Phần đã bàn giao | Phần chưa nghiệm thu hoặc phụ thuộc |
|---|---|---|
| P2-00 | Predicate QoR chuẩn, backward-compatible wrapper, tests, hướng dẫn tích hợp | Tuấn/Đế tích hợp Go/No-Go và duyệt policy; thiếu raw pilot |
| P2-08 | Tests composition, profiler, CLI same-H, artifact và chạy GPU cục bộ | Chưa chạy Colab T4/Compute Sanitizer thành công; chưa có composition quality evaluator |
| P2-09 | Adapter latency/QoR, retry/fallback/overhead, CSV/JSON/SVG và tests | Chưa có terminal controller runs hoặc raw pilot để phân tích thực nghiệm adaptive |
| P2-10 | Workspace OASIcs ẩn danh, script build và kiểm metadata | Chưa phải bản nộp hoàn chỉnh |
| P2-11 | Introduction, Related Work, khung pipeline/controller, S6 và protocol | Nội dung S1 của Tuấn, S2/S3 của Thành, S4/S5/controller của Đế cần owner review; chưa có bảng kết quả cuối |
| P2-12 | Audit, lệnh tái lập, bằng chứng và danh sách phụ thuộc | Nhóm review và chạy môi trường chính thức |

Không triển khai P2-01 đến P2-07, không đổi threshold/method của stage, không chạy quantum. Source kernel S6 và fixed pipeline giữ nguyên. [Quyết định kỹ thuật](../decisions/p2-chuong-qor-blend.md) ghi trade-off và các vấn đề nằm ngoài ownership.

## 2. QoR

`tools/qor/decision.py` cung cấp quyết định thuần; `tools.qor.evaluate.quality_decision` kiểm schema policy rồi gọi lõi. `quality_pass` bool cũ chỉ trả true cho PASS. Các trạng thái còn lại là FAIL, UNVERIFIED, REFERENCE_FAILED, EVALUATION_ERROR. Schema CSV tám cột và policy JSON cũ được giữ. `tools/qor/report.py` thêm `decision_status`, không đổi metric thành giá trị giả để pass.

Ngưỡng vẫn là policy pilot hiện có, chưa được Nhóm duyệt: coverage tối thiểu 0.95, SSIM drop tối đa 0.02, PSNR drop tối đa 0.5 dB. Dấu so sánh là candidate >= reference − drop. Thiếu reference/metric không được pass; NaN và metric sai miền là lỗi; PSNR +inf hợp lệ cho zero MSE. Reference failure và pipeline failure được báo riêng. Chi tiết precedence và adapter cho Tuấn/Đế ở [QoR README](../../tools/qor/README.md).

Audit đã xác nhận Go/No-Go hiện tại bỏ SSIM/PSNR/reference status, dùng fallback khi Oracle không có candidate pass và chưa lọc Pareto đúng nghĩa nondominated. Chương không sửa script đó. Các số pilot 25 cặp vẫn là provisional; không thể kiểm định lại QoR/speedup từ một JSON tổng hợp. Không dùng chúng làm kết quả cuối paper.

## 3. S6 và same-H

Ba B đang có là feather, multiband 3/5 REDUCE, tương ứng tối đa 4/6 mức. API nhận RGB gốc, H B → A, geometry, caller-owned buffers/workspace và stream. Test CUDA bổ sung kiểm tra khoảng trống giữa hai ảnh, preservation của vùng một nguồn, đen hợp lệ, mask/counter/padding và chạy lặp tất định cho cả ba mode.

`tools/bench/same_h.py` tái dùng CPU reference FP64 và `test_s6_cuda`. Mỗi mode có cùng bytes RGB/H/canvas; manifest, H, ảnh, fixture, binary và source đều có hash. Mọi pair lỗi vẫn nằm trong ledger. Không dùng overwrite vì mode đó không tồn tại trong S6. Không thay input/H lỗi bằng identity.

### Bằng chứng đã chạy

| Evidence | Môi trường và input | Kết quả |
|---|---|---|
| Native CTest toàn dự án | WSL Ubuntu 24.04, CUDA 12.5, `sm_75`, GTX 1650 4 GiB | 12/12 pass |
| Oxford same-H | Hai cặp graf 1–2 và 1–3 có sẵn; H từ ORB mock đã lưu, ba B, hai warmup/mười repeat | Sáu fixture pass; mask exact; max RGB error 1, tolerance 2; 72 timing rows |
| Synthetic gap | RGB 4×3 đen/trắng, translation x=7, ba B, 0 warmup/3 repeat | Ba fixture pass, chín timing rows, ảnh được xóa sau kiểm chứng theo mặc định |
| Guard T4 | `test_s6_cuda --require-t4` trên GTX 1650 | Từ chối đúng GPU không phải T4, exit 1 |
| Compute Sanitizer memcheck | Binary đã có trong máy, cùng S6 runner | Bị chặn: WDDM debugger không khởi tạo, device unsupported; exit 1, không phải memory-check pass |

Kết quả máy cục bộ ở `results/s6-same-h-p2-oxford-final/report.json`, các CSV/log/artifact JSON nằm theo từng pair; ảnh chỉ được giữ khi có `--save-images`. Lần chạy trước ở `results/s6-same-h-p2-oxford/` được giữ nguyên. Không commit raw outputs theo quy định repo. Đây là test renderer với ảnh thật và H mock, không phải sweep CUDA S1–S6, UDIS-D hay validation cohort chính thức.

Workspace feather bằng 0 theo API; multiband dùng workspace theo canvas và B. Receipt lưu số byte query thực tế, khác với buffer tối đa đang được harness reserve. Không gọi đây là peak VRAM đo thực nghiệm.

Timing tách geometry/setup/upload một lần, GPU event, host enqueue, host chờ và host invocation steady-state. GPU interval chồng lấp host interval, không cộng lại. Receipt cuối cảnh báo CUDA event lớn hơn host elapsed ở một số repeat. Chưa xác định nguyên nhân; cần đối chiếu T4 trước khi dùng timing làm kết luận hiệu năng. Không đưa số latency này vào paper và không suy ra ưu thế chất lượng blend từ alignment PSNR/SSIM.

### Lệnh tái lập trên T4

Từ root repo trong môi trường CUDA/CMake/Python đã được Nhóm chuẩn bị:

```bash
cmake -S . -B build/p2-t4 -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DS6_REQUIRE_CUDA=ON
cmake --build build/p2-t4 -j2
ctest --test-dir build/p2-t4 --output-on-failure
python -m tools.bench.same_h --manifest input.json --output results/s6-same-h-t4 --runner build/p2-t4/src/s6/test_s6_cuda --warmups 2 --repeats 10 --seed 0 --save-images
compute-sanitizer --tool memcheck --error-exitcode 99 build/p2-t4/src/s6/test_s6_cuda
```

Kỳ vọng: CTest không fail/skip CUDA, benchmark `status=passed`, GPU là T4, mask exact và RGB error <=2; sanitizer không báo lỗi. Đây là lệnh chưa thực hiện trên T4, không phải kết quả pass. `input.json` cần cohort được owner cung cấp đúng schema trong [benchmark README](../../tools/bench/README.md).

## 4. Phân tích P2-09

`tools/analysis/latency_qor.py` nhận terminal runs được gắn nhãn mode tường minh. Không nhận trực tiếp full grid, không tự chọn Fixed-Best hay tính Oracle. Join reference cùng pair, gọi predicate QoR chuẩn, giữ mọi category và mẫu số kỳ vọng/quan sát/thiếu. Median và p95 latency lấy trên pair medians sau khi loại warmup; failure có số đo vẫn được tính. Chỉ xuất paired speedup khi đủ cohort; quality-constrained speedup còn yêu cầu toàn bộ repeat pass policy được truyền vào.

Retry histogram, fallback rate, controller/retry overhead chỉ có khi input cung cấp, không suy missing thành zero. Adapter tùy chọn chưa phải trace contract được owner duyệt. Đọc [analysis README](../../tools/analysis/README.md) về input JSON, cùng split/policy/timing scope và hạn chế QoR theo pair/config.

## 5. Working paper

Source nằm ở [paper](../../paper/README.md); dùng class OASIcs đã có, anonymous mode, author/metadata ẩn danh. Script build kiểm BibTeX, reference, warning, font embedding và các token nhận diện đã biết. README biên tập giữ claim/evidence map, nguồn đã kiểm tra và phần cần owner cung cấp. Không có bảng latency/QoR giả, không coi Oracle là controller online, không claim adaptive đã giảm latency.

Build thực tế bằng `wsl --exec python3 paper/build.py` và lần kiểm tra độc lập với `--output build/paper-root-review` đều pass: sáu trang A4, Author `Anonymous Authors`, 19 font embedded, final log không warning/citation/reference chưa resolve. Đã render và xem sáu trang. Class, asset và license được dùng trực tiếp từ template gốc qua `TEXINPUTS`; `kpsewhich` và hash trong receipt kiểm tra provenance, không thêm bản sao vendor. PDF và log ở `build/paper/`, không commit. Đây là working manuscript, chưa đủ nội dung và bằng chứng để nộp.

## 6. Kiểm tra đã thực hiện

Các lệnh Python sau chạy trong môi trường `build/p1-venv` đã có:

```bash
python -m pytest tests/python/test_qor.py tests/python/test_qor_report.py tests/python/test_qor_decisions.py -q
python -m pytest tests/python/test_same_h_bench.py tests/python/test_s6_validation.py tests/python/test_s6_reference.py -q
python -m pytest tests/python -q
```

Kết quả tương ứng: 157, 70 và **319 test pass**. Analysis có 36 regression tests mới. Build root dùng CMake trong `build/cmake-linux`, NVCC trong `build/cuda-12.5.1`, Release và `sm_75`, output `build/p2-root`. Lệnh CTest đã chạy:

```bash
build/cmake-linux/cmake/data/bin/ctest --test-dir build/p2-root --output-on-failure
```

Kết quả 12/12 pass trên GPU cục bộ; S6 standalone CTest cũng pass 2/2. Build còn warning có sẵn về biến `ch`, `kMaxKeypoints` chưa dùng trong harness và clock skew nhỏ trên mount WSL. Không sửa source của owner khác để xóa warning. `git diff --check` và `git show --check` được kiểm trước/sau các commit; không stage dataset, raw output, build hoặc agent rules.

## 7. Việc cần Nhóm tiếp nhận

1. Tuấn/Đế gửi raw pilot `runs.csv`, `qor.csv`, metadata, policy và manifest; sửa predicate/Oracle/Pareto rồi chạy lại báo cáo. Chưa thể xác nhận lại GO theo QoR đã sửa từ summary cũ.
2. Đế và Chương review quality policy/overlap trước khi gọi bất kỳ so sánh nào là policy được Nhóm duyệt. Không tune trên final data.
3. Tuấn/Đế cung cấp terminal controller runs và xác nhận timing/retry/reuse contract; Chương mới phân tích adaptive/ablation thực tế.
4. Chạy CUDA tests, same-H benchmark và sanitizer trên Colab T4; kiểm clock/timing, lưu artifact đầy đủ và ghi GPU state. Không tính GTX pass là T4 pass.
5. Tuấn, Thành và Đế review phần phương pháp thuộc mình; Chương ghép phần có bằng chứng và bảng sinh từ artifact. Working paper chưa đủ điều kiện nộp.
