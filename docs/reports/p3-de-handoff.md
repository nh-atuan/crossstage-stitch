# Bàn giao Phase 3 của Đế

## Cập nhật 03/11 (Thứ 3)

Đã đo đủ 1.106 cặp UDIS-D và 295 cặp HPatches trên Colab Tesla T4. Đã có
telemetry, A2, bảng MCE, các ca diagnostic và draft Section 5.2/6. Source đo
chính thức là `c58d754`; checkpoint và raw CSV đã tải về local, kiểm hash và
tính lại báo cáo khớp bản T4. Chi tiết và giới hạn ở
[báo cáo Phase 3](p3-de-evaluation.md).

A2 giữ workspace ở cả hai biến thể để tách lợi ích tránh tính lại, theo
[quyết định đã chốt](../decisions/p3-de-a2-reuse.md). Đã sửa bảo đảm
device-ready trước vùng đo và chạy lại campaign. T4 đạt 17/17 CTest,
382 test Python và memcheck regression upload 0 errors. Draft riêng đã
compile PDF 2 trang và audit ẩn danh/font pass.

Reuse tiết kiệm thời gian trên cohort R-only retry, nhưng Controller hiện
tại chưa có lợi thế end-to-end so với Fixed. Bảng HPatches không tự chứng
minh noninferiority trên toàn cohort. Manuscript chung còn bị chặn bởi file
`paper/sections/pipeline-s6.tex` thiếu từ trước task này; không dùng build
draft riêng để tuyên bố manuscript chung đã pass.

Các mục cập nhật 01/11 dưới đây là lịch sử bàn giao, không phải trạng thái
hiện tại. Phần triển khai, đo và draft của Đế đã có artifact để Nhóm review;
việc nghiệm thu Phase 3 và kết luận research question thuộc buổi review của Nhóm.

## Cập nhật sau khi kết nối được T4

Cập nhật 01/11 (Chủ nhật), năm 2026. Code đo đã commit tại `8e1380f`.
Các mục phía dưới phần cập nhật này là bản bàn giao trước khi kết nối được
T4, giữ để đối chiếu; trạng thái hiện tại theo phần cập nhật này.

Bridge MCP đã khôi phục quyền đọc/chạy notebook. Đã quan sát Tesla T4,
15360 MiB VRAM, CUDA, CMake và compute-sanitizer. Source được truyền bằng
archive vì repo không clone được khi thiếu GitHub authentication. Git snapshot
trên Colab được ghi riêng với source revision local và hash source/binary;
không coi snapshot đó là commit của nhánh local.

Kiểm chứng đã quan sát trên T4: bản gốc 15/15 CTest; bản instrumentation
16/16 CTest; 381 test Python; memcheck Controller/S5 và harness với ảnh
HPatches rộng 2456 px đều 0 errors. Smoke adaptive ba pair, 2 warmup/10
repeat có đủ trace và phân tích thành công. Lần chạy Python đầu trên source
archive có sáu test fail vì thiếu `.git`; đã bổ sung metadata Git thật rồi
chạy lại, không sửa hoặc vô hiệu hóa test. Local hiện có 382 test Python pass,
gồm test sinh bảng LaTeX/PDF/SVG và kiểm hai biến thể reuse cùng kết quả.

UDIS-D testing.zip thiếu input1/000001.jpg. Đã khôi phục ảnh từ repo khác của
cùng tác giả; ảnh B khớp từng byte với archive. Receipt ghi URL, commit và
SHA-256. Đã chuẩn bị đủ 1106 pair UDIS-D và 295 pair từ 59 viewpoint sequence
HPatches. Harness cấp phát theo kích thước input trước vùng đo, giữ nguyên
ảnh gốc và kiểm giới hạn canvas. Chi tiết ở
[quyết định campaign](../decisions/p3-de-campaign.md).

Đã khởi chạy worker đo UDIS-D (Fixed/Adaptive/No Reuse) rồi HPatches
(Fixed/Adaptive), seed 20261101, 2 warmup/10 repeat, batch 32; đảo thứ tự mode
theo batch. Không chạy hai campaign GPU song song. Output từ repo Colab là
`../results/p3-de-t4-20261101/`; mỗi batch có CSV, trace, log và hash receipt.

Sau khi khởi chạy, yêu cầu đọc log/notebook chưa được frontend trả lời.
Chưa biết worker còn chạy, đã xong hay bị lỗi; không gửi lại lệnh campaign.
Chưa có số liệu campaign được đọc, tải về local hoặc đưa vào paper. Không
khẳng định giữ được phiên chỉ từ việc Popen trả PID. Runtime chưa mount Drive,
nên cần tải checkpoint về máy ngay khi kết nối phục hồi.

Nếu kiểm tra thủ công, tạo **cell mới**, không chạy lại cell khởi động job.
Với notebook ở thư mục mặc định Colab:

```python
from pathlib import Path
root = Path.cwd() / 'p3-de-work'
print((root / 'campaign.log').read_text()[-5000:])
for path in (root / 'results/p3-de-t4-20261101').glob('*/progress.json'):
    print(path.parent.name, path.read_text())
print('complete:', (root / 'results/p3-de-t4-20261101/measurements-complete.json').exists())
```

Module mới `p3_report.py` sinh JSON, bảng LaTeX, PDF/SVG từ raw CSV;
`p3_cases.py` xuất case ledger và ảnh diagnostic. Dependency báo cáo ở
`tools/controller/requirements-report.txt`. Các module này đã kiểm bằng
fixture local nhưng chưa chạy trên kết quả campaign hoặc compile LaTeX.
Sau khi kết nối phục hồi, cần đồng bộ module mới lên Colab, kiểm receipt
campaign, sinh báo cáo, xem corner case và hoàn tất draft Section 5.2/6.

**Cả bốn task chưa nghiệm thu** vì còn thiếu kết quả đo và phân tích cuối.

## Bản bàn giao trước khi kết nối T4

- **Ngày:** 01/11 (Chủ nhật), năm 2026.
- **Deadline phase:** 05/11 (Thứ 5).
- **Nhánh:** `de/p3-controller-evaluation`.
- **Trạng thái:** Công cụ host đã kiểm thử; cả bốn task chưa được nghiệm thu bằng số liệu T4.

## 1. Kế hoạch và trạng thái

| Bước | Phạm vi | Tiêu chí nghiệm thu | Trạng thái |
|---|---|---|---|
| 1 | Khảo sát main, dữ liệu, công cụ và kết nối | Xác định binary, policy và dataset dùng để đo | Main khớp remote; local chỉ có development; không điều khiển được Colab |
| 2 | Chuẩn bị loader, MCE, phân tích telemetry/A2 và checkpoint | Kiểm thử phép tính, failure accounting, duplicate và resume | Đã có module Python và test host |
| 3 | Instrumentation từng attempt và hai đường retry A2 | Trace đúng action/reason; cùng seed/input; reuse khớp full search | Chưa bắt đầu sửa CUDA/harness vì thiếu môi trường kiểm chứng |
| 4 | Chạy UDIS-D test và HPatches trên T4 | Đủ cohort/repeat; log phần cứng, seed, hash input/binary | Bị chặn ở kết nối Colab và dữ liệu test |
| 5 | Sinh bảng, biểu đồ, phân tích và draft Section 5.2/6 | Các con số truy được về raw CSV; gồm failure và limitations | Chưa có phép đo thật để viết kết luận |

Yêu cầu toàn quyền của Đế cho phép tạo nhánh và thực hiện các lựa chọn trong
task. Không push hoặc tạo PR. Quyết định và trade-off ở
[p3-de-evaluation.md](../decisions/p3-de-evaluation.md).

## 2. Công cụ đã có

| Module | Chức năng | Giới hạn |
|---|---|---|
| `tools/controller/p3_io.py` | Kiểm tra campaign, repeat, duplicate; hash và ghi JSON qua file tạm | Một CSV chứa đúng một mode/cấu hình mỗi pair |
| `tools/controller/p3_telemetry.py` | Phân bố cấu hình cuối, trạng thái, median/p95 theo pair; đọc trace nếu có | Không suy retry từ cấu hình cuối; không chứng nhận QoR |
| `tools/controller/p3_ablation.py` | Ghép cặp with/no reuse, kiểm cùng kết quả và tính tiết kiệm latency | Chưa có producer CSV A2 trên GPU |
| `tools/controller/p3_geometry.py` | Chuẩn bị 59 viewpoint sequence; inverse H_1_i; MCE; so sánh hai mode trên tập giao | Không tự tải dataset; input không được vượt allocation 2048 của harness |
| `tools/controller/p3_campaign.py` | Gọi harness hiện có theo pair/mode; checkpoint có hash; nối CSV hoàn chỉnh | Chỉ fixed/adaptive; không sinh Oracle, trace hoặc A2 |

Module phân tích không loại bỏ số đo chậm hoặc kết quả âm. MCE thống kê trên
các pair có đủ repeat hình học hợp lệ, kèm số pair bị lỗi. Lỗi S6 được báo
riêng vì homography có thể vẫn dùng được cho kiểm chứng hình học.

## 3. Tình trạng Colab và giữ phiên

MCP `open_colab_browser_connection` trả `true`, nhưng danh sách công cụ chỉ
có thao tác mở kết nối, không có đọc/chạy cell. Browser trả
`No browser is available`; danh sách trình duyệt là `[]`. Đã kiểm tra hướng
dẫn khắc phục Browser và tìm plugin liên quan; chưa có đường chạy notebook
khả dụng. Không dùng bridge CDP trong repo để thay thế backend Browser bị thiếu.

Không có lệnh disconnect, restart, đóng tab hoặc thay runtime nào được thực
hiện. Chưa có job nào được khởi chạy trên Colab. Không thể khẳng định phiên
đang được giữ hoặc bảo đảm dịch vụ không tự thu hồi runtime.

Runner không điều khiển lifecycle Colab, không mount/unmount Drive và không
tạo keepalive giả. Nếu có Drive đã mount, đặt output trên Drive để shard và
receipt tồn tại sau khi runtime mất. Job chỉ resume khi input, binary, policy,
seed, cấu hình và GPU identity khớp. Runtime mới có GPU UUID khác cần campaign
mới, không trộn với kết quả phiên cũ.

## 4. Lệnh kiểm chứng trên Colab T4

Chạy từ thư mục gốc repo. Cần source của nhánh này trên Colab, CUDA toolkit,
CMake, compiler, numpy, opencv-python và pytest. Không chạy
`python tools/colab_runner.py --sync`: bản hiện tại xóa thư mục repo trên
Colab khi đồng bộ.

```bash
nvidia-smi
nvcc --version
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j2
ctest --test-dir build --output-on-failure
python -m pytest tests/python -q -p no:cacheprovider --basetemp build/p3-tests
```

Output cần có: GPU Tesla T4; CUDA build nhắm `sm_75`; CTest không có test fail;
pytest không có test fail. Đây là output cần kiểm tra, chưa phải kết quả đã
quan sát trên Colab trong phiên làm việc này.

Với UDIS-D, chuẩn bị `data/udis_test/pairs.json` có đúng 1.106 pair. Root là
object; mỗi key là pair_id, mỗi value có `a`, `b` trỏ tới ảnh P6 PPM. Đường
dẫn ảnh tương đối với manifest. Conversion giữ nguyên kích thước và nội dung
pixel sau decode, không resize để làm vừa allocation. Không dùng development
manifest hoặc thay ngưỡng từ test. Runner hiện chưa tự kiểm con số 1.106;
phải đối chiếu cohort manifest với nguồn dataset trước khi nghiệm thu.

Ví dụ output dưới `results/p3-de-udis-t4`; khi Drive đã mount, có thể thay
`--output` bằng thư mục trên Drive. Lệnh sau chỉ thu kết quả cuối, chưa phải
toàn bộ telemetry cần cho task 1:

```bash
python -m tools.controller.p3_campaign --harness build/harness/stitch_harness --manifest data/udis_test/pairs.json --policy-source harness/main.cpp --dataset udis_d --split test --seed 20261101 --warmup 2 --repeats 10 --output results/p3-de-udis-t4
python -m tools.controller.p3_telemetry --runs results/p3-de-udis-t4/adaptive.csv --manifest data/udis_test/pairs.json --output results/p3-de-udis-t4/controller-summary.json
```

Output cần có: `completed.json` với `status=completed`, `pairs=1106`;
`fixed.csv`, `adaptive.csv`; `controller-summary.json` có `retry=null` khi
chưa cung cấp trace. Cấu hình fixed mặc định là S=0.5, K=1000, R=256,
feather; chỉ dùng làm Fixed-Best chính thức sau khi xác nhận quyết định
development tương ứng. Binary hash truy được executable đã chạy; hash source
policy ghi file được cung cấp, không tự chứng minh binary được build từ file đó.
Cần lưu log build và source receipt của campaign để chứng nhận quan hệ này.

Với HPatches, đặt bản sequence dataset tại `data/hpatches-sequences-release`:

```bash
python -m tools.controller.p3_geometry prepare --root data/hpatches-sequences-release --output data/p3-hpatches
python -m tools.controller.p3_campaign --harness build/harness/stitch_harness --manifest data/p3-hpatches/pairs.json --policy-source harness/main.cpp --dataset hpatches --split viewpoint --seed 20261101 --warmup 2 --repeats 10 --output results/p3-de-hpatches-t4
python -m tools.controller.p3_geometry evaluate --manifest data/p3-hpatches/pairs.json --runs results/p3-de-hpatches-t4/adaptive.csv --compare-runs results/p3-de-hpatches-t4/fixed.csv --output results/p3-de-hpatches-t4/mce.json
```

Output cần có: preparation ghi 59 sequence, 295 pair; campaign có 295 pair;
MCE JSON có `valid_pairs`, `pipeline_success_pairs`, ledger đầy đủ và delta
MCE Adaptive trừ Fixed trên tập giao hợp lệ. Không có ngưỡng pass MCE được
chốt từ số liệu final; kết quả này dùng để đánh giá, không mặc định chứng minh
Controller giữ nguyên độ chính xác.

## 5. Dữ liệu còn phải bổ sung từ harness

Định dạng đầu vào dưới đây là đề xuất riêng cho công cụ phân tích, chưa thay
đổi interface hoặc schema chung. Cần tích hợp và kiểm trên T4 trước khi dùng.

Trace CSV cần `run_id,pair_id,repeat_idx,is_warmup,attempt_idx,seed,S,K,R,
num_inliers,action,reason`. Attempt index bắt đầu bằng 0, liên tiếp; các attempt
trước cuối có action RETRY, cuối có ACCEPT hoặc REJECT. Nên bổ sung revision,
stage timing và generation trong producer để tăng khả năng audit. Chạy:

```bash
python -m tools.controller.p3_telemetry --runs results/p3-de-udis-t4/adaptive.csv --traces results/p3-de-udis-t4/attempts.csv --manifest data/udis_test/pairs.json --output results/p3-de-udis-t4/controller-trace-summary.json
```

A2 CSV cần `pair_id,repeat_idx,is_warmup,seed,variant,S,K,R_old,R_new,status,
num_matches,num_inliers,t_s4,t_s5,t_retry,generation,input_sha256,git_rev,h0,...,h8`.
Variant là `with_reuse` hoặc `no_reuse`; cùng S/K, tăng R. Producer phải lấy
đúng cohort R-only retry từ trace, giữ input/generation bất biến, reset state
đúng cách và xen kẽ thứ tự hai variant để giảm bias do drift GPU. No Reuse
chạy lại S4 và S5; With Reuse chỉ mở rộng S5. `t_retry` là host wall time
của retry, `t_s4/t_s5` là CUDA event; allocation nằm trong vùng nào phải ghi rõ.

```bash
python -m tools.controller.p3_ablation --retries results/p3-de-udis-t4/a2-retries.csv --output results/p3-de-udis-t4/a2-summary.json
```

Output cần có: ledger từng pair/transition, `saved_ms` và `saved_percent` có
thể âm. Đây chỉ là cohort có R-only retry; không suy rộng thành speedup toàn
bộ test set. Còn phải sinh bảng và biểu đồ từ JSON thật.

## 6. Khung phân tích paper sau khi có dữ liệu

Section 5.2 cần phân biệt phân bố nấc đầu và nấc cuối, báo retry theo pair,
số inliers tăng hoặc giảm sau escalation, lý do reject và chi phí retry. A2
báo mức tiết kiệm conditional cùng quy mô cohort. HPatches báo MCE trên tập
giao và failure rate của từng mode. Không dùng chỉ số self-check thay QoR.

Section 6 cần nêu giới hạn của một global homography trước parallax hoặc vật
thể chuyển động; gradient probe không xác định được overlap hay tính lặp
texture; self-check có thể chấp nhận model sai. Không gán nguyên nhân parallax
cho một pair chỉ từ `BUDGET_EXHAUSTED`. Cần kiểm ảnh hoặc annotation độc lập.
Chi phí probe, retry, đồng bộ và công việc phải chạy lại đều nằm trong end-to-end.

Chưa sửa nội dung LaTeX vì không có dữ liệu thật hoặc môi trường compile.
Khung này chưa thay thế draft phân tích có số liệu mà kế hoạch yêu cầu.

## 7. Kiểm chứng local và phần chưa xong

- Python: `python -m pytest tests/python -q -p no:cacheprovider --basetemp build/p3-full-pytest`
  đạt **377 passed**, gồm 39 test mới. Fixture không phải kết quả benchmark.
- Các CLI chạy `--help` được; `compileall` không lỗi.
- `git diff --check` không báo lỗi ở lần rà soát trước commit.
- Chưa build CUDA/C++, chưa chạy CTest/T4, chưa kiểm sanitizer, chưa compile
  paper và chưa đo UDIS-D test/HPatches/A2. Không có số đo GPU mới.
- Warning `.pytest_cache` permission từ git đã có trước task. Test dùng
  `-p no:cacheprovider` và basetemp trong `build/`; không đổi quyền cache cũ.

Để tiếp tục cần công cụ đọc/chạy cell Colab hoặc Browser điều khiển notebook
đã đăng nhập, runtime Tesla T4 và dữ liệu test. Các bước host độc lập đã làm
xong; nghiệm thu bốn task Phase 3 vẫn đang bị chặn.
