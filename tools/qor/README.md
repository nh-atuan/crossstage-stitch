# Bộ đánh giá QoR alignment offline (P1)

- **Ngày cập nhật:** 23/10 (Thứ 6), năm 2026; ngày đối chiếu nguồn SSIM ghi ở cuối.
- **Trạng thái:** Bản pilot; quality policy **chưa được Nhóm phê duyệt**.

Bộ đánh giá trên CPU thực hiện protocol **đề xuất để pilot** trong
[`docs/conventions.md`, mục 5–8](../../docs/conventions.md) và
[`docs/benchmark-qor-s4-s5-decisions.md`, mục 2](../../docs/benchmark-qor-s4-s5-decisions.md).
Các giá trị trong policy không phải ngưỡng chính thức của UDIS-D. Cần kiểm chứng
trên development data trước khi khóa protocol đánh giá cuối.

Công cụ đánh giá alignment từ H, chưa đánh giá seam/blending của S6 và không xác
nhận H là ground truth. Giữ blending cố định trong thí nghiệm S/K/R.

## Đầu vào và cách chạy

Dùng các dependency sẵn có trong `tools/requirements.txt`. Chạy từ thư mục gốc repo:

```powershell
python -m tools.qor.evaluate --manifest data/pairs.json --runs results/runs.csv --policy tools/qor/policy-pilot.json --ref-config S1_K2000_R1024_Bfeather --output results/qor.csv
python -m pytest tests/python/test_qor.py
```

Manifest bắt buộc là JSON ánh xạ rõ `pair_id` sang hai đường dẫn ảnh. Đường dẫn
tương đối được tính từ thư mục chứa manifest:

```json
{
  "pair-0001": {
    "a": "images/a.npy",
    "b": "images/b.png"
  },
  "pair-0002": {
    "a": "images/c.png",
    "b": "images/d.png",
    "reference_status": "reference_unreliable"
  }
}
```

Hai trường `a`, `b` phải là chuỗi đường dẫn không rỗng. File `.npy` phải chứa mảng
RGB uint8 H×W×3; không cho phép nạp pickle. Các định dạng ảnh khác được OpenCV
giải mã, sau đó đổi BGR→RGB. Cờ tùy chọn `reference_unreliable` ghi nhận reference
kém chất lượng sau khi kiểm tra trực quan; evaluator không tự phát hiện được mọi
artifact hình học hoặc thị giác.

`runs.csv` phải có tên cột duy nhất và ít nhất các cột `pair_id`, `config_id`,
`status`, `fail_stage`, `h0`…`h8`. H là row-major double, trong hệ RGB gốc, chiều
**B→A**. Dùng các trường `run_id`, `repeat_idx`, `is_warmup` của harness để phân
biệt lần chạy. Mỗi pair/config có cùng H và trạng thái được đánh giá một lần.
H/trạng thái khác nhau giữa các lần lặp, định danh lần lặp bị trùng, pair không có
trong manifest hoặc thiếu cấu hình reference đều bị báo lỗi. Kiểm tra tính tất
định bao gồm cả warm-up và lần đo. Dòng pipeline thất bại vẫn có trong kết quả;
các trường H của dòng đó không được phân tích hoặc chấm điểm. Trạng thái `OK`
yêu cầu `fail_stage` rỗng, bằng `0`, `none` hoặc `None`.

Đường dẫn output không được trùng manifest, runs CSV, policy hoặc bất kỳ ảnh đầu
vào nào. Kiểm tra cả đường dẫn đã resolve và cùng file theo filesystem, bao gồm
alias hardlink, trước khi mở output để ghi.

Output giữ đúng tám cột `qor.csv` đã quy ước. Điểm không xác định được ghi ô rỗng;
RGB MSE bằng đúng 0 được ghi PSNR là chuỗi `inf`. Candidate không hợp lệ có
`valid_stitch=0`; reference thất bại hoặc không có miền đánh giá đủ dùng làm mọi
cấu hình của pair có `reference_status=reference_failure`. Reference được đánh
dấu kém chất lượng giữ `reference_unreliable` và không thể pass.

Không âm thầm bỏ dòng thất bại. `quality_pass` và `pair_disposition` là helper
phân tích riêng; `Pass_i(c)` và `unachievable` không phải cột bổ sung trong CSV.

## Báo cáo phân loại từ qor.csv

CLI riêng đọc tám cột hiện có và policy pilot để tính pass/fail, lý do và số lượng
pair theo trạng thái; không phân tích latency hoặc chọn Pareto:

```powershell
python -m tools.qor.report --qor results/qor.csv --policy tools/qor/policy-pilot.json --output results/qor-report.json
python -m tools.qor.report --qor results/qor.csv --policy tools/qor/policy-pilot.json --candidate-config S0.50_K1000_R256_Bfeather --candidate-config S1_K2000_R1024_Bfeather --output results/qor-report-selected.json
python -m pytest tests/python/test_qor_report.py
```

Mặc định tập cấu hình hữu hạn là toàn bộ `config_id` xuất hiện trong CSV, **gồm
cấu hình reference**. Reference hợp lệ tự pass, nên nếu nó có trong tập này thì
`unachievable_pairs` có thể luôn bằng 0. Đó là hệ quả của định nghĩa, không tự loại
reference để tạo ra trường hợp unachievable. Caller có thể lặp `--candidate-config`
để chọn rõ tập cần khảo sát, có hoặc không gồm reference. Mọi pair phải có đủ mọi
cấu hình trong tập đã chọn và dòng reference; thiếu cấu hình sẽ báo lỗi. Tập chọn
được ghi trong JSON, không suy ra từ kết quả pass/fail. API cho phép tập rỗng tường
minh và ghi `no_candidates`; CLI luôn có ít nhất một cấu hình khi chọn tường minh.

JSON chứa `schema_version`, toàn bộ policy, `candidate_configs`, `counts` và danh
sách pair/config với `pass`, `reasons`, `is_reference`, `selected_candidate` và
metric. Đếm candidate chỉ áp dụng tập đã chọn; tất cả dòng CSV vẫn được phân loại
và giữ trong JSON. `reference_failure_pairs`, `reference_unreliable_pairs` và
`unachievable_pairs` được tách riêng; reference lỗi/kém tin cậy không được xem là
pair unachievable thông thường. `unachievable` chỉ nói về tập hữu hạn đã ghi,
không kết luận không tồn tại cấu hình tốt ngoài tập đó.

PSNR dương vô hạn được mã hóa bằng chuỗi JSON `"inf"`; metric không xác định là
`null`. Bộ ghi dùng `allow_nan=False`, không sinh token JSON không chuẩn như
`Infinity`/`NaN`. Dòng trùng pair/config, thiếu reference, mapping/trạng thái
reference mâu thuẫn, metric không hợp lệ hoặc coverage ngoài [0,1] đều bị từ chối.
Output không được ghi đè CSV/policy đầu vào, kể cả alias hardlink. Báo cáo tiếp tục
ghi **pilot-not-team-approved**, chưa có xác nhận ngưỡng của Đế hoặc cả Nhóm.

## Định nghĩa metric

Tâm pixel ở tọa độ nguyên. H được chuẩn hóa bằng hệ số có trị tuyệt đối lớn nhất,
kể cả dấu; **không yêu cầu h33 bằng 1**. Kiểm tra tương đối trên singular values
và mẫu số projective loại H suy biến/kém điều kiện số và đường chân trời projective
cắt hình chữ nhật đóng gồm các tâm pixel B. Đây là kiểm tra số học của triển khai,
không phải ngưỡng chất lượng học từ dữ liệu.

Trong hệ A, Ω gồm các pixel có tọa độ ánh xạ ngược qua H reference nằm trong hình
chữ nhật tâm pixel B `[0,width-1]×[0,height-1]`. Candidate dùng cùng quy tắc và
`V = Ω ∩ valid_candidate`. Coverage là `|V|/|Ω|`, **không chia cho overlap riêng
của candidate**. Ω rỗng là reference failure; V rỗng là candidate failure. Warp
dùng `cv2.warpPerspective`, `INTER_LINEAR`, với biên hằng 0. Tolerance tọa độ 1e-9
hấp thụ sai số làm tròn của nghịch đảo ma trận tại biên hình chữ nhật đóng.

PSNR dùng MSE trên **toàn bộ mẫu kênh RGB trong V**, với peak 255. SSIM dùng
moment cục bộ kiểu population có trọng số Gaussian (mặc định 11×11, σ=1.5),
`C1=(0.01×255)^2`, `C2=(0.03×255)^2`, rồi lấy trung bình SSIM từng kênh tại các
tâm hợp lệ. Một tâm chỉ được dùng khi **toàn bộ cửa sổ vuông nằm trong ảnh và V**,
kiểm tra bằng erosion nhị phân với biên hằng 0. Vì vậy pixel không hợp lệ hoặc
padding không ảnh hưởng các cửa sổ được chấm điểm. Miền không chứa cửa sổ SSIM
đầy đủ được báo thất bại, không tự gán SSIM bằng 0 hoặc 1.

Metric reference dùng Ω; metric candidate dùng V. Vì vậy phép so non-inferiority
có thể dùng hai tập pixel khác nhau, dù đã có ràng buộc coverage. Đây là định
nghĩa pilot hiện tại của repo; chưa loại bỏ hoàn toàn selection bias. Cần xem
phân phối chất lượng và failure rate, không tune ngưỡng trên final test. Reference
có PSNR hoàn hảo bằng `inf` chỉ cho phép candidate PSNR hoàn hảo pass với một
PSNR-drop tolerance hữu hạn.

## Nguồn và giới hạn đối chiếu

Công thức SSIM cục bộ dựa trên Wang, Bovik, Sheikh và Simoncelli (2004),
*Image Quality Assessment: From Error Visibility to Structural Similarity*:
[bản paper của tác giả](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf),
[DOI](https://doi.org/10.1109/TIP.2003.819861). Ngày đối chiếu tài liệu và nguồn
gốc protocol của repo: **22/10 (Thứ 5), năm 2026**. Đối chiếu công thức SSIM cục bộ
và cửa sổ Gaussian; chưa tái lập thí nghiệm của paper.

Mask miền cố định, yêu cầu cửa sổ hoàn toàn hợp lệ, cách gộp kênh RGB và các ngưỡng
pilot là lựa chọn đánh giá được ghi rõ của dự án. Chưa tuyên bố khớp SSIM mặc
định của thư viện khác hoặc toàn bộ protocol thực nghiệm của paper gốc. **Policy
vẫn chưa được Nhóm phê duyệt**; cần xác nhận trước khi dùng cho final evaluation.

## P2 — Quyết định QoR chuẩn hóa

Cập nhật 26/10 (Thứ 2), năm 2026. `quality_decision(report, policy,
execution_status="OK", fail_stage="")` được export từ `tools.qor.evaluate`.
Lõi `tools/qor/decision.py` là Python thuần, không chạy ảnh/GPU; hàm export kiểm
tra policy đúng schema trước khi gọi lõi. Kết quả bất biến có `status`, `passed`
và tuple `reasons`. `quality_pass(report, policy)` vẫn là API bool tương thích,
chỉ trả `True` cho `PASS`. Schema tám cột CSV và JSON policy không đổi.

| Status | Ý nghĩa |
|---|---|
| `PASS` | Đủ bằng chứng và đạt mọi ràng buộc của policy được truyền vào |
| `FAIL` | Có pipeline failure, stitch/H không hợp lệ hoặc vi phạm metric |
| `UNVERIFIED` | Thiếu reference/metric/trạng thái cần thiết, hoặc reference unreliable |
| `REFERENCE_FAILED` | Reference được xác nhận đã thất bại; không phải candidate QoR violation thông thường |
| `EVALUATION_ERROR` | Metric/trạng thái sai kiểu, sai miền, NaN/inf không hợp lệ hoặc lỗi evaluator tường minh |

`PASS` chỉ có nghĩa đạt **policy pilot đang truyền vào**, không xác nhận Nhóm đã
phê duyệt policy. Ngưỡng coverage/SSIM/PSNR không thay đổi. Biên so sánh inclusive:
coverage >= minimum, SSIM >= reference−drop, PSNR >= reference−drop. PSNR số học
`+inf` vẫn hợp lệ khi MSE bằng 0; reference `+inf` yêu cầu candidate cũng `+inf`
với tolerance hữu hạn. SSIM/coverage vô hạn, NaN, PSNR âm hoặc `−inf` đều là lỗi;
chuỗi CSV `inf` phải qua parser thành số trước khi gọi predicate thuần.

Caller có execution evidence phải truyền `execution_status`/`fail_stage` vào
hàm mới; hai giá trị mặc định chỉ tương thích với report evaluator đã có. Không
được suy ra runtime success chỉ từ H hoặc `valid_stitch=1`. Predicate không tự
chứng minh H đúng hình học; validity do evaluator xác nhận trước đó.

`tools.qor.report` dùng trực tiếp quyết định chuẩn, bổ sung trường JSON
`decision_status`; giữ `pass`, `reasons` và metric. Khi reference thất bại, lý do
chính là reference failure; không suy diễn candidate chưa chấm điểm là H sai.
`pair_disposition` giữ reference failure/unreliable riêng và trả `unverified` hoặc
`evaluation_error` thay vì `unachievable` khi chưa có đủ bằng chứng. Muốn gọi
unachievable phải có toàn bộ candidate được đánh giá và đều fail trong tập hữu
hạn đã công khai.

### Điểm tích hợp Tuấn/Đế cần review

`tools/analysis/analyze_gonogo.py::analyze_gonogo` hiện gán `quality_pass` từ
status/validity/coverage, chưa dùng SSIM/PSNR/reference status; đối số
`max_ssim_drop` chưa tham gia predicate. Chương chưa sửa script của owner khác.
Bản tích hợp tối thiểu cần:

1. Đọc policy bằng `load_policy`, không thêm threshold mặc định khác; tìm dòng
   reference theo `ref_config_id` cho **cùng pair**.
2. Gắn `reference_metrics` từ dòng reference vào mỗi candidate, rồi gọi
   `quality_decision(candidate, policy, execution_status=run_status,
   fail_stage=run_fail_stage)`.
3. Chỉ `decision.passed` được vào Fixed-Best/Oracle. Giữ riêng counts cho các
   status còn lại; thiếu reference không là pass hoặc unachievable đã chứng minh.
4. Với schema CSV có metric thiếu/sai, dùng parser strict hiện có hoặc validator
   tương đương; không biến NaN hoặc ô rỗng thành 0 bằng `safe_float`.
5. Chạy lại raw pilot sau tích hợp; JSON tổng hợp cũ không đủ dựng lại corrected
   speedup. Oracle fallback và lọc Pareto là vấn đề riêng thuộc Tuấn/Đế.

Ví dụ adapter dùng API hiện có, không phải schema controller tương lai:

```python
from tools.qor.evaluate import quality_decision

candidate = dict(qor_row, reference_metrics=reference_qor_row)
decision = quality_decision(candidate, policy,
                            execution_status=run_row["status"],
                            fail_stage=run_row["fail_stage"])
if decision.passed:
    eligible_for_oracle = True
```

Tests: `python -m pytest tests/python/test_qor.py
 tests/python/test_qor_report.py tests/python/test_qor_decisions.py -q` (viết trên
một dòng khi chạy shell). Fixtures chỉ kiểm logic/serialization; không thay dữ
liệu pilot, phê duyệt policy hay nghiệm thu controller/T4.
