# Chạy score ablation Controller v4 trên Colab T4

Notebook: [controller_v4_score_ablation_t4.ipynb](../notebooks/controller_v4_score_ablation_t4.ipynb).
Đế chạy notebook và gửi ZIP về chat để phân tích. Agent không kết nối Colab,
không khởi chạy GPU và chưa có kết quả benchmark của lần sửa này.

## 1. Chuẩn bị GitHub và Secret

Bản code chứa notebook cần có trên GitHub trước khi clone. Agent chỉ commit
local; Đế tự push nhánh khi muốn chạy. Có thể mở notebook bằng cách upload
file local ngay cả khi repo private.

Tạo token theo [hướng dẫn GitHub](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens):

1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token.
2. Chọn Resource owner sở hữu repo, đặt hạn dùng ngắn phù hợp.
3. Repository access → Only select repositories → chọn đúng repository.
4. Repository permissions → Contents → Read-only. Metadata Read-only được GitHub cấp tự động.
5. Nếu tổ chức yêu cầu approval, đợi duyệt. Tài khoản cũng phải có quyền đọc repo; token không cấp thêm quyền tài khoản.
6. Trong Colab, mở Secrets bằng biểu tượng chìa khóa, thêm tên `GITHUB_TOKEN`, lưu token vào Secret và bật Notebook access.

Không dán token vào CONFIG, cell, URL hoặc file. Nếu GitHub không hỗ trợ
Fine-grained PAT cho loại quyền cộng tác hiện tại, nhờ owner cấp cơ chế đọc
repo phù hợp; notebook dừng, không tự đổi sang token có quyền ghi.

Notebook dùng HTTPS URL sạch và ASKPASS tạm. File ASKPASS không chứa token;
Git nhận token qua môi trường tiến trình con, môi trường notebook không được
cập nhật. Output network được giữ kín. Credential helper bị tắt cho các lệnh
đó; token không được lưu vào remote, checkpoint hoặc ZIP. Sau clone, source
được khóa bằng full commit SHA; rerun cùng runtime dùng lại commit đã khóa.

## 2. Chuẩn bị artifact trên Drive

Cần các input thật từ checkpoint v4, không chỉ source Git:

| CONFIG | Input |
|---|---|
| `POLICY` | `v4-fast-ablations/photo/policy.json` gốc: pre 0.85, gate 0.65, đủ ba ensemble và 48 cấu hình |
| `SPLIT` | File split gốc có SHA-256 khớp `policy.provenance.input_sha256.split`; thường là `v4-inputs/split.json` |
| `CALIBRATION_MANIFEST` | Manifest toàn calibration đã khóa; đường dẫn A/B đọc được trong runtime |
| `EVALUATION_MANIFEST` | Manifest toàn validation của split; đường dẫn A/B đọc được trong runtime |
| `ORACLE_SWEEP_DIR` | Không bắt buộc; full sweep tương thích có `runs.csv`, `qor.csv`, `configs.json`, `receipt.json` |

Manifest có dạng `{ "udis_train_000001": { "a": "images/a.ppm", "b": "images/b.ppm" } }`.
Đường dẫn ảnh tương đối với manifest hoặc là đường dẫn tuyệt đối tồn tại trên
Colab. Ảnh phải là P6 PPM 512×512 gốc; không resize hoặc thay ảnh để làm chạy.
Nếu manifest cũ trỏ tới runtime đã mất, tạo bản relocated riêng, giữ nguyên
manifest và split gốc. Notebook đối chiếu danh sách pair và hash ảnh, từ chối
calibration/validation dùng chung ảnh. Checkpoint không mặc nhiên chứa dataset.

Không sửa split để làm khớp model: hash split và quality policy phải đúng
provenance. Preflight báo đường dẫn thiếu trước build. Không dùng UDIS-D test
hoặc HPatches. Validation v4 đã xem luôn được ghi nhãn diagnostic; không gọi
kết quả này là đánh giá độc lập và không chọn ngưỡng rồi báo lại trên nó.

## 3. Mở notebook và Run all

1. Vào [Google Colab](https://colab.research.google.com/), chọn Open notebook → Upload và mở file `.ipynb` đã bàn giao.
2. Runtime → Change runtime type → GPU → T4 nếu có lựa chọn. Cell kiểm môi trường dừng nếu GPU thực tế không phải Tesla T4.
3. Chỉnh duy nhất cell CONFIG: `REPOSITORY=OWNER/REPO`, `GIT_REF` là branch hoặc full commit đã push, bốn đường dẫn input và `OUTPUT_ROOT`. Không điền token.
4. Để `MODE="FULL"` rồi chọn Runtime → Run all. Notebook tự mount Drive, clone, cài requirements, kiểm input, build, test, smoke và đo đủ sáu policy. Việc cấp quyền Drive/Secret lần đầu có thể cần thao tác trên giao diện Google.
5. Cell cuối tải ZIP. Nếu trình duyệt chặn tải tự động, lấy ZIP trong OUTPUT_ROOT trên Drive hoặc Colab Files, rồi gửi ZIP cùng full commit SHA về chat.

Có thể đặt `MODE="SMOKE"` để kiểm ba pair đầu trước; sau đó đổi thành FULL
và Run all. FULL luôn tự smoke trước nên không bắt buộc chạy hai lần. Không
có cell chọn từng policy hoặc yêu cầu nhập lệnh. Seed 20261101, hai warmup và
mười repeat cố định trong code. Không cài quantum SDK cho campaign này.

Source checkout được dùng lại trong cùng runtime để giữ binary/checkpoint
ổn định. Muốn đổi repo/ref sau khi đã khóa, dùng runtime mới. Nếu clone bị
dở trước khi tạo lock, xóa riêng `controller-score-source` dưới `/content`
bằng Files rồi Run all; giữ nguyên checkpoint trên Drive.

## 4. Notebook đo gì?

| Policy | Score | Pre-threshold |
|---|---|---:|
| P0 | old | 0.85 |
| P1 | old | 0.70 |
| P2 | old | 0.50 |
| P3 | reuse_approx | 0.85 |
| P4 | reuse_approx | 0.70 |
| P5 | reuse_approx | 0.50 |

Giữ model XGBoost, feature, gate 0.65, self-check, QoR và fallback hai attempt.
Không gọi trainer cho model này. Unit tests có thể huấn luyện fixture tổng
hợp để kiểm export; chúng không thay policy thí nghiệm. Policy mặc định của
harness vẫn giữ nguyên.

Notebook build Release `sm_75`, chạy CTest và pytest. Test fail hoặc GPU test
bị skip thì dừng, giữ log. Sau đó đo Reference trên calibration bằng chính
binary mới để lấy profile stage; đây là phép đo thêm, không phải retrain.
R-only trừ S1–S4, đổi K trừ S1, đổi S giữ toàn bộ chi phí Reference. Dùng p95
của tỷ lệ chi phí theo pair; giữ nguyên S5, S6 và host overhead. Các profile
lỗi không tạo discount. Không dùng số retry A2/v3 cho mọi transition.

Công thức mới là ranking approximation, không phải mô hình E2E chính xác:
`L_hat(c) + (1-p_pre) * L_hat(ref) * fraction(c→ref) + extra_ms`.
`1-p_pre` chỉ là failure proxy. `L_hat(c)` đã gồm S6, trong khi attempt bị
reject không render S6; score vẫn thiên bảo thủ ở branch này. Không cộng
score hoặc các stage để báo E2E. E2E là số host wall-clock do harness đo.

Đo Reference và P0–P5 trên cùng cohort, xoay thứ tự theo batch. Smoke và FULL
là hai campaign riêng. Python/C++ kiểm prediction, initial route và post-gate
qua dữ liệu quan sát ngoài vùng timing. Không thêm logging từng candidate
vào đường đo. Candidate diagnostics được tính sau measurement.

Oracle không tự kích hoạt sweep 48 cấu hình. Nếu có sweep đúng binary hash,
seed, ảnh, policy và đủ repeat, notebook audit rồi đo lại cấu hình Oracle đã
chọn trên cùng runtime. Timing sweep cũ giữ riêng, không ghép paired với
campaign mới. Đối chứng đo lại không phải phép tìm minimum mới trên toàn grid.
Thiếu provenance hoặc không có đáp án khả thi thì ghi UNAVAILABLE và lý do.

## 5. Checkpoint, output và gửi kết quả

Dưới OUTPUT_ROOT, mỗi MODE/commit/runtime có thư mục và ZIP riêng. Sau mỗi
policy trong mỗi batch có CSV/trace, receipt/hash và ZIP checkpoint. Rerun
trong cùng runtime chỉ tiếp tục shard đã khớp hash; runtime mới tạo thư mục
mới, không trộn số đo thành paired campaign. Dữ liệu trong Drive tồn tại sau
khi Colab mất runtime; ổ `/content` không có bảo đảm đó.

ZIP hoàn chỉnh chứa:

- `environment.json`, `gpu.log`, `nvcc.log`, `python-packages.log`, `source.tar`, CONFIG và source SHA.
- `baseline-policy.json`, `split.json`, `quality-policy.json`, input hashes và manifest relocated.
- `fallback-profile.json`, `experiment-freeze.json`, `policies/P0.json` đến `P5.json`.
- `build-test/`: log configure/build/CTest/pytest.
- `profile/`, `smoke/`, `campaign/`: raw run CSV, attempt traces, feature observations, checkpoint receipts và freeze.
- `campaign/comparison.csv`, `summary.json`, `pair-metrics.csv`, `qor.csv`, `diagnostics.json`, score diagnostics và parity CSV; ở chế độ SMOKE các báo cáo nằm trong `smoke/`.
- `oracle-comparison.json` và đối chứng Oracle nếu input phù hợp.
- `artifact-hashes.json` kiểm từng file. Nếu lỗi giữa chừng có `failure.json` và ZIP partial; không xem đó là completed campaign.

Latency lấy median mười repeat của từng pair rồi mới mean/median/p95; paired
speedup và bootstrap resample theo pair. Mẫu chính giữ mọi pair reference-valid,
kể cả candidate FAIL; PASS/FAIL/unknown trên toàn cohort vẫn được xuất riêng.
Có retry theo transition, cấu hình đầu/cuối, K500, false accept và candidate
alignment false reject. Candidate bị từ chối chưa render S6 nên metric
alignment đó không được gọi là pipeline success. Không drop lỗi hoặc pair chậm.

Gửi **ZIP nguyên vẹn + full commit SHA**, không cần sửa CSV hay chép số bằng
tay. Agent sẽ xác minh hash/seed/source rồi mới phân tích. Chưa có số liệu T4
thì chưa kết luận score mới cải thiện và chưa đổi default policy.

Nguồn xác thực GitHub và trang rà văn phong
[Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
được đọc ngày 06/11 (Thứ 6), năm 2026.
