# Chạy score ablation từ đầu trên Colab T4

Mở [controller_v4_score_ablation_t4.ipynb](../notebooks/controller_v4_score_ablation_t4.ipynb).
Không cần model, manifest hoặc kết quả v4 cũ. Đế đã đồng ý dựng một baseline
v4 photo mới vì không còn checkpoint. Sau khi dựng xong, cả P0–P5 dùng nguyên
model đã khóa. Đây không phải tái lập đúng weights hoặc timing v4 lịch sử.

## 1. Setup đơn giản

1. Bảo đảm nhánh `de/optimize-controller` trên GitHub đã có commit setup mới. Agent commit local, không tự push.
2. Upload notebook mới vào [Colab](https://colab.research.google.com/). Chọn Runtime → Change runtime type → GPU T4.
3. Trong Secrets, tạo `GITHUB_TOKEN`, bật Notebook access. Token Fine-grained chọn đúng repo và **Contents: Read-only**; xem [hướng dẫn GitHub](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
4. CONFIG đã điền repo `nh-atuan/crossstage-stitch`, branch và ID dữ liệu mặc định. Nếu dùng repo/branch khác thì sửa hai giá trị đó. Giữ `MODE="FULL"` và `MOUNT_DRIVE=True`.
5. Chọn Runtime → Run all, cấp quyền mount Drive nếu được hỏi. Không chọn từng policy, không nhập lệnh hoặc tự tạo manifest.

Dùng notebook bản mới. Với GIT_REF là branch, mỗi Run all fetch lại branch
và khóa full SHA cho lượt đó; có thể giữ runtime hiện tại để nhận bản sửa.
Với GIT_REF là full SHA, notebook giữ đúng commit đó, nên cần đổi SHA khi muốn
nhận bản sửa mới. Không ghi đè tracked files đã sửa tay. `LOCKED COMMIT` được
in sau checkout và lưu trong kết quả. Token không xuất hiện trong URL,
remote, checkpoint hoặc ZIP.

Nếu gặp `ModuleNotFoundError: No module named 'tests.python'` ở pytest:

1. Bảo đảm branch trên GitHub có bản sửa package test và dùng notebook mới.
2. Giữ nguyên WORK_ROOT trên Drive; không xóa raw ZIP, images hoặc prepared.json.
3. Run all lại. Notebook dùng lại archive/ảnh đã kiểm hash và chạy lại build/tests. Lỗi cũ xảy ra trước sweep và training, nên chưa có số đo cần ghép vào campaign mới.

Nguyên nhân đã tái hiện local: thư mục tests chưa là regular package nên có
thể bị package khác cùng tên tests trong site-packages che khuất. Bản sửa
thêm package markers và dùng import tests.python nhất quán; không bỏ qua
pytest hoặc loại test đang lỗi.

## 2. Dữ liệu: tự tải hoặc upload một ZIP

Mặc định để `TRAINING_ZIP=""`. Notebook tải **training.zip** từ File ID
`1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a` đã có trong CONFIG. Dataset do
[tác giả UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching)
cung cấp; công cụ tải dùng [gdown](https://github.com/wkentaro/gdown).
Không tải testing.zip và không chuyển sang synthetic khi thiếu dữ liệu.

Nếu Google Drive báo quota hoặc không tải được:

1. Mở [training.zip của tác giả](https://drive.google.com/file/d/1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a/view) trong trình duyệt và tải về máy.
2. Upload nguyên `training.zip` vào MyDrive, ví dụ thư mục `UDIS`. Không giải nén, không tự đóng lại hoặc đổi cấu trúc.
3. Chỉ sửa dòng CONFIG tương ứng rồi Run all:

```python
"TRAINING_ZIP": "/content/drive/MyDrive/UDIS/training.zip",
```

Nếu đã có bản sao công khai trên Drive, có thể giữ TRAINING_ZIP trống và thay
`TRAINING_DRIVE_ID` bằng File ID trong link `/file/d/FILE_ID/view`. File riêng
tư không cần chia sẻ công khai: dùng đường dẫn đã mount như ví dụ trên.
GitHub PAT chỉ dùng để đọc repo, không dùng để tải dataset.

Notebook đọc metadata của toàn archive để kiểm số pair, counterpart và nhóm
ảnh trùng CRC/size. Chỉ decode 250 pair được chọn sang P6 PPM 512×512 gốc;
không giải nén toàn dataset. Giữ seed 20261101, chia 200 calibration và 50
validation theo component trước khi xem QoR/latency. Hash ZIP là hash quan
sát được; không coi đó là checksum đã được tác giả công bố.

## 3. Lần đầu notebook tự làm gì?

1. Kiểm đúng Tesla T4/nvcc, clone private repo và khóa full source SHA.
2. Tải/đọc training.zip, kiểm cấu trúc và 10.440 pair; tạo split và manifest, giữ các pair khó hoặc lỗi.
3. Build Release `sm_75`, chạy CTest và pytest. Lỗi test thì dừng và giữ log.
4. Sweep 48 cấu hình trên 200 pair calibration, mỗi cấu hình 2 warmup/10 repeat; lưu raw CSV, metadata và checkpoint từng batch.
5. Thu feature riêng, kiểm H/status/count không đổi so với sweep và chấm canonical QoR. Timing có feature instrumentation không thay timing fixed trong nhãn huấn luyện.
6. Chạy pipeline `train_pooled.py` hiện có cho nhóm photo. Chỉ dùng calibration; giữ raw export, outer-fold evidence và training logs.
7. Xuất baseline thử nghiệm với pre 0.85, gate 0.65 và extra_ms 0.429087 trước khi đo validation. Giữ nguyên các cây, feature và self-check của model mới; lưu cả threshold trainer đã chọn để tránh đánh tráo báo cáo training với policy cố định.
8. Đóng model/split/ảnh đã chọn thành ZIP setup trên Drive; khóa model rồi tự smoke và chạy đủ Reference + P0–P5.

Lần đầu lâu hơn notebook chỉ đo scoring vì phải thu lại sweep và huấn luyện.
Chưa có log T4 của luồng setup này nên chưa chốt thời gian tổng. Không chạy
training hoặc solver khác đồng thời với phép đo GPU.

Sau khi setup hoàn tất, Run all dùng lại model/data đã kiểm hash; không fit
lại giữa P0–P5 hoặc mỗi lần chạy. Baseline mặc định trong source không đổi.
Build/test chỉ chạy một lần cho cùng source, binary và runtime; log được sao
vào kết quả. Không ghép partial sweep từ hai phiên GPU.

Nếu MODE=SMOKE khi chưa có model: notebook kiểm dataset/build/test và chỉ đo
Reference trên 3 pair, không sweep/fit hoặc giả lập sáu policy. Đổi FULL rồi
Run all để dựng baseline và chạy đủ thí nghiệm. Nếu model đã lưu, SMOKE chạy
sáu policy trên 3 pair như trước. FULL tự smoke nên không bắt buộc chạy hai lần.

## 4. Những gì được giữ nguyên trong P0–P5

| Policy | Score | Pre-threshold |
|---|---|---:|
| P0 | old | 0.85 |
| P1 | old | 0.70 |
| P2 | old | 0.50 |
| P3 | reuse_approx | 0.85 |
| P4 | reuse_approx | 0.70 |
| P5 | reuse_approx | 0.50 |

Cả sáu dùng cùng model XGBoost đã khóa, feature photo, gate 0.65, self-check,
QoR pilot và seed. Thay score/pre-threshold, không huấn luyện giữa policy.
R-only trừ S1–S4 từ profile Reference calibration; đổi K trừ S1; đổi S giữ
full cost. Giữ full S5/S6 và host overhead. `1-p_pre` là failure proxy;
score là xấp xỉ vì fixed latency candidate đã gồm S6, nhưng candidate bị
reject chưa render S6. E2E báo cáo là wall-clock do harness đo, không cộng stage.

Thứ tự mode xoay theo batch, checkpoint sau mỗi policy/batch. Latency lấy
median mười repeat theo pair rồi mean/median/p95; bootstrap resample theo pair.
Mẫu Reference-valid vẫn gồm candidate FAIL; PASS/FAIL/unknown giữ toàn cohort.
False reject từ H candidate chỉ là alignment diagnostic, chưa chứng minh
candidate pipeline đã render S6 thành công.

Sweep setup thuộc calibration, không dùng làm Oracle validation. Không có
sweep validation tương thích thì Oracle ghi UNAVAILABLE; không âm thầm chạy
thêm full sweep. Không ghép timing v4 cũ với measurement mới, không khẳng định
model dựng lại tương đương v4 cũ. Vì lịch sử split v4 bị mất, không chứng nhận
cohort mới chưa từng được xem; chưa đưa ra claim đánh giá độc lập.

## 5. File được lưu và cách gửi lại

Mặc định notebook tạo hai thư mục trên Drive:

| Vị trí | Nội dung |
|---|---|
| `controller-score-work/` | ZIP gốc tải về, ảnh/split đã chuẩn bị, sweep/feature/training logs theo session, model và setup receipt |
| `controller-score-work/controller_v4_reconstructed_setup_bundle.zip` | Model, split, manifest và ảnh PPM của 250 pair; giữ file này để không mất setup |
| `controller-score-results/` | Campaign đo score và ZIP kết quả để gửi lại |

ZIP kết quả có `comparison.csv`, `summary.json`, pair metrics, canonical QoR,
raw CSV, attempt/feature traces, score diagnostics, model P0–P5, environment,
source SHA, build/test logs và `artifact-hashes.json`. Khi có trên máy chạy,
`setup-evidence/` kèm raw grid, nhãn, feature và export training để truy nguồn.
Không đưa ZIP dataset gốc hoặc Secret vào ZIP kết quả.

Cell cuối tự tải ZIP kết quả. Gửi **ZIP nguyên vẹn + LOCKED COMMIT** trong
chat tiếp theo. Không cần upload bộ dataset lớn lại cho agent. Giữ ZIP setup
trên Drive hoặc tải về máy để dự phòng.

Nếu muốn khôi phục trên Drive/máy khác, upload ZIP setup và điền duy nhất:

```python
"SETUP_BUNDLE_ZIP": "/content/drive/MyDrive/controller_v4_reconstructed_setup_bundle.zip",
```

Dùng WORK_ROOT trống khi restore. Notebook kiểm member/hash, khôi phục model
và ảnh rồi chạy scoring, không cần training.zip hoặc fit lại. File setup
không chứa toàn bộ raw training evidence; giữ thư mục work nếu cần tái lập
quá trình huấn luyện hoặc bàn giao sweep cho Thành.

Nếu ngắt phiên trước khi model được đóng gói, các shard/log vẫn ở Drive.
Runtime mới mở setup measurement riêng để không trộn GPU session; dataset
đã chuẩn bị được dùng lại. Nếu lỗi, gửi ZIP partial/log từ `setup-sessions/`;
không coi partial là completed campaign.

Nguồn UDIS, gdown và hướng dẫn PAT được đối chiếu ngày 06/11 (Thứ 6), năm
2026; văn bản rà theo [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing).
