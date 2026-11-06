# Quyết định: dựng baseline khi checkpoint v4 đã mất

- **Task:** Tự động setup toàn bộ dữ liệu/model cho notebook score ablation.
- **Người quyết định:** Đế; đã chọn “Dựng baseline mới để chạy trọn thí nghiệm”.
- **Ngày ghi nhận:** 06/11 (Thứ 6), năm 2026.
- **Phase:** Phase 3; deadline chung 07/11 (Thứ 7).
- **Trạng thái:** Code/tooling đã kiểm tra local; chưa chạy setup hoặc benchmark GPU trên Colab.

## 1. Bối cảnh

Đế không giữ checkpoint/model/manifest v4 trên Drive. Repo chỉ chứa source
và báo cáo; số liệu latency tổng hợp không phục hồi được các cây XGBoost.
Yêu cầu trước cấm retrain, nên agent đã hỏi lại. Đế cho phép dựng một baseline
mới một lần, rồi khóa nó cho toàn bộ sáu policy. Quyết định này là ngoại lệ
cho bước setup; không cho phép retrain giữa P0–P5 hoặc đổi default policy.

## 2. Phương án chọn

Notebook mặc định tự tải training.zip từ nguồn tác giả; hỗ trợ đường dẫn ZIP
private đã mount hoặc File ID bản sao công khai. Kiểm metadata toàn archive,
chọn 250 pair training theo component CRC/size và seed 20261101 trước khi
đo; chia 200 calibration/50 validation. Chỉ decode các ảnh được chọn sang PPM
gốc 512×512. Không tải test, không thay ảnh lỗi bằng synthetic hoặc pair khác.
Hash archive được ghi là quan sát, không là checksum do tác giả công bố.

Build/test Release sm_75. Thu grid calibration với 2 warmup/10 repeat và feature
observations riêng; đối chiếu H/status/count. Chạy pipeline v4 photo hiện có
trên calibration, giữ nested-fit/calibration và raw training exports. Sau đó
khóa baseline thí nghiệm với pre 0.85, gate 0.65 và extra_ms 0.429087 trước
validation. Ghi cả threshold gốc của trainer và model hash, không dùng báo cáo
training dưới threshold khác làm kết quả của policy cố định.

Từ lúc khóa, P0–P5 dùng cùng ba ensemble và feature. Vẫn chỉ thay scoring và
pre-threshold theo [quyết định score](p3-controller-score-ablation.md). Mọi
output ghi reconstructed baseline; không dùng số v4 lịch sử làm baseline
paired, không nói weights mới tái lập chính xác weights đã mất. Lịch sử split
không còn nên không chứng nhận cohort chưa từng được xem trong các lần v4 cũ.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Hạn chế | Quyết định |
|---|---|---|---|
| Tìm lại policy.json cũ | Giữ nguyên weights lịch sử | Đế không còn artifact | Chỉ dùng nếu tìm lại được; không giả định tồn tại |
| Dựng v4 photo từ training.zip | Tự chạy từ source, có provenance và so công bằng P0–P5 | Tốn sweep/training lần đầu; kết quả khác lịch sử | Đế đồng ý |
| Dùng latency báo cáo để thay model | Không phải train | Không tạo được prediction/route/gate cho input; không thể đo scoring mới | Không dùng |
| Model giả hoặc synthetic data | Có thể demo CLI | Không trả lời câu hỏi trên UDIS-D | Không dùng |

## 4. Checkpoint và fallback

Giữ data/model trên Drive và ZIP setup di động, kiểm hash trước tái sử dụng.
ZIP chỉ có model/split/ảnh đã chọn, không có token hoặc full training archive.
Raw sweep/feature/training evidence lưu riêng dưới work và kèm báo cáo nếu có.
Báo cáo phân biệt cheap_reject (REJECT không retry) với cheap_stop (ACCEPT),
không đếm terminal rejection là một lần dừng rẻ thành công.
Build/test được cache theo source/binary/GPU; partial GPU measurements không
resume qua runtime khác. Setup xong có thể dùng model đã khóa trên runtime
mới; training measurements không được ghép với benchmark paired mới.

Download quota thì dừng với hướng dẫn upload ZIP riêng tư; không tự thay nguồn
không truy vết. Setup chưa xong và MODE=SMOKE chỉ đo Reference trên ba pair,
không fit hoặc giả lập sáu policy. Calibration sweep không phải Oracle cho
validation; vẫn ghi UNAVAILABLE nếu không có sweep validation tương thích.

## 5. Kiểm chứng

Kiểm local bằng CPU fixtures cho ZIP index/split/PPM, checksum/resume,
private-path/public-ID acquisition, bundle restore an toàn, baseline export
và cache không fit lại. Các test không tải UDIS thật và không tạo benchmark
GPU. Kết quả local ngày 06/11 (Thứ 6):

- Toàn suite Python: 497 PASS, 4 SKIP do thiếu OR-Tools/Qiskit ở test quantum ngoài phạm vi; không tắt hoặc nới test.
- 23 test tập trung score/setup/notebook PASS, gồm orchestration, tách timing/feature, checksum/resume và Python/C++ parity trên fixture.
- Notebook qua schema nbformat, compile mọi code cell, output trống; bootstrap Secret được test cách ly, không gọi Colab.
- `compileall` và `git diff --check` không báo lỗi. Chưa chạy download UDIS thật, training baseline đầy đủ, build CUDA hoặc Run all trên T4; môi trường local không có nvcc/GPU.


## 6. Nguồn

- Yêu cầu và lựa chọn trực tiếp của Đế trong phiên ngày 06/11 (Thứ 6).
- [Pipeline huấn luyện hiện có](../../tools/controller/train_pooled.py), [chọn cohort](../../tools/controller/real_cohort.py), [hướng dẫn notebook](../controller-score-colab-run-guide.md).
- [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching), [gdown](https://github.com/wkentaro/gdown), truy cập 06/11 (Thứ 6), năm 2026.
