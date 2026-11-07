# Chạy DCS-R1 trên Colab T4

Notebook duy nhất: [controller_direct_config_selection_t4.ipynb](../notebooks/controller_direct_config_selection_t4.ipynb).

1. Tải notebook và mở [Colab](https://colab.research.google.com/) → File → Upload notebook. Có thể mở từ GitHub nếu Colab đã được cấp quyền repo private.
2. Secrets → `GITHUB_TOKEN` → bật **Notebook access**. Fine-grained PAT cần **Contents: Read-only**, resource owner `nh-atuan`, đúng repository `crossstage-stitch`. Khi tạo token: GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens. Không dán token vào cell.
3. Runtime → Change runtime type → **T4 GPU**.
4. Runtime → **Run all**, chấp nhận quyền mount Drive, tải ZIP cuối notebook.

CONFIG đã điền repo/branch và paths; checkout pin full SHA. Seed/model/search/features/policies tự chọn theo protocol, không chỉnh tay. Pipeline chạy trong subprocess mới sau pip, không yêu cầu restart kernel. CPU dùng cho train/analysis; T4 dùng build/test, bounded profiles và E2E. Giữ nguyên 10 repeat.

## Đầu vào tự tìm

- Full P0–P5 ZIP/grid: Drive `controller-score-work`, `controller-score-results`, các path cũ hoặc repo results. Nếu chưa có trên Drive, đặt `full-1382a8b5cae5-68c68d47bc6c42ad93cc2974fcd59193.zip` trong **MyDrive/controller-score-results/**.
- Frozen baseline v5: giữ ZIP `controller-v5-<run_id>.zip` trong **MyDrive/controller-v5-results/**. Cần `W5/policies/V5_COMBINED.json`, `W5/policies/V4.json`, `provenance.json`, `artifact-hashes.json` khớp grid/split/QoR. Không cần chạy lại W1–W5. Thiếu ZIP này thì native comparison bị chặn, các phase offline vẫn xuất ZIP.
- A–E: notebook tìm `MyDrive/controller-diagnostics-AE/` để kiểm/reuse đúng input/split/capacity. Không coi OOF predictions là raw features.
- Ảnh: khôi phục từ old work inputs/portable setup bundle, kiểm input-hashes. Nếu mất ảnh, dùng training archive cũ hoặc tải Drive ID đã biết từ workflow trước; kiểm archive SHA và đúng cohort. Không tự chọn ảnh mới, không dùng testing.zip thay training.zip.

Nếu dữ liệu thực sự thiếu, xem `MISSING_REQUIRED_INPUTS.md` và `data-inventory.json`. Notebook không thể lấy file còn ở máy cá nhân hoặc trong chat. Chỉ cần bổ sung ZIP đúng thư mục trên Drive rồi Run all; các path override trong CONFIG là tùy chọn khi Drive khác cấu trúc cũ.

## Tiến trình và resume

Thứ tự data → method PRE0 → features → learning → replay Gate → native → gpu. Xem `RUN_MATRIX.md` trước train/đo. Không đo lại 48-grid; tối đa 4.200 native E2E runs cho 50 pair × 7 policies × 12 repeats/warmups, cộng tối đa 432 bounded feature-profile runs. Multiclass có 48 trees mỗi round; CPU CV/learning curve có thể chiếm phần lớn thời gian. Chưa có thời gian đo thật để hứa runtime.

Checkpoint từng model/phase/batch ghi input/model/data/QoR/schema/source/protocol hashes. Run all cùng source/environment/data kiểm checksum rồi resume. Khác fingerprint tạo run mới, không ghép timing campaign khác. Khi chỉ GPU/CUDA đổi và CPU identity còn trùng, copy các phase CPU đã kiểm hash; donor GPU feature costs bị xóa để đo lại trên T4 hiện tại. Frozen v5 inputs cũng nằm trong fingerprint. Phần PARTIAL/BLOCKED được thử lại khi input phù hợp; nếu input/provenance đổi thì chạy dưới run ID mới. Không overwrite P0–P5/A–E/v5 cũ.

## ZIP gửi lại

Bản lưu: **MyDrive/controller-direct-selection-results/dcs-<run_id>.zip**. Cell cuối tự download nếu ZIP dưới 250 MB; lớn hơn thì in Drive path và SHA256. Gửi nguyên ZIP, không chỉ ảnh bảng.

ZIP có training Parquet/CSV, label audit, split/catalog/provenance; objective/top1/ranking comparisons; feature atlas/ablation/permutation/R sensitivity; learning curve CSV và PNG/PDF; Gate false accept/reject ledger; frozen models/native parity; raw E2E/QoR/attempts/stage profiles khi đã đo; logs, status, README và checksums. Ảnh dataset/token không nằm trong ZIP.

Đọc `README-results.md`, `results-summary.json`, `phase-status.csv`. Trạng thái PARTIAL/BLOCKED/FAILED vẫn có ích để debug. Report trả lời objective/ranker-vs-direct48/Oracle-gap/feature/Gate/K500/E2E theo evidence; không tự promote policy khi development tốt.
