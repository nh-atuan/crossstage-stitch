# Chạy Controller v5 bằng một notebook

Notebook duy nhất cần mở: [controller_v5_unified_t4.ipynb](../notebooks/controller_v5_unified_t4.ipynb).

## Các thao tác trên Colab

1. Tải notebook về máy, mở [Colab](https://colab.research.google.com/) → **File → Upload notebook**. Repo private không bắt buộc mở notebook qua GitHub.
2. **Secrets** (chìa khóa) → `GITHUB_TOKEN` → **Notebook access ON**. Token Fine-grained của GitHub phải được cấp quyền **Contents: Read-only** cho resource owner `nh-atuan`, repository `crossstage-stitch`. Nếu cần tạo lại: GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens. Không dán token vào cell hoặc gửi token cho agent.
3. **Runtime → Change runtime type → T4 GPU**. Notebook từ chối gắn nhãn T4 cho GPU khác; các phép CPU còn có thể chạy, W5 bị chặn rõ ràng.
4. **Runtime → Run all**; chấp nhận quyền mount Google Drive khi được hỏi. Không chọn từng phase/model/threshold/policy. Repo/branch đã điền; source được pin full SHA ngay sau checkout.
5. Tải ZIP ở cell cuối và gửi nguyên ZIP lại. Luôn có bản lưu trong **MyDrive/controller-v5-results/controller-v5-<run_id>.zip**; nếu browser chặn auto-download, tải từ Drive theo đường dẫn in cuối notebook.

Không cần thay CONFIG khi dùng repo và Drive cũ. CONFIG chỉ có repo/ref và những path môi trường tùy chọn; seed 20261101, 2 warmup, 10 repeat và danh sách variant đã khóa trong script.

## Dữ liệu notebook tự tìm

| Đầu vào | Cách tìm và tái sử dụng |
|---|---|
| P0–P5/grid/model gốc | Repo `results/`, Drive `controller-score-work`, `controller-score-results`, `controller-v4`, full ZIP hoặc `ARTIFACT_PATH` cuối cùng; kiểm hash/schema/provenance |
| A–E models | Tự tìm `MyDrive/controller-diagnostics-AE/`, ZIP/directory trong thư mục kết quả, hoặc `DIAGNOSTICS_PATH`; chỉ reuse khi data hashes, fit/cal/eval IDs và capacity trùng |
| Ảnh đúng cohort | `controller-score-work/inputs`, `restored-bundle`, `controller-v5-inputs`; kiểm `input-hashes.json` |
| Portable setup bundle | `controller-score-work/controller_v4_reconstructed_setup_bundle.zip`; chỉ lấy image/manifest được khai báo hash |
| Training archive nếu mất ảnh | Tái sử dụng raw ZIP cũ, `TRAINING_ZIP` nếu có; nếu cần tải lại, dùng training Drive ID đã có trong workflow v4. Kiểm archive SHA và giữ nguyên 200 calibration/50 evaluation pair; không chọn cohort mới |

Nếu Drive chưa có gì, đặt **một file** `full-1382a8b5cae5-68c68d47bc6c42ad93cc2974fcd59193.zip` vào **MyDrive/controller-score-results/** trước Run all. Đây là dữ liệu thực nghiệm cũ, không phải ZIP ảnh. Nếu có ZIP A–E thì đặt vào **MyDrive/controller-diagnostics-AE/** để giảm train lại. Notebook tự khôi phục/tải ảnh khi cần; lần đầu có thể tải lại archive khoảng 1,95 GB nếu chỉ còn ZIP kết quả.

Nếu link training bị quota, notebook vẫn xuất báo cáo thiếu dữ liệu. Khi đó tải training ZIP đã dùng lên Drive và điền đường dẫn vào `TRAINING_ZIP` trong CONFIG rồi Run all; không cần lập trình hoặc chạy lệnh shell. Không dùng testing.zip thay thế. Nếu đã mất cả full P0–P5 ZIP/grid/model, notebook báo chính xác artifact cần bổ sung; không âm thầm sweep lại 48 cấu hình.

## Tiến trình và resume

Pipeline chạy trong Python subprocess mới sau pip, không cần restart kernel thủ công. W1 counterfactual dùng CPU; W2–W4 train XGBoost CPU; T4 dùng build/test, feature/transition profile và 7 E2E variants. W4 có nhiều fit và log từng fold/seed/cỡ mẫu. Xem `RUN_MATRIX.md` và `run-matrix.json` được tạo trước khi train/đo; thời gian phụ thuộc CPU/runtime, chưa có ước lượng đo thật.

Nếu runtime ngắt, mở lại notebook cùng CONFIG và Run all. Checkpoint từng model, workstream và policy/batch được kiểm hash. Frozen inference cost không được profile lại giữa các batch resume. Khác code, model/data/QoR/split/package/compiler/CUDA/GPU identity sẽ tạo fingerprint khác; không ghép số đo khác môi trường để báo cùng campaign. Nếu chỉ thay GPU/CUDA và CPU identity còn trùng, notebook copy các workstream CPU đã hoàn tất sau khi kiểm hash; W5 vẫn đo lại dưới fingerprint GPU mới.

## ZIP trả lại gồm gì?

- `data-inventory.json`, `feature-schemas.json`, `provenance.json`, `RUN_MATRIX.md`, model/requirements/source hashes và notebook receipt.
- `W1/counterfactual.json`, `pair-level.csv`: controls perfect pre/post/joint, regret/retry và transition sensitivity.
- `W2/`: pre OOF, fixed-config K500 ranking, model checkpoints và schemas.
- `W3/`: gate OOF, margin targets/component risks, selected-route confusion, selected gate và spatial ablations.
- `W4/`: learning curve, nested split manifests, K500 fixed-config metrics, data counts và expansion status.
- `W5/`: native raw CSV, `variants.csv`, `paired-latency.csv`, `attempts.csv`, `qor.csv`, `route-cost.csv`, `profiles.csv`, feature/transition/inference profiles và frozen policies khi đủ dữ liệu/T4.
- PNG/PDF figures, build/test logs, sanitized `notebook-stdout.log`, `results-summary.json`, `README-results.md`, checksum manifest.

Đọc trạng thái từng workstream. TRAINED/OFFLINE_EVALUATED không thay thế GPU_TESTED; PARTIAL/BLOCKED/FAILED vẫn là ZIP có ích để debug. Không gửi riêng ảnh bảng kết quả. Chưa có fresh unseen holdout thì NOT_QUALIFIED cho tuyên bố generalization, dù development benchmark có thể tốt. Không có policy nào tự ghi đè default.
