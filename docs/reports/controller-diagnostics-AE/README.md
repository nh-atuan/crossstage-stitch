# Chạy chẩn đoán Controller A–E

Notebook: [controller_model_diagnostics_AE_colab.ipynb](../../../notebooks/controller_model_diagnostics_AE_colab.ipynb). Công việc thuộc Phase 3, deadline chung 07/11 (Thứ 7), năm 2026; Đế phụ trách Controller.

## Chuẩn bị một lần

1. Cần push commit bàn giao lên `de/optimize-controller` trước khi chạy. Repo private nên tải file notebook về máy, mở [Colab](https://colab.research.google.com/), chọn **File → Upload notebook** rồi tải `.ipynb` lên. Không cần mở notebook qua GitHub private.
2. Trong GitHub **Settings → Developer settings → Personal access tokens → Fine-grained tokens**, tạo token cho resource owner `nh-atuan`, chỉ repository `crossstage-stitch`, quyền **Repository permissions → Contents: Read-only**. Tài khoản phải được cấp quyền đọc repo; nếu tổ chức yêu cầu approval thì chờ approval. Không cần quyền ghi.
3. Trong Colab, mở **Secrets** (biểu tượng chìa khóa), thêm tên `GITHUB_TOKEN`, đặt token vào giá trị và bật **Notebook access**. Không dán token vào cell, không gửi token cùng kết quả.
4. Giữ thư mục Drive cũ nếu còn. Nếu đã mất dữ liệu làm việc, chỉ upload file `full-1382a8b5cae5-68c68d47bc6c42ad93cc2974fcd59193.zip` vào **MyDrive/controller-score-results/**. Giữ tên file. ZIP P0–P5 hiện có chứa calibration grid và model theo fold; không cần upload lại UDIS-D hoặc ảnh 1,95 GB.
5. Chọn **Runtime → Change runtime type → Python 3, Hardware accelerator: None/CPU**. Notebook này không cần T4. Chọn **Runtime → Run all**, chấp nhận prompt mount Google Drive và cấp quyền Secrets khi được hỏi.

Repo, branch và Drive root đã được điền trong cell CONFIG. Không phải chọn A–E, threshold hoặc tham số model. Nếu dùng tài khoản/repo/thư mục khác, chỉ chỉnh `REPO`, `REF`, `DRIVE_ROOT`, `OUTPUT_ROOT`. `ARTIFACT_PATH` là tùy chọn cuối cùng khi tự tìm không thấy, ví dụ `/content/drive/MyDrive/full-1382a8b5cae5-68c68d47bc6c42ad93cc2974fcd59193.zip`. `REF` phải chứa code mới đã push; full commit SHA giúp tái lập chính xác. Notebook in và lưu full SHA sau checkout.

Mỗi lần Run all gồm audit → CPU tests → A → B → C → D → E → tổng hợp → ZIP. E train các model nghiên cứu trên CPU, có log từng fold/seed/cỡ mẫu; có thể là phần lâu nhất. Checkpoint lưu trên Drive sau từng fit và từng thí nghiệm. Khi Colab ngắt, mở lại cùng notebook, cùng CONFIG và **Run all**. Code, package hoặc input khác sẽ tạo run ID khác; cùng input và môi trường sẽ tái sử dụng checkpoint đã kiểm hash.

## Notebook kiểm tra và xuất gì?

| Phần | Nội dung chính |
|---|---|
| Audit | `data_inventory.json`, `data-inventory.csv`, `fixed-grid-outcomes.csv`: pair/group/cohort, config, H/status, QoR, post invalid, latency median của 10 repeat; hash input/model/source và kiểm tra split |
| A | `pre_diagnostics.csv/json`, `pre-predictions.csv`, `ranking-per-pair.csv`: ROC-AUC/AP theo K/S, Brier, threshold, cheap-feasible/top-k recall, regret và Oracle bị filter hay score bỏ qua |
| B | Reliability PNG/PDF, bins, Brier/log loss/ECE trước/sau probability calibration, confusion ở 0.65/0.85; thống kê theo K/S/component và route đã ghé thăm |
| C | `perfect-pre.json`, bảng từng pair, biểu đồ PNG/PDF: current/perfect probability/perfect latency, PASS/FAIL, retry, config distribution, paired latency scenario và regret |
| D | `perfect-post.json`, bảng calibration và route đã đo, `four-case-studies.csv`: false accept tránh được, false reject cứu được; nhánh thay đổi chưa đo được ghi UNKNOWN |
| E | Learning curve CSV/JSON/PNG/PDF; từng model/split/checkpoint trong `E/fits/`; support K500 của booster-fit và probability calibration |
| Tổng hợp | `experiment-status.csv`, `conclusions.md/json`, `protocol.json`, `environment.json`, notebook execution receipt, CPU test log, `artifact-hashes.json` |

Mặc định kết quả và ZIP được lưu tại **MyDrive/controller-diagnostics-AE/controller-diagnostics-AE-<run_id>** và **controller-diagnostics-AE-<run_id>.zip**. Cell cuối tự gọi `google.colab.files.download` nếu ZIP không quá 250 MB; nếu browser chặn tải, mở Drive và tải ZIP tại đường dẫn đã in. Gửi **nguyên ZIP** lại để phân tích, kể cả khi trạng thái PARTIAL/BLOCKED/FAILED; không chỉ gửi ảnh bảng kết quả.

## Cách đọc và xử lý thiếu dữ liệu

- A–E dùng calibration grid 200 pair; 195 Reference hợp lệ vào phép thử, 5 Reference failure giữ trong audit. OOF và final-model in-sample được ghi riêng. 50 pair evaluation P0–P5 đã xem chỉ dùng mô tả route/case study, không fit hay chọn threshold. Không có full evaluation grid nên không suy Oracle trên 50 pair đó.
- A/B dùng lại outer-fold weights đã lưu nếu hợp lệ. Nếu thiếu fold nhưng grid còn đủ, notebook train grouped CV mới, ghi rõ là model nghiên cứu. E luôn dùng 64 cây depth 2; các cỡ 40/80/120/160/toàn bộ được cap theo outer-training pool, nên với 195 component và 5 fold, mức lớn nhất là 156, không phải train lên outer evaluation.
- CI bootstrap theo component ảnh, giữ mọi config cùng component; seed sampling không được xem như pair độc lập. PR-AUC trong bảng là noninterpolated average precision. Mỗi bin ECE rộng 0.1; bin ít group có nhãn support hạn chế.
- C/D là counterfactual offline. Chi phí stage-only và full-recompute là **ESTIMATED scenarios**, không phải speedup GPU hoặc cận timing đã chứng minh. R-only giữ S1–S4, đổi K giữ S1, đổi S chạy lại từ S1; không mặc định mọi retry đều rẻ. Không dùng thời gian collector toàn bộ feature làm chi phí photo runtime.
- Nếu thiếu artifact, mỗi thí nghiệm báo danh sách file và vẫn xuất ZIP. Bộ đầy đủ cần `baseline-policy.json`, `split.json`, `quality-policy.json`, `calibration.json`, `setup-evidence/calibration-grid/{runs.csv,features.csv,qor.csv,audit.json,configs.json,receipt.json}`. Muốn dùng đúng OOF cũ cần `setup-evidence/fit-000/fold-*.json`. D trên routes cần `campaign/qor.csv`, `P0.csv` đến `P5.csv`, các file `P*-attempts.csv`, `P*-score-diagnostics.csv` và `policies/P*.json`. Cách đơn giản nhất là upload nguyên ZIP P0–P5, không tự ghép file từ campaign khác.
- Notebook không download ảnh, chạy lại sweep, build CUDA, thay model production hay đổi gate mặc định. Nếu mất cả ZIP lẫn grid gốc, gửi ZIP báo thiếu dữ liệu lại để quyết định một workflow bổ sung riêng.

## Kiểm chứng và nguồn

Đã kiểm tra local: audit và A–D trên ZIP thật; A–E, export/resume và các cell phân tích của notebook trên fixture CPU. Notebook được kiểm schema, compile từng cell và giữ trống output. Chưa chạy trên Colab, chưa chạy learning curve đầy đủ trên artifact thật, chưa có kết quả chẩn đoán cuối cùng để kết luận nguyên nhân.

Định nghĩa metric tham khảo [Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) và [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html); capacity giữ theo [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html). Đọc ngày 07/11 (Thứ 7), năm 2026. Quyết định và mapping source nằm trong [ghi chép kỹ thuật](../../decisions/p3-controller-diagnostics-ae.md).
