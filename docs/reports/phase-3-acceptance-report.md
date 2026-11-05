# Báo cáo nghiệm thu từng phần Phase 3

Mốc đối chiếu 05/11 (Thứ 5), năm 2026. Phạm vi gồm CH-00, CH-01, CH-03/A4 và CH-05. Kết luận dưới đây áp dụng cho bằng chứng đã nhận đến mốc này. Deadline chung và phạm vi toàn phase được ghi trong [kế hoạch Phase 3](../plan.md#phase-3-đại-thực-nghiệm-ablation-và-phân-tích-độc-lập).

| Hạng mục | Phần đã nghiệm thu | Phần còn mở |
| --- | --- | --- |
| CH-00 — Baseline | **Đóng.** Receipt đã được xác minh, build `sm_75` và 16/16 CTest pass trên Colab T4. | Kết quả này áp dụng cho snapshot đã kiểm. Kiểm tra Compute Sanitizer cho các campaign GPU được theo dõi riêng. |
| CH-01 — Dữ liệu, protocol và reference | **Đạt tooling, ingestion, freeze protocol và smoke test T4.** Development gồm 200 fit và 50 validation. Test giữ đủ 1.105 cặp canonical. | Chấm QoR toàn bộ development, audit reference và xác nhận Fixed-Best. |
| CH-03/A4 — Same-H blending | **Đạt tooling.** Có runner, bộ kiểm correctness, raw telemetry, bảng và công cụ xuất seam crop. | Campaign A4 với RGB/H đã khóa trên T4. |
| CH-05 — Synthetic multi-resolution | **Đạt tooling.** Có generator, hai track, kiểm soát cấp phát và báo cáo. | Kết quả stress-test GPU ở 720p, 1080p và 4K. |

Lần rà soát báo cáo chạy lại **79 test CPU** cho protocol UDIS-D, A4 và multi-resolution tại revision `33d4b21`. Tất cả đều pass. Các kết quả Colab bên dưới được đối chiếu từ output đã nhận.

## CH-00 — Baseline trên Colab T4

Output notebook lưu cho thấy Tesla T4 với 15.360 MiB VRAM, driver `580.82.07` và CUDA compiler `13.0.88`. Build Release dùng `CMAKE_CUDA_ARCHITECTURES=75`. `generate_receipt --verify-only` trả `receipt_verified`. CTest báo **16/16 pass trong 3,57 giây**, gồm `test_s6_cuda` và `test_admission_control`.

Receipt của lần chạy này ghi snapshot Git Colab `d245f69ecde8aae29d32cc68877203ecd8386bf0`, được tạo từ gói source khi khởi tạo repo trên Colab. Lần rà soát snapshot đã đối chiếu 220 file tracked của checkout `121164c` theo SHA-256. Lần chạy notebook resume có log build lại `stitch_harness` và smoke test riêng. Nhật ký CH-00 ghi Compute Sanitizer có sẵn, còn kết quả chạy sanitizer trên snapshot này chưa có trong bằng chứng đã nhận. Nguồn là output notebook CH-00 đã lưu tại `specs/phase3_ch01_data_protocol.ipynb` ở workspace và [quy trình tạo receipt](../experiments/data-preparation.md#2-ch-00-receipt-và-baseline).

## CH-01 — Ingestion, split và development audit

### Ingestion và kiểm rò rỉ dữ liệu

Phiên resume xác nhận cả hai split vượt qua `udis_ingest --verify-only` sau khi khôi phục bundle. Archive gốc danh nghĩa có 1.106 cặp test. Repo ghi nhận `000001` thiếu ảnh phía `input1`, nên manifest canonical giữ **1.105 cặp đủ hai ảnh**. Freeze và verify đều hoàn tất ở Cell 2. Nguồn là log Colab của [notebook resume](../../notebooks/phase3_ch01_resume.ipynb) và [quyết định về dữ liệu và QoR](../decisions/benchmark-qor-s4-s5.md).

| Thành phần | Số cặp | Cách sử dụng |
| --- | ---: | --- |
| Train gốc đã xác minh | 10.440 | Nguồn duy nhất để lấy development. |
| Train bị cách ly | 6.846 | Thuộc thành phần liên thông chạm test, xấp xỉ 65,6% train gốc. Tất cả thuộc `train-unused`. |
| Train còn đủ điều kiện xét | 3.594 | Còn lại sau cách ly, trước khi chọn development. |
| `train-fit` | 200 | Chọn Fixed-Best bằng QoR và latency theo audit. |
| `train-val` | 50 | Đánh giá lại ứng viên đã chọn trên fit. |
| `train-unused` | 10.190 | Gồm 6.846 cặp cách ly và 3.344 cặp còn lại chưa được chọn. |
| Test canonical | 1.105 | Giữ nguyên ID cho campaign test. |

Graph nối các cặp có chung hash file nguồn hoặc hash RGB đã decode. Một cặp train có thể bị cách ly vì nối bắc cầu tới test qua nhiều cặp trung gian. Tỷ lệ 65,6% mô tả thành phần liên thông, không biểu thị tỷ lệ cặp trùng trực tiếp. Bằng chứng hiện tại chỉ xác lập cách ly theo ảnh trùng chính xác. Cách ly theo cảnh cần metadata cảnh đã xác minh. [Logic split và giới hạn](../../tools/data_prep/udis_split.py)

### Smoke test và diễn giải QoR

Cell 3 chạy hai cặp `000003` và `000015` qua hai cấu hình, mỗi cấu hình có một warmup và hai lượt đo trên mỗi cặp. Kết quả gồm **12 dòng runs** và **4 dòng QoR**, cả hai reference có `reference_status=ok`. So sánh dưới đây dùng ứng viên `S0.75_K2000_R512_Bfeather` và reference `S1.00_K2000_R1024_Bfeather`.

| Cặp | SSIM ứng viên / reference | ΔSSIM | PSNR ứng viên / reference (dB) | ΔPSNR (dB) | Coverage ứng viên |
| --- | --- | ---: | --- | ---: | ---: |
| `000003` | 0,595730 / 0,583459 | +0,012271 | 21,046188 / 20,927600 | +0,118588 | 99,8928% |
| `000015` | 0,542073 / 0,518222 | +0,023851 | 19,507164 / 19,217241 | +0,289923 | 99,6914% |

Cả hai cặp vượt các ngưỡng alignment đã khóa (`ΔSSIM ≥ −0,02`, `ΔPSNR ≥ −0,5 dB`, coverage `≥ 0,95`). Kết quả smoke xác nhận đường chạy từ ảnh thật qua harness và evaluator. Bằng chứng chọn Fixed-Best cần kết quả trên đủ cohort development. [Evaluator QoR](../../tools/qor/evaluate.py)

### Điểm dừng của lần chạy được ghi nhận

Log cuối của Cell 4 cho thấy `train-fit-runs.csv` đã xuất và evaluator bắt đầu chấm QoR fit. Grid có 48 cấu hình Feather, với hai warmup và mười lượt đo mỗi cặp/cấu hình. Tổng kế hoạch là **120.000 lượt đo và 24.000 warmup**, trong đó fit có 96.000 lượt đo và validation có 24.000 lượt đo. Đây là số kỳ vọng từ manifest và grid. Kiểm completeness phải đối chiếu CSV thực tế.

Evaluator gom các lượt đo theo cặp/cấu hình và ghi CSV QoR sau khi chấm xong. Số dòng QoR kỳ vọng là 9.600 cho fit và 2.400 cho validation. Dòng log chờ chấm chưa cho biết số tổ hợp đã xử lý. Chưa nhận được bốn CSV hoàn chỉnh và `development-audit.json` cho lần chạy này. [Thứ tự chạy development](../../notebooks/phase3_ch01_resume.ipynb) · [Cách evaluator ghi kết quả](../../tools/qor/evaluate.py)

Để đóng reference và Fixed-Best, cần gói `p3_evidence/<run-id>/` gồm receipt phiên chạy, `protocol/`, bốn file `train-fit-{runs,qor}.csv` và `train-val-{runs,qor}.csv`, cùng `development-audit.json`. Chạy audit `--verify-only` với input gốc và đối chiếu `complete=true`, `reference_ready_for_review=true`, `selected_fixed_best` có cấu hình và `validation_of_fit_selection` báo kết quả trên cùng cấu hình. Review riêng ledger reference và các trường hợp `missing`, `reference_failure` hoặc `reference_unreliable`. Cờ sẵn sàng chỉ xác nhận các kiểm tra trong protocol, còn độ chính xác homography tuyệt đối cần bằng chứng hình học riêng. [Logic audit development](../../tools/data_prep/audit_development.py)

## CH-03/A4 — Same-H blending

Tooling đã bàn giao trong `4650791`, `b62704c` và `2d7c8e3`. Runner khóa ảnh RGB, H gốc B→A, canvas và thứ tự mode trước khi render Feather, Multiband 3 và Multiband 5. Bộ kiểm đối chiếu union mask theo pixel và RGB với sai số tối đa 2 mức. Overlap count của native runner được đối chiếu với geometry oracle. Số byte workspace lấy từ API và allocation được ghi riêng. [Runbook A4](../runbooks/ch03_same_h_colab.md)

CTest S6 trên T4 là bằng chứng correctness ở phạm vi fixture. Lần thử A4 trong notebook CH-00/CH-01 dừng ở CLI do thiếu `--output`. Notebook resume chỉ chạy development sweep với Feather. Chưa có campaign A4 Phase 3 trong bằng chứng đã nhận.

Để nghiệm thu thực nghiệm, chuẩn bị manifest `s6-same-h-v1` với cohort và H đã khóa, chạy ba mode trên T4 và xác minh campaign bằng `--verify-only`. Gói kết quả cần `registration.json`, `report.json`, `a4_same_h_metrics.csv`, log/timing từng cặp, ảnh và mask phục vụ kiểm lại, bảng LaTeX và hai bộ seam crop cùng tọa độ. Báo latency và workspace theo B. SSIM/PSNR alignment phải bất biến với cùng RGB/H. Nhận xét seam hoặc ghosting dựa trên crop và giữ phạm vi định tính. [Đặc tả A4](../experiments/ablation-studies.md#24-ablation-a4-same-homography-blending-trade-off-trong-s6)

## CH-05 — Stress-test 720p, 1080p và 4K

Tooling đã bàn giao trong `49eb731`, `ac3ba28`, `e859e78` và `eda297e`. Bộ kiểm lưu cho `eda297e` ghi **425 test Python pass**, **9 CTest host-only pass** và `stress_multires_cuda` cùng `stitch_harness` compile/link `sm_75`. Kiểm chứng local dùng fixture nhỏ, mock và kiểm geometry bằng số học. [Runbook CH-05](../runbooks/ch05_multires_colab.md)

| Track | H dùng để render | Bằng chứng cần từ Colab |
| --- | --- | --- |
| `s6_only` | `H_gt` cố định cho từng cặp | Latency S6, kích thước input/canvas, workspace, correctness và phạm vi kiểm full golden hoặc ROI. |
| `end_to_end` | H ước lượng từ S5 | Latency từng stage và toàn pipeline, sai số corner reprojection so với `H_gt`, tình trạng lỗi và workspace. |

Campaign chuẩn có 18 cặp synthetic ở ba độ phân giải, ba mode B và **648 dòng ledger kỳ vọng mỗi track**, gồm cả warmup và lượt lỗi. Khi báo cáo, giữ mẫu số đầy đủ cho OOM, timeout và canvas overflow. Latency dùng các lượt đủ điều kiện đo. Workspace S6, tổng workspace capacity và declared device allocation có scope riêng. ROI xác minh mask và màu vùng một nguồn cho mọi B, đồng thời kiểm màu vùng overlap cho Feather. Màu overlap của Multiband cần full golden để được kiểm trên toàn miền. Chưa có artifact stress-test GPU hoặc sanitizer ở các độ phân giải chuẩn trong bằng chứng đã nhận. [Tiêu chí nghiệm thu và artifact](../runbooks/ch05_multires_colab.md#10-báo-cáo-và-nghiệm-thu)

CH-03 và CH-05 có thể chạy song song khi đã chuẩn bị input theo từng runbook. CH-05 dùng ảnh synthetic và có thể chạy độc lập với development audit UDIS-D. Khi cập nhật báo cáo, gắn từng kết luận với receipt, raw ledger và output `--verify-only` của chính campaign đó.
