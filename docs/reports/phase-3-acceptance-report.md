# Báo cáo nghiệm thu từng phần Phase 3

Mốc đối chiếu 06/11 (Thứ 6), năm 2026. Phạm vi gồm CH-00, CH-01, CH-03/A4 và CH-05. Bản cập nhật này dùng kết quả Colab T4 của lượt `run_20261106_022310Z`. Deadline chung và phạm vi toàn phase được ghi trong [kế hoạch Phase 3](../plan.md#phase-3-đại-thực-nghiệm-ablation-và-phân-tích-độc-lập).

| Hạng mục | Phần đã nghiệm thu | Phần còn mở |
| --- | --- | --- |
| CH-00 — Baseline | **Đóng.** Receipt đã được xác minh, build `sm_75` và 16/16 CTest pass trên Colab T4. | Kết quả này áp dụng cho snapshot đã kiểm. Kiểm tra Compute Sanitizer cho các campaign GPU được theo dõi riêng. |
| CH-01 — Dữ liệu, protocol và reference | **Đạt tooling, ingestion, freeze protocol, smoke test và thu đủ số liệu development T4.** Có 120.000 lượt đo, 24.000 warmup và 12.000 dòng QoR. | **Gate reference chưa đạt.** Reference dùng được trên 238/250 cặp. `selected_fixed_best=null`. |
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

Graph nối các cặp có chung hash file nguồn hoặc hash RGB đã decode. Tính lại từ hai data manifest cho thấy 4.754 cặp train có ít nhất một ảnh trùng trực tiếp với test và 2.092 cặp bị cách ly qua liên kết bắc cầu. Split tái tạo khớp `split_manifest.json`. Mỗi thành phần liên thông xuất hiện trong tối đa một split đang dùng. `scene_disjointness=not_certified` vì phép kiểm hash chỉ phát hiện ảnh trùng chính xác. [Logic split và giới hạn](../../tools/data_prep/udis_split.py)

### Smoke test và diễn giải QoR

Cell 3 chạy hai cặp `000003` và `000015` qua hai cấu hình, mỗi cấu hình có một warmup và hai lượt đo trên mỗi cặp. Kết quả gồm **12 dòng runs** và **4 dòng QoR**, cả hai reference có `reference_status=ok`. So sánh dưới đây dùng ứng viên `S0.75_K2000_R512_Bfeather` và reference `S1.00_K2000_R1024_Bfeather`.

| Cặp | SSIM ứng viên / reference | ΔSSIM | PSNR ứng viên / reference (dB) | ΔPSNR (dB) | Coverage ứng viên |
| --- | --- | ---: | --- | ---: | ---: |
| `000003` | 0,595730 / 0,583459 | +0,012271 | 21,046188 / 20,927600 | +0,118588 | 99,8928% |
| `000015` | 0,542073 / 0,518222 | +0,023851 | 19,507164 / 19,217241 | +0,289923 | 99,6914% |

Cả hai cặp vượt các ngưỡng alignment đã khóa (`ΔSSIM ≥ −0,02`, `ΔPSNR ≥ −0,5 dB`, coverage `≥ 0,95`). Kết quả smoke xác nhận đường chạy từ ảnh thật qua harness và evaluator. Bằng chứng chọn Fixed-Best cần kết quả trên đủ cohort development. [Evaluator QoR](../../tools/qor/evaluate.py)

### Development sweep và đối chiếu artifact

Cell 4 hoàn tất cả hai sweep và xuất đủ bốn CSV. Cell 5 tạo audit, chạy `--verify-only` trên Colab và lưu gói vào `MyDrive/p3_evidence/run_20261106_022310Z/`. Sau khi sao lưu, Cell 5 dừng ở assertion vì audit trả kết quả chưa đạt. Output có trong bản notebook đã chạy tại `specs/phase3_ch01_resume.ipynb`. Gói xuất được lưu ở `specs/p3_evidence/` trong workspace. [Notebook nguồn](../../notebooks/phase3_ch01_resume.ipynb)

| Split | Cặp | Cấu hình | Lượt đo | Warmup | Dòng QoR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train-fit` | 200 | 48 | 96.000 | 19.200 | 9.600 |
| `train-val` | 50 | 48 | 24.000 | 4.800 | 2.400 |
| Tổng | 250 | 48 | 120.000 | 24.000 | 12.000 |

Kiểm tra CSV xác nhận đủ 144.000 lượt chạy harness gồm cả warmup và lượt lỗi, đủ khóa bản ghi duy nhất và đủ 12.000 khóa cặp/cấu hình trong CSV QoR. Tất cả runs có nhãn `UDIS-D`, đúng split và seed harness `0`. Hai audit con tính lại từ CSV khớp hoàn toàn JSON gốc. Hash của năm đầu vào audit, sáu output protocol, hai data manifest và hai source receipt đều khớp metadata. Gói xuất có manifest nhưng thiếu ảnh PPM, nên lần đối chiếu này tái tạo split và audit từ metadata/CSV. Việc kiểm pixel và `--verify-only` đầy đủ dựa trên output Colab. [Audit sweep](../../tools/controller/audit_sweep.py)

Receipt của lượt resume ghi snapshot Colab `c6eb9a1376e536bf6d01ce2b9433d2a351302b16`. Inventory đối chiếu khớp 222 file tracked của `33d4b21` và chứa thêm bảy file pilot Dev25. CSV ghi `git_rev=HEAD`, nên xác định source của phiên bằng receipt và notebook output. Gói audit có SHA-256 `6acaa640a2a43d0eb4a28c978d7319e99fcedaf88d211b92526035d5f150d7a6` cho `development-audit.json`.

Tổng `t_e2e` trong CSV, tính cả warmup và lượt lỗi, là khoảng 475,73 giây cho fit và 114,49 giây cho validation. Hai số này là tổng thời gian được harness ghi theo lượt. Notebook thiếu phép đo wall-clock tách riêng sweep, chấm QoR và audit, nên mốc 26 phút được ghi trước đó không đủ để xác nhận thời lượng từng bước. Evaluator đọc RGB gốc và H từ runs rồi tính alignment trên CPU, với một dòng kết quả mỗi cặp/cấu hình. [Evaluator QoR](../../tools/qor/evaluate.py)

### Reference và nguyên nhân gate chưa đạt

Reference `S1.00_K2000_R1024_Bfeather` dùng được trên **192/200 cặp fit (96,0%)** và **46/50 cặp validation (92,0%)**, tổng cộng **238/250 (95,2%)**. Mỗi cặp dưới đây lặp lại cùng mã lỗi ở cả 10 lượt đo reference. Các mã mô tả lỗi quan sát được trong pipeline. Nguyên nhân cụ thể theo cảnh cần xem lại RGB, H và geometry của từng cặp.

| Mã lỗi ở reference | `train-fit` | `train-val` | Diễn giải |
| --- | --- | --- | --- |
| `ERR_CANVAS_OVERFLOW` ở S6 | `006399`, `009444` | | Canvas của H reference vượt giới hạn runner. |
| `DEGENERATE` ở S5 | `009726`, `009729`, `010327`, `010355` | `007175`, `009964`, `010118` | Estimator báo suy biến ở cấu hình reference. |
| `NOT_ENOUGH_POINTS` ở S5 | `009955`, `009959` | `009957` | S5 báo thiếu điểm cho bước ước lượng. |

Trong 12 cặp này, `006399`, `009444` và `007175` vẫn có lần lượt 33, 28 và 24 cấu hình khác chạy `OK`. Chín cặp còn lại không có cấu hình nào ghi `OK` trong grid đã đo. `OK` xác nhận lượt chạy pipeline thành công. Với ba cặp trên, QoR tương đối vẫn thiếu mốc đối chiếu vì reference thất bại. Kết quả cho thấy cần phân biệt lỗi của cấu hình reference với khả năng thực thi của cả grid.

Hai audit con có đủ lượt đo và không có cấu hình thiếu latency. Gate `complete` còn yêu cầu mỗi lượt được phân loại `PASS` hoặc `FAIL`. Khi reference lỗi, 8 cặp fit và 4 cặp validation tạo lần lượt **3.840** và **1.920** lượt đo có nhãn `REFERENCE_FAILED` trên 48 cấu hình. Vì vậy `complete=false`, `reference_ready_for_review=false`, `selected_fixed_best=null` và `validation_of_fit_selection=null`. Thông báo assertion về đo lường trong notebook cần đọc cùng điều kiện gate này. [Logic audit development](../../tools/data_prep/audit_development.py)

### Xếp hạng quan sát được trên development

Audit con xếp hạng theo số cặp có đủ 10 lượt đạt QoR, rồi đến median latency qua các median theo cặp. Bảng giữ mẫu số toàn cohort, gồm các cặp reference lỗi. Thứ hạng validation dùng để mô tả kết quả đã đo, còn lựa chọn Fixed-Best chính thức cần qua gate trên fit và review validation.

| Cấu hình | Hạng fit / val | QoR fit | Median fit (ms) | QoR val | Median val (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `S1.00_K2000_R1024_Bfeather` | 1 / 1 | 192/200 | 7,51 | 46/50 | 7,27 |
| `S1.00_K2000_R512_Bfeather` | 2 / 2 | 164/200 | 6,70 | 39/50 | 6,42 |
| `S1.00_K1000_R1024_Bfeather` | 3 / 5 | 148/200 | 5,79 | 33/50 | 5,51 |
| `S1.00_K2000_R256_Bfeather` | 4 / 3 | 148/200 | 6,24 | 35/50 | 6,10 |
| `S1.00_K2000_R128_Bfeather` | 5 / 4 | 142/200 | 6,09 | 34/50 | 5,90 |
| `S1.00_K1000_R512_Bfeather` | 6 / 6 | 140/200 | 4,96 | 30/50 | 4,68 |
| `S0.75_K2000_R512_Bfeather` | 8 / 12 | 127/200 | 5,33 | 26/50 | 4,61 |

- Hai cấu hình đầu giữ cùng thứ hạng trên fit và validation. Chỉ **9/48** cấu hình giữ đúng vị trí giữa hai bảng. Cấu hình `S0.75_K2000_R512_Bfeather` qua smoke nhưng chỉ đạt 63,5% trên fit và 52,0% trên validation.
- Ở cùng `S=1.00` và `K=2000`, giảm `R` từ 1024 xuống 512 giảm median khoảng **10,8% trên fit** và **11,7% trên validation**. Tỷ lệ đạt QoR giảm lần lượt **14 điểm phần trăm** và **14 điểm phần trăm**. Đây là so sánh giữa hai cấu hình fixed trên development.
- Mỗi cấu hình trong nhóm `S=0.25` đạt **2–3/200 cặp fit** và **0/50 cặp validation**. Kết quả mô tả grid, cohort và policy đã khóa.
- Oracle của audit có ít nhất một cấu hình đạt trên **192/200 fit** và **46/50 validation**. Reference tự so với cùng H trên những cặp có nhãn hợp lệ, nên các số này phản ánh phạm vi chấm QoR hiện tại. Chúng chưa đánh giá độ chính xác H với ground truth hoặc chứng minh Oracle speedup trên đủ cohort.

Giữ gói dữ liệu, hai split và 12 cặp lỗi khi review reference. Trước khi khóa Fixed-Best hoặc mở campaign test chính, cần quyết định cách xử lý reference failure, lưu quyết định và kiểm lại trên development. Nếu chốt theo gate hiện tại, cần `complete=true`, `reference_ready_for_review=true`, `selected_fixed_best` có giá trị và `validation_of_fit_selection` cho cùng cấu hình. Gói đã đo đủ là căn cứ cho bước review này.

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

## Bước tiếp theo trên Colab T4

1. **Notebook CH-03/A4.** Chuẩn bị cohort UDIS-D và nguồn H gốc B→A cho từng cặp. Kiểm và khóa manifest cùng registration trước khi render. Chạy smoke/correctness, rồi campaign ba mode có lưu ảnh, verify, xuất bảng và crop. Các file `pairs_*.json` từ CH-01 cung cấp đường dẫn ảnh. Manifest A4 cần thêm H và provenance của H. Nếu dùng cohort test, giữ quy tắc chọn cặp và H độc lập với kết quả blender và mọi quyết định hiệu chỉnh benchmark chính. [Runbook A4](../runbooks/ch03_same_h_colab.md)
2. **Notebook CH-05.** Sinh và verify manifest synthetic, build native `sm_75`, chạy fixture nhỏ để kiểm môi trường, rồi chạy `s6_only` và `end_to_end` vào hai thư mục campaign riêng. Verify và xuất báo cáo cho từng track. Bước này độc lập với dữ liệu UDIS-D và gate reference CH-01. [Runbook CH-05](../runbooks/ch05_multires_colab.md)

Hai notebook dùng receipt và thư mục lưu riêng theo lượt chạy. Khi cập nhật nghiệm thu, đối chiếu raw ledger và output `--verify-only` của từng campaign. Bản cập nhật này ghi nhận kế hoạch mở hai campaign, còn kết quả nghiệm thu phụ thuộc vào lượt chạy T4 thực tế.
