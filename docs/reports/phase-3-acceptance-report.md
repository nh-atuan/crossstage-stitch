# Báo cáo nghiệm thu từng phần Phase 3

Mốc đối chiếu 07/11 (Thứ 7), năm 2026. Bằng chứng Colab T4 gồm lượt development `run_20261106_022310Z`, A4/CH-05 `run_20261106_043655Z_ch03_ch05_gbfq3y9z`, CH-01E `run_20261106_081406Z_ch01e_9x1qcuvw`, CH-04a `run_20261106_134702Z_ch04a_gbqf76f6` và CH-02 `run_20261106_164958Z_ch02_inu6unq8`. Deadline chung và phạm vi toàn phase được ghi trong [kế hoạch Phase 3](../plan.md#phase-3-đại-thực-nghiệm-ablation-và-phân-tích-độc-lập).

| Hạng mục | Phần đã nghiệm thu | Phần còn mở |
| --- | --- | --- |
| CH-00 — Baseline | **Đóng.** Receipt đã được xác minh, build `sm_75` và 16/16 CTest pass trên Colab T4. | Kết quả này áp dụng cho snapshot đã kiểm. Kiểm tra Compute Sanitizer cho các campaign GPU được theo dõi riêng. |
| CH-01 — Dữ liệu, protocol và reference | **Đóng trên development theo ngoại lệ đã duyệt.** Reference dùng được trên 238/250 cặp. Receipt riêng khóa Fixed-Best từ train-fit và giữ đủ mẫu số. | Campaign test vẫn phải ghi trạng thái reference thực tế của từng cặp. |
| CH-01E — Hạ tầng adaptive | **Đóng nghiệm thu trên T4 cho revision đã đo.** Build `sm_75`, 4/4 CTest, memcheck fixture 0 lỗi và hai lượt adaptive smoke trên ảnh thật cùng seed. | Smoke bao phủ đường `ACCEPT` ngay attempt đầu với `legacy-v1`. Đánh giá policy đã hiệu chỉnh và QoR toàn test set thuộc các campaign riêng. |
| CH-02 — Benchmark chính và QoR | **Hoàn thành.** Đủ 1.106 disposition mỗi mode, 48 cấu hình Oracle và báo cáo tái tạo từ CSV. | Adaptive `legacy-v1` đạt 394/1.106 QoR PASS. Speedup 2,161× chỉ áp dụng trên giao 394 cặp cùng PASS. |
| CH-03/A4 — Same-H blending | **Đóng campaign T4 trên cohort development đã khóa.** 10/10 cặp đạt ở cả ba blender, có bảng và ảnh truy về artifact. | Phạm vi kết luận giới hạn ở 10 cặp overlap cao. |
| CH-04a — Chuẩn bị A5 | **Đã hoàn tất fit và đo validation trên T4, với kết quả `NO_GO` đã xác minh.** 40/40 ứng viên có đủ ledger. | 0/20 policy đạt sàn Fixed-Best, nên CH-04b chưa đủ điều kiện mở. |
| CH-05 — Synthetic multi-resolution | **Đạt `s6_only` trên T4.** 54/54 case từ 720p đến 4K. `end_to_end` đã đo đủ và có kết quả âm đã xác minh. | Chưa thiết lập được đường scaling end-to-end của các lượt đăng ký ảnh hợp lệ. |
| CH-06 — Đánh giá Đa Kiến trúc | **Hoàn thành trên L4 (`sm_89`) và A100 (`sm_80`).** Đủ 54/54 case mỗi GPU, workspace khớp bitwise với T4, bảng LaTeX và speedup phân tích đầy đủ. | Đóng task CH-06 của Tuấn. |
| CH-07 — Hình và bảng tái lập | **Hoàn thành trên bằng chứng T4 đã khóa và cập nhật CH-06.** 4 cặp PDF/SVG, 6 bảng Booktabs và bảng Cross-architecture tái tạo khớp hash. | Bàn giao đủ. |
| CH-08 — Freeze và bàn giao manuscript | **Đã bàn giao phần T4 của Chương.** Có manifest, claim–evidence map và bản OASIcs ẩn danh. | Freeze bản nộp chung còn review kết quả các owner, CH-06 và quyết định nhóm về A5 sau `NO_GO`. |

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

Chiếu lại bốn góc B bằng H trong CSV và hợp với khung A cho thấy canvas xấp xỉ `8487×1369` ở `006399` và `9758×6497` ở `009444`. Cả hai vượt chiều rộng cấp phát 4096 của harness lượt này. Cặp thứ hai cũng vượt giới hạn diện tích 16.777.216 pixel của geometry policy. Đối chiếu này dùng H đã ghi và kích thước ảnh từ manifest.

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

**Kết quả audit gốc.** Audit giữ reference, ngưỡng QoR, hai split và cả 12 cặp `REFERENCE_FAILED`. Kết quả `OK` ở ba cặp có cấu hình thay thế dùng để chẩn đoán. Thay H theo cấu hình hoặc theo từng cặp sẽ đổi miền so sánh đã khóa. Loại cặp lỗi cũng sẽ đổi mẫu số sau khi đã xem kết quả. Audit gốc kết thúc với disposition reference chưa đạt và Fixed-Best để trống.

Gói export đã được replay bằng lệnh sau. `verification_scope=exported_metadata_and_csv` và exit 0 xác nhận audit lưu trữ tái tạo được. Giá trị `selected_fixed_best=null` là kết quả của gate nghiên cứu. Kiểm chứng đầy đủ trên Colab vẫn dùng bundle có ảnh và bỏ cờ `--exported-data-root`.

```bash
python -m tools.data_prep.audit_development \
  --protocol specs/p3_evidence/protocol \
  --output specs/p3_evidence/development-audit.json \
  --exported-data-root specs/p3_evidence/data --verify-only --verbose
```

### Khóa Fixed-Best theo disposition được duyệt

Ngày 06/11 (Thứ 6), Chương phê duyệt ngoại lệ giới hạn ở đúng 12 lỗi trong audit trên. Công cụ kiểm SHA-256 của audit và mã lỗi qua đủ 10 lượt của từng reference, giữ các cặp trong mẫu số dưới nhãn `REFERENCE_FAILED` và yêu cầu tất cả kết quả khác có disposition `PASS` hoặc `FAIL`. Audit gốc giữ `selected_fixed_best=null` để lần replay cũ vẫn tái lập. Quyết định mới được ghi riêng trong `development-lock.json` và không áp dụng ngoại lệ này cho test. [Ledger phê duyệt](../../tools/data_prep/reference-exceptions-p3.json)

Receipt đã sinh và chạy lại `--verify-only` từ CSV thật. Fixed-Best là **`S1.00_K2000_R1024_Bfeather`**, đạt **192/200** trên fit với median **7,509 ms**. Cùng cấu hình đạt **46/50** trên validation, median **7,270 ms**. Fixed-High và Fixed-Best dùng cùng cấu hình trong lượt này. SHA-256 của receipt là `f204d575f19535c63392c758c9c6b23925a29182ce18e98d90ddbca13260e95b`. [Cách tái tạo receipt](../runbooks/ch04a_a5_calibration.md#1-replay-và-khóa-ch-01)

## CH-01E — Adaptive smoke trên Colab T4

### Nguồn, build và kiểm thử

Lượt `run_20261106_081406Z_ch01e_9x1qcuvw` chạy trên Tesla T4 với compute capability 7.5, VRAM 15.360 MiB, driver `580.82.07` và `nvcc 13.0.88`. Snapshot Colab là `2832a01429b58467b0be776e15e9dc9e130f4bdb`. SHA-256 của source ZIP là `1f061b25b39ae5040293908c270caebd52f5f30b4f5bac68576642c4ff6fbaa3`. ZIP khớp 245/245 hash trong receipt. Mã các cell đã chạy khớp notebook nguồn tại `29938fc`. [Notebook nguồn](../../notebooks/phase3_ch01e_adaptive_verify.ipynb)

Gói xuất được lưu tại `specs/run_20261106_081406Z_ch01e_9x1qcuvw/` trong workspace. Kiểm lại đủ **39/39 file** theo SHA-256 và sáu marker bước đều khớp với `export_verified.json`, có SHA-256 `a4b1c138b6489f5bf48e1b71e94bb22a851745f75974cc3e800ee890bdbc3964`. Dữ liệu thô và output notebook lưu riêng khỏi Git.

Build Release `sm_75` hoàn tất và `compile_commands.json` xác nhận target `sm_75` ở 18/18 mục CUDA. CTest chạy đủ `test_adaptive_upload`, `test_policy_loader`, `test_attempt_trace` và `test_admission_control`, **4/4 pass, 0 skip**, trong 0,45 giây. Compute Sanitizer memcheck trên `test_adaptive_upload` báo **0 lỗi**, exit `0`. Fixture upload kiểm probe nhận ảnh mới, tính nhất quán của trace và việc reset trace khi lỗi xảy ra trước attempt. Phạm vi memcheck là fixture này. [Runbook](../runbooks/ch01e_adaptive_colab.md)

### Replay CH-01 và kiểm input smoke

Cell 2 replay thành công từ metadata và CSV của audit đã khóa, với `verification_scope=exported_metadata_and_csv`. Kết quả vẫn là **192/200 cặp fit** và **46/50 cặp validation** có reference dùng được. Marker replay `READY` xác nhận audit tái lập được. Gate nghiên cứu giữ `complete=false`, `reference_ready_for_review=false` và `selected_fixed_best=null`.

Cell 4 lấy bốn PPM của `000003` và `000015` từ `p3_protocol_bundle.zip`. Cả bốn SHA-256 khớp training manifest đã khóa. `source_receipt.json` cũng khớp hash của audit, protocol, manifest `train-fit` và manifest smoke. Hai lượt chạy dùng `--controller-policy legacy-v1 --seed 0 --warmup 2 --repeats 10`.

### Kết quả terminal và attempt trace

Mỗi lượt có **24/24 terminal `OK`** gồm 4 warmup và 20 lượt đo, kèm 24 attempt. Hai cặp chọn cùng `(S=0.5, K=1000, R=128, B=Feather)` và đều `ACCEPT` với `SELF_CHECK_PASSED` tại attempt `0`.

| Cặp `train-fit` | Inliers / matches | Inlier ratio | Inlier coverage | RMS residual (px S1) | Retry |
| --- | ---: | ---: | ---: | ---: | ---: |
| `000003` | 373 / 391 | 95,40% | 0,3895 | 1,077 | 0 |
| `000015` | 412 / 423 | 97,40% | 0,4665 | 1,148 | 0 |

`inlier_coverage` đo diện tích bounding box của inlier phía A trong hệ S1. Đây là tín hiệu self-check của controller. QoR tương đối dùng overlap coverage, PSNR và SSIM từ evaluator riêng. [Định nghĩa feedback S5](../stages/controller-contract.md)

Đối chiếu lại `runs.csv` và `attempts.csv` của hai lượt xác nhận đủ khóa `(pair_id, is_warmup, repeat_idx)`, `attempt_count=1`, `attempt_idx=0`, status và S/K/R/B khớp terminal. Chín hệ số H khớp giữa terminal và attempt. So sánh giữa hai lượt khớp mọi trường CSV sau khi bỏ `run_id` và các cột `t_*`, gồm quyết định, inlier, residual và H. Kết quả này xác nhận tính tất định của trace được ghi trên hai cặp và cùng seed. Đường retry, reject và policy khác cần evidence từ workload riêng. [Logic đối chiếu trace](../../tools/repro/ch01e_trace.py)

### Timing của hai lượt smoke

Mỗi cột tổng hợp 20 lượt đo, gồm 10 lượt mỗi cặp. Median lấy trực tiếp từ CSV của từng lượt. `determinism_verification.json` giữ median của lượt đầu, nên cột chạy lại lấy từ `smoke-rerun/runs.csv`.

| Chỉ số | Lượt đầu (ms) | Lượt chạy lại (ms) |
| --- | ---: | ---: |
| `t_e2e` | 10,437 | 6,303 |
| `t_probe_gpu` | 0,0568 | 0,0517 |
| `t_controller` | 0,000706 | 0,000450 |
| `t_stream_wait` | 10,244 | 6,073 |
| S1 (gồm probe) | 0,168 | 0,160 |
| S2 | 0,184 | 0,111 |
| S3 | 0,254 | 0,149 |
| S4 | 1,147 | 0,663 |
| S5 | 7,029 | 4,202 |
| S6 | 1,596 | 0,919 |

S5 có median lớn nhất trong cả hai lượt. Ở lượt đầu, `000015` chuyển từ khoảng 10,32–11,05 ms ở năm lượt đo đầu xuống 6,44–6,48 ms ở năm lượt cuối. Các trường ngoài timing vẫn khớp giữa hai lượt. Evidence có thông tin thiết bị nhưng thiếu chuỗi đo xung nhịp và nhiệt độ, nên nguyên nhân dao động timing chưa xác định. `t_e2e` bắt đầu khi RGB đã sẵn sàng trên GPU. `t_controller` chỉ đo phần CPU của `begin_pair` và `observe`, còn `t_stream_wait` chồng lấp với thời gian GPU. Các median mô tả smoke, với hai lượt dùng cùng policy và không có baseline đối chứng để tính speedup. [Ranh giới timing](../runbooks/ch01e_adaptive_colab.md#policy-và-đầu-ra)

**Nghiệm thu.** Đóng CH-01E cho hạ tầng adaptive của revision đã đo trên T4. Khi tích hợp revision có thay đổi harness hoặc controller, kết quả smoke cần được xác nhận lại trước campaign chính.

## CH-02 — Benchmark UDIS-D và nghiệm thu QoR

### Thiết lập và đối soát

Campaign `ch02-b010623ef7d5b02f5e2b` chạy trên một phiên Colab Tesla T4 `sm_75`, VRAM 15.360 MiB, driver `580.82.07`, `nvcc 13.0.88` và Python 3.13.16. Snapshot Colab `5fe8ce799a23f2d6817a9cdd5b6376a4285de37c` khớp 268/268 file trong receipt với ZIP nguồn. ZIP có SHA-256 `8acac3d441f58dd6d484f4bd70826c8420b9598f8d969391a5a30d459d33d17d`. Mã các cell khớp notebook tại `c363631`. Binary Release có SHA-256 `a88d82b77108f8a958bf8826756869b1b6251964e0c70e4d846d9860caf15829`. [Notebook nguồn](../../notebooks/phase3_ch02_benchmark.ipynb)

Plan ghi `legacy-v1` với authority Chương, seed `20261106`, Feather và hai warmup cùng mười lượt đo mỗi nhóm. Fixed-Best từ CH-01 trùng Fixed-High ở cấu hình `S1.00_K2000_R1024_Bfeather`. Để chấm QoR, dùng H của Fixed-High trong grid, với ngưỡng giảm SSIM 0,02, giảm PSNR 0,5 dB và coverage tối thiểu 0,95. Đây là alignment tương đối theo reference, nên giá trị PASS khi H trùng reference không xác nhận độ chính xác hình học độc lập. [Runbook CH-02](../runbooks/ch02_udis_benchmark_colab.md)

Gói `specs/run_20261106_164958Z_ch02_inu6unq8/` khớp **10.009/10.009 hash** và sáu marker bước. SHA-256 của `export_verified.json` là `857bb3ecf2c93d69100772d4e1ab1e3d0503e8ea5d422259d145f7983132b865`. Đủ **3.318 disposition** cho 1.106 cặp × 3 mode. Mỗi mode ghi `ERR_LOADER_IO` cho `000001` thiếu ảnh A và có checkpoint `MEASURED` cho 1.105 cặp còn lại. Grid gồm 48 cấu hình.

Đối chiếu CPU xác nhận **663.000 terminal rows**, gồm 110.500 warmup và **552.500 dòng đã đo và chấm QoR**. Phần đã đo gồm 11.050 Fixed-Best, 11.050 Adaptive và 530.400 grid. Mỗi nhóm có đủ mười repeat. Attempt count khớp cả 13.260 terminal Adaptive, tính cả warmup. Có 503.850 H của lượt đo thành công round-trip chính xác qua CSV/JSON và **0 nhóm cảnh báo H lệch quá `1e-12`**. Cache có 482.530 hit trên 501.790 lookup (96,16%), phản ánh việc tái dùng kết quả theo nội dung ảnh/H. `joined.csv`, receipt kiểm join, báo cáo, ledger, CDF, Pareto và bảng LaTeX tái tạo khớp từng file. Điểm ảnh SSIM/PSNR/coverage được tính trong lượt evaluator trên Colab. Rà soát local kiểm join, precision và tổng hợp từ CSV.

### Kết quả toàn mẫu số

Latency trong bảng là median và p95 của pair medians trên các cặp PASS **riêng của từng mode**.

| Mode | Execution OK / 1.106 | QoR PASS / 1.106 | Median (ms) | p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Fixed-Best | 1.071 (96,84%) | 1.071 (96,84%) | 9,187 | 17,663 |
| Adaptive `legacy-v1` | 942 (85,17%) | 394 (35,62%) | 4,789 | 8,354 |
| Oracle | 1.083 (97,92%) | 1.071 (96,84%) | 3,502 | 5,9215 |

Fixed-Best có 34 `REFERENCE_FAILED` gồm 16 suy biến, 11 canvas overflow và 7 thiếu điểm. Oracle có ít nhất một cấu hình chạy thành công trên 12 trong số 34 cặp này. H tham chiếu vẫn thiếu, nên cả 34 cặp giữ `REFERENCE_FAILED` và Oracle để trống config. `Execution OK` của Oracle đếm cặp có ít nhất một cấu hình grid hoàn thành đủ repeat. Latency Oracle chỉ có ở cặp chọn được cấu hình PASS.

Adaptive có 548 cặp chạy `OK` nhưng fail QoR, 129 `EXECUTION_FAILED`, 34 `REFERENCE_FAILED` và một `ERR_LOADER_IO`. Mã native `BUDGET_EXHAUSTED` xuất hiện ở **163 cặp**, gồm 129 cặp `EXECUTION_FAILED` và 34 cặp thiếu reference. Attempt CSV có 18.204 dòng gồm cả warmup. Có 203/1.105 cặp từng retry trong các lượt đo, trong đó 27 cặp đạt PASS.

Trong 548 cặp fail QoR, 410 vượt dung sai SSIM, 384 vượt dung sai PSNR và 47 dưới coverage 0,95. Các điều kiện có thể cùng xảy ra. Hai nấc `S=0,5, K=1000` với `R=128` hoặc `256` là nấc cuối của 536/548 cặp này, đồng thời cũng là nấc cuối của 367/394 cặp PASS. Nhãn theo cặp cho thấy self-check ở các nấc này có cả ca đạt và ca hụt QoR. Campaign này không cô lập tác động riêng của scale hoặc R.

### Latency ghép cặp và Pareto

| Mode | Median trên cùng 394 cặp PASS (ms) | p95 trên cùng 394 cặp PASS (ms) |
| --- | ---: | ---: |
| Fixed-Best | 10,267 | 18,177 |
| Adaptive | 4,789 | 8,354 |
| Oracle | 3,516 | 4,978 |

Median của tỷ số theo cặp là **2,161×** cho Adaptive so với Fixed-Best và **1,363×** cho Oracle so với Adaptive. Cả hai chỉ áp dụng trên giao 394 cặp PASS. Chỉ tiêu speedup QoR toàn cohort giữ `null`. Oracle là đối chiếu offline trên grid đã đo, với config nhanh nhất theo median trong các config PASS của từng cặp.

Trên cùng 394 cặp, median S4 của Fixed-Best, Adaptive và Oracle lần lượt là **2,517 / 0,420 / 0,187 ms**, còn S5 là **6,253 / 2,926 / 1,834 ms**. S6 nằm trong khoảng 0,913–0,949 ms. Đây là phân rã chi phí quan sát trên cùng cohort. Median probe GPU và thời gian CPU controller của Adaptive lần lượt là 0,051512 ms và 0,000511 ms. Probe nằm trong thời gian pipeline, và các khoảng GPU/stream wait có thể chồng lấp. [Ranh giới timing](../runbooks/ch02_udis_benchmark_colab.md#3-chấm-qor-kiểm-join-và-xuất-báo-cáo)

Pareto cohort có **19 cấu hình** theo ba trục latency, execution OK và QoR PASS trên 1.105 cặp có timing grid. Pareto theo cặp có 10.640 điểm. Ví dụ, `S0.25_K1000_R128_Bfeather` có median grid 1,842 ms và 17 PASS, `S0.75_K500_R128_Bfeather` có 3,308 ms và 535 PASS, còn Fixed-High có 8,989 ms và 1.071 PASS. Các median grid dùng tập có timing chung. Oracle chọn 36 config khác nhau, với ba lựa chọn thường gặp nhất là `S1.00_K500_R128_Bfeather` (238 cặp), `S0.75_K500_R128_Bfeather` (235) và `S0.50_K500_R128_Bfeather` (179).

**Nghiệm thu.** Đóng CH-02 cho campaign T4 của `legacy-v1`. Đã bàn giao ledger, QoR, Oracle, CDF, Pareto và bảng LaTeX. Kết quả thực nghiệm cho thấy lợi ích latency trên giao PASS và mức giảm QoR trên toàn mẫu số. CH-04b vẫn phụ thuộc lock A5 riêng. [Kế hoạch Phase 3](../plan.md#phase-3-đại-thực-nghiệm-ablation-và-phân-tích-độc-lập)

## CH-04a — Adaptive validation A5 trên development

### Nguồn và tính đầy đủ của campaign

Lượt `run_20261106_134702Z_ch04a_gbqf76f6` dùng notebook tại `65ddbb0`, snapshot Colab `4c594daeb0d57a1fcf2ad33e32c55c9db6b5a67d`, Tesla T4 `sm_75`, `nvcc 13.0.88` và driver `580.82.07`. SHA-256 của source ZIP khớp receipt. Build Release và replay CH-01 hoàn tất. Fixed-Best vẫn là `S1.00_K2000_R1024_Bfeather`, đạt 192/200 fit và 46/50 validation. File phê duyệt ngoại lệ trong ZIP dùng CRLF, còn bản local dùng LF. Nội dung phê duyệt và kết quả giống nhau sau khi chuẩn hóa dòng. Khác biệt byte làm đổi `approval_sha256` và kéo theo SHA-256 của development lock và gate plan. [Notebook nguồn](../../notebooks/phase3_ch04a_calibration.ipynb)

Đối chiếu lại đủ **251/251 file** trong gói export và sáu marker bước. `export_verified.json` có SHA-256 `7fd22fec2a3afcdd27cd96b0923ac46a9c256c3213df7e8219ebdaf2ab2632c2`. Cell fit ghi receipt cho 500 ảnh PPM và 40 ứng viên. Ảnh PPM ở lại workspace Colab theo cấu hình export. Đối chiếu local kiểm hash của receipt, gate và candidate, rồi tính lại mẫu số, attempt count và timing từ CSV. Điểm alignment lấy từ ledger đã chấm trên Colab. [Runbook CH-04a](../runbooks/ch04a_a5_calibration.md)

Mỗi ứng viên chạy cùng 50 cặp `train-val`, seed `0`, 2 warmup và 10 lượt đo mỗi cặp. Toàn campaign có **24.000 terminal rows** gồm 4.000 warmup và 20.000 lượt đo, cùng **48.744 attempt rows**. Đối chiếu khóa cặp, warmup và repeat xác nhận số attempt khớp từng terminal. Validation trả exit `2` kèm `a5-no-go.json`, và `--verify-no-go` trả exit `0` trên Colab lẫn local. SHA-256 của kết quả là `7e15bb1d657a1a0e7580cc397964519aac336c0d5dc471974f95259fc095934c`.

### QoR theo policy

Sàn Fixed-Best là **46/50 cặp PASS** ở cả 20 policy. Bốn cặp `007175`, `009957`, `009964`, `010118` giữ `REFERENCE_FAILED` trong mỗi mẫu số 50. Hai ứng viên `r1` và `r4` có cùng tập cặp PASS trong từng policy. Median dưới đây tính trên median 10 lượt đo của mỗi cặp PASS, theo đúng tập cặp của policy đó.

| δSSIM | δPSNR (dB) | Cặp PASS mỗi ứng viên / 50 | Median `r1` (ms) | Median `r4` (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 0,01 | 0,2 | 35 | 15,477 | 8,311 |
| 0,01 | 0,5 | 35 | 15,316 | 8,295 |
| 0,01 | 0,8 | 35 | 8,323 | 8,372 |
| 0,01 | 1,0 | 35 | 8,295 | 8,303 |
| 0,02 | 0,2 | 35 | 15,312 | 8,322 |
| 0,02 | 0,5 | 35 | 8,374 | 8,302 |
| 0,02 | 0,8 | 35 | 8,297 | 8,417 |
| 0,02 | 1,0 | 35 | 8,296 | 8,300 |
| 0,03 | 0,2 | 35 | 16,867 | 8,822 |
| 0,03 | 0,5 | 35 | 15,323 | 8,288 |
| 0,03 | 0,8 | 35 | 15,980 | 8,292 |
| 0,03 | 1,0 | 36 | 15,904 | 8,394 |
| 0,04 | 0,2 | 35 | 16,988 | 8,293 |
| 0,04 | 0,5 | 35 | 15,297 | 8,280 |
| 0,04 | 0,8 | 35 | 16,883 | 8,859 |
| 0,04 | 1,0 | 34 | 16,603 | 8,694 |
| 0,05 | 0,2 | 35 | 15,294 | 8,287 |
| 0,05 | 0,5 | 35 | 15,452 | 8,492 |
| 0,05 | 0,8 | 35 | 16,020 | 8,862 |
| 0,05 | 1,0 | 34 | 16,428 | 8,654 |

Các ứng viên đạt **34–36/50 (68–72%)**, thấp hơn sàn **10–12 cặp**. Trong 2.000 bản ghi cặp–ứng viên, có 1.398 `PASS`, 442 `FAIL` và 160 `REFERENCE_FAILED`. Một policy đạt 36 cặp, 17 policy đạt 35 cặp và hai policy đạt 34 cặp. Mỗi policy fit lại controller theo nhãn riêng, nên thay đổi số cặp PASS phản ánh cả dung sai lẫn hành vi online đã học.

### Reject, false accept và chi phí routing

- Trong 4.420 lượt đo thuộc 442 bản ghi cặp–ứng viên `FAIL`, **4.300** lượt bị `REJECT` ở nấc cuối với `BUDGET_EXHAUSTED`, dù H cuối đạt alignment theo policy. Áp thứ tự self-check của C++ lên telemetry cho thấy ngưỡng vấp đầu tiên là **số inlier ở 3.900 lượt** và **inlier coverage ở 400 lượt**. Kết quả chẩn đoán dùng H trước render, nên chỉ xác nhận alignment của H cuối. [Self-check của controller](../../src/controller/validation.cpp)
- **120 lượt** còn lại được `ACCEPT` và render `OK` nhưng fail QoR, tập trung ở `007753` (70), `007341` (40) và `008159` (10) qua nhiều ứng viên. Ở policy chuẩn `δS=0,02`, `δP=0,5 dB`, cả hai ứng viên có 35/50 PASS, 110 lượt reject có H đạt alignment và 0 false accept quan sát được. Kết quả này chỉ áp dụng cho hai ứng viên của policy chuẩn trong cohort validation.
- `r4` là giới hạn số vùng. Trong 20 model `r4`, 5 model dùng 1 vùng, 9 dùng 2 vùng và 6 dùng 3 vùng. `r4` có median trên cặp PASS thấp hơn ở 16/20 policy. Với `a5-s04-p05`, median giảm từ 15,297 xuống 8,280 ms và số lượt có retry giảm từ 490 xuống 90 trên 500 lượt đo. Trace cho thấy `r4` bắt đầu 400 lượt ngay ở nấc cuối, còn `r1` bắt đầu cả 500 lượt tại `S=0,5, K=1000, R=256`. Đây là đường tiết kiệm retry quan sát được ở policy này.
- Ở 5/20 policy, toàn bộ terminal và attempt của `r1`/`r4` khớp sau khi bỏ trường timing và ID lượt chạy, nhưng latency vẫn dao động. Không có đối chứng timing xen kẽ hoặc chuỗi xung nhịp trong gói evidence. Vì vậy các median mô tả các lượt đã chạy và chưa đủ để quy toàn bộ chênh lệch cho số vùng hoặc kết luận speedup dưới ràng buộc QoR.

**Nghiệm thu.** Hoàn tất campaign CH-04a với kết quả âm đã xác minh. Tất cả 20 policy đều `selected=null`, nên chưa có `a5-lock.json` và CH-04b vẫn bị chặn. Kết quả chỉ áp dụng cho controller family, fallback self-check, cohort và sàn Fixed-Best đã khóa. CH-02 dùng quy trình nghiệm thu controller của benchmark chính. [Đặc tả A5](../experiments/ablation-studies.md#25-ablation-a5-độ-nhạy-của-ngưỡng-chất-lượng-overlap)

## Bằng chứng chung của lượt A4 và CH-05

Gói Colab đã tải về nằm tại `specs/run_20261106_043655Z_ch03_ch05_gbfq3y9z/`. Đối chiếu local xác nhận đủ **2.232 file**, không thiếu hoặc thừa so với inventory, tất cả SHA-256 khớp và 13 marker bước khớp `export_verified.json`. SHA-256 của marker export là `9ab6209f8e89809c2fa04cf205f4ceeeb67b8ff7ab6f3d9bb05a8be98dcf9679`. Hai summary CH-05 và summary A4 tái tạo đúng nội dung JSON đã xuất từ campaign gốc. Các file kết quả được lưu riêng khỏi Git. [Notebook nguồn](../../notebooks/phase3_ch03_ch05_experiments.ipynb)

Thiết bị đo là **Tesla T4**, compute capability **7.5**, VRAM **15.360 MiB**, driver **580.82.07**. CMake cache xác nhận Release và CUDA architecture `75`, `nvcc` là `13.0.88`. Snapshot Colab `aec69534ff73702d545fdc9ce5345a8b8057db60` có inventory khớp 233 file tracked của `fb85db7` sau khi chuẩn hóa line endings khi cần, cùng bảy file pilot bổ sung. Source ZIP ghi SHA-256 riêng và thiếu Git comment, nên dùng inventory để đối chiếu source. Hash hai runner trong các campaign khớp `build-provenance/binaries.json`.

Hai CTest `test_s6_cuda` và `test_admission_control` pass trong 0,48 giây. Fixture S6 có 0 lỗi memcheck, initcheck và synccheck, cùng 0 hazard racecheck. Native smoke CH-05 có 0 lỗi memcheck trên một case Feather 720p. Bằng chứng sanitizer áp dụng cho các fixture và smoke đã chạy. Correctness của campaign kích thước lớn dựa trên ledger, mask và oracle theo phạm vi bên dưới.

## CH-03/A4 — Same-H blending trên T4

Chọn 10 cặp có overlap pixel native cao nhất trong 46 cặp `train-val` có reference sử dụng được. Mỗi cặp khóa cùng RGB, H gốc B→A, canvas và mask cho ba blender. Tỷ lệ diện tích giao so với diện tích ảnh nguồn nhỏ hơn là **86,1–99,1%** theo registration. Đây là cohort development thiên về overlap cao. H được lấy từ cấu hình reference CH-01 để cố định đầu vào A4. [Runbook A4](../runbooks/ch03_same_h_colab.md)

Campaign có **30/30 case cặp–blender đạt**, tương ứng **300 lượt đo và 60 warmup**. Đối chiếu lại 20 ảnh PPM, H và canvas với registration. Cả 30 case khớp union mask từng pixel và sai số RGB tối đa quan sát được là **1 mức** trên oracle FP64, dưới tolerance 2. Intersection được đối chiếu bằng geometry oracle và overlap counter của native runner.

| Blender | Cặp đạt | Median GPU (ms) | p95 GPU (ms) | Median host (ms) | Workspace S6 tối đa (MiB) | Throughput median (MPix/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Feather | 10/10 | 1,260 | 1,859 | 1,271 | 0,000 | 239,40 |
| Multiband 3 | 10/10 | 4,833 | 8,355 | 4,843 | 23,839 | 62,49 |
| Multiband 5 | 10/10 | 5,689 | 9,052 | 5,702 | 23,874 | 53,25 |

Median và p95 lấy qua 10 median theo cặp, sau khi bỏ warmup. Throughput dùng toàn canvas. Workspace là giá trị query tối đa của S6. Giá trị 0 của Feather áp dụng cho workspace phụ trợ. Hai mode Multiband cấp thêm 256 byte guard ngoài query. Trên cohort này, median GPU của Multiband 3 và 5 gấp lần lượt **3,83×** và **4,51×** Feather. Multiband 5 tăng khoảng **17,7%** thời gian so với Multiband 3, trong khi workspace tối đa tăng khoảng **0,035 MiB**. Các con số đo phản ánh tổng chi phí render của từng mode.

Gói visual có **15 PNG lossless** gồm ba bộ panorama đầy đủ và hai bộ crop 96×96, mỗi bộ có ba blender. `captions.txt` và `visual_manifest.json` đưa tổng số file lên 17. Đã so pixel sau decode của cả 15 PNG với PPM tương ứng và kiểm tọa độ crop. Hai crop đã khóa ở `009015` và `009583`. Trong panorama `009583`, Feather có viền kép rõ quanh ô tô trắng và xe đạp, giảm thấy được ở cả hai mode Multiband. Ở `002674`, Feather cho thấy dải bóng cạnh trụ và cờ, trong khi hai panorama Multiband có biên đơn rõ hơn ở các vị trí này. Hai crop hẹp cho thấy khác biệt ít rõ hơn, và quan sát trên các ví dụ đã chọn chưa đủ để xếp hạng Multiband 3 với Multiband 5 về chất lượng.

Alignment SSIM/PSNR được đo một lần mỗi cặp từ RGB/H chung. SSIM quan sát nằm trong 0,462–0,932 và PSNR trong 14,65–33,88 dB. Các giá trị này mô tả alignment đầu vào. Kết luận về seam và ghosting ở trên dựa trên panorama/crop và giữ phạm vi định tính. **Đủ bằng chứng để đóng CH-03/A4 trên cohort development đã khóa.** [Đặc tả A4](../experiments/ablation-studies.md#24-ablation-a4-same-homography-blending-trade-off-trong-s6)

## CH-05 — Stress-test 720p, 1080p và 4K

Hai track dùng cùng manifest synthetic, seed `20261030`, 18 cặp và ba blender. Mỗi track có **54 case cặp–blender**, với hai warmup và mười lượt đo mỗi case. Cả hai ledger đủ **648 dòng**. Tách riêng execution success, kiểm correctness và sai số đăng ký ảnh. [Runbook CH-05](../runbooks/ch05_multires_colab.md)

### S6-only với H ground truth

**54/54 case đạt**, gồm 540 lượt đo và 108 warmup thành công, 0 OOM và 0 timeout. Có sáu full-golden case, đều là Feather 720p. 48 case còn lại dùng oracle ROI, gồm toàn bộ 1080p và 4K. Tất cả mask khớp trong miền đã kiểm, RGB lệch tối đa 1 mức. Ở ROI, Multiband được kiểm màu vùng một nguồn và mask. Màu vùng overlap của Multiband ngoài phạm vi kiểm màu này.

| Input | Blender | Case đạt | Median GPU (ms) | p95 GPU (ms) | Workspace S6 tối đa (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 720p | Feather | 6/6 | 3,039 | 3,074 | 0,000 |
| 720p | Multiband 3 | 6/6 | 11,985 | 21,350 | 105,492 |
| 720p | Multiband 5 | 6/6 | 12,275 | 13,670 | 105,644 |
| 1080p | Feather | 6/6 | 6,849 | 13,362 | 0,000 |
| 1080p | Multiband 3 | 6/6 | 25,188 | 32,308 | 237,110 |
| 1080p | Multiband 5 | 6/6 | 25,455 | 28,837 | 237,451 |
| 4K | Feather | 6/6 | 26,651 | 27,157 | 0,000 |
| 4K | Multiband 3 | 6/6 | 100,095 | 111,687 | 948,265 |
| 4K | Multiband 5 | 6/6 | 101,039 | 112,257 | 949,611 |

Số pixel input tăng 9× từ 720p lên 4K. Median GPU tăng **8,77×**, **8,35×** và **8,23×** lần lượt cho Feather, Multiband 3 và Multiband 5. Đây là xu hướng gần với tăng theo pixel trên các recipe đã đo. Canvas 4K rộng **5.016–6.564 pixel**, khác kích thước input 3.840 pixel. Mức **949,611 MiB** là workspace query tối đa của S6. Tổng device allocation khai báo ở case lớn nhất của Multiband 5 là **1.052,390 MiB**. Cả hai có phạm vi riêng với peak VRAM. Mức 26,651 ms của Feather đo khoảng GPU của S6 với H đã biết. Nó không suy ra tốc độ ghép end-to-end.

Mỗi ô p95 được tính từ sáu pair medians. Ở 720p, hai pair medians của Multiband 3 vượt 21 ms, trong khi bốn pair còn lại nằm khoảng 10,56–12,07 ms. Đây là lý do p95 của ô này cao hơn p95 Multiband 5 trong lượt đo. Cỡ mẫu và các log hiện có chưa tách được nguyên nhân của khác biệt đuôi.

### End-to-end với H từ S5

Track chạy fixed `S=0.5`, `K=1000`, `R=256` và trả `COMPLETED_WITH_FAILURES`. Exit 2 là disposition của campaign đã hoàn tất, với báo cáo và ledger đã verify. Ground truth được dùng sau lượt chạy để tính corner error.

| Input | Cặp hoàn tất / tổng | Case hoàn tất / tổng | Dòng `SUCCESS` | Dòng `ERR_CANVAS_OVERFLOW` | Dòng `ERR_NOT_ENOUGH_POINTS` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p | 1/6 | 3/18 | 36 | 180 | 0 |
| 1080p | 0/6 | 0/18 | 0 | 0 | 216 |
| 4K | 0/6 | 0/18 | 0 | 0 | 216 |
| Tổng | 1/18 | 3/54 | 36 | 180 | 432 |

Các dòng gồm cả warmup. Trong 540 lượt đo có 30 lượt `SUCCESS`, 150 overflow và 360 thiếu điểm. Cả track ghi 0 OOM và 0 timeout. Năm cặp 720p bị canvas guard từ chối. Các cặp 1080p và 4K dừng ở lỗi S5 thiếu điểm. Pattern tuần hoàn của synthetic là giả thuyết hợp lý cho bước chẩn đoán. Ledger này chưa tách được keypoint count, số match vượt ratio test và nguồn gây thiếu correspondences.

Cặp duy nhất thực thi qua cả ba blender là `scene000_affine_overlap00_720p`. Canvas ước lượng **1340×1301**, có **4 inlier**, inlier ratio **1,65%**. Sai số L2 trung bình qua bốn góc là **1.295,27 px** và lớn nhất **2.260,88 px**. Median GPU span 6,622 / 17,881 / 19,037 ms chỉ mô tả lượt chạy xong của cặp này, theo thứ tự Feather / Multiband 3 / Multiband 5. Chỉ có một pair median cho mỗi mode, nên giá trị p95 trong bảng bằng median và không mô tả phân phối rộng. Kết quả này là **execution thành công nhưng đăng ký hình học kém**. `end_to_end` là kết quả âm hợp lệ của workload đã khóa, chưa có cơ sở để kết luận latency scaling cho pipeline đăng ký ảnh thành công ở ba độ phân giải.

## CH-06 — Đánh giá Đa Kiến trúc (Cross-Architecture Study: T4 vs L4 vs A100)

Tuấn phụ trách thực thi và thu thập bằng chứng CH-06 thông qua notebook `notebooks/phase3_ch06_cross_arch.ipynb` trên Google Colab Pro, tái hiện chính xác track synthetic S6-only của CH-05 (18 cặp ảnh, 3 blender Feather, Multiband 3, Multiband 5, seed `20261030`, gồm 2 warmup và 10 repeats mỗi case, tổng cộng 54 case và 648 dòng ledger mỗi GPU). Cả hai phiên chạy đều dùng gói source thống nhất (`51bfef40242946011ef70d804c5d998dcff67d0922b2cf17b52d7b38c1fcc28e`).

### 1. Cấu hình thiết bị đo đạc
- **NVIDIA L4**: Compute capability 8.9 (`sm_89`), 23.034 MiB VRAM, driver `580.82.07`, CUDA 13.0, 58 SMs, 7.424 CUDA cores, băng thông 300 GB/s GDDR6, dung lượng L2 cache 48 MB.
- **NVIDIA A100-SXM4**: Compute capability 8.0 (`sm_80`), 81.920 MiB VRAM (80GB High-RAM), driver `580.82.07`, CUDA 13.0, 108 SMs, 6.912 CUDA cores, băng thông 2.039 GB/s HBM2e, dung lượng L2 cache 40 MB.

### 2. Kết quả thực nghiệm và đối chiếu vi kiến trúc
Cả hai GPU đều đạt **54/54 case thành công (100%)**, 0 OOM và 0 timeout.

| Độ phân giải | Blender | T4 Median (ms) | L4 Median (ms) | Speedup L4 / T4 | A100 Median (ms) | Speedup A100 / T4 | Speedup A100 / L4 | S6 Workspace (MiB) |
| :--- | :--- | ---: | ---: | :---: | ---: | :---: | :---: | :---: |
| 720p | Feather | 3,039 | 1,651 | **1,84×** | 0,091 | **33,35×** | 18,12× | 0,000 |
| 720p | Multiband 3 | 11,985 | 6,303 | **1,90×** | 4,273 | **2,81×** | 1,48× | 105,492 |
| 720p | Multiband 5 | 12,275 | 6,595 | **1,86×** | 4,626 | **2,65×** | 1,43× | 105,644 |
| 1080p | Feather | 6,849 | 3,632 | **1,89×** | 0,193 | **35,58×** | 18,87× | 0,000 |
| 1080p | Multiband 3 | 25,188 | 14,066 | **1,79×** | 7,299 | **3,45×** | 1,93× | 237,110 |
| 1080p | Multiband 5 | 25,455 | 14,336 | **1,78×** | 7,621 | **3,34×** | 1,88× | 237,451 |
| 4K | Feather | 26,651 | 14,365 | **1,86×** | 0,751 | **35,51×** | 19,14× | 0,000 |
| 4K | Multiband 3 | 100,095 | 56,610 | **1,77×** | 22,966 | **4,36×** | 2,47× | 948,265 |
| 4K | Multiband 5 | 101,039 | 56,641 | **1,78×** | 23,450 | **4,31×** | 2,42× | 949,611 |

### 3. Phân tích định lượng và cơ chế phần cứng
- **Ổn định trên Ada Lovelace (L4)**: L4 mang lại mức tăng tốc đều đặn **1,77× – 1,90×** so với T4 trên toàn bộ ma trận test. Độ trễ scale tỉ lệ thuận với số pixel từ 720p lên 4K (8,59× – 8,98× trên mức tăng pixel 9×). L2 cache 48 MB của L4 che giấu hoàn toàn độ trễ đọc ghi của các tầng tháp pyramid trung gian, khiến chênh lệch giữa Multiband 3 và Multiband 5 ở 4K chỉ còn dưới 0,06% (56,610 ms vs 56,641 ms).
- **Bứt phá băng thông trên Ampere (A100)**: Chế độ Feather streaming đạt tốc độ sub-millisecond ở cả 3 mức phân giải (0,091 ms ở 720p và 0,751 ms ở 4K), tương ứng speedup **33,35× – 35,58× so với T4**, phản ánh năng lực truyền tải dữ liệu của bus HBM2e 2.039 GB/s. Đối với Multiband, speedup tăng dần từ 2,65× – 2,81× ở 720p lên 4,31× – 4,36× ở 4K nhờ mức độ bão hòa 108 SMs được cải thiện rõ rệt khi diện tích canvas lớn.
- **Tính bất biến của bộ nhớ S6 (Workspace Invariance)**: Queried S6 workspace ở cả 3 kiến trúc khớp từng byte (105,492 / 105,644 MiB ở 720p, 237,110 / 237,451 MiB ở 1080p, và 948,265 / 949,611 MiB ở 4K). Bằng chứng xác nhận ước lượng bộ nhớ Stage 6 độc lập hoàn toàn với vi kiến trúc GPU.
- **Nghiệm thu.** Đóng chính thức task CH-06. Toàn bộ raw metrics, summary report, environment receipts và device info đã được lưu trữ trong `results/ch06_s6_l4_results` và `results/ch06_s6_a100_results`. Bảng LaTeX đã được tích hợp vào `paper/tables/table_cross_architecture.tex`.

## CH-07/08 — Tái lập và bàn giao phần T4

Tooling đọc hai export đã pin, giữ đủ mẫu số CH-02 và tính lại frontier từ CSV đã join. Biểu đồ CDF phân biệt tập PASS riêng của từng mode. Bảng paired và biểu đồ timing dùng giao ba mode cùng PASS. Tỷ số latency lấy median của các tỷ số theo cặp. Lỗi reference và execution OK được trình bày riêng vì hai trạng thái có thể cùng xuất hiện.

Bundle `build/ch07-final-bundle/` gồm bốn cặp hình PDF/SVG, năm bảng, macro, nguồn dẫn xuất và hồ sơ kiểm. `--verify-only` tái tạo **21/21 artifact** khớp SHA-256. Manifest có SHA-256 `9e61ee7501953b8991e2a23785e5f5cb1b78cb85336025a9eedd5f03df22005b`. Bốn hình PDF đạt kiểm tra font TrueType embed và metadata. Các file kết quả vẫn ở workspace. [Runbook](../runbooks/ch07_ch08_paper_artifacts.md)

Manuscript dùng Pareto, CDF và crop. Biểu đồ timing chi tiết được bàn giao trong bundle. Crop hậu kiểm chỉ minh họa viền kép quanh xe với cùng bbox ở ba blender. Bản thảo giữ kết quả âm của synthetic end-to-end, tách khỏi kết luận render với H đã biết. [Claim–evidence map](../claim_evidence_map.md)

Build bằng OASIcs trên WSL/TeX Live trả `PASS`, có **12 trang gồm 1 trang tiêu đề, 10 trang nội dung và 1 trang references**. Đã xem cả 12 trang render. Final log không có warning, citation/reference chưa resolve hoặc overfull box. Font được embed, author metadata là `Anonymous Authors` và kiểm tra text không phát hiện định danh nhóm. Receipt và PDF nằm trong `build/paper-ch08-accepted/`. [Yêu cầu nộp bài của workshop](https://parma-ditam-workshop.github.io/#information-for-authors)

Đồng bộ `main` trước khi tích hợp manuscript đã qua **14 CTest host-only**, compile/link bốn target `sm_75` và **596 pytest pass, 11 skip**. Số đo T4 tiếp tục truy về binary và source của campaign gốc. Harness sau đồng bộ cần smoke/sanitizer riêng trên T4 trước campaign mới. [Quyết định tích hợp](../decisions/p3-harness-integration.md)

## Công việc còn mở

| Task | Việc tiếp theo |
| --- | --- |
| CH-04b | Bị chặn bởi `NO_GO` của CH-04a. Chỉ mở campaign A5 trên test khi có bộ `a5-lock.json` đã xác minh. |
| CH-05 end-to-end | Giữ nguyên kết quả âm hiện tại. Nếu cần đường scaling đăng ký ảnh hợp lệ, thiết kế và khóa một cohort synthetic bổ sung có tín hiệu đặc trưng phân biệt được, kèm chẩn đoán S2/S4/S5. |
| CH-06 | **Đã hoàn thành.** Tuấn đã chạy thành công 54/54 case trên cả L4 và A100, xác thực tính toàn vẹn 648 rows/GPU, workspace khớp bitwise và đã tích hợp bảng số liệu vào paper. Đóng task. |
| CH-08 — Freeze bản nộp chung | Tổng hợp bằng chứng của các owner, xử lý A5 theo quyết định nhóm, tích hợp CH-06 khi có và audit lại bản nộp cuối. |
