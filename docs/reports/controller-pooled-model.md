# Kết quả mô hình Controller dùng chung

Ngày 04/11 (Thứ 4), năm 2026. Các bảng sinh từ JSON/CSV; không sửa số đo bằng tay.

Biến thể được chọn trước holdout: `v4_photo`. Mất 1 cặp QoR PASS của reference trên tập kiểm tra mới. Chưa đạt ràng buộc giữ chất lượng.

## Phạm vi và protocol

Đã triển khai mô hình dùng chung theo scene và S/K/R, XGBoost cây nông, sigmoid calibration, gate sau S5 và inference C++ không phụ thuộc thư viện ML. Giữ đủ 48 cấu hình, gồm K500. Policy thử tối đa một candidate rồi fallback reference, tổng tối đa hai attempt. Không trích gate feature khi đã chọn reference. Chưa đổi policy mặc định.

Giữ nguyên QoR tương đối: SSIM giảm tối đa 0,02, PSNR giảm tối đa 0,5 dB, coverage tối thiểu 0,95. Reference S1/K2000/R1024, feather. Đây là QoR alignment tương đối; reference không phải homography ground truth và PASS không bảo đảm chất lượng ảnh tuyệt đối cao.

Calibration gồm 200 cặp training; fitter giữ các cặp có reference hợp lệ. Mỗi component ảnh nằm trong một outer fold. Booster fit và sigmoid calibration dùng các component khác nhau; chọn ngưỡng và hyperparameter loại outer evaluation. So sánh sáu nhóm feature bằng calibration trước khi mở holdout. Kết quả của nhóm được chọn trên calibration không được xem là ước lượng generalization không thiên lệch.

Khóa 50 component training mới trước khi fit, loại toàn bộ calibration và cả hai cohort validation đã xem. Kiểm CRC/size toàn index ZIP ngăn trùng ảnh; chưa chứng nhận scene disjointness bằng annotation. Không tune trên UDIS-D test hoặc HPatches. Không điều chỉnh model theo kết quả holdout.

Binary đo cuối: `a4bf474e6d63458eb26fa9f82efaed924eaeb12d`; Tesla T4, `sm_75`, seed 20261101, hai warmup và mười repeat. Grid calibration có mười repeat trên source đã merge main; kernel feature được tối ưu sau đó. Collector đối chiếu H/status/count với grid. Phép đo cuối dùng cùng binary cho reference và các policy.

## Chi phí feature trước và sau song song hóa

Đây là microbenchmark trên calibration với route cố định, không phải policy đạt QoR. Probe/decision dùng model diagnostic nhỏ; inference của mô hình thật được tính trong E2E bên dưới. Không cộng các cột timing chồng lấp để suy ra latency.

| Nhóm | Gate cũ (ms) | Gate mới (ms) | Probe/decision cũ (ms) | Probe/decision mới (ms) | Attempt hợp lệ |
|---|---:|---:|---:|---:|---:|
| existing | 0.3909 | 0.3907 | 0.2615 | 0.2599 | 729 |
| pair | 0.3937 | 0.3901 | 4.7320 | 0.3618 | 729 |
| matching | 0.9393 | 0.4098 | 0.2587 | 0.2594 | 729 |
| geometry | 0.7208 | 0.4482 | 0.2580 | 0.2599 | 729 |
| photo | 1.1942 | 0.4291 | 0.2579 | 0.2597 | 729 |
| all | 2.0371 | 0.4744 | 4.7274 | 0.3620 | 729 |

`pair`: tương quan ảnh nhỏ và vị trí/độ rõ peak. `matching`: phân bố distance và tỷ lệ top2 từ workspace S4. `geometry`: residual p50/p90 và sai số theo vùng. `photo`: SSIM patch 7×7, edge agreement và sai khác chuẩn hóa. `all`: kết hợp. Không chạy thêm reverse matching để lấy mutual fraction; nhóm matching tận dụng kết quả top2 đã có.

## Ablation ngoài fold trên calibration

Chi phí replay dùng timing grid và microbenchmark; probe có dự toán thêm 0,25 ms cho inference đầy đủ. Fallback tính bảo thủ, chưa trừ lợi ích reuse. Các số sau chưa phải speedup đo trên tập mới.

| Nhóm | Cặp | False accept | Retry | Dừng rẻ | Mean replay (ms) | Mean reference (ms) |
|---|---:|---:|---:|---:|---:|---:|
| existing | 191 | 7 | 19 | 26 | 8.9544 | 8.8432 |
| pair | 191 | 5 | 3 | 18 | 8.9506 | 8.8432 |
| matching | 191 | 4 | 17 | 18 | 9.0840 | 8.8432 |
| geometry | 191 | 6 | 21 | 22 | 9.1125 | 8.8432 |
| photo | 191 | 3 | 23 | 20 | 9.2144 | 8.8432 |
| all | 191 | 4 | 6 | 17 | 9.0093 | 8.8432 |

Chọn `v4_photo` bằng calibration, ưu tiên không có false accept quan sát được rồi xét chi phí. Xác suất model và zero error trên phần chọn ngưỡng không bảo đảm zero error trên dữ liệu khác.

## Latency và failure trên tập mới

Mỗi pair lấy median của mười repeat; mean/median/p95 dưới đây tính trên các pair có reference hợp lệ, gồm cả candidate QoR FAIL. Cột PASS/FAIL/unknown giữ đủ cohort 50 pair.

| Mode | PASS | FAIL | Unknown | N latency | Mean (ms) | Median (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 49 | 0 | 1 | 49 | 8.5870 | 7.9215 | 16.5950 |
| fixed_original | 19 | 30 | 1 | 49 | 3.7101 | 3.4230 | 7.3008 |
| legacy | 16 | 33 | 1 | 49 | 4.4490 | 4.2560 | 7.4374 |
| baseline_v2 | 49 | 0 | 1 | 49 | 8.5657 | 7.7445 | 16.7017 |
| baseline_v3 | 49 | 0 | 1 | 49 | 9.1060 | 8.4335 | 16.9765 |
| v4_existing | 49 | 0 | 1 | 49 | 8.7773 | 7.9260 | 16.9659 |
| v4_pair | 49 | 0 | 1 | 49 | 8.7417 | 8.1765 | 16.3642 |
| v4_matching | 49 | 0 | 1 | 49 | 8.7869 | 8.0115 | 16.9315 |
| v4_geometry | 49 | 0 | 1 | 49 | 8.7841 | 8.0045 | 16.9209 |
| v4_photo | 48 | 1 | 1 | 49 | 8.8261 | 7.9950 | 16.5090 |
| v4_all | 49 | 0 | 1 | 49 | 8.7401 | 8.3040 | 16.2747 |

## Speedup trên cùng cặp QoR PASS

Bảng chỉ dùng giao các cặp cùng PASS với reference; không thay cho kiểm mất chất lượng trên toàn cohort. Speedup bằng reference/candidate. CI bootstrap 95% resample theo pair, không coi repeat là mẫu độc lập.

| Mode | N | Tỷ số mean | Tỷ số median | Median paired | CI 95% tỷ số mean | Mất PASS |
|---|---:|---:|---:|---:|---|---:|
| reference | 49 | 1.0000× | 1.0000× | 1.0000× | [1.0000, 1.0000] | 0 |
| fixed_original | 19 | 2.3088× | 2.2855× | 2.2014× | [2.0973, 2.6037] | 30 |
| legacy | 16 | 2.2221× | 2.3111× | 2.1988× | [2.0757, 2.3870] | 33 |
| baseline_v2 | 49 | 1.0025× | 1.0229× | 0.9863× | [0.9861, 1.0282] | 0 |
| baseline_v3 | 49 | 0.9430× | 0.9393× | 0.9006× | [0.8471, 1.0571] | 0 |
| v4_existing | 49 | 0.9783× | 0.9994× | 0.9610× | [0.9609, 1.0036] | 0 |
| v4_pair | 49 | 0.9823× | 0.9688× | 0.9390× | [0.9354, 1.0430] | 0 |
| v4_matching | 49 | 0.9772× | 0.9888× | 0.9607× | [0.9599, 1.0020] | 0 |
| v4_geometry | 49 | 0.9776× | 0.9896× | 0.9604× | [0.9600, 1.0026] | 0 |
| v4_photo | 48 | 0.9704× | 0.9421× | 0.9459× | [0.9336, 1.0058] | 1 |
| v4_all | 49 | 0.9825× | 0.9539× | 0.9346× | [0.9360, 1.0429] | 0 |

## QoR và hành vi policy

Các mean QoR gồm mọi candidate có metric hợp lệ trên tập reference-valid, cả PASS và FAIL; N có thể khác nhau theo metric. Không dùng mean gần reference để bỏ qua một pair FAIL.

| Mode | N SSIM | Mean SSIM | N PSNR | Mean PSNR (dB) | Mean coverage | Retry pair | False accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 49 | 0.7056 | 49 | 21.5921 | 1.000000 | 0 | 0 |
| fixed_original | 47 | 0.6712 | 47 | 20.9905 | 0.957750 | 0 | 0 |
| legacy | 39 | 0.6628 | 39 | 21.3269 | 0.963599 | 11 | 23 |
| baseline_v2 | 49 | 0.7056 | 49 | 21.5921 | 1.000000 | 0 | 0 |
| baseline_v3 | 49 | 0.7055 | 49 | 21.5862 | 0.999973 | 9 | 0 |
| v4_existing | 49 | 0.7056 | 49 | 21.5921 | 1.000000 | 0 | 0 |
| v4_pair | 49 | 0.7055 | 49 | 21.5909 | 0.999979 | 3 | 0 |
| v4_matching | 49 | 0.7056 | 49 | 21.5921 | 1.000000 | 0 | 0 |
| v4_geometry | 49 | 0.7056 | 49 | 21.5921 | 1.000000 | 0 | 0 |
| v4_photo | 49 | 0.7039 | 49 | 21.5664 | 0.998332 | 4 | 1 |
| v4_all | 49 | 0.7055 | 49 | 21.5909 | 0.999979 | 3 | 0 |

Các pair mất PASS của `v4_photo`: `udis_train_009552`.

Phân bố cấu hình ban đầu `v4_photo`:

| Cấu hình | Pair |
|---|---:|
| S1.00_K2000_R1024 | 35 |
| S1.00_K2000_R128 | 14 |
| S1.00_K1000_R128 | 1 |

## Đối chiếu Oracle trên cùng tập mới

Lượt sweep bổ sung tách khỏi campaign trên. Chỉ dùng số trong lượt này để tính chênh với Oracle. Oracle chọn sau khi chạy toàn grid và chấm QoR, không gồm chi phí tìm kiếm online.

| Mode | Mean (ms) | Median (ms) | p95 (ms) |
|---|---:|---:|---:|
| reference | 8.4557 | 7.6775 | 16.6045 |
| v4_photo | 8.9158 | 8.0650 | 16.6221 |
| Oracle | 3.3861 | 3.0990 | 6.2540 |

Oracle có đáp án cho 49/50 pair. Cấu hình đầu trùng Oracle 0; cấu hình cuối trùng 0. Có 49 đáp án Oracle trong tập cấu hình policy. Có 2 cấu hình cuối đạt QoR và có latency không quá 10% so với Oracle.

| Tham số cuối | Trùng Oracle | Mean độ lệch tuyệt đối | Median độ lệch tuyệt đối |
|---|---:|---:|---:|
| S | 22 | 0.2449 | 0.2500 |
| K | 7 | 1193.8776 | 1500.0000 |
| R | 9 | 666.1224 | 896.0000 |

Trên cùng 48 pair PASS: Oracle 3.3774 ms → fixed cấu hình đã chọn 8.1156 ms → Controller E2E 8.9378 ms. Chênh chọn cấu hình 4.7382 ms; phần E2E còn lại 0.8222 ms. Phần sau gồm probe, gate, retry, host và khác biệt lượt đo; không gán toàn bộ cho một kernel.

## Feature thực sự xuất hiện trong cây

Đếm node split của model được chọn; đây không phải mức quan trọng nhân quả.

| Model | Feature | Node split |
|---|---|---:|
| pre_quality | config_scale | 113 |
| pre_quality | config_k_fraction | 30 |
| pre_quality | a_tile3_gradient | 28 |
| pre_quality | a_tile0_gradient | 22 |
| pre_quality | a_tile2_contrast | 19 |
| post_quality | config_scale | 56 |
| post_quality | residual_original | 24 |
| post_quality | b_tile1_gradient | 18 |
| post_quality | config_k_fraction | 17 |
| post_quality | spatial_entropy_b | 16 |
| latency | config_scale | 72 |
| latency | config_k_fraction | 39 |
| latency | config_r_fraction | 37 |
| latency | b_tile2_contrast | 23 |
| latency | pair_tile2_mean_difference | 22 |

## Ảnh kiểm tra

Đây là alignment preview theo H đo được, không phải panorama S6. Metric đối chiếu CSV trong 1e-12. WebP lossless được kiểm pixel trước khi phục hồi PNG ở local.

### cheap-pass: udis_train_006395

Controller `S1.00_K2000_R128_Bfeather`; Oracle `S0.75_K500_R128_Bfeather`.

![udis_train_006395](../../results/controller-v4-t4-20261104/final-checkpoint/v4-images/cheap-pass-udis_train_006395.png)

### qor-failure: udis_train_009552

Controller `S1.00_K2000_R128_Bfeather`; Oracle `S0.75_K500_R128_Bfeather`.

![udis_train_009552](../../results/controller-v4-t4-20261104/final-checkpoint/v4-images/qor-failure-udis_train_009552.png)

### retry: udis_train_005090

Controller `S1.00_K2000_R1024_Bfeather`; Oracle `S0.25_K500_R128_Bfeather`.

![udis_train_005090](../../results/controller-v4-t4-20261104/final-checkpoint/v4-images/retry-udis_train_005090.png)

## Kiểm chứng và artifact

- T4 source `a4bf474`: 23 CTest PASS, 424 test Python PASS.
- Memcheck feature mới, feature cũ và regression harness: 0 errors; Python/C++ kiểm prediction và chọn cấu hình ban đầu trên calibration.
- Trong sweep Oracle, các run v4 thành công được kiểm H/status/count/seed khớp fixed cùng cấu hình; không chỉ kiểm policy giữa hai lượt.
- [JSON thống kê](../../results/controller-v4-t4-20261104/final-checkpoint/v4-detailed/summary.json), [CSV từng pair](../../results/controller-v4-t4-20261104/final-checkpoint/v4-detailed/latency-per-pair.csv), [CSV so với Oracle](../../results/controller-v4-t4-20261104/final-checkpoint/v4-detailed/controller-vs-oracle.csv).
- [Freeze holdout](../../results/controller-v4-t4-20261104/final-checkpoint/v4-validation/freeze.json), [tóm tắt ablation](../../results/controller-v4-t4-20261104/final-checkpoint/v4-fast-ablations/summary.json), [parity C++](../../results/controller-v4-t4-20261104/final-checkpoint/v4-fast-parity/summary.json).
- Không dùng kết quả development này thay bảng toàn UDIS-D test, HPatches hoặc nghiệm thu Phase 3 gốc. Chưa push hoặc tạo PR.
