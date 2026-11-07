# Báo cáo chi tiết latency, QoR và Oracle

Ngày 04/11 (Thứ 4), năm 2026. Bảng được sinh từ raw CSV và canonical QoR bằng `detailed_metrics`/`detailed_export`.
Binary đo `47256d4`, Tesla T4, `sm_75`; ảnh 512 × 512; seed 20261101; hai warmup và mười repeat. Policy đã khóa, không tune lại bằng Oracle holdout.

## 1. Vì sao có median 13,496 ms?

13,496 ms là Fixed-Best trên riêng chín pair mà p80 thử rẻ, dừng ngay và PASS QoR. Fixed-Best luôn chạy một attempt; con số này không cộng retry.
Mốc khoảng 7,5 ms trước đó thuộc một cohort/lượt đo khác. Hai cohort validation cũ/mới không trùng pair. Không dùng tỷ số giữa các hàng dưới đây làm speedup.

| Cohort và mode | N | Mean của pair-median (ms) | Median (ms) | p95 (ms) |
|---|---:|---:|---:|---:|
| Validation cũ: Fixed-Best | 49 | 8.5401 | 7.6260 | 15.8765 |
| Validation cũ: Controller v2 | 49 | 8.6031 | 7.6670 | 15.8906 |
| Validation mới: Fixed-Best | 49 | 9.4779 | 8.5710 | 16.8103 |
| Nhóm chín pair: Fixed-Best | 9 | 14.4438 | 13.4955 | 18.2956 |
| Cùng chín pair: p80 | 9 | 7.7729 | 7.5650 | 9.3658 |

Cả chín pair dùng Fixed-Best S1/K2000/R1024 và Controller S1/K1000/R256, blend feather.

| Pair | Fixed median (ms) | Controller median (ms) | Matches Fixed | Mean S5 Fixed (ms) |
|---|---:|---:|---:|---:|
| udis_train_005298 | 10.8930 | 7.5650 | 634 | 7.8762 |
| udis_train_000671 | 12.0940 | 6.7760 | 637 | 8.1014 |
| udis_train_003153 | 13.1010 | 6.6490 | 691 | 8.7876 |
| udis_train_004604 | 13.3095 | 9.5040 | 848 | 10.2404 |
| udis_train_001606 | 13.4955 | 7.1170 | 749 | 9.1845 |
| udis_train_003198 | 15.0605 | 7.6390 | 883 | 10.7343 |
| udis_train_002784 | 15.6970 | 8.4750 | 950 | 11.3934 |
| udis_train_002702 | 17.5525 | 9.1585 | 1133 | 13.2099 |
| udis_train_000004 | 18.7910 | 7.0725 | 742 | 12.9720 |

Mean stage dưới đây lấy trực tiếp các lượt đo, không cộng các median stage. `t_s6` của fixed bao gồm khoảng chờ host sau S5; không diễn giải mọi cột là thời gian kernel thuần.

| Nhóm chín pair | S1 | S2 | S3 | S4 | S5 | S6 raw | Mean E2E raw (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-Best | 0.3709 | 0.2207 | 0.2670 | 2.3383 | 10.2777 | 0.9767 | 14.4737 |
| Controller | 0.6130 | 0.1787 | 0.1805 | 0.6618 | 4.7434 | 0.8995 | 7.7749 |

Nhóm chín pair có mean 807.4 matches ở Fixed-Best; toàn 49 pair mới là 444.9. S5 chấm hypothesis trên tập matches nên cùng R không đồng nghĩa cùng chi phí. Đây là bằng chứng workload khác; không tách hết ảnh hưởng xung nhịp/lượt đo.

## 2. Latency trên tập mới trong campaign gốc

Mean/median/p95 dưới đây tính trên median của mười repeat mỗi pair. Cùng 49 pair reference hợp lệ, gồm cả các QoR FAIL của từng mode; bảng latency này không chứng nhận giữ QoR. Một reference failure được giữ riêng trong cohort 50 pair.

| Mode | PASS | FAIL | Unknown | Mean (ms) | Median (ms) | Std (ms) | p95 (ms) | Min–max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| reference | 49 | 0 | 1 | 9.4779 | 8.5710 | 4.3042 | 16.8103 | 3.3185–18.7910 |
| fixed_original | 25 | 24 | 1 | 4.2638 | 3.7300 | 1.8839 | 7.3882 | 1.6300–8.2290 |
| legacy | 25 | 24 | 1 | 4.6553 | 4.6615 | 1.7600 | 7.6418 | 2.1595–8.7430 |
| v2 | 49 | 0 | 1 | 9.4801 | 8.7415 | 4.1265 | 15.7153 | 3.4185–17.6820 |
| v3_strict | 49 | 0 | 1 | 10.0914 | 9.2600 | 4.1329 | 16.3203 | 4.0190–18.2655 |
| v3_p90 | 49 | 0 | 1 | 10.0814 | 9.3400 | 4.1337 | 16.3273 | 4.0380–18.3075 |
| v3_p80 | 48 | 1 | 1 | 9.6396 | 8.4750 | 4.2041 | 17.1050 | 3.9940–19.4705 |

`reference_valid_raw_run_distribution` ở mỗi mode và `raw_run_distribution` ở breakdown chứa thêm mean của tất cả lượt đo. Mean raw và mean của pair-median là hai phép tính khác nhau. `t_controller` có phần chồng lấp với probe/feature; không cộng nó vào tổng stage để suy E2E.

## 3. Speedup trên cùng tập QoR PASS

| Mode so với Fixed-Best | N cùng PASS | Tỷ số mean | Tỷ số median | Median tỷ số paired | Geometric mean paired |
|---|---:|---:|---:|---:|---:|
| reference | 49 | 1.0000× | 1.0000× | 1.0000× | 1.0000× |
| fixed_original | 25 | 2.1766× | 2.3493× | 2.1330× | 2.1343× |
| legacy | 25 | 2.1064× | 2.3612× | 2.1272× | 2.0293× |
| v2 | 49 | 0.9998× | 0.9805× | 0.9875× | 0.9921× |
| v3_strict | 49 | 0.9392× | 0.9256× | 0.9240× | 0.9192× |
| v3_p90 | 49 | 0.9401× | 0.9177× | 0.9249× | 0.9202× |
| v3_p80 | 48 | 0.9700× | 0.9929× | 0.8963× | 0.9486× |

P80 nhanh hơn trên 9/48 pair cùng PASS, chậm hơn trên 39. CI bootstrap 95% của tỷ số mean: [0.8688, 1.0936]. Resample theo pair, 10.000 lần, seed 20261101. CI này không mô hình hóa đầy đủ tương quan scene hoặc drift phần cứng.

## 4. QoR thực đo

Các metric dưới đây gồm mọi candidate có metric hợp lệ, cả PASS và FAIL. Cột N cho biết số pair có metric; không nhân N lên theo repeat. PASS hiện là không giảm quá 0,02 SSIM và 0,5 dB PSNR, coverage ít nhất 0,95 so với reference. Reference không phải ground truth và PASS không bảo đảm SSIM tuyệt đối cao.

| Mode | N metric | Mean SSIM | Median SSIM | Mean PSNR (dB) | Mean coverage | Min coverage |
|---|---:|---:|---:|---:|---:|---:|
| reference | 49 | 0.7112 | 0.7476 | 21.8704 | 1.000000 | 1.000000 |
| fixed_original | 49 | 0.6891 | 0.7392 | 21.2077 | 0.977652 | 0.775797 |
| legacy | 46 | 0.6978 | 0.7589 | 21.5156 | 0.984072 | 0.850095 |
| v2 | 49 | 0.7112 | 0.7476 | 21.8704 | 1.000000 | 1.000000 |
| v3_strict | 49 | 0.7112 | 0.7476 | 21.8704 | 1.000000 | 1.000000 |
| v3_p90 | 49 | 0.7112 | 0.7476 | 21.8704 | 1.000000 | 1.000000 |
| v3_p80 | 49 | 0.7089 | 0.7476 | 21.8605 | 0.999448 | 0.977028 |

P80 có một false accept: `udis_train_003317`. Mean SSIM drop 0.002297, max drop 0.087468. Ở pair lỗi, PSNR tốt hơn reference nhưng SSIM giảm 0,087468, vượt ngưỡng 0,02. Không dùng mean gần nhau để bỏ qua pair FAIL.

## 5. Oracle trên đúng 50 pair mới

Đây là lượt đo đánh giá bổ sung, tách khỏi campaign gốc. Quét đủ 48 cấu hình và chạy lại v2/p80 xen kẽ theo batch, cùng binary/policy/seed; H/status/count của policy khớp campaign gốc. Không fit từ grid holdout.

| Mode, 49 reference-valid pair | Mean (ms) | Median (ms) | p95 (ms) |
|---|---:|---:|---:|
| reference | 9.3719 | 8.6595 | 15.6281 |
| v2 | 9.8002 | 8.7600 | 16.1621 |
| v3_p80 | 9.6657 | 8.5650 | 17.1690 |
| Oracle toàn grid | 3.7379 | 3.7460 | 5.9376 |

Fixed-Best/Oracle: tỷ số mean 2.5073×, tỷ số median 2.3117× trên 49 pair. Đây là lợi thế Oracle offline, chưa gồm chi phí tìm kiếm online. Reference failure còn lại không được đổi thành PASS.

Cấu hình ban đầu trùng Oracle: **1/49**. Cấu hình cuối trùng Oracle: **2/49**.

| Tham số cuối | Trùng Oracle | Mean độ lệch tuyệt đối | Median độ lệch tuyệt đối |
|---|---:|---:|---:|
| S | 21/49 | 0.2296 | 0.2500 |
| K | 10/49 | 948.9796 | 1000.0000 |
| R | 4/49 | 632.1633 | 896.0000 |

Trên 48 pair mà cấu hình cuối đạt QoR, chỉ 4 cấu hình có latency không quá 10% so với Oracle. Chênh fixed cùng cấu hình so với Oracle: mean 4.2949 ms, median 3.3440 ms, max 12.8000 ms.

Chỉ 4/49 lựa chọn Oracle nằm trong ladder p80. Oracle giới hạn đúng ladder đó có median 5.1750 ms, so với 3.7460 ms của toàn grid; đây vẫn là upper bound bỏ qua gate/probe/retry.
Không chỉ bộ chọn ban đầu: 45/49 đáp án Oracle không có trong ladder đã khóa, nên bộ chọn hoàn hảo cũng không chọn đúng chúng. Trùng cấu hình không phải tiêu chí duy nhất; regret và số cấu hình nằm trong 10% Oracle được báo kèm để xét các trường hợp gần hòa latency.

Phân rã mean trên cùng 48 pair PASS: Oracle 3.7321 ms → chạy fixed cấu hình Controller đã chọn 8.0270 ms → Controller E2E 9.7264 ms. Chênh cấu hình là 4.2949 ms; chênh E2E so với fixed cùng cấu hình 1.6994 ms. Phần sau gồm probe/retry/host và khác biệt lượt đo, không coi toàn bộ là một kernel overhead.

Các cấu hình Oracle chọn nhiều nhất:

| Cấu hình | Pair |
|---|---:|
| S1.00_K500_R128_Bfeather | 12 |
| S0.50_K500_R128_Bfeather | 8 |
| S0.75_K500_R128_Bfeather | 8 |
| S0.50_K2000_R128_Bfeather | 4 |
| S0.75_K1000_R128_Bfeather | 2 |
| S0.75_K500_R512_Bfeather | 2 |
| S1.00_K1000_R128_Bfeather | 2 |
| S1.00_K2000_R1024_Bfeather | 2 |

Oracle trên 49 pair: mean SSIM 0.7114, mean PSNR 22.0150 dB, mean coverage 0.995652. Cả 49 lựa chọn thỏa predicate tương đối, không phải bảo đảm SSIM tuyệt đối cao.

| Pair lệch lớn | Cấu hình cuối | Oracle | Fixed cấu hình đã chọn (ms) | Oracle (ms) | Controller E2E (ms) |
|---|---|---|---:|---:|---:|
| udis_train_008229 | S1.00_K2000_R1024_Bfeather | S0.75_K500_R128_Bfeather | 17.5520 | 4.7520 | 18.3010 |
| udis_train_002519 | S1.00_K2000_R1024_Bfeather | S0.50_K500_R128_Bfeather | 14.5970 | 3.7980 | 15.4700 |
| udis_train_000007 | S1.00_K2000_R1024_Bfeather | S0.75_K500_R128_Bfeather | 13.1755 | 3.8020 | 19.4870 |
| udis_train_002739 | S1.00_K2000_R1024_Bfeather | S1.00_K500_R128_Bfeather | 13.0980 | 3.8340 | 13.8140 |
| udis_train_000611 | S1.00_K2000_R1024_Bfeather | S1.00_K500_R512_Bfeather | 13.8305 | 5.2185 | 14.6580 |

## 6. Ảnh đối chiếu

Ảnh B được warp vào hệ tọa độ A bằng H đã đo, dùng cùng geometry/interpolation của evaluator. Đây là ảnh alignment để đối chiếu QoR. Ô đen là ngoài miền ảnh B; heatmap là sai khác photometric, không phải SSIM map.
Các preview được kiểm metric khớp CSV trong 1e-12; WebP lossless và PNG khôi phục có cùng hash pixels RGBA. Ảnh gốc A/B 512 × 512 cũng nằm cạnh preview.

### cheap-pass: udis_train_001606

Controller `S1.00_K1000_R256_Bfeather`; Oracle `S1.00_K500_R128_Bfeather`.

![udis_train_001606](../../results/controller-v3-t4-20261104/oracle-checkpoint/images/cheap-pass-udis_train_001606.png)

### ssim-failure: udis_train_003317

Controller `S1.00_K1000_R256_Bfeather`; Oracle `S0.50_K500_R128_Bfeather`.

![udis_train_003317](../../results/controller-v3-t4-20261104/oracle-checkpoint/images/ssim-failure-udis_train_003317.png)

### retry: udis_train_000668

Controller `S1.00_K2000_R1024_Bfeather`; Oracle `S0.75_K500_R128_Bfeather`.

![udis_train_000668](../../results/controller-v3-t4-20261104/oracle-checkpoint/images/retry-udis_train_000668.png)

## 7. Artifact và kiểm chứng

- [JSON đầy đủ: distribution, QoR và CI](../../results/controller-v3-t4-20261104/detailed-report-complete/detailed-statistics.json)
- [CSV latency từng pair, giữ cả FAIL/unknown](../../results/controller-v3-t4-20261104/detailed-report-complete/latency-per-pair.csv)
- [CSV chín pair và stage breakdown](../../results/controller-v3-t4-20261104/detailed-report-complete/nine-pair-breakdown.csv)
- [CSV cấu hình ban đầu/cuối, Oracle và regret](../../results/controller-v3-t4-20261104/detailed-report-complete/controller-vs-oracle.csv)
- Tool thống kê đã kiểm bằng fixture tỷ số paired cố định, bootstrap, infinity/missing và guard partition; toàn bộ Python suite 414 test PASS ở local.
- Oracle giữ đầy đủ 24.000 measurement và 4.800 warmup; audit canonical giữ reference failure. Không tune lại policy và không đổi ngưỡng QoR.
- CTest/CUDA của binary giữ nguyên từ campaign trước; lần bổ sung chỉ chạy đánh giá và xuất ảnh, không sửa kernel hoặc policy.
