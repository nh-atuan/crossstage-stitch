# Kiểm chứng T4 và sweep Phase 2 của Đế

- Người thực hiện: Đế qua agent, theo yêu cầu chạy lại hướng dẫn sweep của Nhóm.
- Ngày: 30/10 (Thứ 6), năm 2026.
- Phase 2; deadline chung 31/10 (Thứ 7).
- Phạm vi: Controller host, tích hợp Controller–S5, grid S×K×R và audit QoR. Chưa có adaptive end-to-end campaign.

## 1. Môi trường và source

| Thành phần | Giá trị thực chạy |
|---|---|
| GPU | Tesla T4, 15.360 MiB, driver 580.82.07 |
| CUDA | 13.0, nvcc V13.0.88, architecture 75 |
| C++ | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| CMake | cmake version 3.31.10 |
| Python | 3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0] |
| NumPy | 2.1.3 |
| OpenCV runtime | 5.0.0; requirements pin opencv-python 5.0.0.93 |
| Source nền | 2df11184a07bdd6c65641ed2de820eab1ed50d93 |

Source được chuyển từ workspace qua archive, kiểm SHA-256 từng file trên T4. Bổ sung GPU test và audit bằng delta riêng đã kiểm hash. Receipt nằm ở `results/de_p2_campaign_20261030/evidence/source-receipt.json` và `delta-core.json`. Thay đổi manuscript có trong source transfer; pipeline/shared interface không đổi. `STITCH_GIT_REV` được chèn bằng header build để CSV ghi đúng revision thay cho chuỗi HEAD.

Harness binary SHA-256: `495f4f6cbf2aecfabb9ce4257e38e8c59d269a6a71c9c557973c789c5d5645d8`.

## 2. Build, test và sanitizer

| Kiểm tra | Kết quả thực chạy |
|---|---|
| Root build + CTest | 12/12 PASS; không SKIP |
| Controller host build + CTest | 2/2 PASS |
| Controller GPU integration build + CTest | 5/5 PASS, gồm Controller, trace, Controller–S5, S5 math và S5 CUDA |
| S5 CUDA: memcheck và racecheck | 0 errors; racecheck 0 hazards, 0 warnings |
| Controller–S5 CUDA: memcheck và racecheck | 0 errors; racecheck 0 hazards, 0 warnings |
| QoR, report và sweep audit pytest trên T4 | 110 PASS |

Test Controller–S5 dùng correspondences tổng hợp, seed 20261030: ép self-check đầu thất bại, nối tiếp R từ 17 lên 257, kiểm H/H_s/mask/tín hiệu khớp chạy full 257, rồi kiểm stale generation bị REJECT/EXECUTION_ERROR. Đây là kiểm chứng integration với output CUDA thật, chưa phải pipeline scene-aware từ ảnh.

```bash
cmake -S src/controller -B build/controller-gpu -DSTITCH_CONTROLLER_GPU_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build/controller-gpu -j2
ctest --test-dir build/controller-gpu --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/controller-gpu/test_controller_s5
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/controller-gpu/test_controller_s5
```

Log gốc: `results/de_p2_campaign_20261030/evidence/build.log` và `sweep.log`.

## 3. Cohort và tính đầy đủ

Chạy theo [guide](sweep-gonogo-guide.md). Trong runtime mới không có UDIS-D, dataset preparer auto sinh đúng 25 pairs tổng hợp bằng seed 42. Đây là synthetic pilot; không gọi là dữ liệu UDIS-D, validation/final hoặc bằng chứng tổng quát hóa.

Lưới có 48 cấu hình: S = 0.25/0.5/0.75/1.0; K = 500/1000/2000; R = 128/256/512/1024; B = feather. Mỗi pair/config có 2 warm-up và 10 lượt đo; RANSAC seed 0. Wrapper chỉ bổ sung nhãn `dataset=synthetic`, `split=synthetic`, seed/run ID; vẫn gọi binary harness gốc.

- Tổng hàng CSV: 14,400; warm-up: 2,400; lượt đo: 12,000.
- Đủ 25 × 48 × 10 lượt đo; không duplicate hoặc missing repeat.
- Trạng thái lượt đo: OK 9,820; S5 NOT_ENOUGH_POINTS 1,440; S5 DEGENERATE 660; S6 INSUFFICIENT_WORKSPACE 80. Giữ nguyên mọi failure trong raw.
- QoR chuẩn pass 459/1.200 pair/config; mọi pair có ít nhất một cấu hình khả thi.
- 50 ảnh được khôi phục cục bộ bằng đúng generator; SHA-256 từng PPM khớp ảnh gốc đã dùng trên T4. Manifest gốc và hash được giữ nguyên. Audit từ CSV tải về tái tạo đúng toàn bộ JSON canonical trên T4.

Raw và ảnh ở [campaign cục bộ](../results/de_p2_campaign_20261030/). Outputs nằm ngoài Git theo conventions; chưa upload Drive hoặc push.

| Artifact | SHA-256 |
|---|---|
| runs.csv | `7c18fc0bafb8b5671a78d6a636afba8de493f8d72f11c00b6853c16aa06647fc` |
| qor.csv | `b34ebc07168c325c1b89f4354eed21781d58e794fac22ed770efda80ecdfbeab` |
| policy | `579d0815eecd1bbf6c3d7276ec4922aee525b8c280ef756db038e835a5f5f37f` |
| pairs manifest | `a4422fbda0e3807b6643ca08684c81bafc3d070f5eea3305e54d728a53957865` |
| configs | `518bfc8927f10f5939f7b00f5e402ef86f2b57f6060c7e0c05ad20e641777cf8` |

Archive bằng chứng tải từ Colab: `f913d3153b9a11c8ad68de08bb58cc7a299feb03ff6163f0e77d73449b1d247b`; đã kiểm đủ hash 20 file trong archive. Tool versions và audit geometry là artifact bổ sung riêng trong campaign.

## 4. QoR chuẩn và kết quả legacy

`tools.controller.audit_sweep` gọi predicate chung cho từng repeat: SSIM/PSNR so với reference cùng pair, coverage, reference status và pipeline status. Không tạo Oracle fallback khi không có cấu hình khả thi. Policy vẫn là pilot-not-team-approved.

| Chỉ số | Kết quả |
|---|---|
| Fixed-Best theo QoR chuẩn | `S1.00_K2000_R128_Bfeather` |
| Fixed-Best pass | 25/25 pairs |
| Fixed-Best median E2E | 3.3360 ms |
| Oracle coverage | 25/25 pairs |
| Canonical median của tỷ số paired | 1.002175× |
| Canonical tỷ số hai median | 1.075435× |
| Legacy tỷ số hai median | 1.618239× |
| Tỉ trọng stage chỉnh được do legacy tính từ cùng raw | 68.58% |

Hai phép tổng hợp speedup khác nhau được ghi tên rõ; không so median paired với tỷ số median như cùng một chỉ số. Cả hai giá trị canonical đều dưới 1.20×. Legacy chỉ kiểm valid stitch và coverage, bỏ SSIM/PSNR/reference status nên báo GO trên cùng raw. Output được giữ ở `legacy-gonogo-not-for-policy.json` để đối chiếu; không dùng nó để khóa policy.

Guide ghi `--out` cho analyzer; CLI hiện có hai lựa chọn `--out-dir`/`--out-json`, nên đã dùng `--out-json`. Không đổi script của Tuấn. Bộ audit riêng có test về reference lỗi, thiếu repeat, pipeline failure và Pareto nondominated.

## 5. Kiểm tra reference bằng geometry đã biết

`tools.controller.audit_synthetic` phát lại RNG và xác nhận từng pixel A/B khớp 25 pairs gốc trước khi phục hồi affine B → A. So sánh reference với mapping đã biết ở bốn góc B:

- Median mean corner error: 0.154133 pixel gốc.
- Mean corner error lớn nhất trong 25 pairs: 0.725661 pixel gốc.

Đã xem [preview bốn ca sai số lớn nhất](../results/de_p2_campaign_20261030/sweep/reference-preview.png): A và B warp theo reference có alignment hợp lý; diff còn photometric shift và nearest-neighbour sampling. Đây là sanity check geometry trên ảnh tổng hợp. Không tự đổi reference statuses, quality policy hoặc suy ra chất lượng perceptual trên ảnh thật.

## 6. Ảnh hưởng tới các task còn lại

Theo tiêu chí mục 7 của [plan](plan.md), điều kiện Oracle ≥ 1.20× không đạt trên synthetic cohort này khi dùng QoR đầy đủ. Đây là evidence mới cần Nhóm review lại; agent không tự sửa quyết định GO đã ghi trong báo cáo Phase 1 hoặc đổi hướng paper.

Controller và test reuse đã có thể bàn giao. Chưa khóa numerical policy, chưa sửa S1/harness của Tuấn, chưa có terminal adaptive traces hoặc overhead end-to-end. Không nới QoR hoặc tune classifier để tìm một speedup mà Oracle chuẩn chưa hỗ trợ. Khuyến nghị review lại gate bằng predicate chuẩn và chọn hướng dữ liệu/thí nghiệm trước khi dành tiếp công sức cho calibration và adaptive campaign.

Paper chỉ mô tả module/method đã kiểm chứng, không đưa các số synthetic này thành kết quả final. Trạng thái build PDF cập nhật trong [bàn giao Controller](p2-de-controller-implementation.md).
