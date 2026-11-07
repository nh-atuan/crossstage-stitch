# Protocol và ngân sách Controller v5

Khóa ngày 07/11 (Thứ 7), năm 2026; Đế yêu cầu triển khai W1–W5. Thuộc Phase 3, deadline chung 07/11 (Thứ 7). Không chọn thông số bằng evaluation đã xem. Thông số thực thi nằm trong `tools/controller/v5/protocol.py`; runtime xuất `RUN_MATRIX.md`/JSON theo input thực tế.

## Dữ liệu và split

- Nguồn cũ: `full-1382a8b5cae5-68c68d47bc6c42ad93cc2974fcd59193.zip`, reconstructed v4 photo 64 cây depth 2; không ghép với weights v4 lịch sử 128 cây.
- Grid có 200 calibration pair × 48 config, 2 warmup/10 repeat; 195 Reference-valid pair vào training/OOF, 5 Reference failure giữ trong ledger. Không chạy lại grid này khi hash/schema/outcome phù hợp.
- Archived W1/diagnostics outer models có capacity theo fold 3/64, 2/128, 3/64, 2/64, 2/128; ghi riêng với final P0 2/64. W2–W4 dùng fixed capacity 2/64 để đối chiếu công bằng.
- 50 evaluation pair P0–P5 đã xem, chỉ dùng development diagnostic E2E sau khi model/policy được freeze. Không có full evaluation sweep nên không so Oracle trên cohort này. Không dùng chúng để fit sigmoid, chọn feature, threshold, seed hoặc model.
- Chia theo shared-image components, kiểm split và archived image hashes. Lịch sử seen cohorts/scene annotations chưa đầy đủ nên chưa chứng nhận fresh independent holdout. Không dùng UDIS test/HPatches để train.
- E2E dùng cùng source, seed 20261101, device-ready boundary và ảnh 512×512. Fixed Reference là S1/K2000/R1024/feather. QoR pilot giữ nguyên coverage 0.95, SSIM drop 0.02, PSNR drop 0.5 dB cùng status/metric validity; Reference failure là UNKNOWN.

## Ma trận bắt buộc

| Phần | Biến thể | Cách đánh giá |
|---|---|---|
| W1 | current, perfect pre, perfect post, perfect latency, perfect pre+post, joint | Calibration OOF, offline truth interventions, hai cost scenarios và transition sensitivity |
| W2 | PRE0 photo; PRE1 photo+NCC | Cùng grouped splits, 64 cây depth 2; fixed-config AUC/AP, calibration, cheap/top-k/near-Oracle recall, weighted regret |
| W3 | POST0 photo binary; POST1 all; POST2 photo+MG không NCC; POST3_PHOTO; POST3_GEOMETRY; POST3; POST4 violation/joint heads | Proper booster/probcal split; chọn threshold trong training-side partition, báo outer predictions; feature selection của POST4 không lấy outer labels |
| W4 | PRE0/ PRE1 × POST0 / nested-selected new gate | 40/80/120/156 component hoặc cap khả thi; 3 seed, 5 fixed outer folds, nested gate/feature selection |
| W5 | REF, V4, SEL, GATE, COST+GATE, STAGED, V5_COMBINED | Native T4 E2E, cùng source/seed/cohort; xoay/đảo thứ tự theo batch; checkpoint từng mode |

PRE2 correspondence probe tùy chọn chưa implement, có status NOT_IMPLEMENTED; không thêm chi phí GPU vào ma trận chính. W4 không thêm pair thiếu provenance: NO_NEW_TRAINING_DATA cho expansion khi chưa có manifest độc lập được duyệt. Việc có training archive không tự chứng nhận scene mới độc lập.

V5_COMBINED hiện cùng stack với STAGED; đo cả hai như identity control, không xem sai biệt timing giữa chúng như đóng góp phương pháp. SEL giữ old gate/direct Reference; GATE giữ old selector/direct Reference; COST+GATE kết hợp new pre/cost và gate; STAGED bổ sung feedback rerank/reuse.

## Budget đã chặn

Với cohort hiện tại:

| Hạng mục | Số lượng |
|---|---:|
| Full GPU grid mới | **0** |
| Spatial candidate slots từ H cũ trên CPU | Tối đa 195 × 48 = 9.360; chỉ extract khi H/payload hợp lệ |
| Transition profiling | 12 calibration pair × 6 transition × 12 run = 864 |
| Feature/group profiling | 12 calibration pair × 8 control × 12 run = 1.152 |
| E2E main | 50 pair × 7 variant × 12 run = 4.200 |
| Tổng profile + E2E | 6.216 run; một run adaptive có thể nhiều attempt |
| W2 CPU model fits | 2 pre families × 4 outputs × 5 folds = 40 |
| W3 CPU model fits | Tối đa (6 binary + 4 margin heads) × 5 folds = 50 |
| W4 CPU model fits | Tối đa (4 pre outputs + 6 binary + 4 margin heads) × 4 sizes × 3 seed × 5 folds = 840, giảm khi reuse A–E tương thích |

Không nhân full Cartesian product feature × threshold × dataset. Thresholds được thử offline trên cùng prediction, không phát sinh GPU sweep. Model final/reject/future fit một lần, lưu riêng và freeze trước evaluation. Không có lời hứa thời gian vì chưa đo trên Colab. Nếu thiếu input hoặc class support, giữ denominator/NULL/status rõ ràng thay vì tạo số đo.

## Các tiêu chí và accounting

- Pre ranking tách khỏi calibration và hard filter. PR-AUC ghi rõ noninterpolated AP; fixed-config K500 có denominator/component riêng, không coi 9.360 config-row là 9.360 ảnh độc lập.
- Gate threshold ứng viên: 0/0.05/0.10/0.20/0.35/0.50 risk; ưu tiên ít false accept rồi nhiều accepted-PASS trên training-side development. Không phải quality guarantee. Pre threshold 0.50 cố định cho new selector; old selector giữ 0.85 và old gate giữ 0.65.
- Reject model target phụ thuộc gate version/self-check; feedback model target là future acceptance của gate đó. Các targets biết qua recorded post observations chỉ xuất hiện offline.
- Replay latency ghi ESTIMATED_REPLAY; perfect targets ghi OFFLINE_ORACLE. Matrix transition chỉ là estimate shrink từ measured GPU controls; unsupported transition không mặc định rẻ. No independence assumption cho margin risks.
- W5 tổng hợp median 10 repeat theo pair, rồi mean các pair cùng Reference-valid denominator. Paired speedup là ratio của paired means; bootstrap component giữ toàn bộ repeat/config cùng group. QoR FAIL và UNKNOWN vẫn có ledger riêng.
- Raw E2E là wall time chính. `t_controller` gồm probe/features/inference, `t_s1` gồm CUDA probe; không cộng chúng để suy E2E. Model cost và measured cost xuất hai trường riêng.
- Điều kiện qualification tương lai: frozen fresh holdout có provenance, không thêm QoR FAIL theo budget nhóm duyệt và paired E2E cải thiện với CI phù hợp. Hiện chưa có fresh holdout nên NOT_QUALIFIED, default không đổi; kết quả âm giữ nguyên.

## Mapping source và kiểm chứng

[Implementation](controller-v5-implementation.md) mô tả C++/CUDA entrypoints; [runbook](controller-v5-runbook.md) là hướng dẫn chạy duy nhất. Test local gồm C++ host CTest, Python grouped training/export/parity, checkpoint/failure packaging, notebook schema/auth/CPU fixture. CUDA spatial parity và native S5/harness integration được build/chạy bởi notebook trên T4; chưa được agent chạy trên GPU.

Công thức metrics và XGBoost dùng lại implementation repo có nguồn trong [boosted model](../tools/controller/boosted_model.py) và [diagnostic metrics](../tools/controller/diagnostics/metrics.py). Không lấy số đo hoặc sửa QoR từ nguồn ngoài.
