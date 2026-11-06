# Kết quả Phase 3 của Đế

- **Ngày bàn giao:** 03/11 (Thứ 3), năm 2026.
- **Deadline phase:** 05/11 (Thứ 5).
- **Nhánh:** `de/p3-controller-evaluation`.
- **Source đo chính thức:** `c58d754a7d7234d6c8610c7181e0e1dc6754fef3`.
- **Phạm vi:** telemetry Controller, A2, HPatches và draft Section 5.2/6. Policy variants cho A5 chưa thuộc lần bàn giao này.

## 1. Kết luận và điều kiện diễn giải

Đã chạy đủ cohort, kiểm tra dữ liệu và sinh báo cáo từ raw CSV. A2 cho thấy
reuse giảm thời gian retry trên cohort chỉ tăng R. Controller hiện tại lại
có median latency cao hơn và ít panorama hoàn tất hơn Fixed trên UDIS-D.
Không dùng kết quả này để khẳng định speedup end-to-end hoặc QoR tốt hơn.

HPatches đã có bảng MCE và so sánh trên tập giao. Những cặp được giữ lại
không giống nhau giữa hai chế độ; median riêng của từng tập không chứng minh
Controller giữ nguyên độ chính xác trên toàn cohort. Không có ngưỡng
noninferiority được chốt để đưa ra kết luận đó.

## 2. Môi trường và protocol

| Mục | Giá trị đã quan sát |
|---|---|
| GPU | Google Colab Tesla T4, 15.360 MiB |
| Driver | 580.82.07 |
| CUDA / nvcc | 13.0 / 13.0.88 |
| Build | Release, `sm_75`, nhúng đầy đủ source revision vào CSV |
| Seed RANSAC | 20261101, in trong log và CSV |
| Lặp | 2 warmup và 10 repeat đo mỗi pair/mode |
| Tổng hợp | Median theo repeat trước; median/p95 trên phân phối theo pair |
| Blend | Feather cố định |
| UDIS-D | Đủ 1.106 cặp test; không dùng để tune policy |
| HPatches | Đủ 59 viewpoint sequences, ảnh 1 so với ảnh 2–6, tổng 295 cặp |

Mỗi batch có 32 cặp, batch cuối giữ nguyên số cặp còn lại. Thứ tự mode đảo
theo batch. Các campaign GPU chạy tuần tự. `t_e2e` bắt đầu sau khi RGB sẵn
sàng trên device, kết thúc khi panorama sẵn sàng hoặc khi run thất bại;
bao gồm probe, Controller, retry và đồng bộ nội bộ. Decode và H2D nằm ngoài
vùng đo. Cột H2D của harness không dùng để kết luận chi phí truyền bằng zero.

Fixed dùng S=0.5, K=1000, R=256. Tên Fixed trong báo cáo này chỉ cấu hình
đối chứng đã giữ nguyên, không tuyên bố chọn lại Fixed-Best trên test.
Ladder và các ngưỡng Controller giữ nguyên; không bổ sung calibration cho A5.

## 3. Telemetry UDIS-D

| Chỉ số | Fixed | Adaptive |
|---|---:|---:|
| Input pairs | 1.106 | 1.106 |
| Panorama hoàn tất | 1.038 | 945 |
| Median latency, mọi pair (ms) | 4.073 | 4.756 |
| Mean latency, mọi pair (ms) | 4.291 | 4.983 |
| p95 latency, mọi pair (ms) | 8.075 | 8.589 |

Adaptive có 204 cặp retry, tương ứng 18,44%; số attempt trung bình là 1,374.
Inliers cuối trừ đầu trung bình là 1,561 trên toàn bộ cohort, kể cả các cặp
không đổi. Đây không phải số inliers được phục hồi riêng trên tập thành công.

| Cấu hình | Số pair khởi đầu | Số pair kết thúc |
|---|---:|---:|
| S=0.25, K=500, R=128 | 3 | 2 |
| S=0.50, K=1000, R=128 | 647 | 613 |
| S=0.50, K=1000, R=256 | 456 | 287 |
| S=0.75, K=2000, R=512 | 0 | 30 |
| S=1.00, K=2000, R=1024 | 0 | 174 |

Phân bố cuối gồm cả run bị reject. Adaptive có 161 cặp
`BUDGET_EXHAUSTED`. Fixed có 38 cặp `DEGENERATE`, 17 cặp
`NOT_ENOUGH_POINTS` tại S5 và 13 cặp `INSUFFICIENT_WORKSPACE` tại S6.
Hai chế độ có cơ chế chấp nhận khác nhau; không đồng nhất `OK` với đạt
predicate QoR offline. Median trên mọi pair cũng chịu ảnh hưởng của các
run kết thúc sớm, nên không thay thế đối sánh latency dưới cùng ràng buộc QoR.

## 4. A2: reuse khi chỉ tăng R

Theo [quyết định A2](../decisions/p3-de-a2-reuse.md), cả hai biến thể giữ
workspace đã cấp phát. No Reuse chạy lại S4 và toàn bộ search S5; With Reuse
giữ match và nối tiếp raw search. Allocation không nằm trong phép đo này.

Trên toàn bộ UDIS-D, hai biến thể có cùng path, trạng thái, số match/inlier
và homography trong sai số đã khóa. Kiểm path gồm cả warmup và repeat đo.
Chỉ 35 cặp có R-only retry trong repeat đo, tạo 35 transition hợp lệ cho A2.

| Chỉ số conditional retry | With Reuse | No Reuse |
|---|---:|---:|
| Median host retry (ms) | 0.5547 | 1.3006 |
| Median CUDA S4+S5 (ms) | 0.5457 | 1.2812 |

Median của mức tiết kiệm ghép cặp là 0,8015 ms, tương đương 58,21%.
Hiệu của hai median và median của các hiệu ghép cặp là hai đại lượng khác
nhau. Con số này chỉ mô tả cohort R-only retry; không phải mức tăng tốc trên
1.106 cặp hoặc lợi ích tránh cấp phát bộ nhớ. Không bỏ pair thất bại khỏi ledger.

## 5. HPatches và MCE

Ground truth dùng `inv(H_1_i)` theo chiều B → A. MCE là trung bình khoảng
cách Euclidean tại bốn tâm pixel góc ảnh B, đơn vị pixel ảnh gốc. Giữ nguyên
độ phân giải nguồn; lỗi render được đếm riêng khi H vẫn được giữ lại.

| Chỉ số | Fixed | Adaptive |
|---|---:|---:|
| Tổng pair | 295 | 295 |
| Có H được giữ lại để đánh giá | 278 | 193 |
| Panorama hoàn tất | 231 | 185 |
| Median MCE trên tập H giữ lại (px) | 18.533 | 7.334 |
| p95 MCE trên tập H giữ lại (px) | 5776.775 | 192.855 |

Trên 193 cặp chung có H được giữ lại, mean MCE là 68,224 px với Fixed và
38,547 px với Adaptive. Median của delta Adaptive trừ Fixed bằng zero;
p95 delta là 3,537 px. Vì vậy không kết luận Adaptive cải thiện mọi pair.
Curve MCE dùng toàn bộ 295 cặp làm mẫu số, để các cặp bị reject vẫn được thể
hiện qua phần coverage chưa đạt. Bảng riêng và curve không che số lỗi lớn.

## 6. Các ca diagnostic đã xem

Ảnh overlay và ledger nằm trong [case-review](../../results/p3-de-t4-20261102/final-checkpoint/report/cases/case-review.png).
Ba ca thất bại được chọn theo ID đầu tiên; ba ca phục hồi được chọn theo
mức tăng inliers. Đây là minh họa có quy tắc chọn, không phải mẫu đại diện.

- `000080`: tăng inliers từ 10 lên 23 sau ba attempt; tại nấc cuối vẫn dưới
  ngưỡng 35. Coverage inlier cuối khoảng 0,028, dù residual khoảng 1,468 px.
  Một residual thấp trên consensus nhỏ chưa đủ để chấp nhận model.
- `000082`, `000084`: inliers cuối lần lượt 18 và 17, vẫn bị reject sau ba
  attempt. Ảnh có vùng tuyết và mặt đường ít texture; overlay còn lệch các
  cấu trúc. Không suy ra nguyên nhân parallax chỉ từ status hoặc ảnh này.
- `000774`, `000776`: tăng S/K cùng R giúp inliers từ 22 lên 76 và 74.
  Vì nhiều núm cùng đổi, không gán mức phục hồi này riêng cho việc tăng R.
- `000649`: R=128 → 256 giữ nguyên 18 inliers; lần tăng S/K tiếp theo đưa
  số inliers lên 62 và đạt self-check. Reuse có thể giảm chi phí thử thêm R
  dù lần tăng R đó không phục hồi thêm inliers.

Không có annotation độc lập cho strong parallax trong cohort này. Global
homography có giới hạn với nhiều độ sâu và vật thể chuyển động; đây là giới
hạn phương pháp, không phải nhãn nguyên nhân đã xác minh cho từng pair.

## 7. Kiểm chứng và artifact

- Tesla T4: 17/17 CTest pass, gồm regression test 96 lần đổi ảnh và kiểm probe đầu.
- Tesla T4: 382 test Python pass; memcheck regression upload báo 0 errors.
- Smoke batch chứa `000424`: path With/No Reuse khớp cả warmup và repeat đo.
- Toàn cohort UDIS-D: path và kết quả With/No Reuse khớp trước khi phân tích A2.
- Đã tải checkpoint cuối về local và xác minh hash của 569 file.
- Tính lại báo cáo trên local: `summary.json` khớp hoàn toàn bản sinh trên T4.
- Local: 382 test Python pass sau các thay đổi trong lần bàn giao này.
- Draft Controller: PDF 2 trang, không còn cảnh báo LaTeX, metadata `Anonymous Authors`, font được nhúng đầy đủ.

Receipt, log build/test, binary hash và source hash nằm trong
[evidence](../../results/p3-de-t4-20261102/final-checkpoint/evidence/).
Raw CSV và metadata nằm trong
[campaigns](../../results/p3-de-t4-20261102/final-checkpoint/campaigns/).
Các bảng LaTeX, macro số liệu, PDF/SVG và ledger nằm trong
[report](../../results/p3-de-t4-20261102/final-checkpoint/report/).
Các figure sau rà soát bố cục nằm trong
[reviewed-report](../../results/p3-de-t4-20261102/reviewed-report/);
chỉ xuống dòng tiêu đề curve HPatches để tránh bị cắt, không đổi số đo.
`summary.json` của bản rà soát vẫn khớp checkpoint T4.
Dataset và output không được commit; đã thêm ignore riêng trong cấu hình Git local.

Lệnh tái sinh báo cáo từ checkpoint, chạy tại root repo:

```bash
python -m tools.controller.p3_report \
  --udis results/p3-de-t4-20261102/final-checkpoint/campaigns/udis \
  --hpatches results/p3-de-t4-20261102/final-checkpoint/campaigns/hpatches \
  --hpatches-manifest results/p3-de-t4-20261102/final-checkpoint/data/hpatches/pairs.json \
  --output results/p3-de-t4-20261102/recomputed-report
```

## 8. Draft và bước bàn giao

Draft tiếng Anh cho Section 5.2 ở
[controller-evaluation.tex](../../paper/sections/controller-evaluation.tex),
Section 6 ở [controller-discussion.tex](../../paper/sections/controller-discussion.tex).
Các số liệu trong draft được nạp từ LaTeX sinh bởi code, không chép vào source.
Hai file là draft riêng để Nhóm tích hợp ở Phase 4; chưa thay bố cục manuscript chung.

PDF và receipt của draft ở
[paper-checkpoint](../../results/p3-de-t4-20261102/paper-checkpoint/).
Khi thử build manuscript chung bằng `paper/build.py`, compiler dừng vì
`paper/sections/pipeline.tex` gọi `sections/pipeline-s6.tex` nhưng file đích
không có. Đã xác nhận lỗi này có ở source `98a44e2` trước task hiện tại.
Log giữ trong `paper-checkpoint/main/`. Không sửa thay nội dung S6 hoặc bỏ
dòng input để làm build pass; Nhóm cần khôi phục phần này trước khi tích hợp
full draft. Việc kiểm chứng draft riêng của Đế vẫn pass.

Phần đo, công cụ phân tích và draft đã có artifact để review. Việc review
Phase 3 của Nhóm vẫn phải đánh giá kết quả âm của Controller và giới hạn
chất lượng; không tự coi số đo A2 là chứng nhận research question toàn hệ thống.
