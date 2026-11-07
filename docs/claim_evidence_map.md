# Claim–evidence map cho bàn giao CH-07/08

Mốc rà soát 07/11 (Thứ 7). Áp dụng cho phần CH-02 `legacy-v1`, A4 và CH-05 của Chương. Các claim của A1, A2, A3, HPatches và quantum do owner bàn giao theo inventory riêng.

Mọi macro định lượng dưới đây được sinh bởi `tools/analysis/paper_tables.py`. Giá trị và query đầy đủ nằm trong `claims.json` và `claim_evidence_map.md` của bundle. `artifact_manifest.json` khóa source CSV, export receipt, code và output. [Lệnh tái tạo](runbooks/ch07_ch08_paper_artifacts.md)

| Vị trí manuscript | Claim hoặc metric | Artifact và truy vấn | Trạng thái |
|---|---|---|---|
| Abstract và Held-Out Quality | Mẫu số danh nghĩa, execution OK, QoR PASS và tỷ lệ PASS từng mode | `report/pair_ledger.csv`, group `mode`, đếm disposition. Đối chiếu `report/report.json`. | PASS khi sinh và verify bundle thành công |
| Held-Out Quality | Speedup có điều kiện và khoảng cách tới Oracle | Pair ledger, chỉ lấy cặp cùng PASS, median của tỷ số latency theo cặp. Macro `ChSpeedup`, `ChOracleGap`. | PASS trên giao PASS, full-cohort speedup giữ undefined |
| Hình Pareto | Latency và QoR PASS của grid | `joined/joined.csv`, median repeat rồi median theo cặp. Tái tạo frontier khớp `report/pareto_cohort.csv`. | PASS trên grid đã đo |
| Hình CDF | Median và p95 trên các tập PASS riêng | `report/latency_cdf.csv`, kiểm lại ECDF từ pair ledger. | PASS, caption phân biệt population |
| Hình timing | S1–S6 và host total trên common-pass cohort | Joined CSV và ledger. Chọn H/config Oracle đã khóa rồi median theo repeat và pair. | PASS, không cộng median hoặc khoảng chồng lấp |
| A4 | Render latency, workspace query, canvas throughput | `a4-tables/summary.csv`, hash đã pin và kiểm raw `a4_same_h_metrics.csv`. | PASS trên cohort development A4 |
| Hình qualitative | Bbox và RGB chung giữa ba blender | `a4-visuals/visual_manifest.json`, PNG có hash, `sources/visual_plan.json`. Crop limitation là hậu kiểm. | PASS cho truy vết và quan sát định tính |
| Scaling | Completion, failure, OOM, timeout, GPU interval và queried workspace | Hai `ch05-*-report/summary.csv`, CSV timing tương ứng. Giữ hai track riêng. | PASS trong phạm vi từng track |
| Scaling và Discussion | Giới hạn registration và corner error | `ch05-end_to_end-report/summary.csv`, giữ case lỗi trong mẫu số. | PASS cho kết quả âm của cấu hình đã đo |
| Cross-architecture | Chuyển workload T4 sang L4/A100 | Bằng chứng CH-06 do Tuấn chạy trên L4 và A100 qua notebook `phase3_ch06_cross_arch.ipynb`, cùng manifest và timing boundary. Đủ 54/54 case mỗi kiến trúc, workspace khớp bitwise. | PASS |
| A5 test | Độ nhạy policy trên test | Cần A5 lock đạt gate development/validation. Hiện đã ghi `NO_GO`. | BLOCKED |
| Runtime sau merge | Suy rộng kết quả T4 cũ sang binary hiện tại | Cần smoke/sanitizer trên T4 cho revision tích hợp. | BLOCKED |

## Rà soát manuscript

Claim định lượng trong abstract dùng cùng macro với evaluation. Tốc độ trên giao PASS được đặt cạnh tỷ lệ QoR toàn cohort. Hình Pareto dùng số PASS theo đúng schema hiện có. Alignment metric chưa có vai trò đo chất lượng seam. Known-H render scaling và estimated-H registration có đoạn kết luận riêng.

Các mục còn mở trong `claims.json` giữ trạng thái BLOCKED. Mẫu số, timing boundary, phạm vi hardware và việc chọn crop hậu kiểm phải còn trong caption khi nhóm chỉnh bài. Kiểm font và metadata bằng tool tự động đi cùng review tất cả trang render. Nghiệm thu CH-07/08 không thay cho review phương pháp và bằng chứng của owner khác.
