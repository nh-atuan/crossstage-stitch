# Quyết định: quantum catalogue offline, Thành chủ trì và baseline chạy song song

- **Task:** Workstream quantum offline Phase 3; liên kết [plan](../plan.md) và [Q00–Q13](../research/quantum-integration-tasks.md).
- **Người quyết định:** Người dùng chốt hướng, phân công Thành và yêu cầu cập nhật/push trực tiếp `main`; Thành chủ trì triển khai, owner stage giữ quyền review phần mình.
- **Ngày ghi nhận:** 02/11 (Thứ 2), năm 2026.
- **Phase:** 3; deadline chung 05/11 (Thứ 5), không thêm deadline task riêng.
- **Trạng thái:** Đã chốt kiến trúc và phân công; code/instance/QAOA experiment chưa triển khai trong lượt cập nhật tài liệu.

## 1. Bối cảnh

Pipeline S1–S6 và controller đã được ghép, Phase 2 được Nhóm nghiệm thu PASS. Câu hỏi nghiên cứu vẫn là giảm latency bằng phân bổ budget dưới QoR. Proposal quantum trước đây mới là nghiên cứu, có phân công đề xuất cho Chương và chưa nằm trong bảng task Phase 3. Người dùng nay chốt bộ tối ưu catalogue offline, giao quantum cho Thành và yêu cầu nhánh không quantum tiếp tục song song.

Catalogue có thể làm controller chọn S/K/R khác nhau, nên có ảnh hưởng gián tiếp tới workload và QoR; việc không sửa kernel không phải bảo đảm chất lượng. Chỉ có circuit toy hoặc QUBO được kiểm bằng cổ điển chưa chứng minh quantum đã giải bài toán phát sinh từ pipeline.

## 2. Phương án chọn

Thành chủ trì Q00–Q13: adapter characterization, native cover, classical controls, QUBO/Ising, exact reductions, validator/repair, môi trường riêng, QAOA simulator p=1,2 và báo cáo. Catalogue fit từ CUDA development sweep đã kiểm QoR, B cố định; validation chọn tham số, test chỉ đánh giá sau khóa protocol. Tối thiểu một instance thật còn biến sau reduction phải có circuit/counts/logs. Các tham số rho, solver budget, quality threshold và split cụ thể vẫn cần owner review.

Giữ ba nhóm: W0 runtime gốc không quantum; W1 catalogue do classical solver; W2 catalogue do QAOA simulator. W0 và A1–A5 không chờ quantum dependencies. W1/W2 online chỉ nhận JSON đã validate, dùng consumer cổ điển cùng selector family/tuning budget, self-check/reuse/fallback. Consumer thuộc Đế, harness thuộc Tuấn; Thành xuất schema/artifact, Chương review QoR/B và tổng hợp paper. Không đổi ABI hoặc thuật toán S1–S6, không gọi QPU trong đường chạy từng ảnh, không đưa quantum SDK vào requirements/CMake của pipeline.

QAOA và controls chạy trên CPU simulator riêng; không tranh tài nguyên với campaign T4 đang đo. Chưa có consumer thì chỉ kết luận về solver offline, không suy speedup runtime. Nhánh đầu không nhân theo toàn lưới A5. A3 của Thành và các owner A1–A5 không đổi.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Nhược điểm | Quyết định |
|---|---|---|---|
| QUBO/QAOA catalogue offline | Tái dùng sweep, không đổi kernels, có exact controls | Slack/qubit/penalty lớn; catalogue có thể trivial; chưa bảo đảm lợi ích | Chọn primary, simulator có budget và validator gốc |
| Classical catalogue | Baseline mạnh, deploy đơn giản, không cần quantum SDK | Không đáp ứng riêng yêu cầu thí nghiệm quantum | Bắt buộc chạy đối chứng, không đổi nhãn thành quantum |
| Quantum selector online | Gần quyết định từng ảnh | Thêm training/inference/embedding và rủi ro latency | Không chọn trong MVP |
| Quantum S5/S6 | Gắn với estimator hoặc seam | Đổi thuật toán/metric/reuse, tăng phạm vi và ownership | Không chọn |
| QPU ngay | Có hardware evidence | Quyền truy cập, queue, chi phí và noise chưa xác nhận | Chỉ định hướng tùy chọn sau prototype |

## 4. Fallback

Quantum lỗi hoặc chưa có dữ liệu không chặn W0/W1; không xuất greedy catalogue mang nhãn quantum. Zero raw feasible samples là negative result chỉ khi execution/counts/controls đầy đủ. Không có circuit hoặc không có instance từ sweep thì quantum vẫn chưa hoàn thành. Nếu consumer chưa có, giữ catalogue validated dưới dạng artifact và phần runtime pending. Nếu mọi instance bị giải hết bởi exact reduction hoặc vượt tài nguyên, báo giới hạn, không tạo độ khó giả hoặc bí mật cắt instance. Classical thắng không làm mất yêu cầu prototype, cũng không cho phép claim advantage.

## 5. Kiểm chứng

Lượt này kiểm tra đồng bộ tài liệu, dependency/owner, liên kết tương đối và Git diff; không cài dependency, thay code CUDA, chạy solver hoặc dựng số liệu. Trước implementation, Thành cần khóa environment/license, tiny energy/Ising/circuit tests và resource admission theo task. Bản kế hoạch được commit/push trực tiếp `main` theo yêu cầu hiện tại; các commit implementation tương lai theo quy tắc nhóm.

## 6. Nguồn

- [Lucas, 2014, mục 5.1 Set cover](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full): penalty/Ising cho set cover; không là benchmark stitching. Truy cập 02/11 (Thứ 2).
- [Farhi, Goldstone và Gutmann, 2014](https://arxiv.org/abs/1411.4028): metadata/abstract về QAOA cho tối ưu tổ hợp. Truy cập 02/11 (Thứ 2); không lấy bảo đảm MaxCut làm bảo đảm catalogue.
- [IBM QAOA tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm): Background và mapping binary/QUBO/Ising sang circuit, vòng lặp hybrid. Truy cập 02/11 (Thứ 2); không dùng thời gian demo làm estimate cho dự án.
- [Proposal và phạm vi đọc nguồn lịch sử](../research/quantum-integration-proposal.md): formulation, baseline, resource budget và giới hạn claim chi tiết.

## 7. Chốt phạm vi bản nộp ngày 07/11 (Thứ 7)

Người dùng chốt quantum là phần bắt buộc của bài toán nghiên cứu và của bản nộp. Thành chủ trì bằng chứng và draft quantum, Chương tích hợp paper, Đế review QoR/split và Tuấn review provenance/hạ tầng đo.

Pilot trên sweep CUDA cho thấy QAOA simulator và đối chứng cổ điển tìm được catalogue cùng kích thước trên instance chính. Bài báo giữ câu hỏi chính về adaptive runtime và đánh giá thêm bài toán con chọn catalogue offline. Phần quantum có formulation, thí nghiệm so solver và giới hạn trong thân bài. `G-PAPER` là điều kiện khóa bản nộp, gồm gói bằng chứng đã audit, phạm vi claim được duyệt và PDF có nội dung này. [Task và tiêu chí nghiệm thu](../research/quantum-integration-tasks.md#8-phase-4--bàn-giao-quantum-cho-paper)

Pilot offline dùng dữ liệu từ pipeline và cho phép so solver trên cùng instance. Đo W0/W1/W2 end-to-end có thể trả lời câu hỏi về tác động lên runtime, nhưng cần consumer, protocol và campaign riêng. Theo luật dừng Q13, nhóm ưu tiên kiểm chứng và bàn giao thí nghiệm hiện tại. QPU và mở rộng depth được xem xét theo một quyết định sau.

Nếu holdout chưa có, nhóm có thể duyệt riêng việc báo pilot calibration ở phạm vi thăm dò và ghi holdout `not_measured`. `G-PAPER` vẫn cần run QAOA thật, controls và khả năng tái lập. Nếu bằng chứng chưa đối chiếu được hoặc phạm vi còn chờ duyệt, ghi `BLOCKED` cho bản nộp và khôi phục bằng chứng hoặc lập run mới theo protocol đã duyệt.
