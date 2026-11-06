# Quyết định: mô hình dùng chung cho cấu hình Controller

- Người quyết định: Đế, yêu cầu triển khai trong phiên làm việc.
- Ngày: 04/11 (Thứ 4), năm 2026.
- Phần cải tiến bổ sung của Phase 3; deadline phase 05/11 (Thứ 5).
- Trạng thái: đang triển khai, chưa có kết quả nghiệm thu.

## Bối cảnh và phương án chọn

Policy trước loại phần lớn cấu hình K500 vì gate riêng chưa phân biệt được
QoR. Đế chấp thuận học mô hình dùng chung, thử XGBoost cây nông, bổ sung
feature quan hệ hai ảnh và kiểm chi phí Controller đầy đủ trên T4.

Đầu vào trước chạy gồm feature scene và S/K/R. Mô hình trước chạy dự đoán
QoR và latency; mô hình sau S5 nhận thêm feature matching/alignment của
candidate đã thực thi. Nhãn là predicate QoR chuẩn và latency đo thật;
không học một lớp duy nhất bằng ID của cấu hình Oracle. Không đọc reference
hoặc nhãn offline trong lúc chọn và kiểm candidate online.

Thử feature hiện có trước, rồi so từng nhóm bổ sung. Giữ feature giúp giảm
chi phí ngoài fold dưới ràng buộc lỗi chất lượng; chi phí trích feature và
inference nằm trong E2E. Tập cấu hình ứng viên được chọn từ calibration,
cho phép K500, không loại toàn bộ cấu hình chỉ vì gate riêng không đạt.
Fallback reference vẫn hữu hạn; không nới SSIM/PSNR/coverage.

Chia toàn bộ pipeline học theo component ảnh. Trong mỗi outer fold, các
bước fit mô hình, calibration xác suất, chọn ngưỡng và cấu hình chỉ được
đọc phần train. Dữ liệu dùng calibration xác suất tách khỏi dữ liệu fit
chính. Báo false accept, false reject, fallback và regret; precision hay
xác suất dự đoán không bảo đảm chất lượng trên dữ liệu mới.

Khóa 50 component training mới trước khi fit, loại mọi component thuộc
200 calibration và hai cohort validation đã xem. Giữ các reference failure;
không thay cặp khó. Không tune trên UDIS-D test hoặc HPatches. Kiểm CRC/size
toàn index ZIP chỉ phát hiện trùng ảnh, chưa chứng nhận scene disjointness.

## Trade-off và fallback

Boosted trees có thể học tương tác tốt hơn cây nhỏ cũ nhưng tăng nguy cơ
overfit trên 191 cặp reference hợp lệ. Dùng cây nông, regularization và số
vòng giới hạn. XGBoost chỉ là dependency huấn luyện; inference C++ dùng
mảng cây có giới hạn, kiểm parity với mô hình Python. Nếu không cải thiện,
giữ đối chứng và báo kết quả âm; không đổi mặc định sang policy chưa đạt.

Feature tương quan, match ambiguity, residual theo vùng và cấu trúc sau
warp có thể bổ sung thông tin nhưng tăng overhead. Chỉ thu feature khi có
đầu vào tương ứng, không đưa feature sau S5 vào selector trước chạy. Thử
mô hình hiện có feature trước để tránh quy mọi cải thiện cho thêm feature.

## Nguồn

- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html):
  mục tiêu phân loại/hồi quy, cây nông và regularization.
- [XGBoost Model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html):
  xuất cây và kiểm objective/base score khi triển khai inference.
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html):
  hiệu chỉnh xác suất bằng dữ liệu độc lập với phần fit mô hình.
- [SATzilla2012](https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/SATzilla2012final.pdf):
  tham khảo chọn thuật toán theo chi phí từng instance.

Các nguồn đọc 04/11 (Thứ 4); không sao chép code. Đã đọc và rà văn phong
theo [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing).
