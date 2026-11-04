# Quyết định: feature và gate học từ QoR

- Người quyết định: Đế, yêu cầu triển khai trong phiên làm việc.
- Ngày: 04/11 (Thứ 4), năm 2026.
- Phase 3, deadline chung 05/11 (Thứ 5).
- Trạng thái: Đã triển khai và đo trên Colab T4; chưa đạt lợi thế latency
  dưới cùng QoR. Xem [báo cáo kết quả](../reports/controller-learned-features.md).

## Bối cảnh và lựa chọn

Policy v2 chọn reference cho mọi pair validation. Gradient đơn chưa mô tả
quan hệ hai ảnh; các ngưỡng S5 đơn điệu bỏ lỡ nhiều positive QoR. Replay
chi phí cũ chưa tính đúng transition có reuse. Đế chấp thuận bổ sung feature,
học cây nhỏ, điều chỉnh gate theo nhãn QoR và đo transition trên T4.

Giữ nguyên predicate SSIM/PSNR/coverage pilot. Feature online không đọc
reference hoặc nhãn Oracle. Module Controller riêng thu thống kê grayscale
theo vùng của cả hai ảnh; sau S5 thu phân bố inlier A/B, hình học H và sai
khác photometric trên lưới nhỏ. Phân biệt inlier coverage với QoR coverage.
Không sửa thuật toán hoặc tham số S1–S6 để tạo nhãn dễ hơn.

Cây có giới hạn độ sâu và số mẫu ở lá, export thành mảng node C++ có kiểm
schema, finite và chỉ số. Không thêm thư viện inference runtime. Gate được
kiểm bằng dự đoán ngoài fold, chia theo pair/component; repeat và cấu hình
cùng pair không nằm ở hai fold. Báo false accept/false reject và độ phủ,
không suy đảm bảo QoR dữ liệu mới từ số lỗi bằng zero trên tập học.

Lựa chọn ban đầu học từ chi phí các cấu hình đạt gate, với nhãn QoR và
latency sweep calibration. Đo chuỗi đổi R, K, S để hiệu chỉnh chi phí dựa
trên stage cần chạy lại và R tăng thêm; chi phí trích feature tính trong
latency adaptive. Policy có fallback reference và retry hữu hạn.
Học đích retry tiến về phía sau theo chi phí trên calibration, không buộc
đi qua mọi nấc còn lại. Kiểm target tiến để không tạo chu trình.

Khóa ba biến thể trước holdout: gate không có false accept ngoài fold;
hai biến thể diagnostic dùng precision ngoài fold tối thiểu 0,90 và 0,80.
Hai mức này chỉ là ràng buộc học gate, không thay ngưỡng QoR và không được
coi là bảo đảm precision trên dữ liệu mới. Báo mọi biến thể; chỉ khuyến
nghị thay baseline khi phép đo không mất các pair QoR PASS của reference.
Loss học bộ chọn tính thêm chi phí reference cho false stop đã biết trên
calibration; đây là penalty offline, không phải phép phát hiện QoR online.

## Dữ liệu và nghiệm thu

Giữ nguyên 200 pair calibration của v2. Tập 50 pair validation cũ chỉ dùng
đối chiếu vì đã được xem trong quá trình thảo luận. Khóa thêm 50 component
training UDIS-D chưa thuộc pool cũ trước phép đo policy mới, kiểm trùng ảnh
bằng toàn bộ index ZIP CRC/size. Đây chưa phải chứng nhận scene disjointness
theo nhãn scene chính thức. Không tune trên UDIS-D test hoặc HPatches.

Chọn feature, độ phức tạp và gate bằng calibration/cross-validation. Freeze
model và protocol trước tập kiểm tra mới. Báo cả failure/reference unknown;
so latency trên cùng cohort QoR PASS và ghi rõ tính có điều kiện. Nếu không
có cải thiện, giữ kết quả âm và fallback; không nới ngưỡng QoR.

## Trade-off

Feature nhiều hơn có thể phân biệt được những trường hợp S5 metrics giống
nhau nhưng QoR khác; overhead có thể xóa lợi ích cấu hình rẻ. Cây nhỏ dễ
đọc và kiểm parity CPU/C++, nhưng có thể bỏ lỡ quan hệ phức tạp. Chỉ nới
ngưỡng v2 ít tốn công hơn, song phân tích calibration cho thấy tăng nhận
đúng đồng thời tăng nhận nhầm; không chọn cách nới hàng loạt.

Giữ baseline v1/v2 và checkpoint cũ riêng. Build `sm_75`, CTest, Python,
kiểm feature CUDA bằng fixture có đáp án và đo T4 trước khi bàn giao.

Nguồn phương pháp cây và giới hạn overfit: tài liệu chính thức
https://scikit-learn.org/stable/modules/tree.html, đọc 04/11 (Thứ 4).
Đây là tham khảo thuật toán; không sao chép code hay thêm dependency.
Đã đọc và rà văn phong theo
https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing cùng ngày.
