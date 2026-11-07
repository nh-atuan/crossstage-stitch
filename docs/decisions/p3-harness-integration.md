# Đồng bộ harness trước bàn giao CH-07/08

## Bối cảnh và quyết định

Ngày 07/11 (Thứ 7), người dùng yêu cầu đồng bộ `origin/main` trước khi tích hợp manuscript. Nhánh Chương có timing, trace lỗi, policy v2 và capacity guards. `main` bổ sung policy v3/v4, learned features, pooled routing và đối chứng A2 `--no-reuse`.

Chọn kết hợp hai hợp đồng trong cùng harness. Policy v2 giữ validation và provenance seed. Policy v3/v4 giữ model và feature layout của Đế. Trace xuất thêm feature columns sau các trường hiện có. Cả hai tên CTest policy loader tiếp tục được nhận diện. Workspace query tính thêm vùng features và các executable sử dụng runner liên kết feature collectors.

## Trade-off và kiểm chứng

Giữ một runner chung tránh hai implementation có timing khác nhau. Đổi lại, logic learned inference phải đi qua cơ chế timing và failure trace của nhánh Chương. Chi phí feature collection được ghi riêng trong `t_features`, nằm trong host invocation. `t_controller` đo lời gọi CPU controller. Khoảng stream wait có thể chồng với GPU intervals và không được cộng vào chúng để suy ra tổng latency.

Chọn explicit `--controller-policy` như `main` để định danh controller mỗi campaign. CLI seed ghi đè grid chỉ khi có đối số `--seed`. Seed trong policy được đối chiếu trước chạy. Đối chứng `--no-reuse` chạy lại S4 và S5 cho R-only retry, giữ các stage khác khi còn hợp lệ.

Thay toàn bộ runner bằng một phía sẽ làm mất hợp đồng của phía kia. Phương án tách hai executable có thể bảo toàn từng snapshot, nhưng tăng rủi ro so sánh sai timing trong campaign tiếp theo.

Compile/link `sm_75` và CTest host-only kiểm tính tích hợp. Campaign đã nghiệm thu tiếp tục gắn với binary và source trong export gốc. Revision sau đồng bộ cần smoke/sanitizer trên T4 trước một campaign GPU mới. CH-07/08 đọc bằng chứng T4 đã khóa và chạy trên CPU.
