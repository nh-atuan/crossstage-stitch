# Quyết định: Kiểm chứng Controller Phase 3

- **Task:** Phân tích Controller, Ablation A2, HPatches và draft Section 5.2/6.
- **Người quyết định:** Đế ủy quyền agent quyết định và thực hiện trong task.
- **Ngày ghi nhận:** 01/11 (Chủ nhật), năm 2026.
- **Phase:** Phase 3; deadline chung 05/11 (Thứ 5).
- **Trạng thái:** Triển khai công cụ Python; phép đo Colab T4 bị chặn bởi kết nối điều khiển.

## 1. Bối cảnh

CSV của harness hiện lưu kết quả cuối lượt chạy, không lưu chuỗi attempt.
Không thể suy ra số retry hoặc inliers phục hồi từ cấu hình cuối vì probe có
thể chọn trực tiếp nấc đó. Dữ liệu có sẵn ở máy là development, không phải
UDIS-D test. Không dùng dữ liệu này thay cho nghiệm thu Phase 3.

MCP Colab trả thành công khi mở kết nối, nhưng không cung cấp công cụ chạy
cell. Browser không có trình duyệt khả dụng. Máy local không có CUDA,
CMake, trình biên dịch C++ hoặc LaTeX trong PATH. Theo mục 3.4 của AGENTS.md,
chưa bắt đầu sửa CUDA/harness khi chưa có môi trường kiểm chứng tương ứng.

## 2. Phương án chọn

Tách công cụ đọc CSV, phân tích telemetry, phân tích A2, kiểm chứng HPatches
và chạy campaign có checkpoint thành các module Python. Dùng numpy,
opencv-python và pytest đã có; không thêm dependency. Các phân tích từ chối
duplicate, repeat thiếu và dữ liệu không hữu hạn thay vì âm thầm loại bỏ.

MCE dùng bốn tâm pixel góc của ảnh B: (0,0), (w-1,0), (w-1,h-1), (0,h-1).
So sánh H của pipeline với inverse của H_1_i trên các góc đó, trong hệ ảnh
gốc. Báo riêng failure, trung bình theo pair, theo sequence và phân phối
theo pair. MCE của tập thành công không đại diện cho toàn bộ cohort. So sánh
hai chế độ thêm trên tập giao các pair có hình học hợp lệ.

A2 cần cùng pair, seed, S, K, R trước/sau và repeat ở hai biến thể. Chỉ xét
transition tăng R và giữ S/K; kiểm H, số match, số inlier và trạng thái để
phát hiện thay đổi kết quả do reuse. Lấy median repeat trước khi tổng hợp
theo pair/transition. Không suy ra A2 từ fixed-grid sweep cũ.

Trace attempt là đầu vào riêng, chưa phải output được harness hỗ trợ. Khi
thiếu trace, chỉ báo cấu hình cuối và trạng thái; retry và phục hồi để null.
Không gắn nhãn parallax từ residual/inliers; cần annotation hoặc ảnh kiểm tra.

Campaign gọi harness hiện có theo từng pair/chế độ, 2 warmup và 10 repeat,
seed 20261101. Lưu receipt gồm hash binary, input, manifest và source chứa
policy. Chỉ reuse checkpoint khi hash và schema khớp; không chép kết quả
development thành test. Seed và ngưỡng không được tune trên final.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Hạn chế và lựa chọn |
|---|---|---|
| Đo từng pair, checkpoint riêng | Khôi phục được sau lỗi; kiểm được dữ liệu thiếu | Có chi phí khởi tạo ngoài vùng đo mỗi pair; chọn cho campaign dài |
| Một tiến trình cho toàn bộ cohort | Ít khởi tạo | CSV dở dang khó phân biệt với run hoàn chỉnh; chưa chọn |
| Suy retry từ cấu hình cuối | Không cần trace mới | Sai khi probe chọn nấc cao; loại bỏ |
| Sửa harness ngay trên máy local | Có thể viết instrumentation sớm | Thiếu CUDA để kiểm; hoãn đến khi truy cập được T4 |

## 4. Fallback

Không điều khiển được Colab: hoàn tất kiểm thử host và bàn giao lệnh chạy;
giữ trạng thái chưa nghiệm thu GPU. Không restart, disconnect runtime, tạo
keepalive giả hoặc thay runtime T4 bằng CPU để báo số đo chính thức.
Checkpoint nên nằm trên Drive đã mount sẵn; công cụ không tự mount hoặc
unmount Drive. Không thể bảo đảm Colab không tự thu hồi runtime.

## 5. Kiểm chứng

Kết quả kiểm thử và phạm vi còn thiếu được ghi trong
[bàn giao Phase 3](../reports/p3-de-handoff.md). Test tổng hợp xác nhận phép
tính và xử lý failure, không phải kết quả benchmark ảnh thật.

## 6. Nguồn

- https://github.com/hpatches/hpatches-dataset — truy cập 01/11 (Chủ nhật),
  năm 2026; nhận diện nguồn dataset. Quy ước chiều H theo conventions.md.
- https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing — truy cập
  01/11 (Chủ nhật), năm 2026; rà soát văn phong, bỏ khẳng định thiếu số liệu.
