# Quyết định: Trace và campaign T4 cho Controller

- **Người quyết định:** Đế ủy quyền agent thực hiện các task Phase 3.
- **Ngày:** 01/11 (Chủ nhật), năm 2026.
- **Phase:** Phase 3; deadline chung 05/11 (Thứ 5).
- **Trạng thái:** Đã build instrumentation trên Colab Tesla T4; chuẩn bị đo.

## Bối cảnh

Bridge MCP đã khôi phục quyền đọc và chạy notebook. Runtime có Tesla T4,
CUDA, CMake và compute-sanitizer. Bản gốc đạt 15/15 CTest; bản có trace đạt
16/16 CTest. Bộ Python local đạt 381 test. Đây là kiểm chứng code, chưa phải
bằng chứng hiệu quả Controller.

## Phương án chọn

Giữ nguyên ladder, gradient rules, self-check và cấu hình Fixed của harness
trên main. Seed 20261101, 2 warmup và 10 repeat, feather; không tune trên
test. Trace lưu trong array host cố định 16 attempt; ghi CSV sau khi kết thúc
lượt chạy. Có thêm chi phí ghi struct trong vùng đo; cả hai biến thể A2 dùng
cùng instrumentation. Không thêm allocation hoặc file I/O vào vùng đo.

`--no-reuse` chỉ thay đường R-only retry: chạy lại S4 và reset S5 với
previous_hypotheses=0. Descriptor và buffer đã cấp phát được giữ. Chọn cách
này để đo phần tính toán bị chạy lại, không gộp cudaMalloc/cudaFree vào lợi
ích của reuse. Phương án giải phóng/cấp phát lại buffer như đặc tả A2 sẽ trộn
chi phí allocator với chi phí S4/S5; chưa dùng. Báo cáo phải nêu rõ giới hạn này.
Đầu ra H, số match/inlier và path phải khớp giữa hai biến thể trước khi phân
tích latency. Chỉ dùng transition R-only thực sự xảy ra trong Controller.

Campaign theo batch 32 pair; xen kẽ thứ tự các mode theo batch để giảm bias
do drift. Checkpoint có hash binary, manifest, ảnh, seed và GPU UUID. Mỗi
mode vẫn dùng 2 warmup/10 repeat cho từng pair; CUDA event đo stage, host
steady_clock đo attempt và end-to-end. Không chạy hai campaign GPU song song.

HPatches có ảnh A của v_talent rộng 2456 px. Harness khảo sát kích thước ảnh
trước vùng đo và cấp phát đủ cho batch, không resize hoặc thay H_gt. Kiểm tra
giới hạn input trước upload và giới hạn canvas 4096 trước render; khi canvas
vượt buffer thì báo INSUFFICIENT_WORKSPACE ở S6. MCE vẫn kiểm được H nếu S5
hợp lệ; lỗi renderer được đếm riêng. Phương án resize bị loại vì thay đổi bài
đo và cần đổi hệ tọa độ ground truth.

## Dữ liệu và fallback

Archive testing.zip hiện tại trên Google Drive của tác giả thiếu
input1/000001.jpg. Khôi phục từ repo Multi-Grid-Deep-Homography của cùng tác
giả, commit 65422f3e012c1dd63fff89b66d8a9b80a8acbec6. input2/000001.jpg
trong repo này khớp từng byte với archive. SHA-256 ảnh A khôi phục:
`bbf99e1817f20f706e3fa2b68f2adaca0ae49dbb8f9a2ccf2410ebbc878f0d18`.
Lưu recovery receipt, hash archive và ảnh; chuẩn bị đủ 1106 pair trước khi chạy.
HPatches giữ đủ 59 viewpoint sequence, 295 pair, ảnh 1 so với ảnh 2–6.

Nếu không có R-only retry, ghi A2 cohort rỗng; không thay policy để tạo retry.
Nếu hai biến thể khác H/path thì dừng phân tích A2 để tìm lỗi. Failure không
bị loại khỏi denominator của cohort; không gọi self-check pass là QoR pass.

## Nguồn

- https://github.com/nie-lang/UnsupervisedDeepImageStitching — nguồn link
  Google Drive testing.zip; truy cập 01/11 (Chủ nhật), năm 2026.
- https://github.com/nie-lang/Multi-Grid-Deep-Homography/tree/65422f3e012c1dd63fff89b66d8a9b80a8acbec6/UDIS-D/testing
  — khôi phục ảnh thiếu; truy cập cùng ngày.
- https://github.com/hpatches/hpatches-dataset — nguồn full sequences trên
  Hugging Face; truy cập cùng ngày. Quy ước chiều H theo conventions.md.
