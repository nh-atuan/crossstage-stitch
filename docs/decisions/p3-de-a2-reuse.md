# Quyết định: Tách reuse kết quả khỏi chi phí cấp phát trong A2

- **Người quyết định:** Đế giao agent đánh giá ưu, nhược điểm và chọn phương án trong phiên làm việc.
- **Ngày:** 03/11 (Thứ 3), năm 2026.
- **Phase:** Phase 3; deadline chung 05/11 (Thứ 5).
- **Phạm vi:** A2 và Controller hiện tại; không bổ sung policy variants cho A5.

## Bối cảnh

Đặc tả A2 trước đây yêu cầu No Reuse giải phóng buffer. Nhánh triển khai giữ
workspace đã cấp phát và chỉ chạy lại S4/S5. Đây là hai phép đo khác nhau:
giải phóng/cấp phát lại đưa thêm chi phí allocator vào thời gian retry.
Buffer là vùng nhớ; giữ vùng nhớ không đồng nghĩa giữ kết quả tính toán.

## Phương án chọn

Giữ workspace đã cấp phát trong cả hai biến thể. Với cùng input, seed, S và K,
khi chỉ tăng R:

- With Reuse giữ match S4 và raw search state S5, chỉ đánh giá phần R bổ sung.
- No Reuse chạy lại S4 và reset S5 để tìm từ giả thuyết đầu đến R mới.

Đường xử lý attempt đầu giống nhau. Không có retry thì A2 không tạo lợi ích
tránh tính lại; không suy ra No Reuse nhanh hơn ở attempt đầu. Đổi S hoặc K
vẫn vô hiệu hóa các kết quả tương ứng theo contract hiện có.

## Trade-off và phương án phụ

| Phương án | Ưu điểm | Hạn chế |
|---|---|---|
| Giữ workspace ở cả hai phía; chọn cho A2 | Tách lợi ích tránh tính lại S4/S5; phù hợp pipeline pre-allocation hiện tại | Không đo lợi ích tránh cấp phát; cần báo rõ phạm vi |
| Giải phóng/cấp phát lại ở No Reuse | Đo thêm tác động quản lý bộ nhớ | Trộn hai nguyên nhân thay đổi latency; kết quả phụ thuộc allocator và đồng bộ |
| Bỏ reuse trong Controller | Đơn giản hóa trạng thái cache | Tính lại khi chỉ tăng R; chưa có bằng chứng lợi hơn trong pipeline hiện tại |

Giữ workspace làm tăng thời gian chiếm dụng VRAM. Reuse còn cần kiểm input,
generation, seed, scale, solver settings và trạng thái hoàn tất GPU. Không
đánh đổi correctness lấy latency. Phương án allocator có thể là nghiên cứu
riêng sau này; chưa đưa vào số đo A2 hiện tại.

## Kiểm chứng và fallback

Hai biến thể phải có cùng path, trạng thái, số match/inlier và homography
trong sai số đã khóa trước khi so thời gian. Đo cả CUDA event S4+S5 và host
retry time; giữ số âm nếu reuse chậm hơn. Chỉ tổng hợp cohort có R-only retry
thật, không đổi policy trên test để tạo thêm retry. Cohort rỗng thì báo không
có ước lượng A2, không tự chuyển thành kết luận tăng tốc toàn bộ dataset.

## Nguồn

- [NVIDIA CUDA C++ Best Practices Guide, Allocation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#allocation),
  truy cập 03/11 (Thứ 3), năm 2026: chi phí cấp phát và giải phóng device memory.
- [Contract Controller](../stages/controller-contract.md): điều kiện hợp lệ của cache và invalidation.
- [Quyết định campaign](p3-de-campaign.md): hai đường thực thi A2 đã triển khai.
