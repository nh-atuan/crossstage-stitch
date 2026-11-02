# Quyết định: Bảo đảm device RGB sẵn sàng trước vùng đo

- **Người quyết định:** Agent xử lý lỗi correctness trong task Phase 3 do Đế giao.
- **Ngày:** 03/11 (Thứ 3), năm 2026.
- **Phase:** Phase 3; deadline chung 05/11 (Thứ 5).

## Bối cảnh và bằng chứng

Campaign A2 dùng binary từ `98a44e2` có một warmup đầu của cặp `000424`
khác nấc khởi đầu giữa hai biến thể. Kết quả cuối của các repeat đo khớp,
nhưng kiểm tra path đầy đủ đã dừng phân tích. Giữ nguyên dữ liệu này làm
diagnostic; chưa dùng để kết luận A2.

Rà code thấy `upload_pair` dùng `cudaMemcpy2D` trên default stream, trong
khi pipeline dùng một non-blocking stream. CUDA không bảo đảm mọi H2D từ
pageable memory đã hoàn tất khi dạng API này trả về. Vì vậy code chưa bảo
đảm contract RGB device-ready trước khi bắt đầu `t_e2e`. Đây là lỗi về
đồng bộ cần sửa; riêng quan hệ nhân quả với warmup trên còn phải kiểm lại.

## Phương án chọn

Đưa cả hai H2D copy lên stream của pipeline bằng `cudaMemcpy2DAsync`, sau đó
`cudaStreamSynchronize` trước khi `upload_pair` trả về. Việc này diễn ra
ngoài vùng đo như contract đã quy định. Không đổi ảnh, probe, ladder,
threshold hoặc seed. Không thêm allocation trong retry.

| Phương án | Đánh giá |
|---|---|
| Copy cùng stream và chờ hoàn tất; chọn | Thiết lập thứ tự rõ ràng, chỉ chờ stream cần dùng; bảo đảm đầu vào cho cả warmup và repeat đo |
| Chờ toàn device | Có thể bảo đảm dữ liệu sẵn sàng nhưng đồng bộ rộng hơn cần thiết |
| Event nối default stream với pipeline stream | Thể hiện dependency nhưng vẫn phải thiết lập ranh giới host device-ready; thêm trạng thái không cần thiết ở đường upload hiện tại |
| Chỉ bỏ warmup khác path khỏi phân tích | Không sửa bảo đảm đầu vào; chưa chọn |

## Kiểm chứng và fallback

Thêm test GPU đổi liên tục giữa ảnh phẳng và ảnh có texture, kiểm ngay
quyết định probe đầu tiên, không dùng warmup để che dữ liệu chưa sẵn sàng.
Sau build, chạy lại CTest, sanitizer và campaign với source/binary hash mới.
Đối sánh path A2 vẫn kiểm cả warmup và repeat đo. Nếu còn khác biệt, tiếp
tục tìm nguyên nhân; không nới điều kiện kiểm tra hoặc sửa raw CSV.

## Nguồn

- [NVIDIA CUDA Runtime API: API Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html),
  truy cập 03/11 (Thứ 3), năm 2026; semantics của synchronous H2D và default stream.
- [Quy ước đo thời gian](../conventions.md): RGB đã ở device trước vùng `t_e2e`.
