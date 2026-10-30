# Quyết định: kiểm lại gate trên UDIS-D training

- Người quyết định: Đế, theo yêu cầu chọn phương án 1.
- Ngày ghi nhận: 30/10 (Thứ 6), năm 2026.
- Phase 2; deadline chung 31/10 (Thứ 7).
- Trạng thái: đã khóa cohort trước khi xem ảnh hoặc chạy pipeline; evidence T4 và [báo cáo gate](../p2-de-real-development-gate.md) đã hoàn thành. Gate toàn cohort chưa xác lập vì hai reference failures.

## 1. Bối cảnh

Synthetic pilot có Oracle chuẩn dưới 1.20×. Đế yêu cầu kiểm lại trên ảnh panorama thật, giữ nguyên canonical QoR và toàn bộ deliverable đã hoàn thành. Không sử dụng legacy speedup để kết luận GO, không nới threshold, không tune bằng final và không mở rộng nghiên cứu ngoài Phase 2.

## 2. Phương án chọn

Lấy ảnh từ `training.zip` trong folder UDIS-D được repository tác giả dẫn chiếu. File training có ID `1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a`, 1.945.998.335 byte. Chỉ đọc metadata và ảnh trong training; không tải testing archive hoặc áp dụng fallback test-as-development.

Chọn development pool 250 pair IDs và pilot 25 pairs đầu, seed 20261030. Metadata archive có 10.440 pairs. Để tránh chọn các pair chung ảnh, nối các pair có cùng `(CRC32, file_size)` ở bất kỳ phía input1/input2 thành component; có 1.789 components. Xếp component theo SHA-256 có seed rồi chọn một pair theo hash trong mỗi component. Đây là proxy exact-file từ archive, không phải nhãn scene chính thức; collision có thể gộp thừa. Không thể chứng nhận scene-disjoint với final khi publisher không cung cấp scene metadata.

Danh sách được khóa trong [cohort manifest](../cohorts/udis-d-development-v1.json). Script chọn chỉ đọc metadata, không nhận latency/QoR. Giữ đủ 25 pairs, không thay pair khó hoặc reference thất bại sau khi chạy. Nhận ảnh JPEG nguyên bản bằng HTTP ranges, kiểm ZIP CRC/size và ghi SHA-256; decode sang RGB8 PPM ở độ phân giải gốc. Không resize, biến đổi hoặc sinh ảnh tổng hợp.

Lưới, B=feather, Fixed-High, warm-up/repeats và seed RANSAC giữ như synthetic pilot: 48 cấu hình S×K×R; reference `S1.00_K2000_R1024_Bfeather`; 2 warm-up, 10 repeats, RANSAC seed 0. Canonical policy JSON giữ nguyên byte/hash. Không sửa Controller, S5 tests, S1/harness hoặc paper.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Hạn chế / quyết định |
|---|---|---|
| 25 pairs từ các training components khác nhau | Kịp Phase 2, không dùng final, giảm ảnh lặp trong pilot | Pilot nhỏ; chưa chứng minh tổng quát hóa; chọn |
| Quét đủ 250 pairs ngay | Cohort lớn hơn | Tăng thời gian trước deadline; chỉ khóa IDs, chưa chạy mở rộng |
| Test split hoặc ảnh tổng hợp fallback | Dễ lấy dữ liệu | Không phù hợp yêu cầu ảnh thật và giữ final; không dùng |
| Tăng grid, nới QoR hoặc tune Controller | Có thể thay kết quả | Thay câu hỏi thí nghiệm; không thực hiện |

## 4. Gate và fallback

Gate speedup dùng **tỷ số median E2E Fixed-Best / median E2E Oracle**, ngưỡng 1.20×, cùng canonical QoR và timing boundary. Báo thêm median tỷ số paired với tên riêng, không dùng hai cách tổng hợp thay thế nhau để vượt gate. Điều kiện stage chỉnh được dùng ngưỡng 30% theo plan.

Giữ mọi pipeline/reference failure và missing evidence trong mẫu số. Chỉ tính speedup toàn cohort khi Fixed-Best và Oracle đủ repeat/latency và đều pass QoR trên toàn bộ 25 pairs, theo audit hiện có. Không có baseline khả thi toàn cohort hoặc reference thiếu tin cậy thì gate chưa được xác lập; không gán speedup giả hoặc loại pairs để tuyên bố GO.

Nếu Oracle đo được dưới 1.20×, báo kết quả và đề xuất Nhóm chuyển No-Go/Pareto theo kế hoạch. Nếu gate không xác lập do failures, báo rõ coverage và nguyên nhân; không tự mở calibration để cứu speedup. Kể cả khi pilot đạt gate, numerical policy và campaign final vẫn cần review riêng.

## 5. Kiểm chứng

Test chọn cohort kiểm deterministic order, shared-image components, train-only boundary, counterpart và duplicate. Ảnh tải về phải khớp CRC/size của archive, có SHA-256 và decode hợp lệ; không được có ảnh trùng trong pilot. Trên Colab kiểm lại GPU là Tesla T4 và source/binary hashes trước grid. Sau chạy kiểm 25×48×10 measurements, canonical QoR và reference previews. Lưu campaign riêng ngoài Git, không sửa synthetic pilot.

## 6. Nguồn

- [Repository UDIS chính thức](https://github.com/nie-lang/UnsupervisedDeepImageStitching), mục Dataset; truy cập 30/10 (Thứ 6), năm 2026.
- [Folder UDIS-D được tác giả công bố](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY); chỉ dùng training.zip.
- [Plan](../plan.md), [benchmark/QoR](../benchmark-qor-s4-s5-decisions.md), [synthetic evidence](../p2-de-t4-validation.md).
