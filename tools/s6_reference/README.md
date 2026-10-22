# Reference FP64 cho S6

Đây là oracle CPU độc lập để kiểm tra `src/s6`, không phải backend chạy chính hoặc benchmark GPU. Ngày cập nhật: 22/10 (Thứ 5), năm 2026. Quy tắc chi tiết ở [bàn giao P1](../../docs/s6-p1-implementation.md) và [phương pháp](../../docs/s6-warp-blend.md).

## Chạy

Từ root repo, với dependency trong `tools/requirements.txt`:

```bash
python -m pytest tests/python/test_s6_reference.py -q
python -m tools.s6_reference.generate_fixtures --output data/s6-fixtures --include-mock
```

Mặc định có 24 fixture: identity kích thước lẻ, tịnh tiến âm/phân số, projective, hai ảnh tách rời, 1 pixel, 1 hàng và 1 cột; mỗi trường hợp chạy ba mode. `--include-mock` thêm 3 fixture từ cặp tổng hợp qua mock S1–S5 dùng ORB và H ước lượng. Không dùng dữ liệu giả làm kết quả HPatches hoặc kết quả CUDA.

```python
from tools.s6_reference.reference import render
result = render(a_rgb8, b_rgb8, H_B_to_A, mode="multiband_3")
# result.panorama: RGB8; result.mask: uint8 0/1
# result.geometry.origin_x/y: coordinates in A at canvas (0, 0)
```

Mode hợp lệ: `feather`, `multiband_3`, `multiband_5`. Số 3/5 là số REDUCE; cap ở 1×1. Tọa độ, màu và filter reference dùng FP64; canvas/mask phải khớp CUDA tuyệt đối, tolerance RGB ban đầu trong CUDA harness là max absolute ≤2, chưa được xác nhận trên GPU.

## Thiết kế và giới hạn

`geometry.py` kiểm tra H và tính canvas trên miền tâm pixel; `reference.py` tự cài bilinear, feather, pull–push extension, REDUCE/EXPAND reflect-101 và Laplacian reconstruction bằng NumPy. Không gọi OpenCV blender hoặc `warpPerspective` để tạo oracle S6. OpenCV chỉ dùng trong đường `--include-mock` để chạy các mock stage trước đó.

Pixel đen không quyết định validity; mask riêng có giá trị 0/1. Output dùng `floor(clamp(255*C)+0.5)`, không dùng bankers rounding của `np.rint`. Tọa độ sát số nguyên được snap theo policy. Đừng yêu cầu bit-identical `warpPerspective`: OpenCV có lựa chọn nội suy/border riêng, còn đây là bilinear tính trực tiếp và weight của dự án.

`max_reference_bytes` mặc định 512 MiB là admission guard theo ước lượng bảo thủ (192 byte/pixel feather, 768 byte/pixel multiband), không phải số đo peak RAM hay ngân sách VRAM. Dùng fixture nhỏ; oracle không thiết kế để sweep ảnh lớn trong benchmark chính.

## Định dạng fixture và tái lập

Mỗi `.bin` little-endian gồm lần lượt:

| Trường | Dữ liệu |
|---|---|
| Magic | 8 byte ASCII `S6GOLD01` |
| Geometry | 9 int32: wa, ha, wb, hb, wc, hc, origin_x, origin_y, mode |
| H | 9 float64 row-major, B → A gốc |
| Ảnh | RGB8 packed A, RGB8 packed B, RGB8 golden panorama |
| Mask | wc×hc byte, 0/1 |

Mode số lần lượt 0/1/2 cho feather/multiband3/multiband5. `manifest.txt` liệt kê tên fixture; `manifest.sha256` chứa hash; `meta.json` ghi Python/NumPy, OpenCV nếu dùng mock và hash mã nguồn oracle. Pin dependency/commit khi so kết quả giữa các máy. Chạy lặp trong cùng môi trường phải giữ hash fixture; không yêu cầu ORB có cùng H giữa mọi phiên bản OpenCV.

```bash
(cd data/s6-fixtures && sha256sum -c manifest.sha256)
build/s6/test_s6_cuda --golden data/s6-fixtures/*.bin
```

Lệnh cuối cần build CUDA và device thật; xem hướng dẫn T4 trong tài liệu bàn giao. Không có CUDA device được test đánh dấu skip, không tính là GPU pass. Dữ liệu sinh trong `data/` đã được gitignore; commit mã nguồn, không commit fixture binary.
