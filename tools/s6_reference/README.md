# Reference FP64 cho S6

Đây là oracle CPU độc lập để kiểm tra `src/s6`, không phải backend chạy chính hoặc benchmark GPU. Ngày cập nhật: 23/10 (Thứ 6), năm 2026. Quy tắc chi tiết ở [bàn giao P1](../../docs/s6-p1-implementation.md) và [phương pháp](../../docs/s6-warp-blend.md).

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

## Bàn giao mock HPatches đã có sang S6

`prepare_mock.py` đọc artifact **đã được** `tools/mock/gen_mock.py` tạo; không tải dataset, không chạy lại S1–S5 và không đổi ảnh/H. Cấu trúc đầu vào mỗi pair là `input_a.npy`, `input_b.npy`, `orb_H.npy` và/hoặc `sift_H.npy`, cùng `meta.json`. H phải là **B → A ở RGB gốc**, đúng output của mock S5; không truyền `*_H_s.npy`.

```bash
python -m tools.s6_reference.prepare_mock --mock-root data/mock --output data/s6-mock-fixtures
python -m tools.s6_reference.prepare_mock --mock-root data/mock --output data/s6-mock-orb --descriptor orb
python -m pytest tests/python/test_s6_mock_fixtures.py -q
```

Mặc định xét **cả ORB và SIFT**, tạo ba mode cho mỗi descriptor/pair có handoff hợp lệ. Header lưu nguyên các giá trị H cung cấp, kể cả hệ số scale; không dùng `h_true_ba.npy`, H làm việc hay identity để thay cho H bị thiếu. Reference S6 vẫn độc lập với panorama/mask do OpenCV mock tạo.

`meta.json` ghi descriptor, metadata upstream, hash các ảnh/H/metadata thực sự tiêu thụ, các hash mã nguồn hiện tại, canvas/levels và fixture hashes. Hash mã nguồn hiện tại **không chứng minh** đó chính là code đã sinh dữ liệu upstream; version upstream được giữ riêng. JSON upstream có NaN/Infinity được giữ dưới tag `upstream_nonfinite_float` trong metadata mới thay vì phát ra JSON không chuẩn.

Mỗi handoff thành công phải tạo **đủ ba mode**. Thiếu H, thiếu/corrupt input hoặc metadata, H không hợp lệ, horizon hay vượt budget được ghi trạng thái/lý do; handoff thất bại không có fixture thay thế. Nếu feather chạy được nhưng multi-band bị guard từ chối, toàn bộ fixture tạm của handoff đó bị bỏ. CLI trả 0 khi có ít nhất một handoff thành công và cảnh báo các handoff bị bỏ; trả 2 nếu không có handoff thành công hoặc lỗi CLI. Luôn đọc failure ledger, không suy ra mọi pair đã pass từ exit 0.

Output phải **mới hoặc rỗng**, tách khỏi cây mock input. Chạy lại vào thư mục mới để đối chiếu hash; công cụ từ chối output không rỗng để tránh dùng nhầm `.bin` cũ sau một lần S5 thất bại.

Budget reference mặc định vẫn là **512 MiB admission guard**, canvas tối đa 16.777.216 pixel; không tự nâng guard hoặc resize/crop để vừa RAM. Canvas HPatches có thể vượt guard đa băng dù ảnh riêng lẻ không lớn. Nếu máy đủ RAM, có thể yêu cầu một ngân sách rõ ràng, ví dụ:

```bash
python -m tools.s6_reference.prepare_mock --mock-root data/mock --output data/s6-mock-2g --max-reference-mib 2048
(cd data/s6-mock-2g && sha256sum -c manifest.sha256)
build/s6/test_s6_cuda --golden data/s6-mock-2g/*.bin
```

Giá trị budget phải dương/hữu hạn; đây vẫn là ước lượng admission, chưa là peak RAM đo được. Ảnh/H không thay đổi; hash của cùng fixture thành công giữ nguyên trong cùng môi trường. Manifest có thể thêm fixture nếu ngân sách mới cho phép xử lý thêm handoff. Chưa chạy HPatches hoặc CUDA không được ghi là đã nghiệm thu bằng dữ liệu thực/GPU; tests mới chỉ dùng mock directory tổng hợp để kiểm tra luồng bàn giao.
