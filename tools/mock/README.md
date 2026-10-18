# Mock data

Script sinh output tham chiếu (golden) của từng stage bằng OpenCV và numpy, từ một phần nhỏ HPatches.

## Dùng

1. `pip install -r tools/requirements.txt`
2. Tải HPatches sequences (`hpatches-sequences-release`) và giải nén.
3. `python tools/mock/gen_mock.py --hpatches <thư mục hpatches-sequences-release> --out data/mock`
4. Chưa có manifest chính thức vì chưa sinh được golden trên HPatches thật (xem trạng thái trong tài liệu bàn giao). Khi sinh, script ghi `tools/mock/manifest.sha256` (SHA256 của mọi file `.npy`). Commit manifest; `data/mock/` nằm trong `.gitignore`, chia sẻ qua Google Drive.

Golden chính thức sinh trên Colab với đúng version trong `tools/requirements.txt`. Chạy lại phải cho cùng SHA256.

## Nội dung mỗi cặp

Mỗi cặp `<seq>_<a>_<b>` nằm trong `data/mock/<pair_id>/`:

| File | Kiểu, shape |
|---|---|
| `input_a.npy`, `input_b.npy` | `uint8 (H, W, 3)`, RGB |
| `h_true_ba.npy` | `float64 (3, 3)`, `inv(H_a_b)` của HPatches, chiều B → A |
| `s1_S<S>_{a,b}.npy` | `uint8 (H_s, W_s)`, với S thuộc {1.00, 0.75, 0.50, 0.25} |
| `s1_S<S>_{a,b}_D.npy` | `float64 (3, 3)` |
| `{orb,sift}_kp_{a,b}.npy` | `float32 (N, 3)`: `x`, `y`, `response`, ở S = 0.5 |
| `{orb,sift}_desc_{a,b}.npy` | ORB `uint8 (N, 32)`, SIFT `float32 (N, 128)` |
| `{orb,sift}_match_idx.npy`, `_match_dist.npy` | `int32 (M, 2)`, `float32 (M,)` |
| `{orb,sift}_H_s.npy`, `_H.npy`, `_inlier_mask.npy` | `float64 (3, 3)` x2, `uint8 (M,)` |
| `s6_panorama.npy`, `s6_mask.npy` | `uint8 (Hc, Wc, 3)`, `uint8 (Hc, Wc)`, từ H của ORB |
| `meta.json` | version thư viện, tham số, `ScaleInfo`, số lượng, sai số góc so với `h_true_ba` |

Tham số của bản giả chỉ dùng cho mock data, không phải quyết định phương pháp của stage.
