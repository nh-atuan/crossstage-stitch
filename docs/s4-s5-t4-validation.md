# Kiểm chứng S4/S5 trên Colab Tesla T4

- Người phụ trách: Đế.
- Ngày chạy: 23/10 (Thứ 6), năm 2026.
- Source revision: `aaa937a0c526f88fafce6d292ba64f75aab173d5`.
- Phạm vi: build độc lập S4/S5, build kiểm thử phối hợp, correctness, memory/race checks và mock pipeline. Không đo latency hoặc quyết định Go/No-Go.

Source được chuyển từ working tree sạch bằng `git archive` của revision trên, gồm `CMakeLists.txt`, `src/`, `tests/`, `tools/`; không lấy source từ một nhánh remote cũ và không push để chuyển file. SHA-256 của archive đã được đối chiếu ở hai máy:

```text
a39ed4af2e2d5859ffabbfe772146c65afca77b406769c85e11d5bbda4396fe3
```

## 1. Môi trường thực chạy

| Thành phần | Giá trị ghi từ runtime |
|---|---|
| GPU | Tesla T4, 15360 MiB |
| NVIDIA driver | 580.82.07 |
| CUDA toolkit / nvcc | 13.0 / V13.0.88 |
| CUDA architecture | 75 (`sm_75`) |
| CMake | 3.31.10 |
| g++ | 13.3.0 |
| Python | 3.13.15 |
| `opencv-python` trong venv | 5.0.0.93 |
| `cv2.__version__` trong venv | 5.0.0 |
| NumPy trong venv | 2.5.4 |
| pytest trong venv | 9.1.1 |

Colab ban đầu có cả `opencv-python`, `opencv-contrib-python` và `opencv-python-headless`. Dù metadata của `opencv-python` ghi đúng bản pin, module được import là OpenCV 4.14.0 của bản contrib. Bộ test Python vẫn pass trên môi trường đó, nhưng kết quả chính thức bên dưới dùng venv riêng để không lẫn binary. Không gỡ hoặc sửa các package có sẵn của Colab.

Các lệnh tạo môi trường Python riêng, chạy từ root source:

```bash
python -m venv --without-pip .venv
python -m pip --python .venv/bin/python install -r tools/requirements.txt
.venv/bin/python -m pytest tests/python -q
```

## 2. Build và test

Các lệnh configure/build/CTest đều trả exit code 0:

```bash
cmake -S tests/s4_s5 -B build/s4_s5 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/s4_s5 -j 2
ctest --test-dir build/s4_s5 --output-on-failure
cmake -S src/s4 -B build/s4 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/s4 -j 2
cmake -S src/s5 -B build/s5 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/s5 -j 2
cmake -S . -B build/common -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/common -j 2
ctest --test-dir build/common --output-on-failure
```

| Bộ kiểm tra | Kết quả |
|---|---|
| CTest S4/S5 | 4/4 pass: `test_s4_config`, `test_s4_cuda`, `test_s5_math`, `test_s5_cuda` |
| CTest common | 2/2 pass: `test_common`, `test_common_cuda` |
| Build độc lập `src/s4` và `src/s5` | Cả hai pass, có library và test CUDA |
| pytest trong venv | 95/95 pass |
| Memcheck S4 và S5 | Cả hai exit 0; `ERROR SUMMARY: 0 errors` |
| Racecheck S4 và S5 | Cả hai exit 0; `0 hazards displayed (0 errors, 0 warnings)` |

Không có test CUDA bị skip. Test S5 dùng seed 20261022, gồm so sánh kết quả R=257 một lần với R=17 rồi tăng lên 257. Thời gian chạy CTest không được dùng làm latency của pipeline.

Lệnh sanitizer đã chạy cho cả hai stage:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 build/s4_s5/test_s4_cuda
compute-sanitizer --tool memcheck --error-exitcode 1 build/s4_s5/test_s5_cuda
compute-sanitizer --tool racecheck --error-exitcode 1 build/s4_s5/test_s4_cuda
compute-sanitizer --tool racecheck --error-exitcode 1 build/s4_s5/test_s5_cuda
```

## 3. Mock pipeline với ground truth

Chạy OpenCV mock S1–S3 → CUDA S4/S5 → OpenCV mock S6, không phải sáu stage GPU. Fixture có hai ảnh RGB 640×480, seed sinh ảnh 17, ảnh B là A dịch `(16,8)` pixel; H đúng B → A là phép dịch `(-16,-8)`.

Fixture có cấu trúc được tạo bằng đoạn sau, dùng chính `gen_pair` của Nhóm:

```python
from pathlib import Path
import cv2
import numpy as np
from tools.mock.gen_mock import gen_pair

rng = np.random.default_rng(17)
a = cv2.GaussianBlur(
    rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (3, 3), 0.7
)
for i in range(140):
    x, y = map(int, rng.integers([24, 24], [616, 456]))
    color = (255, 255, 255) if i % 2 else (0, 0, 0)
    cv2.circle(a, (x, y), int(rng.integers(6, 19)), color, -1)
    cv2.putText(
        a, str(i), (x - 9, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
        tuple(255 - c for c in color), 1, cv2.LINE_AA
    )
forward = np.array([[1., 0., 16.], [0., 1., 8.], [0., 0., 1.]])
b = cv2.warpPerspective(a, forward, (640, 480))
gen_pair(Path("data/mock"), "synthetic_structured", a, b, np.linalg.inv(forward))
```

Chạy hai representation được interface S4 hỗ trợ:

```bash
.venv/bin/python -m tools.s4_s5.check_mock_pipeline \
  build/s4_s5/s4_s5_mock_driver data/mock/synthetic_structured orb results/structured-orb
.venv/bin/python -m tools.s4_s5.check_mock_pipeline \
  build/s4_s5/s4_s5_mock_driver data/mock/synthetic_structured sift results/structured-sift
```

Hai lệnh đều exit 0, driver ghi `GPU=Tesla T4 seed=0 R=2000 S5=OK`. Match indices khớp OpenCV; distance và chuyển hệ tọa độ qua các assertion của runner, không đổi tolerance.

| Descriptor | Matches | Inliers | Mean corner difference với OpenCV, pixel gốc | Mean corner error với H đúng, pixel gốc |
|---|---:|---:|---:|---:|
| ORB / Hamming | 611 | 588 | 0.01028045 | 0.33703143 |
| SIFT / L2 | 821 | 818 | 0.00006370 | 0.00871566 |

Các số này lấy từ `report.json` của runner, không phải số đo chất lượng trên benchmark thật. Ảnh preview gồm reference mock, CUDA ORB và CUDA SIFT đã được mở để kiểm tra: các hình tròn và chữ tương ứng thẳng hàng, không có panorama trống hoặc lệch bố cục rõ rệt ở mức preview.

### Ca thiếu match được giữ lại

Fixture đầu tiên chỉ có noise đã làm mượt, cùng seed và phép dịch, chưa có hình tròn/chữ. ORB có 7 matches và pass; SIFT chỉ có 2 keypoints ở A và 2 matches nên không đủ bốn correspondences để tìm H.

Lệnh smoke SIFT ban đầu fail. Đã giữ log, kiểm tra metadata rồi xác nhận trực tiếp driver trả `S5=NOT_ENOUGH_POINTS`, exit 1. Đây là ca từ chối đúng, không phải lỗi kernel được sửa bằng cách hạ ngưỡng hoặc thay H bằng identity. Ca dương sau đó dùng fixture có cấu trúc ở trên; không đổi thuật toán, tham số stage hoặc assertion để tạo pass.

## 4. Bằng chứng và phần còn lại

Log đầy đủ, metadata lấy từ code, report ORB/SIFT và ảnh preview đã tải về `results/t4-validation-aaa937a/`. Thư mục này bị gitignore theo quy ước; không commit output hoặc dataset. SHA-256 của bundle đã đối chiếu trước khi giải nén local:

```text
a82392ee44391b52c01c68229ac9899f5a99af6973737523d22d0cf45ea3154b
```

Các log gồm cả failure ban đầu `mock-sift.log`, chẩn đoán input và ca âm `expected-insufficient-sift.log`; không bỏ failure khỏi bằng chứng. Panorama PNG và mock data gốc vẫn nằm trong runtime Colab; bản tải local có JPEG preview.

Không cần sửa source để các kiểm tra trên pass. Lần này xác nhận correctness của S4/S5 trên các ca test hiện có và fixture tổng hợp đã nêu. Vẫn cần HPatches/UDIS-D thật, quality-policy pilot, ghép đủ sáu stage GPU và harness của Tuấn, latency trên development pairs và Go/No-Go trước khi coi toàn bộ P1 hoàn tất.
