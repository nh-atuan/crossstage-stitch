# S2/S3: triển khai P1 của Thành

- Người phụ trách: Thành.
- Ngày ghi nhận: 25/10 (Chủ nhật), năm 2026.
- Phase 1, deadline chung: 25/10 (Chủ nhật).
- Trạng thái: đã build và chạy test trên Colab Tesla T4 ngày 25/10 (Chủ nhật). Golden, sanitizer và pipeline mock nối với S4/S5 CUDA đã chạy. Bằng chứng và phần chưa đạt nằm trong [bằng chứng T4](s2-s3-t4-validation.md). Phương pháp và lý do chọn nằm trong [quyết định S2/S3](decisions/s2-s3-orb.md).

Phạm vi theo [kế hoạch, mục 8](plan.md): kernel S2 có núm K, kernel S3, test so với OpenCV, tối ưu chỗ nghẽn. Ghép vào harness và root build thuộc phần của Tuấn và Đế.

## 1. Code và build

| Đường dẫn | Nội dung |
|---|---|
| `src/s2_s3/s2.hpp`, `s2.cu` | S2: FAST-9, NMS, chọn 2K điểm theo FAST score, Harris, chọn top-K |
| `src/s2_s3/s3.hpp`, `s3.cu` | S3: blur 7×7, góc theo intensity centroid, rBRIEF 256 bit |
| `src/s2_s3/s2_s3_internal.hpp`, `s2_s3_config.cpp` | Kiểm tra tham số và layout workspace, chạy trên host, không cần CUDA |
| `src/s2_s3/orb_pattern.hpp` | Pattern `bit_pattern_31_` của OpenCV 5.0.0, giữ nguyên giấy phép BSD |
| `src/s2_s3/CMakeLists.txt` | Build độc lập của stage, cùng mẫu với `src/s4` |
| `tests/test_s2_s3_config.cpp` | Test host |
| `tests/test_s2_s3_cuda.cu` | Test GPU: contract, tính tất định, so golden, đo thời gian; trả 77 khi không có GPU |
| `tests/s2_s3/CMakeLists.txt`, `tests/s2_s3_mock_driver.cu` | Build chung S2/S3 với S4/S5 và driver chạy nối tiếp trên một stream |
| `tools/s2_s3/reference.py` | Bản tham chiếu NumPy, cùng thuật toán và cùng thứ tự phép tính với CUDA |
| `tools/s2_s3/export_golden.py`, `fixtures.py` | Sinh golden nhị phân cho test CUDA từ ảnh tổng hợp |
| `tools/s2_s3/check_mock_pipeline.py` | So H của chuỗi CUDA với mock ORB và với ORB một level của OpenCV |
| `tests/python/test_s2_s3_reference.py` | So bản tham chiếu với `cv2.ORB` và `cv2.FastFeatureDetector` |

Root build chưa đưa stage này vào. Tuấn có thể dùng `add_subdirectory(src/s2_s3)` rồi link `stitch_s2_s3`. Stage không phụ thuộc OpenCV, CUB hay stage khác.

```bash
cmake -S src/s2_s3 -B build/s2_s3 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/s2_s3 -j 2
ctest --test-dir build/s2_s3 --output-on-failure
```

## 2. Contract

API ở [s2.hpp](../src/s2_s3/s2.hpp) và [s3.hpp](../src/s2_s3/s3.hpp). Thứ tự gọi cho một ảnh:

```cpp
stitch::s2::Config c2{20, K, 0.04f};  // fast_threshold, max_keypoints, harris_k
std::size_t bytes2 = 0, bytes3 = 0;
stitch::s2::s2_workspace_bytes(width, height, c2, &bytes2);
stitch::s3::s3_workspace_bytes(width, height, &bytes3);
// Caller allocates: keypoints x/y/response with capacity >= K, d_count,
// descriptors K rows of >= 32 bytes, two 256-byte aligned workspaces.
stitch::s2::s2_detect(gray, c2, keypoints, ws2, bytes2, stream);
stitch::s3::s3_describe(gray, keypoints, descriptors, ws3, bytes3, stream);
// keypoints.d_count and the descriptors are ready when the stream completes.
```

Đoạn trên chỉ minh họa thứ tự gọi. Caller phải kiểm tra từng `Status` và lỗi CUDA khi đồng bộ.

- `gray`: `ImageView` GRAY8 trên device, có pitch. Đây là output S1 với **σ = 0**. S2 và S3 dùng cùng một ảnh, S3 tự blur bên trong.
- `Keypoints`: tọa độ nguyên trong hệ S1, `response` là Harris score dạng float. Thứ tự là Harris giảm dần, bằng score thì chỉ số pixel nhỏ đứng trước. `*d_count = min(số điểm tìm được, K)`.
- Keypoint luôn cách mỗi biên ít nhất `s2::kEdge = 31` pixel. Ảnh có cạnh ≤ 62 px cho `*d_count = 0`, không báo lỗi.
- `Descriptors`: `BINARY_U8`, `dim = 32`, `row_stride_bytes ≥ 32`. Hàng `i` ứng với keypoint `i`, nên S4 dùng thẳng chỉ số này. Byte đệm sau 32 byte không bị ghi.
- Hai hàm không cấp phát, không đồng bộ, không đọc count về host. `Status::OK` chỉ xác nhận tham số hợp lệ và launch thành công. K = 0 không cần workspace.
- Workspace S2 là O(W·H + K): score map, maxima map, histogram, mảng scan theo block và 2K key Harris. Workspace S3 là O(W·H): ảnh blur và buffer float của pass ngang. Kích thước tính được trên host.
- `harris_k` và `fast_threshold` là tham số tường minh. Giá trị 20 và 0.04 lấy theo OpenCV, chưa được tune trên development data.

## 3. Cách S2 chọn K điểm mà không đọc count về host

1. `fast_kernel` tính FAST score cho mọi pixel cách biên ≥ 3 px.
2. `nms_kernel` giữ cực đại nghiêm ngặt 3×3 trong vùng cách biên 31 px và cộng histogram score.
3. `select_kernel` chạy một block 256 thread, tính tổng hậu tố của histogram. Từ đó suy ra ngưỡng score sao cho giữ đúng 2K điểm: giữ toàn bộ điểm có score lớn hơn ngưỡng, và lấy đủ phần còn thiếu trong số các điểm có score bằng ngưỡng theo thứ tự pixel.
4. `count_kernel`, `block_scan_kernel` và `scatter_kernel` scan theo block để ghi các điểm được chọn vào slot liên tục, giữ thứ tự pixel.
5. `harris_kernel` tính Harris và tạo key 64 bit: 32 bit cao là Harris đã đổi sang thứ tự không dấu, 32 bit thấp là `0xffffffff − chỉ số pixel`.
6. `rank_kernel` tính hạng của mỗi key bằng cách đếm số key lớn hơn nó. Key nào có hạng < K được ghi vào vị trí đúng hạng của nó. Mọi key khác nhau, nên kết quả tất định.

Bước 6 tốn O((2K)²) phép so sánh, nên đây là kernel S2 tốn nhiều nhất khi K lớn và ảnh có đủ góc. Nếu sau này dùng K lớn hơn 4000 thì nên thay bằng radix select (xem fallback trong decision).

## 4. Số học và tính tất định

- FAST, NMS, tổng gradient Harris và moment orientation đều là phép tính nguyên.
- Công thức Harris, `fastAtan2`, phép xoay pattern và blur dùng `__fmul_rn`, `__fadd_rn`, `__fsub_rn`, `__fdiv_rn`, để nvcc không gộp thành FMA. Bản tham chiếu NumPy làm từng phép float32 theo đúng thứ tự đó.
- `cos` và `sin` của góc tính bằng double rồi đổi về float. Chỉ lane 0 của warp tính rồi gửi cho các lane khác, vì FP64 trên T4 chậm.
- Blur là bộ lọc tách float32 với hệ số của `cv::getGaussianKernel(7, 2, CV_32F)`. Đây cũng là nhánh `cv::ORB` dùng bên trong (xem decision, mục 3).
- Không có yếu tố ngẫu nhiên. Hai lần chạy cùng input cho output giống từng byte, và test CUDA kiểm tra điều này.

## 5. Tối ưu đã làm

Theo nsys trên T4, ba chỗ tốn nhất ban đầu là:

| Chỗ nghẽn | Tỷ lệ thời gian |
|---|---|
| `describe_kernel`: cả 32 lane tính `cos`/`sin` double | 41% |
| `rank_kernel`: luôn quét đủ 2K slot | 19% |
| `select_kernel`: chạy một thread | 20 µs mỗi lần |

Các sửa đổi:
- chỉ lane 0 tính góc;
- chỉ rank các slot đã chọn;
- chọn ngưỡng bằng scan song song;
- loại sớm pixel bằng 4 điểm la bàn của vòng FAST.

Output không đổi: 41/41 golden vẫn khớp từng bit. Thời gian S2 + S3 mỗi ảnh giảm 1,2–3,5 lần, số liệu ở [bằng chứng T4](s2-s3-t4-validation.md), mục 4.

## 6. Việc cần phối hợp

- **Tuấn:** S1 dùng σ = 0 khi chạy với S2/S3 này. `tools/mock/stages.py` hiện mặc định σ = 1, còn `check_mock_pipeline.py` truyền `sigma=0` mà không sửa file mock. Ngoài ra cần đưa `src/s2_s3` vào root build và harness. `t_s2` tính từ trước `s2_detect` của ảnh A tới sau `s2_detect` của ảnh B, `t_s3` tương tự.
- **Đế:** `num_kp_a`, `num_kp_b` đọc từ `d_count` sau điểm đồng bộ 1. Descriptor dùng Hamming, giống ca ORB đã test ở S4. Dải K đề xuất cho sweep là {500, 1000, 2000, 4000}.
- **Cả Nhóm:** quyết định cách xử lý ba cặp Oxford có thay đổi scale lớn mà ORB một level không ghép được ([bằng chứng T4](s2-s3-t4-validation.md), mục 3).
