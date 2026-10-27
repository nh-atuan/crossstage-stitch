# Benchmark S6 với cùng H

Task P2-08 của Chương, cập nhật 27/10 (Thứ 3), năm 2026. Công cụ tái dùng CPU reference FP64 và golden/profiling runner CUDA hiện có; không chạy lại S1–S5. Ba mode là `feather`, `multiband_3`, `multiband_5`; 3/5 là số REDUCE, tương ứng tối đa 4/6 mức.

## Đầu vào và chạy

Tạo manifest JSON, đường dẫn tương đối tính từ thư mục chứa manifest. `H` là B → A trong hệ RGB gốc, có thể là ma trận 3×3, mảng 9 số, hoặc đường dẫn `.npy`/JSON chứa ma trận. Ảnh `.npy` phải là RGB uint8; ảnh thông thường được đọc và chuyển sang RGB.

```json
{
  "schema_version": "s6-same-h-v1",
  "evidence_type": "real_images",
  "split": "development",
  "pairs": {
    "pair_001": {"a": "a.png", "b": "b.png", "H": "H.npy"}
  }
}
```

Giá trị `evidence_type` còn có `synthetic_fixture`; split phải khai báo `development`, `validation`, `final` hoặc `synthetic`. Một pair trong manifest cũng là cách chạy một cặp ảnh. Không dùng ảnh tổng hợp dưới nhãn dữ liệu thật.

```bash
python -m tools.bench.same_h --manifest input.json --output results/s6-prepared --prepare-only
python -m tools.bench.same_h --manifest input.json --output results/s6-same-h-t4 --runner build/s6/test_s6_cuda --warmups 2 --repeats 10 --seed 0 --save-images
```

Build runner theo [hướng dẫn S6](../s6_reference/README.md), với CUDA `sm_75`. Lệnh thứ hai mặc định yêu cầu T4; chạy GPU khác phải thêm `--allow-non-t4` và kết quả được gắn `preliminary_non_t4`. `--prepare-only` chỉ tạo fixture CPU, không phải CUDA pass. Có thể chọn `--modes feather multiband_3`, `--timeout 600`, `--max-reference-mib 2048`. Memory guard của CPU reference là ước lượng admission, không phải số đo RAM/VRAM.

Thư mục output phải mới hoặc rỗng. Công cụ giữ failure ledger cho mọi pair, kể cả ảnh thiếu, H suy biến, timeout và native exit 77. Không thay H lỗi bằng identity. Thứ tự B được xáo trộn với seed cố định và lưu trong receipt.

## Tính đúng và provenance

Mỗi B nhận cùng byte RGB, H float64 và canvas/offset. Công cụ kiểm tra header/source bytes trước và sau chạy, so canvas/mask giữa các mode, rồi kiểm tra output CUDA với CPU reference độc lập. Mask uint8 dùng 0/1, không suy validity từ màu đen. Mask/canvas phải khớp tuyệt đối; sai số RGB tuyệt đối tối đa cho phép là 2 mức uint8 theo golden runner hiện có. Đây là kiểm chứng implementation, không phải chứng minh chất lượng blend cảm nhận.

`report.json` ghi split, loại bằng chứng, revision, trạng thái working tree, hash manifest/source/binary/ảnh/H/fixture, môi trường, lệnh, thứ tự B, return code và metadata từng mode. Mỗi pair có `runner.log`, fixture, timing CSV và artifact JSON. `--save-images` giữ PPM/PGM; mặc định chỉ xóa các PPM/PGM vừa sinh sau khi kiểm chứng, giữ metadata, fixture và log. Không ghi đè kết quả trước.

## Ranh giới timing và memory

| Trường | Ranh giới đo |
|---|---|
| `geometry_host_ms` | Một lần chuẩn bị geometry/canvas trên host |
| `device_setup_host_ms` | Một lần cấp phát, khởi tạo device buffers/workspace và tài nguyên runner |
| `h2d_host_ms` | Một lần upload input trong runner |
| `gpu_ms` | CUDA events quanh lời gọi render S6; geometry/buffer đã chuẩn bị, statistics tắt |
| `enqueue_host_ms` | Host dispatch từ trước start-event đến sau stop-event record |
| `sync_wait_host_ms` | Host chờ completion đã kiểm tra |
| `host_ms` | Lời gọi S6 steady-state hoàn chỉnh; bằng hai thành phần host trên |
| `runner_wall_ms` | Tiến trình native, gồm correctness render, warmup, repeat, I/O và setup |

GPU interval chồng lấp thời gian host: không cộng `gpu_ms` vào `host_ms`. Setup là số đo một lần, không ghép thành phân phối end-to-end bằng cách cộng median. Không lấy phần thời gian dư sau S1–S5 làm S6 GPU latency. CUDA event interval có thể gồm khoảng stream rỗi do host dispatch; runner không tách riêng từng kernel.

Nếu CUDA event lớn hơn host elapsed, receipt ghi `cuda_event_exceeds_host_elapsed_check_clock_domains`. Cần đối chiếu môi trường/clock trên T4 trước khi dùng số đo đó cho kết luận hiệu năng. Không hiệu chỉnh bằng hệ số suy đoán.

`workspace_bytes` là số byte từ API query S6 cho canvas/B tương ứng, không phải peak VRAM toàn tiến trình. Input/output, mask, runtime CUDA và allocator overhead không được tính vào workspace này. Device allocation ở ngoài vòng timed repeats; cùng buffers được tái dùng. `p95_gpu_repeat_ms` là quantile tuyến tính của repeat trong một pair, chỉ để chẩn đoán, không phải p95 của tập cảnh.

## Phạm vi kết luận

Cùng H làm alignment invariant; PSNR/SSIM alignment không đánh giá seam hay ghosting. `composition_quality` luôn ghi `not_evaluated`. Chưa chọn B mặc định mới và chưa thay quality policy. Mọi bảng workshop cần chạy cohort được duyệt trên Colab T4, giữ dữ liệu thô và đối chiếu policy với Đế.

```bash
python -m pytest tests/python/test_same_h_bench.py tests/python/test_s6_validation.py tests/python/test_s6_reference.py -q
```
