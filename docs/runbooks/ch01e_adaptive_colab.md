# Kiểm chứng harness adaptive trên Colab T4

Cập nhật 06/11 (Thứ 6), năm 2026. Phạm vi là policy loader, attempt trace, ranh giới timing và đường lỗi của harness. Cấu hình `legacy-v1` giữ ladder, probe và self-check đang có. Policy JSON `controller-policy-v2` tương thích với loader ở commit `7da18e8` của Đế. Trace kế thừa các trường reuse từ `8e1380f`. Đồng bộ upload theo `25da69a` và áp dụng CLI seed cho grid theo `e378c53`. Nhánh `de/p3-controller-evaluation` đã kiểm tại `853e791`. Phần tích hợp này dùng controller hiện tại của nhánh Chương. [Nhánh của Đế](https://github.com/nh-atuan/crossstage-stitch/tree/de/p3-controller-evaluation)

## Policy và đầu ra

- Harness đọc `--controller-policy <file>` trước khi cấp phát GPU. Schema giới hạn 16 nấc, B cố định là Feather, K tối đa 4000 và R tối đa 2048. `max_residual_s1` dùng pixel S1. Khi file khai báo `provenance.seed`, `--seed` phải khớp nếu truyền đồng thời. Caller cũ bỏ cờ vẫn dùng `legacy-v1` và có ID tương ứng trong CSV.
- `runs.csv` giữ một dòng terminal mỗi lượt. Các trường mới `policy_id`, `attempt_count`, `t_probe_gpu` và `t_stream_wait` áp dụng cho adaptive. `--trace-out` ghi một dòng cho mỗi attempt đã bắt đầu, theo khóa `run_id,pair_id,repeat_idx,is_warmup`. Join thêm dataset, split, seed và policy khi tổng hợp nhiều campaign. `attempt_idx` bắt đầu từ 0.
- Attempt ghi S/K/R/B, stage bắt đầu lại, generation, `previous_R`, action/reason nếu đã có quyết định, mã lỗi và thời gian. Attempt chưa có model S5 hợp lệ để trống H. Lượt lỗi trước attempt có `attempt_count=0` và trạng thái terminal. Khi gặp lỗi CUDA không an toàn để tiếp tục, harness lưu trạng thái đã thu được rồi dừng với exit khác 0. Campaign khi đó cần được đánh dấu thiếu ở phần chưa chạy.
- `t_e2e` adaptive đo từ RGB device-ready đến khi render hoàn tất hoặc gặp lỗi. Upload chờ trên stream của pipeline trước vùng đo. `t_controller` đo CPU `begin_pair` và `observe`. `t_probe_gpu` nằm trong `t_s1`. `t_stream_wait` là thời gian host chờ stream và chồng lấp các khoảng GPU. Cộng các khoảng chồng lấp sẽ đếm thời gian hai lần. Tải telemetry và ghi CSV diễn ra sau vùng đo thành công.

## Smoke và hồi quy

Chạy từ root repo trên Colab T4, với manifest smoke development đã khóa và thư mục output mới. Bước reference và Fixed-Best cho campaign test chính theo [báo cáo nghiệm thu](../reports/phase-3-acceptance-report.md#reference-và-nguyên-nhân-gate-chưa-đạt).

```bash
set -e
cmake -S . -B build/ch01e-t4 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/ch01e-t4 -j2
ctest --test-dir build/ch01e-t4 --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 build/ch01e-t4/harness/test_adaptive_upload

out=results/ch01e-t4-smoke
mkdir "$out"
python -m tools.env.generate_receipt --output "$out/environment_receipt.json"
build/ch01e-t4/harness/stitch_harness \
  --mode adaptive --controller-policy legacy-v1 \
  --manifest data/p3/smoke/pairs_smoke.json --dataset UDIS-D --split train-fit \
  --seed 0 --warmup 2 --repeats 10 --run-id ch01e-t4-smoke \
  --out "$out/runs.csv" --trace-out "$out/attempts.csv"
```

Kỳ vọng `test_adaptive_upload` pass và memcheck báo 0 lỗi. Test đổi ảnh giữa hai lần upload rồi kiểm probe đầu tiên, đối chiếu attempt cuối và kiểm reset trace sau lỗi trước attempt. Với smoke hai cặp, kỳ vọng 24 dòng terminal gồm warmup. Số dòng trace trên mỗi khóa phải bằng `attempt_count`. Nếu có attempt, S/K/R/B dòng cuối phải khớp terminal. Nếu chạy policy JSON, lưu cả file policy và SHA-256 cùng receipt, log CTest, log sanitizer, hash binary và hai CSV. Chạy lại vào thư mục mới để so path, status và H theo cùng seed. So sánh timing riêng vì timing không có tính tất định từng byte.

Phiên triển khai local kiểm test host, replay audit và compile-only `sm_75`. Runtime CUDA, sanitizer và số đo của revision tích hợp cần lượt chạy Colab ở trên.
