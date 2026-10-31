# Hướng dẫn chạy Grid Sweep và Phân tích Go/No-Go Phase 1

- **Người thực hiện chính:** Đế (phối hợp cùng Tuấn).
- **Môi trường thực thi:** Google Colab với GPU Tesla T4 (`sm_75`).
- **Mục tiêu:** Chạy quét 48 cấu hình ($S \times K \times R$) trên 20–30 cặp ảnh development, đánh giá hai điều kiện Go/No-Go để Nhóm quyết định hướng đi cho Phase 2.

---

## 1. Chuẩn bị môi trường trên Colab T4

Trên phiên Google Colab GPU (Tesla T4):

```bash
# 1. Cài đặt các thư viện cần thiết theo yêu cầu của Nhóm
pip install -r tools/requirements.txt

# 2. Build binary harness đo thời gian CUDA S1-S6
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target stitch_harness -j2

# 3. Kiểm tra binary đã sẵn sàng
./build/harness/stitch_harness --help
```

---

## 2. Chuẩn bị 20–30 cặp ảnh Development

### Định dạng manifest (`pairs.json`)
Script sweep nhận đầu vào là file JSON mô tả danh sách các cặp ảnh. Cấu trúc như sau:

```json
{
  "pair_01": {
    "a": "data/dev_pairs/pair01_a.ppm",
    "b": "data/dev_pairs/pair01_b.ppm"
  },
  "pair_02": {
    "a": "data/dev_pairs/pair02_a.ppm",
    "b": "data/dev_pairs/pair02_b.ppm"
  }
}
```

### Lệnh chuẩn bị dataset tự động (Chạy trực tiếp trên Colab)
Chạy script tự động để chuẩn bị 25 cặp ảnh development dưới định dạng PPM (P6):

```bash
# Tự động tải/chuẩn bị 25 cặp ảnh development và tạo manifest
python -m tools.data_prep.prepare_dev_dataset \
    --num-pairs 25 \
    --out-dir data/dev_pairs \
    --manifest data/dev_pairs/pairs.json
```

*Ghi chú:*
- Script tự động quét thư mục ảnh UDIS-D (nếu đã mount Drive hoặc tải về).
- Nếu chạy trong môi trường mới hoàn toàn chưa có dataset, script tự động sinh 25 cặp ảnh development đa dạng (hình học, texture, noise, độ lệch góc nhìn) với seed cố định `42` theo quy ước của Nhóm để kiểm nghiệm pipeline.

---

## 3. Chạy Grid Sweep 48 cấu hình

Lệnh sau sẽ sinh 48 cấu hình ($S \in \{0.25, 0.5, 0.75, 1.0\}$, $K \in \{500, 1000, 2000\}$, $R \in \{128, 256, 512, 1024\}$), thực thi đo thời gian bằng CUDA event qua `stitch_harness` và tự động đánh giá chất lượng QoR (SSIM, PSNR, Coverage):

```bash
python -m tools.sweep.run_sweep \
    --harness build/harness/stitch_harness \
    --manifest data/dev_pairs/pairs.json \
    --policy tools/qor/policy-pilot.json \
    --ref-config S1.00_K2000_R1024_Bfeather \
    --out-dir results/sweep_p1
```

Kết quả sinh ra trong `results/sweep_p1/`:
- `configs.json`: Danh sách 48 cấu hình lưới.
- `runs.csv`: Dữ liệu đo đạc thời gian từng stage (S1–S6, E2E) và telemetry từ GPU.
- `qor.csv`: Dữ liệu đánh giá chất lượng alignment (SSIM, PSNR, Coverage).

---

## 4. Phân tích kết quả Go/No-Go

Chạy script phân tích để kết hợp `runs.csv` và `qor.csv`, tìm cấu hình Fixed-Best, Oracle per-pair và kiểm tra 2 điều kiện Go/No-Go:

```bash
python -m tools.analysis.analyze_gonogo \
    --runs results/sweep_p1/runs.csv \
    --qor results/sweep_p1/qor.csv \
    --out results/sweep_p1/gonogo_report.json
```

---

## 5. Tiêu chí đọc kết quả cho buổi họp Review

Báo cáo in ra terminal và lưu ở file JSON sẽ đưa ra một trong 3 kết luận:

| Kết quả | Điều kiện | Ý nghĩa & Hành động tiếp theo |
|---|---|---|
| **GO** | Đạt cả 2 điều kiện:<br>1. Stage chỉnh được (S1–S5) chiếm $\ge 30\%$ thời gian.<br>2. Oracle nhanh hơn Fixed-Best $\ge 1.20\times$ (20–25%). | Nhóm giữ nguyên hướng đi: Tiến hành thiết kế và huấn luyện/cài đặt runtime controller thích nghi ở Phase 2. |
| **GO WITH ADJUSTMENT** | Đạt Điều kiện 2, nhưng Điều kiện 1 $< 30\%$. | Báo cáo riêng độ trễ tính toán, hoặc nâng núm blending $B$ (feather vs multiband) lên thành núm điều khiển chính. |
| **NO-GO** | Không đạt Điều kiện 2 (Oracle không nhanh hơn Fixed-Best đáng kể). | Chuyển paper thành bài khảo sát Pareto giữa chất lượng và latency của panorama stitching trên GPU; giữ nguyên toàn bộ lịch trình. |
