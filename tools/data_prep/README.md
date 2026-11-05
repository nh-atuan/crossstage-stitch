# Công cụ tải tự động UDIS-D và chạy thực nghiệm Colab (`tools/data_prep` & `tools/colab_runner.py`)

Thư mục `tools/data_prep/` và script [`tools/colab_runner.py`](../colab_runner.py) cung cấp luồng tải tự động bộ dữ liệu UDIS-D chính thức từ Google Drive của tác giả ([`nie-lang/UnsupervisedDeepImageStitching`](https://github.com/nie-lang/UnsupervisedDeepImageStitching)), chuyển đổi ảnh sang định dạng PPM (P6) kèm `pairs.json`, sinh biên nhận `udis-source-v1` (`testing-source.json` / `training-source.json`) và chạy tự động `stitch_harness` cho 3 chế độ trên Colab Tesla T4.

## 1. Nguồn dữ liệu UDIS-D chính thức được cấu hình sẵn

Thư mục Google Drive gốc của tác giả UDIS-D (`1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY`) gồm 2 gói nén chính thức đã được gắn sẵn File ID trong [`prepare_dev_dataset.py`](prepare_dev_dataset.py):

| Split | File tên gốc | Google Drive File ID | Dung lượng | Số cặp ảnh | Link tải |
|---|---|---|---|---|---|---|
| `testing` | `testing.zip` | `1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0` | ~203 MB | 1.105 cặp (1.106 danh nghĩa; thiếu 000001.jpg trong input1) | [Tải testing.zip](https://drive.google.com/file/d/1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0/view) |
| `training` | `training.zip` | `1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a` | ~1.95 GB | 10.440 cặp | [Tải training.zip](https://drive.google.com/file/d/1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a/view) |
| Cả thư mục | Thư mục UDIS-D tác giả | `1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY` | ~2.15 GB | 11.546 cặp | [Thư mục Drive](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY) |

Khi tải về, script tự động tính mã băm SHA-256 của file nén và ghi file biên nhận `data/p3/testing-source.json` (hoặc `data/p3/training-source.json`) theo đúng schema `udis-source-v1`.

## 2. Các lệnh chuẩn bị dữ liệu cho từng thành viên

### 2.1. Chỉ tải `testing.zip` gốc và tạo `testing-source.json` (dành cho Chương chạy `tools/bench/same_h.py` / Ablation A4)

```bash
python -m tools.data_prep.prepare_dev_dataset \
    --mode gdrive \
    --split testing \
    --download-only
```

Kết quả:
- File nén gốc: `data/raw_udis/testing.zip`
- Biên nhận SHA-256: `data/p3/testing-source.json`

Nếu cần tải cả `training.zip` và `testing.zip`:

```bash
python -m tools.data_prep.prepare_dev_dataset \
    --mode gdrive \
    --split both \
    --download-only
```

### 2.2. Tải và chuẩn bị 25 cặp ảnh UDIS-D thật để kiểm thử nhanh (`data/dev_pairs/pairs.json`)

```bash
python -m tools.data_prep.prepare_dev_dataset \
    --mode gdrive \
    --split testing \
    --num-pairs 25 \
    --seed 42 \
    --out-dir data/dev_pairs \
    --manifest data/dev_pairs/pairs.json
```

### 2.3. Tải và chuẩn bị toàn bộ 1.105 cặp ảnh UDIS-D Test Set (`data/test_pairs/pairs.json`)

```bash
python -m tools.data_prep.prepare_dev_dataset \
    --mode gdrive \
    --split testing \
    --all \
    --out-dir data/test_pairs \
    --manifest data/test_pairs/pairs.json
```

## 3. Chạy tự động trên Colab Tesla T4 (`tools/colab_runner.py`)

Script [`tools/colab_runner.py`](../colab_runner.py) tự động hóa toàn bộ quy trình từ tải dữ liệu, lưu đệm vào Google Drive cá nhân/nhóm (để không phải tải lại khi tắt phiên Colab), đến chạy `stitch_harness` cho cả 3 chế độ (`Fixed-Best`, `Oracle`, `Adaptive`) và xuất file gộp `telemetry_results.csv` cùng `meta.json`.

### 3.1. Chạy thử 25 cặp kiểm thử trên Colab T4

```bash
python -m tools.colab_runner \
    --subset dev25 \
    --gdrive-cache-dir /content/drive/MyDrive/UDIS-D \
    --out-dir results/p3_dev25
```

### 3.2. Chạy đại thực nghiệm 1.105 cặp UDIS-D Test Set trên Colab T4

```bash
python -m tools.colab_runner \
    --subset test1105 \
    --gdrive-cache-dir /content/drive/MyDrive/UDIS-D \
    --out-dir results/p3_test1105
```

Kết quả đầu ra trong thư mục `--out-dir` (và được sao lưu tự động sang `/content/drive/MyDrive/UDIS-D/results/<subset>/` nếu có truyền `--gdrive-cache-dir`):
- `fixed_best_runs.csv`: Kết quả đo cấu hình cố định `Fixed-Best`.
- `oracle_grid_runs.csv`: Kết quả quét lưới cấu hình phục vụ `Oracle` và phân tích của Chương/Thành.
- `adaptive_runs.csv`: Kết quả chạy `Adaptive Controller`.
- `telemetry_results.csv`: Bảng gộp toàn bộ kết quả đo của 3 chế độ.
- `meta.json`: Thông tin GPU, driver, xung nhịp từ `nvidia-smi` và tham số phiên chạy.
