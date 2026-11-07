# CH-07/08 — Hình, bảng và gói bằng chứng T4

Phạm vi hiện tại gồm CH-02 `legacy-v1`, A4 development và hai track synthetic CH-05 đã nghiệm thu. CH-06 thuộc Tuấn. A5 test vẫn chờ một lock đạt gate sau kết quả development `NO_GO`. Gói này phục vụ bàn giao phần của Chương, còn nghiệm thu toàn paper cần review của nhóm.

## 1. Môi trường

Chạy từ gốc repo bằng Python từ 3.10, NumPy, OpenCV và Matplotlib. Dùng môi trường riêng nếu môi trường benchmark chưa có Matplotlib. Script dùng thư viện đã cài, không tự cài package.

```bash
python -m venv build/paper-venv
# Linux
. build/paper-venv/bin/activate
# Windows PowerShell dùng build/paper-venv/Scripts/Activate.ps1
python -m pip install -r tools/requirements-paper.txt
```

Cần có `pdfinfo`, `pdftotext`, `pdffonts` để audit font và metadata. Build manuscript cần thêm `pdflatex`, `bibtex`, `kpsewhich`, các package OASIcs trong [hướng dẫn paper](../../paper/README.md), và `booktabs`. `pdftoppm` dùng để xem lại layout. Không có công cụ thì dừng bước tương ứng, giữ nguyên gate kiểm tra.

## 2. Sinh artifact từ export đã khóa

Đặt hai thư mục export ở vị trí local. Mỗi thư mục chứa `export_verified.json` và `evidence/`. Không đổi nội dung export hoặc tự tính lại hash marker để chấp nhận dữ liệu đã sửa. Hai giá trị dưới đây lấy từ lần audit đã ghi trong [báo cáo nghiệm thu](../reports/phase-3-acceptance-report.md).

```bash
python -m tools.analysis.generate_paper_artifacts \
  --ch02 specs/run_20261106_164958Z_ch02_inu6unq8 \
  --ch02-sha256 857bb3ecf2c93d69100772d4e1ab1e3d0503e8ea5d422259d145f7983132b865 \
  --render specs/run_20261106_043655Z_ch03_ch05_gbfq3y9z \
  --render-sha256 9ab6209f8e89809c2fa04cf205f4ceeeb67b8ff7ab6f3d9bb05a8be98dcf9679 \
  --output build/paper-artifacts
```

Thêm `--dry-run` để kiểm input mà chưa sinh bundle. Output thật phải là thư mục mới. Thư mục lock cùng cấp ngăn hai process ghi đồng thời. File được tạo bằng primitive atomic của repo trong staging, rồi publish cả bundle. Nếu process bị kill và để lại lock, xác minh process đã kết thúc trước khi dọn lock. Chạy lại với thư mục output mới.

Input CSV, manifest, receipt và PNG được đối chiếu với hash trong export đã pin. Script đọc streaming `joined.csv`, kiểm đủ repeat và mẫu số. Bảng và macro định lượng sinh từ CSV hoặc report đã kiểm lại với CSV. Không đọc số trong Markdown để vẽ hình.

## 3. Các output và giới hạn

| Output | Phạm vi |
|---|---|
| `figures/fig_pareto.pdf`, `.svg` | Điểm grid đầy đủ và frontier tính lại từ telemetry. Panel mode dùng median trên tập PASS riêng của mỗi mode. |
| `figures/fig_latency_cdf.pdf`, `.svg` | CDF lấy trực tiếp CSV đã xuất, kiểm lại với pair ledger. Mẫu số PASS của từng mode nằm trên legend. |
| `figures/fig_latency_breakdown.pdf`, `.svg` | Stage medians trên giao PASS ba mode. A4 tách CUDA interval, host invocation và device setup. |
| `figures/fig_qualitative_crops.pdf`, `.svg` | Hai crop theo visual plan và một crop limitation hậu kiểm từ panorama đã chọn. |
| `tables/*.tex` | Booktabs, caption phía trên bảng, đơn vị và failure accounting. |
| `tables/ch_numbers.tex` | Macro định lượng dùng trong abstract và phần kết quả. |
| `sources/pareto_cohort.csv` | Bảng dẫn xuất đầy đủ cho hình Pareto, tách grid và mode-pass. |
| `sources/visual_plan.json` | Pair IDs, bbox, RGB convention và phạm vi lựa chọn crop. |
| `claims.json`, `claim_evidence_map.md` | Claim, giá trị, truy vấn và nguồn dùng cho từng macro hoặc bảng. |
| `figure_audit.json` | Kết quả kiểm font TrueType embed và metadata/text ẩn danh. |
| `artifact_manifest.json` | Hash, byte count, row count, input inventory, source code và phiên bản môi trường. |

CSV `pareto_cohort.csv` gốc chỉ chứa frontier. Tooling tái tạo tất cả điểm grid từ joined telemetry và yêu cầu frontier khớp bản gốc trước khi vẽ. Trục y dùng số cặp QoR PASS vì schema gốc không có median SSIM/PSNR theo cấu hình. Dominance giữ ba mục tiêu đã có trong CH-02, gồm latency, execution OK và QoR PASS. Panel hai chiều có thể che một chiều dominance. Caption ghi rõ điều này.

Fixed-Low là cấu hình có bộ `(S, K, R)` nhỏ nhất theo thứ tự từ điển trong grid. Fixed-Best trùng Fixed-High trong development lock, nên dùng chung marker. Mode Adaptive và Oracle nằm ở panel riêng vì median chỉ dùng các cặp PASS riêng của chúng. Không suy ra paired speedup bằng cách chia median trên các tập khác nhau.

Không có số đo allocator riêng trong CH-02. Bar chart chỉ trình bày timing đã quan sát và A4 device-buffer setup. Stage medians không cộng thành host median. Workspace query không phải peak VRAM.

Manuscript giữ Pareto, CDF và qualitative crops. Biểu đồ timing chi tiết vẫn nằm trong bundle. Phần thân tóm tắt quan sát stage trên cùng cohort để dành trang cho kết quả và giới hạn registration.

Crop thứ ba dùng bbox `[140,155,280,185]` trên panorama `009583` trong visual plan gốc. Đây là lựa chọn hậu kiểm để minh họa viền kép quanh xe trong Feather. Không gán nguyên nhân parallax hoặc low texture từ hình này. Mỗi hàng giữ cùng bbox và display scale giữa ba B. Các hàng có kích thước crop khác nhau. PDF/SVG của hình crop chứa raster RGB trong container, còn biểu đồ số liệu dùng vector.

## 4. Verify và tái tạo một lệnh

```bash
python -m tools.release.freeze_artifacts --root build/paper-artifacts
sh build/paper-artifacts/reproduce_paper_artifacts.sh \
  --ch02 specs/run_20261106_164958Z_ch02_inu6unq8 \
  --ch02-sha256 857bb3ecf2c93d69100772d4e1ab1e3d0503e8ea5d422259d145f7983132b865 \
  --render specs/run_20261106_043655Z_ch03_ch05_gbfq3y9z \
  --render-sha256 9ab6209f8e89809c2fa04cf205f4ceeeb67b8ff7ab6f3d9bb05a8be98dcf9679 \
  --output build/paper-artifacts
```

Trên PowerShell, chạy lại lệnh Python ở mục 2 với `--verify-only`. Verify sinh lại trong thư mục tạm và đối chiếu hash từng output. Cùng input, source và plotting environment phải tái tạo byte giống nhau. Thay NumPy, Matplotlib, Python, platform hoặc code thì tạo freeze mới và review diff. `--verify-only` giữ nguyên bundle đã khóa.

## 5. Build và review manuscript

Trên Linux/WSL có TeX Live, bổ sung đường dẫn artifact vào `TEXINPUTS` rồi chạy build hiện có.

```bash
export TEXINPUTS="$PWD/build/paper-artifacts:$TEXINPUTS"
python paper/build.py --output build/paper-ch07
pdftoppm -png -r 100 build/paper-ch07/main.pdf build/paper-ch07/page
```

`main.tex` input macro, bảng và hình bằng tên tương đối. `paper/build.py` thêm template gốc vào lookup, giữ `TEXINPUTS` caller cung cấp. Thiếu artifact thì build dừng. Review toàn bộ trang render, caption, font, references, metadata, độ dài và claim–evidence map trước nghiệm thu. PDF sinh từ source hiện tại chỉ mô tả campaign đã pin, không xác nhận runtime đã merge được đo lại trên T4.

Gói artifact và PDF nằm trong `build/`. Dataset, export thô, output notebook và kế hoạch cá nhân giữ ngoài commit. CH-08 chỉ đóng toàn bộ khi các owner đã bàn giao đủ bằng chứng, A5 được xử lý theo quyết định nhóm và phiên bản manuscript cuối qua review.
