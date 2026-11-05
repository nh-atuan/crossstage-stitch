# Đặc tả 5 bài thí nghiệm bóc tách (Ablation Studies) — Phase 3

- **Tài liệu tham chiếu:** [Kế hoạch tổng thể (docs/plan.md)](../plan.md), [Báo cáo nghiệm thu Phase 2](../reports/phase-2-review-report.md)
- **Mục tiêu:** Định lượng độc lập đóng góp của từng thành phần kiến trúc và thuật toán trong pipeline thích nghi (Adaptive Runtime Pipeline), cung cấp bằng chứng thực nghiệm vững chắc cho Section 5 của bài báo tại workshop PARMA-DITAM 2027.
- **Nguyên tắc phân công:** Mỗi bài thí nghiệm bóc tách có **đúng 1 người chủ trì duy nhất (Single Owner)** chịu trách nhiệm thực thi, lấy số liệu và viết phân tích.

---

## 1. Bảng tổng hợp 5 bài thí nghiệm bóc tách

| Mã | Tên bài Ablation | Mục tiêu khoa học | Biến thể đối đầu (Variants) | Chỉ số đo lường chính | Người chủ trì duy nhất |
|---|---|---|---|---|---|
| **A1** | **Hiệu quả của Probe Signal (S1 Gradient)** | Đo lường xem chi phí probe (~0.07 ms) có thực sự giúp giảm số lần retry và tiết kiệm latency tổng thể hay không | 1. **Adaptive with Probe** (đề xuất)<br>2. **Blind Escalation** (luôn bắt đầu từ nấc 1 nhẹ nhất, thiếu inliers thì leo thang)<br>3. **Conservative High** (luôn chạy nấc tối đa) | - $t_{\text{probe}}$, $t_{\text{controller}}$, $t_{\text{e2e}}$<br>- Số attempt trung bình/pair<br>- Tỉ lệ retry (%) | **Tuấn** |
| **A2** | **Đóng góp của State/Match Reuse trong S5** | Định lượng thời gian tiết kiệm được khi tăng $R$ nhờ việc tái sử dụng match S4 và inliers S5 cũ | 1. **With Reuse** (đề xuất: giữ nguyên match S4, chỉ mở rộng RANSAC evaluation từ $R_{\text{old}}$ đến $R_{\text{new}}$)<br>2. **No Reuse** (chạy lại toàn bộ S4 matching và S5 RANSAC từ đầu) | - Thời gian chạy lại $t_{\text{retry}}$<br>- Delta latency tiết kiệm được (ms)<br>- Tốc độ cải thiện của attempt 2/3 | **Đế** |
| **A3** | **Tác động của Ngân sách $K$ động trong S2–S3** | Đo đạc độ co giãn thời gian của FAST/ORB theo núm vặn $K$ và kiểm chứng tính ổn định bộ nhớ (zero-allocation với view `capacity = K`) | 1. **Fixed pre-allocated buffer** (đề xuất: cấp phát một lần theo $K_{\max}=4000$, truyền view `capacity = K` động theo từng attempt)<br>2. **Dynamic re-allocation / Unbounded capacity** (cấp phát lại theo từng $K$ hoặc giữ `capacity = K_max` cố định)<br>3. Quét $K \in \{500, 1000, 2000, 4000\}$ | - Kernel time $t_{s2}$, $t_{s3}$<br>- Memory allocation / idle-block overhead<br>- Số lượng keypoints hợp lệ | **Thành** |
| **A4** | **Same-Homography Blending Trade-off (S6)** | Định lượng sự đánh đổi giữa latency render và chất lượng cảm nhận vùng biên (seam quality) | Cố định cùng 1 homography $H$ đã chuẩn hóa:<br>1. **Feather Blending**<br>2. **Multiband 3-level**<br>3. **Multiband 5-level** | - Kernel time $t_{s6}$ (render + pyramid)<br>- VRAM footprint của workspace<br>- Overlap SSIM / PSNR<br>- Hiện tượng seam/ghosting | **Chương** |
| **A5** | **Độ nhạy của Ngưỡng Chất lượng ($\delta_S, \delta_P$)** | Khảo sát phản ứng của Controller và Oracle khi người dùng siết chặt hoặc nới lỏng dung sai chất lượng | Quét lưới dung sai trên tập test UDIS-D:<br>- $\delta_S \in [0.01, 0.05]$ (bước 0.01)<br>- $\delta_P \in [0.2, 1.0\text{ dB}]$ (bước 0.2 dB) | - Tỉ lệ đạt QoR (pass rate %)<br>- Tỉ lệ các nấc ladder được chọn<br>- Speedup so với Fixed-Best | **Chương** |

---

## 2. Chi tiết đặc tả từng bài thí nghiệm

```mermaid
flowchart TD
    subgraph S1_S3["Front-end Stages"]
        A1["Ablation A1: S1 Probe Signal<br/>(Chủ trì: Tuấn)"]
        A3["Ablation A3: S2-S3 Dynamic K<br/>(Chủ trì: Thành)"]
    end
    subgraph S4_S5["Matching & Estimation"]
        A2["Ablation A2: S5 Match/State Reuse<br/>(Chủ trì: Đế)"]
    end
    subgraph S6_QoR["Back-end & Policy"]
        A4["Ablation A4: Same-H Blending Modes<br/>(Chủ trì: Chương)"]
        A5["Ablation A5: Quality Threshold Sensitivity<br/>(Chủ trì: Chương)"]
    end
    A1 --> Paper["Section 5: Experimental Evaluation"]
    A2 --> Paper
    A3 --> Paper
    A4 --> Paper
    A5 --> Paper
```

### 2.1. Ablation A1: Hiệu quả của Probe Signal (S1 Gradient)

- **Người chủ trì duy nhất:** Tuấn

- **Bối cảnh:** Controller đề xuất sử dụng kernel S1 trích xuất độ biến thiên gradient trung bình trên ảnh probe thu nhỏ (scale 0.25, chi phí $<0.1\text{ ms}$) để chọn mức cấu hình ban đầu (initial configuration).
- **Câu hỏi khoa học:** Chi phí probe $t_{\text{probe}} \approx 0.07\text{ ms}$ có thực sự giúp giảm số vòng lặp retry và tiết kiệm latency tổng hay không so với việc luôn khởi đầu mù (blind) từ nấc nhẹ nhất?
- **Phương pháp thực hiện:**
  - Chạy `stitch_harness` trên toàn bộ tập test UDIS-D ở 3 chế độ:
    1. *Adaptive with Probe:* Sử dụng gradient probe để phân loại cảnh (dễ $\to$ Level 1, thường $\to$ Level 2, phức tạp $\to$ Level 3).
    2. *Blind Escalation:* Bỏ qua probe, 100% các cặp ảnh đều bắt đầu từ Level 1 ($S=0.5, K=1000, R=128$); nếu S5 không đủ inliers thì leo thang tuần tự lên Level 2, 3, 4.
    3. *Conservative High:* Luôn chạy ở cấu hình an toàn cao nhất ($S=1.0, K=2000, R=1024$).
- **Đầu ra bàn giao:**
  - Bảng so sánh 3 chế độ: Mean/Median $t_{\text{e2e}}$, tỉ lệ retry (%), và số attempt trung bình/pair.
  - Đoạn phân tích kỹ thuật cho Section 5.2 minh chứng giá trị của Scene-Awareness.

---

### 2.2. Ablation A2: Đóng góp của State/Match Reuse trong S5

- **Người chủ trì duy nhất:** Đế

- **Bối cảnh:** Khi Controller phát hiện S5 thiếu inliers nhưng cấu hình hiện tại chỉ cần tăng số vòng lặp RANSAC $R$ (ví dụ từ $R=256$ lên $512$ hoặc $1024$), danh sách match của S4 hoàn toàn không đổi.
- **Câu hỏi khoa học:** Cơ chế tái sử dụng match của S4 và mở rộng đánh giá giả thuyết trên GPU tiết kiệm được bao nhiêu ms so với việc chạy lại từ đầu?
- **Phương pháp thực hiện:**
  - Lọc ra các cặp ảnh kích hoạt escalation trong tập UDIS-D test set.
  - So sánh thời gian thực thi của attempt leo thang ở 2 trường hợp:
    1. *With Reuse:* Giữ nguyên buffer descriptor match trong VRAM, kernel S5 chỉ sinh thêm $R_{\text{new}} - R_{\text{old}}$ giả thuyết và cập nhật best consensus model.
    2. *No Reuse:* Giải phóng buffer, chạy lại toàn bộ kernel S4 Hamming matching và S5 RANSAC từ vòng 0 đến $R_{\text{new}}$.
- **Đầu ra bàn giao:**
  - Bảng đối sánh độ trễ $t_{\text{s4}} + t_{\text{s5}}$ giữa 2 trường hợp.
  - Biểu đồ minh họa phần trăm thời gian tiết kiệm được ở các nấc retry.

---

### 2.3. Ablation A3: Tác động của Ngân sách K động trong S2-S3

- **Người chủ trì duy nhất:** Thành

- **Bối cảnh:** Khác với các pipeline truyền thống phải cấp phát lại buffer GPU mỗi khi thay đổi số lượng keypoints $K$, kiến trúc S2–S3 của nhóm cấp phát một bộ buffer/workspace cố định một lần theo $K_{\max}=4000$ và kích thước ảnh lớn nhất ngay từ đầu (pre-allocated), sau đó truyền tham số $K$ động kèm view `Keypoints` có `capacity = K` vào từng attempt (xem [quyết định K động](../decisions/s2-s3-dynamic-k.md)).
- **Câu hỏi khoa học:** Thời gian thực thi của S2 (FAST + Harris) và S3 (ORB) đánh đổi (trade-off) và co giãn như thế nào khi vặn núm $K \in \{500, 1000, 2000, 4000\}$? Cơ chế zero-allocation kết hợp view `capacity = K` giúp triệt tiêu hiện tượng jitter cấp phát và tránh phóng thừa block ở S3/S4 ra sao?
- **Phương pháp thực hiện:**
  - Quét qua các giá trị $K \in \{500, 1000, 2000, 4000\}$ trên lưới tỉ lệ $S \in \{0.25, 0.5, 0.75, 1.0\}$ (hoặc cố định $S=0.5$ cho đường cong chi tiết từng kernel).
  - Đo đạc thời gian $t_{s2}$, $t_{s3}$ (và chi tiết các bước FAST, Harris rank $O(K^2)$, rBRIEF).
  - Đối sánh giữa 3 chế độ: (1) cấp phát một lần theo $K_{\max}=4000$ kèm view `capacity = K`, (2) cấp phát một lần nhưng giữ `capacity = K_max` (gây thừa block ở S3 khi $K$ nhỏ), và (3) cấp phát vừa đúng theo từng $K$ (`cudaMalloc`/`cudaFree`).
- **Đầu ra bàn giao:**
  - Bảng phân bổ thời gian chi tiết của S2 và S3 cùng footprint bộ nhớ theo từng nấc $K \in \{500, 1000, 2000, 4000\}$.
  - Đồ thị đường cong thể hiện sự đánh đổi (trade-off) giữa ngân sách $K$ và thời gian trích xuất đặc trưng.

---

### 2.4. Ablation A4: Same-Homography Blending Trade-off trong S6

- **Người chủ trì duy nhất:** Chương

- **Bối cảnh:** Stage 6 hỗ trợ 3 chế độ blending: Feather (nhẹ), Multiband 3 mức, và Multiband 5 mức.
- **Câu hỏi khoa học:** Chi phí tính toán và bộ nhớ của Multiband tăng lên gấp mấy lần so với Feather? Sự cải thiện về chất lượng ghép hình ở vùng biên (seamline) có tương xứng với độ trễ bỏ ra không?
- **Phương pháp thực hiện:**
  - Sử dụng công cụ `tools/bench/same_h.py` trên các cặp ảnh UDIS-D có độ phức tạp thị giác cao (độ tương phản sáng tối mạnh, parallax nhẹ).
  - Cố định cùng một homography $H$, chỉ tráo đổi chế độ blending của S6.
  - Đo đạc: $t_{\text{s6}}$ (thời gian kernel render trên CUDA stream), bộ nhớ tạm (workspace memory), Overlap SSIM, Overlap PSNR.
  - Xuất 2 cặp ảnh crop phóng to vùng đường nối (seam) để kiểm tra hiện tượng bóng mờ (ghosting).
- **Đầu ra bàn giao:**
  - Bảng so sánh 3 chế độ: Latency S6 (ms), Peak VRAM (MB), Overlap SSIM/PSNR.
  - Hình ảnh trực quan so sánh vết ghép ở vùng biên cho bài báo.

---

### 2.5. Ablation A5: Độ nhạy của Ngưỡng Chất lượng Overlap

- **Người chủ trì duy nhất:** Chương

- **Bối cảnh:** Chính sách QoR xác định một cấu hình $c$ là hợp lệ nếu $\mathrm{SSIM}(c) \ge \mathrm{SSIM}_{\mathrm{ref}} - \delta_S$ và $\mathrm{PSNR}(c) \ge \mathrm{PSNR}_{\mathrm{ref}} - \delta_P$.
- **Câu hỏi khoa học:** Khi người dùng siết chặt yêu cầu chất lượng ($\delta_S=0.01$) hoặc nới lỏng ($\delta_S=0.05$), hành vi chọn cấu hình của Controller và trần lý thuyết của Oracle thay đổi như thế nào?
- **Phương pháp thực hiện:**
  - Chạy công cụ đánh giá QoR `tools/qor/evaluate.py` quét qua lưới tham số:
    - $\delta_S \in \{0.01, 0.02, 0.03, 0.04, 0.05\}$
    - $\delta_P \in \{0.2, 0.5, 0.8, 1.0\text{ dB}\}$
  - Thống kê trên toàn bộ 1.105 cặp ảnh test UDIS-D: tỉ lệ cặp ảnh tìm được cấu hình đạt chuẩn (feasible rate), phân bố nấc ladder của Controller, và tốc độ trung bình của Oracle.
- **Đầu ra bàn giao:**
  - Đồ thị đường cong độ nhạy: Trục hoành là $\delta_S$, trục tung là Mean Latency của Controller và Oracle.
  - Bảng tóm tắt kết luận: Ngưỡng $\delta_S=0.02, \delta_P=0.5\text{ dB}$ hiện tại là điểm cân bằng tối ưu giữa chất lượng hình ảnh và latency.
