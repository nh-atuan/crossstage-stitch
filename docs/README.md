# Mục lục tài liệu dự án

Tài liệu kỹ thuật của workshop paper PARMA-DITAM 2027 được tổ chức theo miền chức năng nhằm phục vụ tra cứu, phát triển và nghiệm thu độc lập giữa các thành viên.

## 1. Cấu trúc thư mục

```text
docs/
├── README.md                              # Mục lục và sơ đồ tổ chức tài liệu (file này)
├── plan.md                                # Kế hoạch tổng thể, phân vai, timeline các phase
├── conventions.md                         # Quy ước chung: ABI, cấu trúc repo, file kết quả, Git
├── stages.md                              # Kiến trúc tổng quan 6 stage của pipeline stitching
├── decisions/                             # Architecture Decision Records (ADR)
│   ├── TEMPLATE.md                        # Mẫu ghi quyết định kỹ thuật
│   ├── benchmark-qor-s4-s5.md            # Quyết định benchmark UDIS-D, QoR và phương pháp S4/S5 (Đế)
│   ├── s2-s3-orb.md                       # Quyết định thuật toán FAST/Harris/rBRIEF cho S2–S3 (Thành)
│   ├── s2-s3-dynamic-k.md                 # Quyết định cấp phát dynamic K một lần cho S2–S3 (Thành)
│   ├── p2-de-controller.md               # Quyết định thiết kế và ladder escalation của Controller (Đế)
│   └── p2-chuong-qor-blend.md             # Quyết định tích hợp QoR và multiband blend cho S6 (Chương)
├── stages/                                # Đặc tả kỹ thuật, giao diện và nghiệm thu theo từng stage
│   ├── s1-preprocessing.md                # Đặc tả thuật toán và bộ đệm S1 Preprocessing (Tuấn)
│   ├── s2-s3-implementation.md            # Hiện thực chi tiết S2 Keypoints & S3 Descriptors (Thành)
│   ├── s2-s3-validation.md                # Bằng chứng kiểm chứng và so sánh bit-exact S2–S3 trên Colab T4 (Thành)
│   ├── s4-s5-implementation.md            # Hiện thực S4 Matching & S5 RANSAC Homography (Đế)
│   ├── s4-s5-validation.md                # Bằng chứng kiểm chứng S4–S5 trên Colab T4 (Đế)
│   ├── s6-warp-blend.md                   # Phương pháp warping, inverse mapping và blending cho S6 (Chương)
│   ├── s6-p1-implementation.md            # Bàn giao triển khai S6 Phase 1 (Chương)
│   ├── s6-p2-implementation.md            # Hiện thực mở rộng S6 Phase 2: QoR feedback & blend modes (Chương)
│   ├── controller-contract.md             # Đặc tả giao diện Contract v1 của Adaptive Controller (Đế)
│   └── controller-p2-implementation.md    # Bàn giao triển khai Controller Phase 2 (Đế)
├── experiments/                           # Thiết kế thực nghiệm và thí nghiệm bóc tách (Phase 3)
│   └── ablation-studies.md                # Đặc tả 5 bài thí nghiệm bóc tách A1–A5 (Đế, Chương, Thành)
├── reports/                               # Báo cáo các phase và cẩm nang vận hành
│   ├── phase-1-gonogo-report.md           # Báo cáo kết quả Go/No-Go Phase 1 (sweep S×K×R)
│   ├── phase-2-review-report.md           # Báo cáo review và pilot evaluation Phase 2
│   └── sweep-gonogo-guide.md              # Cẩm nang thực thi chạy sweep cấu hình và Go/No-Go
└── research/                              # Lý thuyết nền tảng và nghiên cứu mở rộng
    ├── workshop-foundations.md            # Lý thuyết hình học, toán học và cẩm nang triển khai paper
    ├── quantum-integration-proposal.md    # Đề xuất tích hợp QUBO/QAOA chọn catalogue cấu hình offline
    └── quantum-integration-tasks.md       # Danh sách nhiệm vụ và backlog nghiên cứu quantum
```

## 2. Bảng tra cứu tài liệu theo vai trò

| Nhóm tài liệu | Đường dẫn | Nội dung chính | Người phụ trách chính |
|---|---|---|---|
| **Cốt lõi (Core)** | [plan.md](plan.md) | Kế hoạch cuốn chiếu, câu hỏi nghiên cứu, phân vai và deadline | Toàn nhóm |
| | [conventions.md](conventions.md) | Struct dùng chung, điểm đồng bộ stream, quy ước hình học và telemetry | Tuấn |
| | [stages.md](stages.md) | Tổng quan 6 stage, luồng dữ liệu và vai trò runtime controller | Toàn nhóm |
| **Quyết định (ADR)** | [decisions/](decisions/) | Lưu trữ các quyết định kỹ thuật theo mẫu `TEMPLATE.md` | Từng thành viên |
| **Stage S1** | [stages/s1-preprocessing.md](stages/s1-preprocessing.md) | Downsample bilinear, RGB to Grayscale, ma trận co giãn $D$ | Tuấn |
| **Stage S2–S3** | [stages/s2-s3-implementation.md](stages/s2-s3-implementation.md)<br>[stages/s2-s3-validation.md](stages/s2-s3-validation.md) | FAST-9, Harris response, rBRIEF 32-byte, dynamic $K$, kiểm chứng bit-exact T4 | Thành |
| **Stage S4–S5** | [stages/s4-s5-implementation.md](stages/s4-s5-implementation.md)<br>[stages/s4-s5-validation.md](stages/s4-s5-validation.md) | Brute-force Hamming match, RANSAC 4-point homography, inlier mask | Đế |
| **Stage S6** | [stages/s6-warp-blend.md](stages/s6-warp-blend.md)<br>[stages/s6-p1-implementation.md](stages/s6-p1-implementation.md)<br>[stages/s6-p2-implementation.md](stages/s6-p2-implementation.md) | Warp phối cảnh ngược, feather blend, multiband pyramid blend | Chương |
| **Controller** | [stages/controller-contract.md](stages/controller-contract.md)<br>[stages/controller-p2-implementation.md](stages/controller-p2-implementation.md) | Interface ABI, ladder escalation policy, tái sử dụng kết quả S5 | Đế |
| **Thực nghiệm (Experiments)** | [experiments/ablation-studies.md](experiments/ablation-studies.md) | Đặc tả 5 bài thí nghiệm bóc tách (A1–A5) cho Phase 3 | Toàn nhóm |
| **Báo cáo & Vận hành** | [reports/phase-1-gonogo-report.md](reports/phase-1-gonogo-report.md)<br>[reports/phase-2-review-report.md](reports/phase-2-review-report.md)<br>[reports/sweep-gonogo-guide.md](reports/sweep-gonogo-guide.md) | Báo cáo Go/No-Go P1, báo cáo nghiệm thu P2, hướng dẫn sweep harness | Toàn nhóm |
| **Nghiên cứu & Mở rộng** | [research/workshop-foundations.md](research/workshop-foundations.md)<br>[research/quantum-integration-proposal.md](research/quantum-integration-proposal.md)<br>[research/quantum-integration-tasks.md](research/quantum-integration-tasks.md) | Lý thuyết toán hình học; workstream QUBO/QAOA offline chạy song song với baseline không quantum, chốt ngày 02/11 (Thứ 2) | Thành chủ trì quantum; Chương review/tổng hợp; Tuấn/Đế hỗ trợ dữ liệu và integration đúng ownership |

## 3. Quy tắc cập nhật tài liệu

1. **Quyết định kỹ thuật mới:** Tạo file mới trong `docs/decisions/` theo mẫu [decisions/TEMPLATE.md](decisions/TEMPLATE.md), đặt tên file theo cú pháp `<stage>-<chu-de>.md` hoặc `p<phase>-<ten>-<chu-de>.md`.
2. **Cập nhật stage:** Tài liệu giải thuật hoặc bằng chứng đo kiểm đặt trong `docs/stages/`.
3. **Báo cáo mốc:** Kết thúc mỗi phase, báo cáo tổng kết được lưu trong `docs/reports/`.
4. **Đường dẫn liên kết:** Luôn sử dụng đường dẫn tương đối (relative path) và kiểm tra tính toàn vẹn của liên kết khi chuyển đổi hoặc tạo tài liệu mới.

