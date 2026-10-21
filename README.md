# CrossStage-Stitch

Repo của Nhóm cho workshop paper [PARMA-DITAM 2027](https://parma-ditam-workshop.github.io/). Paper nghiên cứu một runtime chọn lượng tính toán cho từng cặp ảnh trong pipeline ghép panorama trên GPU, dựa trên độ khó của cảnh, để giảm latency mà vẫn giữ chất lượng đạt một ngưỡng cho trước. Mọi build, test và đo đạc chạy trên Google Colab với GPU Tesla T4.

Mốc nộp bài của Nhóm: 13/11/2026.

## Cấu trúc thư mục

| Đường dẫn | Ý nghĩa |
|---|---|
| `README.md` | Giới thiệu chung về repo và ý nghĩa của các file, thư mục |
| `AGENTS_TEMPLATE.md` | Bộ quy tắc tối thiểu cho AI agent của từng thành viên. Mỗi người sao chép thành `AGENTS.md` hoặc `CLAUDE.md` của riêng mình và không commit file đó |
| `.gitignore` | Các file không đưa lên Git: file hướng dẫn agent cá nhân, dataset, output, file build, file tạm của LaTeX |
| `docs/plan.md` | Kế hoạch: mục tiêu, phân vai, nguyên tắc làm việc, lịch các phase, task Phase 0, Go/No-Go, dataset, môi trường, paper, rủi ro |
| `docs/stages.md` | Các stage của pipeline: mục tiêu, đầu vào, đầu ra, núm chỉnh, người phụ trách, controller |
| `docs/conventions.md` | Quy ước chung: cấu trúc repo, môi trường, interface giữa các stage, file kết quả, đo thời gian, Git |
| `docs/s1-preprocessing.md` | Quyết định kỹ thuật S1 |
| `docs/decisions/TEMPLATE.md` | Mẫu ghi quyết định kỹ thuật |
| `src/common/` | Header interface dùng chung |
| `tools/mock/` | Script sinh mock data và bản giả OpenCV của từng stage |
| `tests/` | Test CTest (C++/CUDA) và pytest (`tests/python/`) |

Cấu trúc thư mục đầy đủ của mã nguồn nằm trong [docs/conventions.md](docs/conventions.md).

## Bắt đầu

1. Đọc [docs/plan.md](docs/plan.md) để biết hướng đi, phân vai và task của phase hiện tại.
2. Đọc [docs/stages.md](docs/stages.md) để nắm pipeline và stage mình phụ trách.
3. Nếu dùng AI agent, sao chép [AGENTS_TEMPLATE.md](AGENTS_TEMPLATE.md) thành `AGENTS.md` hoặc `CLAUDE.md` ở thư mục gốc repo. Mỗi lần `git pull`, kiểm tra template có thay đổi không để cập nhật lại file của mình.
