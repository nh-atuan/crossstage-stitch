# Quyết định: môi trường riêng cho tooling quantum (Q10)

- **Task:** Q10 trong [kế hoạch quantum](../research/quantum-integration-tasks.md), Phase 3.
- **Người quyết định:** Thành.
- **Ngày ghi nhận:** 04/11 (Thứ 4), năm 2026.
- **Phase:** 3; deadline chung 05/11 (Thứ 5).
- **Trạng thái:** Đã chọn và đã cài trên máy local của Thành. Import, smoke test Aer, CP-SAT và test Python đã chạy; chưa thử trên máy khác.

## 1. Bối cảnh

Proposal (mục 8) yêu cầu một môi trường Python 3.12 tách khỏi pipeline, pin Qiskit, Qiskit Aer và OR-Tools vào `tools/quantum/requirements-lock.txt`. Pipeline không được import các package này. Máy local chỉ có Python 3.11 (môi trường pipeline, đã có sẵn qiskit 2.5.2 và qiskit-aer 0.17.2 nhưng không có OR-Tools) và Python 3.13; không có Python 3.12.

## 2. Phương án chọn

Dùng venv Python 3.13.7 đặt ngoài repo, tại `%USERPROFILE%\.venvs\crossstage-quantum`. Version đã cài và chạy được:

| Package | Version | License khai báo trong metadata |
|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| qiskit | 2.6.1 | Apache-2.0 |
| qiskit-aer | 0.18.2 | Apache Software License |
| ortools | 9.15.6755 | Apache Software License |
| pytest | 9.2.0 | MIT |

Lockfile ghi đủ 22 package (gồm cả phụ thuộc gián tiếp) kèm SHA-256 của wheel win_amd64. Hash lấy từ `pip install --dry-run --ignore-installed --report`, rồi kiểm lại bằng `pip install --dry-run --require-hashes`. `tools/quantum/env_receipt.py` ghi OS, CPU, RAM, version và license của từng package vào `env-receipt.json` (lưu trong `data/`, không commit). Không sửa `tools/requirements.txt` hay CMake.

Qiskit và OR-Tools chỉ được import bên trong `qaoa.py` và `classical.cpsat`. Các module còn lại chỉ cần NumPy, nên chạy được trong môi trường pipeline. `tests/python/test_quantum_no_sdk.py` kiểm điều này bằng cách chặn import ba package trên.

## 3. Trade-off và phương án phụ

| Phương án | Ưu điểm | Nhược điểm | Vì sao không chọn |
|---|---|---|---|
| venv Python 3.13 (chọn) | Có sẵn trên máy; có wheel cho cả bốn package; tách khỏi pipeline | Lệch version 3.12 của proposal | Chọn; ghi lệch ở đây |
| Cài thêm Python 3.12 | Đúng proposal | Thêm một bản cài đặt hệ thống; proposal không nêu lý do kỹ thuật phải là 3.12 | Không cần cho MVP |
| Dùng Python 3.11 global | Nhanh, đã có Qiskit và Aer | Không tách khỏi pipeline; version Qiskit trong đó do cài tay, không pin | Trái yêu cầu tách môi trường |

## 4. Fallback

Nếu người khác không cài được lockfile (ví dụ trên Linux hay Colab), cài cùng version nhưng không dùng `--require-hashes`, sinh lại hash cho nền tảng đó và ghi vào receipt. Nếu một package không có wheel cho Python 3.13 trên nền tảng đó thì dùng Python 3.12 với cùng version. QAOA không chạy trên T4 để tránh tranh tài nguyên với campaign đo.

## 5. Kiểm chứng

- Smoke test: circuit Bell 2 qubit trên `AerSimulator`, 1000 shot, cho `{'00': 482, '11': 518}`. CP-SAT giải một model 1 biến và trả về `OPTIMAL`.
- `pip install --dry-run --require-hashes -r tools/quantum/requirements-lock.txt` resolve được.
- Venv: `python -m pytest tests/python/test_quantum_*.py -q` cho 44 pass, 1 skip. Test bị skip là test characterize, vì fixture của nó cần `cv2` của pipeline.
- Python 3.11 của pipeline: `python -m pytest tests/python -q` cho 384 pass, 2 skip. Hai test cần OR-Tools bị skip.
- Chưa kiểm: cài lockfile trên máy khác.

## 6. Nguồn

- [Qiskit](https://github.com/Qiskit/qiskit), [Qiskit Aer](https://github.com/Qiskit/qiskit-aer), [OR-Tools](https://github.com/google/or-tools): package cài từ PyPI ngày 04/11 (Thứ 4); license đọc từ metadata đã cài, theo đối chiếu D4–D6 trong [proposal](../research/quantum-integration-proposal.md) mục 12.
