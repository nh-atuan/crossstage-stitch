# Báo cáo pilot solver offline cho quantum catalogue (Q00–Q13)

- **Owner:** Thành.
- **Ngày chạy:** 05/11 (Thứ 5), năm 2026, trên CPU máy local của Thành; không chạy trên T4.
- **Nhánh:** `thanh/p3-quantum`.
- **Trạng thái nhãn:** pilot. Policy QoR `qor-pilot-v1` và `tools/quantum/protocol-pilot.json` chưa được Đế, Tuấn và Chương duyệt.
- **Kế hoạch và yêu cầu:** [quantum-integration-tasks.md](quantum-integration-tasks.md), [quantum-integration-proposal.md](quantum-integration-proposal.md).

Mọi bảng số dưới đây chép từ `data/quantum/report-tables.md`. File này do `python -m tools.quantum.report` sinh ra từ hai thư mục run. Thư mục `data/` không commit, nên mỗi run có `SHA256SUMS.json` để đối chiếu.

## 1. Kết quả chính

Lượt chính đạt **PASS_OFFLINE** theo định nghĩa ở mục 1.2 của kế hoạch: có instance thật với `q_after = 13 > 0`, QAOA p = 1 và p = 2 chạy đủ 10 seed, có counts xác nhận, và đủ các solver đối chứng. PASS ở đây không có nghĩa QAOA tốt hơn classical.

Trên instance này, mọi solver đều tìm ra optimum 17 cấu hình, kể cả greedy, SA, hai control ngẫu nhiên và repair-only xuất phát từ catalogue rỗng. Exhaustive giải phần còn lại sau reduction trong khoảng 30 µs. Xác suất QAOA trả về một minimum cover ở lượt xác nhận có median 0,031 (p = 1) và 0,032 (p = 2), khoảng gấp đôi xác suất 0,016 của một bitstring ngẫu nhiên đều. Theo Go/No-Go của Q13 ("nếu exact/greedy thắng hoặc repair-only tương đương, chốt feasibility/negative result và dừng mở rộng"), G3 không đạt và không nên mở rộng depth, seed hay QPU trên instance này. Nhóm chưa khóa criterion riêng cho G3, nên kết luận này dựa vào luật dừng của Q13.

Catalogue W1 (exhaustive) và W2 (QAOA, raw, không repair) gồm cùng 17 cấu hình.

Lượt phụ trên instance đầy đủ (q = 21, optimum 19) ghi `SECONDARY`. Cả 20 run QAOA chạm cap 60 s trước lượt xác nhận, nên nếu xét như lượt chính thì G2 không đạt.

## 2. Dữ liệu và provenance

| Thành phần | Giá trị |
|---|---|
| Nguồn | Output `sweep_calibration` (PR #16) do Đế chạy, file `data/sweep/v4-grid.zip` |
| Campaign | `calibration-grid-v2`, Tesla T4, seed 20261101, 2 warmup và 10 repeat |
| Phạm vi | 200 cặp calibration của UDIS-D train × 48 cấu hình, B = feather |
| `runs.csv` sha256 | `7758ed5f87bd692803226ea749516cebc097307b6866918f3c599594124e4238` |
| `qor.csv` sha256 | `b177284e513c023c7d2403fdada8c2272c830fc5d38caac5d8f7d8a0d37e94cc` |
| `configs.json` sha256 | `611cbd536a7d08411535084aec9a709483ac1ec53e0645ec168003de59b2ad8c` |
| Policy sha256 (bản CRLF local) | `579d0815eecd1bbf6c3d7276ec4922aee525b8c280ef756db038e835a5f5f37f` |
| Split dùng để chạy | `split-rebuilt.json`, sha256 `f1cf10a51eececea6afccc7539fdf8b8bbbe69b17cabb98bb386be2086da6c80` |
| Split ghi trong receipt | sha256 `275d7cba34414b29e82938d03b460b16a4bed064ee9a894f5d704b0879b3b85f`, không có trong zip |

Hash của runs, qor, configs và policy khớp `audit.inputs_sha256` ở chế độ `raw`. Zip không có `split.json` và không có sweep validation. Split được dựng lại bằng `freeze_split` của PR #16 từ `data/udis-d/udis-d-development-v1.json`. 200 cặp calibration của bản dựng lại trùng đúng 200 cặp trong audit, nhưng hash khác receipt, nên chưa chứng minh được hai file giống nhau từng byte (amendment A1).

`python -m tools.quantum.characterize` cho `completeness.json` ở trạng thái complete:

| Trạng thái dòng (pair, config) | Số dòng |
|---|---|
| `ok` | 8543 (3633 dòng pass QoR, 4910 dòng không pass) |
| `reference_failure` | 432, tức 9 cặp × 48 cấu hình |
| `runtime_failure` | 625 |
| Tổng | 9600 = 200 × 48 |

`runtime_failure` là các dòng có ít nhất một repeat lỗi ở S5 (`DEGENERATE`, `NOT_ENOUGH_POINTS`) hoặc S6 (`INSUFFICIENT_WORKSPACE`) trên cặp có reference ok. Các dòng này được tính là không pass, không bị bỏ khỏi mẫu số.

## 3. Môi trường

Venv Python 3.13.7 ngoài repo, theo [p3-quantum-environment.md](../decisions/p3-quantum-environment.md): qiskit 2.6.1, qiskit-aer 0.18.2 (AerSimulator statevector, CPU), ortools 9.15.6755, numpy 2.5.4. Máy chạy Windows 10, Intel 8 thread, RAM 8 GB (`data/quantum/env-receipt.json`). Simulator chạy trên CPU, không phải phần cứng quantum.

## 4. Protocol và amendment

Budget dùng đúng `protocol-pilot.json`, không override: 10 seed; QAOA 4 init × 100 objective call × 1024 shot cộng 4096 shot xác nhận; SA 4 restart × 1000 sweep; mỗi control ngẫu nhiên 413696 draw; cap 60 s cho mỗi seed và depth. Bốn amendment ngày 05/11 (Thứ 5) được ghi vào protocol trước khi chạy solver:

| ID | Nội dung |
|---|---|
| A1 | Dùng split dựng lại; khi có `split.json` gốc thì thay vào và chạy lại characterize |
| A2 | Không có sweep validation; partition đánh giá không có dòng nào được ghi `not_measured` |
| A3 | Thêm lượt phụ trên instance đầy đủ ở rho = 0,10, `q_max` nâng lên bằng `q_after` của instance đó, cùng budget, gate ghi `SECONDARY` và không thay kết quả chính |
| A4 | W2 là catalogue raw cover-feasible nhỏ nhất trên mọi run QAOA trong budget; hòa thì xét danh sách config đã sắp xếp, rồi p nhỏ hơn, rồi seed nhỏ hơn |

## 5. Chọn instance

Luật chọn instance đã khóa ngày 04/11 (Thứ 4): lấy rho nhỏ nhất có `0 < q_after ≤ 20` trên toàn bộ fit; nếu không có thì dùng restricted cohort theo thứ tự `sha256("20261104:" + pair_id)`.

| rho | Cặp trong I_f | q trước reduction | q sau reduction | Cột bị ép | Cột còn lại | Hàng còn lại |
|---|---|---|---|---|---|---|
| 0 | 191 | 50 | 0 | 28 | 0 | 0 |
| 0,05 | 191 | 230 | 0 | 24 | 0 | 0 |
| 0,10 | 191 | 376 | 21 | 15 | 11 | 8 |
| 0,20 | 191 | 522 | 26 | 11 | 14 | 9 |

Ở rho = 0 và 0,05, reduction giải hết. Ở rho = 0,10, q = 21 vượt `q_max = 20`, nên luật chuyển sang restricted cohort ở rho = 0,10 và giảm dần số cặp. Từ 199 xuống 181 cặp, q vẫn là 21. Ở 180 cặp (173 cặp thuộc I_f), q giảm còn 13 và instance được nhận. Danh sách 180 cặp nằm trong `selection.json` của lượt chính.

Reduction trên instance được chọn chạy theo thứ tự: ép 11 cột singleton (149 hàng), bỏ 8 hàng trùng, bỏ 7 hàng superset, bỏ 22 cột zero, gộp 5 cột trùng, rồi ép thêm 3 cột. Kết quả còn 7 cột, 6 hàng, mỗi hàng có degree 2, cộng 14 cột bị ép. Nếu tắt gộp cột trùng thì q tăng lên 26.

## 6. Lượt chính: restricted, rho = 0,10, q = 13

### 6.1. QUBO và tài nguyên

| Đại lượng | Giá trị |
|---|---|
| Biến y / slack / tổng q | 7 / 6 / 13 |
| `k_ub` từ greedy, lambda, alpha | 3, 1, 4 |
| Số coupling, mật độ, degree lớn nhất | 18, 0,231, 4 |
| Dynamic range của hệ số | 2,333 |
| Statevector | 131072 byte |
| Optimum sau restore (exhaustive, CP-SAT có reduction, CP-SAT không reduction) | 17, 17, 17 |
| Số minimum cover trên instance đã reduce | 2 |

Ở cả năm giá trị alpha của grid sensitivity, kể cả hai giá trị stress-only (tỉ lệ 0,5 và 1 so với lambda·k_ub), ground state của QUBO đều gồm 2 trạng thái và cả hai đều là minimum cover. G1 đạt.

### 6.2. Solver không có seed

| Solver | Trạng thái | Kích thước | Gap | Fit QoR coverage | Thời gian |
|---|---|---|---|---|---|
| exhaustive, có reduction | OPTIMAL | 17 | 0 | 1,000 | 2,99e-05 s |
| CP-SAT, có reduction | OPTIMAL | 17 | 0 | 1,000 | 0,011 s |
| greedy, có reduction | FEASIBLE | 17 | 0 | 1,000 | 5,98e-05 s |
| CP-SAT, không reduction | OPTIMAL | 17 | 0 | 1,000 | 0,003 s |
| greedy, không reduction | FEASIBLE | 17 | 0 | 1,000 | 0,002 s |
| repair-only từ catalogue rỗng | ok | 17 | 0 | 1,000 | 0,002 s |

Fit QoR coverage tính trên toàn bộ 191 cặp I_f của partition calibration, gồm cả 18 cặp I_f nằm ngoài cohort 180 cặp. Validation ghi `not_measured` ở mọi dòng.

### 6.3. Solver có seed (10 seed mỗi solver)

| Solver | Run có raw cover | Kích thước raw (min–max) | Raw cover-feasible probability, median (min–max) | Cột repair phải thêm, max | Thời gian median |
|---|---|---|---|---|---|
| SA | 10/10 | 17–17 | 1,000 (1,000–1,000) | 0 | 0,177 s |
| random_qubo | 10/10 | 17–17 | 0,274 (0,273–0,275) | 0 | 0,027 s |
| random_y | 10/10 | 17–17 | 0,437 (0,436–0,437) | 0 | 5,094 s |
| QAOA p = 1 | 10/10 | 17–17 | 0,670 (0,234–0,684) | 0 | 8,156 s |
| QAOA p = 2 | 10/10 | 17–17 | 0,664 (0,579–0,975) | 0 | 9,883 s |

Với SA và hai control ngẫu nhiên, xác suất tính trên mọi sample của run. Với QAOA, xác suất chỉ tính trên 4096 shot xác nhận. Catalogue raw tốt nhất của QAOA lấy từ shot xác nhận cộng với sample tốt nhất gặp trong trace optimizer. Ở mọi run QAOA, shot xác nhận đã chứa ít nhất một minimum cover.

### 6.4. QAOA

Cả 20 run có trạng thái `ok`, chạy đủ 401/401 call (gồm call xác nhận) và 413696/413696 shot, không run nào chạm cap 60 s. Circuit p = 1 có 18 cổng 2 qubit, depth 10; p = 2 có 36 cổng 2 qubit, depth 17. Peak RSS của tiến trình không vượt 237 MiB.

| Depth | P(minimum cover) ở shot xác nhận, median | min | max | Run thấp hơn bitstring đều | Median / bitstring đều |
|---|---|---|---|---|---|
| p = 1 | 0,031 | 0,009 | 0,036 | 3/10 | 2,0 |
| p = 2 | 0,032 | 2,44e-04 | 0,044 | 2/10 | 2,0 |

Bitstring đều trên 13 qubit là minimum cover với xác suất 2 × 2^6 / 2^13 = 0,016 (2 minimum cover, mỗi cover đi với 2^6 cách đặt slack). Năm run thấp hơn mức này là p = 1 seed 0, 1, 6 và p = 2 seed 1, 9. QAOA đẩy xác suất cover-feasible lên khoảng 0,67, so với 0,27 của bitstring đều, nhưng xác suất rơi vào optimum chỉ gấp khoảng hai lần bitstring đều.

## 7. Lượt phụ: toàn bộ fit, rho = 0,10, q = 21

Lượt phụ dùng toàn bộ 191 cặp I_f ở rho = 0,10, bỏ qua luật chọn instance theo amendment A3. `q_max` được nâng từ 20 lên 21. `gates.json` ghi `SECONDARY`; nếu đây là lượt chính thì trạng thái sẽ là `BLOCKED`, vì G2 không đạt.

| Đại lượng | Giá trị |
|---|---|
| Biến y / slack / tổng q | 11 / 10 / 21 |
| `k_ub`, alpha | 4, 5 |
| Số coupling, degree lớn nhất, dynamic range | 43, 9, 5 |
| Statevector | 33554432 byte (32 MiB) |
| Optimum (exhaustive, CP-SAT có và không reduction) | 19, 19, 19 |
| Số minimum cover trên instance đã reduce | 3 |
| P(minimum cover) của bitstring đều | 3 × 2^10 / 2^21 ≈ 0,001 |
| q khi tắt gộp cột trùng | 34 |

Ground-state check đạt ở cả năm giá trị alpha (3 ground state, đều là minimum cover). Exhaustive, CP-SAT, greedy, repair-only và SA đều cho 19. Raw cover-feasible probability của random_qubo là 0,221, của random_y là 0,392. Bản repaired của random_qubo, xuất phát từ sample năng lượng thấp nhất, cho 20 cấu hình (gap 1) ở 3/10 seed (seed 0, 2, 6). Bản raw của random_qubo vẫn có 19 ở mọi seed.

Cả 20 run QAOA dừng ở cap 60 s với trạng thái `timeout_before_confirmation`, nên không có shot xác nhận và không có raw probability nào cho QAOA:

| Depth | Cổng 2 qubit / depth circuit | Call đã chạy trên 401 yêu cầu (min–max) | Thời gian mỗi run |
|---|---|---|---|
| p = 1 | 43 / 18 | 334–392 | 60,0–60,4 s |
| p = 2 | 86 / 30 | 190–252 | 60,1–60,6 s |

Mọi run vẫn gặp một sample cover-feasible có 19 cấu hình trong trace optimizer, nên `catalogue-w2.json` vẫn được xuất (QAOA p = 1, seed 2) và trùng W1. Catalogue này lấy từ trace optimizer, tức là từ các circuit có góc thay đổi theo từng call, không từ circuit xác nhận cố định. Q11 trong kế hoạch chỉ coi sample gộp từ optimizer là best-found, không phải xác suất raw, nên W2 của lượt phụ không được dùng làm bằng chứng cho G2. Nhóm cần quyết định tại review có giữ file này với nhãn `quantum` hay không. Peak RSS của tiến trình không vượt 284 MiB; tổng thời gian lượt phụ là 1294 s.

Ở 21 qubit, budget của protocol không chạy hết được trong cap 60 s trên máy này. Lượt phụ không nới cap.

## 8. Trạng thái gate

| Gate | Lượt chính | Bằng chứng |
|---|---|---|
| G0 | Đạt, có nhãn pilot | `completeness.json` complete, hash khớp; policy chưa duyệt và split dựng lại (A1) |
| G1 | Đạt | q = 13 ≤ 20; ground state là minimum cover ở mọi alpha không stress-only |
| G2 | Đạt | Counts xác nhận ở p = 1 và p = 2, 10 seed mỗi depth, đủ control |
| G3 | Không đạt theo luật dừng của Q13 | Classical, SA và repair-only đều tìm ra optimum; QAOA không cho catalogue tốt hơn; Nhóm chưa khóa criterion G3 |
| G4 | Chưa xét | Chưa có consumer, chưa có đo e2e |

`gates.json` của lượt chính ghi `PASS_OFFLINE`; trạng thái này chỉ xét G0–G2. Lượt phụ ghi `SECONDARY` với `outcome_if_primary = BLOCKED`: G1 đạt ở q = 21, G2 không đạt vì không có counts xác nhận ở depth nào.

## 9. Giới hạn

- Policy `qor-pilot-v1` và protocol chưa được duyệt; mọi kết quả mang nhãn pilot.
- Split dựng lại có hash khác receipt (A1).
- Không có sweep validation, nên không có coverage hay regret ngoài tập fit (A2). Catalogue được đánh giá trên chính dữ liệu dùng để fit.
- Instance chính là restricted cohort 180/200 cặp, chọn theo luật đã khóa. Optimum 17 là optimum của cohort, không phải của toàn bộ fit.
- Instance nhỏ: 7 biến y sau reduction và chỉ 2 minimum cover. Một instance không đủ để nói gì về scaling.
- Chưa có consumer catalogue (Q14+), nên chưa có số runtime nào.
- Thời gian đo trên một máy Windows 8 thread, một tiến trình, không cô lập tải nền. Cap 60 s vì vậy phụ thuộc máy: ở q = 21, một máy nhanh hơn có thể chạy hết budget.

Các claim chưa được viết, theo bảng claim ở mục 10 của proposal: quantum advantage hay speedup; QAOA tốt hơn classical; catalogue giữ QoR trên dữ liệu mới; simulator dự báo hiệu năng QPU; cải thiện runtime e2e. Claim được phép viết: encoding đúng cover tại rho = 0,10 (test exhaustive và ground-state check), và QAOA simulator tạo ra một catalogue cụ thể có hash và sample truy vết được, nhưng không tốt hơn catalogue classical.

## 10. Tái lập

Trong venv quantum, từ gốc repo:

```text
python -m tools.quantum.characterize --split data/sweep/v4-grid/split-rebuilt.json --configs data/sweep/v4-grid/configs.json --policy tools/qor/policy-pilot.json --partition calibration data/sweep/v4-grid/audit.json data/sweep/v4-grid/runs.csv data/sweep/v4-grid/qor.csv --pair-prefix udis_train_ --output data/quantum/v4-char
python -m tools.quantum.pilot --characterization data/quantum/v4-char/characterization.csv --configs data/sweep/v4-grid/configs.json --protocol tools/quantum/protocol-pilot.json --completeness data/quantum/v4-char/completeness.json --label real --output data/quantum/pilot-v4-primary
python -m tools.quantum.pilot --characterization data/quantum/v4-char/characterization.csv --configs data/sweep/v4-grid/configs.json --protocol tools/quantum/protocol-pilot.json --completeness data/quantum/v4-char/completeness.json --label real --secondary-full-rho 0.10 --output data/quantum/pilot-v4-full-q21
python -m tools.quantum.report --run data/quantum/pilot-v4-primary --run data/quantum/pilot-v4-full-q21 --output data/quantum/report-tables.md
```

Seed của mọi solver nằm trong `run-config.json` và `solver_runs.jsonl`: solver seed 0–9, CP-SAT seed 0, seed chọn cohort 20261104. Thời gian chạy sẽ khác theo máy. Chưa chạy lại lần hai để kiểm tra counts có trùng từng shot hay không. Ở lượt phụ, số call chạy được trước cap phụ thuộc tốc độ máy, nên counts và trace sẽ khác trên máy khác.
