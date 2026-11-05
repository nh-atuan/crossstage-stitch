# Kế hoạch triển khai quantum catalogue selection

## Material Passport

- Origin Skill: academic-research-suite / deep-research.
- Origin Mode: backlog triển khai từ nghiên cứu có RQ xác định.
- Origin Date: 02/11 (Thứ 2), năm 2026; thay phân công đề xuất ngày 24/10 (Thứ 7).
- Verification Status: PILOT_RUN, cập nhật 05/11 (Thứ 5). Tooling `tools/quantum/` đã có trên nhánh `thanh/p3-quantum`, pilot đã chạy trên sweep calibration thật và đạt PASS_OFFLINE cho G0–G2 với nhãn pilot; G3 không đạt. Trạng thái từng ticket ở mục 11, kết quả ở [quantum-pilot-report.md](quantum-pilot-report.md).
- Version Label: quantum_catalogue_tasks_v2_thanh.
- Tài liệu phương pháp: [quantum-integration-proposal.md](quantum-integration-proposal.md).

## 1. Phạm vi, owner và Definition of Done

Primary path đã chọn: **A — offline QUBO/QAOA catalogue selection**, B_fixed. Tạo bảng đo CUDA trước, xây native cover/QUBO, chạy QAOA simulator p=1,2 có raw samples, rồi so với classical baselines và tích hợp catalogue vào controller cổ điển. Không chuyển sang S5 estimator, S6 seam hoặc quantum classifier để thay thế phần còn thiếu.

Phân công được người dùng chốt ngày 02/11 (Thứ 2): **Thành là single owner Q00–Q13 và quantum tooling/solver report**. Thành tiếp tục S2–S3/A3; các việc quantum được làm song song với baseline không quantum. Tuấn cung cấp raw sweep và giữ harness/S1/common/merge; Đế duyệt policy/reference và giữ controller/S4–S5; Chương giữ S6/A4/A5, review QoR và tổng hợp paper. Review hoặc cung cấp input không chuyển trách nhiệm implementation của stage sang Thành. Quyết định ở [p3-offline-quantum-catalogue.md](../decisions/p3-offline-quantum-catalogue.md).

Phase hiện tại là **Phase 3**, deadline chung 05/11 (Thứ 5), review cùng ngày. Q00–Q13 là backlog chi tiết của phase này; chưa có task nào được đánh dấu DONE bằng lượt cập nhật kế hoạch. Effort dưới đây là ước lượng person-day tập trung khoảng 6 giờ, không phải deadline hoặc cam kết tất cả đều kịp; không cộng đôi việc harness/stage đã có. Thành báo tiến độ theo gate và các thiếu hụt dữ liệu/tài nguyên; baseline/A1–A5 tiếp tục độc lập. Q14–Q22 và QH chỉ là định hướng sau review, không giao lịch hoặc acceptance chi tiết của phase sau. P0 là ưu tiên offline; P1/P2 trong backlog cũ chỉ biểu thị ưu tiên, không phải tên phase.

Definition of Done cho tích hợp quantum ứng dụng: Q00–Q13 có artifact kiểm tra được, QAOA thực sự chạy trên ít nhất một instance từ CUDA sweep còn q_after>0 sau reduction, native/classical baselines đủ, raw và repaired outputs tách biệt. Optimum=1 hoặc classical giải nhanh không miễn prototype khi q_after>0 và đủ tài nguyên. Kết quả QAOA không có feasible raw sample vẫn là negative experiment nếu circuit/counts/logs hợp lệ; không được xuất catalogue thành công giả. Khi đó muốn hoàn tất tích hợp vào controller phải có catalogue hybrid hợp lệ, truy vết nguồn và repair, hoặc ghi rõ phần tích hợp chưa hoàn thành. Claim hệ thống còn cần Q14–Q20; Q21–Q22 để bàn giao paper/tái lập. QPU không bắt buộc.

CLI và file code `tools/quantum/*` của Q00–Q13 đã có từ 05/11 (Thứ 5); trạng thái và bằng chứng từng ticket ở mục 11. Các CLI và file nêu cho Q14 trở đi vẫn là dự kiến, chưa tồn tại. Artifacts data/results/build không commit theo conventions; lưu manifest/hash và vị trí chia sẻ. Small test fixtures có thể commit trong tests theo quy ước Nhóm; không commit dataset hoặc raw benchmark outputs.


### 1.1. Hai nhánh song song và ba nhóm đối chứng

| Nhóm | Vai trò | Quantum khi chạy từng ảnh |
|---|---|---|
| W0 — Adaptive-original | Runtime/ladder gốc của nhóm, cùng Fixed-Best và Oracle; giữ campaign baseline riêng | Không |
| W1 — Adaptive-classical-catalogue | Catalogue do exact/CP-SAT/greedy hoặc classical solver đã khai báo chọn; dùng catalogue consumer cổ điển | Không |
| W2 — Adaptive-QAOA-catalogue | Catalogue từ QAOA simulator, validated và truy vết được sample; repair nếu có phải mang nhãn hybrid | Không |

W0 chạy được khi không cài quantum SDK và không cần đợi W1/W2. Quantum SDK không nằm trong root CMake, stage API hoặc `tools/requirements.txt`. W1/W2 chỉ được gọi là kết quả runtime sau khi Đế/Tuấn tích hợp consumer và đo thật; trước đó chỉ có so sánh solver offline. W1/W2 dùng cùng candidate set/reduction, split, quality policy, B, consumer, selector family và tuning budget, reuse/self-check/fallback. W0 so với W1 đo tác động thiết kế catalogue/consumer; W1 so với W2 mới kiểm tra đóng góp solver. Kích thước catalogue khác nhau vẫn được giữ trong bảng, kèm matched-size control khi có.

Thành triển khai adapter, classical solvers, encoding, QAOA và validator dưới `tools/quantum/` cùng tests riêng; các đường dẫn đó hiện chưa tồn tại. Các stage S1–S6 không nhận quantum states hoặc circuit. Catalogue là JSON gồm config IDs/S/K/R/B, hashes và provenance, không chứa RGB. Một catalogue không tự xác định thứ tự ladder; Đế chuyển nó sang policy/ladder hợp lệ trên development và giữ fallback đã duyệt.

### 1.2. Mức nghiệm thu trong Phase 3

- **PASS_OFFLINE:** Q00–Q13 đủ artifact trên ít nhất một instance từ sweep CUDA còn `q_after > 0`, có QAOA p=1,2 và controls. PASS không có nghĩa QAOA thắng classical hoặc runtime nhanh hơn.
- **NEGATIVE_RESULT:** Circuit/counts và đối chứng đầy đủ nhưng raw feasibility bằng 0, classical mạnh hơn hoặc repair quyết định kết quả; bàn giao nguyên dữ liệu và giới hạn, không xuất catalogue thành công giả.
- **BLOCKED/INCOMPLETE:** Thiếu sweep/policy approval, environment không chạy, encoding chưa đúng, mọi instance bị exact reduction giải hết hoặc vượt tài nguyên; toy-only không đóng ứng dụng quantum.

Không dùng một kết quả baseline để đóng task quantum. Negative result không chặn baseline, không tự cho phép bỏ workstream quantum, đổi stage hoặc mua quyền QPU. Nếu chưa có e2e consumer thì phần tích hợp runtime còn pending dù solver offline đã nghiệm thu.

### 1.3. Protocol và phân bổ công việc của Thành

Thành bắt đầu Q02/Q10 và tiny exact tests trong lúc chờ raw sweep Q00. Sau G0 mới tạo instance ứng dụng; sau G1 mới chạy QAOA; Q13 gom tất cả seeds/failures và công bố G2/G3. QAOA/classical controls có ngân sách solver riêng, CPU simulator tách khỏi phiên đo GPU T4 để tránh tranh tài nguyên. Không chạy thêm solver trên T4 trong lúc baseline đang đo nếu chưa có cô lập tài nguyên.

A3 và quantum là hai deliverable riêng của Thành; không chuyển A3 sang owner khác. Để giới hạn khối lượng trong Phase 3, bắt đầu một instance thật đã kiểm tra cùng các controls, rồi mới mở rộng cohort/seeds khi còn budget. Không đồng thời mở multiobjective, quantum selector, seam, QPU hoặc toàn bộ sensitivity grid. Mọi run config nhỏ hơn các budget pilot đề xuất phải được ghi trước khi chạy và báo số thực thi; không giảm sau khi nhìn kết quả để bỏ seed thất bại. Nếu tổng effort vượt sức chứa phase, Thành báo phần PASS/NEGATIVE/BLOCKED tại review; Nhóm quyết định phần tiếp theo, không ghi tất cả DONE.

Nhánh quantum đầu tiên chỉ dùng **một** policy QoR đã được Đế duyệt và B cố định được Chương xác nhận. A5 là workstream sensitivity riêng; không tự nhân quantum theo 20 tổ hợp hoặc dùng test UDIS-D để fit catalogue. Rho, seeds, sampling/budget và restriction manifest ở proposal là tham số pilot đề xuất; Thành lập run config, Tuấn/Đế review trước khi dùng validation/final. Quyết định offline/owner hiện tại không đồng thời duyệt mọi ngưỡng trong proposal.

## 2. Dependency graph và critical path

Q00 được đặt đầu vì chưa có bảng characterization đủ provenance được tooling quantum nghiệm thu; kernel/controller nền đã có và Phase 2 đã PASS theo báo cáo nhóm. Q01/Q02 là prerequisite chuẩn bị của Q00, có thể làm trước khi đủ kernel. Q04–Q12 có thể phát triển bằng toy fixtures khi chờ, nhưng không đóng task nghiệm thu dữ liệu thật bằng toy. Graph có cả chuỗi tooling song song và đường dữ liệu chính.

```mermaid
flowchart TD
    Q01["Q01: protocol, split, policy"] --> Q00["Q00: CUDA development sweep"]
    Q02["Q02: schema/adapter"] --> Q00
    Q00 --> Q03["Q03: join, failures, characterization"]
    Q03 --> Q04["Q04: Oracle / native cover"]
    Q04 --> Q05["Q05: greedy / exhaustive / CP-SAT"]
    Q04 --> Q07["Q07: QUBO / Ising"]
    Q07 --> Q06["Q06: SA / random"]
    Q07 --> Q08["Q08: exact reductions"]
    Q05 --> Q09["Q09: penalty / resource admission"]
    Q08 --> Q09
    Q05 --> Q12["Q12: validator / shared repair"]
    Q06 --> Q12
    Q09 --> Q11["Q11: QAOA p=1,2"]
    Q10["Q10: isolated environment"] --> Q11
    Q10 --> Q05
    Q12 --> Q11
    Q11 --> Q13["Q13: real-instance solver pilot"]
    Q13 --> Q14["Q14: catalogue loader"]
    Q14 --> Q15["Q15: classical selector / escalation"]
    Q15 --> Q16["Q16: reuse / invalidation"]
    Q16 --> Q17["Q17: e2e instrumentation"]
    Q17 --> Q18["Q18: validation / ablations"]
    Q18 --> Q19["Q19: protocol lock"]
    Q19 --> Q20["Q20: final evaluation"]
    Q20 -->|"system claims"| Q21["Q21: paper / claims"]
    Q13 -. "offline claims" .-> Q21
    Q21 --> Q22["Q22: reproducibility release"]
    Q13 -. "optional" .-> QH["QH: hardware replication"]
```

Critical path: CUDA đủ stage/harness → Q00 → Q03 → Q04 → encoding/admission → Q11/Q13 → controller replay → final. Q10, tiny-instance tests và validator có thể làm song song với Q00 và nhánh baseline không quantum. Nếu critical path bị chặn, báo missing artifact; không thay bằng timing S6-only hay mock CPU.

## 3. Nhóm dữ liệu characterization

### Q00 — Complete the development CUDA sweep

- **Owner:** Thành (yêu cầu, tiếp nhận và kiểm tra campaign); Tuấn cung cấp sweep/harness, Đế review pipeline/QoR; **Priority:** P0.
- **Dependencies:** Q01, Q02; stage GPU và T4/harness đủ điều kiện theo plan. **Estimated effort:** 1–2 ngày chạy/kiểm tra sau tích hợp; không bao gồm viết stage còn thiếu.
- **Deliverable:** Thành yêu cầu và tiếp nhận sweep CUDA từ Tuấn cho các development-fit/validation pairs được phép, C=(S,K,R,B_fixed), Fixed-High và baseline cần thiết; có 2 warmup + 10 đo/pair/config.
- **Acceptance criteria:** cùng timing boundary t_e2e của conventions; algorithm seed cố định, H tất định; đủ stage thực chạy GPU; H trong hệ RGB gốc B→A; lưu mọi launch/runtime failure; config/repeats coverage được kiểm bằng manifest. Không thay measurements thiếu bằng CPU/mock hoặc latency S6-only.
- **Go/No-Go:** G0; thiếu stage/harness/T4 thì BLOCKED ứng dụng, không đánh sweep complete. Toy test của tooling vẫn có thể chạy độc lập.
- **Artifacts:** `runs.csv`, `meta.json`, config-grid/split manifests, logs, GPU/driver/CUDA/clock/git/source hashes; vị trí dự kiến `results/quantum-characterization/<run_id>/`.

### Q01 — Freeze the pilot protocol and scene-grouped splits

- **Owner:** Thành lập protocol quantum; Đế duyệt policy/reference/split, Tuấn review CSV/measurement, Chương review B; **Priority:** P0.
- **Dependencies:** quyết định hiện có trong docs; không phụ thuộc Q00. **Estimated effort:** 0,5 ngày review, chưa tính chuẩn bị dataset lớn.
- **Deliverable:** protocol pilot có C, B_fixed, Fixed-High, candidate/fallback spaces, split/grouping, seeds và policy version/hash.
- **Acceptance criteria:** validation tách từ development được phép; final không đổi; scene/sequence/ảnh dùng chung không leakage khi metadata cho phép. Policy pilot 0.95/0.02/0.5 được ghi là đề xuất cho tới Đế duyệt. Khóa quantile method và failure/repeat handling. Sensitivity grids/rho/budget có người duyệt, không tự thay Go/No-Go runtime.
- **Go/No-Go:** chưa có owner decision thì chỉ cho synthetic tooling/pilot có nhãn, chưa chạy final hoặc claim QoR chính thức.
- **Artifacts:** `protocol-pilot.json`, `splits.json`, `config-grid.json`, `policy.json`, decision record; metadata thiếu scene được ghi rõ.

### Q02 — Implement the characterization schema and adapters

- **Owner:** Thành (adapter riêng), Tuấn review schema/common boundary; **Priority:** P0.
- **Dependencies:** đọc conventions và evaluator hiện có; test schema không cần sweep. **Estimated effort:** 0,5–1 ngày.
- **Deliverable:** `tools/quantum/characterize.py`, schema bảng dẫn xuất và tiny CSV fixtures/tests.
- **Acceptance criteria:** join đúng pair_id/config_id/run_id/split, gọi evaluator/report để tính pass; không sửa runs.csv/qor.csv chung. Kiểm tra duplicate/missing rows, config mismatch, warmup, đủ 10 repeats, status/hashes. `latency_p95_repeat_ms` chỉ diagnostic; p95 main trên pair medians. Không drop silent failures.
- **Go/No-Go:** input chưa đầy đủ phải error/ledger, không tự tạo latency hoặc labels.
- **Artifacts:** schema version, tests cho malformed/missing/policy mismatch, `characterization.csv` dự kiến; implementation không có quantum dependency.

### Q03 — Validate failures and publish characterization

- **Owner:** Thành; Đế review reference status/QoR, Tuấn review timing; **Priority:** P0.
- **Dependencies:** Q00, Q02. **Estimated effort:** 0,5 ngày.
- **Deliverable:** bảng pair/config median latency, diagnostic repeat p95, QoR pass, coverage, status và completeness report.
- **Acceptance criteria:** reference_failure/reference_unreliable, unachievable_in_C và incomplete phân loại riêng; giữ denominator toàn split. QoR pass tái tạo từ evaluator, không nhập tay. Fixed-High hợp lệ tự pass nếu nằm trong C; explicit candidate exclusion được ghi. Mỗi dòng truy về run/meta/policy hashes.
- **Go/No-Go:** G0 chỉ qua khi đủ bảng và provenance; incomplete chặn instance chính, không coi là no-pass. S6/Oxford smoke hiện có không đủ đóng task.
- **Artifacts:** `characterization.csv`, `failure-ledger.csv`, `completeness.json`, input manifest/hash và review record.

## 4. Nhóm native cover và classical baselines

### Q04 — Build the Oracle and original cover instances

- **Owner:** Thành; Tuấn review Oracle consistency; **Priority:** P0.
- **Dependencies:** Q03 cho dữ liệu thật; phát triển trước bằng toy. **Estimated effort:** 0,5–1 ngày.
- **Deliverable:** `tools/quantum/build_instance.py`, C/I_f/t_i*/U_i(rho), original incidence matrix và original-problem validator API.
- **Split guard:** chỉ development-fit tạo incidence/objective cho solver; validation/final được xuất riêng cho đánh giá catalogue, không thêm vào cover constraints hoặc repair. Tests phải từ chối mixed-split instance và rò rỉ final vào fit.
- **Acceptance criteria:** Oracle từ median t_e2e và pass trong C, không từ S6 latency; rho/epsilon comparison versioned; no-pass/empty I_f là status rõ. Fixture giải tích xác nhận U_i, full-grid coverage và infinity regret. Có hashes/scope full hoặc restricted.
- **Go/No-Go:** thiếu measurements hoặc I_f rỗng thì không xuất instance hợp lệ giả.
- **Artifacts:** `instance.json`, `incidence.npz`, `oracle.csv`, tests và configuration mapping.

### Q05 — Implement greedy, exhaustive and native CP-SAT

- **Owner:** Thành; Tuấn review exact/bound reporting; **Priority:** P0.
- **Dependencies:** Q04; CP-SAT execution cần Q10, pure-Python tests làm trước. **Estimated effort:** 1 ngày.
- **Deliverable:** `tools/quantum/solve_classical.py` với maximum-gain greedy/reverse-delete, exhaustive-small và native integer CP-SAT.
- **Acceptance criteria:** cùng original instance/reduction và tie-break; greedy/exact feasibility được kiểm gốc; known-optimum tests. Native enumeration admission theo n_y=|C_remaining|, không theo số hàng. CP-SAT OPTIMAL/FEASIBLE/UNKNOWN và incumbent/bound tách rõ; exhaustive timeout chưa là certificate. Ghi actual wall time kể cả khi exact chạy gần như tức thì.
- **Go/No-Go:** optimum/bound sai hoặc thiếu certificate thì chặn objective-gap claims; classical mạnh không phải lý do bỏ baseline hoặc bỏ prototype quantum.
- **Artifacts:** native model export, solver logs, known-optimum fixtures, per-instance catalogue/coverage/regret report.

### Q06 — Implement QUBO SA and matched random controls

- **Owner:** Thành; **Priority:** P0.
- **Dependencies:** Q07; phát triển trên tiny QUBO trước. **Estimated effort:** 0,5–1 ngày.
- **Deliverable:** seeded Metropolis SA, uniform-QUBO random và random-y distribution có khai báo; same-time/matched-sample runs.
- **Acceptance criteria:** tính delta energy khớp full energy; schedule/proposals/restarts, draws/evaluations và threads lưu vào log. Cùng QUBO với QAOA, không coi shot bằng SA move. Ghi raw sample/energy/status; raw feasibility không thay bằng repaired feasibility.
- **Go/No-Go:** không tái lập seed hoặc không có comparable budgets thì chưa nghiệm thu solver comparison.
- **Artifacts:** solver configs/logs, raw sample files, tests, anytime traces.

## 5. Nhóm encoding và tài nguyên

### Q07 — Implement slack QUBO, Ising conversion and decode

- **Owner:** Thành; Tuấn review logic/formulation; **Priority:** P0.
- **Dependencies:** Q04. **Estimated effort:** 1 ngày.
- **Deliverable:** slack binary b_i=ceil(log2 d_i), upper-triangular QUBO, Ising h/J/offset, bit mapping/decode.
- **Acceptance criteria:** enumerate tiny (y,z), so residual/QUBO/Ising energies; cover-feasible y có ít nhất một zero-penalty slack, equality-feasible sample luôn cover gốc. Với alpha>lambda|C|, ground state cardinality bằng exact cover; test singleton, duplicate, non-power-of-two degree, empty/invalid row, endian/factor-2. QUBO-feasible phải định nghĩa là residual zero, không là mọi bitstring.
- **Go/No-Go:** bất kỳ mismatch nào chặn QAOA; không sửa tolerance/penalty để che lỗi conversion.
- **Artifacts:** `qubo.json`, `ising.json`, `variables.json`, proof note trong proposal và `test_quantum_encoding.py`.

### Q08 — Implement optimum-preserving reductions

- **Owner:** Thành; **Priority:** P0.
- **Dependencies:** Q07, Q04. **Estimated effort:** 0,5–1 ngày.
- **Deliverable:** duplicate/superset rows, forced singletons, zero columns, duplicate columns và full reconstruction ledger.
- **Acceptance criteria:** exhaustive nhỏ xác nhận optimum cardinality trước/sau và feasibility sau restore trên original matrix; fixed-point và idempotence tests. Giữ forced cost/representatives/hashes. Duplicate columns chỉ giữ cardinality optimum, không hứa validation-equivalence; có on/off ablation. Không sparsify mất constraints rồi nhận exact reduction.
- **Go/No-Go:** mismatch thì tắt reduction đó và sửa; q=0 xuất solved_by_reduction, không gọi quantum solution.
- **Artifacts:** `reduction.json`, before/after matrices, reconstruction tests và optimum certificates nhỏ.

### Q09 — Audit penalties, couplings and simulator admission

- **Owner:** Thành; **Priority:** P0.
- **Dependencies:** Q05, Q07, Q08. **Estimated effort:** 0,5 ngày.
- **Deliverable:** bound alpha>lambda k_ub từ feasible cover, sensitivity grid, q_before/q_after/couplings/dynamic-range/RAM report.
- **Acceptance criteria:** default alpha=k_ub+1 khi lambda=1; below-bound tests được nhãn stress-only. Đếm q_after theo row degrees đã cập nhật sau reduction. Positive global scaling giữ mapping/offset, không clip. q<=20 là proposed admission, đo thêm actual memory. Full instance quá lớn chỉ cho restricted cohort manifest khóa trước; đánh giá full-data coverage sau đó, không claim full optimum.
- **Go/No-Go:** G1; sai bound/precision hoặc vượt RAM/time thì reject run với status rõ. Mọi instance trivial/quá lớn phải báo feasibility limitation, không lựa dataset sau khi xem quantum score.
- **Artifacts:** `resource-audit.json`, penalty-sensitivity config, cohort selection manifest và admission logs.

## 6. Nhóm QAOA prototype bắt buộc

### Q10 — Pin and verify an isolated simulator environment

- **Owner:** Thành; Tuấn review dependency boundary; **Priority:** P0.
- **Dependencies:** không cần sweep; package/install plan trong proposal. **Estimated effort:** 0,5 ngày.
- **Deliverable:** Python 3.12 environment riêng, pinned/hashes Qiskit/Aer/OR-Tools, license inventory và install/run instructions.
- **Tách dependencies:** classical core dùng stdlib/NumPy, CP-SAT dùng extra OR-Tools; Qiskit/Aer chỉ import khi chọn backend QAOA. CLI/tests classical phải chạy khi thiếu quantum SDK; không import SDK trong package initializer hoặc runtime consumer.
- **Acceptance criteria:** compatibility import + tiny circuit smoke, CPU/RAM/threads/OS/version receipt, direct/transitive licenses/NOTICE. Không đổi pipeline requirements hoặc root build, không cần cloud credentials. Pin versions sau kiểm tra thực, không sao chép version chưa chạy.
- **Go/No-Go:** thiếu compatible environment thì chặn QAOA execution; dependency failure không được ghi thành quantum negative result về thuật toán.
- **Artifacts:** `tools/quantum/requirements-lock.txt`, environment receipt và license inventory; venv/build không commit.

### Q11 — Implement and run QAOA p=1,2 on a simulator

- **Owner:** Thành; **Priority:** P0.
- **Dependencies:** Q07, Q09, Q10, Q12. **Estimated effort:** 1–2 ngày prototype; pilot budget giới hạn theo config.
- **Deliverable:** `tools/quantum/solve_qaoa.py`, uniform initial state, X mixer, cost Hamiltonian, finite-shot expectation/SPSA và p=1,2.
- **Acceptance criteria:** circuit phase/basis test khớp Ising oracle; raw counts tổng đúng executed shots. Lưu angles, optimizer/sampler/transpiler seeds, requested/executed calls/shots, restarts/timeouts, backend/version/circuit depth/two-qubit gates. Pilot đề xuất 10 seeds, 4 starts, 100 calls/start, 1024 shots/call + 4096 confirmation; 60 s cap gồm mọi bước solver/repair, dừng sớm ghi rõ. Raw probabilities chính từ confirmation circuit cố định; samples gộp optimizer chỉ là adaptive mixture/best-found. Exact-expectation diagnostic tách khỏi finite shots.
- **Go/No-Go:** G2 cần ít nhất một nontrivial real-sweep instance, có counts ở cả p. Feasibility=0 vẫn giữ thí nghiệm; no circuit/crash không hoàn thành task. Không nhãn simulator là hardware.
- **Artifacts:** circuits QPY/QASM phù hợp version, raw counts, angles/traces, `solver_runs.jsonl`, simulator memory/time logs.

### Q12 — Implement shared validation, repair and catalogue provenance

- **Owner:** Thành; Đế review consumption contract; **Priority:** P0.
- **Dependencies:** Q05, Q06, Q07; có thể test không cần QAOA. **Estimated effort:** 0,5–1 ngày.
- **Deliverable:** shared original-matrix validator, canonical slack decode, deterministic add/reverse-delete repair và catalogue schema.
- **Acceptance criteria:** raw equality/cover feasibility tách; repaired y luôn được kiểm lại gốc. Cùng repair cho quantum/SA/random và empty-start control; đếm bit flips/added/deleted/time/rate. Không dùng validation/final labels. No-feasible trả status thật; catalogue success truy về sample/forced/repair, không mặc định greedy rồi gắn nhãn quantum.
- **Go/No-Go:** provenance đứt hoặc repair chiếm kết quả mà không báo thì fail nghiệm thu; hybrid repaired catalogue cần nhãn riêng.
- **Artifacts:** `catalogue.schema.json`, `catalogue.json`, raw/repaired audit, repair-only tests và logs.

### Q13 — Run the real-instance solver pilot and evaluate Go/No-Go

- **Owner:** Thành; Tuấn/Đế review kết luận; **Priority:** P0.
- **Dependencies:** Q03–Q12 có artifact áp dụng, CP-SAT environment sẵn. **Estimated effort:** 0,5–1 ngày, không gồm sweep.
- **Deliverable:** comparison exhaustive/CP-SAT/greedy/SA/random/QAOA trên cùng instance; raw/post-repair và resource/time breakdown.
- **Acceptance criteria:** cardinality/feasibility/success/optimality gap hoặc bounds, raw feasibility probability, validation coverage/regret, repair rate và actual time đủ. CP-SAT native được phép mạnh hơn QUBO baseline. Cùng wall cap và separate resource audit; seeds/restarts retained, không chọn best seed rồi bỏ failures. Catalogue lớn/nhỏ đều giữ, thêm matched-size nếu có.
- **Go/No-Go:** G3; nếu exact/greedy thắng hoặc repair-only tương đương, chốt feasibility/negative result và dừng mở rộng. Quantum execution evidence vẫn bắt buộc; chưa có real instance thì ghi chưa hoàn tất.
- **Artifacts:** `solver-comparison.csv`, pilot report, logs/hashes, catalogue-size/regret và solver-time/instance-size plots; không vẽ scaling claim từ một instance.

## 7. Handoff sau prototype offline — chỉ định hướng

Q14–Q17 giữ làm tham chiếu backlog cũ: Thành xuất catalogue có schema/hash và validator; Đế chủ trì consumer/ladder và giữ reuse/invalidation đúng contract; Tuấn giữ harness/telemetry và deployment không quantum SDK. Consumer validation không được thay nguồn gốc catalogue hoặc ẩn classical repair. Chưa coi các API/CLI catalogue dự kiến là đã tồn tại. Nếu các owner có đủ thời gian trong Phase 3, chỉ tích hợp sau khi catalogue qua validator và cùng contract; nếu chưa thì báo phần hệ thống pending và lập task chi tiết tại review.

## 8. Validation, final và paper — chỉ định hướng

Q18–Q22: sau review lập chi tiết matched W0/W1/W2, validation, protocol lock và final evaluation. Thành giữ analysis/solver artifacts và draft nội dung quantum; Chương tổng hợp paper; Đế review controller/QoR; Tuấn review systems/merge. Phải có output runtime thật mới viết e2e claims. QH (QPU/annealer) là tùy chọn, cần quyền/chi phí được duyệt riêng; simulator đủ cho quantum offline MVP. Không đặt deadline riêng hoặc chuyển các phase sau thành kế hoạch chi tiết trong tài liệu này.

## 9. Gate và quyết định dừng

| Gate | Tickets / điều kiện | Nếu không đạt |
|---|---|---|
| G0 | Q00–Q03: CUDA characterization đầy đủ, policy/split có owner/hash | Chặn real-instance work; không giả dữ liệu; tooling test tiếp được |
| G1 | Q04–Q10/Q12: native/QUBO/Ising/reduction đúng, RAM/qubit phù hợp | Sửa lỗi; trivial/resource limitation báo rõ, không tạo circuit giả |
| G2 | Q11/Q13: p=1,2 real-instance circuit/counts và classical controls | Zero-feasible là negative experiment; crash/no circuit là chưa hoàn thành |
| G3 | Q13/Q18: trade-off theo criteria Nhóm khóa trước | Dừng mở rộng QPU/depth/B–F; giữ feasibility/negative report |
| G4 | Q17–Q22: controller thật cho system claims, final locked và reproducible | Giới hạn claim offline, hoặc ghi task chưa hoàn thành nếu thiếu minimum evidence |

**Go cho simulator MVP A; No-Go cho mở rộng khi không có giá trị.** Exact/greedy thắng không làm biến mất yêu cầu quantum: vẫn bàn giao circuit/logs, bảng đối chứng và negative/feasibility result. Nếu mọi real instance sau exact reduction là trivial hoặc vượt tài nguyên thì báo chưa thể hoàn thành real-application prototype trong budget, không claim toy thay ứng dụng. Không nới QoR/split/Go–No-Go sau xem kết quả. Quantum là mandatory experiment của backlog này; thắng classical là giả thuyết có thể bị bác bỏ.

## 10. Các commit dự kiến theo deliverable

| Nhóm commit tiếng Anh | Tickets liên quan |
|---|---|
| `feat(quantum): add characterization and native cover instances` | Q02–Q04 |
| `feat(quantum): add classical catalogue baselines` | Q05–Q06 |
| `feat(quantum): encode and validate catalogue QUBOs` | Q07–Q09/Q12 |
| `feat(quantum): add reproducible QAOA simulator experiments` | Q10–Q13 |
| `feat(controller): consume versioned configuration catalogues` | Q14–Q17, do Đế/Tuấn phụ trách |
| `docs(quantum): report catalogue experiments and limitations` | Q18–Q22 |

Danh sách này là cách tách commit implementation cho Thành và các owner integration khi task được kích hoạt; lượt hiện tại chỉ cập nhật tài liệu, không tạo solver hoặc catalogue consumer. Nguồn học thuật, license và giới hạn đọc nằm tại proposal §12; không sao chép citation metadata chưa kiểm chứng vào paper.

## 11. Trạng thái Q00–Q13 tại review 05/11 (Thứ 5)

Nhãn trạng thái: PASS (có artifact và test, chạy trên dữ liệu thật), PARTIAL (thiếu một phần dữ liệu hoặc kiểm chứng), PENDING_REVIEW (đã làm, chờ người duyệt). Mọi kết quả trên dữ liệu thật vẫn mang nhãn pilot vì policy và protocol chưa được duyệt. Output nằm trong `data/quantum/` (không commit), mỗi run có `SHA256SUMS.json`.

| Ticket | Trạng thái | Bằng chứng | Còn thiếu |
|---|---|---|---|
| Q00 | PARTIAL | Sweep calibration của Đế: 200 cặp × 48 cấu hình × 10 repeat trên T4, hash khớp audit | Sweep validation 50 cặp; `split.json` gốc |
| Q01 | PENDING_REVIEW | `tools/quantum/protocol-pilot.json` với amendment A1–A4 ngày 05/11 (Thứ 5) | Đế duyệt policy và split, Tuấn duyệt đo, Chương duyệt B |
| Q02 | PASS | `tools/quantum/characterize.py`, `tests/python/test_quantum_characterize.py` | |
| Q03 | PASS | `data/quantum/v4-char/`: 9600 dòng, complete, có failure ledger | Chạy lại khi có split gốc |
| Q04 | PASS | `tools/quantum/instance.py`, split guard có test; `instance.json` của hai run | |
| Q05 | PASS | Exhaustive, CP-SAT và greedy cho optimum 17 trên instance chính | |
| Q06 | PASS | SA, random_qubo, random_y, 10 seed, sample lưu trong `samples/` | |
| Q07 | PASS | `tests/python/test_quantum_encoding.py`; ground-state check trên instance thật ở 5 giá trị alpha | |
| Q08 | PASS | `tools/quantum/reduce.py`, `reduction.json` có ledger; ablation tắt gộp cột trùng cho q = 26 | |
| Q09 | PASS | `resource-audit.json`: q = 13, statevector 131072 byte, admitted | |
| Q10 | PASS | Venv Python 3.13, `tools/quantum/requirements-lock.txt`, env receipt | Chưa cài thử trên máy khác |
| Q11 | PASS | 20 run QAOA p = 1, 2 trên instance chính, đủ call và shot, counts xác nhận | Lượt phụ q = 21: cả 20 run chạm cap 60 s trước lượt xác nhận |
| Q12 | PASS | `tools/quantum/validate.py`, `catalogue.schema.json`; W1 và W2 qua `check_catalogue` | |
| Q13 | PASS_OFFLINE, G3 không đạt | [quantum-pilot-report.md](quantum-pilot-report.md): mọi solver, kể cả repair-only, đạt optimum 17; QAOA không tốt hơn classical | Validation, consumer (Q14+) |

Kiểm chứng local ngày 05/11 (Thứ 5): venv quantum `python -m pytest tests/python/test_quantum_*.py -q` cho 48 pass, 1 skip (test characterize cần `cv2`); Python 3.11 của pipeline `python -m pytest tests/python -q` cho 385 pass, 3 skip (các test cần OR-Tools).
