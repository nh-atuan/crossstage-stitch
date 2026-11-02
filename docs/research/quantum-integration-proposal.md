# Quantum integration cho adaptive GPU panorama-stitching runtime

## Material Passport

- Origin Skill: academic-research-suite / deep-research.
- Origin Mode: nghiên cứu định hướng và thiết kế thực nghiệm theo RQ đã xác định.
- Origin Date: 02/11 (Thứ 2), năm 2026; kế thừa khảo sát ngày 24/10 (Thứ 7).
- Verification Status: ANALYZED / PLANNED — hướng offline và owner Thành đã được người dùng chốt; chưa chạy QAOA của dự án.
- Version Label: quantum_catalogue_v3_thanh_parallel.
- Snapshot hiện trạng: `main@ea992fa`; bản lịch sử `d5d5acd` chỉ phản ánh khảo sát trước đây. Hướng/owner đã chốt không đồng nghĩa mọi policy, split hoặc budget đề xuất dưới đây được duyệt.

## 1. Quyết định và phạm vi bàn giao

**Primary Quantum Integration Path: Hướng A — Offline QUBO/QAOA catalogue selection for quality-constrained adaptive GPU panorama stitching.** Quantum giải bài toán chọn tập cấu hình nhỏ dùng chung cho nhiều cảnh từ bảng CUDA latency–QoR. Controller online và S1–S6 tiếp tục chạy cổ điển. B cố định trong thí nghiệm đầu; đề xuất feather, cần Chương và Nhóm xác nhận.

**Quyết định ngày 02/11 (Thứ 2): Thành chủ trì quantum tooling và thí nghiệm offline; chạy song song với runtime không quantum.** Owner stage, controller và harness không đổi. Chương review QoR/B và tổng hợp paper; Đế giữ policy/reference và consumer catalogue; Tuấn cung cấp sweep và giữ hạ tầng đo. [Quyết định kỹ thuật](../decisions/p3-offline-quantum-catalogue.md) và [task Phase 3](quantum-integration-tasks.md) thay phân công đề xuất cũ.

Catalogue được tạo offline là dữ liệu cấu hình, không là quantum state truyền vào CUDA. Pipeline mặc định phải build/chạy khi không cài quantum dependencies; giữ nguyên các kernels và public ABI S1–S6. QAOA không nhận RGB, không sửa H và không chạy trong đường xử lý từng ảnh. Chỉ sau validator và validation, Đế mới đưa catalogue vào consumer cổ điển; catalogue kém có thể ảnh hưởng budget/retry/QoR gián tiếp nên không được mặc định là an toàn hoặc nhanh hơn.

Quantum là thành phần bắt buộc của kế hoạch: phải chạy QAOA circuit trên simulator cho ít nhất một instance phát sinh từ sweep CUDA, lưu raw distribution/bitstrings và so với classical solvers. Chỉ có công thức QUBO, SA hoặc toy circuit chưa hoàn thành tích hợp ứng dụng. QPU là tùy chọn; phiên này chưa xác nhận quyền truy cập QPU. Hai tài liệu hiện tại bàn giao nghiên cứu và backlog, chưa báo kết quả quantum đã thực hiện.

Kết quả mong đợi là biết catalogue có nén được không, QAOA tạo nghiệm khả thi ở mức nào và lợi ích đến từ solver, catalogue hay controller/reuse. Quantum không bắt buộc phải thắng. Nếu kết quả âm, giữ artifact và báo cáo; không chuyển toàn bộ quantum thành Future Work, không đổi nhãn classical thành quantum. Nếu không có dữ liệu/circuit thực, ghi phần thực nghiệm chưa hoàn thành.

Đưa vào paper khi có provenance dữ liệu, encoding đúng, baseline mạnh và kết quả simulator thật trên instance của pipeline. Claim về runtime cần thêm chạy controller thật; nếu chưa có, giới hạn kết luận ở tối ưu offline. Đề xuất một subsection 1–1,5 trang gồm bảng/plot trong ngân sách nội dung đã có; không cộng thêm ngoài outline hiện tại. Không thay plan, common ABI hoặc estimator của thành viên khác trong lượt viết này.

### Primary Quantum Integration Path — đoạn dùng trong internal planning

Chọn Hướng A: chọn catalogue cấu hình offline bằng QUBO và QAOA cho runtime ghép panorama GPU dưới ràng buộc QoR. Đầu vào là bảng sweep CUDA đã kiểm tra, gồm latency, trạng thái và QoR cho từng cặp ảnh, cấu hình. Với B cố định, bài toán chọn ít cấu hình nhất sao cho mỗi cặp development có một lựa chọn đạt QoR và nằm trong mức regret so với Oracle. QAOA p=1,2 chạy trên simulator, tạo bitstrings để giải bài toán này; exhaustive, CP-SAT, greedy, SA và random dùng cùng instance làm đối chứng. Mọi nghiệm được giải mã và kiểm tra lại coverage; raw samples và nghiệm sau repair được báo riêng. Catalogue được xuất thành artifact có hash, rồi controller cổ điển dùng để chọn budget, escalate và reuse trên GPU. Quantum không nằm trong đường chạy mỗi ảnh. Validation chọn tham số; final chỉ chạy sau khi khóa protocol. Đánh giá gồm kích thước catalogue, coverage, regret, feasibility, chi phí solver và latency thực của controller. Nếu classical solver giải nhanh hoặc repair quyết định kết quả, báo feasibility hay negative result. Hướng này giữ nguyên S1–S6 và RQ chính, đồng thời tạo thí nghiệm quantum có thể kiểm chứng.

## 2. Hiện trạng và câu hỏi nghiên cứu

Đã đọc [plan](../plan.md), [stages](../stages.md), [conventions](../conventions.md), [S1](../stages/s1-preprocessing.md), [benchmark/QoR/S4–S5](../decisions/benchmark-qor-s4-s5.md), [S6](../stages/s6-warp-blend.md), [bàn giao P1](../stages/s6-p1-implementation.md), [khung workshop](../../PARMA-DITAM_2027_Workshop/README.md); kiểm tra source, test, mock và results trong checkout.

| Có bằng chứng tại checkout | Chưa có bằng chứng tại checkout |
|---|---|
| Source S1–S6, controller và harness fixed/oracle/adaptive đã merge | Quantum adapter/solver/circuit trong `tools/quantum/` |
| Báo cáo nghiệm thu Phase 2 ghi root CTest 15/15 và sanitizer trên T4 | Raw characterization đủ provenance được tooling quantum kiểm tra; không thay bằng summary |
| `tools/qor/`, policy versioned và `tools/controller/audit_sweep.py` dùng predicate chuẩn | Policy file vẫn `pilot-not-team-approved`; owner phải duyệt policy dùng cho quantum |
| FP64 reference, same-H runner và historical classical formulation check | Catalogue từ QAOA, raw counts, solver logs, QPU runs hoặc benchmark W1/W2 |

`results/s6-validation-20261023-final/s6-timings.csv` chỉ đo S6 steady-state. `results/s6-qor-smoke/runs.csv` có H mock ORB/SIFT nhưng không có t_e2e; đổi descriptor không phải sweep S/K/R. Không dùng chúng làm characterization chính. Không thấy artifact trong checkout không chứng minh thành viên khác chưa có dữ liệu trên nhánh riêng.

RQ chính giữ nguyên:

*Can a scene-aware cross-stage runtime reduce GPU panorama-stitching latency by dynamically allocating computational budgets while maintaining a predefined Quality-of-Result constraint?*

**RQ-Q:** Với cùng bảng CUDA latency–QoR, cùng coverage/regret constraints và ngân sách tìm kiếm đã khóa, QAOA simulator p=1,2 tạo catalogue có cardinality, feasibility, validation regret và thời gian offline như thế nào so với exhaustive/CP-SAT, greedy, SA và random?

Tách ba kiểm định: catalogue có ích không; encoding có đúng không; QAOA có thêm giá trị không. Biết trước mọi latency/QoR của một pair thì chọn cấu hình nhanh nhất chỉ cần quét C. Tính tổ hợp nằm ở việc nhiều pair cùng chia sẻ catalogue; không gán NP-hard cho một phép argmin online.

Phase hiện tại là Phase 3, deadline chung 05/11 (Thứ 5); Phase 4 là 09/11 (Thứ 2), Phase 5/nộp nội bộ là 13/11 (Thứ 6). Nhánh runtime không quantum tiếp tục dù quantum thiếu input hoặc thất bại. Thiếu prototype quantum được ghi là chưa hoàn thành, không đổi deadline hoặc đánh dấu quantum đã làm bằng kết quả baseline.

## 3. Khảo sát và bảng so sánh A–F

Khảo sát định hướng đến ngày cập nhật, không phải systematic review/PRISMA. Truy vấn gồm minimum set cover QAOA/QUBO, configuration selection, penalty scaling, multi-objective QAOA, quantum robust fitting, graph-cut segmentation và QML classical benchmarks. Ưu tiên paper gốc, code của tác giả và docs chính thức; không dùng blog, citation count hoặc uy tín venue thay bằng chứng phương pháp. Preprint/bản xuất bản của cùng công trình được tính một nguồn. Phạm vi đọc ghi ở §12; chưa kiểm toán retraction toàn corpus.

| Tiêu chí | A — catalogue cover | B — Pareto catalogue | C — escalation ladder | D — S5 robust fitting | E — S6 seam | F — quantum selector |
|---|---|---|---|---|---|---|
| Khớp RQ | Cao: nén lựa chọn budget | Cao, thêm objective | Cao: retry cost | Gián tiếp: đổi estimator | Thấp khi QoR đo alignment | Có thể khớp selection |
| Dữ liệu | CUDA sweep + QoR + split | A + margin/memory/retry | Transition/reuse traces | Matches/hypotheses, lỗi H | Overlap + composition labels/metric | Features/labels và split đủ lớn |
| Công triển khai | Thấp nhất tương đối | Vừa–cao | Cao | Cao | Cao | Vừa–cao |
| Rủi ro | Slack bits, trivial cover, repair | Trọng số tùy ý, thêm assignment | Cost không cộng độc lập | Thay R/estimator/reuse | Metric sai mục tiêu, pixel quá lớn | Baseline mạnh, embedding cost |
| Vai trò quantum thật | Giải QUBO cover | Giải họ QUBO theo trọng số | Chọn thứ tự/chuyển tiếp | Chọn outlier/hypothesis/subset | Nhãn A/B pixel/superpixel | Quantum kernel/VQC |
| Classical baseline | Exhaustive, CP-SAT, greedy, SA, random | Native multiobjective MIP, greedy/SA | Enumeration/DP/MIP, heuristic ladder | GPU RANSAC/MSAC, refinement | Min-cut nếu submodular, SA/MIP | Logistic/SVM/RF/MLP, separable ablation |
| Giá trị kiểm chứng | Cardinality, coverage/regret, cost | Pareto/hypervolume, sensitivity | Expected/measured e2e/retry | Geometry/consensus và e2e | Seam/ghosting và runtime | Calibration, prediction và overhead |
| Tác động S1–S6 | Không đổi kernel/ABI | Không đổi nếu chỉ offline | Cần transition telemetry | Đổi S5, tương tác S4 | Đổi S6/blender | Thêm feature extraction |
| Controller/reuse | Cùng controller giữa solver | Có thể đổi preferences | Tác động trực tiếp | Invalidate assumptions | Có thể đổi B/gate | Train/inference path mới |
| Khả năng kịp lịch | Cao nhất nếu có sweep sớm | Sau A | Thấp khi thiếu controller | Thấp | Thấp | Thấp khi thiếu feature dataset |
| Khuyến nghị | **Primary, bắt buộc prototype** | Extension sau A | Backlog | Không chọn hiện tại | Không chọn khi thiếu metric | Không chọn cho MVP |

A có tiền lệ set cover và configuration selection [R1–R3]. B có multiobjective QAOA [R4], nhưng regret chứa min/assignment, không tự thành quadratic bằng cộng vài metric. Mở B phải định nghĩa biến, chuẩn hóa objective, quét trọng số/Pareto và so classical cùng objective; không chuyển kết quả MO-MAXCUT thành catalogue evidence.

Catalogue/portfolio selection cũng có tiền lệ cổ điển [R13]. Việc thêm QAOA vào một bài toán cover đã biết chưa đủ làm novelty; đóng góp phải đến từ workload runtime, benchmark và kết quả kiểm chứng được.

C cần chi phí chuyển tiếp thực: tăng R có thể giữ matches/hypotheses; đổi S/K có thể recompute. Sweep độc lập chưa cung cấp mô hình đó. D có CVPR/ECCV papers [R6–R8], nhưng quy mô/backend không chứng minh thay S5 GPU khả thi. E có tiền lệ segmentation quantum [R9], khác panorama seam; energy nhị phân submodular có baseline min-cut mạnh [R10]. F cần baseline ML tương đương và tính cả preprocessing/inference [R11]. Công sức và tiến độ trong bảng là suy luận cho repo, không phải số đo của các paper.

## 4. Dữ liệu, Oracle và bài toán gốc

### 4.1. Characterization là task đầu tiên

Hoàn thành CUDA sweep qua harness của Tuấn và pipeline Đế tích hợp. Mỗi pair/config cần 2 warmup + 10 đo theo conventions, seed thuật toán cố định và H tất định. Nếu nghiên cứu nhiều seed RANSAC, dùng cohort/run riêng; không trộn thành 10 timing repeats.

Thành nhận và kiểm tra dữ liệu qua adapter riêng; tái dùng semantics của `tools.controller.audit_sweep` và `tools.qor.evaluate.quality_decision`, không viết một predicate pass khác. `--mode oracle` của harness thực thi các cấu hình đầu vào, chưa tự bảo đảm đó là Oracle dưới QoR: phải audit grid rồi chọn offline. Các lỗi legacy của `tools.analysis.analyze_gonogo` không được đưa vào nhánh quantum bằng cách đọc nhãn pass cũ.

Join `runs.csv`, `qor.csv`, `meta.json`; bảng dẫn xuất riêng `characterization.csv` có:

```text
pair_id,scene_id,split,config_id,S,K,R,B_fixed,latency_median_ms,
latency_p95_repeat_ms,qor_pass,overlap_coverage,reference_status,status,
n_measured,policy_version,run_id
```

QoR pass gọi lại evaluator/report hiện có, không nhập tay hoặc tự thêm threshold. Metadata kèm revision/dirty/source hashes, GPU/clock/driver/CUDA, seeds, grid, split/policy/input hashes, timestamp, units và timing boundary. Không sửa schema common. Join thiếu/trùng, repeat trùng, config thiếu hoặc policy lẫn nhau phải báo lỗi; không nội suy cost.

`latency_p95_repeat_ms` chỉ là chẩn đoán nhiễu 10 repeat, ghi quantile method. Theo conventions, p95 chính của hệ thống tính trên **phân phối pair medians**, không xem repeat là scene độc lập. Oracle chính dùng median. Không đủ 10 repeat thành công thì ghi incomplete/runtime failure, giữ ledger và rerun theo protocol; không nhận median từ phần repeat còn sống như đủ dữ liệu.

`t_ic=median(t_e2e)` đo từ RGB đã trên device đến panorama sẵn sàng trên device, gồm host/controller/sync, loại decode/H2D/D2H theo conventions. Sweep fixed-config không có retry; adaptive phải đo lại toàn đường chạy. S6-only timing không thay thế t_e2e.

### 4.2. Split và policy

UDIS-D là benchmark chính theo Đế. Giữ final set đã có; đề xuất chia phần development được phép dùng thành development-fit và validation theo **scene/sequence/source-image group**, seed/hash manifest khóa trước fit. Nếu dùng fallback hash split của conventions, validation chỉ tách từ development đó, không lấy thêm final. Thiếu scene metadata phải ghi hạn chế và kiểm tra ảnh chung/duplicate; không tạo scene labels giả. Tỉ lệ/danh sách cần Đế/Tuấn duyệt.

Policy hiện là `qor-pilot-v1`, `pilot-not-team-approved`: coverage 0.95, SSIM drop 0.02, PSNR drop 0.5 dB. Chưa phải threshold chính thức. Policy sensitivity phải được Đế duyệt trước final; không nới để quantum trông tốt hơn. Metric hiện đo alignment từ H trên miền reference cố định, chưa đo composition; giữ B_fixed.

Gọi a_ic=1 khi run hợp lệ, đủ repeats, reference đáng tin và evaluator xác nhận pass. C là configuration space hữu hạn đã đăng ký và đo đầy đủ:

$$
I_f=\{i:\mathrm{reference\_status}_i=\mathrm{ok},\ \exists c\in C:a_{ic}=1\},\qquad
t_i^*=\min_{c\in C:a_{ic}=1}t_{ic}.
$$

Oracle chỉ là cận offline trên C. Fixed-High quality reference khác Fixed-Best latency baseline. Nếu Fixed-High nằm trong C và reference hợp lệ, nó tự pass policy tương đối; không loại reference chỉ để tạo unachievable. Nếu candidate-space loại reference thì công khai, fallback-space riêng.

Reference failure/unreliable không vào cover constraints, nhưng giữ denominator toàn tập. Reference hợp lệ mà không cấu hình pass là unachievable_in_C; thiếu measurements là incomplete. I_f rỗng thì dừng instance generation, không trả catalogue rỗng như thành công.

### 4.3. Catalogue dưới regret constraint

Với rho >= 0:

$$
U_i(\rho)=\{c\in C:a_{ic}=1,\ t_{ic}\le(1+\rho)t_i^*\},\qquad
\min_{y\in\{0,1\}^{|C|}}\sum_c y_c\quad\mathrm{s.t.}\quad
\sum_{c\in U_i(\rho)}y_c\ge1\quad(i\in I_f).
$$

Catalogue P={c:y_c=1}. Khi P có cấu hình pass cho pair i, regret là `r_i(P)=min_{c in P:a_ic=1}t_ic/t_i* - 1`; không có thì regret +∞ và covered=false. Dùng token rõ cho infinity, không JSON NaN. Báo QoR coverage và near-Oracle coverage (`r_i<=rho`) riêng trên I_f và toàn tập; không chỉ báo mean regret của survivors.

Objective chính là cardinality, regret là constraint. Tie-break config_id cố định, không dùng final latency. Fixed-Best chọn trên development/validation theo protocol Nhóm, không chọn lại trên final. Thứ tự escalation/reuse, memory và margin chưa nằm trong cover objective. Cover guarantee chỉ đúng trên bảng fit; không bảo đảm online selector chọn đúng hoặc QoR trên ảnh mới.

## 5. QUBO, reduction và validator

### 5.1. Encoding và điều kiện đủ

Đặt A_ic=1[c in U_i], d_i=|U_i|>=1, b_i=ceil(log2 d_i), slack s_i=sum_k 2^k z_ik. d_i=1 không cần slack. Các mã slack lớn hơn d_i−1 không thể tạo residual zero với y không hợp lệ, nên không cần constraint phụ để cấm chúng.

$$
E(y,z)=\lambda\sum_c y_c+\alpha\sum_i\left(\sum_c A_{ic}y_c-1-s_i\right)^2,\qquad \lambda>0.
$$

Đây là cụ thể hóa penalty/slack đã biết [R1,R2,D2]. Suy dẫn dự án: cover toàn C có cost lambda|C|; assignment vi phạm equality có residual nguyên với bình phương >=1. Vì vậy **alpha > lambda|C|** đủ để mọi ground state là cover tối thiểu trong số học chính xác. Sau reduction có thể dùng `alpha>lambda k_ub`, k_ub là số biến chưa cố định của một cover đã được validator xác nhận; fixed-variable offset giữ riêng. Mặc định đề xuất lambda=1, alpha=k_ub+1, không claim penalty tối thiểu hoặc QAOA tìm được ground state.

Với x=(y,z), m là số constraint đang encode và v_i chứa A cùng hệ số slack âm, dùng upper-triangular convention:

$$
E(x)=e_0+\sum_jQ_{jj}x_j+\sum_{j<k}Q_{jk}x_jx_k,
$$
$$
e_0=\alpha m,\quad Q_{jj}=\lambda\mathbf1_{j\in y}+\alpha\sum_i(v_{ij}^2-2v_{ij}),\quad
Q_{jk}=2\alpha\sum_i v_{ij}v_{ik}.
$$

Không double-count off-diagonal khi dùng API dạng matrix. Đổi x_j=(1−Z_j)/2 cho Ising: J_jk=Q_jk/4; h_j=−Q_jj/2−sum_{k!=j}Q_min(j,k),max(j,k)/4. Giữ constant để test energy. Duyệt basis states so residual/QUBO/Ising, bit-order/endian và factor 2 trước khi chạy circuit.

### 5.2. Reduction với mapping phục hồi

Lặp đến fixed point: gộp hàng trùng; U_i subset U_j thì bỏ hàng j; singleton buộc cột bằng 1 rồi bỏ hàng đã phủ; bỏ cột không phủ hàng còn lại; incidence cột trùng giữ đại diện config_id lexicographic. Lưu forced columns, removed rows, representative map, objective offset và hashes. Hàng rỗng phải báo infeasible, không bỏ đi.

Reduction bảo toàn optimum cardinality trên development tại rho đang xét. Gộp cột trùng không giữ mọi catalogue đồng tối ưu hoặc validation performance của từng cột. Primary solvers nhận cùng reduction; thêm unreduced classical và duplicate-column on/off ablation. Không xóa edge/cột để vừa qubit rồi gọi reduction exact. Chưa dùng column dominance trong MVP.

Decode phục hồi forced columns và kiểm tra **toàn bộ original incidence matrix**. Phân biệt equality-feasible QUBO sample; cover-feasible y nhưng slack sai; y không cover. Tái tính slack là canonicalization không sửa membership, báo riêng với repair thêm/bớt cấu hình. Energy thấp chưa chứng nhận feasibility.

### 5.3. Tài nguyên và sensitivity

$$
q_{before}=|C|+\sum_i\lceil\log_2d_i\rceil,\qquad
q_{after}=|C_{remaining}|+\sum_{i\in I_{remaining}}\lceil\log_2d'_i\rceil,\qquad d'_i=\sum_{c\in C_{remaining}}A'_{ic}.
$$

Báo q, nonzero couplings/density/max degree, coefficient min_nonzero/max_abs và ratio. Mỗi residual tạo clique nên ít bits chưa nghĩa ít two-qubit gates. Scale toàn Hamiltonian bằng một hằng dương, lưu scale/energy gốc; không clip coefficients. Scaling giữ optimum nhưng thay góc/landscape và precision hữu hạn.

Pilot đề xuất q_after<=20. Complex128 statevector cần `16*2^q` bytes chỉ cho vector: 20 bits = 16 MiB; 30 bits = 16 GiB, chưa buffers. Đo peak RAM/threads và admission trước chạy. q=0 sau reduction là solved_by_reduction, không tạo circuit giả. Optimum=1 hoặc greedy giải rất nhanh nhưng q_after>0 vẫn phải chạy QAOA trong budget, báo feasibility/control result. Full instance quá lớn: cohort lấy từ development theo manifest/seed khóa trước khi xem solver, ghi restricted_instance và toàn mapping/số pair/config bị loại; đánh giá thêm full development/validation, không claim optimum full grid. Nếu mọi instance đều q_after=0 hoặc vượt tài nguyên thì báo giới hạn ứng dụng; toy circuit chưa hoàn thành workload quantum.

Grid pilot đề xuất rho={0,0.05,0.10,0.20}; alpha/(lambda k_ub)={0.5,1,2,4} cộng mặc định (k_ub+1)/k_ub khi k_ub>0. Mức vi phạm strict bound chỉ là stress test, không có bảo đảm tương đương. Đây là tham số nghiên cứu chờ duyệt; không thay quality policy và không tune trên final.

## 6. Solvers, budget và artifact quantum

### 6.1. Baseline và công bằng

Exhaustive/CP-SAT/greedy giải native cover; SA/QAOA giải cùng QUBO. Không ép exact solver dùng encoding bất lợi. Exhaustive native duyệt y; encoding exhaustive duyệt (y,z) chỉ cho unit-size.

| Solver | Cấu hình pilot đề xuất | Bắt buộc báo |
|---|---|---|
| Exhaustive | n_y=|C_remaining|<=20, timeout 60 s, cardinality tăng | Optimum hoặc incomplete, không giả certificate khi timeout |
| CP-SAT | Integer cover, 1 thread, seed cố định, timeout chung | OPTIMAL/FEASIBLE/UNKNOWN, incumbent/bound/gap; FEASIBLE không là optimal [D3] |
| Greedy | Maximum uncovered gain, tie config_id, reverse-delete | Feasible cover và wall time, kể cả khi rất nhanh |
| SA | QUBO Metropolis, 4 restarts, schedule/beta/proposals được khóa | Raw bits, energy/feasibility, objective/delta-evaluation count |
| Random | Uniform QUBO bits và random-y với cardinality distribution khóa trước | Draws, raw success, wall time, same-repair controls |
| QAOA | p=1,2 riêng; uniform initial state, X mixer; expected Ising energy finite shots [R12,D1] | Circuit, angles, optimizer trace, seeds, shots, counts và decode |

Hai phép so: (1) anytime/time-to-quality với cùng cap wall-clock đề xuất 60 s/seed/instance/depth sau preprocessing chung; (2) resource audit gồm objective calls, samples, shots, proposals và restarts. Một shot không bằng một SA move hoặc CP-SAT node. SA/random có thêm matched-sample run, nhưng không chuyển phép so đó thành equal-compute claim. Mọi solver dùng cùng host/thread policy; backend CPU simulator được ghi rõ.

QAOA pilot: 10 seeds [0..9], 4 initializations/seed, tối đa 100 objective evaluations/initialization, 1024 shots/evaluation, 4096 shots xác nhận cho angles được chọn. Tối đa 413696 shots/seed/instance/depth gồm xác nhận; timeout có thể kết thúc sớm. Optimizer đề xuất SPSA; 100 là **objective calls**, không phải 100 SPSA steps. Lưu optimizer/sampler/transpiler seeds. Training, compile/sampling, decode và repair đều tính vào run wall cap khi so equal-wall-time; nếu hết cap trước confirmation thì ghi incomplete, không giấu thời gian. p=1 và p=2 có budget riêng.

Exact statevector expectation chỉ là diagnostic, không tính là finite-shot run. Việc chọn p/optimizer/penalty dùng validation; tính cả hyperparameter search của mọi solver. Warm-start là extension có chi phí tạo seed riêng. Các budget ở đây chưa được đo hoặc khóa chính thức.

### 6.2. Decode, repair và lựa chọn nghiệm

Chọn raw cover-feasible P có cardinality nhỏ nhất từ samples trong budget, tie lexicographic. Samples trong optimizer trace được phép đóng góp vào best-found catalogue; distribution gộp đó là adaptive mixture, không phải xác suất của một circuit cố định. Raw cover-feasibility/equality-feasibility probability và xác suất optimum chính dùng riêng 4096 confirmation shots tại angles cuối đã chọn, với số shots thực chạy ghi rõ. Báo pooled best-found và confirmation distribution riêng. Không có feasible sample thì trả no_feasible_sample, không bí mật thay greedy; timeout trước confirmation không được phát probability estimate giả.

Repair optional: thêm cấu hình phủ nhiều uncovered rows nhất rồi reverse-delete theo thứ tự khóa. Áp dụng giống nhau cho QAOA/SA/random và **empty-start repair-only**. Lưu raw/post-repair size, coverage, added/deleted configs, bit flips, repair rate/time. Không dùng validation/final labels để repair. Nếu repair tạo phần lớn kết quả, gọi hybrid QAOA+classical repair và báo giới hạn đóng góp quantum [R5].

`solver_runs.jsonl` cần instance/reducer hashes, solver/backend/version, p/angles, seeds, requested/executed shots, evaluations/restarts, timeout/status, raw-counts path, raw/repaired energy/feasibility/objective, optimum/bound nếu có, wall-time breakdown và peak RAM. `catalogue.json` chứa config_id/S/K/R/B, selected/forced/repaired provenance, fallback_id, policy/split/source hashes và full/restricted scope. Catalogue truy ngược được đến sample; solver fail không xuất file mang nhãn quantum thành công.

## 7. Protocol đánh giá hệ thống

### Hai nhánh triển khai và đối chứng W0/W1/W2

| Nhóm | Cách chọn cấu hình/catalogue | Vai trò thực nghiệm |
|---|---|---|
| W0 — Adaptive-original | Runtime/ladder baseline hiện có, cùng Fixed-Best và Oracle được audit | Nhánh không quantum độc lập, không bị chặn bởi solver/môi trường quantum |
| W1 — Adaptive-classical-catalogue | Exact/CP-SAT/greedy hoặc classical solver được khai báo chọn catalogue | Đo giá trị thiết kế catalogue và làm đối chứng cho solver quantum |
| W2 — Adaptive-QAOA-catalogue | QAOA simulator chọn catalogue; raw/repaired được ghi riêng | Đo đóng góp solver trên cùng bài toán; online consumer vẫn cổ điển |

Thành triển khai và chạy solvers offline cho W1/W2; Đế/Tuấn quản lý consumer và instrumentation online. Không tạo runtime branch phụ thuộc Qiskit/Aer. W0 không cần đợi consumer mới; W1 cũng không cần đợi QAOA. Khi consumer chưa có, chỉ so solver/catalogue offline và ghi W1/W2 e2e `not_measured`, không suy latency online từ bảng cover.

W1/W2 phải dùng cùng preprocessing/reduction, candidate set, policy/reference, split, B, consumer, selector family và tuning budget, self-check, reuse, retry/fallback cap. Lựa chọn catalogue và repair chỉ dùng development-fit; validation chọn tham số; test chỉ đánh giá sau lock. Không lấy seed tốt nhất trên test. Bảng chính giữ tất cả nghiệm và failures; matched-size control là bổ sung, không thay bảng chính. W0/W1 khác consumer hoặc policy phải công khai; không quy mọi khác biệt đó cho QAOA.

Nhánh quantum đầu dùng một quality policy đã được Đế duyệt, không đồng thời chạy toàn lưới A5. User đã chốt A5 là controller theo policy hiệu chỉnh trên development; đó là sensitivity riêng do Chương chủ trì và không tự đổi policy của instance quantum.

### Protocol khi có consumer đã kiểm chứng

1. Khóa split trước fit. Development-fit tạo catalogue/selector; validation chọn rho, penalty, solver budget và controller thresholds. Final chỉ mở sau protocol lock, không dùng chọn seed đẹp hoặc fit lại.
2. Group theo scene/source ảnh; report số groups/pairs, duplicate/missing/failure mỗi split. Khóa C, reference, B và policy; Đế xác nhận reference đáng tin.
3. So Fixed-High, Fixed-Best, Oracle-C, Adaptive-original (W0), Adaptive-classical-catalogue (W1), Adaptive-QAOA-catalogue (W2) và Oracle-P. W0 là ladder baseline đã đăng ký, không tự gọi là toàn grid nếu chưa có implementation đó. Oracle-P tách catalogue loss khỏi controller loss. Giữ cùng selector family/data/tuning budget giữa W1/W2, self-check/reuse/fallback; thêm matched-size comparison khi có, không xóa nghiệm khác size khỏi bảng chính.
4. Initial selector chỉ dùng shape/features rẻ có trước quyết định, tính extraction cost; sau chạy mới dùng tín hiệu S5. Cấm pair_id lookup, H_ref, Oracle/Pass labels hoặc features của cấu hình chưa chạy. Offline labels chỉ dùng train/evaluate. Fixed-High fallback có thể ngoài P: báo |P|, |P union fallback|, retry/fallback rate, quality và latency sau fallback.
5. P là tập, chưa phải ladder đơn điệu. Đế xác định thứ tự classical theo development cost, khóa retry cap/termination; không suy ra tăng S/K/R luôn tăng quality. Chỉ tăng R mới có thể giữ matches/hypothesis state theo S5 contract; đổi S/K invalidate dependencies; đổi H làm lại S6. Reuse on/off cần đối chiếu cả output/failure.
6. `t_e2e` gồm controller, sync, retry và cache bookkeeping; giữ cả latency run thất bại. False acceptance = accepted-but-offline-fail/accepted-evaluable; false rejection = rejected-but-offline-pass/rejected-evaluable. Ghi counts và denominator, reference failure riêng; tín hiệu S5 không bảo đảm QoR.
7. Report median/p95 pair latency, QoR violations, coverage và regret distribution. Paired difference cùng pairs, scene-cluster bootstrap 95% CI; đề xuất 2000 resamples/seed khóa. Thiếu group dùng pair bootstrap và ghi hạn chế. Solver seeds không là image pairs; uncertainty solver báo riêng. Ít instance chỉ là descriptive case study.
8. Final là một campaign protocol đã khóa, gồm repeats định trước. Crash có thể resume với manifest/log; không tune theo final feedback. Thay đổi phương pháp sau xem final phải ghi contamination và cần holdout mới được nhóm duyệt.

### 7.1. Chi phí offline và online

Ghi riêng T_characterization, T_reduction, T_encoding, T_parameter_search, T_compile_or_embed, T_sampling, T_decode_validate, T_repair, T_online_controller, T_end_to_end. Optimizer loop có thể bao gồm compile/sampling/decode: dùng exclusive timers để tính tổng, inclusive timer chỉ tham khảo, không cộng đôi. QPU thêm queue/network/programming/readout/access time; anneal duration không là wall time.

Quantum offline không cộng vào từng ảnh. Công bố tổng offline thực; với N pairs, cost phân bổ `T_offline/N + mean(T_e2e)`. Characterization chung báo một lần, solver-specific marginal cost riêng. So với pipeline không cần sweep phải tính toàn characterization khi xét break-even. Nếu tiết kiệm online delta_t>0, hòa vốn chi phí bổ sung tại `N>DeltaT_offline/delta_t`; delta_t<=0 không có hòa vốn theo mô hình này.

### 7.2. Ablation bắt buộc

| Ablation | Nội dung kiểm tra |
|---|---|
| W0 original vs W1 classical catalogue | Giá trị catalogue/consumer; công khai khác biệt, giữ selector family/reuse và budget đối chứng |
| Classical vs quantum catalogue | Cùng instance/policy/controller; thêm matched-size |
| QAOA vs SA/random/exact/greedy | Same-QUBO QAOA/SA, native cover exact/greedy |
| Reduction on/off | Optimum/mapping, bits/couplings/time; resource-limit ghi rõ |
| Penalty/rho sensitivity | Raw feasibility, dynamic range và validation regret |
| Raw vs repaired vs repair-only | Đóng góp quantum tách greedy hậu xử lý |
| Reuse on/off; escalation on/off | Retry cost, false acceptance/rejection, termination |
| Policy sensitivity | Chỉ policy versions được Đế duyệt trước final |
| Ideal finite-shot vs noise model | Model/version rõ, noise simulation chưa là QPU |
| Simulator vs hardware | Chỉ khi có hardware thực; cùng instance/decode |

Quantum selector không thuộc primary: classical-vs-quantum selector ablation là N/A có lý do. B ngoài MVP; nếu mở rộng phải có Pareto/weight sensitivity, không chỉ một scalar score.

## 8. Môi trường và ranh giới ownership

Không thêm/cài dependency trong lượt viết. Đề xuất môi trường riêng Python 3.12 CPU, tách `tools/requirements.txt`. Task môi trường thử compatibility, pin versions/hashes vào `tools/quantum/requirements-lock.txt`, lưu OS/CPU/RAM/threads và license/NOTICE trực tiếp/transitive.

Classical core/CLI/tests chỉ cần stdlib/NumPy, CP-SAT có extra OR-Tools riêng; Qiskit/Aer được import muộn trong backend QAOA. Package initializer và consumer catalogue online không import quantum SDK. Điều kiện này phải có smoke test trong môi trường không cài Qiskit/Aer để W0/W1 tiếp tục độc lập; lockfile quantum không trở thành requirements của pipeline.

- Qiskit và Qiskit Aer: Apache-2.0 theo LICENSE chính thức [D4,D5]. QAOAAnsatz hoặc circuit tương đương cần test với oracle NumPy; CPU simulator đủ cho pilot, không cần account [D1].
- OR-Tools CP-SAT: Apache-2.0 [D6]; native integer cover và status handling theo [D3].
- NumPy có sẵn ở tooling; validator/exhaustive/greedy/SA/random có thể dựa NumPy/stdlib. SPSA nhỏ có thể cài riêng với test, chưa thêm optimizer package mặc định.
- Ocean/QPU ngoài install plan MVP. Trước dùng cần xác minh version/license/API, quyền truy cập, chi phí, embedding/chain strength/break handling/reads. Backend chỉ cần instance được phép chia sẻ, không cần RGB/manuscript.

CLI dự kiến, **chưa tồn tại**: `tools.quantum.characterize`, `build_instance`, `solve_classical`, `solve_qaoa`, `evaluate_catalogue`. Tests dự kiến `tests/python/test_quantum_*.py`; data/results/build theo gitignore. Catalogue schema riêng, không sửa common ABI. Thành chủ trì quantum tooling, classical controls, simulator, tests và báo cáo solver, đồng thời giữ S2–S3/A3. Đế giữ controller/S4–S5 và duyệt policy; Tuấn giữ harness/S1/common/merge; Chương giữ S6/A4/A5, review QoR và tổng hợp paper. Chi tiết ở [quantum-integration-tasks.md](quantum-integration-tasks.md).

## 9. Rủi ro và tiêu chí phản bác

| Rủi ro lớn | Giảm thiểu / bằng chứng |
|---|---|
| Chưa có sweep, gần mốc nhóm | Q00 prerequisite; toy chỉ kiểm tooling, không gọi workload thật |
| Exact/greedy giải tức thì hoặc optimum=1 | Báo kết quả; không tạo độ khó giả để cứu quantum |
| Slack/couplings vượt RAM | Exact reduction, admission, cohort khóa trước; báo restriction/full-validation |
| Penalty/encoding sai | Proof, exhaustive energy/basis tests, sensitivity, original validator |
| Repair tạo lợi ích | Raw/post-repair/empty-start controls và charge repair time |
| Leakage/reduction chọn cột đại diện có lợi giả | Scene split/final lock, duplicate-column ablation |
| Cover đúng nhưng selector sai | Oracle-P vs adaptive-P, false acceptance/retry/fallback cost |
| B và QoR không tương thích | B_fixed; seam claim cần composition evaluator được duyệt |
| Speedup CUDA/reuse gán cho quantum | Cùng controller/reuse giữa catalogues, cost offline/online tách |
| QPU chưa có hoặc noise cao | Simulator evidence ghi rõ; hardware là optional, không trì hoãn MVP |

Nếu quantum không tốt hơn greedy cùng size hoặc exact giải rất nhanh, bác bỏ lợi ích solver trên workload đã xét; không suy ra mọi quantum algorithm vô dụng. Nếu catalogue cổ điển cũng không giúp sau overhead, báo negative result hệ thống. Nếu chỉ training cover tốt mà validation xấu, kết luận không generalize. Không đổi objective/metric sau kết quả để cứu giả thuyết.

## 10. Go/No-Go và claim

| Gate | Go | No-Go và đầu ra bắt buộc |
|---|---|---|
| G0 — dữ liệu | CUDA sweep complete, policy/split/grid có owner/hash | Chặn ứng dụng; tooling tests và missing-data ledger, quantum chưa hoàn tất |
| G1 — encoding/tài nguyên | Native/QUBO/Ising/reduction tests đúng, instance thật phù hợp | Sửa lỗi trước solver; triviality/resource negative report nếu không có instance |
| G2 — quantum tối thiểu | QAOA p=1,2 trên instance thật, raw samples/logs, controls đủ | Không feasible vẫn giữ kết quả; crash/no circuit chưa là prototype hoàn tất |
| G3 — giá trị | Trade-off validation có ích theo criterion nhóm khóa trước | Dừng mở depth/QPU/B–F, giữ feasibility/negative result và artifacts |
| G4 — paper/hệ thống | Controller thật cho system claims; final protocol lock; số liệu truy vết | Giới hạn claim offline; thiếu minimum evidence thì ghi chưa hoàn thành |

Khuyến nghị Go cho A ở quy mô simulator có budget, bắt đầu bằng sweep CUDA/native cover. Dù classical mạnh, vẫn thực hiện prototype quantum tối thiểu trên instance thật đủ nhỏ và báo đối chứng. Dừng mở rộng khi hết budget, instances không phù hợp hoặc classical/repair chi phối. Không đổi S5/S6 hoặc mở QPU trả phí để kéo dài hướng đã thất bại. Quantum gates không thay Go/No-Go runtime trong plan. Mọi threshold hệ thống mới cần Nhóm duyệt trước final.

| Claim có thể viết khi đủ evidence | Evidence cần |
|---|---|
| Encoding đúng cover tại rho | Proof, exhaustive tests, decode validation |
| QAOA simulator tạo catalogue cụ thể | Circuit/counts và instance/sample/catalogue hashes |
| Cardinality/coverage/regret đã đo | Split-aware tables và failure denominator |
| Case study feasibility/negative result | Classical controls, raw/repair metrics và timers |
| Runtime nhanh hơn trên tập đã đo | Actual e2e dưới cùng policy, CI/ablation; không tự quy cho quantum |

| Claim chưa được phép | Lý do |
|---|---|
| Quantum advantage / exponential speedup | Thiếu scaling và so tài nguyên tương ứng |
| QAOA vượt mọi classical methods | Chỉ có baseline/configurations thực sự chạy |
| SOTA panorama quality / quantum cải thiện ảnh | Alignment policy chưa đánh giá đầy đủ composition |
| Quantum guarantees QoR | Training cover không bảo đảm selector/ảnh mới |
| Simulator dự báo QPU performance | Noise/connectivity/cost khác |
| Realtime quantum stitching | Quantum offline, chưa có benchmark realtime |

## 11. Trạng thái và quyết định còn chờ

Ngày 02/11 (Thứ 2) đã chốt hướng offline, Thành chủ trì và nhánh không quantum chạy song song; cập nhật hiện trạng source/plan sau Phase 2. Lượt này chỉ cập nhật nghiên cứu/kế hoạch, chưa triển khai solver, chạy QAOA/QPU hoặc tạo sweep CUDA mới. Kết quả 441 ma trận cover trong bản cũ là lịch sử kiểm tra nhỏ, không thay tests của implementation mới hay chứng nhận mọi reduction.

Ngày 24/10 (Thứ 7) đã chạy lại kiểm tra thiết kế độc lập bằng Python/stdlib: **441 ma trận có 1–3 hàng và 1–3 cột, mọi hàng khác zero, tổng 19.374 assignments (y,z)**, lambda=1 và alpha=|C|+1. Residual energy, upper-triangular QUBO và Ising có offset khớp trên mọi assignment; mọi ground state cover ma trận gốc với cardinality bằng native optimum. Receipt và hash script ở `results/quantum-design-check-20261024.json`, script cục bộ `build/verify_quantum_formulation.py`. Đây chưa kiểm tra code reducer/QAOA, noise hoặc performance; không tính là prototype quantum hoàn tất.

Còn cần owner review C/B_fixed, split/grouping, policy/Fixed-High, rho/sensitivity, budget/seeds và threshold hệ thống trước thí nghiệm. Owner Thành và kiến trúc offline là quyết định đã chốt; các tham số khác vẫn là đề xuất. Khảo sát nguồn lịch sử có agent riêng tra optimization; tổng hợp cùng ngữ cảnh, không phải hội đồng độc lập hoặc cross-model review. Lượt cập nhật này đối chiếu nguồn inline, không đánh dấu nguồn như người dùng đã đọc.

## 12. Nguồn đã kiểm tra và phạm vi đọc

Đối chiếu bổ sung ngày 02/11 (Thứ 2): metadata/abstract QAOA gốc [R12], phần set cover của Lucas [R1] và [IBM QAOA tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm), mục Background và From classical optimization to quantum circuits. Dùng cho ranh giới hybrid offline và mapping QUBO/Ising/circuit; không coi tutorial là benchmark catalogue hoặc chứng cứ quantum advantage. Không mở lại toàn bộ corpus bên dưới trong lượt này; phạm vi đọc lịch sử được giữ nguyên.

Bảng tiếp theo ghi nguồn gốc, formulation/backend, baseline/metric, kết quả và giới hạn thực sự được kiểm tra. Không dùng số citation hoặc impact factor làm bằng chứng. Các suy dẫn QUBO và khuyến nghị architecture trong tài liệu là thiết kế của dự án, không phải kết luận paper ngoài.

| ID / nguồn gốc | Phạm vi thực sự đọc hoặc kiểm tra | Bài toán, backend, baseline/metric, kết quả và giới hạn chuyển giao |
|---|---|---|
| **R1 — Lucas, A. (2014). Ising formulations of many NP problems. Frontiers in Physics 2:5.** [DOI/full text](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full) — 10.3389/fphy.2014.00005 | Metadata và §5.1 Set cover trong HTML publisher | Chọn tập nhị phân, biến phụ và penalty cardinality/coverage. Nguồn lý thuyết, không QAOA/QPU benchmark hay runtime baseline. Connectivity/biến phụ gây chi phí embedding. Chuyển giao: mapping là kỹ thuật đã biết; proof/encoding cụ thể của repo phải kiểm riêng. |
| **R2 — Djidjev, H. N. (2023). Quantum Annealing with Inequality Constraints: The Set Cover Problem. Advanced Quantum Technologies 6(11):2300104.** [DOI](https://doi.org/10.1002/qute.202300104), [publisher full text](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/qute.202300104), [preprint](https://arxiv.org/abs/2302.11185) | Agent tra nguồn đọc §§1.2–1.3, 3.1–3.3, 4.1–4.3 và conclusion; tổng hợp đối chiếu publisher passages. URL Wiley không có subdomain advanced trả 403; không tính lần lỗi là đọc thành công | Weighted SCP; slack-QUBO, augmented Lagrangian và HUBO, D-Wave Advantage cùng SA. So quality/embedding; AL/HUBO tốt hơn slack trong các instance đã xét, không có strong native exact/MIP comparison để suy rộng. Slack tăng qubit, penalty normalization ảnh hưởng precision. Chuyển giao cao cho resource audit, chưa chứng minh lợi ích catalogue stitching. |
| **R3 — Ammermann, J., Brenneisen, F. J., Bittner, T., & Schaefer, I. (2024). Quantum Solution for Configuration Selection and Prioritization. Q-SE 2024.** [DOI](https://doi.org/10.1145/3643667.3648221), [author PDF tại KIT](https://publikationen.bibliothek.kit.edu/1000173454/154123677) | Abstract/metadata, formulation và execution §§4.3–4.4, evaluation/discussion §5, đặc biệt ideal-simulator và width/depth passages | Attributed feature-model constraints thành PUBO/QUBO/Ising; QAOA ideal simulator. So encoding, validity/cost quality, rank-biased overlap và circuit width/depth; prioritization kém hơn khi scale. Không phải strong classical cover benchmark. Tiền lệ configuration optimization, khác chọn catalogue dùng chung nhiều scene; không nhập claim speedup trong introduction của nguồn vào dự án. |
| **R4 — Kotil, A., et al. (2025). Quantum approximate multi-objective optimization. Nature Computational Science 5:1168–1177.** [DOI/full text](https://www.nature.com/articles/s43588-025-00873-y) — 10.1038/s43588-025-00873-y | Main, Results, parameter transfer, classical comparisons và MPS/hardware sampling; chưa đọc supplementary | MO-MAXCUT, scalarization theo trọng số, IBM ibm_fez thật và MPS; hardware p=3–6, sparse heavy-hex 42-node. Baseline DPA-a/DCM/epsilon-constraint, metric hypervolume. Có kết quả hứa hẹn trong phạm vi bài; MPS timing dùng giả định sampling QPU, hardware sampling loại compile. Không suy ra p=1,2 constrained-cover advantage; dùng làm căn cứ sensitivity của B. |
| **R5 — Koch, T., et al. (2026). The Quantum Optimization Benchmarking Library. Nature Computational Science 6:653–671.** [DOI](https://doi.org/10.1038/s43588-026-00991-1), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13293884/), [official library](https://zib-aopt.github.io/QOBLIB/) | Benchmark rationale, independent-set example/Tables 2–3; Methods metrics, solution quality, computational resources và quantum runtimes | Native optimization/QUBO benchmark framework, feasibility/objective/bounds và runtime breakdown. Ví dụ p=1 ibm_fez 52 biến không có raw feasible sample, greedy postprocessing tạo nghiệm tốt; circuit chỉ thực thi một phần constraints. Không phải proof quantum hơn exact/greedy. Chuyển giao: original-problem validator, raw/repaired control, training/compile/sampling/repair timers; không gọi sample tốt là optimal certificate. |
| **R6 — Doan, A.-D., Sasdelli, M., Suter, D., & Chin, T.-J. (2022). A Hybrid Quantum-Classical Algorithm for Robust Fitting. CVPR.** [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Doan_A_Hybrid_Quantum-Classical_Algorithm_for_Robust_Fitting_CVPR_2022_paper.html), [full text v4](https://arxiv.org/html/2201.10110v4) | Introduction, §§4.2–4.3, 6.2 và weaknesses | Hypergraph vertex-cover/QUBO subproblems cho consensus maximization. D-Wave trên synthetic nhỏ; real-data fundamental matrix/triangulation dùng SA CPU. So RANSAC/LO-RANSAC, deterministic methods; metric consensus/error bound/runtime. Không chứng minh quantum thay GPU homography nhanh hơn. Chuyển giao sang D đòi estimator/evaluator/reuse mới. |
| **R7 — Yang, F. F., Sasdelli, M., & Chin, T.-J. (2024). Robust Fitting on a Gate Quantum Computer. ECCV.** [DOI](https://doi.org/10.1007/978-3-031-73232-4_7), [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/37_ECCV_2024_paper.php), [full text v1](https://arxiv.org/html/2409.02006v1) | §§1.3–1.4, 4.2 và 6–7; IonQ/SV1 và fundamental-matrix experiment passages | Boolean influence/Bernstein–Vazirani với feasibility circuit 1D, không phải QUBO catalogue. IonQ Aria minh họa nhỏ, SV1 simulator; real fundamental-matrix method dùng classical influence. Metric influence/robust estimation; noise/qubit limits. Không chứng minh thay RANSAC homography toàn pipeline; không chọn làm primary. |
| **R8 — Farina, M., et al. (2023). Quantum Multi-Model Fitting. CVPR.** [CVF PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Farina_Quantum_Multi-Model_Fitting_CVPR_2023_paper.pdf), [full text v1](https://arxiv.org/html/2303.15444v1), [official code](https://github.com/FarinaMatteo/qmmf) | §§3, 5, iterative decomposition và implementation/runtime paragraphs; chưa audit code | Preference-matrix set-cover-like selection, QuMF/DeQuMF qua QA và SA; so RanSaCov/HQC-RF, misclassification và runtime. Decomposition xử lý subproblems, không bảo đảm optimum đầy đủ; timing có implementation/hardware khác nhau. Hữu ích cho cảnh báo decomposition, không thay global H của repo bằng nhiều model. |
| **R9 — Venkatesh, S. M., et al. (2024). Q-Seg: Quantum Annealing-Based Unsupervised Image Segmentation. IEEE Computer Graphics and Applications.** [DOI](https://doi.org/10.1109/MCG.2024.3455012), [author PDF](https://www.dfki.de/fileadmin/user_upload/import/15308_Q-Seg_Quantum_Annealing-Based_Unsupervised_Image_Segmentation.pdf) | Graph/QUBO formulation, experimental setup, runtime analysis và quality passages; author accepted manuscript, không coi placeholder DOI đầu trang là metadata cuối | Signed-edge graph segmentation/QUBO, D-Wave Advantage thật. So Gurobi QUBO, relative cut error, runtime/scalability và imagery examples. Nguồn báo nhanh hơn Gurobi ở các phép thử, Gurobi có objective tốt hơn; có cả measured remote và hypothetical onsite timing. Không chuyển thành panorama/seam superiority hoặc ưu thế mọi classical solver. |
| **R10 — Kolmogorov, V., & Zabih, R. (2004). What Energy Functions Can Be Minimized via Graph Cuts? IEEE TPAMI 26(2):147–159.** [Author page](https://pub.ista.ac.at/~vnk/papers/WHAT_ENERGIES.html), [full text](https://pub.ista.ac.at/~vnk/papers/KZ-PAMI-graph_cuts.pdf) | Abstract/metadata, binary-energy characterization và regularity/construction passages; không đọc toàn proofs | Lý thuyết classical graph-representable binary energies, không quantum backend/benchmark. Căn cứ min-cut cho energy phù hợp, không phải mọi signed QUBO đều submodular. Chuyển giao: nếu chọn E, phải kiểm tra class energy và dùng exact classical baseline khi áp dụng được. |
| **R11 — Bowles, J., Ahmed, S., & Schuld, M. (2024). Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv preprint 2403.07059v2.** [Full text](https://arxiv.org/html/2403.07059v2), [official code](https://github.com/XanaduAI/qml-benchmarks) | §§II.1, II.2.4, II.3–II.4, IV, V.1/V.3; chưa đọc mọi appendix hoặc xác minh journal version | Classical simulation PennyLane/JAX; quantum classifiers đối chiếu SVC/MLP/CNN với tuning. Accuracy/cross-validation trên bộ classification tasks; classical và non-entangling variants mạnh trong các phép thử. Không suy ra mọi QML bất khả thi. Chuyển giao F: matched tuning/features/resources, inference cost và separable ablation. |
| **R12 — Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. arXiv:1411.4028.** [Metadata](https://arxiv.org/abs/1411.4028), [paper](https://arxiv.org/pdf/1411.4028) | Agent tra nguồn đọc abstract và §§I–đầu II, equations (1)–(14); không toàn paper | QAOA uniform initial state, cost phase/X mixer và 2p angles; lý thuyết, không catalogue hardware benchmark. Kỳ vọng tối ưu có thể cải thiện khi tăng p không bảo đảm optimizer hữu hạn làm được. Chuyển giao: định nghĩa circuit p=1,2, không hứa advantage. |
| **R13 — Xu, L., Hoos, H. H., & Leyton-Brown, K. (2010). Hydra: Automatically Configuring Algorithms for Portfolio-Based Selection. AAAI 24(1):210–216.** [Proceedings/DOI](https://ojs.aaai.org/index.php/AAAI/article/view/7565) — 10.1609/aaai.v24i1.7565 | Metadata và abstract, chưa full-text review | Classical algorithm configuration/portfolio selection, SAT study; nguồn này chỉ xác nhận tiền lệ portfolio. Chưa dùng chi tiết implementation/results làm evidence cho catalogue stitching. Không claim novelty từ việc có catalogue và selector đơn thuần. |

### Tài liệu triển khai và license

| ID | Nguồn đã kiểm tra | Phạm vi và cách dùng |
|---|---|---|
| D1 | [IBM QAOAAnsatz API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.QAOAAnsatz), [Qiskit Algorithms QAOA tutorial](https://qiskit-community.github.io/qiskit-algorithms/tutorials/05_qaoa.html) | Cost/mixer/initial-state/reps API và ví dụ; tài liệu không thay compatibility test của phiên bản sẽ pin. |
| D2 | [D-Wave Penalty Models](https://docs.dwavequantum.com/en/latest/concepts/penalty.html) | Penalty làm infeasible states có energy cao hơn; không tự bảo đảm solver tìm feasible. Không dùng docs thay proof bound của encoding riêng. |
| D3 | [OR-Tools CP-SAT solver](https://developers.google.com/optimization/cp/cp_solver) | Integer constraints và return statuses; timeout FEASIBLE/UNKNOWN không là optimal certificate. |
| D4 | [Qiskit LICENSE](https://github.com/Qiskit/qiskit/blob/main/LICENSE.txt) | Apache License 2.0; version/commit license cần lưu lại khi pin. |
| D5 | [Qiskit Aer LICENSE](https://github.com/Qiskit/qiskit-aer/blob/main/LICENSE.txt) | Apache License 2.0; không giả mọi transitive package cùng license. |
| D6 | [OR-Tools LICENSE](https://github.com/google/or-tools/blob/stable/LICENSE) | Apache License 2.0; kiểm lại tại phiên bản cài thực tế. |

Các nguồn R1–R13 và D1–D6 được kiểm tra trong phiên nghiên cứu kết thúc 24/10 (Thứ 7), năm 2026; phạm vi đọc ở trên bao gồm phần được agent tra nguồn bàn giao có phân biệt rõ. Không nguồn nào là số đo quantum của dự án. Những mục chỉ abstract/metadata không dùng để bảo vệ claim phương pháp chi tiết.
