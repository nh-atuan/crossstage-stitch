# Bản thảo workshop

## Build bản thảo Phase 3

`main.tex` đọc macro, bảng và hình sinh từ export T4 đã khóa. Tạo bundle trước và đưa đường dẫn bundle vào `TEXINPUTS` theo [runbook CH-07/08](../docs/runbooks/ch07_ch08_paper_artifacts.md). Nếu thiếu input sinh tự động, LaTeX dừng ở file chưa có. Hai export thô và PDF được giữ ngoài Git.

Bản hiện tại trình bày CH-02 `legacy-v1`, A4 và CH-05. Section về phương pháp và các kết quả do owner khác bàn giao được quản lý riêng. [Claim–evidence map](../docs/claim_evidence_map.md)

## Lịch sử Phase 2

Cập nhật: 27/10 (Thứ 3). Phạm vi của Chương theo [plan](../docs/plan.md): khung OASIcs ẩn danh, Introduction, Related Work và phương pháp S6. Deadline chung Phase 2: 31/10 (Thứ 7). Đây là **working manuscript**, chưa phải bản nộp hoàn chỉnh.

Đã đối chiếu [Information for Authors của workshop](https://parma-ditam-workshop.github.io/) ngày 27/10 (Thứ 3): full paper PDF, double-blind, OASIcs, tối đa 10 trang single-column không tính trang tiêu đề và references. Mốc chính thức là 16/11 (Thứ 2), 23:59 UTC; mốc nộp nội bộ 13/11 (Thứ 6) trong plan sớm hơn và không mâu thuẫn. Trang hướng dẫn Dagstuhl không truy cập được qua công cụ web trong lần kiểm tra này; tính nguyên trạng của class/assets được xác nhận với template đã có trong repo.

## Cấu trúc và trạng thái

`main.tex` là entry point; `sections/` tách nội dung theo phần; `references.bib` chứa mười một nguồn đang được trích dẫn. `build.py` build bằng công cụ đã cài và ghi biên nhận kiểm tra vào `build/paper/` ở gốc repo. Không commit PDF, ảnh render, log, `.aux`, `.bbl` hoặc output khác.

Build dùng trực tiếp class `oasics-v2021.cls` và ba asset `cc-by.pdf`, `oasics-logo-bw.pdf`, `orcid.pdf` trong [template đã có](../PARMA-DITAM_2027_Workshop/), qua biến môi trường `TEXINPUTS`. Không nhân bản hoặc sửa class, asset hay [license đi kèm](../PARMA-DITAM_2027_Workshop/LICENSE.md) trong `paper/`. SHA-256 của class tại thời điểm kiểm tra là `3721904fe65763359493f44c604a161af875f0119fde12923a4096aa2ceb14bf`. Chế độ `anonymous` và `\hideOASIcs` dùng trực tiếp interface của class; không sửa class để ẩn danh. Nội dung paper không có tên thành viên, trường, repository hay địa chỉ liên hệ của nhóm.

Hash của các asset nguồn đang được dùng trong thư mục template:

| Asset nguồn | SHA-256 |
|---|---|
| `cc-by.pdf` | `91ab98989270efcbd884dff88f0cb67e48e4a7b8368895d761a070b2c43cf58a` |
| `oasics-logo-bw.pdf` | `3771a7329cfc7524c1610f405b9f949c9cde6f7d4cb4ccad8167196e24a1d89d` |
| `orcid.pdf` | `503580772e93ea803050d3a9db56736517ba78592010f486ba007e03e5a439cf` |

Ba PDF này là tài nguyên nguồn của publisher, không phải PDF manuscript sinh ra. Không cần stage thêm bản sao trong `paper/`. Script kiểm tra `kpsewhich` thực sự resolve class và cả ba asset tới thư mục template trong repo, đồng thời ghi hash của class, asset và license vào receipt. Cách dùng lại này giữ nguyên byte nguồn publisher và tránh phát sinh thay đổi vendor không cần thiết.

| Tệp | Nội dung hiện có | Phần còn thiếu / người phụ trách |
|---|---|---|
| `sections/abstract.tex` | Phạm vi, câu hỏi nghiên cứu, ranh giới bằng chứng | Viết lại sau khi có kết quả adaptive đã kiểm chứng |
| `sections/introduction.tex` | Workload, tương tác budget, RQ và điều kiện chứng minh đóng góp | Chương điều chỉnh theo kết quả cuối |
| `sections/related-work.tex` | Stitching/composition, blending, autotuning và quality-aware GPU execution | Chương cập nhật khi method/controller được chốt |
| `sections/pipeline.tex`, `sections/s4-s5.tex` | Interface sáu stage; phương pháp S2–S3, S4–S5 và S6 đã đối chiếu source | Tuấn: S1; Chương: review tích hợp manuscript |
| `sections/controller.tex` | Logic host đã test: policy tường minh, scene rules, feedback, escalation và retry plan | Đế hiệu chỉnh policy, nghiệm thu adaptive integration và bổ sung evidence T4 |
| `sections/experiments.tex` | Định nghĩa chất lượng và thiết kế so sánh; trạng thái bằng chứng | Đế review QoR; Tuấn cung cấp harness/T4; cả nhóm kiểm tra raw artifacts |
| `sections/discussion.tex` | Giới hạn global homography, alignment/composition và điều kiện kết luận | Cập nhật sau thí nghiệm, kể cả negative result |

Phần S1 hiện chỉ mô tả interface đã kiểm tra. Đế bổ sung phương pháp S4–S5 và Controller ngày 30/10 (Thứ 6); Section 4 phân biệt code host/Controller–S5 đã test với calibration và pipeline scene-aware còn thiếu. Bản sửa này đã build 9 trang và pass audit trên Colab. Không có bảng kết quả giả hoặc con số speedup lấy từ pilot. Quantum cần artifact và thí nghiệm theo đề xuất riêng trước khi được đưa vào phần đóng góp của paper.

## Build và tái kiểm tra

Bản ngày 30/10 (Thứ 6) bổ sung S4–S5 và Controller: `python paper/build.py --output build/paper-de-p2` PASS trên Colab/TeX Live 2023. PDF 9 trang, author `Anonymous Authors`, warnings rỗng, citation/reference resolve và mọi font embed. PDF/receipt/log ở `build/paper-de-p2/`. SHA-256 PDF: `e43cc423abb325cf4180092ba083a70928a3ae5d925cf16ae9a9d63dae0349c7`. Đây là working manuscript; chưa có adaptive campaign hoặc kết quả final.

Môi trường Colab mới cần `texlive-latex-extra`, `texlive-science`, `texlive-fonts-recommended`, `texlive-bibtex-extra`, `texlive-plain-generic`, `texlive-fonts-extra`, `cm-super` và `poppler-utils`. Thiếu `soul`/fontawesome5 hoặc font scalable gây build fail; không tắt microtype hoặc bỏ audit để vượt lỗi môi trường.

Chạy từ gốc repo; WSL kế thừa thư mục hiện tại:

```powershell
wsl --exec python3 paper/build.py
```

Hoặc trong Linux/WSL, từ gốc repo:

```bash
python3 paper/build.py
pdftoppm -png -r 85 build/paper/main.pdf build/paper/page
```

Yêu cầu có sẵn Python từ 3.9, `pdflatex`, `bibtex`, `kpsewhich`, `pdfinfo`, `pdftotext`, `pdffonts`; `pdftoppm` dùng cho kiểm tra hình thức. Script không cài package và không dùng shell escape. Môi trường đã chạy là WSL với TeX Live 2023/Debian, pdfTeX 1.40.25, BibTeX 0.99d và Poppler. Đây là kiểm chứng tài liệu, không thay thế kiểm chứng CUDA trên Colab T4.

Script thực hiện pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX; trả mã lỗi nếu compile thất bại, citation/reference chưa resolve, final log còn warning/overfull/underfull, author metadata sai hoặc font chưa embed. Nó quét một số token nhận diện đã biết trong PDF text/metadata; kiểm tra này hỗ trợ review thủ công, không chứng minh ẩn danh tuyệt đối. Chạy lỗi ghi lại receipt lỗi để không giữ trạng thái pass từ lần trước. Output phải nằm trong `build/` của repo.

Đã kiểm tra syntax script bằng `python3 -m py_compile paper/build.py`; build thật là kiểm tra tích hợp chính. Kiểm tra `--output paper` bị từ chối với exit code 2 trước khi ghi file, xác nhận output không làm bẩn thư mục nguồn. Không tạo một bộ unit test chỉ lặp lại lệnh build.

Output thành công có `status: passed`, `pages: 6`, `author: Anonymous Authors`. Các file cần đọc là `build/paper/verification.json`, `main.log`, `bibtex.txt`, `pdfinfo.txt`, `pdffonts.txt`, `paper-text.txt`, `lookup-*.txt` và `main.pdf`. Receipt lưu hash của PDF, source TeX/BibTeX/script và `template_provenance_sha256` cho class/asset/license gốc để đối chiếu đúng lần build. Số trang sáu là trạng thái draft hiện tại, không phải mục tiêu độ dài bản nộp.

Kết quả kiểm tra thực tế ngày 27/10 (Thứ 3):

| Kiểm tra | Kết quả |
|---|---|
| Build script trên WSL | Exit code 0; sáu trang A4; PDF 1.5 |
| BibTeX và final TeX log | Không có citation/reference chưa resolve, warning, overfull hoặc underfull |
| PDF metadata | Author `Anonymous Authors`; Creator `LaTeX with oasics-v2021.cls`; không có tên nhóm; không có metadata stream/XMP |
| Font | Cả 19 font đều được embed |
| PDF text | Không thấy tên thành viên, trường, GitHub cá nhân hoặc đường dẫn checkout trong nội dung manuscript |
| Render | `pdftoppm -png -r 85`; đã xem cả sáu trang, không thấy chữ/công thức/citation bị cắt hoặc đè nhau |

Kiểm tra ngày 28/10 (Thứ 4), sau khi thêm mục S2–S3, trên Windows với MiKTeX 26.2 (BibTeX 0.99e):

- `build.py` dừng ở bước `kpsewhich cc-by.pdf`. `kpsewhich` của MiKTeX tìm được `oasics-v2021.cls` qua `TEXINPUTS` nhưng không tìm file `.pdf` theo biến này. Script không bị sửa; receipt `status: passed` của script chưa có cho bản này.
- Build tay cùng trình tự pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX, cùng `TEXINPUTS` và `BIBINPUTS`, output ngoài repo: cả bốn bước exit 0. Final log không có warning, overfull, underfull hay citation chưa resolve. PDF dài 7 trang, author `Anonymous Authors`, mọi font được embed. Text PDF không chứa tên thành viên, trường hay GitHub cá nhân.
- Cần chạy lại `build.py` trên WSL hoặc Linux có TeX Live để có receipt đầy đủ.

Chưa có figure thực nghiệm; vì vậy chưa có figure-quality audit. Trước bản nộp cần kiểm tra lại hướng dẫn chính thức, độ dài sau khi các chủ sở hữu bổ sung method/results, PDF links và toàn bộ metadata. Template/build pass không đồng nghĩa paper đã đủ nội dung hoặc kết quả để nộp.

## Claim–evidence map

| Claim / nội dung | Bằng chứng sử dụng | Kết luận được phép / còn thiếu |
|---|---|---|
| RQ: giảm latency bằng phân bổ budget giữa các stage dưới QoR constraint | [plan](../docs/plan.md), [stages](../docs/stages.md) | Câu hỏi nghiên cứu; chưa là kết quả |
| S6 nhận RGB gốc, H B→A; geometry, stream và caller-owned workspace | [interface S6](../src/s6/s6.hpp), [geometry](../src/s6/geometry.cpp), [CUDA](../src/s6/s6.cu) | Mô tả implementation; không suy ra speedup |
| Feather và multi-band 3/5 có quy tắc sampling, ownership, extension và reconstruction cụ thể | [thiết kế S6](../docs/stages/s6-warp-blend.md), [CUDA](../src/s6/s6.cu), [reference độc lập](../tools/s6_reference/reference.py) | Không phải thuật toán blending mới; 3/5 là số REDUCE, tương ứng 4/6 mức nếu kích thước đủ |
| Tính đúng số học renderer | [CUDA tests](../tests/test_s6_cuda.cu), [geometry tests](../tests/test_s6_geometry.cpp), [reference validation](../tools/s6_reference/validate_cuda.py) | Correctness trong các ca đã chạy; không chứng minh chất lượng perceptual hoặc hiệu năng T4 |
| S2–S3 là ORB một level: FAST-9, NMS 3×3, chọn 2K theo FAST rồi K theo Harris, intensity centroid, rBRIEF 32 byte; chọn điểm trên device không đọc count về host | [decision ORB](../docs/decisions/s2-s3-orb.md), [triển khai](../docs/stages/s2-s3-implementation.md), [S2](../src/s2_s3/s2.cu), [S3](../src/s2_s3/s3.cu) | Mô tả implementation; không phải detector mới. Hằng số 20 và 0.04 là mặc định của OpenCV, chưa tune |
| S2–S3 khớp bản tham chiếu từng bit; bản tham chiếu được so với ORB một level của OpenCV | [bằng chứng T4](../docs/stages/s2-s3-validation.md), mục 2 và 6.3; [test tham chiếu](../tests/python/test_s2_s3_reference.py) | Correctness trên golden đã chạy; không chứng minh chất lượng ghép |
| Workspace không giảm theo W, H, K nên cấp một lần và đổi K mỗi attempt không cấp phát lại; caller truyền view `capacity = K`; output giữa các K không có tính prefix | [decision K động](../docs/decisions/s2-s3-dynamic-k.md), [test host](../tests/test_s2_s3_config.cpp), [bằng chứng T4](../docs/stages/s2-s3-validation.md), mục 6 | Câu về view `capacity = K` chỉ nói tránh warp thừa, không đưa số đo vào paper |
| ORB một level không ghép được cặp có thay đổi scale lớn | [bằng chứng T4](../docs/stages/s2-s3-validation.md), mục 3: ba cặp Oxford, `cv2.ORB` một level cũng sai | Suy luận từ ba cặp; chưa kiểm tra trên UDIS-D |
| Alignment QoR dùng cùng domain tham chiếu và homography, độc lập blender | [QoR evaluator](../tools/qor/evaluate.py), [decision](../tools/qor/decision.py), [benchmark decisions](../docs/decisions/benchmark-qor-s4-s5.md) | PSNR/SSIM alignment không chứng minh seam/ghosting tốt hơn; Đế phải review policy và ngưỡng |
| Runtime adaptive tốt hơn Fixed-Best | Chưa có campaign hoàn chỉnh với cùng predicate và timing boundary | Không được khẳng định; cần raw measurements, policy hash, split, traces và T4 |
| Oracle gap từ pilot là tiềm năng đã xác nhận | Chỉ có summary; raw sweep/manifest chưa đầy đủ để audit predicate | Không đưa số pilot vào manuscript; cần tính lại cùng quality predicate |
| Multi-band sâu hơn cải thiện QoR chính | Alignment metric không dùng blended output | Không được khẳng định; cần composition evaluator riêng nếu nghiên cứu chất lượng blend |

Các test CUDA và same-H chạy trên GTX 1650 là evidence phát triển, không phải T4. Timing CUDA-event và host clock quan sát trong WSL còn chênh lệch; paper không dùng các số này làm kết quả hiệu năng. Tất cả bảng/đồ thị định lượng trong bản tiếp theo phải được sinh từ code và artifact của nhóm, không chép tay số từ ghi chú.

## Reverse outline và tự review

Introduction đi từ workload → tương tác budget → cấu hình/runtime → RQ và nghĩa vụ bằng chứng → vị trí đóng góp. Related Work chia theo vấn đề mà mỗi nhóm nghiên cứu giải quyết, không liệt kê paper rời rạc. Method đi từ interface → geometry/validity → feather → pyramid/reconstruction/resource contract. Evaluation tách chất lượng alignment, correctness renderer, cost blender và hiệu quả adaptive.

| Chiều review | Tự đánh giá hiện tại | Việc cần làm tiếp |
|---|---|---|
| Đóng góp rõ ràng | RQ và contribution dự kiến đã nêu; chưa claim novelty hoặc speedup | Viết lại contribution theo ablation thực tế |
| Chứng cứ cho claim | S6 dựa trên source; claim thực nghiệm chưa đủ đều được đánh dấu pending | Bổ sung artifact/traces và kết quả T4 đã audit |
| Method có thể tái lập | Công thức, toạ độ, mask, biên, numeric và resource S6 đã mô tả | Chủ sở hữu viết S1–S5 và controller, kiểm tra interface khi tích hợp |
| Thiết kế thực nghiệm | Fixed-Best/Oracle/Adaptive, failures và timing boundary tách rõ | Chốt split/policy; chạy campaign; ablation reuse/selection/escalation |
| Văn phong và trình bày | Đã rà paragraph roles, tránh claim không có số liệu; PDF sáu trang đọc được | Rà lại toàn bài khi ghép các phần, bỏ trạng thái editorial khỏi bản nộp |

Đã đọc [Wikipedia:Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) theo quy định repo. Sau viết đã rà các đoạn mở đầu chung chung, câu nhấn mạnh không có chứng cứ, lặp kết luận và từ ngữ quảng bá. Trạng thái pending được ghi cụ thể theo bằng chứng thiếu, không dùng văn bản chung để lấp Section 4.

## Tài liệu gốc đã đối chiếu

Ngày truy cập: 27/10 (Thứ 3). Phạm vi đọc dưới đây chỉ phục vụ các claim hẹp trong draft, không đồng nghĩa đã tái lập thí nghiệm của các tác giả.

| Nguồn | URL / DOI | Phạm vi thực sự đọc và sử dụng |
|---|---|---|
| Brown và Lowe, 2007 | [PDF của tác giả](https://www.cs.ubc.ca/~lowe/papers/07brown.pdf), DOI `10.1007/s11263-006-0002-3` | Abstract và phần pipeline/stitching/multi-band làm bối cảnh; đối chiếu thêm caption Figure 1 qua nội dung index từ PDF gốc về matching → RANSAC → inlier verification. Không suy ra repo dùng SIFT; không lấy tốc độ làm baseline GPU |
| Burt và Adelson, 1983 | [PDF MIT](https://persci.mit.edu/pub_pdfs/spline83.pdf), DOI `10.1145/245.247` | Header, abstract/introduction qua nội dung được index từ PDF gốc; mở PDF trực tiếp timeout. Chỉ dùng nguyên lý ghép band-pass rồi reconstruction, không tuyên bố đã đọc toàn bộ chứng minh |
| Zhu và cộng sự, 2018 | [IEEE PDF](https://ieeexplore.ieee.org/ielx7/83/8310970/08300640.pdf), DOI `10.1109/TIP.2018.2808766`; [code chính thức](https://github.com/ajex1988/blending) | Abstract/introduction từ nguồn IEEE và mô tả repo: so sánh time, memory, visual quality của blending; không chuyển số liệu video sang T4 hai ảnh |
| Nie và cộng sự, 2023, UDIS++ | [arXiv của tác giả](https://arxiv.org/abs/2302.08207), [HTML](https://arxiv.org/html/2302.08207v2) | Abstract/introduction và mô tả global homography, local TPS, seam-driven composition; CVF HTML bị từ chối truy cập nên dùng bản tác giả |
| Ansel và cộng sự, 2014, OpenTuner | [PDF MIT](https://commit.csail.mit.edu/papers/2014/ansel-pact14-opentuner.pdf), DOI `10.1145/2628071.2628092` | Abstract/introduction: representation và nhiều search techniques; đối chiếu bối cảnh autotuning |
| Samadi và cộng sự, 2013, SAGE | [PDF của nhóm tác giả](https://cccp.eecs.umich.edu/papers/samadi-micro13.pdf) | Abstract/introduction: compiler tạo CUDA variants, runtime chọn dưới quality target; không trích speedup, không tự thêm DOI chưa kiểm tra |

Năm nguồn của phần S2–S3, thêm ngày 28/10 (Thứ 4). Metadata và DOI đối chiếu qua bản ghi Crossref (`api.crossref.org/works/<DOI>`) cùng ngày. Các nguồn chỉ được trích làm nguồn gốc của từng bước thuật toán. Chi tiết thuật toán được đối chiếu với mã nguồn OpenCV 5.0.0 trong [decision ORB](../docs/decisions/s2-s3-orb.md), mục Nguồn; lần này không đọc lại toàn văn các bài.

| Nguồn | DOI | Dùng cho |
|---|---|---|
| Rublee và cộng sự, 2011, ORB | `10.1109/ICCV.2011.6126544` | Tổ hợp FAST, Harris ranking, orientation và rBRIEF |
| Rosten và Drummond, 2006 | `10.1007/11744023_34` | FAST corner score |
| Harris và Stephens, 1988 | `10.5244/C.2.23` | Harris response; Crossref ghi trang 23.1–23.6 theo đánh số của BMVA |
| Calonder và cộng sự, 2010, BRIEF | `10.1007/978-3-642-15561-1_56` | Descriptor nhị phân so bằng Hamming |
| Rosin, 1999 | `10.1006/cviu.1998.0719` | Orientation theo intensity centroid |

Không lấy nội dung blog để chứng minh method; không upload manuscript riêng lên dịch vụ ngoài. Giữ số liệu tham khảo ngoài paper tách khỏi số đo của nhóm.
