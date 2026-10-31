# Khung paper PARMA-DITAM 2027

Cập nhật: 18/10 (Chủ nhật), năm 2026. Đây là khung nội dung Phase 0, chưa phải full paper hoàn chỉnh hoặc kết quả của Phase 1–2.

## File làm việc

- `oasics-v2021-sample-article.tex`: bản outline tiếng Anh đã thay nội dung minh họa, bật `anonymous`.
- `crossstage-stitch.bib`: ba nguồn dùng trong outline; style BibTeX bắt buộc là `plainurl`.
- `oasics-v2021.cls` và các tài nguyên publisher: giữ nguyên. PDF tên `oasics-v2021-sample-article.pdf` có sẵn trong gói vẫn là ví dụ của publisher, không phải PDF của outline đã sửa.

Theo yêu cầu hiện tại, chỉnh trực tiếp trong thư mục này. [Kế hoạch Nhóm](../docs/plan.md) quy định vị trí paper lâu dài là `paper/`; chưa tự di chuyển/đổi tên thư mục, chưa thay quy ước chung. Trước khi ghép paper cần thống nhất vị trí này với Tuấn.

## 1. Kết quả kiểm tra template

Template được chọn đúng: publisher hiện liệt kê `oasics-v2021 v2021.1.3`. Class local ghi `[2023/05/11 v3.1.3 OASIcs articles]`; đây là cách đánh số nội bộ khác tên gói, không phải hai phiên bản mâu thuẫn. Đã tải archive chính thức và so SHA-256 của class:

`3721904FE65763359493F44C604A161AF875F0119FDE12923A4096AA2CEB14BF`

Hash local và class trong archive khớp nhau. Việc repo styles có release mới hơn cho các series khác không chứng minh OASIcs cần đổi class.

Nguồn đối chiếu, truy cập 18/10 (Chủ nhật): [workshop](https://parma-ditam-workshop.github.io/), [hướng dẫn camera-ready](https://parma-ditam-workshop.github.io/Submission_Guidelines.html), [trang OASIcs của publisher, Templates and Example Files](https://drops.dagstuhl.de/entities/series/OASIcs#author), [archive chính thức v2021.1.3](https://submission.dagstuhl.de/styles/download-tag/oasics/v2021.1.3/authors/zip). Đường documentation/authors mở thất bại ở công cụ web; dùng trang publisher và hướng dẫn PDF có sẵn để đối chiếu, không coi lỗi truy cập là bằng chứng template sai.

### Review và camera-ready là hai giai đoạn

Review: full paper PDF qua EasyChair, double-blind, tối đa 10 trang nội dung single-column. Trang hướng dẫn chi tiết tính figures/tables/algorithms/footnotes/appendices vào running text; liệt kê title page, abstract, acknowledgements và references ngoài giới hạn này. Không tự nhận 12 trang là hạn review. Dùng budget bảo thủ, không chèn trang trắng hoặc ép ngắt trang để tận dụng cách đếm.

Camera-ready: trang workshop hiện ghi 12 trang running text, tối đa 15 trang tổng, deadline 07/01/2027, và package source/PDF sau acceptance. Không áp dụng các yêu cầu điền tên tác giả thật của camera-ready cho bản review. Chưa upload hoặc nộp tài liệu.

Trang camera-ready cũng chứa chỉ dẫn kiểu “left hand column” và “no headers/page numbers” không khớp hoàn toàn với class single-column. Giữ class publisher nguyên vẹn trong bản review, chưa vá `.cls` hoặc ép pagestyle theo những câu này; trước camera-ready cần xác nhận với ban tổ chức nếu hướng dẫn vẫn còn khác nhau. Không suy đoán tên proceedings editors hoặc mã xuất bản từ danh sách chairs.

### Sửa các khẳng định trong bản gợi ý được gửi

| Nội dung gợi ý | Điều chỉnh |
|---|---|
| `anonymous` che toàn bộ mọi dấu vết | Class xử lý author-related metadata chuẩn, không tự che tên ở body, ảnh, URL, file đính kèm hoặc metadata của hình. Bản review không chứa tên thật/email/ORCID trong source đang dùng; phải kiểm tra cả PDF/XMP |
| Tự điền editors, volume=1, ArticleNo=1 | Không có căn cứ cho các giá trị đó. Bỏ metadata giả của sample; dùng lệnh class `hideOASIcs` cho working review draft để không in DOI/logo/khối xuất bản chưa được gán |
| Phải thêm multirow/algorithm2e | `multirow`, `tabularx`, `amsmath`, `graphicx`, `subcaption` đã được class nạp. Outline hiện không cần thêm package; chỉ thêm khi có nội dung cần dùng |
| Bật mọi option cleveref/autoref/thm-restate | Giữ cleveref/autoref để tham chiếu; không dùng thm-restate khi chưa có nhu cầu restatable theorems |
| Dùng `newpage` sau abstract để tăng diện tích | Không dùng mẹo này. Giữ dòng chảy tự nhiên của class và kiểm tra số trang khi bài đủ nội dung |
| Introduction phải có speedup/FPS nổi bật | Chỉ viết sau khi có kết quả. Hiện dùng câu hỏi, intended contribution và evidence plan; không điền số giả |
| Memory hierarchy, shared memory, streams là các đóng góp sẵn có | Chỉ mô tả những gì đã cài và chứng minh bằng profiling/ablation; đóng góp chính dự kiến là quyết định budget giữa các stage |
| Tổng phân bổ các mục của bản gợi ý | Các mức cao cộng tới 11,5 trang, vượt review budget; dùng budget 9,5 trang dưới đây, chừa 0,5 trang dự phòng |

Abstract đặt sau `maketitle`; giữ numbered citations/BibTeX `plainurl`, sectioning chuẩn, font/lề/spacing của class. Bản review giữ line numbers mặc định. Không dùng `apalike`, `natbib`, `geometry`, `times`, `enumitem`, `paralist`, `wrapfig`; không scale bảng đến mức chữ khó đọc. Việc class biên dịch được không tự chứng nhận bài đủ điều kiện nộp.

## 2. Mini-outline và budget dự kiến

- **Introduction — 1 trang:** bài toán, giả thuyết fixed budget, research question, intended contributions; cần số liệu chứng minh headroom.
- **Background and Problem Formulation — 1,25 trang:** sáu stage, c=(s,k,r,b), quality policy nhiều metric, feasibility, Fixed-Best và Oracle.
- **Related Work — 0,75 trang:** stitching/alignment/composition, GPU blending trade-off, adaptive runtime/approximate computing; chưa chốt novelty trước khi bổ sung nhánh runtime literature.
- **Cross-Stage Runtime Design — 2,75 trang:** characterization và ladder, initial selection/check/escalation, reuse/dependency/buffer lifecycle; S6 như một policy có ranh giới rõ.
- **Experimental Evaluation — 3 trang:** setup/protocol, fixed/adaptive/oracle, stage costs/overhead/ablation, sensitivity/coverage/failure; có main table và Pareto/breakdown plots khi có số liệu.
- **Discussion and Conclusion — 0,75 trang:** điều kiện adaptation có ích, hạn chế QoR/global H/T4, trả lời RQ bằng dữ liệu; fallback Pareto nếu giả thuyết không đứng vững.

Tổng dự kiến 9,5 trang running text, chừa 0,5 trang để chỉnh layout. Đây là ngân sách cho full paper sau này, không phải số trang outline hiện tại. Không có mục “Scalability” hàm ý đã chạy multi-GPU; chỉ resolution/workload scaling trên một GPU nếu thực hiện.

## 3. Story, vai trò đoạn và bằng chứng

Story: chi phí căn chỉnh cần thiết có thể thay đổi theo cảnh; runtime chọn budget và reuse có điều kiện; kiểm tra lợi ích sau khi trừ toàn bộ overhead và xét chất lượng từng pair. Đây là giả thuyết cần đo, không phải kết quả mặc định.

Mỗi đoạn trong `.tex` có comment ngắn `Role:`: opening/background, challenge, design/method, evidence plan hoặc limitation. Reverse outline đã rà theo mạch: bài toán → quality constraint → prior work → runtime → phép thử → phạm vi kết luận. Không dựng riêng sáu mục mô tả stage dài làm lu mờ đóng góp runtime.

| Claim | Evidence | Status |
|---|---|---|
| Feature-based stitching là bối cảnh pipeline, không là thuật toán mới | Brown–Lowe và mô tả dự án | supported |
| Prior blending work xét time/memory/quality | Zhu et al., TIP 2018 | supported |
| UDIS++ thay cả alignment/composition | CVF abstract/metadata | supported |
| Adaptive budgets giảm latency so Fixed-Best ở cùng policy | Cần held-out end-to-end measurements, coverage/violation rate | needs evidence |
| Scene-aware initial selection có ích | Cần so policy khởi đầu/ablation và overhead | needs evidence |
| Reuse tạo lợi ích thực tế | Cần bật/tắt reuse, cùng output và tính cả materialization/invalidation | needs evidence |
| B có thể được chọn dưới composition constraint | Overlap metrics không đủ; cần output-sensitive criterion và same-H B sweep | needs evidence |
| Chất lượng thị giác được bảo đảm hoặc đạt realtime/energy saving | Chưa có phép đo/chứng minh tương ứng; không viết như claim | needs evidence |

Bám [quyết định benchmark/QoR](../docs/decisions/benchmark-qor-s4-s5.md): UDIS-D là benchmark chính; HPatches dùng phát triển/geometry, synthetic high-resolution là stress test riêng. Main S/K/R giữ B cố định trước khi có composition policy. Hai tolerance 0,02 SSIM và 0,5 dB trong tài liệu Đế vẫn là pilot proposals, không tự nâng thành ngưỡng chuẩn đã khóa hoặc ngưỡng của dataset.

## 4. Tự phản biện năm chiều

| Chiều | Câu hỏi và trả lời ở trạng thái outline | Trạng thái |
|---|---|---|
| Contribution | Đóng góp là gì? Dự kiến là quality-constrained budget selection và dependency-aware reuse; chưa được phép gọi first/new hơn mọi prior runtime | needs revision: bổ sung runtime literature |
| Writing clarity | Có mạch và thuật ngữ ổn định không? Có sáu mục, c/QoR/reference/Fixed-Best/Oracle phân biệt; body và abstract ghi đúng trạng thái dự kiến | pass ở mức outline |
| Experimental strength | Đã có lợi ích đáng kể chưa? Chưa. Không đưa số speedup, occupancy hoặc energy tưởng tượng | needs new experiment |
| Evaluation completeness | Có baseline/ablation/failure không? Đã có kế hoạch; thiếu protocol chi tiết, final split, frozen thresholds và dữ liệu | needs new experiment |
| Method soundness | Proxy có bảo đảm QoR không? Không. Phải đo false acceptance, retry cost, infeasible/reference failure và cache correctness; B có gate riêng | needs revision và new experiment |

Rà ngắn: clarity/flow/terminology đạt ở mức khung; unsupported claims được hạ thành hypotheses/planned work; thiếu evidence được nêu trong bảng. Khi có kết quả, viết lại Abstract/Introduction/Conclusion thay vì chỉ chèn số vào các câu hứa trước.

## 5. Biên dịch và kiểm tra

Entry point là `oasics-v2021-sample-article.tex`. Dùng pdflatex → BibTeX → pdflatex → pdflatex; chạy từ thư mục này để tìm class và assets. `plainurl.bst` phải có trong TeX distribution; không đổi sang style khác chỉ để build qua.

Kiểm tra log, citations/cross-references, PDF metadata/text và render từng trang. File `.cls` phải giữ đúng hash nêu trên. Trạng thái kiểm chứng thực tế được ghi trong bàn giao; chưa có kiểm chứng CUDA/T4 vì task này chỉ dựng LaTeX. Sau khi có full paper còn phải kiểm tra giới hạn trang, ẩn danh của hình/supplement và yêu cầu EasyChair trước nộp.

### Kết quả kiểm tra bản outline

- Đã build bằng TeX Live 2023/Debian có sẵn trong WSL Ubuntu 24.04: pdflatex, BibTeX `plainurl`, hai lượt pdflatex tiếp theo. Không cài thêm package và không sửa class để vượt lỗi môi trường.
- MiKTeX Windows thiếu `microtype.sty`; compiler tích hợp bị timeout. Đây là lỗi môi trường build tương ứng, không phải căn cứ đổi template.
- Preview có 4 trang A4; đã render và xem cả bốn trang. Không có bằng chứng về giới hạn trang của full paper tương lai từ độ dài outline này.
- PDF Info ghi author ẩn danh; không có XMP metadata stream trong build này. Toàn bộ font đã embed dạng Type 1. Class vẫn đúng SHA-256 của archive chính thức.
- Thanh màu vàng ở chân trang lẻ là phần class v2021 tạo sẵn, kể cả với `hideOASIcs`; không vá class để loại bỏ trang trí này.
- Nguồn preview và các file build được lưu ngoài repo. Không dùng PDF ví dụ đi kèm template để nộp outline mới.
- Lượt build cuối không còn Warning, undefined references, Overfull hoặc Underfull box diagnostics; bibliography đã rút gọn URL tới bản tác giả để tránh dòng quá thưa.
