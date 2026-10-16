# Quyết định Phase 0: benchmark, QoR, S4 Matching và S5 Homography

- **Task:** Khảo sát và chốt benchmark (mục 9 của `plan.md`), định nghĩa QoR và `Qmin`; khảo sát và chốt phương pháp chủ chốt cho S4 Matching, S5 Homography.
- **Người phụ trách và quyết định:** Đế.
- **Ngày ghi nhận:** 15/10 (Thứ 5), năm 2026.
- **Phase:** Phase 0; deadline chung 15/10 (Thứ 5).
- **Trạng thái:** Đã chọn benchmark và phương pháp chủ chốt. Các ngưỡng QoR là đề xuất để kiểm chứng trên development data, chưa có kết quả đo trên GPU Tesla T4.

Tài liệu chỉ ghi quyết định thuộc hai task trên. Tham chiếu [kế hoạch](plan.md) và [các stage của pipeline](stages.md). Interface, định dạng CSV và những quy ước dùng chung do Tuấn quyết định; phương pháp S2–S3 và S6 do Thành và Chương phụ trách.

## 1. BENCH-01 — Benchmark cho panorama stitching

### Bối cảnh và phương án chọn

**Chọn UDIS-D làm benchmark chính** để đánh giá toàn pipeline S1–S6. Không giữ CA-UDHN trong bộ thí nghiệm chính vì dataset này được xây dựng cho homography estimation trên các cặp ảnh chủ yếu có small baseline, không có protocol đo chất lượng panorama hoàn chỉnh [1], [2].

| Bộ dữ liệu | Vai trò | Quyết định |
| --- | --- | --- |
| **UDIS-D** (TIP 2021) | Đánh giá panorama stitching end-to-end: latency, overlap alignment quality, tỷ lệ thất bại và hiện tượng seam/ghosting | **Benchmark chính** |
| **CA-UDHN** (ECCV 2020) | Đối chiếu geometric error của chuỗi S1–S5 bằng annotated correspondences khi cần debug hoặc làm thí nghiệm phụ | Không bắt buộc trong paper |
| **HPatches** (CVPR 2017) | Kiểm tra homography/matching hoặc dùng mock/reference data theo kế hoạch của Tuấn | Chỉ dùng trong phát triển nếu cần |
| **Synthetic high-resolution pairs** | Kiểm tra GPU latency và resolution scaling ở độ phân giải lớn | Stress test tùy chọn, không phải benchmark chất lượng chính |
| **StitchBench** (AAAI 2025) | Kiểm tra khả năng tổng quát trên bộ stitching độc lập | Chỉ bổ sung nếu còn thời gian |

**UDIS-D** có 10.440 training pairs và 1.106 testing pairs theo repository và mô tả dataset [3]. UDIS++ (ICCV 2023) tiếp tục sử dụng UDIS-D cho đánh giá warping/alignment và composition, với các ví dụ có parallax hoặc low-texture [4], [5]. Dataset không cung cấp ground-truth homography và panorama hoàn chỉnh tương đương những benchmark tổng hợp có nhãn; vì vậy không thể dùng một lỗi hình học có ground truth làm `Qmin` chính.

**Protocol sử dụng:**

1. Ưu tiên tập test công khai của UDIS-D cho đánh giá cuối. Chọn development data tách biệt để quyết định quality tolerances, tham số S4/S5 và configuration ladder; không tune bằng final test.
2. Chạy cùng các image pairs và cấu hình môi trường trên pipeline CUDA của Nhóm. Báo cáo số pairs chạy được, lỗi và failure rate; không loại bỏ các trường hợp thất bại một cách âm thầm.
3. Dùng phân nhóm Easy/Moderate/Hard nếu có nhãn hoặc quy tắc chia được tái lập đúng theo nguồn [4]; không tự gán nhãn độ khó rồi gọi đó là nhãn của dataset.
4. Nếu cần kiểm tra scalability, tạo một tập synthetic nhỏ từ ảnh gốc độ phân giải cao, ghi seed, source image, phép biến đổi và kích thước. Kết quả này báo cáo riêng, không trộn vào UDIS-D.

### Trade-off và fallback

- **Ưu điểm:** Một bộ dữ liệu thống nhất cho kiểm tra S1–S6, QoR và end-to-end latency; giảm công chuẩn hóa dữ liệu và số bảng kết quả trong workshop paper.
- **Hạn chế:** Không có ground-truth panorama, nên overlap PSNR/SSIM chỉ là *proxy* cho chất lượng alignment; chúng không xác nhận toàn bộ tính tự nhiên của panorama hoặc độ chính xác hình học tuyệt đối.
- **Fallback:** Nếu chưa xây dựng được protocol đánh giá QoR đáng tin cậy trên UDIS-D, không tự chuyển CA-UDHN thành bằng chứng cho chất lượng panorama. Báo rõ giới hạn và trao đổi với Nhóm trước khi đổi benchmark hoặc thu hẹp tuyên bố nghiên cứu.

## 2. QOR-01 — Quality-of-Result và `Qmin`

### 2.1. Metric được sử dụng

QoR là chất lượng kết quả của ứng dụng, không phải tên một metric. Với UDIS-D, chọn:

| Metric | Phạm vi đánh giá | Vai trò |
| --- | --- | --- |
| **Overlap SSIM** (cao hơn tốt hơn) | Mức độ tương đồng cấu trúc của hai ảnh sau khi căn chỉnh trong vùng chồng lấn | Metric QoR chính cho alignment |
| **Overlap PSNR**, dB (cao hơn tốt hơn) | Khác biệt cường độ pixel trong vùng chồng lấn | Metric QoR bổ trợ và ràng buộc thứ hai |
| **Valid stitch / failure rate** | Homography, vùng ghép và panorama có hợp lệ không | Điều kiện bắt buộc, báo cáo trên toàn bộ dữ liệu |
| **Seam/ghosting inspection** | Artifact sau blending ở panorama cuối | Kiểm tra composition bổ sung, chưa có ngưỡng tự động đã được kiểm chứng |

UDIS++ dùng overlap PSNR/SSIM để đánh giá warping/alignment [4]. Không được xem các metric này là thước đo đầy đủ cho seam quality hoặc geometric correctness trong cảnh có parallax.

### 2.2. Định nghĩa ràng buộc trên từng image pair

Với image pair `i`, gọi `c_ref` là **Fixed-High**, tức một cấu hình cố định có budget cao, được xác định và kiểm tra trên development data. Chạy `c_ref` trên cùng pair để nhận `SSIM_ref,i` và `PSNR_ref,i`. Cấu hình này làm **quality reference**, không phải Fixed-Best dùng để đo speedup.

Với một cấu hình `c`, điều kiện QoR *đề xuất để pilot* là:

\[
\operatorname{Pass}_i(c)=\operatorname{ValidStitch}_i(c)
\land [\operatorname{SSIM}_i(c)\ge \operatorname{SSIM}_{ref,i}-0.02]
\land [\operatorname{PSNR}_i(c)\ge \operatorname{PSNR}_{ref,i}-0.5\ \mathrm{dB}]
\land \operatorname{CoverageOK}_i(c).
\]

Trong đó `CoverageOK` kiểm tra cấu hình không đạt điểm số cao chỉ bằng cách làm mất những vùng overlap khó căn chỉnh. Cách tính overlap mask và coverage tolerance cần thống nhất với Chương và Tuấn trước khi chạy pilot.

Có thể gọi cặp tolerance `Qmin = (ΔSSIM_max = 0.02, ΔPSNR_max = 0.5 dB)` trong tài liệu thiết kế; về mặt toán học, đây là **ràng buộc QoR nhiều metric**, không phải một quality score vô hướng chuẩn hóa. Nếu code cần một biến `Qmin` duy nhất, nên lưu một *quality policy* chứa hai tolerance và các điều kiện validity/coverage; không tự cộng SSIM với PSNR.

**Phân biệt rõ nguồn gốc ngưỡng:** Hai giá trị `0.02 SSIM` và `0.5 dB PSNR` là *working tolerances* do Nhóm đề xuất, **không phải ngưỡng được UDIS-D hay UDIS++ quy định**. Chúng chỉ được khóa sau khi kiểm tra trên development pairs về quality distribution, failure rate và hình ảnh trực quan; không thay đổi sau khi nhìn kết quả final test chỉ để cải thiện speedup.

**Lưu ý về chất lượng của tham chiếu:** Fixed-High không tự động là panorama đúng. Pair mà reference không tạo được kết quả hợp lệ, hoặc reference bị artifact nghiêm trọng, phải ghi `reference_failure`/`reference_unreliable` và báo cáo riêng; không cho kết quả đó mặc nhiên vượt `Qmin` nhờ so tương đối. Nếu reference không đáng tin trên nhiều pairs, cần kiểm tra pipeline và thảo luận lại quality policy.

### 2.3. Protocol đo và khả năng dựng Oracle

- Chuẩn hóa color space, dynamic range, độ phân giải và image coordinate frame trước khi tính PSNR/SSIM. Giữ một **evaluation domain cố định theo pair**, xây dựng độc lập với cấu hình candidate; không để mỗi cấu hình tự thu nhỏ overlap mask nhằm tăng metric. Báo cáo valid-overlap coverage và quy tắc xử lý vùng thiếu pixel.
- Tính PSNR/SSIM cho **từng pair và từng cấu hình**. Điều kiện `Pass_i(c)` ở trên dùng để tìm cấu hình rẻ nhất đạt QoR cho pair `i` trong một configuration space hữu hạn (*offline Oracle*).
- Ngoài pair-wise pass rate, báo cáo trung bình và phân phối chênh lệch `SSIM(c) - SSIM_ref` và `PSNR(c) - PSNR_ref`, cùng failure rate và các panorama có artifact. Trung bình đạt ngưỡng **không** đồng nghĩa mọi pair đều đạt.
- Trường hợp không có cấu hình nào đạt được quality policy, ghi `unachievable` trong configuration space. Báo cáo Oracle coverage, không bỏ những pairs này khỏi thống kê tổng.
- Để đánh giá blending knob `B`, overlap PSNR/SSIM là chưa đủ. **Đề xuất giữ `B` cố định trong thí nghiệm S/K/R chính** cho tới khi Chương và Nhóm chốt được tiêu chí composition quality; đây là đề xuất phối hợp, không phải quyết định thay S6.

### 2.4. Offline evaluation và online controller

- **Offline evaluator:** Được sử dụng ảnh đầu vào và panorama/warps của các cấu hình để tính metric, `Pass_i`, Oracle và false-accept rate; không có ground-truth panorama để chứng nhận tuyệt đối.
- **Online controller:** Chỉ dùng tín hiệu sẵn có lúc chạy từ S4/S5 như `num_matches`, `num_inliers`, `inlier_ratio`, reprojection residual, inlier spatial coverage và homography validity. Không được đọc trước quality scores của mọi cấu hình hay kết quả từ offline Oracle.
- Các tín hiệu self-check tốt không bảo đảm PSNR/SSIM tốt. Cần đo tỷ lệ controller chấp nhận cấu hình mà offline evaluator đánh dấu fail.

### Trade-off và fallback

Ưu điểm của quality non-inferiority là nó áp dụng trực tiếp cho end-to-end stitching trên UDIS-D và tạo điều kiện tính Oracle cho từng pair. Hạn chế là ngưỡng phụ thuộc chất lượng `c_ref`, quy tắc overlap mask và tolerance do Nhóm đặt. Nếu pilot cho thấy PSNR/SSIM không phân biệt được các panorama lỗi, phải bổ sung kiểm tra composition/geometry hoặc thu hẹp quality claim; không tự nhận đã bảo đảm chất lượng panorama chỉ bằng overlap metrics.

## 3. S4-01 — CUDA feature matching

### Phương pháp được chọn

**Tiled exact brute-force kNN (top-2) trên GPU + Lowe ratio test**. Mutual nearest-neighbor check (*cross-check*) là phương án ablation, chỉ bật mặc định nếu kiểm thử cho thấy có lợi cho homography/stitching success rate [6], [7].

Các bước chính:

1. Nhận keypoints/descriptors của A và B từ S2–S3; không tự quyết descriptor thay Thành.
2. Tính descriptor distances theo tile trên GPU, duy trì nearest và second-nearest neighbor cho từng query; tránh lưu ma trận khoảng cách `K_A × K_B` nếu không cần thiết.
3. Dùng ratio test để loại correspondence mơ hồ; ratio threshold chưa khóa và phải chọn trên development data sau khi biết descriptor thực tế.
4. Xuất `(index_A, index_B, distance)` cùng các thông tin cần thiết theo interface do Tuấn chốt.

Distance metric phụ thuộc S3: Hamming cho binary descriptor thông thường, L2/squared L2 cho float descriptor. Trường hợp ORB `WTA_K = 3` hoặc `4` phải dùng quy ước Hamming tương ứng [7]. Nếu so squared L2 distances thì cần bình phương ratio threshold tương ứng.

### Lý do, trade-off và fallback

Exact nearest-neighbor search dễ kiểm tra tính đúng đắn, nhiều distance calculations có thể chạy song song trên GPU, không phải dựng ANN index cho từng cặp ảnh. Khi K thay đổi, số phép so sánh thay đổi rõ ràng, thuận tiện đo latency. Bù lại, complexity là `O(K_A K_B D)` và nearest descriptor không nhất thiết là correspondence đúng; BF **không phải SOTA về matching quality**. LightGlue có adaptive learned matching, dùng làm related-work reference chứ không nằm trong phạm vi tự triển khai S4 [8].

**Fallback:** So BF only, BF + ratio và BF + ratio + cross-check trên cùng input để chọn biến thể có stitching success/QoR tốt. OpenCV BFMatcher chỉ dùng làm correctness baseline hoặc mock implementation, không phải CUDA kernel chính.

## 4. S5-01 — CUDA homography estimation

### Phương pháp được chọn

**Batched GPU RANSAC với MSAC-style truncated residual scoring và inlier-based local refinement**. Thiết kế dựa trên robust hypothesis fitting và local optimization [9], [10]; không tuyên bố đã triển khai đầy đủ MAGSAC++.

1. Xác minh có đủ ít nhất bốn correspondences và loại các sample suy biến, trùng điểm hoặc gần thẳng hàng.
2. Sinh các minimal 4-point samples với seed cố định; ước lượng homography theo chiều **B → A**, chuẩn hóa tọa độ để giảm mất ổn định số học.
3. Chấm điểm song song các hypotheses theo batches bằng reprojection residual và truncated MSAC-style loss `sum(min(residual², threshold²))`.
4. Chọn best hypothesis theo quy tắc tie-break xác định, refine bằng inliers và kiểm tra lại validity.
5. Trả `H`, inlier mask/count/ratio, residual statistics và inlier spatial coverage để controller sử dụng.

`R` là maximum RANSAC hypothesis budget. R values, batch size, residual threshold và số refinement steps vẫn cần pilot trên descriptor/matches thực tế; chưa tự khóa các tham số này trong Phase 0.

Khi chỉ tăng R, có thể giữ matches, best hypothesis và trạng thái sample sequence để đánh giá thêm các hypotheses chưa chạy. Khi thay S hoặc K, correspondences có thể thay đổi, nên không được mặc định reuse RANSAC state. Nếu S1 resize ảnh thì phải đổi homography từ working coordinates về original coordinates trước khi gửi S6.

### Trade-off và fallback

- **Ưu điểm:** Dễ song song hóa hypotheses và chủ động giới hạn computation bằng R; local refinement cải thiện model từ các inliers.
- **Nhược điểm:** Chất lượng phụ thuộc matches và ngưỡng residual; GPU batch launch/synchronization và numerical instability có thể làm mất lợi ích latency.
- **Fallback:** Batched vanilla RANSAC (inlier count) + một lần least-squares refinement nếu MSAC/refinement chưa ổn định. Đối chiếu với OpenCV RANSAC/`USAC_MAGSAC` để kiểm tra accuracy; PROSAC [11] là hướng nâng cấp, không phải yêu cầu Phase 0.

## 5. Protocol kiểm chứng của hai task

| Hạng mục | Cách kiểm chứng | Trạng thái |
| --- | --- | --- |
| Benchmark | Kiểm tra UDIS-D test set, định dạng, số pairs và protocol chia development/final | Chưa chạy |
| Quality evaluator | Kiểm thử overlap mask, PSNR/SSIM, coverage, invalid warps và pair-wise pass với ảnh giả lập | Chưa chạy |
| S4 correctness | So top-2 neighbor indices/distances với OpenCV trên cùng descriptors; ablation ratio/cross-check | Chưa chạy |
| S5 correctness | Known-H/synthetic correspondences, sample suy biến, reproducible seed, chuyển đổi tọa độ | Chưa chạy |
| Controller evaluation | Fixed-Low, Fixed-High, Fixed-Best, Adaptive và offline Oracle trên cùng UDIS-D pairs | Chưa chạy |
| QoR/failure analysis | Pair-wise pass, coverage, PSNR/SSIM difference, failure rate và panorama artifacts | Chưa chạy |
| GPU performance | Stage/end-to-end latency, controller overhead, escalation/reuse overhead | Chưa chạy |

**Fixed-Best** là một cấu hình dùng cho mọi pair được chọn trên development data theo quality policy, không phải cấu hình chọn riêng cho từng ảnh. **Oracle** là cấu hình nhanh nhất *trên từng pair* thỏa `Pass_i(c)`, chỉ biết được qua khảo sát offline. Đánh giá speedup phải đo với **cùng phạm vi end-to-end latency** của GPU pipeline, không so kernel-only time với host wall-clock time.

Go/No-Go vẫn theo `plan.md`: tỷ trọng thời gian của các stage bị tác động bởi S/K/R và khoảng cách latency giữa Oracle và Fixed-Best. Các mốc 30% và 20–25% trong kế hoạch là gợi ý; Nhóm quyết định ngưỡng chính thức. Một lượt thử nhanh 20–30 pairs chỉ đủ phục vụ Go/No-Go ban đầu, không thay thế final evaluation.

Môi trường đo chính thức là Colab Tesla T4; build `nvcc -arch=sm_75`. Lưu seed, thông tin GPU, cấu hình và code revision trong kết quả. Chỉ báo cáo số đo đã được sinh từ pipeline thực tế.

## 6. Việc còn cần phối hợp

- **Tuấn:** Chốt quy ước dữ liệu, CSV, overlap evaluation output và development/final split; xác nhận quality policy thuộc giao diện chung trước khi dùng.
- **Thành:** Chốt descriptor representation và distance metric để hoàn thiện S4; kiểm tra ảnh hưởng của K tới match quality.
- **Chương:** Chốt mask, cách đánh giá seam/ghosting và khả năng đưa B vào QoR; việc giữ B cố định trước mắt cần được thống nhất.
- **Cả Nhóm:** Xác nhận ngưỡng QoR sau pilot và các điều kiện Go/No-Go. Không sửa ngưỡng theo kết quả final test.

## 7. Tài liệu tham khảo

Ngày tra cứu: 15/10 (Thứ 5), năm 2026. Những số liệu hoặc metric lấy từ paper được phân biệt với ngưỡng do Nhóm đề xuất trong phần tương ứng.

1. Zhang, J. et al. (ECCV 2020), *Content-Aware Unsupervised Deep Homography Estimation*. [Paper](https://arxiv.org/abs/1909.05983): cơ sở và giới hạn của CA-UDHN.
2. Zhang, J. et al., [DeepHomography repository](https://github.com/JirongZhang/DeepHomography): nguồn dữ liệu/annotation CA-UDHN.
3. Nie, L. et al. (IEEE TIP 2021), *Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images*. [UDIS-D repository](https://github.com/nie-lang/UnsupervisedDeepImageStitching): nguồn dữ liệu và kết quả test.
4. Nie, L. et al. (ICCV 2023), *Parallax-Tolerant Unsupervised Deep Image Stitching*. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Nie_Parallax-Tolerant_Unsupervised_Deep_Image_Stitching_ICCV_2023_paper.html): warping, overlap evaluation, composition.
5. Nie, L. et al., [UDIS++ repository](https://github.com/nie-lang/UDIS2): triển khai và evaluation reference.
6. Lowe, D. G. (IJCV 2004), *Distinctive Image Features from Scale-Invariant Keypoints*. [DOI](https://doi.org/10.1023/B:VISI.0000029664.99615.94): nearest-neighbor distance ratio.
7. OpenCV, [BFMatcher documentation](https://docs.opencv.org/4.8.0/d3/da1/classcv_1_1BFMatcher.html): distance metrics, cross-check và kNN.
8. Lindenberger, P. et al. (ICCV 2023), *LightGlue: Local Feature Matching at Light Speed*. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html): learned/adaptive feature matching reference.
9. Fischler, M. A. and Bolles, R. C. (1981), *Random Sample Consensus*. [DOI](https://doi.org/10.1145/358669.358692): RANSAC.
10. Chum, O., Matas, J. and Kittler, J. (2003), *Locally Optimized RANSAC*. [DOI](https://doi.org/10.1007/978-3-540-45243-0_31): local refinement.
11. Chum, O. and Matas, J. (CVPR 2005), *Matching with PROSAC — Progressive Sample Consensus*. [DOI](https://doi.org/10.1109/CVPR.2005.221): sampling alternative.
12. Barath, D. et al. (CVPR 2020), *MAGSAC++, a Fast, Reliable and Accurate Robust Estimator*. [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html): robust estimation quality reference.
13. Cai, J. and Yang, Y. (AAAI 2025), *Object-level Geometric Structure Preserving for Natural Image Stitching*. [Repository/StitchBench](https://github.com/RussRobin/OBJ-GSP): optional stitching benchmark.
