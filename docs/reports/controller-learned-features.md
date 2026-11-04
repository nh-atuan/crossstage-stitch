# Feature nhiều chiều, gate QoR và chi phí retry

Ngày 04/11 (Thứ 4), năm 2026. Phần bổ sung của Phase 3 theo yêu cầu của
Đế; deadline chung 05/11 (Thứ 5). Source đo `47256d4`, Colab Tesla T4,
Release `sm_75`, seed RANSAC 20261101, hai warmup và mười repeat.

Đã triển khai bốn hướng cải tiến và đo trên tập mới. Biến thể p80 chọn
cấu hình rẻ ở một phần dữ liệu, nhưng mất một pair QoR PASS và chưa có lợi
thế latency trên tập cùng đạt chất lượng. Chưa có cơ sở thay baseline bằng
policy này. Các biến thể giữ đủ QoR vẫn chọn reference và chịu thêm overhead.

## Dữ liệu và protocol

Giữ 200 pair calibration cũ, trong đó 191 pair có reference hợp lệ. Không
dùng 50 pair validation v2 để tune hoặc làm tập kiểm tra mới. Trước khi fit,
khóa 50 component training UDIS-D ngoài toàn bộ 250 component cũ. Index ZIP
cho 1.789 component theo liên kết ảnh có CRC/size trùng; phép chọn dùng hash
cố định của component, không đọc QoR/latency. Đây là kiểm trùng ảnh, chưa
phải chứng nhận scene disjointness theo nhãn chính thức.

SHA-256 split mới:
`a315d1714bc3fbabc741e55c19b2698f1f853e95f0ee34fcfd75d475dfe2cdc6`.
Script `freeze_learned_split` tái tạo đúng bytes của split, restore split và
selected index đã khóa. Giữ nguyên predicate pilot: SSIM giảm tối đa 0,02,
PSNR giảm tối đa 0,5 dB, coverage tối thiểu 0,95 so với reference. Không dùng
UDIS-D test hoặc HPatches trong quá trình hiệu chỉnh.

Thu feature của 48 cấu hình trên 200 pair bằng binary `c44dcc8`. Cả 9.600
quan sát được kiểm H/status/count/seed khớp grid `e378c53`; NaN H của các
model lỗi được so đúng với NaN tương ứng, không tính thành model hợp lệ.
Latency/QoR dùng để học vẫn lấy từ grid lặp đã có, không lấy latency fixed
có bật instrumentation feature.

Khóa ba policy v3 trước khi đo tập mới: strict, p90 và p80. Hai tên sau chỉ
precision ngoài fold tối thiểu khi chọn gate, không phải ngưỡng SSIM/PSNR,
không phải bảo đảm precision trên dữ liệu mới. Đo thêm reference/Fixed-Best,
Fixed cũ, legacy và v2. Đảo thứ tự mode theo batch; hash từng ảnh input,
binary, model, split và QoR policy được ghi trước phép đo và kiểm lại sau đo.

## Phân tích mẫu: S5 score và QoR không tương đương

Hai mẫu calibration cùng chạy S0.75/K500/R128:

| Chỉ số | Cảnh hồ `000069` | Cảnh đêm `000628` |
|---|---:|---:|
| Inliers | 148 | 311 |
| Residual S1 | 1,00985 | 0,97740 |
| Inlier bbox coverage | 0,16532 | 0,25024 |
| SSIM giảm so với reference | -0,00087 | 0,02027 |
| PSNR giảm so với reference (dB) | 0,00769 | 0,65234 |
| QoR | PASS | FAIL |
| RMSE photometric lấy mẫu | 0,01681 | 0,03879 |
| Correlation photometric lấy mẫu | 0,99528 | 0,97461 |

Mẫu cảnh đêm có cả bốn metric S5 tốt hơn mẫu cảnh hồ, nhưng trượt cả ngưỡng
SSIM và PSNR tương đối. Gate ngưỡng cũ loại cả hai. Feature photometric mới
phân biệt được hai mẫu này rõ hơn; hai ví dụ không chứng minh khả năng tổng
quát trên toàn dataset. Ảnh A/B, warp theo candidate/reference và error map
ở `v3-analysis/calibration-counterexamples.png` trong checkpoint.

QoR ở đây đánh giá alignment tương đối với reference, không phải điểm thẩm
mỹ của panorama hay homography ground truth. Inlier bbox coverage, tỷ lệ
overlap lấy mẫu trong frame A và coverage QoR dùng các mẫu số khác nhau;
không dùng chung một ngưỡng cho ba đại lượng này.

## Thay đổi đã thực hiện

| Phần | Triển khai |
|---|---|
| Probe và bộ chọn | Giữ riêng gradient A/B; mean, contrast, gradient trên bốn vùng của mỗi ảnh; bốn chênh lệch mean A/B theo vùng. Tổng 30 đầu vào selector |
| Gate sau S5 | 24 feature: match/inlier, residual S1/pixel gốc, hình học H, photometric trên lưới 64 × 64, occupancy/largest-cell/entropy inlier trên lưới 4 × 4 ở cả A/B |
| Cách học | Cây giới hạn depth và leaf size; năm fold theo pair/component; dùng nhãn QoR và latency sweep; export mảng node C++ cùng quy tắc `<=` |
| Chi phí retry | Đo transition đổi R, K, S; tính stage phải chạy lại và phần R tăng thêm; học retry target tiến về phía sau để có thể đi thẳng fallback |
| Kiểm tra runtime | Loader kiểm schema, finite, feature/output index và target tiến; không allocation, không đọc nhãn/reference trong per-pair Controller |

Gate chọn depth 2 hoặc 3, ít nhất tám mẫu ở lá, chỉ ACCEPT lá thuần positive
trong tập fit. Kiểm ngoài fold vẫn có thể nhận nhầm: không coi zero training
error là bảo đảm QoR. Selector depth tối đa ba, ít nhất 20 pair ở lá, có
penalty chia nhánh. Loss cho false stop cộng chi phí reference trên dữ liệu
calibration; đây là penalty học offline, không phải kiểm QoR online.

Kernel feature nằm riêng trong Controller; không đổi thuật toán/tham số
S1–S6 hoặc evaluator chuẩn. Workspace cấp trước vùng đo. Chi phí feature,
synchronization và decision được giữ trong end-to-end adaptive. Policy v1/v2
vẫn có đường chạy tường minh và checkpoint riêng.

## Đo chi phí transition

Chọn trước 32 pair calibration bằng hash, đo các chuỗi R-only, đổi K và đổi
S. Đường R-only có cả reuse/no-reuse để đối chiếu. Fit từ 3.140 attempt đo
của các chuỗi có reuse; repeat cùng pair không chia sang hai phía khi kiểm
sai số. Trên tám pair giữ riêng, MAE là 0,30865 ms và p95 absolute error là
0,80666 ms; sau đó fit cuối trên 32 pair.

Đây là mô hình chi phí xấp xỉ, không phải ground truth latency. Các stage
có chi phí tương quan nên hệ số không được diễn giải thành mức quan trọng
nhân quả. Mọi kết luận hiệu năng dưới đây dùng raw end-to-end thực đo.

Sau khi học retry target, p80 bắt đầu ở S1/K1000/R256 cho 72/191 pair
calibration hợp lệ, còn lại chọn reference. Strict và p90 vẫn chọn reference
cho mọi pair. Đây là replay calibration, không dùng thay kết quả holdout.

## Kết quả trên 50 pair mới

Một pair reference không đủ điểm; giữ riêng dưới REFERENCE_FAILED, không
gán PASS và không chọn pair thay thế.

| Mode | QoR PASS | FAIL | Reference unknown | Bắt đầu cấu hình rẻ | Pair retry |
|---|---:|---:|---:|---:|---:|
| Reference/Fixed-Best S1/K2000/R1024 | 49 | 0 | 1 | 0 | 0 |
| Fixed cũ S0.5/K1000/R256 | 25 | 24 | 1 | 50 | 0 |
| Legacy | 25 | 24 | 1 | 50 | 6 |
| v2 | 49 | 0 | 1 | 0 | 0 |
| v3 strict | 49 | 0 | 1 | 0 | 0 |
| v3 p90 | 49 | 0 | 1 | 0 | 0 |
| v3 p80 | 48 | 1 | 1 | 22 | 12 |

P80 bắt đầu 22 pair ở S1/K1000/R256 và 28 pair tại reference. Có một false
accept đã biết, pair `udis_train_003317`; không tune lại gate từ pair này.
Legacy có 21 false accept đã biết. P80 giảm false accept và chọn ban đầu
đa dạng hơn legacy, nhưng vẫn mất một PASS so với reference/v2.

Trong 22 pair bắt đầu rẻ, mười pair dừng ngay: chín PASS và một FAIL; 12 pair
phải retry rồi PASS. Cây selector cuối chỉ dùng `b_tile2_gradient` với ngưỡng
0,03897764 để chọn giữa cấu hình rẻ và reference. Dù có 30 đầu vào, dữ liệu
calibration và tiêu chí CV chưa chọn được bộ phân biệt nhiều feature tốt
hơn. Không đồng nhất việc bổ sung đầu vào với việc đã học được scene-aware
selection hiệu quả.

Phân nhóm theo hành vi p80, chỉ so cùng các pair QoR PASS trong mỗi hàng:

| Nhóm | Pair | Median Fixed-Best (ms) | Median p80 (ms) |
|---|---:|---:|---:|
| Dừng ở cấu hình rẻ và PASS | 9 | 13,4955 | 7,5650 |
| Thử rẻ, phải retry rồi PASS | 12 | 8,5140 | 13,15025 |
| Bắt đầu reference và PASS | 27 | 6,9105 | 7,6360 |

Fixed-Best luôn chạy một attempt; tên nhóm mô tả hành vi của p80. Chín pair
đầu có tỷ số hai median khoảng 1,784×, nhưng đây là tập con có điều kiện.
Chi phí thử sai ở 12 pair và overhead trên các pair chọn reference làm mất
lợi ích ở mức toàn policy. Không so trực tiếp latency giữa các hàng như
thể đó là cùng một cohort.

Ở false accept `003317`, PSNR candidate tốt hơn reference khoảng 0,562 dB,
nhưng SSIM giảm từ 0,54952 xuống 0,46205, vượt giới hạn giảm 0,02. Coverage
0,97703 vẫn đạt. Sai ở tiêu chí SSIM, không phải PSNR hoặc coverage. Điều
này cho thấy MAE/RMSE/correlation photometric và điểm S5 chưa thay được
predicate kết hợp; gate mới vẫn chỉ là predictor cần kiểm độc lập.

Trên cùng 49 pair mà reference, v2, strict và p90 đều đạt QoR:

| Mode | Median end-to-end (ms) |
|---|---:|
| Reference/Fixed-Best | 8,5710 |
| v2 | 8,7415 |
| v3 strict | 9,2600 |
| v3 p90 | 9,3400 |

Gate feature có median tổng chi phí theo pair khoảng 0,40 ms; đây chưa gồm
toàn bộ probe và decision. Bản v3 chỉ chọn reference không thu được lợi ích
cấu hình rẻ và phát sinh overhead; kết quả đo chậm hơn baseline trong lần
đo này. Không cộng các cột timing chồng lấp để suy ra end-to-end.

P80 chỉ có 48 pair cùng PASS với reference: median reference 8,50275 ms,
p80 8,56325 ms; tỷ số hai median 0,99293×. Median tỷ số paired là 0,89625×.
Hai thống kê khác nhau và đều chưa cho thấy speedup. Không dùng median
toàn cohort có failure hoặc bỏ pair QoR FAIL rồi tuyên bố giữ nguyên chất
lượng. Kết quả thuộc development holdout này, chưa có kiểm định bất định
hay phép đo toàn UDIS-D test/HPatches cho v3.

## Kiểm chứng và artifact

T4 source `47256d4`: 21/21 CTest, 409 test Python PASS; compute-sanitizer
memcheck kernel feature có 0 errors. Helper tái tạo split bổ sung có hai
test về component leakage và số component không đủ; kiểm bytes tái tạo
khớp protocol thực đã khóa. Hai test helper cũng PASS trên T4; toàn bộ 411
test Python PASS ở local. Các kết quả raw và model không được commit.

Checkpoint cục bộ:
[`results/controller-v3-t4-20261104/final-checkpoint/`](../../results/controller-v3-t4-20261104/final-checkpoint/).
`v3-validation/freeze.json` ghi provenance trước phép đo;
`v3-validation/summary.json` và `qor.csv` chứa kết quả;
`artifact-hashes.json` kiểm các file tải từ T4. `external-input-hashes.json`
trỏ tới grid v2 và policy cũ với SHA-256, được bổ sung vào checkpoint từ
artifact local đã kiểm hash. Dùng các lệnh và giới hạn ở
[`tools/controller/README.md`](../../tools/controller/README.md) để tái lập.

## Kết luận bàn giao

Đã có feature, bộ học, gate và retry routing mới cùng phép đo độc lập.
Chưa đạt mục tiêu nhanh hơn dưới cùng QoR. Giữ Fixed-Best hoặc policy v2
tường minh làm đối chứng chất lượng; chưa khuyến nghị thay bằng p80.
Không đổi mặc định wrapper sang policy chưa đạt tiêu chí. Cải tiến tiếp
cần xử lý đồng thời khả năng phân biệt QoR tương đối và overhead feature;
không dùng tập 50 pair vừa đo để hiệu chỉnh rồi gọi lại là holdout mới.
