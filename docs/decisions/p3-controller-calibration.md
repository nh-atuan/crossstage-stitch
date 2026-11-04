# Quyết định: hiệu chỉnh policy Controller từ development

- **Người quyết định:** Đế, đồng ý triển khai các đề xuất trong phiên làm việc.
- **Ngày:** 03/11 (Thứ 3), năm 2026.
- **Phase:** Phase 3; deadline chung 05/11 (Thứ 5).
- **Trạng thái:** Đang triển khai và kiểm chứng trên Colab Tesla T4.

## Bối cảnh

Campaign development ảnh thật đã xét SSIM/PSNR. Trên 23 cặp có reference hợp
lệ, tỷ số median Fixed-Best/Oracle là 2,048. Hai reference failures vẫn được
giữ trong cohort 25 cặp. Không dùng kết quả synthetic để phủ nhận bằng chứng
này. Trong 23 cấu hình Oracle chọn, 22 cấu hình không có trong ladder v1.

Policy v1 gán S cao với K/R cao và chọn ban đầu bằng các ngưỡng gradient gán
trong harness. Self-check chưa được hiệu chỉnh theo predicate QoR. Trace
test v1 còn có các retry chỉ đổi R khi số match thấp hơn yêu cầu inlier của
cấu hình kế tiếp; những retry này không thể được chấp nhận.

## Phương án chọn

Giữ module host, scene signals hiện có và metric S5. Thêm cấu hình policy
tường minh và công cụ hiệu chỉnh offline; online không đọc nhãn QoR hoặc
kết quả Oracle theo pair. Ladder được chọn từ grid development, cho phép
S cao nhưng K/R thấp. Self-check được học từ cùng nhãn SSIM/PSNR/coverage
hiện có; không thay đổi quality policy để tạo speedup.

Khóa split trước phép đo mới: 200 component đầu của pool development 250
component cũ dùng calibration; 50 component cuối dùng validation. Mỗi
component có một pair, không chia các repeat của cùng pair sang hai tập.
Pool chỉ lấy từ training UDIS-D. Các cặp có reference failure giữ nguyên;
không gán chúng thành QoR pass hoặc chọn pair thay thế. Không dùng UDIS-D
test hoặc HPatches để hiệu chỉnh. Seed RANSAC của campaign mới là 20261101.

Với scene signal hiện có, chọn các khoảng gradient theo chi phí rollout
trên calibration, có giới hạn số khoảng và số pair tối thiểu mỗi khoảng.
Các nấc được phép làm điểm khởi đầu, kể cả scale cao. Chọn self-check theo
khả năng phân biệt QoR trong calibration; báo cả false accept và false
reject trên validation. Kết quả replay là chẩn đoán, không thay latency đo
thực tế trên GPU. Freeze policy trước khi xem kết quả validation và không
sửa threshold theo các ca validation lỗi.

Giới hạn trước validation: tối đa 12 cấu hình được chọn và một reference
cuối; mỗi gate cần ít nhất bốn positive và không có false accept đã biết
trên calibration. Tìm tối đa 12 giá trị mỗi trục, có tính sai số làm tròn
CSV. Rule gradient có tối đa bốn khoảng, mỗi khoảng ít nhất 20 pair; phạt
thêm 0,05 ms/pair cho mỗi khoảng để tránh chia nhỏ chỉ vì dao động latency.
Reference cuối chấp nhận model S5 hợp lệ để làm baseline QoR tương đối;
không đồng nhất nó với ground truth hoặc bảo đảm chất lượng tuyệt đối.

Khi xét retry, nếu S/K không đổi và số match nhỏ hơn `min_inliers` của ứng
viên thì bỏ ứng viên đó. Quy tắc dựa trên giới hạn `inliers <= matches`,
không dựa trên nhãn chất lượng hoặc giả định tăng R luôn có lợi. Ứng viên
đổi S/K vẫn được chạy vì có thể tạo tập match mới.

## Trade-off và phương án phụ

| Phương án | Ưu điểm | Hạn chế và lựa chọn |
|---|---|---|
| Hiệu chỉnh rule và self-check hiện có | Ít thay đổi API, kiểm được nguồn dữ liệu và chi phí online | Gradient và metric S5 có thể chưa phân biệt đủ QoR; chọn để kiểm chứng trước |
| Giữ ladder v1, chỉ giảm retry | Loại được công việc chắc chắn không hữu ích | Không truy cập được phần lớn cấu hình Oracle; không đủ để xử lý vấn đề |
| Thêm feature hoặc mô hình phức tạp | Có thể phân biệt geometry/overlap tốt hơn | Cần thêm phép đo, thay giao diện và kiểm overhead; chưa chọn trong lần này |
| Dùng Oracle hoặc SSIM/PSNR để quyết định online | Biết cấu hình pass khi đã khảo sát đủ | Rò nhãn offline và bỏ chi phí tìm kiếm; không dùng |

## Fallback và nghiệm thu

Cập nhật 04/11 (Thứ 4), năm 2026: kiểm raw phát hiện harness chưa truyền seed
CLI vào các phần tử đọc từ `--configs`; đường single-config và adaptive đã
nhận seed. Đã dừng grid trước bước fit và validation. Giữ các batch seed 0
làm diagnostic, không dùng chúng để nghiệm thu protocol seed 20261101.
Sửa tại harness, thêm regression chạy binary thật cho cả fixed/oracle và
kiểm identity/seed/repeat của từng shard trước khi chạy batch tiếp theo.
Fitter từ chối cả trường hợp mọi dòng đồng nhất nhưng dùng sai seed.
Không sửa cột seed trong CSV để hợp thức hóa dữ liệu đã đo; chạy lại grid.

Nếu validation không chứng minh được lợi ích dưới cùng QoR, báo kết quả đó;
không nới QoR, bỏ pair lỗi hoặc tune trên validation/test. Policy cũ và kết
quả cũ là baseline có provenance riêng, không ghi đè checkpoint đã bàn giao.
Chỉ gọi policy mới là đã kiểm chứng khi có build `sm_75`, test liên quan và
campaign thực chạy trên T4. Không hứa trước speedup 2x của controller online.

## Nguồn

- `results/udis_train_gate_20261030/`: raw sweep, canonical QoR và manifest
  development đã khóa; artifact cục bộ, không commit dataset hoặc kết quả.
- https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing — đọc và rà
  văn phong ngày 03/11 (Thứ 3), năm 2026.
