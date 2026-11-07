# Quyết định: thí nghiệm DCS-R1

- Người yêu cầu/phê duyệt phạm vi: Đế.
- Ngày: 07/11 (Thứ 7), năm 2026.
- Phase 3, deadline chung 07/11 (Thứ 7).
- Trạng thái: implementation nghiên cứu, chưa có số đo T4 DCS; default không đổi.

Đế yêu cầu đổi objective trước, giữ PRE0, sau đó feature ablation/Gate/native/T4, thêm learning curve. Agent triển khai theo spec DCS-R1; không ghi agent là nguồn đề xuất nghiên cứu hoặc gán mọi chi tiết thực hiện cho Đế tự nghĩ ra.

Chọn XGBoost NDCG/pairwise, direct48 và ridge controls; search hẹp trong inner folds, family/features không chọn trên outer evaluation. Mỗi query đủ 48 configs, Reference có trong candidate list. Raw score chỉ có nghĩa thứ tự. Export scalar trees/native linear, giữ gate và fallback riêng. Baseline v5 frozen chỉ dùng khi kiểm đúng ZIP; grouped OOF v5 là nested reconstruction có tên riêng.

Trade-off: direct48 có thể overfit vì số query ít và 48 cây mỗi round; ranker còn có thể xếp FAIL đầu. Phải báo FAIL cùng regret và CI. NCC/interaction được benchmark với overhead, post information chỉ sau attempt quan sát. Learning curve dùng capacity cố định và cùng outer queries để xem ảnh hưởng số pair; chưa có approved data expansion hoặc fresh holdout thì không tạo dữ liệu thay thế.

Fallback: model/schema/catalog/feature lỗi dẫn Reference; execution/CUDA errors theo contract. Thiếu ZIP/ảnh báo path cần bổ sung, tiếp tục phần độc lập và xuất PARTIAL. Không thay QoR/Reference/default để tạo kết quả tốt. Không dùng MCP Colab; Đế tự Run all.

Kiểm chứng local gồm host build/CTest, Python CPU fixtures, raw export và full ranking parity, label/split/QoR regression, notebook kernel/auth, ZIP và checksum resume. CUDA/T4/full research CV cần notebook thực thi. Xem [implementation](../controller-direct-selection-implementation.md) và [runbook](../controller-direct-selection-runbook.md).
