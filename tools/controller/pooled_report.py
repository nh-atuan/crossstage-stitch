"""Generate the calibration-only pooled-model progress report from frozen JSON."""
import argparse
import json
from pathlib import Path
from tools.controller.calibrate import digest


def report(summary_path,split_path,output):
    s=json.loads(summary_path.read_text());selected=s['final_fit']['selected']
    lines=['# Mô hình dùng chung: phép thử với feature hiện có','',
        'Ngày 04/11 (Thứ 4), năm 2026. Đây là báo cáo tiến độ phần cải tiến bổ sung; chưa nghiệm thu Controller mới.',
        '', '## Thiết kế và trạng thái','',
        'Đã train ba mô hình XGBoost dùng chung: chất lượng trước chạy, latency và chất lượng sau S5. '
        'Feature scene kết hợp S/K/R; mô hình sau S5 nhận thêm 24 feature đã có. '
        'Giữ toàn bộ 48 cấu hình, gồm K500. Không đổi predicate QoR.', '',
        'Outer CV chia theo component ảnh. Bên trong mỗi outer train, booster fit trên khoảng hai phần ba '
        'component; phần còn lại dùng sigmoid calibration, chọn ngưỡng và hyperparameter. '
        'Mọi bước trên loại outer evaluation. Mỗi cặp chỉ xuất hiện ở một outer evaluation.', '',
        'Chi phí replay dùng latency grid cũ, cộng dự toán probe/inference 0,5 ms và feature đã đo. '
        'Fallback tính lại đầy đủ, chưa trừ reuse; đây là mô hình chi phí bảo thủ, chưa phải E2E mới trên T4.', '',
        '## Kết quả ngoài fold','',
        '| Fold | Cặp | Mean reference (ms) | Mean replay (ms) | False accept | Retry | Dừng rẻ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for f in s['folds']:
        lines.append(f"| {f['fold']} | {f['pairs']} | {f['reference_mean_ms']:.4f} | {f['mean_ms']:.4f} | {f['false_accept']} | {f['retries']} | {f['cheap_stops']} |")
    lines+=['',f"Tổng {s['pairs']} cặp: {s['outer_false_accept']} false accept, {s['outer_retries']} retry, "
        f"{s['outer_cheap_stops']} lần dừng rẻ. Mean replay {s['outer_estimated_mean_ms']:.4f} ms, "
        f"reference {s['reference_mean_ms']:.4f} ms. Chưa có lợi thế dưới cùng QoR.", '',
        f"Fit cuối chọn depth {selected['depth']}, {selected['rounds']} cây mỗi mô hình; "
        f"ngưỡng trước chạy {selected['pre_threshold']:.3f}, sau S5 {selected['gate_threshold']:.3f}. "
        'Không có false accept trên phần chọn ngưỡng không bảo đảm zero error ngoài fold.', '',
        '## Phần tiếp theo','',
        'Đã khóa 50 component training mới, loại cả hai cohort validation đã xem và toàn calibration. '
        'Chưa chạy đánh giá policy trên tập này. Đã phục hồi ảnh trên Colab; không dùng nhãn holdout để học.', '',
        'Feature CUDA quan hệ hai ảnh đã có module và fixture riêng. Chưa tích hợp các feature này '
        'xong vào phép đo fitter/harness; chưa có ablation lợi ích hoặc chi phí E2E. Loader/inference v4 '
        'đã được triển khai và đang kiểm chứng tích hợp trên T4. Chưa thay policy mặc định.', '',
        'Cần chốt source sau cập nhật main, tích hợp loader/inference/feature, kiểm parity và thu feature '
        'calibration, rồi mới so ablation và chạy tập kiểm tra đã khóa. Nếu source mới thay output '
        'H/status/count, phải sinh lại nhãn; không trộn ngầm kết quả hai binary.', '',
        '## Truy xuất dữ liệu','',
        f"- SHA-256 split đã khóa: `{digest(split_path)}`.",
        f"- SHA-256 summary: `{digest(summary_path)}`.",
        '- Artifact local: `results/controller-v4-t4-20261104/existing-features-fit/`.',
        '- Nguồn thiết kế và trade-off: [quyết định mô hình dùng chung](../decisions/p3-controller-pooled-model.md).','']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['summary-path','split-path','output']:p.add_argument('--'+name,type=Path,required=True)
    report(**vars(p.parse_args()))
