"""Render the Vietnamese pooled-controller report from verified result artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re


def number(value,digits=4):
    return '—' if value is None else str(value) if isinstance(value,str) else f'{value:.{digits}f}'


def export(checkpoint,output):
    data=json.loads((checkpoint/'v4-detailed/summary.json').read_text())
    ablations=json.loads((checkpoint/'v4-fast-ablations/summary.json').read_text())
    slow=json.loads((checkpoint/'v4-feature-cost/summary.json').read_text())['groups']
    fast=json.loads((checkpoint/'v4-fast-feature-cost/summary.json').read_text())['groups']
    environment=json.loads((checkpoint/'v4-fast-evidence/environment.json').read_text())
    selected=data['selected_before_holdout'];group=selected.removeprefix('v4_')
    model=json.loads((checkpoint/'v4-fast-ablations'/group/'policy.json').read_text())
    mode=data['holdout']['modes'][selected];routing=data['routing'][selected]
    lost=len(routing['lost_reference_pass_pairs'])
    prefix=Path(os.path.relpath(checkpoint.resolve(),output.parent.resolve())).as_posix()
    lines=['# Kết quả mô hình Controller dùng chung','',
        'Báo cáo ngày 05/11 (Thứ 5), năm 2026; campaign T4 ngày 04/11 (Thứ 4). Các bảng sinh từ JSON/CSV.','',
        f"Biến thể được chọn trước holdout: `{selected}`. Mất {lost} cặp QoR PASS của reference trên tập kiểm tra mới. "
        +('Chưa đạt ràng buộc giữ chất lượng.' if lost else 'Không mất cặp QoR PASS của reference trong cohort đã đo.'),'',
        '## Phạm vi và protocol','',
        'Đã triển khai mô hình dùng chung theo scene và S/K/R, XGBoost cây nông, sigmoid calibration, '
        'gate sau S5 và inference C++ không phụ thuộc thư viện ML. Giữ đủ 48 cấu hình, gồm K500. '
        'Policy thử tối đa một candidate rồi fallback reference, tổng tối đa hai attempt. '
        'Không trích gate feature khi đã chọn reference. Chưa đổi policy mặc định.','',
        'Giữ nguyên QoR tương đối: SSIM giảm tối đa 0,02, PSNR giảm tối đa 0,5 dB, coverage tối thiểu 0,95. '
        'Reference S1/K2000/R1024, feather. Đây là QoR alignment tương đối; reference không phải homography ground truth '
        'và PASS không bảo đảm chất lượng ảnh tuyệt đối cao.','',
        'Calibration gồm 200 cặp training; fitter giữ các cặp có reference hợp lệ. Mỗi component ảnh nằm trong một outer fold. '
        'Booster fit và sigmoid calibration dùng các component khác nhau; chọn ngưỡng và hyperparameter loại outer evaluation. '
        'So sánh sáu nhóm feature bằng calibration trước khi mở holdout. Kết quả của nhóm được chọn trên calibration '
        'không được xem là ước lượng generalization không thiên lệch.','',
        'Khóa 50 component training mới trước khi fit, loại toàn bộ calibration và cả hai cohort validation đã xem. '
        'Kiểm CRC/size toàn index ZIP ngăn trùng ảnh; chưa chứng nhận scene disjointness bằng annotation. '
        'Không tune trên UDIS-D test hoặc HPatches. Không điều chỉnh model theo kết quả holdout.','',
        f"Binary đo cuối: `{environment['revision']}`; Tesla T4, `sm_75`, seed 20261101, hai warmup và mười repeat. "
        'Grid calibration có mười repeat trên source đã merge main; kernel feature được tối ưu sau đó. '
        'Collector đối chiếu H/status/count với grid. Phép đo cuối dùng cùng binary cho reference và các policy.','',
        '## Chi phí feature trước và sau song song hóa','',
        'Đây là microbenchmark trên calibration với route cố định, không phải policy đạt QoR. '
        'Probe/decision dùng model diagnostic nhỏ; inference của mô hình thật được tính trong E2E bên dưới. '
        'Không cộng các cột timing chồng lấp để suy ra latency.','',
        '| Nhóm | Gate cũ (ms) | Gate mới (ms) | Probe/decision cũ (ms) | Probe/decision mới (ms) | Attempt hợp lệ |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ['existing','pair','matching','geometry','photo','all']:
        a,b=slow[name],fast[name]
        lines.append(f"| {name} | {a['median_gate_ms']:.4f} | {b['median_gate_ms']:.4f} | {a['median_probe_decision_ms']:.4f} | {b['median_probe_decision_ms']:.4f} | {b['measured_attempts']} |")
    lines+=['','`pair`: tương quan ảnh nhỏ và vị trí/độ rõ peak. `matching`: phân bố distance và tỷ lệ top2 từ workspace S4. '
        '`geometry`: residual p50/p90 và sai số theo vùng. `photo`: SSIM patch 7×7, edge agreement và sai khác chuẩn hóa. '
        '`all`: kết hợp. Không chạy thêm reverse matching để lấy mutual fraction; nhóm matching tận dụng kết quả top2 đã có.','',
        '## Ablation ngoài fold trên calibration','',
        'Chi phí replay dùng timing grid và microbenchmark; probe có dự toán thêm 0,25 ms cho inference đầy đủ. '
        'Fallback tính bảo thủ, chưa trừ lợi ích reuse. Các số sau chưa phải speedup đo trên tập mới.','',
        '| Nhóm | Cặp | False accept | Retry | Dừng rẻ | Mean replay (ms) | Mean reference (ms) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in ablations['variants'].items():
        lines.append(f"| {name} | {s['pairs']} | {s['outer_false_accept']} | {s['outer_retries']} | {s['outer_cheap_stops']} | {s['outer_estimated_mean_ms']:.4f} | {s['reference_mean_ms']:.4f} |")
    minimum_errors=min(s['outer_false_accept'] for s in ablations['variants'].values())
    lines+=['',f"Chọn `{selected}` để đánh giá tiếp bằng calibration, ưu tiên ít false accept rồi xét chi phí. "
        +(f"Không nhóm nào đạt zero false accept ngoài fold; mức thấp nhất là {minimum_errors}. Đây là candidate nghiên cứu, chưa phải policy đạt ràng buộc để triển khai. " if minimum_errors else '')
        +'Xác suất model và zero error trên phần chọn ngưỡng không bảo đảm zero error trên dữ liệu khác.','',
        '## Latency và failure trên tập mới','',
        'Mỗi pair lấy median của mười repeat; mean/median/p95 dưới đây tính trên các pair có reference hợp lệ, '
        'gồm cả candidate QoR FAIL. Cột PASS/FAIL/unknown giữ đủ cohort 50 pair.','',
        '| Mode | PASS | FAIL | Unknown | N latency | Mean (ms) | Median (ms) | p95 (ms) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in data['holdout']['modes'].items():
        q=s['quality_status'];d=s['reference_valid_pair_median_distribution'];unknown=s['pairs']-q.get('PASS',0)-q.get('FAIL',0)
        lines.append(f"| {name} | {q.get('PASS',0)} | {q.get('FAIL',0)} | {unknown} | {d['n']} | {number(d['mean'])} | {number(d['median'])} | {number(d['p95'])} |")
    lines+=['','## Speedup trên cùng cặp QoR PASS','',
        'Bảng chỉ dùng giao các cặp cùng PASS với reference; không thay cho kiểm mất chất lượng trên toàn cohort. '
        'Speedup bằng reference/candidate. CI bootstrap 95% resample theo pair, không coi repeat là mẫu độc lập.','',
        '| Mode | N | Tỷ số mean | Tỷ số median | Median paired | CI 95% tỷ số mean | Mất PASS |',
        '|---|---:|---:|---:|---:|---|---:|']
    for name,s in data['holdout']['modes'].items():
        speed=s.get('speedup_vs_reference_joint_pass')
        if not speed:continue
        ci=speed['ratio_of_means_bootstrap95'];lost_n=len(data['routing'][name].get('lost_reference_pass_pairs',[]))
        lines.append(f"| {name} | {speed['pairs']} | {speed['ratio_of_means']:.4f}× | {speed['ratio_of_medians']:.4f}× | {speed['median_paired_speedup']:.4f}× | [{ci[0]:.4f}, {ci[1]:.4f}] | {lost_n} |")
    lines+=['','## QoR và hành vi policy','',
        'Các mean QoR gồm mọi candidate có metric hợp lệ trên tập reference-valid, cả PASS và FAIL; '
        'N có thể khác nhau theo metric. Không dùng mean gần reference để bỏ qua một pair FAIL.','',
        '| Mode | N SSIM | Mean SSIM | N PSNR | Mean PSNR (dB) | Mean coverage | Retry pair | False accept |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in data['holdout']['modes'].items():
        m=s['qor_metrics_reference_valid'];route=data['routing'][name]
        lines.append(f"| {name} | {m['ssim_overlap']['n']} | {number(m['ssim_overlap']['mean'])} | {m['psnr_overlap']['n']} | {number(m['psnr_overlap']['mean'])} | {number(m['overlap_coverage']['mean'],6)} | {route.get('retry_pairs',0)} | {route.get('false_accept_pairs',0)} |")
    lines+=['',f"Các pair mất PASS của `{selected}`: "+(', '.join('`'+p+'`' for p in routing['lost_reference_pass_pairs']) or 'không có trong cohort đã đo')+'.','',
        f"Phân bố cấu hình ban đầu `{selected}`:",'', '| Cấu hình | Pair |','|---|---:|']
    for name,count in routing['initial_config_counts'].items():lines.append(f'| {name} | {count} |')
    diagnostic_path=checkpoint/'v4-detailed/diagnostics.json'
    if diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text())
        lines+=['','## Vì sao mô hình vẫn chọn đắt','',
            'Bộ lọc trước chạy của model được chọn giữ xác suất từ '
            f"{diagnostic['pre_threshold']:.2f}; gate sau S5 dùng {diagnostic['gate_threshold']:.2f}. "
            'Bảng dưới replay model cuối trên calibration, gồm cả nhóm fit và calibration xác suất. '
            'Đây là chẩn đoán quyết định, không phải accuracy ngoài fold.','',
            '| Cấu hình | QoR PASS | Qua lọc trước chạy | PASS qua lọc trước chạy | Gate chấp nhận | Gate nhận nhầm |',
            '|---|---:|---:|---:|---:|---:|']
        for name,s in diagnostic['calibration_replay'].items():
            lines.append(f"| {name} | {s['qor_pass']} | {s['pre_eligible']} | {s['true_pass_pre_eligible']} | {s['post_accepted']} | {s['post_false_accept']} |")
        lines+=['','Model chưa phân biệt được phần lớn cơ hội chạy K500. Đồng thời gate đứng riêng vẫn nhận nhầm; '
            'hạ lọc trước chạy sẽ mở thêm cả positive lẫn negative. Tăng số cây chưa thay được thông tin về '
            'matching thực tế. 191 cặp ảnh vẫn là 191 đơn vị dữ liệu độc lập, dù mỗi cặp có 48 cấu hình.','',
            'Thông tin sau S5 hiện chỉ quyết định accept hoặc fallback reference. Chưa dùng nó để chọn lại '
            'một S/K/R khác theo chi phí transition. Đây là hướng cần cân nhắc cho lượt nghiên cứu tiếp theo, '
            'cùng với thêm calibration; không tune lại bằng tập kiểm tra vừa xem.','',
            '| Hành vi | Tổng pair | Cùng PASS | Mean reference (ms) | Mean Controller (ms) |',
            '|---|---:|---:|---:|---:|']
        for name,s in diagnostic['branch_breakdown'].items():
            lines.append(f"| {name} | {s['all_pairs']} | {s['joint_pass_pairs']} | {number(s['reference']['mean'])} | {number(s['controller']['mean'])} |")
        for failure in diagnostic['false_accept_details']:
            lines+=['',f"Pair `{failure['pair_id']}`, cấu hình `{failure['config']}`; lý do FAIL: "+', '.join(failure['reasons'])+'.','',
                '| Kết quả | SSIM | PSNR (dB) | Coverage |','|---|---:|---:|---:|']
            for name in ['candidate','reference']:
                s=failure[name]
                lines.append(f"| {name} | {s['ssim_overlap']:.4f} | {s['psnr_overlap']:.4f} | {s['overlap_coverage']:.6f} |")
    lines+=['','## Đối chiếu Oracle trên cùng tập mới','',
        'Lượt sweep bổ sung tách khỏi campaign trên. Chỉ dùng số trong lượt này để tính chênh với Oracle. '
        'Oracle chọn sau khi chạy toàn grid và chấm QoR, không gồm chi phí tìm kiếm online.','',
        '| Mode | Mean (ms) | Median (ms) | p95 (ms) |','|---|---:|---:|---:|']
    for name,s in data['oracle_followup']['modes'].items():
        d=s['reference_valid_pair_median_distribution']
        lines.append(f"| {name} | {number(d['mean'])} | {number(d['median'])} | {number(d['p95'])} |")
    o=data['oracle_comparison'];d=o['oracle_latency']
    lines.append(f"| Oracle | {number(d['mean'])} | {number(d['median'])} | {number(d['p95'])} |")
    lines+=['',f"Oracle có đáp án cho {o['oracle_available_pairs']}/{o['pairs']} pair. "
        f"Cấu hình đầu trùng Oracle {o['initial_exact_oracle']}; cấu hình cuối trùng {o['final_exact_oracle']}. "
        f"Có {o['oracle_config_available_in_policy']} đáp án Oracle trong tập cấu hình policy. "
        f"Có {o['near_oracle_within_10percent_pairs']} cấu hình cuối đạt QoR và có latency không quá 10% so với Oracle.",'',
        '| Tham số cuối | Trùng Oracle | Mean độ lệch tuyệt đối | Median độ lệch tuyệt đối |',
        '|---|---:|---:|---:|']
    for k in ['S','K','R']:
        d=o['absolute_parameter_delta'][k]
        lines.append(f"| {k} | {o['parameter_exact_counts'][k]} | {number(d['mean'])} | {number(d['median'])} |")
    decomposition=o['mean_latency_decomposition_joint_pass']
    if decomposition:
        d=decomposition
        lines+=['',f"Trên cùng {d['pairs']} pair PASS: Oracle {d['oracle_ms']:.4f} ms → fixed cấu hình đã chọn "
            f"{d['chosen_configuration_fixed_ms']:.4f} ms → Controller E2E {d['controller_e2e_ms']:.4f} ms. "
            f"Chênh chọn cấu hình {d['configuration_gap_ms']:.4f} ms; phần E2E còn lại {d['e2e_minus_chosen_fixed_ms']:.4f} ms. "
            'Phần sau gồm probe, gate, retry, host và khác biệt lượt đo; không gán toàn bộ cho một kernel.']
    lines+=['','## Feature thực sự xuất hiện trong cây','',
        'Đếm node split của model được chọn; đây không phải mức quan trọng nhân quả.','',
        '| Model | Feature | Node split |','|---|---|---:|']
    for name,ensemble in model['models'].items():
        names=model['provenance']['feature_names']['post' if name=='post_quality' else 'pre']
        counts=Counter(node['feature'] for tree in ensemble['trees'] for node in tree if 'feature' in node)
        for index,count in counts.most_common(5):lines.append(f'| {name} | {names[index]} | {count} |')
    previews=json.loads((checkpoint/'v4-images/previews.json').read_text())
    lines+=['','## Ảnh kiểm tra','',
        'Đây là alignment preview theo H đo được, không phải panorama S6. Metric đối chiếu CSV trong 1e-12. '
        'WebP lossless được kiểm pixel trước khi phục hồi PNG ở local.']
    for case in previews['cases']:
        lines+=['',f"### {case['category']}: {case['pair_id']}",'',
            f"Controller `{case['controller_config']}`; Oracle `{case['oracle_config']}`.",'',
            f"![{case['pair_id']}]({prefix}/v4-images/{case['preview']})"]
    ctest=(checkpoint/'v4-fast-evidence/ctest.log').read_text();python=(checkpoint/'v4-fast-evidence/pytest.log').read_text()
    c=re.search(r'0 tests failed out of (\d+)',ctest);p=re.search(r'(\d+) passed',python)
    lines+=['','## Kiểm chứng và artifact','',
        f"- T4 source `{environment['revision'][:7]}`: {c.group(1) if c else 'xem log'} CTest PASS, {p.group(1) if p else 'xem log'} test Python PASS.",
        '- Memcheck feature mới, feature cũ và regression harness: 0 errors; Python/C++ kiểm prediction và chọn cấu hình ban đầu trên calibration.',
        '- Trong sweep Oracle, các run v4 thành công được kiểm H/status/count/seed khớp fixed cùng cấu hình; không chỉ kiểm policy giữa hai lượt.',
        f"- [JSON thống kê]({prefix}/v4-detailed/summary.json), [CSV từng pair]({prefix}/v4-detailed/latency-per-pair.csv), [CSV so với Oracle]({prefix}/v4-detailed/controller-vs-oracle.csv).",
        f"- [Freeze holdout]({prefix}/v4-validation/freeze.json), [tóm tắt ablation]({prefix}/v4-fast-ablations/summary.json), [parity C++]({prefix}/v4-fast-parity/summary.json).",
        '- Không dùng kết quả development này thay bảng toàn UDIS-D test, HPatches hoặc nghiệm thu Phase 3 gốc. Chưa push hoặc tạo PR.','']
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    export(**vars(p.parse_args()))
