"""Render the Vietnamese report directly from audited statistics and ledgers."""
import argparse
import csv
import json
from pathlib import Path


def export(report_path,output):
    data=json.loads(report_path.read_text());original=data['original_campaign']['modes'];old=data['previous_campaign']['modes']
    oracle=data['oracle_comparison'];follow=data['oracle_followup']['modes'];branches=data['original_p80_branch_breakdown']
    fmt=lambda x,n=4: '—' if x is None else str(x) if isinstance(x,str) else f'{x:.{n}f}'
    lines=['# Báo cáo chi tiết latency, QoR và Oracle','',
        'Ngày 04/11 (Thứ 4), năm 2026. Bảng được sinh từ raw CSV và canonical QoR bằng `detailed_metrics`/`detailed_export`.',
        'Binary đo `47256d4`, Tesla T4, `sm_75`; ảnh 512 × 512; seed 20261101; hai warmup và mười repeat. Policy đã khóa, không tune lại bằng Oracle holdout.','',
        '## 1. Vì sao có median 13,496 ms?','',
        '13,496 ms là Fixed-Best trên riêng chín pair mà p80 thử rẻ, dừng ngay và PASS QoR. Fixed-Best luôn chạy một attempt; con số này không cộng retry.',
        'Mốc khoảng 7,5 ms trước đó thuộc một cohort/lượt đo khác. Hai cohort validation cũ/mới không trùng pair. Không dùng tỷ số giữa các hàng dưới đây làm speedup.','',
        '| Cohort và mode | N | Mean của pair-median (ms) | Median (ms) | p95 (ms) |',
        '|---|---:|---:|---:|---:|']
    contexts=[('Validation cũ: Fixed-Best',old['fixed_best']['reference_valid_pair_median_distribution']),
        ('Validation cũ: Controller v2',old['adaptive']['reference_valid_pair_median_distribution']),
        ('Validation mới: Fixed-Best',original['reference']['reference_valid_pair_median_distribution']),
        ('Nhóm chín pair: Fixed-Best',branches['cheap_stop']['metrics']['reference']['pair_median_distribution']),
        ('Cùng chín pair: p80',branches['cheap_stop']['metrics']['v3_p80']['pair_median_distribution'])]
    for name,s in contexts:lines.append(f"| {name} | {s['n']} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['p95'])} |")
    nine=list(csv.DictReader((report_path.parent/'nine-pair-breakdown.csv').open(newline='')));nine.sort(key=lambda r:float(r['fixed_median_ms']))
    lines+=['','Cả chín pair dùng Fixed-Best S1/K2000/R1024 và Controller S1/K1000/R256, blend feather.','',
        '| Pair | Fixed median (ms) | Controller median (ms) | Matches Fixed | Mean S5 Fixed (ms) |','|---|---:|---:|---:|---:|']
    for r in nine:lines.append(f"| {r['pair_id']} | {fmt(float(r['fixed_median_ms']))} | {fmt(float(r['controller_median_ms']))} | {r['fixed_matches']} | {fmt(float(r['t_s5_mean_ms']))} |")
    lines+=['','Mean stage dưới đây lấy trực tiếp các lượt đo, không cộng các median stage. `t_s6` của fixed bao gồm khoảng chờ host sau S5; không diễn giải mọi cột là thời gian kernel thuần.','',
        '| Nhóm chín pair | S1 | S2 | S3 | S4 | S5 | S6 raw | Mean E2E raw (ms) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,label in [('reference','Fixed-Best'),('v3_p80','Controller')]:
        s=branches['cheap_stop']['metrics'][mode]
        lines.append('| '+label+' | '+' | '.join(fmt(s['stage_means'][k]) for k in ['t_s1','t_s2','t_s3','t_s4','t_s5','t_s6'])+f" | {fmt(s['raw_run_distribution']['mean'])} |")
    lines+=['',f"Nhóm chín pair có mean {fmt(branches['cheap_stop']['metrics']['reference']['matches_mean'],1)} matches ở Fixed-Best; toàn 49 pair mới là {fmt(original['reference']['matches_mean_reference_valid'],1)}. S5 chấm hypothesis trên tập matches nên cùng R không đồng nghĩa cùng chi phí. Đây là bằng chứng workload khác; không tách hết ảnh hưởng xung nhịp/lượt đo.",
        '','## 2. Latency trên tập mới trong campaign gốc','',
        'Mean/median/p95 dưới đây tính trên median của mười repeat mỗi pair. Cùng 49 pair reference hợp lệ, gồm cả các QoR FAIL của từng mode; bảng latency này không chứng nhận giữ QoR. Một reference failure được giữ riêng trong cohort 50 pair.','',
        '| Mode | PASS | FAIL | Unknown | Mean (ms) | Median (ms) | Std (ms) | p95 (ms) | Min–max (ms) |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,m in original.items():
        s=m['reference_valid_pair_median_distribution'];q=m['quality_status']
        lines.append(f"| {name} | {q.get('PASS',0)} | {q.get('FAIL',0)} | {q.get('REFERENCE_FAILED',0)} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['std'])} | {fmt(s['p95'])} | {fmt(s['min'])}–{fmt(s['max'])} |")
    lines+=['','`reference_valid_raw_run_distribution` ở mỗi mode và `raw_run_distribution` ở breakdown chứa thêm mean của tất cả lượt đo. Mean raw và mean của pair-median là hai phép tính khác nhau. `t_controller` có phần chồng lấp với probe/feature; không cộng nó vào tổng stage để suy E2E.','',
        '## 3. Speedup trên cùng tập QoR PASS','',
        '| Mode so với Fixed-Best | N cùng PASS | Tỷ số mean | Tỷ số median | Median tỷ số paired | Geometric mean paired |','|---|---:|---:|---:|---:|---:|']
    for name,m in original.items():
        s=m['speedup_vs_reference_joint_pass']
        lines.append(f"| {name} | {s['pairs']} | {fmt(s['ratio_of_means'])}× | {fmt(s['ratio_of_medians'])}× | {fmt(s['median_paired_speedup'])}× | {fmt(s['geometric_mean_paired_speedup'])}× |")
    s=original['v3_p80']['speedup_vs_reference_joint_pass'];ci=s['ratio_of_means_bootstrap95']
    lines+=['',f"P80 nhanh hơn trên {s['faster_pairs']}/{s['pairs']} pair cùng PASS, chậm hơn trên {s['slower_pairs']}. CI bootstrap 95% của tỷ số mean: [{fmt(ci[0])}, {fmt(ci[1])}]. Resample theo pair, 10.000 lần, seed 20261101. CI này không mô hình hóa đầy đủ tương quan scene hoặc drift phần cứng.",
        '','## 4. QoR thực đo','',
        'Các metric dưới đây gồm mọi candidate có metric hợp lệ, cả PASS và FAIL. Cột N cho biết số pair có metric; không nhân N lên theo repeat. PASS hiện là không giảm quá 0,02 SSIM và 0,5 dB PSNR, coverage ít nhất 0,95 so với reference. Reference không phải ground truth và PASS không bảo đảm SSIM tuyệt đối cao.','',
        '| Mode | N metric | Mean SSIM | Median SSIM | Mean PSNR (dB) | Mean coverage | Min coverage |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in original.items():
        q=m['qor_metrics_reference_valid'];ss=q['ssim_overlap'];ps=q['psnr_overlap'];cv=q['overlap_coverage']
        lines.append(f"| {name} | {ss['n']} | {fmt(ss['mean'],4)} | {fmt(ss['median'],4)} | {fmt(ps['mean'])} | {fmt(cv['mean'],6)} | {fmt(cv['min'],6)} |")
    drops=original['v3_p80']['qor_drop_reference_minus_candidate']
    lines+=['',f"P80 có một false accept: `udis_train_003317`. Mean SSIM drop {fmt(drops['ssim_overlap']['mean'],6)}, max drop {fmt(drops['ssim_overlap']['max'],6)}. Ở pair lỗi, PSNR tốt hơn reference nhưng SSIM giảm 0,087468, vượt ngưỡng 0,02. Không dùng mean gần nhau để bỏ qua pair FAIL.",
        '','## 5. Oracle trên đúng 50 pair mới','',
        'Đây là lượt đo đánh giá bổ sung, tách khỏi campaign gốc. Quét đủ 48 cấu hình và chạy lại v2/p80 xen kẽ theo batch, cùng binary/policy/seed; H/status/count của policy khớp campaign gốc. Không fit từ grid holdout.','',
        '| Mode, 49 reference-valid pair | Mean (ms) | Median (ms) | p95 (ms) |','|---|---:|---:|---:|']
    for name in ['reference','v2','v3_p80']:
        s=follow[name]['reference_valid_pair_median_distribution'];lines.append(f"| {name} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['p95'])} |")
    s=oracle['oracle_latency'];lines.append(f"| Oracle toàn grid | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['p95'])} |")
    s=oracle['reference_over_oracle']
    lines+=['',f"Fixed-Best/Oracle: tỷ số mean {fmt(s['ratio_of_means'])}×, tỷ số median {fmt(s['ratio_of_medians'])}× trên {s['pairs']} pair. Đây là lợi thế Oracle offline, chưa gồm chi phí tìm kiếm online. Reference failure còn lại không được đổi thành PASS.",
        '',f"Cấu hình ban đầu trùng Oracle: **{oracle['initial_exact_oracle']}/{oracle['oracle_available_pairs']}**. Cấu hình cuối trùng Oracle: **{oracle['final_exact_oracle']}/{oracle['oracle_available_pairs']}**.",'',
        '| Tham số cuối | Trùng Oracle | Mean độ lệch tuyệt đối | Median độ lệch tuyệt đối |','|---|---:|---:|---:|']
    for k in ['S','K','R']:
        d=oracle['absolute_parameter_delta'][k];lines.append(f"| {k} | {oracle['parameter_exact_counts'][k]}/{oracle['oracle_available_pairs']} | {fmt(d['mean'])} | {fmt(d['median'])} |")
    s=oracle['configuration_regret_ms'];d=oracle['mean_latency_decomposition_joint_pass'];restricted=oracle['oracle_restricted_to_policy']['latency']
    lines+=['',f"Trên 48 pair mà cấu hình cuối đạt QoR, chỉ {oracle['near_oracle_within_10percent_pairs']} cấu hình có latency không quá 10% so với Oracle. Chênh fixed cùng cấu hình so với Oracle: mean {fmt(s['mean'])} ms, median {fmt(s['median'])} ms, max {fmt(s['max'])} ms.",
        '',f"Chỉ {oracle['oracle_config_available_in_policy']}/49 lựa chọn Oracle nằm trong ladder p80. Oracle giới hạn đúng ladder đó có median {fmt(restricted['median'])} ms, so với {fmt(oracle['oracle_latency']['median'])} ms của toàn grid; đây vẫn là upper bound bỏ qua gate/probe/retry.",
        'Không chỉ bộ chọn ban đầu: 45/49 đáp án Oracle không có trong ladder đã khóa, nên bộ chọn hoàn hảo cũng không chọn đúng chúng. Trùng cấu hình không phải tiêu chí duy nhất; regret và số cấu hình nằm trong 10% Oracle được báo kèm để xét các trường hợp gần hòa latency.',
        '',f"Phân rã mean trên cùng 48 pair PASS: Oracle {fmt(d['oracle_ms'])} ms → chạy fixed cấu hình Controller đã chọn {fmt(d['chosen_configuration_fixed_ms'])} ms → Controller E2E {fmt(d['controller_e2e_ms'])} ms. Chênh cấu hình là {fmt(d['configuration_gap_ms'])} ms; chênh E2E so với fixed cùng cấu hình {fmt(d['e2e_minus_chosen_fixed_ms'])} ms. Phần sau gồm probe/retry/host và khác biệt lượt đo, không coi toàn bộ là một kernel overhead.",
        '','Các cấu hình Oracle chọn nhiều nhất:','', '| Cấu hình | Pair |','|---|---:|']
    for config,count in sorted(oracle['oracle_config_counts'].items(),key=lambda x:(-x[1],x[0]))[:8]:lines.append(f'| {config} | {count} |')
    q=oracle['oracle_qor_metrics']
    lines+=['',f"Oracle trên 49 pair: mean SSIM {fmt(q['ssim_overlap']['mean'],4)}, mean PSNR {fmt(q['psnr_overlap']['mean'])} dB, mean coverage {fmt(q['overlap_coverage']['mean'],6)}. Cả 49 lựa chọn thỏa predicate tương đối, không phải bảo đảm SSIM tuyệt đối cao.",
        '', '| Pair lệch lớn | Cấu hình cuối | Oracle | Fixed cấu hình đã chọn (ms) | Oracle (ms) | Controller E2E (ms) |',
        '|---|---|---|---:|---:|---:|']
    for r in sorted((r for r in oracle['ledger'] if r.get('configuration_regret_ms') is not None),key=lambda r:-r['configuration_regret_ms'])[:5]:
        lines.append(f"| {r['pair_id']} | {r['final_config']} | {r['oracle_config']} | {fmt(r['chosen_fixed_ms'])} | {fmt(r['oracle_ms'])} | {fmt(r['controller_e2e_ms'])} |")
    lines+=['','## 6. Ảnh đối chiếu','',
        'Ảnh B được warp vào hệ tọa độ A bằng H đã đo, dùng cùng geometry/interpolation của evaluator. Đây là ảnh alignment để đối chiếu QoR. Ô đen là ngoài miền ảnh B; heatmap là sai khác photometric, không phải SSIM map.',
        'Các preview được kiểm metric khớp CSV trong 1e-12; WebP lossless và PNG khôi phục có cùng hash pixels RGBA. Ảnh gốc A/B 512 × 512 cũng nằm cạnh preview.','']
    image_root='../../results/controller-v3-t4-20261104/oracle-checkpoint/images/'
    previews=json.loads((report_path.parent.parent/'oracle-checkpoint/images/previews.json').read_text())
    for case in previews['cases']:
        lines+=[f"### {case['category']}: {case['pair_id']}",'',
            f"Controller `{case['controller_config']}`; Oracle `{case['oracle_config']}`.",'',
            f"![{case['pair_id']}]({image_root+case['preview']})",'']
    base='../../results/controller-v3-t4-20261104/'+report_path.parent.name+'/'
    lines+=['## 7. Artifact và kiểm chứng','',
        f"- [JSON đầy đủ: distribution, QoR và CI]({base}detailed-statistics.json)",
        f"- [CSV latency từng pair, giữ cả FAIL/unknown]({base}latency-per-pair.csv)",
        f"- [CSV chín pair và stage breakdown]({base}nine-pair-breakdown.csv)",
        f"- [CSV cấu hình ban đầu/cuối, Oracle và regret]({base}controller-vs-oracle.csv)",
        '- Tool thống kê đã kiểm bằng fixture tỷ số paired cố định, bootstrap, infinity/missing và guard partition; toàn bộ Python suite 414 test PASS ở local.',
        '- Oracle giữ đầy đủ 24.000 measurement và 4.800 warmup; audit canonical giữ reference failure. Không tune lại policy và không đổi ngưỡng QoR.',
        '- CTest/CUDA của binary giữ nguyên từ campaign trước; lần bổ sung chỉ chạy đánh giá và xuất ảnh, không sửa kernel hoặc policy.','']
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.report,a.output)

if __name__=='__main__':main()
