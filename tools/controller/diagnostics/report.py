"""Evidence-linked Vietnamese conclusions; INCONCLUSIVE when a test is missing."""
from .io import write_json,write_csv


def number(value):return 'chưa có' if value is None else f'{value:.4f}'


def conclude(results,inventory,protocol,output):
    status=[dict(experiment=k,status=results.get(k,{}).get('status','BLOCKED'),
        reason=results.get(k,{}).get('reason',''),pairs=inventory.get('grid',{}).get('reference_valid',0)) for k in 'ABCDE']
    write_csv(output/'experiment-status.csv',status)
    conclusions=[]
    a=results.get('A',{})
    ranking=[r for r in a.get('ranking',[]) if r['scope']=='calibration_oof' and r['ranking']=='pre_probability' and r['top_k']==3 and r['threshold']==.85]
    k500=[r for r in a.get('classification',[]) if r['scope']=='calibration_oof' and r['slice']=='K=500']
    if ranking and k500:
        row=ranking[0];metrics=k500[0]
        conclusions.append(dict(question='Pre-model, threshold hay score?',status='DIAGNOSTIC',
            evidence=f"K500 OOF ROC-AUC={number(metrics['roc_auc'])}; AP={number(metrics['pr_auc_average_precision'])}; cheap-feasible recall tại 0.85={number(row['cheap_feasible_recall']['mean'])}; top-3 near-Oracle recall={number(row['top_k_near_oracle_recall']['mean'])}. Oracle bị filter ở {row['oracle_filtered_pairs']} pair; giữ nhưng không chọn ở {row['oracle_retained_not_selected_pairs']} pair.",
            source='A/pre_diagnostics.json; A/ranking-per-pair.csv'))
    else:conclusions.append(dict(question='Pre-model, threshold hay score?',status='INCONCLUSIVE',evidence='Thiếu full grid hoặc prediction OOF hợp lệ.',source='data_inventory.json'))
    b=results.get('B',{});cal=[r for r in b.get('metrics',[]) if r['scope']=='calibration_oof' and r['slice']=='all']
    if cal:
        text=[]
        for stage in ['pre','post']:
            rows=[r for r in cal if r['stage']==stage]
            text.extend(f"{stage}/{r['variant']}: Brier={number(r['brier']['mean'])}, ECE={number(r['ece'])}, group={r['pairs_or_components']}" for r in rows)
            if stage=='post':
                for r in rows:
                    if r['variant']=='stored_probability_calibration':
                        text.extend(f"gate {t['threshold']}: FP={t['fp']}, FN={t['fn']}, TP={t['tp']}, TN={t['tn']} (valid payloads, mọi candidate)"
                            for t in r['confusion'] if t['threshold'] in protocol.gate_thresholds)
        conclusions.append(dict(question='Probability calibration và gate?',status='DIAGNOSTIC',evidence='; '.join(text)+'. Confusion tại gate 0.65/0.85 có trong bảng; không chọn ngưỡng bằng evaluation đã xem.',source='B/probability-calibration.json'))
    else:conclusions.append(dict(question='Probability calibration và gate?',status='INCONCLUSIVE',evidence='Chỉ có route đã chọn hoặc thiếu OOF; không thay bằng accuracy trên tập fit.',source='B/probability-calibration.json'))
    c=[r for r in results.get('C',{}).get('comparisons',[]) if r['scope']=='calibration_oof' and r['policy']=='P0']
    d=results.get('D',{})
    post_rows=[r for r in (d.get('calibration_comparisons') or []) if r['scope']=='calibration_oof' and r['policy']=='P0']
    post_text='; '.join(f"post/{r['variant']}: PASS={r['QoR_PASS']}/{r['pairs']}, false accept={r['false_accept']}, false reject={r['false_reject_alignment']}, cheap-stop PASS={r['cheap_stop_pass']}, retry={r['retry_pairs']}, full-recompute={number(r['full_recompute_ms']['mean'])} ms" for r in post_rows)
    post_text='; '+post_text if post_text else '; perfect-post INCONCLUSIVE'
    if c:
        text='; '.join(f"{r['variant']}: PASS={r['QoR_PASS']}/{r['pairs']}, retry={r['retry_pairs']}, scenario full-recompute={number(r['full_recompute_ms']['mean'])} ms" for r in c)
        conclusions.append(dict(question='Perfect pre/post cải thiện tối đa bao nhiêu?',status='ESTIMATED',evidence=text+post_text+'. Perfect-post giữ initial route; xem D. Đây là counterfactual offline, không phải model deploy hay speedup đo thật.',source='C/perfect-pre.json; D/perfect-post.json'))
    else:conclusions.append(dict(question='Perfect pre/post cải thiện tối đa bao nhiêu?',status='INCONCLUSIVE',evidence='Không đủ grid; D chỉ có thể báo thay đổi gate trên route đã đo, timing nhánh mới UNKNOWN.',source='data_inventory.json; D/perfect-post.json'))
    e=results.get('E',{});gain=e.get('paired_smallest_to_largest_brier_gain',{}).get('pre_brier',{})
    curve=e.get('curve',[])
    learning_support=bool(gain.get('mean') is not None and gain['mean']>=protocol.material_brier_gain and gain.get('ci95') and gain['ci95'][0]>0)
    if len(curve)>=2:
        interpretation='Có bằng chứng ưu tiên thêm pair độc lập theo tiêu chí pilot.' if learning_support else 'Chưa đủ bằng chứng kết luận thiếu pair hoặc plateau; không suy rằng XGBoost vô dụng.'
        conclusions.append(dict(question='Có thiếu pair huấn luyện?',status='DIAGNOSTIC' if learning_support else 'INCONCLUSIVE',
            evidence=f"Support K500 fit/probcal: {e.get('K500_positive_support',{})}. Cỡ group {curve[0]['training_groups']} → {curve[-1]['training_groups']}; Brier gain pre={number(gain.get('mean'))}, CI95={gain.get('ci95')}. "+interpretation,
            source='E/learning-curve.json; E/fit-partitions.csv'))
    else:conclusions.append(dict(question='Có thiếu pair huấn luyện?',status='INCONCLUSIVE',evidence='Learning curve chưa có ít nhất hai cỡ train với evaluation groups cố định.',source='E/learning-curve.json'))
    if learning_support:
        priority='more independent pairs';next_test='Thu thêm component training đa dạng theo protocol khóa trước; giữ outer evaluation và budget/model capacity để so đường học, không thêm repeat/config cùng ảnh thay cho pair.'
    elif any(r['stage']=='post' and r['variant']=='stored_probability_calibration' and (r['ece'] or 0)>protocol.material_ece for r in cal):
        priority='recalibration / post-gate diagnosis';next_test='Trên cùng outer groups, so margin model và các calibrator với probability-calibration partition riêng; giữ feature/capacity. Chỉ quyết định trên OOF, không tune vào bốn case evaluation.'
    elif k500 and k500[0]['roc_auc'] is not None and k500[0]['roc_auc']<protocol.weak_auc:
        priority='better pre information, chưa chứng minh nguyên nhân';next_test='Phân tích phụ ngoài fold: so feature photo hiện có với pair-relation/NCC đã ghi trong archive, giữ cùng capacity/split và tính chi phí riêng; chưa thêm kernel hoặc triển khai v5.'
    else:
        priority='INCONCLUSIVE: ranking / score / retry cần đối chứng';next_test='Giữ model và initial candidate, đo transition theo S/K/R trên calibration riêng nếu Nhóm duyệt; kiểm expected reject cost. Không dùng timing cố định như timing retry thật.'
    conclusions.append(dict(question='Ưu tiên và phép thử phân biệt tiếp theo?',status='PROPOSAL_ONLY',evidence=priority+'. '+next_test,source='A–E; tiêu chí trong protocol.json'))
    write_json(output/'conclusions.json',conclusions)
    lines=['# Chẩn đoán Controller A–E','', 'Kết quả offline trên dữ liệu được kiểm kê; không sửa Controller mặc định hoặc triển khai v5.','',
        '| Thí nghiệm | Trạng thái | Pair calibration hợp lệ | Lý do |','|---|---|---:|---|']
    lines += [f"| {r['experiment']} | {r['status']} | {r['pairs']} | {r['reason']} |" for r in status]
    for row in conclusions:lines += ['',f"## {row['question']}",'',row['status']+': '+row['evidence'],'','Nguồn: `'+row['source']+'`.']
    lines += ['','## Giới hạn','',
        '50 pair P0–P5 đã xem chỉ dùng chẩn đoán mô tả; không fit/chọn hyperparameter hoặc threshold bằng chúng. Calibration final-model là in-sample; OOF được ghi riêng.',
        'Reference failure/unknown và post-feature invalid giữ denominator riêng. Không coi 48 cấu hình của một pair là 48 ảnh độc lập.',
        'Scenario latency là ESTIMATED, không phải measured GPU E2E hoặc upper bound timing được chứng minh. Không giả lập post-feature của config chưa chạy.',
        'Nếu thiếu artifact, xem data_inventory.json; notebook không tải dataset hoặc chạy lại CUDA sweep.']
    (output/'conclusions.md').write_text('\n'.join(lines)+'\n')
    return status,conclusions
