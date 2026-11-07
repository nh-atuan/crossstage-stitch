"""Evidence-linked answers, including negative/missing results."""
from tools.controller.diagnostics.io import read_json,write_json,write_csv,sha,package
from .plots import render


def finish(context,results):
    root=context.output;render(root)
    statuses=[dict(phase=n,status=r['status'],reason=r.get('reason','')) for n,r in results.items()]
    write_csv(root/'phase-status.csv',statuses)
    answers=[]
    method=results.get('method',{})
    for comp in method.get('comparisons',[]):
        pass_ci=comp['PASS_delta']['ci95'];regret_ci=comp['regret_saved_ms']['ci95']
        verdict='INCONCLUSIVE'
        if pass_ci and regret_ci and pass_ci[0]>=0 and regret_ci[0]>0:verdict='OOF_IMPROVEMENT_DEVELOPMENT_ONLY'
        elif pass_ci and regret_ci and (pass_ci[1]<0 or regret_ci[1]<0):verdict='NO_IMPROVEMENT_UNDER_JOINT_CRITERIA'
        answers.append(dict(verdict=verdict,question='Objective PRE0 / ranker versus planner or direct48',comparison=comp,
            conclusion='Inspect paired PASS and regret CI jointly; no automatic win from point estimate.'))
    answers += [dict(question='Oracle gap with same PRE0',evidence='method/objective-ablation.csv; oracle_gap_closed uses FAIL-penalized denominator'),
        dict(question='NCC/texture/gradient/interactions contribution and overhead',evidence='features/ablation.csv + features/feature-cost.csv; native variants measure combined overhead; no online post-feature leakage'),
        dict(question='K500/R128 selection',evidence='method/top1-per-pair.csv, features/R-sensitivity.csv and objective-ablation.csv'),
        dict(question='Gate bottleneck',evidence='replay/gate-breakdown.csv; selected cheap PASS rejected is gate error, not selector failure'),
        dict(question='Too few training pairs?',evidence='learning/learning-curve.csv and figures/learning-curve.png; fixed capacity, identical outer groups; inconclusive if CI overlaps'),
        dict(question='E2E versus Reference/v4/v5',status=results.get('gpu',{}).get('status','NOT_MEASURED'),evidence='gpu/variants.csv and paired CIs, only when actual T4 native runs exist')]
    result=dict(status='PARTIAL' if any(r['status'] not in ['PREPARED','OFFLINE_EVALUATED','TRAINED','GPU_TESTED','TEST_FIXTURE'] for r in results.values()) else 'DEVELOPMENT_ONLY',
        phases=statuses,answers=answers,qualification='NOT_QUALIFIED',blind_holdout='NONE_CERTIFIED',default_changed=False,
        execution='TEST_FIXTURE' if context.test_fixture else 'REAL_PIPELINE',source=context.inventory['v5_source_revision'])
    write_json(root/'results-summary.json',result)
    lines=['# DCS-R1: kết quả và giới hạn','','DEVELOPMENT_ONLY; NOT_QUALIFIED cho generalization. Default không đổi.','',
        '| Phase | Status | Lý do |','|---|---|---|']
    lines += [f"| {r['phase']} | {r['status']} | {r['reason']} |" for r in statuses]
    summaries=method.get('summaries',[])
    gpu=results.get('gpu',{})
    if gpu.get('variants'):
        lines += ['','| Native variant | PASS/FAIL/UNKNOWN | E2E mean ms [CI95] | Speedup vs REF | Speedup vs v4 |','|---|---|---|---|---|']
        for r in gpu['variants']:lines.append(f"| {r['variant']} | {r['PASS']}/{r['FAIL']}/{r['UNKNOWN']} | {r['mean_e2e_ms']} {r.get('e2e_ms_ci95')} | {r['paired_reference_speedup']} | {r['paired_v4_speedup']} |")
        lines += ['', 'Paired v5 comparisons: gpu/paired-v5-comparison.csv. Cần quality và CI speedup so với cả ba baseline; đây vẫn là viewed evaluation.']
    if method.get('comparisons'):
        lines += ['','| So sánh PRE0 | Kết luận CI joint PASS/regret |','|---|---|']
        for answer in answers:
            if 'comparison' in answer:
                comp=answer['comparison'];lines.append(f"| {comp['left']} / {comp['right']} | {answer['verdict']} |")
    if summaries:
        lines += ['','| Method PRE0 | n | QoR FAIL | Top1 PASS | Regret có penalty (ms) | Oracle gap closed |','|---|---:|---:|---:|---:|---:|']
        for r in summaries:lines.append(f"| {r['method']} | {r['n']} | {r['FAIL']} | {r['top1_PASS']['mean']} | {r['penalized_regret_ms']['mean']} | {r['oracle_gap_closed']} |")
    lines += ['','Mỗi FAIL giữ trong denominator; penalty = thời gian config chọn + Reference nếu FAIL. Không dùng mean chỉ trên PASS để tuyên bố thắng.',
        '', 'Xem method/ranking-comparison.csv để đối chiếu CI paired. Rank score không phải xác suất PASS; perfect selector/post chỉ là counterfactual offline.',
        '', 'Feature post chỉ ở continuation sau source attempt đã quan sát. NCC/interaction overhead chưa đo thì không tuyên bố có lợi E2E.',
        '', 'Learning curve có cùng outer evaluation, 3 sampling seeds, capacity cố định; không suy số pair cần thiết chỉ bằng extrapolation.',
        '', 'E2E phải có GPU_TESTED/MEASURED_GPU, QoR và paired CI cùng cohort/source. Không có fresh blind cohort thì không claim generalization.']
    (root/'README-results.md').write_text('\n'.join(lines)+'\n')
    missing=[r for r in statuses if r['status'].startswith('BLOCKED') or r['status']=='PARTIAL']
    if missing:(root/'MISSING_REQUIRED_INPUTS.md').write_text('# Đầu vào còn thiếu\n\n'+'\n\n'.join(r['phase']+': '+r['reason'] for r in missing)+'\n\nĐặt full P0 ZIP vào MyDrive/controller-score-results/ và ZIP v5 gốc vào MyDrive/controller-v5-results/. Không chạy lại GPU sweep.\n')
    write_json(root/'manifest-sha256.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ['manifest-sha256.json','artifact-hashes.json'] and p.suffix!='.partial'})
    destination=package(root);print('DCS RESULT ZIP:',destination,flush=True);return destination
