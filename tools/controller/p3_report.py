"""Generate Phase 3 summaries, publication tables and vector figures from raw measurements."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from tools.controller.p3_ablation import analyze
from tools.controller.p3_extract_a2 import extract
from tools.controller.p3_evidence import matching_campaigns, matching_retry_paths, measurement_protocol
from tools.controller.p3_geometry import compare, evaluate
from tools.controller.p3_io import measurements, sha256, write_json
from tools.controller.p3_telemetry import summarize
from tools.controller.p3_plots import export_distributions


def check_reuse_equivalence(left: Path, right: Path, repeats: int = 10, *, same_run: bool = True) -> None:
    a, b = measurements(left, repeats), measurements(right, repeats)
    if set(a) != set(b):
        raise ValueError('reuse variants have different cohorts')
    for pair in a:
        for x, y in zip(a[pair], b[pair]):
            for field in ('run_id', 'git_rev', 'dataset', 'split', 'seed', 'repeat_idx',
                          'config_id', 'status', 'fail_stage', 'num_matches', 'num_inliers'):
                if field == 'run_id' and not same_run:
                    continue
                if x[field] != y[field]:
                    raise ValueError(f'reuse changes {field} for {pair}')
            if x['status'] == 'OK' or x['fail_stage'].lower() == 's6':
                ha = np.array([float(x[f'h{i}']) for i in range(9)])
                hb = np.array([float(y[f'h{i}']) for i in range(9)])
                if not np.allclose(ha, hb, rtol=1e-9, atol=1e-9):
                    raise ValueError('reuse changes homography')


def build(udis: Path, hp: Path, hp_manifest: Path, output: Path, *, a2_campaign: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    study = a2_campaign if a2_campaign is not None else udis
    matching_campaigns(udis, study)
    if study.resolve() != udis.resolve():
        check_reuse_equivalence(udis / 'adaptive.csv', study / 'adaptive.csv', same_run=False)
    check_reuse_equivalence(study / 'adaptive.csv', study / 'no_reuse.csv')
    matching_retry_paths(study / 'adaptive-attempts.csv', study / 'no_reuse-attempts.csv')
    controller = summarize(udis / 'adaptive.csv', traces=udis / 'adaptive-attempts.csv')
    fixed = summarize(udis / 'fixed.csv')
    if {r['pair_id'] for r in controller['ledger']} != {r['pair_id'] for r in fixed['ledger']}:
        raise ValueError('fixed/adaptive cohort mismatch')
    a2_csv = output / 'a2-retries.csv'
    try:
        extract(study, a2_csv)
    except ValueError as error:
        if str(error) != 'no actual R-only retries; A2 cohort is empty':
            raise
        a2 = dict(status='empty_cohort', pairs=0, transitions=0, ledger=[])
    else:
        a2 = analyze(a2_csv)
        a2['status'] = 'measured'
    hp_fixed = evaluate(hp / 'fixed.csv', hp_manifest)
    hp_adaptive = evaluate(hp / 'adaptive.csv', hp_manifest)
    comparison = compare(hp_fixed, hp_adaptive)
    lookup = {r['pair_id']: r for r in hp_fixed['ledger']}
    shared = [r for r in hp_adaptive['ledger'] if r['mce_px'] is not None and lookup[r['pair_id']]['mce_px'] is not None]
    comparison['paired_fixed_mean_px'] = statistics.mean(lookup[r['pair_id']]['mce_px'] for r in shared) if shared else None
    comparison['paired_adaptive_mean_px'] = statistics.mean(r['mce_px'] for r in shared) if shared else None
    report = dict(controller=controller, fixed=fixed, a2=a2,
                  hpatches_fixed=hp_fixed, hpatches_adaptive=hp_adaptive,
                  geometry_comparison=comparison, reuse_equivalence='passed_all_pairs_all_repeats',
                  protocol=measurement_protocol(udis / 'adaptive.csv'))
    if measurement_protocol(udis / 'fixed.csv') != report['protocol']:
        raise ValueError('fixed/adaptive measurement protocol mismatch')
    write_json(output / 'summary.json', report)
    for name, data in report.items():
        if isinstance(data, dict):
            write_json(output / f'{name}.json', data)
    export(report, output)
    return report


def export(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c, f, a = report['controller'], report['fixed'], report['a2']
    hpf, hpa = report['hpatches_fixed'], report['hpatches_adaptive']
    ok = lambda data: sum(r['pairs'] for r in data['status_counts'] if r['status'] == 'OK')
    retry = c['retry']
    protocol = report['protocol']
    fmt = lambda value: '--' if value is None else f'{value:.3f}'
    lines = [r'\subsection{Controller Behaviour and Match Reuse}',
             'The frozen policy was evaluated on a Tesla T4 with CUDA targeting '
             r'\texttt{sm\_75}, '
             f"seed {protocol['seed']}, {protocol['warmups']} warmups and {protocol['repeats']} measured repeats per pair. "
             'We take repeat medians before aggregating over pairs. '
             'These execution statistics do not replace the offline overlap-quality constraint.',
             r'\begin{center}\begin{tabular}{lrr}\hline',
             r'UDIS-D execution & Fixed & Adaptive \\\hline',
             f"Input pairs & {f['pairs']} & {c['pairs']} " + r'\\',
             f"Completed panoramas & {ok(f)} & {ok(c)} " + r'\\',
             f"Median latency (ms), all pairs & {f['pair_median_e2e_ms']['median']:.3f} & {c['pair_median_e2e_ms']['median']:.3f} " + r'\\',
             r'\hline\end{tabular}\end{center}',
             f"Adaptive retried {retry['pairs_with_retry']} of {c['pairs']} pairs "
             f"({100*retry['pair_retry_rate']:.2f}\\%). The mean attempt count was "
             f"{retry['attempts_per_pair']['mean']:.3f}; the mean final-minus-initial inlier count was "
             f"{retry['final_minus_initial_inliers']['mean']:.3f} over all pairs, including zero changes.",
             'A2 reruns matching and the complete RANSAC search for an actual R-only retry, '
             'while keeping preallocated buffers. Allocation cost is excluded in both variants. '
             'All final statuses, match/inlier counts and valid homographies agree between variants.']
    if a['status'] == 'measured':
        median_reuse = statistics.median(r['reuse_retry_ms'] for r in a['ledger'])
        median_fresh = statistics.median(r['no_reuse_retry_ms'] for r in a['ledger'])
        lines += [f"The conditional A2 cohort contained {a['pairs']} pairs and {a['transitions']} transitions. "
                  f"Median retry time was {median_reuse:.4f} ms with reuse and {median_fresh:.4f} ms without reuse. "
                  f"The median paired saving was {a['transition_saved_percent']['median']:.2f}\\%. "
                  'This is a component comparison on the retry cohort, not a whole-dataset speedup.']
        lines += [r'\begin{center}\begin{tabular}{lrr}\hline',
                  r'A2 conditional retry & Reuse & Recompute \\\hline',
                  f"Median host retry (ms) & {median_reuse:.4f} & {median_fresh:.4f} " + r'\\',
                  f"Median CUDA S4+S5 (ms) & {statistics.median(r['reuse_s4_s5_ms'] for r in a['ledger']):.4f} & "
                  f"{statistics.median(r['no_reuse_s4_s5_ms'] for r in a['ledger']):.4f} " + r'\\',
                  r'\hline\end{tabular}\end{center}']
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        ax.bar(['Reuse', 'Recompute S4/S5'], [median_reuse, median_fresh], color=['#247881','#b85c38'])
        ax.set_ylabel('Median retry time (ms)')
        fig.tight_layout()
        fig.savefig(output / 'a2-retry.pdf', metadata={'Author': 'Anonymous Authors'})
        fig.savefig(output / 'a2-retry.svg')
        plt.close(fig)
    else:
        lines += ['No actual R-only retry occurred, so A2 has no latency estimate for this policy.']
    lines += [r'\paragraph{Independent geometry check.}',
              f"HPatches uses {len({row['sequence'] for row in hpf['ledger']})} viewpoint sequences "
              f"({hpf['pairs']} pairs) at original resolution. "
              'We compare estimated B-to-A homographies with the inverse supplied ground truth at '
              'the four B-image corner pixel centres. Renderer failures are counted separately '
              'from valid S5 estimates.',
              r'\begin{center}\begin{tabular}{lrr}\hline',
              r'HPatches & Fixed & Adaptive \\\hline',
              f"Retained homographies & {hpf['valid_pairs']} & {hpa['valid_pairs']} " + r'\\',
              f"Completed panoramas & {hpf['pipeline_success_pairs']} & {hpa['pipeline_success_pairs']} " + r'\\',
              f"Median MCE (px), valid subset & {fmt(hpf['pair_mce_px']['median'])} & {fmt(hpa['pair_mce_px']['median'])} " + r'\\',
              f"p95 MCE (px), valid subset & {fmt(hpf['pair_mce_px']['p95'])} & {fmt(hpa['pair_mce_px']['p95'])} " + r'\\',
              r'\hline\end{tabular}\end{center}']
    comparison = report['geometry_comparison']
    if comparison['shared_valid_pairs']:
        lines += [f"On the {comparison['shared_valid_pairs']} jointly valid pairs, mean MCE was "
                  f"{comparison['paired_fixed_mean_px']:.3f} px for Fixed and "
                  f"{comparison['paired_adaptive_mean_px']:.3f} px for Adaptive. "
                  'A successful self-check alone does not establish geometric accuracy.']
    (output / 'controller-results.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    macros = dict(ControllerPairs=c['pairs'], ControllerCompleted=ok(c), FixedCompleted=ok(f),
                  ControllerRetried=retry['pairs_with_retry'],
                  ControllerRetryPercent=f"{100*retry['pair_retry_rate']:.2f}",
                  ControllerMedianMs=fmt(c['pair_median_e2e_ms']['median']),
                  FixedMedianMs=fmt(f['pair_median_e2e_ms']['median']),
                  ControllerBudgetRejected=sum(r['pairs'] for r in c['status_counts'] if r['status']=='BUDGET_EXHAUSTED'),
                  GeometryPairs=hpf['pairs'], GeometryFixedRetained=hpf['valid_pairs'],
                  GeometryAdaptiveRetained=hpa['valid_pairs'], GeometryJointPairs=comparison['shared_valid_pairs'],
                  GeometryFixedCompleted=hpf['pipeline_success_pairs'],
                  GeometryAdaptiveCompleted=hpa['pipeline_success_pairs'], ATwoPairs=a['pairs'],
                  GeometryPairedDelta=fmt(comparison['paired_delta_second_minus_first_px']['mean']))
    if a['status'] == 'measured':
        macros['ATwoSavingPercent'] = f"{a['transition_saved_percent']['median']:.2f}"
    (output / 'controller-summary-macros.tex').write_text(
        '\n'.join('\\newcommand{\\'+key+'}{'+str(value)+'}' for key,value in macros.items())+'\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    configs = c['final_config_counts']
    labels = [label.replace('_Bfeather', '').replace('_', '\n') for label in configs]
    ax.bar(labels, list(configs.values()), color='#247881')
    ax.set_ylabel('Pairs')
    ax.set_xlabel('Final configuration')
    fig.tight_layout()
    fig.savefig(output / 'controller-ladder.pdf', metadata={'Author': 'Anonymous Authors'})
    fig.savefig(output / 'controller-ladder.svg')
    plt.close(fig)
    export_distributions(report, output)
    write_json(output / 'generation.json', dict(inputs={name: data.get('inputs_sha256') for name, data in report.items()
                                                      if isinstance(data, dict)},
                                               formatter_sha256={p.name: sha256(p) for p in (
                                                   Path(__file__), Path(__file__).with_name('p3_evidence.py'),
                                                   Path(__file__).with_name('p3_plots.py'))},
                                               outputs={p.name: sha256(p) for p in output.iterdir()
                                                        if p.suffix in ('.pdf','.svg','.tex')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('udis','hpatches','hpatches-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--a2-campaign', type=Path)
    args = parser.parse_args()
    build(args.udis,args.hpatches,args.hpatches_manifest,args.output,a2_campaign=args.a2_campaign)
