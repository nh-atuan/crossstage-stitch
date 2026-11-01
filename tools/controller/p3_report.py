"""Generate Phase 3 summaries, publication tables and vector figures from raw measurements."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from tools.controller.p3_ablation import analyze
from tools.controller.p3_extract_a2 import extract
from tools.controller.p3_geometry import compare, evaluate
from tools.controller.p3_io import measurements, sha256, write_json
from tools.controller.p3_telemetry import summarize


def check_reuse_equivalence(left: Path, right: Path, repeats: int = 10) -> None:
    a, b = measurements(left, repeats), measurements(right, repeats)
    if set(a) != set(b):
        raise ValueError('reuse variants have different cohorts')
    for pair in a:
        for x, y in zip(a[pair], b[pair]):
            for field in ('run_id', 'git_rev', 'dataset', 'split', 'seed', 'repeat_idx',
                          'config_id', 'status', 'fail_stage', 'num_matches', 'num_inliers'):
                if x[field] != y[field]:
                    raise ValueError(f'reuse changes {field} for {pair}')
            if x['status'] == 'OK' or x['fail_stage'].lower() == 's6':
                ha = np.array([float(x[f'h{i}']) for i in range(9)])
                hb = np.array([float(y[f'h{i}']) for i in range(9)])
                if not np.allclose(ha, hb, rtol=1e-9, atol=1e-9):
                    raise ValueError('reuse changes homography')


def build(udis: Path, hp: Path, hp_manifest: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    check_reuse_equivalence(udis / 'adaptive.csv', udis / 'no_reuse.csv')
    controller = summarize(udis / 'adaptive.csv', traces=udis / 'adaptive-attempts.csv')
    fixed = summarize(udis / 'fixed.csv')
    a2_csv = output / 'a2-retries.csv'
    try:
        extract(udis, a2_csv)
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
                  geometry_comparison=comparison, reuse_equivalence='passed_all_pairs_all_repeats')
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
    fmt = lambda value: '--' if value is None else f'{value:.3f}'
    lines = [r'\subsection{Controller Behaviour and Match Reuse}',
             'The frozen policy was evaluated on a Tesla T4 with CUDA targeting '
             r'\texttt{sm\_75}, seed 20261101, two warmups and ten measured repeats per pair. '
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
              'HPatches uses all 59 viewpoint sequences (295 pairs) at original resolution. '
              'We compare estimated B-to-A homographies with the inverse supplied ground truth at '
              'the four B-image corner pixel centres. Renderer failures are counted separately '
              'from valid S5 estimates.',
              r'\begin{center}\begin{tabular}{lrr}\hline',
              r'HPatches & Fixed & Adaptive \\\hline',
              f"Valid geometry pairs & {hpf['valid_pairs']} & {hpa['valid_pairs']} " + r'\\',
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
    write_json(output / 'generation.json', dict(inputs={name: data.get('inputs_sha256') for name, data in report.items()
                                                      if isinstance(data, dict)},
                                               outputs={p.name: sha256(p) for p in output.iterdir()
                                                        if p.suffix in ('.pdf','.svg','.tex')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('udis','hpatches','hpatches-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    build(args.udis,args.hpatches,args.hpatches_manifest,args.output)
