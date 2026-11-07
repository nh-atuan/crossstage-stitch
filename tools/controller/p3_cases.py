"""Export evidence for rejected and recovered pairs without assigning visual causes."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import cv2
import numpy as np

from tools.controller.p3_io import read_csv, write_json


def export(campaign: Path, manifest: Path, output: Path) -> dict:
    pairs = json.loads(manifest.read_text())
    runs = {r['pair_id']: r for r in read_csv(campaign / 'adaptive.csv', {'pair_id','repeat_idx','is_warmup'})
            if r['repeat_idx'] == '0' and r['is_warmup'] == '0'}
    trace = defaultdict(list)
    for row in read_csv(campaign / 'adaptive-attempts.csv', {'pair_id','attempt_idx','repeat_idx','is_warmup'}):
        if row['repeat_idx'] == '0' and row['is_warmup'] == '0':
            trace[row['pair_id']].append(row)
    ledger = []
    for pair, run in sorted(runs.items()):
        attempts = sorted(trace[pair], key=lambda r: int(r['attempt_idx']))
        if not attempts:
            raise ValueError('missing case trace')
        ledger.append(dict(pair_id=pair, status=run['status'], attempts=len(attempts),
                           initial_inliers=int(attempts[0]['num_inliers']),
                           final_inliers=int(attempts[-1]['num_inliers']),
                           recovered_inliers=int(attempts[-1]['num_inliers'])-int(attempts[0]['num_inliers']),
                           first_reason=attempts[0]['reason'], final_reason=attempts[-1]['reason']))
    failed = [r for r in ledger if r['status'] != 'OK'][:3]
    recovered = sorted((r for r in ledger if r['status']=='OK' and r['attempts']>1),
                       key=lambda r: (-r['recovered_inliers'],r['pair_id']))[:3]
    selected = failed + recovered
    output.mkdir(parents=True, exist_ok=True)
    with (output/'case-ledger.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    if selected:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(selected), 3, figsize=(12, 2.7*len(selected)), squeeze=False)
        for row, case in enumerate(selected):
            pair=case['pair_id']; info=pairs[pair]
            a=cv2.imread(str(manifest.parent/info['a']));b=cv2.imread(str(manifest.parent/info['b']))
            h=np.array([float(runs[pair][f'h{i}']) for i in range(9)]).reshape(3,3)
            axes[row,0].imshow(cv2.cvtColor(a,cv2.COLOR_BGR2RGB))
            axes[row,1].imshow(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))
            axes[row,0].set_title(f"{pair}: {case['status']}; attempts={case['attempts']}",fontsize=9)
            axes[row,1].set_title(f"inliers {case['initial_inliers']} to {case['final_inliers']}",fontsize=9)
            if np.isfinite(h).all() and np.linalg.matrix_rank(h)==3:
                warped=cv2.warpPerspective(b,h,(a.shape[1],a.shape[0]))
                axes[row,2].imshow(cv2.cvtColor(cv2.addWeighted(a,.5,warped,.5,0),cv2.COLOR_BGR2RGB))
                axes[row,2].set_title('Diagnostic overlay; no causal label',fontsize=9)
            for axis in axes[row]:axis.axis('off')
        fig.tight_layout()
        fig.savefig(output/'case-review.png',dpi=120)
        plt.close(fig)
    result=dict(selected=selected, cohort_pairs=len(ledger),
                limits='Descriptive trace and image evidence; no automatic parallax annotation.')
    write_json(output/'case-review.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.campaign,args.manifest,args.output)))
