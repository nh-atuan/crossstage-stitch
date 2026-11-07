"""Standalone plots retain sample count, failures and descriptive-only labels."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.controller.diagnostics.io import read_json,read_csv


def render(root):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    kind=read_json(root/'run-receipt.json').get('kind','REAL_PIPELINE') if (root/'run-receipt.json').is_file() else 'REAL_PIPELINE'
    def save(name):
        plt.tight_layout();plt.savefig(out/(name+'.png'),dpi=160);plt.savefig(out/(name+'.pdf'));plt.close()
    path=root/'method/objective-ablation.csv'
    if path.is_file():
        rows=read_csv(path);names=[r['method'] for r in rows];n=int(rows[0]['n']);fig,axes=plt.subplots(1,2,figsize=(12,4))
        for axis,field,title in [(axes[0],'top1_PASS','Top-1 QoR PASS fraction'),(axes[1],'penalized_regret_ms','Penalized regret (ms)')]:
            values=[json.loads(r[field]) for r in rows];means=[v['mean'] for v in values]
            axis.bar(range(len(rows)),means)
            if all(v.get('ci95') is not None for v in values):axis.errorbar(range(len(rows)),means,yerr=[[m-v['ci95'][0] for m,v in zip(means,values)],[v['ci95'][1]-m for m,v in zip(means,values)]],fmt='none',color='black',capsize=3)
            axis.set_xticks(range(len(rows)),names,rotation=50,ha='right');axis.set_title(title)
        fig.suptitle(f'{kind}: objective only, PRE0; n={n} grouped OOF queries; development only');save('objective-PRE0')
    path=root/'learning/learning-curve.csv'
    if path.is_file() and 'method' in read_csv(path)[0]:
        rows=read_csv(path);fig,axes=plt.subplots(1,2,figsize=(11,4))
        for method in sorted({r['method'] for r in rows}):
            rs=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['training_groups']))
            for ax,field,title in [(axes[0],'top1_PASS','Top-1 PASS fraction'),(axes[1],'penalized_regret_ms','Penalized regret (ms)')]:
                stats=[json.loads(r[field]) for r in rs];xs=[int(r['training_groups']) for r in rs];ax.plot(xs,[s['mean'] for s in stats],marker='o',label=method)
                # Helper reports CI under ci95; support both current and older metric schemas.
                ci=[s.get('ci95',s.get('ci')) for s in stats]
                if all(c is not None for c in ci):ax.fill_between(xs,[c[0] for c in ci],[c[1] for c in ci],alpha=.15)
                ax.set_xlabel('Requested training components (capped counts in CSV)');ax.set_ylabel(title);ax.legend(fontsize=7)
        fig.suptitle(f"{kind}: learning curve; n={rows[0]['n']} outer queries; fixed capacity; component CI95");save('learning-curve')
    path=root/'features/ablation.csv'
    if path.is_file():
        rows=read_csv(path);values=[json.loads(r['penalized_regret_ms']) for r in rows];means=[v['mean'] for v in values]
        plt.figure(figsize=(10,4));plt.bar(range(len(rows)),means)
        if all(v.get('ci95') is not None for v in values):plt.errorbar(range(len(rows)),means,yerr=[[m-v['ci95'][0] for m,v in zip(means,values)],[v['ci95'][1]-m for m,v in zip(means,values)]],fmt='none',color='black',capsize=3)
        plt.xticks(range(len(rows)),[r['feature'] for r in rows],rotation=30);plt.ylabel('Penalized regret (ms), CI95');plt.title(f"{kind}: feature ablation; n={rows[0]['n']} grouped OOF; overhead measured separately");save('feature-ablation')
    path=root/'replay/gate-breakdown.csv'
    if path.is_file():
        rows=read_csv(path);rows=[r for r in rows if r['feature']=='F0' and r['method'].startswith('DCS_')]
        if rows:
            plt.figure(figsize=(12,4));plt.bar(range(len(rows)),[int(r['rejected_but_PASS']) for r in rows]);plt.xticks(range(len(rows)),[r['method']+' / '+r['gate'] for r in rows],rotation=65,ha='right')
            plt.ylabel('Selected cheap PASS rejected by gate (pairs)');plt.title('Selector success blocked by gate; grouped development replay');save('gate-bottleneck')
    path=root/'gpu/results.json'
    if path.is_file():
        result=read_json(path);rows=result.get('variants',[])
        if rows:
            plt.figure(figsize=(10,4));plt.bar(range(len(rows)),[r['mean_e2e_ms'] or 0 for r in rows]);
            if all(r.get('e2e_ms_ci95') for r in rows):plt.errorbar(range(len(rows)),[r['mean_e2e_ms'] for r in rows],yerr=[[r['mean_e2e_ms']-r['e2e_ms_ci95'][0] for r in rows],[r['e2e_ms_ci95'][1]-r['mean_e2e_ms'] for r in rows]],fmt='none',color='black',capsize=3)
            plt.xticks(range(len(rows)),[r['variant'] for r in rows],rotation=45,ha='right');plt.ylabel('Mean of pair median E2E (ms)')
            plt.title(f"{result['measurement']}; viewed evaluation only; FAIL/UNKNOWN retained in CSV");save('native-e2e')
