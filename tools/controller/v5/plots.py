"""Scientific figures from emitted tables, with explicit measurement labels."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.controller.diagnostics.plots import save
from tools.controller.diagnostics.io import read_csv,read_json


def render(output):
    output=Path(output)
    if (output/'W1/counterfactual.json').is_file():
        rows=[r for r in read_json(output/'W1/counterfactual.json')['comparisons'] if r['policy']=='P0']
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(rows))
        ax.bar(x-.18,[r['optimistic_ms']['mean'] for r in rows],.36,label='Stage-only scenario')
        ax.bar(x+.18,[r['full_recompute_ms']['mean'] for r in rows],.36,label='Full-recompute scenario')
        ax.set_xticks(x,[r['variant'] for r in rows],rotation=20);ax.legend();ax.set(ylabel='Estimated ms',title='W1 OFFLINE_ORACLE / ESTIMATED_REPLAY')
        save(fig,output/'W1/figures/counterfactual')
    if (output/'W2/fixed-config-ranking.csv').is_file():
        rows=read_csv(output/'W2/fixed-config-ranking.csv');fig,ax=plt.subplots(figsize=(8,4))
        keys=['S0.50_K500_R128_Bfeather','S0.75_K500_R128_Bfeather']
        for offset,name in [(-.18,'PRE0'),(.18,'PRE1')]:
            selected={r['config_id']:r for r in rows if r['model']==name}
            values=[float(selected[k]['roc_auc']) if k in selected and selected[k]['roc_auc'] else np.nan for k in keys]
            ax.bar(np.arange(2)+offset,values,.36,label=name)
        ax.set_xticks(range(2),keys);ax.set(ylim=(0,1),ylabel='ROC-AUC',title='W2 fixed-config grouped development OOF');ax.legend()
        save(fig,output/'W2/figures/fixed-config-auc')
    if (output/'W3/result.json').is_file() and read_json(output/'W3/result.json').get('summary'):
        rows=read_json(output/'W3/result.json')['summary'];fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(rows))
        ax.bar(x-.18,[r['false_accept'] for r in rows],.36,label='False accept')
        ax.bar(x+.18,[r['false_reject'] for r in rows],.36,label='False reject (valid payloads)')
        ax.set_xticks(x,[r['gate'] for r in rows],rotation=20);ax.legend();ax.set(ylabel='Candidate rows',title='W3 training-side thresholds; outer predictions')
        save(fig,output/'W3/figures/gate-confusion')
    if (output/'W4/result.json').is_file() and read_json(output/'W4/result.json').get('curve'):
        rows=read_json(output/'W4/result.json')['curve']
        for metric in ['pre_brier','post_brier','top3_near_oracle','QoR_PASS','estimated_conservative_ms']:
            fig,ax=plt.subplots(figsize=(7,4))
            for model in sorted({r['model'] for r in rows}):
                selected=[r for r in rows if r['model']==model and r[metric]['mean'] is not None]
                x=[r['training_groups'] for r in selected];y=[r[metric]['mean'] for r in selected]
                ax.plot(x,y,'o-',label=model)
                for r in selected:
                    ci=r[metric]['ci95']
                    if ci:ax.plot([r['training_groups']]*2,ci,alpha=.3,color='gray')
            ax.legend(fontsize=7);ax.set(xlabel='Outer-training components (fit + probcal)',ylabel=metric,title='W4 nested development CV; seeds averaged within pair')
            save(fig,output/('W4/figures/'+metric))
    if (output/'W5/results.json').is_file():
        result=read_json(output/'W5/results.json');rows=[r for r in result['variants'] if r['mean_e2e_ms'] is not None];fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(rows))
        ax.bar(x,[r['mean_e2e_ms'] for r in rows]);ax.set_xticks(x,[r['variant'] for r in rows],rotation=15)
        ax.set(ylabel='Mean of pair-median E2E (ms)',title='W5 '+result['measurement']+'; viewed evaluation, Reference-valid pairs')
        save(fig,output/'W5/figures/latency')

        bins=read_csv(output/'W5/reject-reliability.csv')
        fig,ax=plt.subplots(figsize=(6,4));ax.plot([0,1],[0,1],'--',color='gray')
        for name in sorted({r['variant'] for r in bins}):
            selected=[r for r in bins if r['variant']==name and int(r['n'])>0]
            if selected:ax.plot([float(r['probability_mean']) for r in selected],[float(r['observed_pass']) for r in selected],'o-',label=name)
        if any(int(r['n']) for r in bins):ax.legend(fontsize=7)
        ax.set(xlabel='Predicted reject probability',ylabel='Observed REJECT/RETRY fraction',title='W5 descriptive behavior calibration; viewed cohort')
        save(fig,output/'W5/figures/reject-reliability')
