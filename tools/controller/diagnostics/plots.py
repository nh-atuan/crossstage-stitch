"""Export scientific diagnostic figures as standalone PNG and PDF artifacts."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=160)
    fig.savefig(path.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def reliability(records,output):
    for stage in ['pre','post']:
        fig,ax=plt.subplots(figsize=(6,4))
        ax.plot([0,1],[0,1],'--',color='gray',label='Ideal')
        for variant in ['raw_sigmoid','stored_probability_calibration']:
            rows=[r for r in records if r['scope']=='calibration_oof' and r['stage']==stage and r['slice']=='all' and r['variant']==variant and r['n']]
            if rows:
                x=[r['probability_mean'] for r in rows];y=[r['observed_pass'] for r in rows]
                ax.plot(x,y,'o-',label=variant)
                for a,b,r in zip(x,y,rows):
                    ci=r['observed_pass_ci95']
                    if ci:ax.plot([a,a],ci,color='gray',alpha=.4)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed QoR PASS fraction',title=f'{stage}: calibration OOF; component bootstrap CI')
        ax.legend(fontsize=8);save(fig,Path(output)/('reliability-'+stage))


def learning(curve,output):
    for metric in ['pre_brier','post_brier','cheap_feasible_recall','top3_near_oracle']:
        fig,ax=plt.subplots(figsize=(6,4))
        rows=[r for r in curve if r[metric]['mean'] is not None]
        if rows:
            x=[r['training_groups'] for r in rows];y=[r[metric]['mean'] for r in rows]
            ax.plot(x,y,'o-')
            for a,b,r in zip(x,y,rows):
                ci=r[metric]['ci95']
                if ci:ax.plot([a,a],ci,color='gray')
        ax.set(xlabel='Independent outer-training components (fit + probability calibration)',ylabel=metric,title='Fixed outer evaluation; seeds averaged within pair')
        save(fig,Path(output)/('learning-'+metric))


def counterfactual(comparisons,output):
    rows=[r for r in comparisons if r['scope']=='calibration_oof' and r['policy']=='P0']
    fig,ax=plt.subplots(figsize=(7,4))
    if rows:
        x=np.arange(len(rows));a=[r['optimistic_ms']['mean'] for r in rows];b=[r['full_recompute_ms']['mean'] for r in rows]
        ax.bar(x-.18,a,.36,label='Stage-only scenario');ax.bar(x+.18,b,.36,label='Full-recompute scenario')
        ax.set_xticks(x,[r['variant'] for r in rows],rotation=18);ax.legend()
    ax.set(ylabel='Estimated ms (not GPU E2E)',title='Calibration OOF counterfactual; P0 selector settings')
    save(fig,Path(output)/'perfect-pre-scenarios')
