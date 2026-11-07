"""Calibration-only sample diagnosis; labels never enter online features."""
from pathlib import Path
import json
import cv2
import numpy as np
from tools.controller.calibrate import REFERENCE, load_samples
from tools.controller.fit_checks import accepts
from tools.controller.feature_schema import GATE, read_features
from tools.qor.report import read_qor


def analyze(runs,audit,manifest,split,configs,features,old_fit,qor,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cohort,_,samples,labels,valid,_=load_samples(runs,audit,manifest,split)
    definitions={c['config_id']:c for c in json.loads(configs.read_text())}
    observations,_=read_features(features,set(cohort),set(definitions))
    checks=json.loads(old_fit.read_text())['gates'];quality=read_qor(qor)
    config='S0.75_K500_R128_Bfeather';check=checks[config]['check']
    positive=[p for p in valid if labels[p,config] and not accepts(samples[p,config],check)]
    negative=[p for p in valid if not labels[p,config] and samples[p,config]['status']=='OK'
              and not samples[p,config]['fail_stage']]
    metric=lambda p:np.array([float(samples[p,config][f]) for f in ['num_inliers','inlier_ratio','inlier_coverage','reproj_residual']])
    examples=None;dominated=[]
    for p in positive:
        good=metric(p)
        bad=[n for n in negative if np.all(metric(n)[:3]>=good[:3]) and metric(n)[3]<=good[3]]
        if bad:
            dominated.append(p)
            if examples is None:examples=[p,bad[0]]
    if examples is None:raise ValueError('expected a calibration counterexample')
    output.mkdir(parents=True,exist_ok=False);ledger=[]
    figure,axes=plt.subplots(2,5,figsize=(15,6))
    for row,pair in enumerate(examples):
        a=cv2.cvtColor(cv2.imread(str(manifest.parent/cohort[pair]['a'])),cv2.COLOR_BGR2RGB)
        b=cv2.cvtColor(cv2.imread(str(manifest.parent/cohort[pair]['b'])),cv2.COLOR_BGR2RGB)
        h=np.array([float(samples[pair,config][f'h{i}']) for i in range(9)]).reshape(3,3)
        ref=np.array([float(samples[pair,REFERENCE][f'h{i}']) for i in range(9)]).reshape(3,3)
        shape=(a.shape[1],a.shape[0]);warped=cv2.warpPerspective(b,h,shape);reference=cv2.warpPerspective(b,ref,shape)
        mask=cv2.warpPerspective(np.ones(b.shape[:2],np.uint8),h,shape,flags=cv2.INTER_NEAREST)>0
        error=np.abs(a.astype(float)-warped).mean(axis=2)/255.;error[~mask]=np.nan
        for column,(name,value) in enumerate([('A',a),('B',b),('candidate warp',warped),('reference warp',reference),('candidate |A-warp|',error)]):
            axes[row,column].imshow(value,**(dict(cmap='magma',vmin=0,vmax=.25) if column==4 else {}))
            axes[row,column].set_title(f'{pair}\n{name}');axes[row,column].axis('off')
        obs=observations[pair,config]
        ledger.append(dict(pair_id=pair,config_id=config,qor_pass=labels[pair,config],
            old_gate_accept=accepts(samples[pair,config],check),qor=quality[pair][config],
            reference_qor=quality[pair][REFERENCE],gate_features=dict(zip(GATE,map(float,obs['gate'])))))
    figure.tight_layout();figure.savefig(output/'calibration-counterexamples.png',dpi=110);plt.close(figure)
    report=dict(config_id=config,old_gate=check,old_gate_rejected_positive_count=len(positive),
                rejected_positive_dominated_by_negative=len(dominated),examples=ledger,
                interpretation='S5 score dominance does not imply relative image QoR. Selected on calibration only.')
    (output/'case-ledger.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
