"""Export alignment previews using measured H and canonical metric geometry."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from tools.controller.calibrate import REFERENCE,digest
from tools.controller.detailed_metrics import read_runs
from tools.qor import evaluate as canonical
from tools.qor.report import read_qor


def export(manifest,oracle,quality_policy,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():raise ValueError('preview output already exists')
    pairs=json.loads(manifest.read_text());audit=json.loads((oracle/'audit.json').read_text())
    chosen={r['pair_id']:r['config_id'] for r in audit['oracle']}
    records={}
    for row in csv.DictReader((oracle/'grid.csv').open(newline='')):
        if row['is_warmup']=='0' and row['repeat_idx']=='0':records[row['pair_id'],row['config_id']]=row
    adaptive=read_runs(oracle/'v3_p80.csv');labels=read_qor(oracle/'qor.csv');policy=canonical.load_policy(quality_policy)
    traces={}
    for row in csv.DictReader((oracle/'v3_p80-attempts.csv').open(newline='')):
        if row['is_warmup']=='0' and row['repeat_idx']=='0':traces.setdefault(row['pair_id'],[]).append(row)
    retry=sorted(p for p,rows in traces.items() if len(rows)>1)
    # Predeclared diagnostic categories: successful cheap stop, known SSIM false
    # accept, and the median-latency retry. No model is updated from these images.
    retry_median=float(np.median([np.median([float(r['t_e2e']) for r in adaptive[p]]) for p in retry]))
    retry_pair=min(retry,key=lambda p:(abs(float(np.median([float(r['t_e2e']) for r in adaptive[p]]))-retry_median),p))
    cases=[('cheap-pass','udis_train_001606'),('ssim-failure','udis_train_003317'),('retry',retry_pair)]
    output.mkdir(parents=True);ledger=[]
    for category,pair in cases:
        a=canonical._read_image(manifest.parent/pairs[pair]['a']);b=canonical._read_image(manifest.parent/pairs[pair]['b'])
        ref=records[pair,REFERENCE];controller=adaptive[pair][0];optimum=records[pair,chosen[pair]]
        raw=[ref,controller,optimum];hs=[canonical._geometry(np.array([float(r[f'h{i}']) for i in range(9)]).reshape(3,3),b.shape) for r in raw]
        warped=[cv2.warpPerspective(b,h,(a.shape[1],a.shape[0]),flags=cv2.INTER_LINEAR) for h in hs]
        reports=[]
        for row,h in zip(raw,hs):
            report=canonical.evaluate_pair(a,b,h,hs[0],policy);expected=labels[pair][row['config_id']]
            for metric in ['ssim_overlap','psnr_overlap','overlap_coverage']:
                if not np.isclose(report[metric],expected[metric],rtol=0,atol=1e-12,equal_nan=True):
                    raise ValueError('preview metrics differ from audited H/labels')
            reports.append(report)
        mask=canonical.valid_mask(hs[0],a.shape,b.shape)&canonical.valid_mask(hs[1],a.shape,b.shape)
        error=np.abs(a.astype(float)-warped[1]).mean(axis=2)/255.;error[~mask]=np.nan
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        panels=[('Input A',a),('Input B',b),('Fixed-Best\n'+ref['config_id'],warped[0]),
            ('Controller\n'+controller['config_id'],warped[1]),('Oracle\n'+optimum['config_id'],warped[2]),('Controller |A-warp(B)|',error)]
        for i,(title,im) in enumerate(panels):
            ax=axes.flat[i];ax.imshow(im,**(dict(cmap='magma',vmin=0,vmax=.25) if i==5 else {}));ax.axis('off')
            if i in [2,3,4]:
                report=reports[{2:0,3:1,4:2}[i]]
                title+=f"\nSSIM={report['ssim_overlap']:.4f}, PSNR={report['psnr_overlap']:.2f} dB"
            ax.set_title(title,fontsize=9)
        fig.suptitle(pair+' / '+category+' / measured H; previews in A coordinates',fontsize=12)
        fig.tight_layout();name=category+'-'+pair+'.png';fig.savefig(output/name,dpi=130);plt.close(fig)
        # Keep exact input PNGs as well as the composed scientific preview.
        for side,im in [('a',a),('b',b)]:cv2.imwrite(str(output/(pair+'-'+side+'.png')),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        ledger.append(dict(pair_id=pair,category=category,preview=name,controller_config=controller['config_id'],
            oracle_config=optimum['config_id'],reference_config=REFERENCE,metrics=[{k:r[k] for k in ['ssim_overlap','psnr_overlap','overlap_coverage']} for r in reports]))
    result=dict(cases=ledger,metric_parity='passed_at_1e-12',
        inputs_sha256={name:digest(path) for name,path in [('manifest',manifest),('grid',oracle/'grid.csv'),('controller',oracle/'v3_p80.csv'),('qor',oracle/'qor.csv')]},
        limits='Alignment previews for canonical QoR, not S6 panorama compositions. Black pixels show image support boundaries; heatmap is photometric error, not SSIM.')
    (output/'previews.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','oracle','quality-policy','output']:p.add_argument('--'+name,type=Path,required=True)
    export(**vars(p.parse_args()))

if __name__=='__main__':main()
