"""ROC rank AUC, noninterpolated AP and component-bootstrap calibration metrics.

Definitions: https://scikit-learn.org/stable/modules/calibration.html and
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
Accessed 2026-11-07. PR-AUC is reported explicitly as average precision (AP).
"""
from __future__ import annotations
import numpy as np


def validate(y,p):
    y=np.asarray(y);p=np.asarray(p,float)
    if y.shape!=p.shape or y.ndim!=1 or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('finite aligned binary labels/probabilities required')
    return y.astype(bool),p


def roc_auc(y,p):
    y,p=validate(y,p);positive=int(y.sum());negative=len(y)-positive
    if not positive or not negative:return None
    order=np.argsort(p,kind='stable');sorted_p=p[order];ranks=np.empty(len(p),float)
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_p))+1];ends=np.r_[starts[1:],len(p)]
    for start,end in zip(starts,ends):ranks[order[start:end]]=(start+1+end)/2
    return float((ranks[y].sum()-positive*(positive+1)/2)/(positive*negative))


def average_precision(y,p):
    y,p=validate(y,p)
    if not y.sum():return None
    order=np.argsort(-p,kind='stable');ys=y[order];scores=p[order]
    ends=np.r_[np.flatnonzero(np.diff(scores))+1,len(y)]
    true=np.cumsum(ys)[ends-1];recall=true/y.sum();precision=true/ends
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def confusion(y,p,threshold):
    y,p=validate(y,p);positive=p>=threshold
    tp=int((y&positive).sum());fp=int((~y&positive).sum());fn=int((y&~positive).sum());tn=int((~y&~positive).sum())
    return dict(threshold=threshold,tp=tp,fp=fp,fn=fn,tn=tn,
        recall=tp/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None,
        false_positive_rate=fp/(fp+tn) if fp+tn else None)


def group_mean_ci(values,groups,seed=20261101,repeats=1000):
    values=np.asarray(values,float);groups=np.asarray(groups)
    if values.shape!=groups.shape or values.ndim!=1 or not np.isfinite(values).all():raise ValueError('invalid cluster bootstrap input')
    unique=np.unique(groups)
    if not len(values):return dict(n=0,groups=0,mean=None,ci95=None)
    sums=np.array([values[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique])
    if len(unique)<2:return dict(n=len(values),groups=len(unique),mean=float(values.mean()),ci95=None)
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(unique),(repeats,len(unique)))
    boot=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return dict(n=len(values),groups=len(unique),mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        bootstrap_unit='component; preserve all rows of each sampled component')


def reliability(y,p,groups,bins=10,seed=20261101,repeats=1000):
    y,p=validate(y,p);groups=np.asarray(groups)
    if groups.shape!=y.shape:raise ValueError('calibration groups mismatch')
    indices=np.minimum((p*bins).astype(int),bins-1);rows=[]
    for b in range(bins):
        mask=indices==b;n=int(mask.sum());ci=group_mean_ci(y[mask].astype(float),groups[mask],seed,repeats)
        rows.append(dict(bin=b,left=b/bins,right=(b+1)/bins,n=n,groups=ci['groups'],
            probability_mean=float(p[mask].mean()) if n else None,observed_pass=float(y[mask].mean()) if n else None,
            observed_pass_ci95=ci['ci95'],support='limited' if ci['groups']<10 else 'available'))
    ece=sum(r['n']*abs(r['probability_mean']-r['observed_pass']) for r in rows if r['n'])/len(y) if len(y) else None
    return rows,ece


def classification(y,p,groups,protocol):
    y,p=validate(y,p);groups=np.asarray(groups)
    bins,ece=reliability(y,p,groups,protocol.bins,protocol.seed,protocol.bootstrap_replicates)
    clipped=np.clip(p,1e-15,1-1e-15)
    brier=(p-y)**2;logloss=-(y*np.log(clipped)+(~y)*np.log1p(-clipped))
    return dict(rows=len(y),pairs_or_components=len(np.unique(groups)),positives=int(y.sum()),negatives=int((~y).sum()),
        roc_auc=roc_auc(y,p),pr_auc_average_precision=average_precision(y,p),pr_definition='noninterpolated average precision; ties grouped',
        brier=group_mean_ci(brier,groups,protocol.seed,protocol.bootstrap_replicates),
        log_loss=group_mean_ci(logloss,groups,protocol.seed,protocol.bootstrap_replicates),ece=ece,
        bins=bins,confusion=[confusion(y,p,t) for t in sorted(set(protocol.thresholds+protocol.gate_thresholds))],
        limits='Point metrics pool recorded rows; confidence intervals resample entire components, not configurations independently.')
