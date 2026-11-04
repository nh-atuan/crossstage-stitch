"""Two-attempt pooled policy: choose a candidate, then accept or reference fallback."""
from __future__ import annotations
import numpy as np
from tools.controller.boosted_model import predict


def predictions(models, grid):
    n,c,_=grid.pre.shape
    return (predict(models['pre_quality'],grid.pre.reshape(n*c,-1)).reshape(n,c),
            predict(models['latency'],grid.pre.reshape(n*c,-1)).reshape(n,c),
            predict(models['post_quality'],grid.post.reshape(n*c,-1)).reshape(n,c))


def choose(pre_probability, latency, reference, pre_threshold, extra_ms):
    # Approximate expected cost; calibration replay includes gate rejection and
    # measured feature costs. This is a ranking score, not a QoR guarantee.
    if pre_threshold>=1.:
        return np.full(len(latency),reference,dtype=int)
    score = latency+(1.-pre_probability)*latency[:, reference,None]+extra_ms
    score[pre_probability<pre_threshold] = np.inf
    score[:,reference] = latency[:,reference]
    result=np.argmin(score, axis=1)
    result[score[np.arange(len(score)),result]>=latency[:,reference]]=reference
    return result


def replay(grid, prediction, *, pre_threshold, gate_threshold, extra_ms=.5,probe_ms=.5):
    probability,latency,post=prediction
    initial=choose(probability,latency,grid.reference,pre_threshold,extra_ms)
    i=np.arange(len(grid.pairs))
    cheap=initial!=grid.reference
    accepted=(grid.present[i,initial] & (post[i,initial]>=gate_threshold)) | ~cheap
    final=np.where(accepted,initial,grid.reference)
    cost=grid.latency[i,initial]+probe_ms
    rejected=cheap & ~accepted
    # Conservative full fallback cost, no fictional reuse saving. Runtime reuses
    # stages where eligible; the final performance claim must use measured E2E.
    cost[rejected]=grid.attempt[i[rejected],initial[rejected]]+grid.feature_ms[i[rejected],initial[rejected]]+grid.latency[rejected,grid.reference]+probe_ms
    cost[cheap & accepted]+=grid.feature_ms[i[cheap & accepted],initial[cheap & accepted]]
    correct=grid.truth[i,final]
    return dict(initial=initial,final=final,cost=cost,correct=correct,
        false_accept=int((~correct).sum()),retry=int(rejected.sum()),
        cheap_stops=int((cheap & accepted).sum()),mean_ms=float(cost.mean()),
        reference_mean_ms=float(grid.latency[:,grid.reference].mean()))


def tune(grid, prediction, max_false=0,extra_ms=.5,probe_ms=.5):
    alternatives=[]
    for before in [0.,.5,.7,.85,.95,1.]:
        for after in [.5,.65,.75,.85,.9,.95,.98,.995,1.]:
            result=replay(grid,prediction,pre_threshold=before,gate_threshold=after,extra_ms=extra_ms,probe_ms=probe_ms)
            if result['false_accept']<=max_false:
                alternatives.append((result['mean_ms'],-after,-before,before,after,result))
    if not alternatives:
        raise ValueError('reference fallback must remain feasible')
    _,_,_,before,after,result=min(alternatives,key=lambda v:v[:3])
    return dict(pre_threshold=before,gate_threshold=after,extra_ms=extra_ms,probe_ms=probe_ms),result
