"""Training-only QoR/latency supervision; no offline fields enter features."""
import numpy as np

def targets(grid):
    if grid.truth.shape[1]!=48 or not grid.truth[:,grid.reference].all():raise ValueError('48 items and valid Reference per supervised query required')
    if not np.isfinite(grid.latency).all() or (grid.latency<=0).any():raise ValueError('positive measured latencies required')
    ref=grid.latency[:,grid.reference,None];gain=np.maximum(0.,(ref-grid.latency)/ref)
    graded=np.where(grid.truth,1+(gain>=.10)+(gain>=.30)+(gain>=.50),0).astype(int)
    precedence=np.zeros_like(graded)
    names=np.array([c['config_id'] for c in grid.definitions])
    for i in range(len(grid.pairs)):
        order=sorted(np.flatnonzero(grid.truth[i]),key=lambda j:(-grid.latency[i,j],names[j]))
        rank=0;previous=None
        for j in order:
            if previous is None or grid.latency[i,j]!=previous:rank+=1
            precedence[i,j]=rank;previous=grid.latency[i,j]
    oracle=np.array([min(np.flatnonzero(grid.truth[i]),key=lambda j:(grid.latency[i,j],names[j])) for i in range(len(grid.pairs))])
    utility=np.where(grid.truth,1.+gain,-1.)
    return dict(graded=graded,precedence=precedence,oracle=oracle,utility=utility)

def order(scores,definitions):
    scores=np.asarray(scores)
    if scores.shape[-1]!=48 or not np.isfinite(scores).all():raise ValueError('all 48 finite scores required')
    names=np.array([c['config_id'] for c in definitions]);tie=np.argsort(np.argsort(names))
    return np.array([np.lexsort((tie,-row)) for row in scores.reshape(-1,48)]).reshape(scores.shape)
