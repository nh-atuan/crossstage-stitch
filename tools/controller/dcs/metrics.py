"""Selector decisions with explicit FAIL penalty and paired component intervals."""
import numpy as np
from tools.controller.diagnostics.metrics import group_mean_ci
from .labels import order,targets

def ledger(grid,scores,method,feature='F0',fold=-1):
    ranking=order(scores,grid.definitions);labels=targets(grid);rows=[]
    for i,pair in enumerate(grid.pairs):
        j=int(ranking[i,0]);oracle=int(labels['oracle'][i]);ref=grid.latency[i,grid.reference];lat=grid.latency[i,j];ok=bool(grid.truth[i,j])
        penalized=lat+(0. if ok else ref)
        row=dict(method=method,feature=feature,fold=fold,pair_id=pair,group_id=str(grid.groups[i]),selected=grid.definitions[j]['config_id'],
            selected_index=j,oracle=grid.definitions[oracle]['config_id'],PASS=ok,selected_ms=lat,reference_ms=ref,oracle_ms=grid.latency[i,oracle],
            penalized_ms=penalized,penalized_regret_ms=penalized-grid.latency[i,oracle],exact_oracle=j==oracle,
            K500=grid.definitions[j]['K']==500,R128=grid.definitions[j]['R']==128,
            K500_R128_PASS=ok and grid.definitions[j]['K']==500 and grid.definitions[j]['R']==128,
            K500_R128_available=bool(any(grid.truth[i,k] and c['K']==500 and c['R']==128 for k,c in enumerate(grid.definitions))),
            pass_only_ratio=lat/grid.latency[i,oracle] if ok else None,rank_score=float(scores[i,j]),information='GROUPED_OOF',measurement='FIXED_GRID_LABELS_NOT_ONLINE_E2E')
        for k in [1,3,5]:
            row[f'top{k}_PASS']=bool(grid.truth[i,ranking[i,:k]].any())
            row[f'top{k}_near_oracle']=bool((grid.truth[i,ranking[i,:k]]&(grid.latency[i,ranking[i,:k]]<=grid.latency[i,oracle]*1.05)).any())
        rows.append(row)
    return rows

def summary(rows,protocol):
    def ci(field):return group_mean_ci([float(r[field]) for r in rows],[r['group_id'] for r in rows],protocol.seed,protocol.bootstrap_replicates)
    if not rows:return dict(n=0)
    denominator=sum(r['reference_ms']-r['oracle_ms'] for r in rows)
    unique=sorted({r['group_id'] for r in rows});rng=np.random.default_rng(protocol.seed)
    numerator=np.array([sum(r['reference_ms']-r['penalized_ms'] for r in rows if r['group_id']==g) for g in unique])
    den=np.array([sum(r['reference_ms']-r['oracle_ms'] for r in rows if r['group_id']==g) for g in unique])
    sample=rng.integers(0,len(unique),(protocol.bootstrap_replicates,len(unique)));ds=den[sample].sum(axis=1);ns=numerator[sample].sum(axis=1)
    gap_ci=np.quantile(ns[ds>0]/ds[ds>0],[.025,.975]).tolist() if (ds>0).any() and len(unique)>1 else None
    return dict(method=rows[0]['method'],feature=rows[0]['feature'],n=len(rows),FAIL=sum(not r['PASS'] for r in rows),
        top1_PASS=ci('PASS'),penalized_regret_ms=ci('penalized_regret_ms'),mean_fixed_ms=ci('selected_ms'),
        oracle_gap_closed=sum(r['reference_ms']-r['penalized_ms'] for r in rows)/denominator if denominator>0 else None,
        oracle_gap_closed_ci95=gap_ci,gap_definition='sum(Tref - [Tselected + Tref if FAIL]) / sum(Tref - Toracle); all Reference-valid queries, can be negative',
        exact_oracle=ci('exact_oracle'),top3_near_oracle=ci('top3_near_oracle'),top5_near_oracle=ci('top5_near_oracle'),
        K500=sum(r['K500'] for r in rows),K500_R128_PASS=sum(r['K500_R128_PASS'] for r in rows),
        feasible_ratio_mean=float(np.mean([r['pass_only_ratio'] for r in rows if r['PASS']])) if any(r['PASS'] for r in rows) else None,
        p95_fixed_ms=float(np.quantile([r['selected_ms'] for r in rows],.95)),median_fixed_ms=float(np.median([r['selected_ms'] for r in rows])))

def paired(a,b,protocol):
    left={r['pair_id']:r for r in a};right={r['pair_id']:r for r in b}
    if left.keys()!=right.keys():raise ValueError('paired selector population mismatch')
    groups=[left[p]['group_id'] for p in left]
    return dict(left=a[0]['method'],right=b[0]['method'],n=len(left),
        PASS_delta=group_mean_ci([int(left[p]['PASS'])-int(right[p]['PASS']) for p in left],groups,protocol.seed,protocol.bootstrap_replicates),
        regret_saved_ms=group_mean_ci([right[p]['penalized_regret_ms']-left[p]['penalized_regret_ms'] for p in left],groups,protocol.seed,protocol.bootstrap_replicates))

def selection_key(rows):return (sum(not r['PASS'] for r in rows),float(np.mean([r['penalized_ms'] for r in rows])))
