"""W3 binary versus violation-risk gates, with spatial group ablations."""
import numpy as np
from tools.controller.diagnostics.io import write_json,write_csv
from tools.controller.diagnostics.metrics import classification
from tools.controller.pooled_policy import choose
from .training import partitions,gate_models,choose_gate
from .data import margins


def w3(context,output,protocol,spatial):
    grid=context.grid;values,target=margins(context);records=[];summaries=[];route_rows=[];available=[]
    n,c=grid.truth.shape
    stored=np.load(context.output/'W2'/('PRE0-oof.npz'))
    old_initial=choose(stored['pre'],stored['latency'],grid.reference,.85,context.model['extra_ms'])
    families=['POST0','POST1','POST2']+(['POST3_PHOTO','POST3_GEOMETRY','POST3'] if spatial is not None else [])
    # POST4's feature family is selected using training-side cal groups in every fold.
    folds=partitions(context,protocol);binary_by_fold={}
    for name in families+['POST4']:
        risk=np.full((n,c),np.nan);valid=np.zeros((n,c),bool);accept=np.zeros((n,c),bool);thresholds=[];components=np.full((n,c,4),np.nan)
        for fold,fit,cal,test in folds:
            if name=='POST4':
                candidates=binary_by_fold[fold]
                feature=min(candidates,key=lambda r:(r['selection']['false_accept'],-r['selection']['accepted_pass'],r['name']))['name']
                # Margin heads use the chosen feature set, without changing capacity.
                trained=gate_models(context,'POST4',target,spatial,(fit,cal,test),output/'models'/name/f'fold-{fold}',protocol,feature_name=feature)
            else:
                feature=name;trained=gate_models(context,name,target,spatial,(fit,cal,test),output/'models'/name/f'fold-{fold}',protocol)
            selection,trials=choose_gate(trained['risk'],trained['valid']&grid.self_ok,grid.truth,cal,protocol.risk_thresholds)
            if name!='POST4':binary_by_fold.setdefault(fold,[]).append(dict(name=name,selection=selection))
            risk[test]=trained['risk'][test];valid[test]=trained['valid'][test]
            if name=='POST4':components[test]=np.stack(trained['components'],axis=2)[test]
            accept[test]=trained['valid'][test]&grid.self_ok[test]&(trained['risk'][test]<=selection['threshold'])
            thresholds.append(dict(fold=fold,threshold=selection['threshold'],selected_features=feature,training_side_trials=trials))
        metric=classification((~grid.truth)[valid],risk[valid],np.broadcast_to(grid.groups[:,None],grid.truth.shape)[valid],protocol)
        false=int((accept&~grid.truth).sum());true=int((accept&grid.truth).sum())
        summaries.append(dict(gate=name,false_accept=false,accepted_pass=true,valid_rows=int(valid.sum()),
            false_reject=int((~accept&grid.truth&valid).sum()),metrics={k:v for k,v in metric.items() if k!='bins'},fold_thresholds=thresholds))
        available.append(name)
        for i,pair in enumerate(grid.pairs):
            for j,config in enumerate(grid.definitions):records.append(dict(gate=name,pair_id=pair,group_id=str(grid.groups[i]),config_id=config['config_id'],
                post_valid=bool(valid[i,j]),self_check=bool(grid.self_ok[i,j]),risk=float(risk[i,j]) if valid[i,j] else None,
                p_fail_ssim=components[i,j,0],p_fail_psnr=components[i,j,1],p_fail_coverage=components[i,j,2],p_fail_joint_head=components[i,j,3],
                accept=bool(accept[i,j]),truth=bool(grid.truth[i,j]),ssim_margin=values[i,j,0],psnr_margin=values[i,j,1],coverage_margin=values[i,j,2]))
            j=old_initial[i];route_rows.append(dict(gate=name,selector='PRE0 old-score OOF',pair_id=pair,config_id=grid.definitions[j]['config_id'],
                is_reference=bool(j==grid.reference),valid=bool(valid[i,j]),risk=float(risk[i,j]) if valid[i,j] else None,
                accept=bool(accept[i,j]) if j!=grid.reference else bool(grid.self_ok[i,j]),truth=bool(grid.truth[i,j])))
        np.savez_compressed(output/(name+'-oof.npz'),risk=risk,valid=valid,accept=accept)
    best=min(summaries,key=lambda r:(r['false_accept'],-r['accepted_pass'],r['gate']))
    # Freeze a deployment threshold on development OOF. The reported OOF values
    # above used inner thresholds; selecting this threshold is not blind testing.
    selected=np.load(output/(best['gate']+'-oof.npz'))
    selected_threshold,trials=choose_gate(selected['risk'],selected['valid']&grid.self_ok,grid.truth,np.arange(n),protocol.risk_thresholds)
    best_binary=min([r for r in summaries if r['gate']!='POST4'],key=lambda r:(r['false_accept'],-r['accepted_pass'],r['gate']))['gate']
    final=dict(feature_family=best_binary if best['gate']=='POST4' else best['gate'],gate=best['gate'],risk_threshold=selected_threshold['threshold'],trials=trials,
        selection_role='Grouped development only; final viewed evaluation is never used for selection',
        feature_schema='controller-v5-features-1',risk_rule='POST4 max(joint head, union bound of three violations); binary otherwise',
        thresholds_role='Zero observed development false accepts, not a certified risk bound')
    write_json(output/'selected-gate.json',final);write_csv(output/'gate-oof.csv',records);write_csv(output/'route-gate-analysis.csv',route_rows)
    write_csv(output/'feature-cost.csv',[dict(gate=name,cost_status='MEASURE_ON_T4',collector='real candidate-only CUDA features; mask/group flags frozen per policy') for name in available])
    result=dict(status='OFFLINE_EVALUATED' if spatial is not None else 'PARTIAL',selected_gate=final,summary=summaries,
        missing=[] if spatial is not None else ['POST3/POST4 require recovered images and spatial CPU observations'],
        hypothesis='More feature groups or margin heads may lose; no assumption of improvement.')
    write_json(output/'result.json',result);return result
