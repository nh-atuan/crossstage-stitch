"""W4 nested component learning curves with fixed outer evaluations."""
from collections import defaultdict
import numpy as np
from tools.controller.diagnostics.learning import sample_groups
from tools.controller.diagnostics.prediction import inner_partition,Prediction,disjoint_partitions
from tools.controller.diagnostics.metrics import group_mean_ci,classification
from tools.controller.diagnostics.counterfactual import replay
from tools.controller.diagnostics.io import write_json,write_csv,read_json
from .training import partitions,pre_models,gate_models,choose_gate
from .data import margins
from .reuse import compatible_fit


def w4(context,output,protocol,spatial):
    grid=context.grid;_,target=margins(context);folds=partitions(context,protocol)
    maximum=min(len(np.unique(grid.groups[np.r_[fit,cal]])) for _,fit,cal,_ in folds)
    sizes=sorted({min(s,maximum) for s in protocol.sizes}|{maximum});records=[];splits=[];fixed_records=[]
    families=['POST0','POST1','POST2']+(['POST3_PHOTO','POST3_GEOMETRY','POST3'] if spatial is not None else [])
    for fold,outer_fit,outer_cal,test in folds:
        pool=np.sort(np.r_[outer_fit,outer_cal])
        for seed in protocol.sampling_seeds:
            for size in sizes:
                if size<3:continue
                selected=sample_groups(grid,pool,size,seed);fit,cal=inner_partition(grid,selected,seed)
                disjoint_partitions(grid.groups,fit,cal,test)
                folder=output/'fits'/f'fold-{fold}-seed-{seed}-groups-{size}';folder.mkdir(parents=True,exist_ok=True)
                parts=(fit,cal,test);gate_trials=[];post0=None
                legacy=compatible_fit(getattr(context,'diagnostics_source',None),fold,seed,size,grid,fit,cal,test,protocol)
                for name in families:
                    gate=gate_models(context,name,target,spatial,parts,folder/name,protocol,legacy=legacy if name=='POST0' else None)
                    selection,_=choose_gate(gate['risk'],gate['valid']&grid.self_ok,grid.truth,cal,protocol.risk_thresholds)
                    item=dict(name=name,selection=selection,gate=gate)
                    gate_trials.append(item)
                    if name=='POST0':post0=item
                binary=min(gate_trials,key=lambda r:(r['selection']['false_accept'],-r['selection']['accepted_pass'],r['name']))
                if gate_trials:
                    gate=gate_models(context,'POST4',target,spatial,parts,folder/'POST4',protocol,feature_name=binary['name'])
                    selection,_=choose_gate(gate['risk'],gate['valid']&grid.self_ok,grid.truth,cal,protocol.risk_thresholds)
                    gate_trials.append(dict(name='POST4',selection=selection,gate=gate,features=binary['name']))
                new=min(gate_trials,key=lambda r:(r['selection']['false_accept'],-r['selection']['accepted_pass'],r['name']))
                split=dict(fold=fold,seed=seed,groups=size,fit_pairs=[grid.pairs[i] for i in fit],
                    probability_calibration_pairs=[grid.pairs[i] for i in cal],outer_evaluation_pairs=[grid.pairs[i] for i in test],
                    selected_gate=new['name'],selected_features=new.get('features',new['name']),
                    gate_selection_role='training-side probability calibration; outer fold never selects',
                    compatible_AE_checkpoint=legacy['source'] if legacy else None,
                    gate_trials=[dict(name=r['name'],**r['selection']) for r in gate_trials],
                    K500_fit_positive_rows=int(grid.truth[fit][:,[c['K']==500 for c in grid.definitions]].sum()),
                    K500_probcal_positive_rows=int(grid.truth[cal][:,[c['K']==500 for c in grid.definitions]].sum()))
                write_json(folder/'split.json',split);splits.append(split)
                for name in ['PRE0','PRE1']:
                    pre=pre_models(context,name,parts,folder/name,protocol,costs=False,legacy=legacy)
                    for gate_label,item in [('POST0',post0),('NEW_GATE',new)]:
                        gate=item['gate'];threshold=item['selection']['threshold']
                        prediction=Prediction(pre['pre'][test],1.-gate['risk'][test],pre['latency'][test],pre['pre'][test],1.-gate['risk'][test],
                            'learning_outer',np.full(len(test),fold))
                        policy=dict(context.model,pre_threshold=protocol.pre_threshold,gate_threshold=1.-threshold)
                        subset=grid.take(test);result=replay(subset,prediction,policy)
                        for local,i in enumerate(test):
                            cheap=grid.truth[i]&(grid.latency[i]<grid.latency[i,grid.reference]);cheap[grid.reference]=False
                            near=grid.truth[i]&(grid.latency[i]<=np.min(grid.latency[i,grid.truth[i]])*1.1);near[grid.reference]=False
                            order=np.argsort(-pre['pre'][i],kind='stable');order=order[order!=grid.reference]
                            valid=gate['valid'][i];key=name+'_'+gate_label
                            records.append(dict(model=key,fold=fold,seed=seed,training_groups=size,pair_id=grid.pairs[i],group_id=str(grid.groups[i]),
                                pre_brier=float(np.mean((pre['pre'][i]-grid.truth[i])**2)),
                                post_brier=float(np.mean((gate['risk'][i,valid]-(~grid.truth[i,valid]))**2)) if valid.any() else None,
                                cheap_pass_recall=float((cheap&(pre['pre'][i]>=protocol.pre_threshold)).any()) if cheap.any() else None,
                                top3_near_oracle=float(near[order[:3]].any()) if near.any() else None,
                                gate_false_accept=int(result['false_accept'][local]),QoR_PASS=bool(result['quality'][local]),
                                estimated_optimistic_ms=float(result['optimistic_ms'][local]),estimated_conservative_ms=float(result['full_recompute_ms'][local])))
                    for i in test:
                        for j,c in enumerate(grid.definitions):
                            if c['K']==500:fixed_records.append(dict(model=name,training_groups=size,seed=seed,pair_id=grid.pairs[i],group_id=str(grid.groups[i]),
                                config_id=c['config_id'],probability=float(pre['pre'][i,j]),truth=bool(grid.truth[i,j])))
                write_json(folder/'complete.json',dict(fit_complete=True,models_confined_to='research outputs'))
                print('W4',fold,seed,size,new['name'],'complete',flush=True)
    curve=[]
    metrics=['pre_brier','post_brier','cheap_pass_recall','top3_near_oracle','gate_false_accept','QoR_PASS','estimated_optimistic_ms','estimated_conservative_ms']
    for model,size in sorted({(r['model'],r['training_groups']) for r in records}):
        by_pair=defaultdict(list)
        for r in records:
            if (r['model'],r['training_groups'])==(model,size):by_pair[r['pair_id']].append(r)
        row=dict(model=model,training_groups=size,pairs=len(by_pair),groups=len({rs[0]['group_id'] for rs in by_pair.values()}))
        for metric in metrics:
            values=[];groups=[]
            for rs in by_pair.values():
                valid=[r[metric] for r in rs if r[metric] is not None]
                if valid:values.append(float(np.mean(valid)));groups.append(rs[0]['group_id'])
            row[metric]=group_mean_ci(values,groups,protocol.seed,protocol.bootstrap_replicates)
        curve.append(row)
    fixed=[]
    for model,size,config in sorted({(r['model'],r['training_groups'],r['config_id']) for r in fixed_records}):
        by_pair=defaultdict(list)
        for r in fixed_records:
            if (r['model'],r['training_groups'],r['config_id'])==(model,size,config):by_pair[r['pair_id']].append(r)
        y=[rs[0]['truth'] for rs in by_pair.values()];p=[np.mean([r['probability'] for r in rs]) for rs in by_pair.values()]
        groups=[rs[0]['group_id'] for rs in by_pair.values()]
        stats=classification(y,p,groups,protocol)
        fixed.append(dict(model=model,training_groups=size,config_id=config,**{k:v for k,v in stats.items() if k!='bins'}))
    gains=[]
    for model in sorted({r['model'] for r in records}):
        by_size_pair=defaultdict(list)
        for r in records:
            if r['model']==model:by_size_pair[r['training_groups'],r['pair_id']].append(r)
        available=sorted({key[0] for key in by_size_pair})
        if len(available)<2:continue
        for metric in ['pre_brier','post_brier','top3_near_oracle','QoR_PASS','estimated_conservative_ms']:
            values=[];groups=[]
            for pair in grid.pairs:
                small=by_size_pair[available[0],pair];large=by_size_pair[available[-1],pair]
                a=[r[metric] for r in small if r[metric] is not None];b=[r[metric] for r in large if r[metric] is not None]
                if not a or not b:continue
                gain=float(np.mean(b)-np.mean(a))
                if metric in ['pre_brier','post_brier','estimated_conservative_ms']:gain=-gain
                values.append(gain);groups.append(large[0]['group_id'])
            gains.append(dict(model=model,metric=metric,smallest_groups=available[0],largest_groups=available[-1],
                gain=group_mean_ci(values,groups,protocol.seed,protocol.bootstrap_replicates)))
    supported=[];limited=[]
    def positive(row):return row and row['gain'].get('ci95') and row['gain']['ci95'][0]>0
    for model in sorted({r['model'] for r in records}):
        m={r['metric']:r for r in gains if r['model']==model}
        brier=m.get('pre_brier');recall=m.get('top3_near_oracle');quality=m.get('QoR_PASS')
        if positive(brier) and brier['gain']['mean']>=protocol.material_brier_gain and (positive(recall) or positive(quality)):supported.append(model)
        largest=max([r for r in curve if r['model']==model],key=lambda r:r['training_groups'],default=None)
        if brier and brier['gain'].get('ci95') and brier['gain']['ci95'][1]<protocol.material_brier_gain and largest:
            ci=largest['top3_near_oracle']['ci95']
            if ci and ci[1]<protocol.weak_near_oracle_recall:limited.append(model)
    interpretation=dict(label='data-limited' if supported else 'feature-limited' if limited else 'inconclusive',
        supported_models=supported,feature_limited_hypothesis_models=limited,
        reason='Predeclared pilot: component-paired Brier plus recall/PASS evidence. Flat curves only suggest feature/target/capacity limits; no online speedup or XGBoost failure is inferred.')
    result=dict(status='OFFLINE_EVALUATED' if spatial is not None else 'PARTIAL',curve=curve,interpretation=interpretation,paired_smallest_to_largest_gain=gains,
        expansion=dict(status='NO_NEW_TRAINING_DATA',pairs_added=0,reason='No approved disjoint expansion manifest. Existing viewed evaluation is excluded; archive availability alone does not certify new scene independence.'),
        limits='Outer evaluations fixed. Gate/feature selection repeated inside each training pool. Seeds averaged per pair before component bootstrap. Replay uses frozen legacy score to isolate pre/gate sample effects; W5 measures cost planner separately.')
    write_csv(output/'learning-curve.csv',curve);write_csv(output/'learning-per-pair.csv',records);write_csv(output/'fixed-config-K500.csv',fixed)
    write_json(output/'split-manifest.json',splits);write_json(output/'data-counts.json',dict(calibration_pairs=len(grid.pairs),groups=len(np.unique(grid.groups)),sizes=sizes,expansion=result['expansion']))
    write_json(output/'result.json',result);return result
