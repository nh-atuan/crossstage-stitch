"""Frozen run matrix; no user-facing phase/model/threshold switches."""
from dataclasses import dataclass,asdict
@dataclass(frozen=True)
class Protocol:
    seed:int=20261101
    warmup:int=2
    repeats:int=10
    outer_folds:int=5
    rounds:int=64
    depth:int=2
    sizes:tuple=(40,80,120,156)
    sampling_seeds:tuple=(20261101,20261102,20261103)
    risk_thresholds:tuple=(.05,.10,.20,.35,.50)
    pre_threshold:float=.50
    profile_pairs:int=12
    max_attempts:int=4
    bootstrap_replicates:int=1000
    thresholds:tuple=(.85,.70,.50)
    gate_thresholds:tuple=(.65,.85)
    bins:int=10
    top_k:tuple=(1,3,5)
    near_oracle_ratio:float=1.10
    variants:tuple=('REF','V4','SEL','GATE','COST+GATE','STAGED','V5_COMBINED')
    training_expansion_pairs:int=0
    material_brier_gain:float=.02
    weak_near_oracle_recall:float=.80
    def mapping(self):return asdict(self)
