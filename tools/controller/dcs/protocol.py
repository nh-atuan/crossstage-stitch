"""Prespecified search and bounded run matrix; no evaluation-driven choices."""
from dataclasses import dataclass
from tools.controller.v5.protocol import Protocol as V5Protocol
@dataclass(frozen=True)
class Protocol(V5Protocol):
    inner_folds:int=2
    search:tuple=((1,32,40.),(2,64,20.),(3,96,40.))
    families:tuple=('DCS_RANK_NDCG','DCS_RANK_PAIR','DCS_DIRECT48','DCS_LINEAR')
    feature_sets:tuple=('F0','F1','F2','F0_NO_TEXTURE','F0_NO_GRADIENT')
    near_oracle_ratio:float=1.05
    sizes:tuple=(40,80,120,156)
    variants:tuple=('REF','V4','V5_COMBINED','DCS_F0_OLD_GATE','DCS_F0_V5_GATE','DCS_FBEST_GATE','DCS_FBEST_STAGE')
