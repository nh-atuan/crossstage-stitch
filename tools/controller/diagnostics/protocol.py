"""Predeclared diagnostic budgets; evaluation outcomes never tune these values."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Protocol:
    seed: int = 20261101
    outer_folds: int = 5
    rounds: int = 64
    depth: int = 2
    sampling_seeds: tuple = (20261101, 20261102, 20261103)
    requested_sizes: tuple = (40, 80, 120, 160)
    thresholds: tuple = (.85, .70, .50)
    gate_thresholds: tuple = (.65, .85)
    top_k: tuple = (1, 3, 5)
    near_oracle_ratio: float = 1.10
    bins: int = 10
    bootstrap_replicates: int = 1000
    # Pilot interpretation criteria, never runtime acceptance thresholds.
    weak_recall: float = .80
    weak_auc: float = .70
    material_ece: float = .10
    material_brier_gain: float = .02
    material_replay_gain: float = .05

    def mapping(self):
        return dict(asdict(self), evaluation_role='P0-P5 evaluation already seen; descriptive only',
            learning_sizes='Cap at available outer-training groups; never train on outer evaluation.',
            bootstrap_unit='image-sharing component; all configurations move together; no model refit in CI',
            capacity=f'Fixed depth {self.depth} / {self.rounds} trees; no capacity search or production changes',
            gpu_sweep='NEVER launched by this workflow',
            criteria_role='Predeclared pilot diagnostics; no evaluation-driven selection or online guarantee')
