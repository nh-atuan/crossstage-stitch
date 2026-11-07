"""Q08: optimum-preserving reductions of a set-cover instance.

Rules, applied until a fixed point: forced singletons (and rows they cover),
duplicate rows, rows that are proper supersets of another row, columns that
cover no remaining row and, optionally, columns with identical incidence
(the lexicographically first config is kept). Every rule preserves the
minimum cover cardinality; duplicate-column merging does not preserve every
co-optimal catalogue. Column dominance is not used.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class InfeasibleInstance(ValueError):
    """A row has no candidate configuration."""


@dataclass
class Reduction:
    n_columns: int
    forced: list
    columns: list                      # remaining original column indices, ascending
    rows: dict                         # original row index -> frozenset of remaining columns
    representatives: dict = field(default_factory=dict)   # merged column -> kept column
    ledger: list = field(default_factory=list)

    @property
    def solved(self):
        return not self.rows

    def restore(self, selected):
        """Original column set from columns chosen on the reduced instance."""
        selected = set(selected)
        if not selected <= set(self.columns):
            raise ValueError("selection contains columns removed by the reduction")
        return sorted(set(self.forced) | selected)

    def to_dict(self, names):
        return {"forced": [names[c] for c in self.forced],
                "columns": [names[c] for c in self.columns],
                "rows": {str(i): sorted(names[c] for c in u) for i, u in sorted(self.rows.items())},
                "representatives": {names[a]: names[b] for a, b in sorted(self.representatives.items())},
                "ledger": [dict(step, columns=[names[c] for c in step.get("columns", [])])
                           for step in self.ledger]}


def reduce_cover(rows, n_columns, *, merge_duplicate_columns=True):
    active = {i: frozenset(u) for i, u in enumerate(rows)}
    if any(not u for u in active.values()):
        raise InfeasibleInstance("instance has an empty row")
    alive = set(range(n_columns))
    forced, representatives, ledger = set(), {}, []
    changed = True
    while changed:
        changed = False
        singles = sorted({next(iter(u)) for u in active.values() if len(u) == 1})
        if singles:
            forced.update(singles)
            covered = [i for i, u in active.items() if u & set(singles)]
            for i in covered:
                del active[i]
            ledger.append({"rule": "forced_singleton", "columns": singles, "rows": covered})
            changed = True
        seen, duplicates = {}, []
        for i in sorted(active):
            if active[i] in seen:
                duplicates.append(i)
            else:
                seen[active[i]] = i
        if duplicates:
            for i in duplicates:
                del active[i]
            ledger.append({"rule": "duplicate_row", "rows": duplicates})
            changed = True
        supersets = sorted(j for j in active for i in active
                           if i != j and active[i] < active[j])
        supersets = sorted(set(supersets))
        if supersets:
            for j in supersets:
                del active[j]
            ledger.append({"rule": "superset_row", "rows": supersets})
            changed = True
        used = set().union(*active.values()) if active else set()
        zero = sorted(alive - used - forced)
        if zero:
            alive -= set(zero)
            ledger.append({"rule": "zero_column", "columns": zero})
            changed = True
        if merge_duplicate_columns:
            groups = {}
            for c in sorted(alive - forced):
                groups.setdefault(frozenset(i for i, u in active.items() if c in u), []).append(c)
            merged = []
            for members in groups.values():
                for c in members[1:]:
                    representatives[c] = members[0]
                    merged.append(c)
            if merged:
                alive -= set(merged)
                active = {i: u & alive for i, u in active.items()}
                ledger.append({"rule": "duplicate_column", "columns": sorted(merged)})
                changed = True
    remaining = sorted(alive - forced)
    active = {i: u & set(remaining) for i, u in active.items()}
    if any(not u for u in active.values()):
        raise InfeasibleInstance("reduction produced an empty row")
    return Reduction(n_columns, sorted(forced), remaining, dict(sorted(active.items())),
                     representatives, ledger)
