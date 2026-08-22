"""The hero table: real-only vs standard augmentation vs the adaptive loop.

Three systems, one holdout, one operating point.

  A  baseline  real data only                      — what a team has today
  B  smote     real + off-the-shelf oversampling   — what a team tries first
  C  adaptive  real + the attacker-defender loop   — the claim

B exists to make C falsifiable. If adaptive generation does not beat naive oversampling on a
held-out attack family, the whole project reduces to an expensive way of duplicating rows, and
this table is where that shows up.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np

from afl.contract.metrics import MetricResult
from afl.contract.schema import Transaction
from afl.evaluation.leave_one_attack_out import DEFAULT_HOLDOUT, LeaveOneAttackOut
from afl.loop.closed_loop import run_closed_loop
from afl.tracking import InMemoryTracker
from afl.utils.seed import rng as make_rng

log = logging.getLogger(__name__)


# ── System B's augmentation ─────────────────────────────────────────────────────
def smote_transactions(
    txns: list[Transaction], ratio: float = 1.0, k: int = 5, seed: int = 1337
) -> list[Transaction]:
    """Interpolate between neighbouring fraud rows, per vector.

    Row-level SMOTE can move an amount and a timestamp. It cannot invent a new fan-in shape, a
    new pacing strategy, or a beneficiary that never existed — which is precisely the gap the
    adaptive system is claiming to fill.
    """
    r = make_rng(seed)
    by_vector: dict[str | None, list[Transaction]] = {}
    for t in txns:
        if t.is_fraud:
            by_vector.setdefault(t.vector_id, []).append(t)

    out: list[Transaction] = []
    for vid, rows in by_vector.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda t: t.ts)
        space = np.column_stack(
            [
                np.log1p([t.amount for t in rows]),
                np.array([t.ts.timestamp() for t in rows]) / 86_400.0,
            ]
        )
        space = (space - space.mean(0)) / (space.std(0) + 1e-9)
        dist = np.linalg.norm(space[:, None, :] - space[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        neighbours = np.argsort(dist, axis=1)[:, : min(k, len(rows) - 1)]

        for i in range(int(len(rows) * ratio)):
            a = rows[i % len(rows)]
            b = rows[int(r.choice(neighbours[i % len(rows)]))]
            u = float(r.random())
            out.append(
                a.model_copy(
                    update={
                        "txn_id": f"smote-{vid}-{i:06d}",
                        "amount": round(max(0.01, a.amount + u * (b.amount - a.amount)), 2),
                        "ts": a.ts + timedelta(seconds=u * (b.ts - a.ts).total_seconds()),
                        "attack_run_id": "smote",
                    }
                )
            )
    return out


# ── the table ───────────────────────────────────────────────────────────────────
@dataclass
class SystemResult:
    """One system's row in the three-system table."""

    name: str
    metrics: MetricResult
    operational: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_train_fraud: int = 0
    rounds: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "pr_auc": self.metrics.pr_auc,
            f"recall@{self.metrics.fixed_fpr:.0%}fpr": self.metrics.recall_at_fixed_fpr,
            f"precision@{self.metrics.k}": self.metrics.precision_at_k,
            "evasion_rate": round(self.operational.get("evasion_rate", float("nan")), 4),
            "train_rows": self.n_train,
            "train_fraud": self.n_train_fraud,
            "rounds": self.rounds,
        }


def _confine_to_training_window(simulator, train: list[Transaction]) -> None:
    """Shrink the loop's simulation window to end where training data ends.

    Left alone, the loop keeps generating across the simulator's full window — which straddles
    the evaluation period. System C would then be training on traffic contemporaneous with the
    data it is graded on, and the recall lift would be leakage wearing a convergence curve.
    """
    inner = getattr(simulator, "inner", simulator)
    if not (hasattr(inner, "start_ts") and hasattr(inner, "window_days")) or not train:
        log.warning(
            "cannot confine %r to the training window — check for temporal leakage",
            type(inner).__name__,
        )
        return
    days = (max(t.ts for t in train) - inner.start_ts).total_seconds() / 86_400
    inner.window_days = max(1, int(days))


def _measure(name: str, detector, evaluator: LeaveOneAttackOut, train, **extra) -> SystemResult:
    return SystemResult(
        name=name,
        metrics=evaluator.leave_one_attack_out(detector),
        operational=evaluator.operational(detector),
        n_train=len(train),
        n_train_fraud=sum(1 for t in train if t.is_fraud),
        **extra,
    )


def run_three_systems(
    pool: list[Transaction],
    detector_factory: Callable[[], Any],
    simulator=None,
    optimiser=None,
    held_out_vector: str = DEFAULT_HOLDOUT,
    rounds: int = 10,
    smote_ratio: float = 1.0,
    train_frac: float = 0.7,
    embargo_days: float = 1.0,
    seed: int = 1337,
    tracker=None,
    real_vectors: tuple[str, ...] = (),
) -> list[SystemResult]:
    """All three systems, same split, same operating point, fresh detector each time.

    `pool` is real transactions plus one batch per attack vector; `simulator`/`optimiser` are
    only needed for System C.

    `real_vectors` names synthetic families that stand in for "fraud the team already has labels
    for" — needed when there is no labelled real dataset, or System A has nothing to learn from
    and the control is vacuous rather than merely weak.
    """
    evaluator, train = LeaveOneAttackOut.from_pool(
        pool, held_out_vector=held_out_vector, train_frac=train_frac, embargo_days=embargo_days
    )
    # what a team already had before any of this: real rows plus the families they have labels for
    historical = [t for t in train if t.vector_id is None or t.vector_id in real_vectors]
    if not any(t.is_fraud for t in historical):
        log.warning(
            "System A has no fraud to train on — it will score 0.0 for lack of a model, not for "
            "lack of skill. Use a labelled dataset, or name known families in `real_vectors`."
        )
    results: list[SystemResult] = []

    a = detector_factory()
    a.fit(historical)
    results.append(_measure("A_baseline", a, evaluator, historical))

    # B — the same rows, oversampled. It can only oversample what System A actually had.
    smote_train = historical + smote_transactions(historical, ratio=smote_ratio, seed=seed)
    b = detector_factory()
    b.fit(smote_train)
    results.append(_measure("B_smote", b, evaluator, smote_train))

    # C — the same starting rows, plus whatever the loop generates. The loop is the only
    # difference between C and A; anything else in the diff would confound the claim.
    if simulator is not None and optimiser is not None:
        if getattr(optimiser, "vector_id", None) == held_out_vector:
            raise ValueError(
                f"optimiser is searching {held_out_vector!r}, which is the held-out family — "
                "System C would be training on the answer"
            )
        _confine_to_training_window(simulator, historical)
        c = detector_factory()
        c.fit(historical)
        loop_tracker = tracker or InMemoryTracker("system_c")
        run_closed_loop(simulator, optimiser, c, evaluator, rounds=rounds, tracker=loop_tracker)
        results.append(
            _measure(
                "C_adaptive", c, evaluator, historical, rounds=rounds, history=loop_tracker.history
            )
        )

    return results


def to_frame(results: list[SystemResult]):
    """The table as a dataframe."""
    import pandas as pd

    return pd.DataFrame([r.row() for r in results])


def to_markdown(results: list[SystemResult]) -> str:
    """The table as markdown, ready to paste into the deck."""
    rows = [r.row() for r in results]
    if not rows:
        return "_no systems run_"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]}" for c in cols) + " |")
    return "\n".join(lines)


def lift(results: list[SystemResult], metric: str = "recall_at_fixed_fpr") -> dict[str, float]:
    """C's improvement over A and over B — the two numbers the claim stands or falls on."""
    by_name = {r.name: getattr(r.metrics, metric) for r in results}
    c = by_name.get("C_adaptive")
    if c is None:
        return {}
    return {
        "vs_baseline": round(c - by_name.get("A_baseline", 0.0), 6),
        "vs_smote": round(c - by_name.get("B_smote", 0.0), 6),
    }
