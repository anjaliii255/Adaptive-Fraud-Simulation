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

import json
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from afl.contract.metrics import MetricResult
from afl.contract.schema import Transaction
from afl.data.splits import CommittedSplit
from afl.evaluation import leave_one_attack_out as loao
from afl.evaluation import protocol
from afl.evaluation.leave_one_attack_out import DEFAULT_HOLDOUT, LeaveOneAttackOut
from afl.loop.closed_loop import find_evasions, run_closed_loop
from afl.tracking import InMemoryTracker
from afl.utils.runcard import with_provenance
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
    #: Backend, params and what the fit saw — the provenance of this row. Kept out of `row()`
    #: so it reaches the run artefact without turning the hero table into a config dump.
    model_card: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "pr_auc": self.metrics.pr_auc,
            f"recall@{self.metrics.fixed_fpr:.0%}fpr": self.metrics.recall_at_fixed_fpr,
            f"precision@{self.metrics.k}": self.metrics.precision_at_k,
            "evasion_rate": round(self.operational.get("evasion_rate", float("nan")), 4),
            # The realised cost of catching it. Under a cost-derived policy the FPR is an
            # OUTPUT, not a target, so the column that reconciles `recall@FPR` with what the
            # policy actually did has to be in the table rather than in a sibling artefact.
            "friction_rate": round(self.operational.get("friction_rate", float("nan")), 4),
            "train_rows": self.n_train,
            "train_fraud": self.n_train_fraud,
            "rounds": self.rounds,
        }

    @property
    def backend(self) -> str:
        card = self.model_card.get("supervised", self.model_card)
        b = card.get("backend") or {}
        return f"{b.get('name', 'unknown')} {b.get('version', '')}".strip()


def confine_to_training_window(simulator, train: list[Transaction]) -> None:
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


def _card(detector) -> dict[str, Any]:
    """A detector's model card, duck-typed.

    Duck-typed rather than imported: `afl.evaluation` measures detectors, it does not know how
    any of them are built, and a concrete import here would be the first thread of that coupling.

    A detector's *own* card wins when it has one. The wrapper below is the fallback for a
    composite that does not — before ticket 10 it was the only path, so an ensemble's card was
    its supervised half's card under a different `detector` name and the anomaly layer that
    produced 30% of every blended score did not appear in the run artefact at all.
    """
    own = getattr(detector, "model_card", None)
    if callable(own):
        return own()
    inner = getattr(detector, "supervised", detector)
    card = getattr(inner, "model_card", None)
    if not callable(card):
        return {"detector": type(detector).__name__}
    return {"detector": type(detector).__name__, "supervised": card()}


def plain_fit(detector, rows: list[Transaction]) -> None:
    """The default `fit_detector`: fit, and leave the configured action bands alone."""
    detector.fit(rows)


def measure(name: str, detector, evaluator: LeaveOneAttackOut, train, **extra) -> SystemResult:
    return SystemResult(
        name=name,
        metrics=evaluator.leave_one_attack_out(detector),
        operational=evaluator.operational(detector),
        n_train=len(train),
        n_train_fraud=sum(1 for t in train if t.is_fraud),
        model_card=_card(detector),
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
    split: CommittedSplit | None = None,
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
    fit_detector: Callable[[Any, list[Transaction]], None] = plain_fit,
) -> list[SystemResult]:
    """All three systems, same split, same operating point, fresh detector each time.

    `pool` is real transactions plus one batch per attack vector; `simulator`/`optimiser` are
    only needed for System C.

    `fixed_fpr` and `k` are the operating point, and they are arguments rather than defaults
    picked up here so that one config value reaches all three rows. Two systems compared at two
    thresholds is not a comparison.

    `fit_detector` is how a system is trained, applied identically to all three. It is a hook
    because calibrating the action bands needs a validation split of the training rows, and a
    system whose bands were set differently from its neighbour's is not on the same operating
    point either — even when its metrics are.

    `real_vectors` names synthetic families that stand in for "fraud the team already has labels
    for" — needed when there is no labelled real dataset, or System A has nothing to learn from
    and the control is vacuous rather than merely weak.
    """
    evaluator, train = LeaveOneAttackOut.from_pool(
        pool,
        held_out_vector=held_out_vector,
        train_frac=train_frac,
        embargo_days=embargo_days,
        split=split,
        fixed_fpr=fixed_fpr,
        k=k,
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
    fit_detector(a, historical)
    results.append(measure("A_baseline", a, evaluator, historical))

    # B — the same rows, oversampled. It can only oversample what System A actually had.
    smote_train = historical + smote_transactions(historical, ratio=smote_ratio, seed=seed)
    b = detector_factory()
    fit_detector(b, smote_train)
    results.append(measure("B_smote", b, evaluator, smote_train))

    # C — the same starting rows, plus whatever the loop generates. The loop is the only
    # difference between C and A; anything else in the diff would confound the claim.
    if simulator is not None and optimiser is not None:
        if getattr(optimiser, "vector_id", None) == held_out_vector:
            raise ValueError(
                f"optimiser is searching {held_out_vector!r}, which is the held-out family — "
                "System C would be training on the answer"
            )
        confine_to_training_window(simulator, historical)
        c = detector_factory()
        fit_detector(c, historical)
        loop_tracker = tracker or InMemoryTracker("system_c")
        run_closed_loop(simulator, optimiser, c, evaluator, rounds=rounds, tracker=loop_tracker)
        results.append(
            measure(
                "C_adaptive", c, evaluator, historical, rounds=rounds, history=loop_tracker.history
            )
        )

    return results


def to_frame(results: list[SystemResult]):
    """The table as a dataframe."""
    import pandas as pd

    return pd.DataFrame([r.row() for r in results])


def to_markdown(results: list[SystemResult]) -> str:
    """The table as markdown, ready to paste into the deck.

    The backend goes underneath it rather than in a column: it is the same for every row, and
    "LightGBM" naming a table that a fallback actually produced is the misreading ticket 08
    exists to prevent.
    """
    rows = [r.row() for r in results]
    if not rows:
        return "_no systems run_"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]}" for c in cols) + " |")

    backends = sorted({r.backend for r in results if r.backend})
    if backends:
        lines.append("")
        lines.append(f"_detector backend: {', '.join(backends)}_")
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


# ══════════════════════════════════════════════════════════════════════════════════
# Ticket 16 — the table as a committed artefact, over seeds, in two columns.
#
# Everything above builds one table from one run. What a claim needs is narrower and harder:
# the same three systems, on the same guarded fold, over several seeds, reported in *both*
# columns — the family nobody trained on and the families everybody did — with the run-to-run
# spread next to every number so a two-point gap is not read as a result.
#
# Ticket 11's carry-out is the reason the machinery below is not just a mean: the held-out fold
# it hands this table is withheld on PaySim and vacuous on AMLSim. So the held-out column gets
# the same treatment every row of that matrix gets — outcome, reason, and numbers that move out
# of `metrics` when they cannot be quoted — rather than a bare float in a slide.
# ══════════════════════════════════════════════════════════════════════════════════

#: The two questions one holdout has to answer. `unseen` is the claim: recall on the family no
#: system trained on. `known` is the price of it: what the same detector still does on the
#: families it did train on, measured on the same window and the same haystack. A system that
#: buys the first by giving up the second has traded rather than improved, and a one-column
#: table cannot see the trade — which is why both are reported or neither is.
UNSEEN = "unseen"
KNOWN = "known"
COLUMNS = (KNOWN, UNSEEN)

#: The three rows, in the order they are argued. Fixed here so the artefact, the document and
#: the figure cannot disagree about what System B is called.
SYSTEMS = ("A_baseline", "B_smote", "C_adaptive")

#: Bump when the fields change shape, so an old file fails loudly instead of being read with the
#: wrong meaning. Same discipline as `LOAO_ARTEFACT_VERSION`.
THREE_SYSTEM_ARTEFACT_VERSION = 1

DEFAULT_TABLE_DIR = Path(os.getenv("AFL_TABLE_DIR", "artifacts/three_system"))


# ── the two columns ─────────────────────────────────────────────────────────────
def known_column(fold: loao.Fold, known_vectors: tuple[str, ...] = ()) -> list[Transaction]:
    """The attacks every system *did* train on, on the same test window as the holdout.

    The mirror image of the fold's holdout, and defined by what reached training rather than by
    what happens to be in the pool. The holdout keeps the held-out family's fraud and drops every
    other positive; this column keeps the fraud the systems were trained on — the anchor's own
    labelled fraud, which is the most known attack there is, plus any family named in
    `known_fraud_vectors` for an anchor that has no labels of its own — and drops the rest.

    The families the pool carries but nobody trained on belong in neither column: they are not
    the claim and they are not the control, and counting them as negatives would label real
    fraud as legit traffic.

    The legit rows are the same rows in both, which is what makes the two columns comparable at
    one operating point, and it is asserted rather than assumed — see `assert_same_haystack`.
    """
    if fold.held_out_vector in known_vectors:
        raise ValueError(
            f"{fold.held_out_vector!r} is the held-out family and is also named as a known one — "
            "the known column would then contain the family the unseen column is measuring"
        )
    return [
        t
        for t in fold.test_side
        if not t.is_fraud or t.vector_id is None or t.vector_id in known_vectors
    ]


def assert_same_haystack(known: list[Transaction], unseen: list[Transaction]) -> dict[str, Any]:
    """Both columns are scored against the same legit traffic, or they are not one table.

    A fixed-FPR threshold is a quantile of the negatives. Two columns with two different sets of
    negatives are two operating points, and the difference between them would then be partly a
    difference of denominator — the exact confusion the fixed operating point exists to prevent.
    """
    a = {t.txn_id for t in known if not t.is_fraud}
    b = {t.txn_id for t in unseen if not t.is_fraud}
    if a != b:
        raise loao.GuardFailed(
            f"the two columns do not share a haystack: {len(a - b)} legit row(s) only in "
            f"{KNOWN}, {len(b - a)} only in {UNSEEN} — recall at a fixed FPR is a quantile of "
            "the negatives, so two haystacks are two operating points wearing one table"
        )
    return {"legit_rows": len(a), "shared": True}


def column_counts(rows: list[Transaction]) -> dict[str, Any]:
    """What a column is made of — and how much of it the generator wrote.

    `positives_anchor_own` is the field that decides whether a column can carry a claim at all.
    Where it is zero, every positive is an injected row and every negative is a real one, so
    "caught the fraud" and "spotted the synthetic row" are the same label. That is ticket 11's
    finding, and it applies to any column, not only to the held-out one.
    """
    positives = [t for t in rows if t.is_fraud]
    injected = [t for t in positives if t.vector_id is not None]
    return {
        "rows": len(rows),
        "positives": len(positives),
        "positives_injected": len(injected),
        "positives_anchor_own": len(positives) - len(injected),
        "legit": len(rows) - len(positives),
        "base_rate": round(len(positives) / len(rows), 8) if rows else 0.0,
        "families": sorted({t.vector_id for t in positives if t.vector_id}),
    }


@dataclass(frozen=True)
class ColumnResult:
    """One cell block: an outcome, a reason, and numbers only when they may be quoted.

    The same three-state contract as `loao.FoldResult`, applied to the *known* column — which
    the leave-one-attack-out harness never measures, because it is not a carve-out. The held-out
    column of this table is a `FoldResult` proper; this type is its counterpart, and the two are
    read the same way on purpose.
    """

    column: str
    outcome: str
    reason: str = ""
    metrics: MetricResult | None = None
    withheld_metrics: MetricResult | None = None
    operational: dict[str, float] = field(default_factory=dict)
    floor: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in loao.OUTCOMES:
            raise ValueError(
                f"unknown column outcome {self.outcome!r}; expected one of {loao.OUTCOMES}"
            )
        if self.outcome != loao.MEASURED and self.metrics is not None:
            raise ValueError(
                f"{self.column}: a {self.outcome} column carries numbers under "
                "`withheld_metrics`, never under `metrics` — that field is what a reader quotes"
            )
        if self.outcome != loao.MEASURED and not self.reason:
            raise ValueError(
                f"{self.column}: a column that is not measured has to say why — a "
                f"{self.outcome} column with no reason is indistinguishable from a forgotten one"
            )

    @property
    def reported(self) -> bool:
        return self.outcome == loao.MEASURED

    @property
    def any_metrics(self) -> MetricResult | None:
        """Whichever block exists — for a document that prints withheld numbers under a warning."""
        return self.metrics or self.withheld_metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "outcome": self.outcome,
            "reason": self.reason,
            "metrics": self.metrics.model_dump() if self.metrics else None,
            "withheld_metrics": self.withheld_metrics.model_dump()
            if self.withheld_metrics
            else None,
            "operational": self.operational,
            "floor": self.floor,
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ColumnResult:
        return cls(
            column=raw["column"],
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
            metrics=MetricResult(**raw["metrics"]) if raw.get("metrics") else None,
            withheld_metrics=MetricResult(**raw["withheld_metrics"])
            if raw.get("withheld_metrics")
            else None,
            operational=raw.get("operational", {}),
            floor=raw.get("floor", {}),
            counts=raw.get("counts", {}),
        )


def measure_known_column(
    detector,
    fold: loao.Fold,
    rows: list[Transaction] | None = None,
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
    min_positives: int = loao.MIN_MEANINGFUL_POSITIVES,
    known_vectors: tuple[str, ...] = (),
) -> ColumnResult:
    """Score a fitted detector on the families it trained on, and say what the number is worth.

    Deliberately not a leave-one-attack-out fold: nothing is held out here, so this column
    measures memorisation as much as skill. That is the point of it — it is the control on the
    other column, not a second claim — and it is why the `known` cell is never quoted on its own.
    """
    rows = known_column(fold, known_vectors) if rows is None else rows
    counts = column_counts(rows)
    if not counts["positives"]:
        return ColumnResult(
            column=KNOWN,
            outcome=loao.SKIPPED,
            reason="the test window carries no fraud outside the held-out family — there is "
            "nothing for a known-attack column to measure",
            counts=counts,
        )

    scores = protocol.score_transactions(detector, rows, run_id="three-system-known")
    y, s = protocol.align(rows, scores)
    result = protocol.evaluate(y, s, fixed_fpr, k)
    measured = {
        "operational": {
            key: round(v, 6) for key, v in protocol.operational_rates(rows, scores).items()
        },
        "floor": loao.amount_floor(fold.train, rows, fixed_fpr, k),
        "counts": counts,
    }

    def withheld(reason: str) -> ColumnResult:
        return ColumnResult(
            column=KNOWN,
            outcome=loao.WITHHELD,
            reason=reason,
            withheld_metrics=result,
            **measured,
        )

    if counts["positives"] < min_positives:
        return withheld(
            f"{counts['positives']} positives against a floor of {min_positives} — recall moves "
            f"{1 / counts['positives']:.1%} per row here, so this is reported as missing rather "
            "than as a low score"
        )
    if not counts["positives_anchor_own"]:
        return withheld(
            "every positive in this column is an injected row and every negative is real, so it "
            "cannot tell detection apart from provenance any more than the held-out column can. "
            "An anchor with its own labelled fraud is what makes this column mean something"
        )
    return ColumnResult(column=KNOWN, outcome=loao.MEASURED, metrics=result, **measured)


# ── System C, with the audit gate respected ─────────────────────────────────────
@dataclass
class AdaptiveRun:
    """What the loop produced, and what it was refused for."""

    rows: list[Transaction] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    rejected: int = 0

    @property
    def evasion_trajectory(self) -> list[float]:
        return [float(h.get("evasion_rate", 0.0)) for h in self.history]


def run_adaptive_loop(
    simulator,
    optimiser,
    detector,
    rounds: int,
    evaluator: LeaveOneAttackOut | None = None,
    tracker=None,
) -> AdaptiveRun:
    """attack → detect → log evasions → mutate → retrain, keeping only what the audit allowed.

    The one difference from `run_closed_loop`, and the reason this exists beside it: a batch the
    commensurability audit rejected is **not** trained on. `MultiVectorOptimiser` already refuses
    to score such a candidate — it is the gate that stops the search being rewarded for drifting
    off-anchor — but the loop above retrains on every batch it is handed, so a rejected candidate
    still reached the detector's corpus. In a table whose whole purpose is to ask whether
    System C beat System B honestly, training on rows the anchor can pick out by one field is the
    one way to win that would not mean anything.

    An optimiser with no trials to inspect is treated as accepting everything, so a stub
    optimiser still drives the loop.
    """
    run = AdaptiveRun(rounds=rounds)
    for r in range(rounds):
        batch = simulator.generate(optimiser.propose())
        scores = detector.score(batch)
        evasions = find_evasions(batch, scores)
        optimiser.update(evasions)

        trial = (getattr(optimiser, "trials", None) or [None])[-1]
        rejected = bool(getattr(trial, "rejected", False))
        fraud = [t for t in batch.transactions if t.is_fraud]
        if rejected:
            run.rejected += 1
        else:
            run.rows.extend(fraud)
            detector.retrain(batch, evasions)

        record: dict[str, Any] = {
            "round": r,
            "vector_id": batch.params.vector_id,
            "n_transactions": len(batch.transactions),
            "n_fraud": len(fraud),
            "n_evasions": len(evasions),
            # over fraud rows, never over all rows — see afl/contract/metrics.py
            "evasion_rate": (len(evasions) / len(fraud)) if fraud else 0.0,
            "rejected_by_audit": rejected,
            "realism_penalty": float(getattr(trial, "realism_penalty", 0.0) or 0.0),
            "audit_score": float(getattr(trial, "audit_score", 0.0) or 0.0),
            "audit_base_rate": float(getattr(trial, "audit_base_rate", 0.0) or 0.0),
            "audit_worst": getattr(trial, "audit_worst", None),
            # what each gate rule said about this batch, not only the one in force. The two
            # disagree as a function of anchor size, and a run that recorded only the verdict it
            # acted on would have to be repeated to answer the obvious next question.
            "audit_rule": getattr(trial, "audit_rule", None),
            "rejected_by_lift": bool(getattr(trial, "rejected_by_lift", False)),
            "rejected_by_envelope": bool(getattr(trial, "rejected_by_envelope", False)),
            "fitness": float(getattr(trial, "fitness", 0.0) or 0.0),
            "allocation": dict(getattr(trial, "allocation", {}) or {}),
            "rows_kept": len(run.rows),
        }
        if evaluator is not None:
            # the convergence curve: what the held-out family looks like to the detector the
            # attacker is currently probing. Measured on the fold, so it is the same number the
            # table reports — a curve measured on anything else is a different claim.
            record.update(evaluator.leave_one_attack_out(detector).model_dump())
        if tracker is not None:
            tracker.log(**record)
        run.history.append(record)
        log.info(
            "round %d/%d: %d fraud rows, evasion %.3f, realism penalty %.3f%s",
            r + 1,
            rounds,
            len(fraud),
            record["evasion_rate"],
            record["realism_penalty"],
            "  [REJECTED BY THE AUDIT GATE]" if rejected else "",
        )
    return run


# ── one seed's three rows ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class SystemRow:
    """One system, one seed, both columns."""

    name: str
    known: ColumnResult
    unseen: loao.FoldResult
    n_train: int = 0
    n_train_fraud: int = 0
    #: Rows the augmentation added. SMOTE's copies for B, the loop's accepted output for C, and
    #: zero for A — the column that says how much of the difference is data rather than method.
    n_generated: int = 0
    rounds: int = 0
    rejected_rounds: int = 0
    model_card: dict[str, Any] = field(default_factory=dict)
    loop: list[dict[str, Any]] = field(default_factory=list)

    def column(self, column: str) -> ColumnResult | loao.FoldResult:
        if column == KNOWN:
            return self.known
        if column == UNSEEN:
            return self.unseen
        raise ValueError(f"unknown column {column!r}; expected one of {COLUMNS}")

    @property
    def backend(self) -> str:
        card = self.model_card.get("supervised", self.model_card)
        b = card.get("backend") or {}
        return f"{b.get('name', 'unknown')} {b.get('version', '')}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "known": self.known.to_dict(),
            "unseen": self.unseen.to_dict(),
            "n_train": self.n_train,
            "n_train_fraud": self.n_train_fraud,
            "n_generated": self.n_generated,
            "rounds": self.rounds,
            "rejected_rounds": self.rejected_rounds,
            "model_card": self.model_card,
            "loop": self.loop,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SystemRow:
        return cls(
            name=raw["system"],
            known=ColumnResult.from_dict(raw["known"]),
            unseen=loao.FoldResult.from_dict(raw["unseen"]),
            n_train=int(raw.get("n_train", 0)),
            n_train_fraud=int(raw.get("n_train_fraud", 0)),
            n_generated=int(raw.get("n_generated", 0)),
            rounds=int(raw.get("rounds", 0)),
            rejected_rounds=int(raw.get("rejected_rounds", 0)),
            model_card=raw.get("model_card", {}),
            loop=raw.get("loop", []),
        )


@dataclass(frozen=True)
class SeedRun:
    """One seed: three systems on one carve-out, with the guards that carve-out passed."""

    seed: int
    systems: list[SystemRow]
    counts: dict[str, Any] = field(default_factory=dict)
    guards: dict[str, Any] = field(default_factory=dict)
    separability: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    #: The same question as `provenance`, asked with System C's advantage: a model that learns
    #: "the generator wrote this row" from the families the loop produced, applied to the family
    #: nobody trained on. The fold probe learns from the holdout's own handful of positives;
    #: System C learns from thousands, and ticket 11's carry-out is explicit that a low fold
    #: probe on a thin fold is weak evidence of anything. Recorded for every seed, and it is
    #: what judges System C's held-out cell whenever it is the more damning of the two.
    loop_provenance: dict[str, Any] | None = None
    seconds: float = 0.0

    def __post_init__(self) -> None:
        names = [s.name for s in self.systems]
        if len(set(names)) != len(names):
            raise ValueError(f"seed {self.seed}: two rows share a system name ({names})")

    def system(self, name: str) -> SystemRow | None:
        return next((s for s in self.systems if s.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "systems": [s.to_dict() for s in self.systems],
            "counts": self.counts,
            "guards": self.guards,
            "separability": self.separability,
            "provenance": self.provenance,
            "loop_provenance": self.loop_provenance,
            "seconds": round(self.seconds, 2),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SeedRun:
        return cls(
            seed=int(raw["seed"]),
            systems=[SystemRow.from_dict(s) for s in raw["systems"]],
            counts=raw.get("counts", {}),
            guards=raw.get("guards", {}),
            separability=raw.get("separability"),
            provenance=raw.get("provenance"),
            loop_provenance=raw.get("loop_provenance"),
            seconds=float(raw.get("seconds", 0.0)),
        )


# ── the artefact ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ThreeSystemReport:
    """One anchor's table, every seed of it, with the config that produced the numbers.

    The same division as `artifacts/loao/` and `artifacts/splits/`: config holds the inputs,
    the artefact holds the decision. Nothing downstream — document, figure, README — is allowed
    to hold a number that is not in here.
    """

    dataset: str
    held_out_vector: str
    config: dict[str, Any]
    operating_point: dict[str, Any]
    runs: list[SeedRun]
    split: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    version: int = THREE_SYSTEM_ARTEFACT_VERSION

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError(
                f"{self.dataset}: a three-system report with no runs is not a table — a run "
                "where every system failed still has to name them and say why"
            )
        shapes = {tuple(s.name for s in r.systems) for r in self.runs}
        if len(shapes) > 1:
            raise ValueError(
                f"{self.dataset}: the seeds did not run the same systems ({sorted(shapes)}) — "
                "averaging across them would compare a different table per seed"
            )

    @property
    def seeds(self) -> list[int]:
        return [r.seed for r in self.runs]

    @property
    def systems(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.runs[0].systems)

    def cells(self, system: str, column: str) -> list[ColumnResult | loao.FoldResult]:
        """One system's cell in one column, once per seed. Missing seeds are dropped, not zeroed."""
        out = []
        for run in self.runs:
            row = run.system(system)
            if row is not None:
                out.append(row.column(column))
        return out

    def rows_of(self, system: str) -> list[SystemRow]:
        return [r.system(system) for r in self.runs if r.system(system) is not None]

    @property
    def backends(self) -> list[str]:
        return sorted({row.backend for run in self.runs for row in run.systems if row.backend})

    def summary(self) -> str:
        head = compare(self)
        return (
            f"{self.dataset}: {len(self.runs)} seed(s), holdout {self.held_out_vector} — "
            f"{head.verdict if head else 'no comparison available'}"
        )

    # ── round trip ──────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset": self.dataset,
            "held_out_vector": self.held_out_vector,
            "config": self.config,
            "operating_point": self.operating_point,
            "split": self.split,
            "data": self.data,
            "runs": [r.to_dict() for r in self.runs],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ThreeSystemReport:
        if int(raw.get("version", 0)) != THREE_SYSTEM_ARTEFACT_VERSION:
            raise ValueError(
                f"three-system artefact version {raw.get('version')} != "
                f"{THREE_SYSTEM_ARTEFACT_VERSION}; rebuild it with "
                "scripts/build_three_system.py rather than reading it as-is"
            )
        return cls(
            dataset=raw["dataset"],
            held_out_vector=raw["held_out_vector"],
            config=raw.get("config", {}),
            operating_point=raw.get("operating_point", {}),
            runs=[SeedRun.from_dict(r) for r in raw["runs"]],
            split=raw.get("split", {}),
            data=raw.get("data", {}),
            meta=raw.get("meta", {}),
        )

    def save(self, directory: str | Path = DEFAULT_TABLE_DIR) -> Path:
        path = Path(directory) / f"{self.dataset}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(with_provenance(self.to_dict()), indent=2, default=str) + "\n")
        return path

    @classmethod
    def load(cls, dataset: str, directory: str | Path = DEFAULT_TABLE_DIR) -> ThreeSystemReport:
        path = Path(directory) / f"{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no three-system table for {dataset!r} at {path} — run `make table` once and "
                "commit the result"
            )
        return cls.from_dict(json.loads(path.read_text()))


def load_all(directory: str | Path = DEFAULT_TABLE_DIR) -> dict[str, ThreeSystemReport]:
    """Every committed table on disk, keyed by dataset."""
    directory = Path(directory)
    if not directory.exists():
        return {}
    return {
        path.stem: ThreeSystemReport.from_dict(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
    }


# ── reading the table ───────────────────────────────────────────────────────────
#: How close a provenance-only model has to come to System C's score before the score is read
#: as provenance rather than detection. Two points of PR-AUC: inside the seed-to-seed spread of
#: every cell in this table, so a probe that lands there has reproduced the number.
PROVENANCE_MARGIN = 0.02

#: The metric the project's single claim is written in — "improves detection recall on a
#: held-out attack family". PR-AUC is reported beside it everywhere, and a win has to show in
#: both before it is called one.
HEADLINE_METRIC = "recall_at_fixed_fpr"
REPORTED_METRICS = ("pr_auc", "recall_at_fixed_fpr", "precision_at_k")


@dataclass(frozen=True)
class Spread:
    """One cell of the table across seeds: the mean, and the spread that qualifies it.

    The spread is not decoration. On this fold the seed-to-seed range has been larger than every
    gap between systems, and a mean quoted without it turns noise into a result.
    """

    system: str
    column: str
    metric: str
    values: list[float]
    outcome: str
    reason: str = ""

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return round(float(np.mean(self.values)), 6) if self.values else float("nan")

    @property
    def sd(self) -> float:
        """Sample standard deviation. Zero on one seed, which is a statement about the run."""
        return round(float(np.std(self.values, ddof=1)), 6) if len(self.values) > 1 else 0.0

    @property
    def lo(self) -> float:
        return round(min(self.values), 6) if self.values else float("nan")

    @property
    def hi(self) -> float:
        return round(max(self.values), 6) if self.values else float("nan")

    @property
    def reported(self) -> bool:
        return self.outcome == loao.MEASURED

    def text(self, dp: int = 3) -> str:
        """`0.271 ± 0.014`, in brackets when the number may not be quoted."""
        if not self.values:
            return "—"
        body = f"{self.mean:.{dp}f} ± {self.sd:.{dp}f}" if self.n > 1 else f"{self.mean:.{dp}f}"
        return body if self.reported else f"[{body}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "column": self.column,
            "metric": self.metric,
            "outcome": self.outcome,
            "reason": self.reason,
            "n_seeds": self.n,
            "mean": self.mean,
            "sd": self.sd,
            "lo": self.lo,
            "hi": self.hi,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Spread:
        """The constructor fields only — `mean`, `sd`, `lo` and `hi` are recomputed from them.

        Reading them back off the file instead would let a hand-edited artefact carry a mean its
        own values do not support, which is the one thing a committed number must not be able
        to do.
        """
        return cls(
            system=raw["system"],
            column=raw["column"],
            metric=raw["metric"],
            values=[float(v) for v in raw.get("values", [])],
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
        )


def spread_of(
    report: ThreeSystemReport, system: str, column: str, metric: str = HEADLINE_METRIC
) -> Spread:
    """One system's cell in one column, across every seed that ran it.

    Withheld numbers are aggregated too — they are the evidence for the verdict — but the
    aggregate inherits the withholding: one seed that could not carry a claim is enough, because
    a mean of two quotable seeds and one withheld one is not a quotable number.
    """
    cells = report.cells(system, column)
    values, outcomes, reasons = [], [], []
    for c in cells:
        m = c.any_metrics
        if m is None:
            outcomes.append(loao.SKIPPED)
            reasons.append(c.reason)
            continue
        values.append(round(float(getattr(m, metric)), 6))
        outcomes.append(c.outcome)
        reasons.append(c.reason)

    if not outcomes:
        outcome, reason = loao.SKIPPED, "this system did not run on any seed"
    elif all(o == loao.MEASURED for o in outcomes):
        outcome, reason = loao.MEASURED, ""
    elif any(o == loao.WITHHELD for o in outcomes):
        outcome = loao.WITHHELD
        reason = next(
            (r for o, r in zip(outcomes, reasons, strict=False) if r and o != loao.MEASURED), ""
        )
    else:
        outcome = loao.SKIPPED
        reason = next((r for r in reasons if r), "")
    return Spread(
        system=system, column=column, metric=metric, values=values, outcome=outcome, reason=reason
    )


def summarise(
    report: ThreeSystemReport, metric: str = HEADLINE_METRIC
) -> dict[str, dict[str, Spread]]:
    """`{system: {column: Spread}}` — the whole table as numbers plus their spread."""
    return {
        system: {column: spread_of(report, system, column, metric) for column in COLUMNS}
        for system in report.systems
    }


def floor_spread(report: ThreeSystemReport, column: str, metric: str = HEADLINE_METRIC) -> Spread:
    """The amount floor for a column, across seeds. No model, no features, no training.

    Read off the first system's cells because the floor is a property of the *fold*, not of any
    detector: it is the same number under every row. A system that does not clear it has not
    detected anything, which is a verdict two results in this repo were walked back for want of.
    """
    values = []
    for cell in report.cells(report.systems[0] if report.systems else "", column):
        value = (cell.floor or {}).get(metric)
        if value is not None:
            values.append(round(float(value), 6))
    return Spread(
        system="amount_floor", column=column, metric=metric, values=values, outcome=loao.MEASURED
    )


def sign_test(wins: int, trials: int) -> float:
    """One-sided binomial p: how surprising is `wins` of `trials` if the direction were a coin?

    The seed-to-seed spread on this fold is larger than the gap between systems, which makes the
    mean the wrong summary — it is dominated by whichever seed swung furthest. Counting which way
    each seed fell, and asking whether that count could be chance, is what the data can support.
    Three seeds cannot reach p < 0.05 by construction; that is a fact about the run, and it is
    reported rather than worked around.
    """
    if trials <= 0:
        return 1.0
    return sum(math.comb(trials, i) for i in range(wins, trials + 1)) / 2**trials


@dataclass(frozen=True)
class Comparison:
    """Challenger minus incumbent, seed by seed, with the spread and the sign test."""

    challenger: str
    incumbent: str
    column: str
    metric: str
    per_seed: list[dict[str, Any]]
    outcome: str
    reason: str = ""

    @property
    def deltas(self) -> list[float]:
        return [d["delta"] for d in self.per_seed]

    @property
    def n(self) -> int:
        return len(self.per_seed)

    @property
    def mean_delta(self) -> float:
        return round(float(np.mean(self.deltas)), 6) if self.per_seed else float("nan")

    @property
    def sd_delta(self) -> float:
        return round(float(np.std(self.deltas, ddof=1)), 6) if self.n > 1 else 0.0

    @property
    def wins(self) -> int:
        return sum(1 for d in self.deltas if d > 0)

    @property
    def p_value(self) -> float:
        return round(sign_test(self.wins, self.n), 4)

    @property
    def beats(self) -> bool:
        return bool(self.per_seed) and self.mean_delta > 0

    @property
    def inside_noise(self) -> bool:
        """Is the difference smaller than its own seed-to-seed spread?

        One seed has no spread to be outside of, so a single-seed run is always inside it. That
        is the honest reading of one run, not a technicality.
        """
        if self.n < 2:
            return True
        return abs(self.mean_delta) <= self.sd_delta

    @property
    def verdict(self) -> str:
        if not self.per_seed:
            return f"{self.challenger} vs {self.incumbent}: not run"
        direction = "beats" if self.beats else "does not beat"
        seeds = "1 seed" if self.n == 1 else f"{self.n} seeds"
        spread = f" ± {self.sd_delta:.4f}" if self.n > 1 else ""
        line = (
            f"{self.challenger} {direction} {self.incumbent} on the {self.column} column: "
            f"{self.metric} {self.mean_delta:+.4f}{spread} over {seeds}, "
            f"{self.wins}/{self.n} in that direction, sign-test p = {self.p_value:.3f}"
        )
        if self.n < 2:
            line += " — one seed has no spread to be outside of, so nothing follows from the sign"
        elif self.inside_noise:
            line += " — inside the run-to-run spread, so it is not a result either way"
        if self.outcome != loao.MEASURED:
            line += f". The column itself is {self.outcome}: {self.reason}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenger": self.challenger,
            "incumbent": self.incumbent,
            "column": self.column,
            "metric": self.metric,
            "outcome": self.outcome,
            "reason": self.reason,
            "per_seed": self.per_seed,
            "n_seeds": self.n,
            "mean_delta": self.mean_delta,
            "sd_delta": self.sd_delta,
            "wins": self.wins,
            "p_value": self.p_value,
            "beats": self.beats,
            "inside_noise": self.inside_noise,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Comparison:
        """The per-seed deltas and the labels; every summary is recomputed from them.

        Same rule as `Spread.from_dict`: `mean_delta`, `p_value` and `verdict` are written into
        the artefact for a reader, never read back out of it, so a committed comparison cannot
        disagree with the seeds it is made of.
        """
        return cls(
            challenger=raw["challenger"],
            incumbent=raw["incumbent"],
            column=raw["column"],
            metric=raw["metric"],
            per_seed=list(raw.get("per_seed", [])),
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
        )


def compare(
    report: ThreeSystemReport,
    challenger: str = "C_adaptive",
    incumbent: str = "B_smote",
    column: str = UNSEEN,
    metric: str = HEADLINE_METRIC,
) -> Comparison | None:
    """The paired difference the claim stands or falls on. Paired by seed, never pooled.

    Pairing matters: both systems see the same fold, the same generated pool and the same fitted
    procedure on a given seed, so the difference between them is the only thing the seed does not
    also move. Comparing two pooled means would put the seed's whole variance into the gap.
    """
    per_seed: list[dict[str, Any]] = []
    outcomes, reasons = [], []
    for run in report.runs:
        a, b = run.system(incumbent), run.system(challenger)
        if a is None or b is None:
            continue
        ma, mb = a.column(column).any_metrics, b.column(column).any_metrics
        if ma is None or mb is None:
            continue
        va, vb = float(getattr(ma, metric)), float(getattr(mb, metric))
        per_seed.append(
            {
                "seed": run.seed,
                "incumbent": round(va, 6),
                "challenger": round(vb, 6),
                "delta": round(vb - va, 6),
            }
        )
        for cell in (a.column(column), b.column(column)):
            outcomes.append(cell.outcome)
            reasons.append(cell.reason)

    if not per_seed:
        return None
    if all(o == loao.MEASURED for o in outcomes):
        outcome, reason = loao.MEASURED, ""
    else:
        outcome = loao.WITHHELD
        reason = next(
            (r for o, r in zip(outcomes, reasons, strict=False) if r and o != loao.MEASURED), ""
        )
    return Comparison(
        challenger=challenger,
        incumbent=incumbent,
        column=column,
        metric=metric,
        per_seed=per_seed,
        outcome=outcome,
        reason=reason,
    )


def comparisons(report: ThreeSystemReport) -> list[Comparison]:
    """Every comparison the table is obliged to report, in both columns and both metrics."""
    out = []
    for column in COLUMNS:
        for metric in ("pr_auc", HEADLINE_METRIC):
            for challenger, incumbent in (
                ("C_adaptive", "B_smote"),
                ("C_adaptive", "A_baseline"),
                ("B_smote", "A_baseline"),
            ):
                got = compare(report, challenger, incumbent, column, metric)
                if got is not None:
                    out.append(got)
    return out


def diagnose(report: ThreeSystemReport) -> list[dict[str, Any]]:
    """Why the table landed where it did — read off the run, never typed in.

    Ticket 16 asks for the likely reason to be stated when the adaptive loop does not beat SMOTE.
    A sentence somebody writes after seeing the number is worth less than one the run can derive
    from its own logs, so each finding below carries the evidence that produced it. They are
    candidate explanations in the order a reader should rule them out, not a ranked cause.
    """
    findings: list[dict[str, Any]] = []
    head = compare(report)
    if head is None:
        return findings

    # 1. Can the column carry a claim at all? Everything else is downstream of this.
    unseen_cells = [c for s in report.systems for c in report.cells(s, UNSEEN)]
    not_measured = [c for c in unseen_cells if c.outcome != loao.MEASURED]
    if not_measured:
        findings.append(
            {
                "finding": "the held-out column cannot carry a claim on this anchor, so neither "
                "system's number on it is quotable and the difference between them is not "
                f"either — {not_measured[0].reason}",
                "evidence": {
                    "outcome": not_measured[0].outcome,
                    "cells_affected": f"{len(not_measured)}/{len(unseen_cells)}",
                },
            }
        )

    # 2. Did the control do anything? If B == A, C is being compared with A wearing B's name.
    b_vs_a = compare(report, "B_smote", "A_baseline", UNSEEN, HEADLINE_METRIC)
    if b_vs_a is not None and max((abs(d) for d in b_vs_a.deltas), default=0.0) < 1e-9:
        findings.append(
            {
                "finding": "SMOTE reproduced the baseline exactly on this column — it can only "
                "interpolate between fraud rows the baseline already had, so System B is System "
                "A with more rows. The control is behaving as designed; there is no augmentation "
                "effect for the loop to beat, and C vs B is C vs A under another name",
                "evidence": {"max_abs_delta_B_minus_A": 0.0, "seeds": b_vs_a.n},
            }
        )

    # 3. Did anything clear the floor? A table under the amount floor is not a detection result.
    floor = floor_spread(report, UNSEEN)
    if floor.values:
        best = max(
            (spread_of(report, s, UNSEEN).mean for s in report.systems),
            default=float("nan"),
        )
        if not math.isnan(best) and best <= floor.mean:
            findings.append(
                {
                    "finding": "no system clears the amount floor on the held-out column — "
                    "sorting the window by amount alone ranks the family at least as well as "
                    "every trained model here, so the column is not measuring detection",
                    "evidence": {"best_system": round(best, 6), "amount_floor": floor.mean},
                }
            )

    # 4. Did the loop have any leverage? Rows it could not add cannot move a fit.
    for row in report.rows_of("C_adaptive"):
        if row.n_train_fraud and row.n_generated / row.n_train_fraud < 0.05:
            findings.append(
                {
                    "finding": "the loop's output is a rounding error in the training set — the "
                    "rows it added are a few percent of the fraud already there, so System C is "
                    "System A plus noise however good the attacks are",
                    "evidence": {
                        "generated_rows": row.n_generated,
                        "train_fraud": row.n_train_fraud,
                        "share": round(row.n_generated / row.n_train_fraud, 4),
                    },
                }
            )
            break

    # 5. Did the attacker still find anything by the end? A collapsed evasion rate means the
    #    rows the loop kept are rows the detector already catches.
    trajectories = [row.loop for row in report.rows_of("C_adaptive") if row.loop]
    if trajectories:
        finals = [float(t[-1].get("evasion_rate", 0.0)) for t in trajectories]
        firsts = [float(t[0].get("evasion_rate", 0.0)) for t in trajectories]
        if max(finals) < 0.05:
            findings.append(
                {
                    "finding": "the attacker's evasion rate collapsed inside the loop, so the "
                    "rows System C trained on are rows the detector had already learned to "
                    "catch. A converged loop teaches the detector nothing new by definition",
                    "evidence": {
                        "evasion_first_round": round(float(np.mean(firsts)), 4),
                        "evasion_final_round": round(float(np.mean(finals)), 4),
                    },
                }
            )
        rejected = sum(row.rejected_rounds for row in report.rows_of("C_adaptive"))
        if rejected:
            findings.append(
                {
                    "finding": "the commensurability audit rejected candidate batches inside the "
                    "loop, so part of the search budget bought nothing. The gate is working as "
                    "intended — those rows would have won by provenance — but the loop paid for "
                    "them",
                    "evidence": {"rejected_rounds": rejected, "seeds": len(trajectories)},
                }
            )

    # 6. Is System C's held-out number reproduced by a model that only knows provenance? This
    #    is the one check the fold-level probe cannot make: it learns "injected" from the
    #    holdout's own positives, where System C learns it from every row the loop generated.
    for row in report.rows_of("C_adaptive"):
        probe = row.unseen.provenance or {}
        measured = row.unseen.any_metrics
        if not probe.get("checked") or measured is None:
            continue
        if (
            str(probe.get("trained_on", "")).startswith("the loop")
            and float(probe["pr_auc"]) >= float(measured.pr_auc) - PROVENANCE_MARGIN
        ):
            findings.append(
                {
                    "finding": "a model that knows nothing except which rows the generator wrote "
                    "reaches System C's held-out score on the same rows — so C's number on that "
                    "column is the generator's fingerprint transferring between families, not a "
                    "detector generalising to an unseen attack. Both models trained on the same "
                    "injected rows; only one of them was told what fraud is",
                    "evidence": {
                        "system_c_pr_auc": round(float(measured.pr_auc), 6),
                        "provenance_only_pr_auc": round(float(probe["pr_auc"]), 6),
                        "trained_on": probe.get("trained_on"),
                    },
                }
            )
            break

    # 7. Is the control column at the ceiling? Then "no regression on known attacks" is cheap.
    known_means = [spread_of(report, s, KNOWN).mean for s in report.systems]
    known_means = [m for m in known_means if not math.isnan(m)]
    if known_means and min(known_means) >= 0.99:
        findings.append(
            {
                "finding": "every system is at the ceiling on the known column — this anchor's "
                "own fraud is separable by almost anything, so holding that column steady is "
                "not evidence of much, and neither is losing a little of it",
                "evidence": {
                    "known_column_range": [round(min(known_means), 4), round(max(known_means), 4)]
                },
            }
        )

    # 8. Did C trade the known column for the unseen one?
    known = compare(report, "C_adaptive", "A_baseline", KNOWN, HEADLINE_METRIC)
    if known is not None and known.mean_delta < -1e-6:
        findings.append(
            {
                "finding": "System C lost ground on the attacks it already knew about — the "
                "generated rows moved the fit away from the families the anchor actually "
                "carries. Whatever the held-out column says, this is the cost side of it",
                "evidence": {
                    "known_delta_C_minus_A": known.mean_delta,
                    "sd": known.sd_delta,
                    "seeds": known.n,
                },
            }
        )

    # 9. Is the gap smaller than the noise around it?
    if head.inside_noise and head.n > 1:
        findings.append(
            {
                "finding": "the gap between C and B is smaller than the seed-to-seed spread, so "
                "the ordering of the two rows is not stable across seeds and no direction should "
                "be read from it",
                "evidence": {
                    "mean_delta": head.mean_delta,
                    "sd_delta": head.sd_delta,
                    "seeds": head.n,
                },
            }
        )
    return findings


def table_markdown(report: ThreeSystemReport, dp: int = 3) -> str:
    """The hero table: both columns, mean ± sd across seeds, brackets where withheld.

    Generated from the artefact and nowhere else, so the deck, the README and the document
    cannot disagree with the run. A bracketed number is one the harness refuses to stand behind;
    it is printed rather than hidden, because hiding evidence is its own kind of dishonesty.
    """
    fpr = report.operating_point.get("fixed_fpr", protocol.DEFAULT_FPR)
    k = report.operating_point.get("k", protocol.DEFAULT_K)
    header = [
        "system",
        "known PR-AUC",
        f"known recall@{float(fpr):.0%}FPR",
        f"unseen PR-AUC ({report.held_out_vector})",
        f"unseen recall@{float(fpr):.0%}FPR",
        f"unseen P@{int(k)}",
        "train rows",
        "train fraud",
        "generated",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]

    for system in report.systems:
        rows = report.rows_of(system)
        cells = [
            system,
            spread_of(report, system, KNOWN, "pr_auc").text(dp),
            spread_of(report, system, KNOWN, HEADLINE_METRIC).text(dp),
            spread_of(report, system, UNSEEN, "pr_auc").text(dp),
            spread_of(report, system, UNSEEN, HEADLINE_METRIC).text(dp),
            spread_of(report, system, UNSEEN, "precision_at_k").text(2),
            f"{int(np.mean([r.n_train for r in rows])):,}" if rows else "—",
            f"{int(np.mean([r.n_train_fraud for r in rows])):,}" if rows else "—",
            f"{int(np.mean([r.n_generated for r in rows])):,}" if rows else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    floor_known = floor_spread(report, KNOWN)
    floor_unseen = floor_spread(report, UNSEEN)
    lines.append(
        "| "
        + " | ".join(
            [
                "amount floor",
                floor_spread(report, KNOWN, "pr_auc").text(dp),
                floor_known.text(dp),
                floor_spread(report, UNSEEN, "pr_auc").text(dp),
                floor_unseen.text(dp),
                floor_spread(report, UNSEEN, "precision_at_k").text(2),
                "—",
                "—",
                "—",
            ]
        )
        + " |"
    )
    lines.append("")
    lines.append(
        f"_mean ± sd over {len(report.runs)} seed(s) {report.seeds}; "
        f"**[brackets]** mark numbers the harness withholds; "
        f"detector backend: {', '.join(report.backends) or 'unknown'}_"
    )
    return "\n".join(lines)
