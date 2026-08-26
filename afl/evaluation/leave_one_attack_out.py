"""Leave-one-attack-out: the only evaluation that answers the question we actually asked.

Reporting recall on a family the model trained on measures memorisation. The claim here is
*generalisation to an unseen attack*, so one whole vector is pulled out of training and the
headline number is measured on it alone.

Two guards, both easy to lose and expensive to lose quietly:
  1. the held-out family never appears in training — not one row, not in the replay buffer;
  2. the split is still out-of-time, so "unseen family" does not smuggle in "seen future".

Both are `assert_*` functions that raise, not comments, and both are exercised by tests that
deliberately try to leak a row past them. A third keeps the holdout's legit rows: without a
haystack, FPR and precision@k are arithmetic over an empty denominator.

**A fold that runs is not a fold that means something.** Ticket 07 and ticket 10 both landed on
the same finding from different directions — on a real anchor, every positive in this fold is an
injected synthetic row and every negative is a real one, so a fold can report a confident number
while measuring which generator wrote the row. So the outcome of a fold is one of three things,
and only the first carries a claim:

  * `measured`  — the numbers stand.
  * `withheld`  — the fold ran and the numbers exist, but nothing may be concluded from them:
                  too few positives; the family is separable from the anchor by one contract
                  field; a classifier can sort the injected rows from the anchor's own, so the
                  fold is measuring provenance; or the vector is a template whose defining tell
                  is not modelled yet. They live under `withheld_metrics`, never under
                  `metrics`, so a consumer that reads the obvious field gets `None` rather than
                  a number it should not quote.
  * `skipped`   — the fold never ran, and says why.

A low score and a meaningless one are different claims. Nothing here is allowed to report the
second as the first.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from afl.contract.metrics import MetricResult
from afl.contract.schema import Transaction
from afl.data.splits import CommittedSplit, assert_no_leakage, out_of_time_split
from afl.evaluation import protocol

log = logging.getLogger(__name__)

#: First-party / friendly fraud. Chosen as the holdout because `user == fraudster` breaks the
#: legit-vs-attacker assumption every supervised feature quietly rests on: no compromised device,
#: no new operator, no ring. A family that is merely *unseen* still shares that assumption; this
#: one does not, which is what makes it a real generalisation test rather than a harder fold.
#: It is also why the anomaly layer exists at all — see afl/defend/models/anomaly.py.
DEFAULT_HOLDOUT = "M3"

#: Below this many positives a fold is reported as too thin to carry a claim rather than as a
#: low score. Recall on 20 rows moves five points per row, so the difference between 0.30 and
#: 0.45 there is which episodes the out-of-time cut happened to land after. Same order as
#: `afl.defend.calibration.MIN_POSITIVES`, for the same reason.
MIN_MEANINGFUL_POSITIVES = 30

#: A fold whose positives a classifier can pick out of the anchor's own traffic *by provenance*
#: is a fold whose recall is a statement about the generator. Ticket 07 measured exactly this at
#: AUC 1.00 on PaySim and it lived in a carry-out rather than in the harness, which is how a
#: number like that gets quoted around. Above this PR-AUC the fold's numbers are withheld.
#:
#: The bar has the same shape as `envelope.audit`'s: a floor, or a multiple of the base rate,
#: whichever is higher — so a fold with three positives cannot pass by having a base rate small
#: enough to make any score look impressive. The floor is half of a perfect probe: below it the
#: probe is finding some of the tell, above it the fold cannot tell "caught the fraud" apart from
#: "spotted the synthetic row", because in this fold those are the same label.
#:
#: **Read the verdict asymmetrically.** The probe learns "injected" from the fold's own
#: positives, so it is underpowered exactly where the fold is thin. A score over the bar is
#: strong evidence the fold is measuring provenance; a score under it, on a fold with few
#: positives, is weak evidence of anything. `MIN_MEANINGFUL_POSITIVES` is applied first so the
#: weakest probes belong to folds that were already withheld.
PROVENANCE_FLOOR = 0.5
PROVENANCE_LIFT = 5.0

#: What a fold's numbers may be used for. See the module docstring.
MEASURED = "measured"
WITHHELD = "withheld"
SKIPPED = "skipped"
OUTCOMES = (MEASURED, WITHHELD, SKIPPED)

_MISSING = object()


class GuardFailed(AssertionError):
    """A leave-one-attack-out guard could not be satisfied — or could not be checked.

    An `AssertionError` so it reads like the guard it is, and a named type so a caller can tell
    "the fold is unsound" apart from every other assertion in a run.
    """


def is_provenance_bound(pr_auc: float, base_rate: float) -> bool:
    """Can provenance alone explain the fold?

    The rule, not the measurement: the probe is built where the feature space lives (see
    `scripts/build_loao.py`), and the verdict is applied here so that every caller applies the
    same one.
    """
    return pr_auc > max(PROVENANCE_FLOOR, PROVENANCE_LIFT * base_rate)


# ── the guards ──────────────────────────────────────────────────────────────────
def training_rows(detector) -> list[Transaction]:
    """Every row a detector has fitted on, replay buffer included.

    A detector that cannot say what it trained on is a detector the carve-out cannot be checked
    on, and that is a failure rather than a pass: the whole point of these guards is that the
    expensive way to lose a fold is quietly. Implemented as a `training_rows` property on
    `LGBMDetector`, `AnomalyDetector` and `EnsembleDetector`.
    """
    rows = getattr(detector, "training_rows", _MISSING)
    if rows is _MISSING:
        raise GuardFailed(
            f"{type(detector).__name__} has no `training_rows`, so there is no way to check that "
            "the held-out family stayed out of its training set — including its replay buffer. "
            "Add the property, or do not claim a leave-one-attack-out number from it"
        )
    return list(rows)


def assert_family_held_out(
    train: list[Transaction], held_out_vector: str, detector=None
) -> dict[str, Any]:
    """Not one row of the family reaches training. With a detector, its whole memory too.

    The list handed to `fit` is the easy half. The half that goes wrong is the replay buffer:
    it accumulates evasions across rounds, so a family carved out of the split in round one can
    walk back into training in round four without anything in the split changing.
    """
    leaked = {t.txn_id for t in train if t.vector_id == held_out_vector}
    checked, audited = len(train), "the training rows"
    if detector is not None:
        seen = training_rows(detector)
        leaked |= {t.txn_id for t in seen if t.vector_id == held_out_vector}
        checked += len(seen)
        audited = "the training rows and every row the detector has fitted on, replay included"
    if leaked:
        raise GuardFailed(
            f"{len(leaked)} {held_out_vector} row(s) reached training, e.g. "
            f"{sorted(leaked)[:3]} — the holdout is not held out, so its recall measures "
            "memorisation rather than generalisation"
        )
    return {
        "held_out_vector": held_out_vector,
        "rows_checked": checked,
        "leaked_rows": 0,
        "audited": audited,
        "detector": type(detector).__name__ if detector is not None else None,
    }


def assert_embargo_intact(
    train: list[Transaction], holdout: list[Transaction], embargo: timedelta
) -> dict[str, Any]:
    """The carve-out must not have closed the out-of-time gap it was applied on top of.

    Removing a family shortens both sides, and the arithmetic says the gap can only widen — but
    that is an argument, and an argument is what this replaces. "Unseen family" must not be able
    to smuggle in "seen future" through a filter somebody edits next year.
    """
    if not train or not holdout:
        return {"checked": False, "reason": "one side of the fold is empty"}
    latest_train = max(t.ts for t in train)
    earliest_holdout = min(t.ts for t in holdout)
    gap = earliest_holdout - latest_train
    if gap < embargo:
        raise GuardFailed(
            f"the embargo did not survive the carve-out: {gap} between the last training row "
            f"({latest_train}) and the first holdout row ({earliest_holdout}), against the "
            f"{embargo} the split committed to"
        )
    assert_no_leakage(train, holdout)
    return {
        "checked": True,
        "embargo_seconds": int(embargo.total_seconds()),
        "gap_seconds": int(gap.total_seconds()),
        "last_train_row": latest_train.isoformat(),
        "first_holdout_row": earliest_holdout.isoformat(),
    }


def assert_haystack_intact(
    test_side: list[Transaction], holdout: list[Transaction]
) -> dict[str, Any]:
    """Every legit row of the test window is still in the holdout.

    The carve-out drops the *other* families' fraud, and it would be one keystroke to drop the
    legit rows with them. An FPR measured without negatives is not an FPR, and precision@100 on
    a holdout of 100 positives is 1.0 by construction.
    """
    legit = {t.txn_id for t in test_side if not t.is_fraud}
    kept = {t.txn_id for t in holdout if not t.is_fraud}
    dropped = legit - kept
    if dropped:
        raise GuardFailed(
            f"{len(dropped)} legit row(s) were dropped from the holdout, e.g. "
            f"{sorted(dropped)[:3]} — a fixed-FPR number needs the whole haystack, not the "
            "share of it that survived a filter"
        )
    return {"legit_rows_in_window": len(legit), "legit_rows_kept": len(kept), "dropped": 0}


# ── the carve-out ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Fold:
    """One carve-out: a family pulled out of training, and the window it is measured on.

    Built by `Fold.carve`, which runs all three guards before it hands anything back and keeps
    what each one checked in `guards`. That record travels into the artefact, so a committed
    number says the fold was guarded rather than leaving a reader to assume it.
    """

    held_out_vector: str
    train: list[Transaction]
    holdout: list[Transaction]
    embargo: timedelta = timedelta(0)
    guards: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def carve(
        cls,
        pool: list[Transaction],
        held_out_vector: str,
        split: CommittedSplit | None = None,
        train_frac: float = 0.7,
        embargo_days: float = 1.0,
    ) -> Fold:
        """Out-of-time first, then the family carve-out, then the guards.

        `split` is the committed boundary for a real anchor. Prefer it: a fraction re-derives
        the cut from whatever rows it was handed, so the partition moves whenever the pool
        composition does, and two runs stop being comparable without anything visibly changing.
        """
        if split is not None:
            train_side, test_side = split.apply(pool)
            embargo = split.embargo
        else:
            train_side, test_side = out_of_time_split(
                pool, train_frac=train_frac, embargo_days=embargo_days
            )
            embargo = timedelta(days=embargo_days)

        train = [t for t in train_side if t.vector_id != held_out_vector]
        holdout = [t for t in test_side if t.vector_id == held_out_vector or not t.is_fraud]
        guards = {
            "family": assert_family_held_out(train, held_out_vector),
            "embargo": assert_embargo_intact(train, holdout, embargo),
            "haystack": assert_haystack_intact(test_side, holdout),
        }
        return cls(
            held_out_vector=held_out_vector,
            train=train,
            holdout=holdout,
            embargo=embargo,
            guards=guards,
        )

    # ── what the fold contains ──────────────────────────────────────────────────
    @property
    def n_positives(self) -> int:
        """Held-out-family rows in the holdout. The number every metric below rests on."""
        return sum(1 for t in self.holdout if t.is_fraud)

    @property
    def n_train_fraud(self) -> int:
        return sum(1 for t in self.train if t.is_fraud)

    def counts(self) -> dict[str, Any]:
        legit = len(self.holdout) - self.n_positives
        return {
            "train_rows": len(self.train),
            "train_fraud": self.n_train_fraud,
            "holdout_rows": len(self.holdout),
            "holdout_positives": self.n_positives,
            "holdout_legit": legit,
            "holdout_base_rate": round(self.n_positives / len(self.holdout), 8)
            if self.holdout
            else 0.0,
            "embargo_seconds": int(self.embargo.total_seconds()),
        }


def make_splits(
    txns: list[Transaction],
    held_out_vector: str,
    train_frac: float = 0.7,
    embargo_days: float = 1.0,
    split: CommittedSplit | None = None,
) -> tuple[list[Transaction], list[Transaction]]:
    """(train, holdout) — the guarded carve-out, for callers that only want the two lists."""
    fold = Fold.carve(txns, held_out_vector, split, train_frac, embargo_days)
    return fold.train, fold.holdout


# ── what a fold is allowed to claim ─────────────────────────────────────────────
@dataclass(frozen=True)
class FoldResult:
    """One row of the matrix: an outcome, a reason, and numbers only when they may be quoted.

    `metrics` is `None` unless `outcome == "measured"`. Numbers from a fold that cannot carry a
    claim still exist — they are the evidence for the verdict — but they live under
    `withheld_metrics`, so quoting one is a decision somebody had to make rather than the
    default a naive reader falls into.
    """

    held_out_vector: str
    outcome: str
    reason: str = ""
    metrics: MetricResult | None = None
    withheld_metrics: MetricResult | None = None
    operational: dict[str, float] = field(default_factory=dict)
    floor: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    guards: dict[str, Any] = field(default_factory=dict)
    separability: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown fold outcome {self.outcome!r}; expected one of {OUTCOMES}")
        if self.outcome != MEASURED and self.metrics is not None:
            raise ValueError(
                f"{self.held_out_vector}: a {self.outcome} fold carries numbers under "
                "`withheld_metrics`, never under `metrics` — that field is what a reader quotes"
            )
        if self.outcome != MEASURED and not self.reason:
            raise ValueError(
                f"{self.held_out_vector}: a fold that is not measured has to say why — a "
                f"{self.outcome} fold with no reason is indistinguishable from a forgotten one"
            )

    @classmethod
    def skipped(cls, held_out_vector: str, reason: str, **kwargs) -> FoldResult:
        """A fold that never ran. There are no numbers at all, and the reason is the result."""
        return cls(held_out_vector=held_out_vector, outcome=SKIPPED, reason=reason, **kwargs)

    @property
    def reported(self) -> bool:
        return self.outcome == MEASURED

    @property
    def any_metrics(self) -> MetricResult | None:
        """Whichever block exists — for a doc that prints withheld numbers under their warning."""
        return self.metrics or self.withheld_metrics

    def summary(self) -> str:
        """One line, in the form the run prints it."""
        m = self.metrics
        if m is None:
            return f"{self.held_out_vector}: {self.outcome} — {self.reason}"
        return (
            f"{self.held_out_vector}: PR-AUC {m.pr_auc:.3f}  "
            f"recall@{m.fixed_fpr:.0%}FPR {m.recall_at_fixed_fpr:.3f}  "
            f"precision@{m.k} {m.precision_at_k:.2f}  ({m.n_positives:,} positives)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_vector": self.held_out_vector,
            "outcome": self.outcome,
            "reason": self.reason,
            "metrics": self.metrics.model_dump() if self.metrics else None,
            "withheld_metrics": self.withheld_metrics.model_dump()
            if self.withheld_metrics
            else None,
            "operational": self.operational,
            "floor": self.floor,
            "counts": self.counts,
            "guards": self.guards,
            "separability": self.separability,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FoldResult:
        return cls(
            held_out_vector=raw["held_out_vector"],
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
            metrics=MetricResult(**raw["metrics"]) if raw.get("metrics") else None,
            withheld_metrics=MetricResult(**raw["withheld_metrics"])
            if raw.get("withheld_metrics")
            else None,
            operational=raw.get("operational", {}),
            floor=raw.get("floor", {}),
            counts=raw.get("counts", {}),
            guards=raw.get("guards", {}),
            separability=raw.get("separability"),
            provenance=raw.get("provenance"),
        )


def amount_floor(
    train: list[Transaction],
    holdout: list[Transaction],
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
) -> dict[str, Any]:
    """Rank by amount alone — no model, no features, no training. The floor under every fold.

    Two results in this repo were walked back for want of this column: the transfer test and the
    BankSim spike both produced a detector that lost to sorting by amount. A held-out-family
    recall that does not clear it has not detected anything.

    The direction is chosen on the **training** window and applied to the holdout, because
    picking it on the window it is reported from is the tuning-on-test the whole harness exists
    to prevent.
    """
    if not train or not holdout:
        return {}
    y_tr = np.array([int(t.is_fraud) for t in train], dtype=int)
    a_tr = np.array([t.amount for t in train], dtype=float)
    high_first = protocol.pr_auc(y_tr, a_tr) >= protocol.pr_auc(y_tr, -a_tr)

    y_te = np.array([int(t.is_fraud) for t in holdout], dtype=int)
    a_te = np.array([t.amount for t in holdout], dtype=float)
    result = protocol.evaluate(y_te, a_te if high_first else -a_te, fixed_fpr=fixed_fpr, k=k)
    return {
        **result.model_dump(exclude={"held_out_vector"}),
        "direction": "largest amount first" if high_first else "smallest amount first",
        "direction_chosen_on": "train",
    }


def run_fold(
    fold: Fold,
    detector,
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
    min_positives: int = MIN_MEANINGFUL_POSITIVES,
    separability: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    not_reportable: str = "",
) -> FoldResult:
    """Score a fitted detector on one carve-out and decide what its numbers may be used for.

    The family guard runs again here, against the *detector* rather than the split, because
    fitting is where the replay buffer gets a say. `separability` is the red side's audit of
    this family against the anchor (`afl.attack.envelope.audit`) — passed in rather than
    computed, so the blue side's harness stays out of the generator's business.

    `provenance` is the blue side's version of the same question, asked in the feature space
    the detector actually sees: can a classifier sort the injected rows from the anchor's own?
    In this fold those two labels are the same one — the carve-out drops the anchor's own fraud
    from the holdout, so every positive is injected and every negative is real — which is why a
    fold can post a confident recall while measuring which generator wrote the row.

    `not_reportable` carries a reason the family itself cannot be claimed for — a template
    vector whose defining tell is not modelled yet. The fold still runs: the number is evidence
    about the pipeline even when it is not evidence about the family.
    """
    family = assert_family_held_out(fold.train, fold.held_out_vector, detector)
    guards = {**fold.guards, "family": family}
    counts = fold.counts()
    common = {
        "counts": counts,
        "guards": guards,
        "separability": separability,
        "provenance": provenance,
    }

    if not fold.n_positives:
        # the out-of-time cut can land after every episode of the held-out family. Every metric
        # then reads 0.0, which looks like failure but is the absence of a measurement.
        return FoldResult.skipped(
            fold.held_out_vector,
            f"the holdout carries no {fold.held_out_vector} rows — every metric would read 0.0 "
            "without having measured anything. Widen the window or raise eval.holdout_episodes",
            **common,
        )

    # one scoring pass, both halves of the story: the ranking metrics and what the policy did.
    # Scoring twice would rebuild the design matrix and re-run SHAP over the flagged rows, which
    # on a 600k-row holdout is the expensive half of the fold.
    scores = protocol.score_transactions(detector, fold.holdout, run_id="loao")
    y, s = protocol.align(fold.holdout, scores)
    result = protocol.evaluate(y, s, fixed_fpr, k, held_out_vector=fold.held_out_vector)
    measured = {
        **common,
        "operational": {
            key: round(v, 6) for key, v in protocol.operational_rates(fold.holdout, scores).items()
        },
        "floor": amount_floor(fold.train, fold.holdout, fixed_fpr, k),
    }

    def withheld(reason: str) -> FoldResult:
        """The numbers exist; they move to `withheld_metrics` and the reason takes their place."""
        return FoldResult(
            held_out_vector=fold.held_out_vector,
            outcome=WITHHELD,
            reason=reason,
            withheld_metrics=result,
            **measured,
        )

    # Ordered cheapest-and-most-damning first, so a fold that fails on several grounds is
    # reported on the one a reader can check without re-running anything.
    if not_reportable:
        return withheld(not_reportable)
    if fold.n_positives < min_positives:
        return withheld(
            f"{fold.n_positives} positives against a floor of {min_positives} — recall moves "
            f"{1 / fold.n_positives:.1%} per row here, so this is reported as missing rather "
            "than as a low score"
        )
    if separability and separability.get("trivially_separable"):
        return withheld(
            f"`{separability.get('worst')}` alone separates the injected "
            f"{fold.held_out_vector} rows from the anchor at PR-AUC "
            f"{float(separability.get('score', 0.0)):.3f} — this fold measures provenance, "
            "not detection, and every number in it inherits that"
        )
    if provenance and provenance.get("separable"):
        return withheld(
            f"a classifier sorts the injected {fold.held_out_vector} rows from this anchor's "
            f"own traffic at PR-AUC {float(provenance.get('pr_auc', 0.0)):.3f}, against "
            f"{result.pr_auc:.3f} for the detector on the same rows — every positive here is "
            "injected and every negative is real, so the fold cannot tell detection apart "
            "from provenance"
        )
    return FoldResult(
        held_out_vector=fold.held_out_vector, outcome=MEASURED, metrics=result, **measured
    )


@dataclass
class LeaveOneAttackOut:
    """■ B's half of the seam's return leg. Scores a detector; never trains one."""

    holdout: list[Transaction]
    held_out_vector: str = DEFAULT_HOLDOUT
    fixed_fpr: float = protocol.DEFAULT_FPR
    k: int = protocol.DEFAULT_K
    history: list[MetricResult] = field(default_factory=list)

    @classmethod
    def from_pool(
        cls,
        txns: list[Transaction],
        held_out_vector: str = DEFAULT_HOLDOUT,
        train_frac: float = 0.7,
        embargo_days: float = 1.0,
        split: CommittedSplit | None = None,
        **kwargs,
    ) -> tuple[LeaveOneAttackOut, list[Transaction]]:
        """Returns (evaluator, train_rows) so a caller cannot accidentally train on the holdout."""
        fold = Fold.carve(txns, held_out_vector, split, train_frac, embargo_days)
        if not fold.n_positives:
            log.warning(
                "holdout for %r contains no fraud rows — widen the window or add episodes; "
                "the numbers from this split measure nothing",
                held_out_vector,
            )
        return cls(holdout=fold.holdout, held_out_vector=held_out_vector, **kwargs), fold.train

    def leave_one_attack_out(self, detector) -> MetricResult:
        result = protocol.evaluate_detector(
            detector,
            self.holdout,
            fixed_fpr=self.fixed_fpr,
            k=self.k,
            held_out_vector=self.held_out_vector,
        )
        if not result.n_positives:
            log.warning(
                "scored a holdout with no positives — treat this row as missing, not as 0.0"
            )
        self.history.append(result)
        return result

    def operational(self, detector) -> dict[str, float]:
        scores = protocol.score_transactions(detector, self.holdout, run_id="loao")
        return protocol.operational_rates(self.holdout, scores)


def sweep(
    txns: list[Transaction],
    detector_factory,
    vectors: list[str] | None = None,
    train_frac: float = 0.7,
    embargo_days: float = 1.0,
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
    split: CommittedSplit | None = None,
    min_positives: int = MIN_MEANINGFUL_POSITIVES,
    fit=None,
    separability: dict[str, dict[str, Any]] | None = None,
    provenance: dict[str, dict[str, Any]] | None = None,
    not_reportable: dict[str, str] | None = None,
) -> list[FoldResult]:
    """The full matrix: hold out each family in turn, refit from scratch, score it.

    One row per vector is the honest version of a single headline number — a system that only
    generalises to one family will show it here. Every requested vector gets a row, including
    the ones that could not be run: a fold that vanishes from the output is a fold that reads
    as "not applicable" when it means "we did not look".

    `fit(detector, rows)` is how a caller trains — calibration on a validation tail of the
    training window, typically. The default fits on everything and calibrates nothing.
    """
    vectors = vectors or sorted({t.vector_id for t in txns if t.vector_id})
    fit = fit or (lambda detector, rows: detector.fit(rows))
    separability = separability or {}
    provenance = provenance or {}
    not_reportable = not_reportable or {}

    out: list[FoldResult] = []
    for vid in vectors:
        if not any(t.vector_id == vid for t in txns):
            out.append(
                FoldResult.skipped(
                    vid, "no rows of this family are in the pool — nothing to hold out"
                )
            )
            continue
        fold = Fold.carve(txns, vid, split, train_frac, embargo_days)
        if not fold.n_train_fraud:
            out.append(
                FoldResult.skipped(
                    vid,
                    "the carve-out left no fraud in the training window — a single-class fit is "
                    "not a detector, so there is nothing to measure this family against",
                    counts=fold.counts(),
                    guards=fold.guards,
                )
            )
            continue
        detector = detector_factory()
        fit(detector, fold.train)
        out.append(
            run_fold(
                fold,
                detector,
                fixed_fpr=fixed_fpr,
                k=k,
                min_positives=min_positives,
                separability=separability.get(vid),
                provenance=provenance.get(vid),
                not_reportable=not_reportable.get(vid, ""),
            )
        )
    return out


# ── the artefact ────────────────────────────────────────────────────────────────
#: Bump when the fields change shape, so an old file fails loudly instead of being read with the
#: wrong meaning.
LOAO_ARTEFACT_VERSION = 1

DEFAULT_LOAO_DIR = Path(os.getenv("AFL_LOAO_DIR", "artifacts/loao"))


@dataclass(frozen=True)
class LeaveOneAttackOutReport:
    """One anchor's whole matrix, with the config and the seed that produced it.

    Same discipline as `artifacts/splits/` and `artifacts/detector/` — the config holds the
    inputs, the artefact holds the decision. A headline number quoted from a slide is a number
    nobody can re-derive; this file carries the operating point, the committed boundary, the
    eval config as it was read, and the seed, so the row and the run that made it stay attached.
    """

    dataset: str
    seed: int
    config: dict[str, Any]
    operating_point: dict[str, Any]
    folds: list[FoldResult]
    split: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    model_card: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    version: int = LOAO_ARTEFACT_VERSION

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError(
                f"{self.dataset}: a leave-one-attack-out report with no folds is not a result — "
                "even a run where every fold was skipped has to name them and say why"
            )
        if "seed" not in self.config:
            object.__setattr__(self, "config", {**self.config, "seed": self.seed})

    @property
    def headline(self) -> FoldResult | None:
        """The fold `config.held_out_vector` names — the one the rest of the project quotes."""
        wanted = str(self.config.get("held_out_vector", DEFAULT_HOLDOUT))
        return next((f for f in self.folds if f.held_out_vector == wanted), None)

    def fold(self, vector_id: str) -> FoldResult | None:
        return next((f for f in self.folds if f.held_out_vector == vector_id), None)

    @property
    def measured(self) -> list[FoldResult]:
        return [f for f in self.folds if f.outcome == MEASURED]

    @property
    def not_measured(self) -> list[FoldResult]:
        """Every fold that carries no quotable number, with its reason. Named, never dropped."""
        return [f for f in self.folds if f.outcome != MEASURED]

    def summary(self) -> str:
        head = self.headline
        tail = (
            f"{len(self.measured)}/{len(self.folds)} folds measured"
            if self.folds
            else "no folds attempted"
        )
        return f"{self.dataset}: {head.summary() if head else 'no headline fold'} — {tail}"

    # ── round trip ──────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset": self.dataset,
            "seed": self.seed,
            "config": self.config,
            "operating_point": self.operating_point,
            "split": self.split,
            "data": self.data,
            "folds": [f.to_dict() for f in self.folds],
            "skipped": [
                {"held_out_vector": f.held_out_vector, "outcome": f.outcome, "reason": f.reason}
                for f in self.not_measured
            ],
            "model_card": self.model_card,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LeaveOneAttackOutReport:
        if int(raw.get("version", 0)) != LOAO_ARTEFACT_VERSION:
            raise ValueError(
                f"leave-one-attack-out artefact version {raw.get('version')} != "
                f"{LOAO_ARTEFACT_VERSION}; rebuild it with scripts/build_loao.py rather than "
                "reading it as-is"
            )
        return cls(
            dataset=raw["dataset"],
            seed=int(raw["seed"]),
            config=raw["config"],
            operating_point=raw["operating_point"],
            folds=[FoldResult.from_dict(f) for f in raw["folds"]],
            split=raw.get("split", {}),
            data=raw.get("data", {}),
            model_card=raw.get("model_card", {}),
            meta=raw.get("meta", {}),
        )

    def save(self, directory: str | Path = DEFAULT_LOAO_DIR) -> Path:
        path = Path(directory) / f"{self.dataset}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        return path

    @classmethod
    def load(
        cls, dataset: str, directory: str | Path = DEFAULT_LOAO_DIR
    ) -> LeaveOneAttackOutReport:
        path = Path(directory) / f"{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no leave-one-attack-out report for {dataset!r} at {path} — run "
                "`make loao` once and commit the result"
            )
        return cls.from_dict(json.loads(path.read_text()))


def load_all(directory: str | Path = DEFAULT_LOAO_DIR) -> dict[str, LeaveOneAttackOutReport]:
    """Every committed report on disk, keyed by dataset."""
    directory = Path(directory)
    if not directory.exists():
        return {}
    return {
        path.stem: LeaveOneAttackOutReport.from_dict(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
    }
