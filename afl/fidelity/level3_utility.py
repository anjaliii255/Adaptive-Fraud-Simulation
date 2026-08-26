"""Level 3 — does the synthetic data *do work*? This is the bar.

Levels 1 and 2 measure resemblance. A generator can resemble the real thing closely and still
teach a model nothing. The only question that matters downstream:

  TSTR gap        train without a real fraud label, test on real. How much worse than
                  train-real/test-real, on the same real test window?
  augmentation    real vs real+synthetic, both tested on real. Does adding it help?
  amount floor    rank the same real test window by amount, no model at all. Did either beat it?

**What "train on synthetic" means here is a definition, not a result, so it is written down
before the run** — in `config/fidelity/thresholds.yaml`, under `tstr_definition`. This generator
injects attacks into a real anchor's traffic; it does not claim to synthesise a payment system.
So TSTR trains on the anchor's real *legit* rows plus the generated *fraud*, and never sees a
real fraud label. That is the same system the transfer test called `synthetic` (commit de254c0),
so the two numbers are comparable rather than two different questions wearing one name.

The stricter reading — train on the generator's whole standalone output, background and all —
is reported next to it as `standalone`, because it is what TSTR means in the synthetic-data
literature and a reader is entitled to both. It does not gate: failing a test of something the
generator does not claim tells you nothing about the generator.

The floor is the third column because two results in this repo were walked back for want of it:
the transfer test produced a synthetic-trained detector at 0.238 against a 0.702 amount floor,
and the BankSim spike repeated it. A TSTR PR-AUC that loses to sorting by amount is not a small
gap in utility, it is the absence of any.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from afl.contract.metrics import MetricResult
from afl.contract.schema import Transaction
from afl.evaluation import protocol

log = logging.getLogger(__name__)


class _NoUncalibratedBandWarning(logging.Filter):
    """Drop the decision layer's uncalibrated-bands warning for the duration of this level.

    It is a correct warning and it is about a field nothing here reads. Level 3 measures
    ranking metrics only — PR-AUC, recall at a fixed FPR, precision@k — and the decision layer
    cannot move one of those by construction (`docs/decisions.md`). Firing it four times per
    anchor trains a reader to scroll past a warning that does matter elsewhere.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "UNCALIBRATED" not in str(record.msg)


def _fit_and_score(
    detector_factory: Callable[[], Any],
    train: list[Transaction],
    test: list[Transaction],
    fixed_fpr: float,
    k: int,
) -> MetricResult:
    """One fit, one measurement, at the run's operating point.

    Deliberately `detector.fit`, not the loop's calibrating `build_fit`: every number on this
    level is a ranking metric, and the decision layer cannot move a ranking metric by
    construction (see `docs/decisions.md`). Calibrating here would double the fits and change
    nothing but the runtime.
    """
    decision_log = logging.getLogger("afl.defend.decision")
    quiet = _NoUncalibratedBandWarning()
    decision_log.addFilter(quiet)
    try:
        detector = detector_factory()
        detector.fit(train)
        return protocol.evaluate_detector(detector, test, fixed_fpr, k)
    finally:
        decision_log.removeFilter(quiet)


def _row(result: MetricResult) -> dict[str, Any]:
    return {
        "pr_auc": result.pr_auc,
        "recall_at_fixed_fpr": result.recall_at_fixed_fpr,
        "precision_at_k": result.precision_at_k,
    }


def amount_floor(
    train: list[Transaction],
    test: list[Transaction],
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
) -> dict[str, Any]:
    """Rank by amount alone: no model, no features, no training. The line utility has to clear.

    The direction — largest first or smallest first — is chosen on the **training** window and
    applied to test, because choosing it on the window it is reported from is the tuning-on-test
    everything else here is arranged to avoid. Same rule as `build_baseline.amount_only_reference`
    and `build_anomaly.amount_only`, so the three floors in this repo are one floor.
    """
    y_tr = np.array([int(t.is_fraud) for t in train], dtype=int)
    a_tr = np.array([t.amount for t in train], dtype=float)
    high_first = protocol.pr_auc(y_tr, a_tr) >= protocol.pr_auc(y_tr, -a_tr)

    y_te = np.array([int(t.is_fraud) for t in test], dtype=int)
    a_te = np.array([t.amount for t in test], dtype=float)
    result = protocol.evaluate(y_te, a_te if high_first else -a_te, fixed_fpr=fixed_fpr, k=k)
    return {
        **_row(result),
        "n_positives": result.n_positives,
        "direction": "largest amount first" if high_first else "smallest amount first",
        "direction_chosen_on": "train",
    }


def report(
    real_train: list[Transaction],
    real_test: list[Transaction],
    synth_fraud: list[Transaction],
    detector_factory: Callable[[], Any],
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
    max_gap: float = 0.15,
    standalone: list[Transaction] | None = None,
    min_positives: int = 0,
) -> dict[str, object]:
    """Level 3 verdict. `score` blends "close to real utility" with "actually helps".

    Four systems, one real test window, one operating point:

        trtr        real rows, real labels                 the ceiling: what is learnable here
        tstr        real legit + generated fraud           the gate: does the label transfer?
        augmented   real rows + generated fraud            does adding it help?
        standalone  the generator's whole output           the literature's TSTR, reported only

    `min_positives` is the same bar the leave-one-attack-out matrix uses. Below it the test
    window cannot move a metric by less than a rounding error per row, and the level reports
    `withheld` rather than a number — a thin window is not a failing generator.
    """
    n_positives = sum(1 for t in real_test if t.is_fraud)
    if n_positives < min_positives:
        return {
            "level": 3,
            "outcome": "withheld",
            "why": (
                f"the real test window holds {n_positives} fraud rows, below the "
                f"{min_positives} needed for a metric to mean anything"
            ),
            "n_real_train": len(real_train),
            "n_real_test": len(real_test),
            "n_real_test_positives": n_positives,
            "n_synth_fraud": len(synth_fraud),
            "score": None,
        }

    real_legit = [t for t in real_train if not t.is_fraud]
    if not synth_fraud:
        log.warning("no generated fraud rows — TSTR and augmentation have nothing to add")

    # the ceiling and the augmentation baseline are the same fit, so it is done once
    trtr = _fit_and_score(detector_factory, real_train, real_test, fixed_fpr, k)
    tstr = _fit_and_score(detector_factory, real_legit + synth_fraud, real_test, fixed_fpr, k)
    aug = _fit_and_score(detector_factory, real_train + synth_fraud, real_test, fixed_fpr, k)
    floor = amount_floor(real_train, real_test, fixed_fpr, k)

    gap = round(trtr.pr_auc - tstr.pr_auc, 6)
    lift = round(aug.recall_at_fixed_fpr - trtr.recall_at_fixed_fpr, 6)

    systems = {
        "trtr": {**_row(trtr), "trained_on": "real rows, real labels", "n_train": len(real_train)},
        "tstr": {
            **_row(tstr),
            "trained_on": "real legit + generated fraud, no real fraud label",
            "n_train": len(real_legit) + len(synth_fraud),
        },
        "augmented": {
            **_row(aug),
            "trained_on": "real rows + generated fraud",
            "n_train": len(real_train) + len(synth_fraud),
        },
        "amount_floor": {**floor, "trained_on": "nothing", "n_train": 0},
    }
    if standalone:
        alone = _fit_and_score(detector_factory, standalone, real_test, fixed_fpr, k)
        systems["standalone"] = {
            **_row(alone),
            "trained_on": "the generator's whole output, background included",
            "n_train": len(standalone),
            "n_train_fraud": sum(1 for t in standalone if t.is_fraud),
            "note": "reported, never gating — see this module's docstring",
        }

    gap_score = max(0.0, 1.0 - abs(gap) / max_gap)
    lift_score = min(1.0, max(0.0, 0.5 + lift * 5))  # 0.5 = no lift, 1.0 = +10pp
    return {
        "level": 3,
        "outcome": "measured",
        "systems": systems,
        "tstr": {
            "trtr_pr_auc": trtr.pr_auc,
            "tstr_pr_auc": tstr.pr_auc,
            "trtr_recall": trtr.recall_at_fixed_fpr,
            "tstr_recall": tstr.recall_at_fixed_fpr,
            "tstr_gap": gap,
            # a ratio, so the gap is judged against how learnable the problem was in the first place
            "tstr_ratio": round(tstr.pr_auc / trtr.pr_auc, 6) if trtr.pr_auc > 0 else 0.0,
        },
        "augmentation": {
            "base_pr_auc": trtr.pr_auc,
            "augmented_pr_auc": aug.pr_auc,
            "base_recall": trtr.recall_at_fixed_fpr,
            "augmented_recall": aug.recall_at_fixed_fpr,
            "recall_lift": lift,
            "pr_auc_lift": round(aug.pr_auc - trtr.pr_auc, 6),
        },
        "amount_floor": floor,
        # the comparison the learned columns are read against, computed rather than eyeballed
        "beats_amount_floor": {
            name: bool(body["pr_auc"] > floor["pr_auc"])
            for name, body in systems.items()
            if name != "amount_floor"
        },
        "n_real_train": len(real_train),
        "n_real_test": len(real_test),
        "n_real_test_positives": n_positives,
        "n_synth_fraud": len(synth_fraud),
        "score": round(0.5 * gap_score + 0.5 * lift_score, 4),
    }
