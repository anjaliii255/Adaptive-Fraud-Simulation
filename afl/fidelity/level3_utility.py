"""Level 3 — does the synthetic data *do work*? This is the bar.

Levels 1 and 2 measure resemblance. A generator can resemble the real thing closely and still
teach a model nothing. The only question that matters downstream:

  TSTR gap        train on synthetic, test on real. How much worse than train-real/test-real?
  augmentation    real vs real+synthetic, both tested on real. Does adding it help?

Both are measured on **real** held-out data at the same operating point as everything else.
A generator that fails here does not get to be defended by its level-1 histograms.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from afl.contract.schema import Transaction
from afl.evaluation import protocol

log = logging.getLogger(__name__)


def _has_both_classes(txns: list[Transaction]) -> bool:
    labels = {t.is_fraud for t in txns}
    return len(labels) == 2


def _fit(detector_factory: Callable[[], Any], rows: list[Transaction]):
    detector = detector_factory()
    detector.fit(rows)
    return detector


def tstr(
    real_train: list[Transaction],
    real_test: list[Transaction],
    synth: list[Transaction],
    detector_factory: Callable[[], Any],
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
) -> dict[str, float]:
    """Train-Synthetic-Test-Real against Train-Real-Test-Real, same test set both times."""
    if not _has_both_classes(synth):
        log.warning(
            "synthetic set is single-class (%d rows) — TSTR needs its own legit background",
            len(synth),
        )
    trtr = protocol.evaluate_detector(_fit(detector_factory, real_train), real_test, fixed_fpr, k)
    tstr_m = protocol.evaluate_detector(_fit(detector_factory, synth), real_test, fixed_fpr, k)

    gap = trtr.pr_auc - tstr_m.pr_auc
    return {
        "trtr_pr_auc": trtr.pr_auc,
        "tstr_pr_auc": tstr_m.pr_auc,
        "trtr_recall": trtr.recall_at_fixed_fpr,
        "tstr_recall": tstr_m.recall_at_fixed_fpr,
        "tstr_gap": round(gap, 6),
        # a ratio, so the gap is judged against how learnable the problem was in the first place
        "tstr_ratio": round(tstr_m.pr_auc / trtr.pr_auc, 6) if trtr.pr_auc > 0 else 0.0,
    }


def augmentation_lift(
    real_train: list[Transaction],
    real_test: list[Transaction],
    synth: list[Transaction],
    detector_factory: Callable[[], Any],
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
) -> dict[str, float]:
    """Held-out recall lift from adding the synthetic rows to a real training set."""
    base = protocol.evaluate_detector(_fit(detector_factory, real_train), real_test, fixed_fpr, k)
    aug = protocol.evaluate_detector(
        _fit(detector_factory, real_train + synth), real_test, fixed_fpr, k
    )
    return {
        "base_pr_auc": base.pr_auc,
        "augmented_pr_auc": aug.pr_auc,
        "base_recall": base.recall_at_fixed_fpr,
        "augmented_recall": aug.recall_at_fixed_fpr,
        "recall_lift": round(aug.recall_at_fixed_fpr - base.recall_at_fixed_fpr, 6),
        "pr_auc_lift": round(aug.pr_auc - base.pr_auc, 6),
    }


def report(
    real_train: list[Transaction],
    real_test: list[Transaction],
    synth: list[Transaction],
    detector_factory: Callable[[], Any],
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
    max_gap: float = 0.15,
) -> dict[str, object]:
    """Level 3 verdict. `score` blends "close to real utility" with "actually helps"."""
    t = tstr(real_train, real_test, synth, detector_factory, fixed_fpr, k)
    a = augmentation_lift(real_train, real_test, synth, detector_factory, fixed_fpr, k)

    gap_score = max(0.0, 1.0 - abs(t["tstr_gap"]) / max_gap)
    lift_score = min(1.0, max(0.0, 0.5 + a["recall_lift"] * 5))  # 0.5 = no lift, 1.0 = +10pp
    return {
        "level": 3,
        "tstr": t,
        "augmentation": a,
        "n_real_train": len(real_train),
        "n_real_test": len(real_test),
        "n_synth": len(synth),
        "score": round(0.5 * gap_score + 0.5 * lift_score, 4),
    }
