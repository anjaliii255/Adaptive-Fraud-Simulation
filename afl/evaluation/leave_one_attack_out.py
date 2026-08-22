"""Leave-one-attack-out: the only evaluation that answers the question we actually asked.

Reporting recall on a family the model trained on measures memorisation. The claim here is
*generalisation to an unseen attack*, so one whole vector is pulled out of training and the
headline number is measured on it alone.

Two guards, both easy to lose and expensive to lose quietly:
  1. the held-out family never appears in training — not one row, not in the replay buffer;
  2. the split is still out-of-time, so "unseen family" does not smuggle in "seen future".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from afl.contract.metrics import MetricResult
from afl.contract.schema import Transaction
from afl.data.splits import assert_no_leakage, out_of_time_split
from afl.evaluation import protocol

log = logging.getLogger(__name__)

DEFAULT_HOLDOUT = "M3"  # the research-maturity drift family: no event to anchor on


def make_splits(
    txns: list[Transaction],
    held_out_vector: str,
    train_frac: float = 0.7,
    embargo_days: float = 1.0,
) -> tuple[list[Transaction], list[Transaction]]:
    """(train, holdout) — out-of-time first, then the family carve-out.

    The holdout keeps all legit rows: without a haystack, FPR and precision@k mean nothing.
    """
    train, test = out_of_time_split(txns, train_frac=train_frac, embargo_days=embargo_days)
    train = [t for t in train if t.vector_id != held_out_vector]
    test = [t for t in test if t.vector_id == held_out_vector or not t.is_fraud]
    return train, test


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
        **kwargs,
    ) -> tuple[LeaveOneAttackOut, list[Transaction]]:
        """Returns (evaluator, train_rows) so a caller cannot accidentally train on the holdout."""
        train, holdout = make_splits(txns, held_out_vector, train_frac, embargo_days)
        assert_no_leakage(train, holdout)
        if not any(t.is_fraud for t in holdout):
            # the out-of-time cut can land after every episode of the held-out family. Every
            # metric then reads 0.0, which looks like failure but is the absence of a measurement.
            log.warning(
                "holdout for %r contains no fraud rows — widen the window or add episodes; "
                "the numbers from this split measure nothing",
                held_out_vector,
            )
        return cls(holdout=holdout, held_out_vector=held_out_vector, **kwargs), train

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
) -> dict[str, MetricResult]:
    """The full matrix: hold out each family in turn, refit from scratch, score it.

    One row per vector is the honest version of a single headline number — a system that only
    generalises to one family will show it here.
    """
    vectors = vectors or sorted({t.vector_id for t in txns if t.vector_id})
    out: dict[str, MetricResult] = {}
    for vid in vectors:
        train, holdout = make_splits(txns, vid, train_frac, embargo_days)
        if not any(t.is_fraud for t in holdout) or not any(t.is_fraud for t in train):
            continue  # nothing to learn from, or nothing to be measured on
        detector = detector_factory()
        detector.fit(train)
        out[vid] = protocol.evaluate_detector(
            detector, holdout, fixed_fpr=fixed_fpr, k=k, held_out_vector=vid
        )
    return out
