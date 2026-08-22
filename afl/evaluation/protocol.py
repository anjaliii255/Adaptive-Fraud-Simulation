"""The metric protocol. One definition, used by every system in the table.

Accuracy and ROC-AUC are useless at a 0.1% base rate — both are dominated by the negatives.
The three numbers here are the ones a fraud team actually operates on:

  * PR-AUC              — ranking quality where the positives live
  * recall @ fixed FPR  — how much fraud we catch at a review budget we can afford
  * precision @ k       — what the analyst queue looks like on the day

Every one of them is computed at a *fixed* operating point, agreed once, so two systems are
never compared at two different thresholds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from afl.contract.metrics import Action, DetectorScore, MetricResult
from afl.contract.schema import AttackBatch, AttackParams, Transaction

DEFAULT_FPR = 0.01
DEFAULT_K = 100


def pr_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Average precision. The ranking metric that survives a 0.1% base rate."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return 0.0
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def recall_at_fixed_fpr(y_true: ArrayLike, scores: ArrayLike, fpr: float = DEFAULT_FPR) -> float:
    """Recall at the threshold that spends exactly `fpr` of the legit traffic."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    neg, pos = s[y == 0], s[y == 1]
    if neg.size == 0 or pos.size == 0:
        return 0.0
    thr = float(np.quantile(neg, 1.0 - fpr))
    return float((pos > thr).mean())


def precision_at_k(y_true: ArrayLike, scores: ArrayLike, k: int = DEFAULT_K) -> float:
    """Precision in the top-k queue — the analyst's day, not the model's ROC curve."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return 0.0
    k = min(k, y.size)
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].mean())


def evaluate(
    y_true: ArrayLike,
    scores: ArrayLike,
    fixed_fpr: float = DEFAULT_FPR,
    k: int = DEFAULT_K,
    held_out_vector: str | None = None,
) -> MetricResult:
    """All three metrics at one agreed operating point."""
    y = np.asarray(y_true, dtype=int)
    return MetricResult(
        pr_auc=round(pr_auc(y, scores), 6),
        recall_at_fixed_fpr=round(recall_at_fixed_fpr(y, scores, fixed_fpr), 6),
        fixed_fpr=fixed_fpr,
        precision_at_k=round(precision_at_k(y, scores, k), 6),
        held_out_vector=held_out_vector,
        k=min(k, int(y.size)),
        n_positives=int(y.sum()),
    )


def align(txns: list[Transaction], scores: list[DetectorScore]) -> tuple[np.ndarray, np.ndarray]:
    """(labels, scores) aligned by txn_id — never by position."""
    by_id = {s.txn_id: s for s in scores}
    missing = [t.txn_id for t in txns if t.txn_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} transaction(s) unscored, e.g. {missing[:3]}")
    y = np.array([int(t.is_fraud) for t in txns], dtype=int)
    s = np.array([by_id[t.txn_id].score for t in txns], dtype=float)
    return y, s


def score_transactions(
    detector, txns: list[Transaction], run_id: str = "eval"
) -> list[DetectorScore]:
    """Run a detector over a bare transaction list by wrapping it in a throwaway batch."""
    batch = AttackBatch(
        run_id=run_id,
        params=AttackParams(vector_id="eval", engine="none"),
        transactions=list(txns),
        seed=0,
    )
    return detector.score(batch)


def evaluate_detector(
    detector,
    txns: list[Transaction],
    fixed_fpr: float = DEFAULT_FPR,
    k: int = DEFAULT_K,
    held_out_vector: str | None = None,
) -> MetricResult:
    """Score a detector on a transaction list and return the measurement."""
    y, s = align(txns, score_transactions(detector, txns))
    return evaluate(y, s, fixed_fpr, k, held_out_vector)


def operational_rates(txns: list[Transaction], scores: list[DetectorScore]) -> dict[str, float]:
    """What the policy actually did — the half of the story the ranking metrics cannot tell.

    A model can rank perfectly and still let everything through if the bands are set wrong.
    """
    by_id = {s.txn_id: s for s in scores}
    fraud = [t for t in txns if t.is_fraud]
    legit = [t for t in txns if not t.is_fraud]
    evaded = [t for t in fraud if by_id[t.txn_id].action == Action.ALLOW]
    declined_legit = [t for t in legit if by_id[t.txn_id].action == Action.DECLINE]
    frictioned_legit = [t for t in legit if by_id[t.txn_id].action != Action.ALLOW]
    return {
        "evasion_rate": len(evaded) / len(fraud) if fraud else 0.0,
        "caught_rate": 1.0 - (len(evaded) / len(fraud)) if fraud else 0.0,
        "false_decline_rate": len(declined_legit) / len(legit) if legit else 0.0,
        "friction_rate": len(frictioned_legit) / len(legit) if legit else 0.0,
        "amount_evaded": float(sum(t.amount for t in evaded)),
    }
