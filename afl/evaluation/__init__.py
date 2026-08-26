"""■ B — how a number becomes a claim.

Nothing here trains a model; it only measures one, at an operating point fixed in advance.
"""

from afl.evaluation.leave_one_attack_out import (
    MEASURED,
    MIN_MEANINGFUL_POSITIVES,
    SKIPPED,
    WITHHELD,
    Fold,
    FoldResult,
    GuardFailed,
    LeaveOneAttackOut,
    LeaveOneAttackOutReport,
    assert_embargo_intact,
    assert_family_held_out,
    assert_haystack_intact,
    make_splits,
    run_fold,
    sweep,
)
from afl.evaluation.protocol import (
    evaluate,
    evaluate_detector,
    operational_rates,
    pr_auc,
    precision_at_k,
    recall_at_fixed_fpr,
)
from afl.evaluation.three_system import measure, run_three_systems, smote_transactions

__all__ = [
    "MEASURED",
    "MIN_MEANINGFUL_POSITIVES",
    "SKIPPED",
    "WITHHELD",
    "Fold",
    "FoldResult",
    "GuardFailed",
    "LeaveOneAttackOut",
    "LeaveOneAttackOutReport",
    "assert_embargo_intact",
    "assert_family_held_out",
    "assert_haystack_intact",
    "evaluate",
    "evaluate_detector",
    "make_splits",
    "measure",
    "operational_rates",
    "pr_auc",
    "precision_at_k",
    "recall_at_fixed_fpr",
    "run_fold",
    "run_three_systems",
    "smote_transactions",
    "sweep",
]
