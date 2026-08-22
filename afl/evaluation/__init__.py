"""■ B — how a number becomes a claim.

Nothing here trains a model; it only measures one, at an operating point fixed in advance.
"""

from afl.evaluation.leave_one_attack_out import LeaveOneAttackOut, make_splits, sweep
from afl.evaluation.protocol import (
    evaluate,
    evaluate_detector,
    operational_rates,
    pr_auc,
    precision_at_k,
    recall_at_fixed_fpr,
)
from afl.evaluation.three_system import run_three_systems, smote_transactions

__all__ = [
    "LeaveOneAttackOut",
    "evaluate",
    "evaluate_detector",
    "make_splits",
    "operational_rates",
    "pr_auc",
    "precision_at_k",
    "recall_at_fixed_fpr",
    "run_three_systems",
    "smote_transactions",
    "sweep",
]
