"""Scoring + evaluation types — the return half of the seam."""

from __future__ import annotations

from enum import Enum

import pydantic


class Action(str, Enum):
    """Graded response. A binary block/allow decision throws away the interesting middle."""

    ALLOW = "allow"
    STEP_UP = "step_up"
    HOLD = "hold"
    REVIEW = "review"
    DECLINE = "decline"


#: An attack "evades" only if it was let through untouched. Anything friction-bearing counts
#: as caught for loop purposes — both sides must agree on this, so it lives in the contract.
EVASION_ACTIONS: frozenset[Action] = frozenset({Action.ALLOW})


class DetectorScore(pydantic.BaseModel):
    """One transaction's verdict: probability, action taken, and why."""

    txn_id: str
    score: float  # P(fraud)
    action: Action
    reasons: list[str] = pydantic.Field(default_factory=list)  # SHAP top features

    @pydantic.field_validator("score")
    @classmethod
    def _score_in_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("score must be a probability in [0, 1]")
        return v

    @property
    def evaded(self) -> bool:
        return self.action in EVASION_ACTIONS


class MetricResult(pydantic.BaseModel):
    """A measurement plus the operating point it was measured at."""

    pr_auc: float
    recall_at_fixed_fpr: float
    fixed_fpr: float
    precision_at_k: float
    held_out_vector: str | None = None  # set for leave-one-attack-out runs
    k: int | None = None
    n_positives: int | None = None
