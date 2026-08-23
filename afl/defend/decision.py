"""Score → action. Graded, and priced.

A binary block/allow decision throws away the part of the distribution where the money is: the
uncertain middle, where friction is cheap and a decline is expensive. Thresholds here are derived
from a cost model rather than picked by eye, so "what FPR do we run at" becomes a business
question with an auditable answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from afl.contract.metrics import Action, DetectorScore


@dataclass(frozen=True)
class CostModel:
    """Costs per transaction, as a fraction of amount unless the field says otherwise."""

    fraud_loss_rate: float = 1.0  # a missed fraud loses the full amount
    false_decline_cost: float = 0.35  # lost margin + churn risk on a wrongly declined good txn
    review_cost: float = 4.0  # flat analyst cost per manual review
    step_up_cost: float = 0.5  # flat friction cost (abandonment risk) per challenge
    step_up_efficacy: float = 0.8  # share of fraud a challenge actually stops
    hold_cost: float = 1.5  # flat cost of delaying settlement

    def expected_cost(self, action: Action, p_fraud: float, amount: float) -> float:
        loss = p_fraud * amount * self.fraud_loss_rate
        if action is Action.ALLOW:
            return loss
        if action is Action.STEP_UP:
            return loss * (1 - self.step_up_efficacy) + self.step_up_cost
        if action is Action.HOLD:
            return loss * 0.5 + self.hold_cost
        if action is Action.REVIEW:
            return self.review_cost + loss * 0.1
        if action is Action.DECLINE:
            return (1 - p_fraud) * amount * self.false_decline_cost
        raise ValueError(action)


#: The house cost model. Override per call rather than mutating this.
DEFAULT_COSTS = CostModel()


@dataclass
class DecisionPolicy:
    """Either fixed thresholds, or cost-minimising (`mode="cost"`) per transaction."""

    step_up_at: float = 0.20
    hold_at: float = 0.50
    review_at: float = 0.70
    decline_at: float = 0.90
    mode: str = "threshold"  # "threshold" | "cost"
    costs: CostModel = field(default_factory=CostModel)

    def act(self, score: float, amount: float = 100.0) -> Action:
        if self.mode == "cost":
            return min(Action, key=lambda a: self.costs.expected_cost(a, score, amount))
        if score >= self.decline_at:
            return Action.DECLINE
        if score >= self.review_at:
            return Action.REVIEW
        if score >= self.hold_at:
            return Action.HOLD
        if score >= self.step_up_at:
            return Action.STEP_UP
        return Action.ALLOW

    def decide(
        self, txn_id: str, score: float, amount: float = 100.0, reasons: list[str] | None = None
    ) -> DetectorScore:
        return DetectorScore(
            txn_id=txn_id,
            score=float(min(max(score, 0.0), 1.0)),
            action=self.act(score, amount),
            reasons=reasons or [],
        )

    def calibrate_to_fpr(self, scores, labels, target_fpr: float = 0.01) -> DecisionPolicy:
        """Set `decline_at` to the score that hits a target FPR on a held-out set.

        The other bands are placed proportionally below it. Calibrate on validation data only —
        calibrating on the test set is how an honest pipeline quietly becomes a dishonest one.
        """
        import numpy as np

        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        negatives = np.sort(scores[labels == 0])[::-1]
        if len(negatives) == 0:
            return self
        idx = min(len(negatives) - 1, max(0, int(round(target_fpr * len(negatives))) - 1))
        self.decline_at = float(negatives[idx])
        self.review_at = self.decline_at * 0.8
        self.hold_at = self.decline_at * 0.6
        self.step_up_at = self.decline_at * 0.3
        return self


def assert_one_operating_point(calibrate_to_fpr: float | None, fixed_fpr: float) -> None:
    """The action bands and the reported metric are one decision, so they take one number.

    `defend.supervised.decision.calibrate_to_fpr` places the bands; `eval.fixed_fpr` is where
    recall is read off. Two different values means the table's `recall@1%FPR` column and its
    `evasion_rate` column describe two different systems — which is the sort of discrepancy
    nobody notices until they are asked to explain it on a slide.
    """
    if calibrate_to_fpr is None:  # calibration off; the fixed bands in config stand
        return
    if float(calibrate_to_fpr) != float(fixed_fpr):
        raise ValueError(
            f"two operating points: the bands are calibrated to {calibrate_to_fpr} but recall is "
            f"reported at {fixed_fpr}. Set defend.supervised.decision.calibrate_to_fpr and "
            "eval.fixed_fpr to the same number, or set the former to null to keep fixed bands."
        )


def total_cost(
    scores: list[DetectorScore],
    amounts: dict[str, float],
    labels: dict[str, int],
    costs: CostModel | None = None,
) -> float:
    """Realised cost of a decision set — the number a fraud lead actually cares about."""
    costs = costs or DEFAULT_COSTS
    return sum(
        costs.expected_cost(s.action, float(labels.get(s.txn_id, 0)), amounts.get(s.txn_id, 0.0))
        for s in scores
    )
