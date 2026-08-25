"""Score → action. Graded, and priced.

A binary block/allow decision throws away the part of the distribution where the money is: the
uncertain middle, where friction is cheap and a decline is expensive. So there are five actions,
and which one a transaction gets is whichever **minimises expected cost** at that transaction's
own probability and its own amount. "What FPR do we run at" stops being a number somebody picked
and becomes a business question with an auditable answer: it is an *output* of the cost model,
reported next to the metric rather than assumed by it.

Three things make that claim survive contact with the data, and each of them was a bug the first
time round:

**The score has to be a probability.** `p x amount` against a flat analyst cost is arithmetic on
a probability, and a boosted tree's `predict_proba` is a ranking score. `afl/defend/calibration.py`
maps one to the other on the validation tail; the map is monotone, so not one reported metric
moves.

**The flat costs have to be portable.** PaySim's median payment is 74,872 and AMLSim's is 157.
A review priced at "4.0" is a rounding error on one anchor and a fortune on the other, and the
same config would decline everything on one and nothing on the other while looking principled
both times. Flat costs are therefore quoted in units of `unit_amount` — the anchor's median
payment — and resolved to absolute currency at load. See `CostModel.from_config`.

**The ladder has to actually be a ladder.** An action that stops less fraud than the cheaper
action below it is dominated: it can never minimise expected cost, at any probability or any
amount, and the graded policy quietly collapses to four bands. The house numbers had exactly
this defect — a hold cost 3x a step-up and stopped less — so `CostModel.__post_init__` now
refuses a non-monotone ladder instead of silently dropping a rung.

The bands (`step_up_at` ... `decline_at`) are still here, because a threshold is what an auditor,
a model card and a Streamlit gauge can read. They are **derived** from the cost model at a
reference amount, not typed in: `DecisionPolicy()` computes them, and
`tests/test_decision.py` asserts the derived ladder and the per-transaction argmin agree
row for row at that amount.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from afl.contract.metrics import Action, DetectorScore
from afl.defend.calibration import ScoreCalibrator

log = logging.getLogger(__name__)

#: The ladder, least to most severe. Everything in this module assumes this order: an action's
#: efficacy and its cost must both rise along it, or the rung below dominates it.
SEVERITY: tuple[Action, ...] = (
    Action.ALLOW,
    Action.STEP_UP,
    Action.HOLD,
    Action.REVIEW,
    Action.DECLINE,
)

#: Which policy attribute holds each action's band edge.
BAND_ATTR: dict[Action, str] = {
    Action.STEP_UP: "step_up_at",
    Action.HOLD: "hold_at",
    Action.REVIEW: "review_at",
    Action.DECLINE: "decline_at",
}

#: A band edge above 1.0 is one no score can cross. Used where the cost model never chooses that
#: action at the reference amount — recorded as an unreachable band rather than dropped, because
#: "the review band never opens on this anchor" is a finding and a missing key is not.
UNREACHABLE = 2.0


class DominatedAction(ValueError):
    """Raised when a cost model prices a rung so that a cheaper one always beats it."""


@dataclass(frozen=True)
class CostModel:
    """What each action costs, per transaction, in the units the anchor quotes `amount` in.

    Amount-proportional terms (`fraud_loss_rate`, `false_decline_cost`) are fractions of the
    transaction amount. Flat terms (`step_up_cost`, `hold_cost`, `review_cost`) are absolute
    currency here — `from_config` is what converts the portable, `unit_amount`-relative numbers
    in `config/costs/*.yaml` into these.

    `*_efficacy` is the share of fraud the action actually stops. A decline stops all of it by
    construction and pays for that in false declines instead; an allow stops none.
    """

    fraud_loss_rate: float = 1.0  # a missed fraud loses this share of the amount
    false_decline_cost: float = 0.35  # lost margin + churn on a wrongly declined good txn
    step_up_cost: float = 0.5  # flat friction cost (abandonment risk) per challenge
    step_up_efficacy: float = 0.80
    hold_cost: float = 1.5  # flat cost of delaying settlement
    hold_efficacy: float = 0.85
    review_cost: float = 4.0  # flat analyst cost per manual review
    review_efficacy: float = 0.95
    #: The currency unit the flat costs above were quoted in before resolution. 1.0 means they
    #: are absolute. Carried so an artefact can say what "review_cost = 4,492" was 0.06 *of*.
    unit_amount: float = 1.0
    #: `parameter -> why this number`, straight from config. Empty for a hand-built model.
    rationale: dict[str, str] = field(default_factory=dict)
    source: str = "defaults in afl/defend/decision.py"

    # ── the ladder ──────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        efficacies = [self.efficacy(a) for a in SEVERITY]
        flats = [self.flat_cost(a) for a in SEVERITY[:-1]]  # decline's cost is proportional
        for i in range(1, len(SEVERITY) - 1):
            lower, upper = SEVERITY[i - 1], SEVERITY[i]
            if efficacies[i] <= efficacies[i - 1] or flats[i] <= flats[i - 1]:
                raise DominatedAction(
                    f"{upper.value} is dominated by {lower.value}: it stops "
                    f"{efficacies[i]:.0%} of fraud for {flats[i]:,.4g} against {lower.value}'s "
                    f"{efficacies[i - 1]:.0%} for {flats[i - 1]:,.4g}. A rung that costs more "
                    "and catches less can never minimise expected cost at any probability or "
                    "any amount, so the graded ladder silently loses it. Fix the cost model, "
                    "or drop the action."
                )
        if self.false_decline_cost < 0.0 or self.fraud_loss_rate < 0.0:
            raise ValueError("costs cannot be negative")
        if not 0.0 <= self.review_efficacy < 1.0:
            raise DominatedAction(
                f"review_efficacy={self.review_efficacy} leaves nothing for a decline to add: a "
                "decline stops all of the fraud by construction, so a review that also stops all "
                "of it dominates the decline and the top of the ladder disappears."
            )

    def efficacy(self, action: Action) -> float:
        """Share of fraud this action stops."""
        return {
            Action.ALLOW: 0.0,
            Action.STEP_UP: self.step_up_efficacy,
            Action.HOLD: self.hold_efficacy,
            Action.REVIEW: self.review_efficacy,
            Action.DECLINE: 1.0,
        }[action]

    def flat_cost(self, action: Action) -> float:
        """Cost paid on every transaction receiving this action, fraud or not."""
        return {
            Action.ALLOW: 0.0,
            Action.STEP_UP: self.step_up_cost,
            Action.HOLD: self.hold_cost,
            Action.REVIEW: self.review_cost,
            Action.DECLINE: 0.0,  # a decline pays in false declines, below, not a flat fee
        }[action]

    # ── the arithmetic ──────────────────────────────────────────────────────────
    def expected_cost(self, action: Action, p_fraud: float, amount: float) -> float:
        """Expected cost of taking `action` on a transaction worth `amount` at risk `p_fraud`."""
        if action is Action.DECLINE:
            return (1.0 - p_fraud) * amount * self.false_decline_cost
        loss = p_fraud * amount * self.fraud_loss_rate
        return loss * (1.0 - self.efficacy(action)) + self.flat_cost(action)

    def costs(self, p_fraud: float, amount: float) -> dict[Action, float]:
        """Every action priced at once — what the reason code quotes."""
        return {a: self.expected_cost(a, p_fraud, amount) for a in SEVERITY}

    def action_for(self, p_fraud: float, amount: float) -> Action:
        """The cost-minimising action. Ties go to the *less* severe rung, which is the point.

        Written out rather than `min(SEVERITY, key=...)` because this runs once per transaction
        and the loop is the whole hot path of the decision layer.
        """
        best, best_cost = Action.ALLOW, p_fraud * amount * self.fraud_loss_rate
        for action in SEVERITY[1:]:
            cost = self.expected_cost(action, p_fraud, amount)
            if cost < best_cost - 1e-12:
                best, best_cost = action, cost
        return best

    def bands(self, amount: float) -> dict[Action, float]:
        """The probability at which each action becomes cost-minimising, at this amount.

        Every action's cost is linear in `p` — slope `amount x fraud_loss_rate x (1 - efficacy)`,
        or `-amount x false_decline_cost` for a decline — so the cost-minimising action as `p`
        sweeps 0 to 1 is the lower envelope of five straight lines. The crossovers of that
        envelope *are* the action bands, and there is nothing left to choose by eye.

        Actions the envelope never selects inside [0, 1] are absent from the result; the caller
        records them as `UNREACHABLE`. That happens for real and it is worth seeing: on a small
        enough payment, a flat analyst cost exceeds the entire amount at risk and the review band
        never opens at all.
        """
        lines = [
            (
                action,
                amount * self.fraud_loss_rate * (1.0 - self.efficacy(action))
                if action is not Action.DECLINE
                else -amount * self.false_decline_cost,
                self.flat_cost(action)
                if action is not Action.DECLINE
                else amount * self.false_decline_cost,
            )
            for action in SEVERITY
        ]
        return {a: p for a, p in _lower_envelope(lines) if 0.0 < p <= 1.0}

    def dominated_at(self, amount: float) -> list[Action]:
        """Rungs the cost model never chooses at this amount. Reported, never silently dropped."""
        reachable = set(self.bands(amount)) | {Action.ALLOW}
        return [a for a in SEVERITY if a not in reachable]

    # ── loading ─────────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg: dict[str, Any], unit_amount: float | None = None) -> CostModel:
        """Build from `config/costs/*.yaml`, where every number carries its own rationale.

        Each parameter is a `{value, why}` pair and a blank `why` is refused. That is the
        enforceable half of "the cost model's parameters live in config with a stated rationale
        per number": a comment can be deleted without anything noticing, a required field cannot.

        Flat costs in the config are quoted as multiples of `unit_amount` and are multiplied out
        here. `unit_amount` comes from the anchor (its median payment) when the caller measured
        one, else from the config's own value, else 1.0 — which leaves the flat costs absolute.
        """
        flat = ("step_up_cost", "hold_cost", "review_cost")
        wanted = (
            "fraud_loss_rate",
            "false_decline_cost",
            "step_up_efficacy",
            "hold_efficacy",
            "review_efficacy",
            *flat,
        )
        values: dict[str, float] = {}
        rationale: dict[str, str] = {}
        missing: list[str] = []
        for name in (*wanted, "unit_amount"):
            entry = cfg.get(name)
            if not isinstance(entry, dict) or "value" not in entry:
                missing.append(f"{name} (expected a {{value, why}} pair)")
                continue
            why = str(entry.get("why") or "").strip()
            if not why:
                missing.append(f"{name} (no `why`)")
                continue
            rationale[name] = why
            if entry["value"] is not None:
                values[name] = float(entry["value"])
        if missing:
            raise ValueError(
                "every cost parameter needs a value and a stated rationale; missing: "
                + ", ".join(missing)
            )

        unit = float(unit_amount or values.get("unit_amount") or 1.0)
        resolved = {k: v for k, v in values.items() if k in wanted}
        for name in flat:
            resolved[name] = resolved[name] * unit
        return cls(
            **resolved,
            unit_amount=unit,
            rationale=rationale,
            source=str(cfg.get("name", "config/costs")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "unit_amount": round(self.unit_amount, 6),
            "fraud_loss_rate": self.fraud_loss_rate,
            "false_decline_cost": self.false_decline_cost,
            "step_up_cost": round(self.step_up_cost, 6),
            "step_up_efficacy": self.step_up_efficacy,
            "hold_cost": round(self.hold_cost, 6),
            "hold_efficacy": self.hold_efficacy,
            "review_cost": round(self.review_cost, 6),
            "review_efficacy": self.review_efficacy,
            "rationale": dict(self.rationale),
        }


def _lower_envelope(lines: list[tuple[Action, float, float]]) -> list[tuple[Action, float]]:
    """(action, p at which it takes over), in increasing p, for lines given as (key, slope, y0).

    The standard convex-hull trick for a minimum over lines: sort by decreasing slope, then
    push, popping any line the new one has already overtaken by the time the previous crossover
    arrives — that line is never the cheapest anywhere and must not become a band.
    """
    hull: list[tuple[Action, float, float, float]] = []  # action, slope, intercept, p_start
    for action, slope, intercept in sorted(lines, key=lambda line: (-line[1], line[2])):
        if hull and abs(hull[-1][1] - slope) < 1e-15:
            continue  # same slope, and sorted so this one's intercept is no lower: dominated
        while hull:
            _, m0, b0, p0 = hull[-1]
            if (intercept - b0) / (m0 - slope) <= p0:
                hull.pop()
            else:
                break
        if not hull:
            hull.append((action, slope, intercept, 0.0))
        else:
            _, m0, b0, _ = hull[-1]
            hull.append((action, slope, intercept, (intercept - b0) / (m0 - slope)))
    return [(action, p) for action, _, _, p in hull]


#: The house cost model. Override per call rather than mutating this.
DEFAULT_COSTS = CostModel()


@dataclass
class DecisionPolicy:
    """Score → graded action, priced by `costs` and reported as bands.

    `mode="cost"` (the default) prices every action at the transaction's own amount and takes the
    cheapest. `mode="threshold"` compares against the four band edges, which is what a detector
    with no amount to hand — or a demo gauge — needs.

    The bands default to **derived**: left at `None` they are read off `costs.bands(
    reference_amount)`, so even threshold mode is running numbers the cost model chose. Passing
    an explicit band keeps it, which is what `calibrate_to_fpr` and the legacy configs rely on.
    """

    step_up_at: float | None = None
    hold_at: float | None = None
    review_at: float | None = None
    decline_at: float | None = None
    mode: str = "cost"  # "cost" | "threshold"
    costs: CostModel = field(default_factory=CostModel)
    #: Maps the detector's score onto P(fraud) before any of the above is applied. Unfitted, it
    #: is the identity, and the policy behaves exactly as it did before calibration existed.
    calibrator: ScoreCalibrator = field(default_factory=ScoreCalibrator)
    #: The amount the reported bands are quoted at. Not used by cost mode, which has the real
    #: amount; used by threshold mode, by the model card, and by `docs/decisions.md`.
    reference_amount: float = 100.0
    #: Set when the bands were typed into config rather than derived, so an artefact can say so.
    bands_source: str = "derived from the cost model"
    _warned_uncalibrated: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("cost", "threshold"):
            raise ValueError(f"unknown decision mode {self.mode!r}; expected 'cost' or 'threshold'")
        explicit = [a for a in BAND_ATTR.values() if getattr(self, a) is not None]
        if explicit and len(explicit) < len(BAND_ATTR):
            raise ValueError(
                f"partial bands {explicit}: set all four or none. Half a ladder derived from the "
                "cost model and half typed in is the ambiguity this ticket removed."
            )
        if explicit:
            self.bands_source = "explicit, from the caller"
        else:
            self.derive_bands()

    # ── bands ───────────────────────────────────────────────────────────────────
    def derive_bands(self, amount: float | None = None) -> DecisionPolicy:
        """Recompute the band edges from the cost model; unreachable rungs get `UNREACHABLE`."""
        self.reference_amount = float(amount if amount is not None else self.reference_amount)
        derived = self.costs.bands(self.reference_amount)
        for action, attr in BAND_ATTR.items():
            setattr(self, attr, float(derived.get(action, UNREACHABLE)))
        self.bands_source = "derived from the cost model"
        return self

    @property
    def band_edges(self) -> dict[Action, float]:
        return {action: float(getattr(self, attr)) for action, attr in BAND_ATTR.items()}

    # ── deciding ────────────────────────────────────────────────────────────────
    def probability(self, score: float) -> float:
        """The detector's score as P(fraud). The identity until a calibrator has been fitted."""
        return self.calibrator.probability(score)

    def act_on_probability(self, p_fraud: float, amount: float) -> Action:
        """The action for an already-calibrated probability."""
        if self.mode == "cost":
            return self.costs.action_for(p_fraud, amount)
        if p_fraud >= self.decline_at:
            return Action.DECLINE
        if p_fraud >= self.review_at:
            return Action.REVIEW
        if p_fraud >= self.hold_at:
            return Action.HOLD
        if p_fraud >= self.step_up_at:
            return Action.STEP_UP
        return Action.ALLOW

    def act(self, score: float, amount: float = 100.0) -> Action:
        """The action for a raw detector score."""
        self._warn_if_uncalibrated()
        return self.act_on_probability(self.probability(score), amount)

    def _warn_if_uncalibrated(self) -> None:
        """Say it once, loudly: cost mode on a ranking score is arithmetic in the wrong units.

        Measured rather than feared. On the synthetic loop, the same run with
        `decision.calibration=none` puts friction on **99.3%** of legit traffic against 9.3%
        with the map fitted — because an untuned tree's raw scores sit in a range where a band
        placed at a genuine probability of 0.005 catches nearly everything. The policy still
        runs, because a caller may have calibrated the scores upstream, but nobody should reach
        that state without being told.
        """
        if self._warned_uncalibrated or self.mode != "cost" or self.calibrator.fitted:
            return
        self._warned_uncalibrated = True
        log.warning(
            "cost-mode decisions on an UNCALIBRATED score (%s). The bands are probabilities and "
            "the score is a ranking statistic, so the two are not on the same scale. Fit the "
            "calibrator on a validation tail before deciding, or set decision.mode=threshold.",
            self.calibrator.note,
        )

    def rationale(self, p_fraud: float, amount: float, action: Action) -> str:
        """Why this action and not the one below it — in the reason list, next to the features."""
        if self.mode == "cost":
            priced = self.costs.costs(p_fraud, amount)
            runner_up = min((a for a in SEVERITY if a is not action), key=lambda a: priced[a])
            return (
                f"decision: {action.value} at p={p_fraud:.4g} on {amount:,.2f} — expected cost "
                f"{priced[action]:,.2f} against {runner_up.value} at {priced[runner_up]:,.2f}"
            )
        edge = self.band_edges[action]
        return (
            f"decision: {action.value} — p={p_fraud:.4g} is over the "
            f"{action.value} band at {edge:.4g}"
        )

    def decide(
        self, txn_id: str, score: float, amount: float = 100.0, reasons: list[str] | None = None
    ) -> DetectorScore:
        """The seam's return type: the detector's score, a graded action, and why.

        **`DetectorScore.score` stays the detector's own score, not the calibrated probability.**
        The calibrated probability is what chooses the action and what the reason code quotes;
        it does not reach the field every ranking metric is computed from. That division is the
        point: PR-AUC, recall @ fixed FPR and precision@k then cannot move when the decision
        layer changes, by construction rather than by argument.

        The argument was tried first, and it did not hold. Platt scaling is monotone in exact
        arithmetic, so reporting the calibrated probability *looked* safe — but `1/(1+exp(-z))`
        rounds to exactly 1.0 in float64 past z ≈ 37, and on PaySim's committed test window the
        fitted map collapsed 129 distinct scores in the top 200 rows to one value across 480 of
        them. precision@100 on the stock-params control moved 0.14 → 0.06 as a result: a
        detection metric changed because of a decision-layer knob. `Z_LIMIT` fixes the
        saturation, and this keeps the two apart so that the next such bug cannot reach a
        reported number at all.

        The band edges in `to_dict()` are therefore in *calibrated probability* units and this
        field is not. `band_units` says so, and the reason code prints the probability it acted
        on next to the amount it acted on.
        """
        self._warn_if_uncalibrated()
        p = self.probability(score)
        action = self.act_on_probability(p, amount)
        out = list(reasons or [])
        if action is not Action.ALLOW:
            out.append(self.rationale(p, amount, action))
        return DetectorScore(
            txn_id=txn_id, score=min(max(float(score), 0.0), 1.0), action=action, reasons=out
        )

    # ── calibrating ─────────────────────────────────────────────────────────────
    def reset_calibration(self) -> DecisionPolicy:
        """Back to the identity map, keeping the configured method.

        Called before the scores a calibrator is about to be fitted from are produced. Without
        it, re-calibrating an already-calibrated policy fits a second map on top of the first
        and every probability afterwards is squashed twice — a bug that leaves the ranking
        untouched, so no metric moves and nothing looks wrong.
        """
        self.calibrator = ScoreCalibrator(method=self.calibrator.method)
        # A reset is the first half of "score the validation rows, then fit the map on those
        # scores", so the pass that follows is *meant* to run uncalibrated. Warning there would
        # cry wolf on the one code path doing the right thing, and a warning people learn to
        # ignore is worth less than no warning at all.
        self._warned_uncalibrated = True
        return self

    def fit_calibrator(self, scores, labels) -> DecisionPolicy:
        """Fit the score → probability map on held-out validation rows. Never on test."""
        self.calibrator.fit(scores, labels)
        # If the fit was refused — too few positives to be anything but noise — then from here
        # on this policy really is pricing a ranking score, and that is worth saying out loud.
        self._warned_uncalibrated = self.calibrator.fitted
        return self

    def calibrate_to_fpr(self, scores, labels, target_fpr: float = 0.01) -> DecisionPolicy:
        """Set `decline_at` to the score that hits a target FPR on a held-out set.

        The other bands keep their spacing relative to it. This is the *threshold*-mode operating
        point and it is kept for configs that ask for it; in cost mode it is contradictory —
        the bands are the cost model's answer, not an FPR target — so it warns and
        `assert_one_operating_point` refuses the combination in config outright.

        Calibrate on validation data only. Calibrating on the test set is how an honest pipeline
        quietly becomes a dishonest one.
        """
        import numpy as np

        if self.mode == "cost":
            log.warning(
                "calibrate_to_fpr on a cost-mode policy: the bands it sets are reporting-only, "
                "because the action comes from the cost model. Set decision.mode=threshold if "
                "an FPR target is what you want."
            )
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        negatives = np.sort(scores[labels == 0])[::-1]
        if len(negatives) == 0:
            return self
        idx = min(len(negatives) - 1, max(0, int(round(target_fpr * len(negatives))) - 1))
        decline = float(negatives[idx])
        spacing = {
            Action.REVIEW: self.review_at / self.decline_at if self.decline_at else 0.8,
            Action.HOLD: self.hold_at / self.decline_at if self.decline_at else 0.6,
            Action.STEP_UP: self.step_up_at / self.decline_at if self.decline_at else 0.3,
        }
        self.decline_at = decline
        for action, ratio in spacing.items():
            setattr(self, BAND_ATTR[action], decline * min(ratio, 1.0))
        self.bands_source = f"calibrated to {target_fpr:.4g} FPR on validation"
        return self

    def to_dict(self) -> dict[str, Any]:
        """What the model card and every run artefact carry about the decision layer."""
        return {
            "mode": self.mode,
            # Rounded for the artefact only; the policy compares against the exact values. A
            # band printed as 0.16000000000000023 is a band nobody reads.
            **{attr: round(float(getattr(self, attr)), 8) for attr in BAND_ATTR.values()},
            "bands_source": self.bands_source,
            # The bands are probabilities; `DetectorScore.score` is the detector's own score.
            # Said out loud because they are two scales and an artefact that showed both without
            # naming them would invite exactly one wrong comparison.
            "band_units": "calibrated probability (see `calibration` below)",
            "reference_amount": round(self.reference_amount, 6),
            "unreachable_bands": [a.value for a in self.costs.dominated_at(self.reference_amount)],
            "costs": self.costs.to_dict(),
            "calibration": self.calibrator.to_dict(),
        }


#: The band placement this ticket replaced: `decline_at` pinned to a target FPR, and the other
#: three at these fractions of it. The fractions were calibrated to nothing — that is the whole
#: complaint — and they are kept here, exactly, so that `make decisions` can measure the new
#: policy against the real old one rather than against a flattering approximation of it.
LEGACY_RATIOS: dict[Action, float] = {
    Action.REVIEW: 0.8,
    Action.HOLD: 0.6,
    Action.STEP_UP: 0.3,
}


def ratio_band_policy(
    scores, labels, target_fpr: float, costs: CostModel | None = None
) -> DecisionPolicy:
    """The pre-ticket-09 decision policy, reproduced exactly. The control, not the product.

    `DecisionPolicy.calibrate_to_fpr` no longer does this: it preserves whatever spacing the
    policy already had, which for a cost-derived policy means the cost model's spacing. That is
    better behaviour and it is *not* what shipped before, so a control built from it would be
    comparing against a policy that never ran. This function is the historical one, kept
    deliberately unpleasant.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    negatives = np.sort(scores[labels == 0])[::-1]
    policy = DecisionPolicy(
        mode="threshold",
        costs=costs or CostModel(),
        step_up_at=0.20,
        hold_at=0.50,
        review_at=0.70,
        decline_at=0.90,
    )
    if len(negatives) == 0:
        return policy
    idx = min(len(negatives) - 1, max(0, int(round(target_fpr * len(negatives))) - 1))
    policy.decline_at = float(negatives[idx])
    for action, ratio in LEGACY_RATIOS.items():
        setattr(policy, BAND_ATTR[action], policy.decline_at * ratio)
    policy.bands_source = (
        f"pre-ticket-09: decline_at at {target_fpr:.4g} FPR on validation, the rest at "
        f"{', '.join(f'{r:g}' for r in LEGACY_RATIOS.values())} of it"
    )
    return policy


def median_amount(txns) -> float:
    """The amount scale a cost model is denominated in — the median payment, not the mean.

    Median because these distributions are lognormal with a long tail: PaySim's mean payment is
    179,862 against a median of 74,872, so pricing an analyst's time against the mean would set
    it 2.4x too high on the strength of the largest transfers in the file.
    """
    amounts = sorted(float(t.amount) for t in txns)
    if not amounts:
        return 1.0
    mid = len(amounts) // 2
    return amounts[mid] if len(amounts) % 2 else 0.5 * (amounts[mid - 1] + amounts[mid])


def cost_model_for(cfg: dict[str, Any], rows=None) -> CostModel:
    """The run's cost model, denominated in the amount scale of the traffic it will decide on.

    One function rather than one per script, because a cost model built two slightly different
    ways in `run_experiment.py` and `build_baseline.py` is two operating points again.
    """
    unit = median_amount(rows) if rows else None
    model = CostModel.from_config(cfg, unit_amount=unit)
    log.info(
        "cost model %r denominated in unit_amount=%.4g (%s): step_up %.4g, hold %.4g, review %.4g",
        model.source,
        model.unit_amount,
        "median payment of the anchor" if unit else "config / absolute",
        model.step_up_cost,
        model.hold_cost,
        model.review_cost,
    )
    return model


def policy_from_config(decision_cfg: dict[str, Any], costs: CostModel) -> DecisionPolicy:
    """Build the shipped decision policy from `defend.supervised.decision` plus the cost model.

    Bands are never read from `decision_cfg` — there are none in it. That is the point of the
    ticket, and a config that grew one back would be silently ignored here rather than silently
    honoured, which is the safer direction to fail.
    """
    return DecisionPolicy(
        mode=str(decision_cfg.get("mode", "cost")),
        costs=costs,
        calibrator=ScoreCalibrator(method=str(decision_cfg.get("calibration", "sigmoid"))),
        reference_amount=costs.unit_amount,
    )


def assert_one_operating_point(
    calibrate_to_fpr: float | None, fixed_fpr: float, mode: str = "threshold"
) -> None:
    """The action bands and the reported metric are one decision, so they take one number.

    In **threshold** mode, `defend.supervised.decision.calibrate_to_fpr` places the bands and
    `eval.fixed_fpr` is where recall is read off. Two different values means the table's
    `recall@1%FPR` column and its `evasion_rate` column describe two different systems — which is
    the sort of discrepancy nobody notices until they are asked to explain it on a slide.

    In **cost** mode the bands are not an FPR target at all: the cost model places them and the
    realised false-positive rate is an *output*, reported as `false_decline_rate` beside
    `recall@1%FPR` in the same table. Naming a `calibrate_to_fpr` as well would be two operating
    points wearing one config, so it is refused rather than quietly ignored.
    """
    if str(mode) == "cost":
        if calibrate_to_fpr is not None:
            raise ValueError(
                f"two operating points: decision.mode=cost places the bands by expected cost, but "
                f"calibrate_to_fpr={calibrate_to_fpr} also asks for them to be placed at an FPR "
                "target. Set calibrate_to_fpr to null, or set mode to threshold. The realised "
                "FPR of the cost policy is reported as `false_decline_rate`."
            )
        return
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


def action_mix(scores: list[DetectorScore]) -> dict[str, float]:
    """Share of decisions landing on each rung. The thing a changed cost parameter has to move."""
    if not scores:
        return {a.value: 0.0 for a in SEVERITY}
    counts = dict.fromkeys((a.value for a in SEVERITY), 0)
    for s in scores:
        counts[s.action.value] += 1
    return {name: n / len(scores) for name, n in counts.items()}


__all__ = [
    "BAND_ATTR",
    "DEFAULT_COSTS",
    "SEVERITY",
    "UNREACHABLE",
    "CostModel",
    "DecisionPolicy",
    "DominatedAction",
    "action_mix",
    "LEGACY_RATIOS",
    "assert_one_operating_point",
    "cost_model_for",
    "ratio_band_policy",
    "median_amount",
    "policy_from_config",
    "total_cost",
]
