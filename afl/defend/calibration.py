"""Score → probability, so that a cost threshold means what it says.

A cost model compares `p x amount` against the flat cost of an analyst looking at the case. That
comparison is arithmetic on a probability, and it is only meaningful if `p` is one. A
gradient-boosted tree's `predict_proba` is not: at a 0.13% base rate, with `class_weight` a
searched knob and an evasion-weighted replay buffer in the mix, its output is a *ranking* score
whose absolute value is off by a factor nobody has measured. Feeding it to a cost model would be
the quiet kind of wrong this project is arranged to avoid — the arithmetic right, the inputs
meaningless, and a number at the end that looks derived.

So the score is mapped onto a probability on the same validation tail the rest of the operating
point is set on, and the reliability of that map is measured and committed rather than assumed.

**Platt scaling (`sigmoid`) is the default, and it is chosen for the thin tail.** The tuning
validation window carries 46 fraud rows on PaySim and 196 on AMLSim (`artifacts/detector/`), and
a two-parameter fit is what survives that; isotonic buys resolution this much data cannot pay
for. Platt is also *strictly monotone*, which matters more than it sounds: PR-AUC, recall @ fixed
FPR and precision@k are rank statistics, so calibrating cannot move them. Only the units move.
`tests/test_decision.py` asserts exactly that, because it is the property that lets this land
without disturbing a single reported metric.

Fitted on validation, never on test — an operating point chosen on the window it is reported from
is not an operating point, it is a result.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)

#: Below this many positives the map is noise wearing a curve, and the calibrator stays an
#: identity rather than inventing precision. The count that triggered the number: PaySim's
#: tuning validation tail holds 46.
MIN_POSITIVES = 15

#: Equal-count bins for the reliability diagram and the calibration error.
N_BINS = 10

METHODS = ("sigmoid", "isotonic", "none")

#: Where the logistic is clamped before it stops being a function. `1/(1+exp(-z))` rounds to
#: exactly 1.0 in float64 at about z = 37, and to exactly 0.0 at about z = -746, so past those
#: points the map returns the same number for inputs that were far apart. Measured, not feared:
#: fitted on PaySim's validation tail, the unclamped map collapsed the 129 distinct scores in the
#: test window's top 200 to a single value of 1.0 across 480 rows. Clamping keeps the last
#: distinguishable value instead, which is why this module can promise a probability without
#: promising a probability that has lost the ordering it came from.
Z_LIMIT = 36.0


def _logit(p: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def brier(probabilities: ArrayLike, labels: ArrayLike) -> float:
    """Mean squared error against the outcome. Lower is better; 0.0 is clairvoyance."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else 0.0


def reliability_bins(probabilities: ArrayLike, labels: ArrayLike, n_bins: int = N_BINS) -> list:
    """Equal-count bins of (predicted, observed). The material for a reliability diagram.

    Equal-count rather than equal-width: at this base rate almost every row lands in the first
    equal-width bin, and nine empty buckets are not a diagnostic.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return []
    order = np.argsort(p, kind="stable")
    out = []
    for chunk in np.array_split(order, min(n_bins, p.size)):
        if chunk.size == 0:
            continue
        out.append(
            {
                "n": int(chunk.size),
                "predicted": round(float(p[chunk].mean()), 8),
                "observed": round(float(y[chunk].mean()), 8),
            }
        )
    return out


def expected_calibration_error(probabilities: ArrayLike, labels: ArrayLike) -> float:
    """Weighted mean gap between what was predicted and what happened, over equal-count bins."""
    bins = reliability_bins(probabilities, labels)
    if not bins:
        return 0.0
    total = sum(b["n"] for b in bins)
    return float(sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in bins) / total)


@dataclass
class ScoreCalibrator:
    """A monotone map from a detector's score to P(fraud), with its own reliability attached.

    Unfitted, it is the identity — so a detector with no calibration behaves exactly as it did
    before this module existed, which is what keeps the fallback path honest rather than absent.
    """

    method: str = "sigmoid"  # sigmoid | isotonic | none
    n_fit: int = 0
    n_positives: int = 0
    base_rate: float = 0.0
    #: Brier and ECE before and after the map, on the rows it was fitted from. Reported, not
    #: promised: a calibrator that made things worse should be visible in the artefact.
    reliability: dict[str, Any] = field(default_factory=dict)
    note: str = "not fitted - scores pass through unchanged"
    _model: Any = None
    #: The fitted map, unpacked into plain numbers. `decide` calls `probability` once per
    #: transaction, and a per-row trip through a scikit-learn estimator costs tens of
    #: microseconds — which on a 150k-row test window is a minute of nothing.
    _sigmoid: tuple[float, float] | None = None
    _knots: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(
                f"unknown calibration method {self.method!r}; expected one of {METHODS}"
            )

    @property
    def fitted(self) -> bool:
        return self._model is not None

    # ── fitting ─────────────────────────────────────────────────────────────────
    def fit(self, scores: ArrayLike, labels: ArrayLike) -> ScoreCalibrator:
        """Fit on a *validation* set. Refuses to invent precision out of too few positives."""
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int)
        self.n_fit = int(s.size)
        self.n_positives = int(y.sum())
        self.base_rate = float(y.mean()) if y.size else 0.0
        self._model = self._sigmoid = self._knots = None

        if self.method == "none":
            self.note = "calibration disabled in config - scores pass through unchanged"
        elif self.n_positives < MIN_POSITIVES or self.n_positives == y.size:
            self.note = (
                f"{self.n_positives} positive(s) in the calibration set, below the floor of "
                f"{MIN_POSITIVES} - left as the identity rather than fitting a curve to noise"
            )
            log.warning("calibration skipped: %s", self.note)
        elif self.method == "sigmoid":
            self._sigmoid = self._fit_sigmoid(s, y)
            self._model = self._sigmoid
            self.note = f"{self.method}, fitted on {self.n_fit:,} validation rows"
        else:
            self._model = self._fit_isotonic(s, y)
            self._knots = (
                np.asarray(self._model.X_thresholds_, dtype=float),
                np.asarray(self._model.y_thresholds_, dtype=float),
            )
            self.note = f"{self.method}, fitted on {self.n_fit:,} validation rows"

        # Reported whether or not a map was fitted. Unfitted the two blocks are equal, because
        # the identity is a calibration too — and a reader wanting to know how badly the raw
        # score was calibrated should not have to infer it from a missing key.
        after = self(s)
        self.reliability = {
            "before": {
                "brier": round(brier(s, y), 8),
                "ece": round(expected_calibration_error(s, y), 8),
            },
            "after": {
                "brier": round(brier(after, y), 8),
                "ece": round(expected_calibration_error(after, y), 8),
            },
            "bins": reliability_bins(after, y),
        }
        if self.fitted and self.reliability["after"]["brier"] > self.reliability["before"]["brier"]:
            log.warning(
                "calibration made the Brier score worse (%g -> %g) - it is recorded, not hidden",
                self.reliability["before"]["brier"],
                self.reliability["after"]["brier"],
            )
        return self

    @staticmethod
    def _fit_sigmoid(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Platt scaling on the logit of the score: `p = sigmoid(a * logit(s) + b)`.

        Two parameters, fitted directly rather than through a logistic regression, for two
        reasons. Platt's **target smoothing** is the part people drop, and a general-purpose
        classifier cannot take the continuous targets it produces: fitting to raw 0/1 labels
        drives the slope to infinity the moment the validation tail happens to be separable —
        which at 46 positives is not a remote possibility — and an infinite slope is a step
        function, a threshold pretending to be a probability. The smoothing pulls both ends in
        by one pseudo-count, which makes the optimum finite without a second regulariser
        shrinking the map toward saying nothing.

        The second reason is numerical. Routed through `LogisticRegression` with a large `C`,
        the same fit overflows inside the solver's line search on exactly this data. Written out,
        the objective is `softplus(-z) + (1 - t) * z` with gradient `sigmoid(z) - t` — both
        evaluated stably — and the starting point `(1, 0)` is the identity, so the fit begins
        from "the score is already a probability" and moves only as far as the data pushes it.
        """
        from scipy.optimize import minimize
        from scipy.special import expit

        n_pos, n_neg = float(y.sum()), float(y.size - y.sum())
        target = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        x = _logit(s)

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = float(theta[0]), float(theta[1])
            z = a * x + b
            loss = float(np.sum(np.logaddexp(0.0, -z) + (1.0 - target) * z))
            residual = expit(z) - target
            # np.sum(a * b) rather than `a @ b`: on this platform numpy's matmul raises a
            # spurious "divide by zero" FP warning from BLAS flag leakage on finite inputs,
            # and a warning that is not about the data is worse than no warning at all.
            return loss, np.array([float(np.sum(residual * x)), float(residual.sum())])

        fit = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        if not fit.success:
            log.warning("Platt scaling did not converge (%s) — using the last iterate", fit.message)
        return float(fit.x[0]), float(fit.x[1])

    @staticmethod
    def _fit_isotonic(s: np.ndarray, y: np.ndarray):
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)

    # ── applying ────────────────────────────────────────────────────────────────
    def __call__(self, scores: ArrayLike) -> np.ndarray:
        s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        if self._sigmoid is not None:
            a, b = self._sigmoid
            z = np.clip(a * _logit(s) + b, -Z_LIMIT, Z_LIMIT)
            return 1.0 / (1.0 + np.exp(-z))
        if self._knots is not None:
            x, y = self._knots
            return np.clip(np.interp(s, x, y), 0.0, 1.0)
        return s

    def probability(self, score: float) -> float:
        """One score. The hot path — `decide` calls this once per transaction."""
        s = min(max(float(score), 1e-6), 1.0 - 1e-6)
        if self._sigmoid is not None:
            a, b = self._sigmoid
            z = min(max(a * math.log(s / (1.0 - s)) + b, -Z_LIMIT), Z_LIMIT)
            return 1.0 / (1.0 + math.exp(-z))
        if self._knots is not None:
            return float(np.interp(s, self._knots[0], self._knots[1]))
        return min(max(float(score), 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "fitted": self.fitted,
            "n_fit": self.n_fit,
            "n_positives": self.n_positives,
            "base_rate": round(self.base_rate, 8),
            "note": self.note,
            "reliability": self.reliability,
        }


__all__ = [
    "MIN_POSITIVES",
    "ScoreCalibrator",
    "brier",
    "expected_calibration_error",
    "reliability_bins",
]
