"""The workhorse. Gradient-boosted trees over the causal feature table.

This is the detector every headline number is measured against. The exotic layers (sequence,
GNN) have to beat *this*, on the same split, before they earn a place in the table.

Falls back to sklearn's HistGradientBoosting when LightGBM is not installed, so the loop runs
anywhere; the fallback is logged, never silent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from afl.contract.metrics import DetectorScore
from afl.contract.schema import AttackBatch, Transaction
from afl.defend.decision import DecisionPolicy
from afl.defend.features import FeatureBuilder

log = logging.getLogger(__name__)

_FALLBACK_WARNED = False  # the backend is an environment fact; say it once, not once per fit


class LGBMDetector:
    """■ B's half of the seam: `score(batch)` and `retrain(batch, evasions)`."""

    def __init__(
        self,
        policy: DecisionPolicy | None = None,
        features: FeatureBuilder | None = None,
        params: dict[str, Any] | None = None,
        seed: int = 1337,
        replay_weight: float = 3.0,
        explain: bool = False,
    ) -> None:
        self.policy = policy or DecisionPolicy()
        self.features = features or FeatureBuilder(stateful=True)
        self.seed = seed
        self.replay_weight = replay_weight  # evasions are the expensive examples; weight them up
        self.explain = explain
        self.params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "n_estimators": 300,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "verbosity": -1,
            **(params or {}),
        }
        self.model = None
        self.backend = "untrained"
        self.feature_names: list[str] = []
        self._replay: list[Transaction] = []  # evasion memory across rounds
        self._corpus: list[Transaction] = []  # everything trained on so far
        self._explainer = None

    # ── training ────────────────────────────────────────────────────────────────
    def _new_model(self):
        """The configured backend, or the sklearn fallback, saying loudly which one it got."""
        global _FALLBACK_WARNED
        try:
            import lightgbm as lgb

            self.backend = "lightgbm"
            return lgb.LGBMClassifier(**self.params)
        except (ImportError, OSError) as e:
            # ImportError: not installed. OSError: installed but libomp is missing, which is the
            # usual macOS case — the wheel imports and then fails to dlopen its own shared library.
            from sklearn.ensemble import HistGradientBoostingClassifier

            if not _FALLBACK_WARNED:
                log.warning(
                    "LIGHTGBM UNAVAILABLE (%s) - running on the sklearn "
                    "HistGradientBoosting fallback, which is NOT the headline detector. "
                    "On macOS: brew install libomp",
                    type(e).__name__,
                )
                _FALLBACK_WARNED = True
            self.backend = "sklearn-hgb"
            return HistGradientBoostingClassifier(
                learning_rate=self.params["learning_rate"],
                max_iter=self.params["n_estimators"],
                random_state=self.seed,
            )

    def fit(self, txns: list[Transaction], sample_weight: np.ndarray | None = None) -> LGBMDetector:
        """Fit from scratch on `txns`, which become the corpus later rounds accumulate onto."""
        self._corpus = list(txns)
        return self._fit_model(txns, sample_weight)

    def _fit_model(self, txns: list[Transaction], sample_weight: np.ndarray | None) -> LGBMDetector:
        # a full refit rebuilds the feature history from scratch, so the entity state the model
        # scores against is a pure function of its corpus rather than of the order of past calls
        self.features.reset()
        X = self.features.transform(txns, update=True)
        y = FeatureBuilder.labels(txns)
        if len(set(y.tolist())) < 2:
            log.warning("single-class training set (%d rows) — model not fitted", len(y))
            return self
        self.feature_names = list(X.columns)
        self.model = self._new_model()
        try:
            self.model.fit(X.to_numpy(), y, sample_weight=sample_weight)
        except TypeError:  # backend without sample_weight support
            self.model.fit(X.to_numpy(), y)
        self._explainer = None
        return self

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        """Add the round to the corpus and refit, with the rows that got through weighted up.

        The round is *added*, not substituted. Fitting on the latest batch alone would make the
        detector forget its whole history every round, and the convergence curve would then be
        measuring recency rather than learning.

        Full refit rather than incremental: with a few thousand rows it costs seconds, and it
        keeps every round's model reproducible from (seed, data) alone.
        """
        self._replay.extend(evasions)
        known = {t.txn_id for t in self._corpus}
        self._corpus.extend(t for t in batch.transactions if t.txn_id not in known)

        heavy = {t.txn_id for t in self._replay}
        weights = np.array(
            [self.replay_weight if t.txn_id in heavy else 1.0 for t in self._corpus], dtype=float
        )
        self._fit_model(self._corpus, weights)

    # ── scoring ─────────────────────────────────────────────────────────────────
    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(txns), dtype=float)
        X = self.features.transform(txns, update=False)
        X = X.reindex(columns=self.feature_names, fill_value=0.0)
        return self.model.predict_proba(X.to_numpy())[:, 1]

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        txns = batch.transactions
        probs = self.predict_proba(txns)
        reasons: list[list[str]] = [[]] * len(txns)
        if self.explain and self.model is not None:
            from afl.defend.explain import reason_codes

            reasons = reason_codes(self, txns)
        return [
            self.policy.decide(t.txn_id, float(p), amount=t.amount, reasons=list(rs))
            for t, p, rs in zip(txns, probs, reasons, strict=False)
        ]

    # ── introspection ───────────────────────────────────────────────────────────
    def feature_importance(self) -> dict[str, float]:
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(
            sorted(
                zip(
                    self.feature_names,
                    (float(v) for v in self.model.feature_importances_),
                    strict=False,
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
