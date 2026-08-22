"""The zero-day layer.

The supervised model can only catch what it has labels for — which by construction excludes the
attack family the loop is about to invent. An unsupervised score trained on *legit traffic only*
degrades more gracefully against an unseen vector, so it is the honest floor under the ensemble
in leave-one-attack-out runs.
"""

from __future__ import annotations

import numpy as np

from afl.contract.metrics import DetectorScore
from afl.contract.schema import AttackBatch, Transaction
from afl.defend.decision import DecisionPolicy
from afl.defend.features import FeatureBuilder


class AnomalyDetector:
    """Isolation Forest over the same causal features. `kind="ae"` swaps in an autoencoder."""

    def __init__(
        self,
        kind: str = "iforest",  # "iforest" | "ae"
        contamination: float = 0.01,
        policy: DecisionPolicy | None = None,
        features: FeatureBuilder | None = None,
        seed: int = 1337,
    ) -> None:
        self.kind = kind
        self.contamination = contamination
        self.policy = policy or DecisionPolicy()
        self.features = features or FeatureBuilder(stateful=True)
        self.seed = seed
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self._corpus: list[Transaction] = []

    def fit(self, txns: list[Transaction], legit_only: bool = True) -> AnomalyDetector:
        """Fit on legit rows only — training on fraud defeats the point of an outlier score."""
        rows = [t for t in txns if not t.is_fraud] if legit_only else list(txns)
        self._corpus = list(rows)
        if len(rows) < 10:
            return self
        self.features.reset()
        X = self.features.transform(rows, update=True)
        self.feature_names = list(X.columns)

        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X.to_numpy())
        Xs = self.scaler.transform(X.to_numpy())

        if self.kind == "iforest":
            from sklearn.ensemble import IsolationForest

            self.model = IsolationForest(
                n_estimators=200, contamination=self.contamination, random_state=self.seed
            ).fit(Xs)
        elif self.kind == "ae":
            # MLP autoencoder via sklearn so the base install stays torch-free; the torch version
            # lives behind the `deep` extra and only earns its place if it beats this.
            from sklearn.neural_network import MLPRegressor

            self.model = MLPRegressor(
                hidden_layer_sizes=(
                    max(2, Xs.shape[1] // 2),
                    max(2, Xs.shape[1] // 4),
                    max(2, Xs.shape[1] // 2),
                ),
                max_iter=300,
                random_state=self.seed,
            ).fit(Xs, Xs)
        else:
            raise ValueError(f"unknown anomaly kind {self.kind!r}")
        return self

    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(txns), dtype=float)
        X = self.features.transform(txns, update=False).reindex(
            columns=self.feature_names, fill_value=0.0
        )
        Xs = self.scaler.transform(X.to_numpy())
        if self.kind == "iforest":
            raw = -self.model.score_samples(Xs)  # higher = more anomalous
        else:
            raw = np.mean((self.model.predict(Xs) - Xs) ** 2, axis=1)  # reconstruction error
        lo, hi = float(np.min(raw)), float(np.max(raw))
        return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        probs = self.predict_proba(batch.transactions)
        return [
            self.policy.decide(
                t.txn_id, float(p), amount=t.amount, reasons=[f"anomaly:{self.kind}"]
            )
            for t, p in zip(batch.transactions, probs, strict=False)
        ]

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        """Widen the notion of normal with this round's legit traffic; never narrow it to it."""
        known = {t.txn_id for t in self._corpus}
        rows = self._corpus + [t for t in batch.transactions if t.txn_id not in known]
        self.fit(rows, legit_only=True)


class EnsembleDetector:
    """Supervised + unsupervised, blended. The blend weight is a config knob, not a constant."""

    def __init__(
        self, supervised, unsupervised, weight: float = 0.7, policy: DecisionPolicy | None = None
    ) -> None:
        self.supervised = supervised
        self.unsupervised = unsupervised
        self.weight = weight
        self.policy = policy or DecisionPolicy()

    def fit(
        self, txns: list[Transaction], sample_weight: np.ndarray | None = None
    ) -> EnsembleDetector:
        self.supervised.fit(txns, sample_weight=sample_weight)
        self.unsupervised.fit(txns, legit_only=True)
        return self

    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        return self.weight * self.supervised.predict_proba(txns) + (
            1 - self.weight
        ) * self.unsupervised.predict_proba(txns)

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        p_sup = self.supervised.predict_proba(batch.transactions)
        p_uns = self.unsupervised.predict_proba(batch.transactions)
        blended = self.weight * p_sup + (1 - self.weight) * p_uns
        return [
            self.policy.decide(t.txn_id, float(p), amount=t.amount, reasons=["ensemble"])
            for t, p in zip(batch.transactions, blended, strict=False)
        ]

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        self.supervised.retrain(batch, evasions)
        self.unsupervised.retrain(batch, evasions)
