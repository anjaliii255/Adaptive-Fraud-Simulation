"""The zero-day layer.

The supervised model can only catch what it has labels for — which by construction excludes the
attack family the loop is about to invent. An unsupervised score trained on *legit traffic only*
degrades more gracefully against an unseen vector, so it is the honest floor under the ensemble
in leave-one-attack-out runs.
"""

from __future__ import annotations

import numpy as np

from afl.contract.metrics import Action, DetectorScore
from afl.contract.schema import AttackBatch, Transaction
from afl.defend import explain
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

    def scaled(self, txns: list[Transaction]) -> np.ndarray:
        """The features standardised against what *legit* traffic looked like when this was fit.

        Kept public and separate from `predict_proba` because it is also the explanation: a row
        is an outlier in the coordinates the scaler defines, so the columns with the largest
        magnitude here are literally the reason the forest isolated it.
        """
        X = self.features.transform(txns, update=False).reindex(
            columns=self.feature_names, fill_value=0.0
        )
        return self.scaler.transform(X.to_numpy())

    def predict_proba(
        self, txns: list[Transaction], scaled: np.ndarray | None = None
    ) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(txns), dtype=float)
        Xs = self.scaled(txns) if scaled is None else scaled
        if self.kind == "iforest":
            raw = -self.model.score_samples(Xs)  # higher = more anomalous
        else:
            raw = np.mean((self.model.predict(Xs) - Xs) ** 2, axis=1)  # reconstruction error
        lo, hi = float(np.min(raw)), float(np.max(raw))
        return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

    def reason_codes(self, txns: list[Transaction], top_k: int = 2, scaled=None) -> list[list[str]]:
        """Why *this* row looked abnormal: the columns furthest from legit traffic's normal.

        A local explanation, not a global one, so it carries no `GLOBAL_PREFIX` — the z-score
        quoted is this transaction's. There is no SHAP path here: an isolation forest has no
        `TreeExplainer` contract, and standardised distance is the honest thing an outlier score
        can say about itself.
        """
        if self.model is None or not txns:
            return [[] for _ in txns]
        Xs = self.scaled(txns) if scaled is None else scaled
        out: list[list[str]] = []
        for row in np.asarray(Xs, dtype=float):
            order = np.argsort(-np.abs(row))[:top_k]
            out.append(
                [
                    f"{'↑' if row[j] > 0 else '↓'} {explain.readable(self.feature_names[j])} "
                    f"({row[j]:+.1f}σ vs legit traffic)"
                    for j in order
                ]
            )
        return out

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        txns = batch.transactions
        if not txns:
            return []
        scaled = self.scaled(txns) if self.model is not None else None
        probs = self.predict_proba(txns, scaled=scaled)
        reasons = self.reason_codes(txns, top_k=explain.MIN_REASONS, scaled=scaled)
        return [
            self.policy.decide(
                t.txn_id, float(p), amount=t.amount, reasons=[f"anomaly:{self.kind}", *rs]
            )
            for t, p, rs in zip(txns, probs, reasons, strict=False)
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
        """Blend, decide, and explain the flagged rows from *both* halves.

        `reasons=["ensemble"]` was the whole explanation before ticket 09, which meant the
        detector every anchored run actually uses — `defend.unsupervised.ensemble.enabled` is
        true — produced one uninformative string per flagged transaction while the reason-code
        machinery sat behind a supervised detector nothing scored through. An analyst holding a
        blended score needs to know which half raised it, so the split is quoted too.
        """
        txns = batch.transactions
        if not txns:
            return []
        # Duck-typed, and deliberately: the ensemble takes *a* supervised detector, and only the
        # gradient-boosted one can hand back the design matrix it just built. Anything else is
        # scored the plain way and explained from its own features, at the cost of one extra
        # pass. A hard dependency here would quietly make `supervised` mean `LGBMDetector`.
        reuses_features = hasattr(self.supervised, "design_matrix") and (
            getattr(self.supervised, "model", None) is not None
        )
        values = self.supervised.design_matrix(txns) if reuses_features else None
        scaled = self.unsupervised.scaled(txns) if self.unsupervised.model is not None else None
        p_sup = (
            self.supervised.predict_proba(txns, values=values)
            if reuses_features
            else self.supervised.predict_proba(txns)
        )
        p_uns = self.unsupervised.predict_proba(txns, scaled=scaled)
        blended = self.weight * p_sup + (1 - self.weight) * p_uns

        actions = [
            self.policy.act(float(p), amount=t.amount) for t, p in zip(txns, blended, strict=False)
        ]
        flagged = [i for i, a in enumerate(actions) if a is not Action.ALLOW]
        reasons: list[list[str]] = [[] for _ in txns]
        if flagged:
            rows = [txns[i] for i in flagged]
            sup_codes = (
                explain.reason_codes(self.supervised, rows, values=values[flagged])
                if values is not None
                else [[] for _ in rows]
            )

            uns_codes = self.unsupervised.reason_codes(
                rows, scaled=scaled[flagged] if scaled is not None else None
            )
            for n, i in enumerate(flagged):
                reasons[i] = [
                    *sup_codes[n],
                    *uns_codes[n],
                    f"blend: supervised {p_sup[i]:.3f} x {self.weight:.2f} + "
                    f"outlier {p_uns[i]:.3f} x {1 - self.weight:.2f}",
                ]

        return [
            self.policy.decide(t.txn_id, float(p), amount=t.amount, reasons=rs)
            for t, p, rs in zip(txns, blended, reasons, strict=False)
        ]

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        self.supervised.retrain(batch, evasions)
        self.unsupervised.retrain(batch, evasions)
