"""The zero-day layer.

The supervised model can only catch what it has labels for — which by construction excludes the
attack family the loop is about to invent. The bet behind this module is that an unsupervised
score trained on *legit traffic only* degrades more gracefully against an unseen vector, and so
is the honest floor under the ensemble in leave-one-attack-out runs.

**Ticket 10 measured the bet and it lost.** On the M3 fold the supervised model does not collapse
(PaySim: PR-AUC 0.524 on the held-out family against 0.152 on the anchor's own fraud) and this
layer sits far below it (0.033), below the amount floor on AMLSim. It stays because the *blend*
measurably needs it on PaySim — an interior optimum on the weight curve, w=0.5 at 0.551 against
0.524 for the supervised model alone — not because it holds up on its own. `docs/anomaly.md` has
the tables; `artifacts/anomaly/<anchor>.json` has the evidence.

Two properties this module owes the rest of the project, both of them enforced here rather than
promised in a docstring:

**No fraud row ever enters training.** `fit` filters, records what it filtered, and there is no
argument that turns the filter off — `contaminated_control` is a separately named function so
that the contaminated variant can be *measured* (ticket 10 does) without being reachable by
accident from the shipped path.

**A row's score does not depend on which other rows it was scored with.** The map from the raw
outlier score to [0, 1] is fixed at fit time against legit traffic. It used to be a min-max over
the batch, which meant the same transaction scored two different values in two different
batches, and made the ensemble blend a probability with a batch-relative rank statistic. See
`ScoreMap`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from afl.contract.metrics import Action, DetectorScore
from afl.contract.schema import AttackBatch, Transaction
from afl.defend import explain
from afl.defend.decision import DecisionPolicy
from afl.defend.features import FeatureBuilder

log = logging.getLogger(__name__)

#: Below this, the notion of "normal" is one account's afternoon. The detector stays unfitted and
#: says so rather than shipping a scaler fitted on ten rows.
MIN_FIT_ROWS = 10


@dataclass(frozen=True)
class ScoreMap:
    """Raw outlier score → [0, 1], decided at fit time and never by the batch being scored.

    The previous version min-maxed each call's raw scores. Three things were wrong with that, and
    only the first is obvious:

      * the same row scored differently depending on its company, so there was no operating
        point to speak of — `DetectorScore.score` was a statement about a batch;
      * one extreme row compressed everything else towards zero, so a single wild outlier could
        quietly disable the layer for the rest of the batch;
      * the ensemble blended it 0.3-to-0.7 against a *probability*, which is arithmetic on two
        different scales — the same complaint ticket 09 made about cost bands on a ranking score.

    Ranking metrics never noticed, because a within-batch min-max is monotone and PR-AUC only
    reads order. The actions did, and so did the blend.

    `kind="identity"` is what an isolation forest gets: sklearn's `-score_samples` is
    ``2 ** (-E[h(x)] / c(n))``, already in (0, 1) by construction, so there is nothing to rescale
    and rescaling was pure loss. `kind="saturating"` is what a reconstruction error gets — it is
    unbounded above, so it is divided by a fit-time reference (`scale`, the median legit error)
    through ``e / (e + scale)``: strictly monotone and batch-independent.

    Any map from an unbounded quantity into [0, 1) saturates somewhere in float64, and ticket 09
    paid for the habit of naming where rather than asserting it does not happen: this one returns
    exactly 1.0 once ``e / scale`` passes ``2 ** 53`` — about nine quadrillion times the median
    legit reconstruction error. That is not a number a fitted MLP produces on standardised
    features, and it is written down here so the next person does not have to rediscover it the
    way the logistic's `Z_LIMIT` was rediscovered.
    """

    kind: str = "identity"  # "identity" | "saturating"
    scale: float = 1.0

    def apply(self, raw: np.ndarray) -> np.ndarray:
        r = np.asarray(raw, dtype=float)
        if self.kind == "identity":
            return np.clip(r, 0.0, 1.0)
        if self.kind == "saturating":
            r = np.maximum(r, 0.0)
            # `inf / (inf + scale)` is NaN, and an untrained MLP overflowing its own matmul is
            # how that arrives — observed, not hypothetical. An infinite reconstruction error is
            # the most anomalous a row can be, so it maps to the top of the range rather than to
            # a value `DetectorScore` would reject with an error about the wrong thing.
            with np.errstate(invalid="ignore"):
                out = r / (r + self.scale)
            return np.clip(np.nan_to_num(out, nan=1.0, posinf=1.0), 0.0, 1.0)
        raise ValueError(f"unknown score map {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scale": round(float(self.scale), 8),
            "batch_dependent": False,
            "why": (
                "an isolation forest's -score_samples is 2**(-E[h]/c(n)), already in (0,1)"
                if self.kind == "identity"
                else "reconstruction error is unbounded, so it is divided by the median legit "
                "error measured at fit time"
            ),
        }


@dataclass
class AnomalyTraining:
    """What one fit actually saw — including, and especially, what it refused to see.

    `n_fraud_seen` is on the card so that "fits on legit rows only" is a number in the artefact
    rather than a claim in a comment. It is 0 on every shipped path; `contaminated_control` is
    the only thing that can make it anything else, and it is named so that an artefact carrying
    a non-zero here is unmistakable.
    """

    n_rows: int = 0
    n_fraud_seen: int = 0
    n_fraud_excluded: int = 0
    n_features: int = 0
    legit_only: bool = True
    fitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_fraud_seen": self.n_fraud_seen,
            "n_fraud_excluded": self.n_fraud_excluded,
            "n_features": self.n_features,
            "legit_only": self.legit_only,
            "fitted": self.fitted,
        }


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
        if kind not in ("iforest", "ae"):
            raise ValueError(f"unknown anomaly kind {kind!r}; expected 'iforest' or 'ae'")
        self.kind = kind
        self.contamination = contamination
        self.policy = policy or DecisionPolicy()
        self.features = features or FeatureBuilder(stateful=True)
        self.seed = seed
        self.model = None
        self.scaler = None
        self.score_map = ScoreMap()
        self.feature_names: list[str] = []
        self.training = AnomalyTraining()
        self._corpus: list[Transaction] = []

    # ── training ────────────────────────────────────────────────────────────────
    def fit(self, txns: list[Transaction]) -> AnomalyDetector:
        """Fit on the legit rows of `txns`. There is no argument that keeps the fraud ones.

        Training an outlier score on fraud teaches it that fraud is normal, which is the one
        thing this layer exists not to learn. The filter is unconditional for that reason; the
        control that does not filter has its own name, `contaminated_control`, and ticket 10
        measures the difference rather than assuming it.
        """
        return self._fit(txns, legit_only=True)

    def _fit(self, txns: list[Transaction], legit_only: bool) -> AnomalyDetector:
        n_fraud = sum(1 for t in txns if t.is_fraud)
        rows = [t for t in txns if not t.is_fraud] if legit_only else list(txns)
        self._corpus = list(rows)
        self.training = AnomalyTraining(
            n_rows=len(rows),
            n_fraud_seen=0 if legit_only else n_fraud,
            n_fraud_excluded=n_fraud if legit_only else 0,
            legit_only=legit_only,
        )
        if len(rows) < MIN_FIT_ROWS:
            log.warning(
                "%d rows is not a notion of normal — the anomaly layer stays unfitted and every "
                "row will score 0.0, which reads in a metric exactly like catching nothing",
                len(rows),
            )
            return self
        self.features.reset()
        X = self.features.transform(rows, update=True)
        self.feature_names = list(X.columns)
        self.training.n_features = X.shape[1]

        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X.to_numpy())
        Xs = self.scaler.transform(X.to_numpy())

        if self.kind == "iforest":
            from sklearn.ensemble import IsolationForest

            self.model = IsolationForest(
                n_estimators=200, contamination=self.contamination, random_state=self.seed
            ).fit(Xs)
            self.score_map = ScoreMap(kind="identity")
        else:
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
            # the reference the map is quoted against: what reconstruction error legit traffic
            # produced on the rows this was fitted on. Fixed here, never recomputed at score time.
            legit_error = self.raw_scores(scaled=Xs)
            median = float(np.median(legit_error))
            self.score_map = ScoreMap(kind="saturating", scale=median if median > 0 else 1.0)
        self.training.fitted = True
        return self

    @property
    def training_rows(self) -> list[Transaction]:
        """Every row this detector has fitted on — the legit ones, after the fraud filter.

        What the leave-one-attack-out guard audits. `fit` drops the fraud rows before it gets
        here, so a carved-out family reaching this list would mean the carve-out failed on the
        *legit* side, which is the case nobody checks for.
        """
        return list(self._corpus)

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        """Widen the notion of normal with this round's legit traffic; never narrow it to it."""
        known = {t.txn_id for t in self._corpus}
        rows = self._corpus + [t for t in batch.transactions if t.txn_id not in known]
        self.fit(rows)

    # ── scoring ─────────────────────────────────────────────────────────────────
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

    def raw_scores(
        self, txns: list[Transaction] | None = None, scaled: np.ndarray | None = None
    ) -> np.ndarray:
        """The model's own units: isolation depth for a forest, reconstruction error for an AE.

        Higher is more anomalous in both cases. Public because `score_map.scale` is measured from
        it at fit time, and because an artefact that quotes a mapped score should be able to
        quote what it was mapped from.
        """
        Xs = self.scaled(txns or []) if scaled is None else np.asarray(scaled, dtype=float)
        if self.kind == "iforest":
            return -self.model.score_samples(Xs)  # 2 ** (-E[h(x)] / c(n)), higher = more isolated
        return np.mean((self.model.predict(Xs) - Xs) ** 2, axis=1)

    def predict_proba(
        self, txns: list[Transaction], scaled: np.ndarray | None = None
    ) -> np.ndarray:
        """The outlier score in [0, 1] — a fixed function of the row, not of the batch."""
        if self.model is None:
            return np.zeros(len(txns), dtype=float)
        if not txns:
            return np.zeros(0, dtype=float)
        Xs = self.scaled(txns) if scaled is None else scaled
        return self.score_map.apply(self.raw_scores(scaled=Xs))

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

    # ── introspection ───────────────────────────────────────────────────────────
    def model_card(self) -> dict[str, Any]:
        """Everything a reported number needs to be traceable back to the model that made it."""
        return {
            "detector": type(self).__name__,
            "kind": self.kind,
            "contamination": self.contamination,
            "seed": self.seed,
            "training": self.training.to_dict(),
            "score_map": self.score_map.to_dict(),
            "decision": self.policy.to_dict(),
        }


def contaminated_control(
    txns: list[Transaction], template: AnomalyDetector | None = None, **kwargs
) -> AnomalyDetector:
    """The same detector fitted on fraud rows too — a *control*, never a shipped configuration.

    "Fit on legit only" is a design claim, and a design claim with nothing measured against it is
    a preference. This builds the variant the claim rules out so that ticket 10 can put the two
    in the same table. Its `model_card()` reports `legit_only: false` and a non-zero
    `n_fraud_seen`, so a number produced by this cannot be mistaken for a number produced by the
    detector the loop runs.
    """
    detector = template or AnomalyDetector(**kwargs)
    return detector._fit(txns, legit_only=False)


class EnsembleDetector:
    """Supervised + unsupervised, blended. The blend weight is a config knob, not a constant."""

    def __init__(
        self, supervised, unsupervised, weight: float = 0.7, policy: DecisionPolicy | None = None
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"blend weight {weight} is not a share of the score")
        self.supervised = supervised
        self.unsupervised = unsupervised
        self.weight = weight
        self.policy = policy or DecisionPolicy()

    def fit(
        self, txns: list[Transaction], sample_weight: np.ndarray | None = None
    ) -> EnsembleDetector:
        self.supervised.fit(txns, sample_weight=sample_weight)
        self.unsupervised.fit(txns)
        return self

    @property
    def training_rows(self) -> list[Transaction]:
        """Both halves' training rows. Either one is enough to lose a carve-out."""
        return [*self.supervised.training_rows, *self.unsupervised.training_rows]

    def split_proba(self, txns: list[Transaction]) -> tuple[np.ndarray, np.ndarray]:
        """(supervised, unsupervised) probabilities for the same rows, computed once.

        The blend is one line of arithmetic over these two vectors, so a sweep over the weight
        costs one scoring pass rather than one per weight — which is what makes "the blend is
        measured, not assumed" affordable on a 150k-row holdout.
        """
        # Duck-typed, and deliberately: the ensemble takes *a* supervised detector, and only the
        # gradient-boosted one can hand back the design matrix it just built. Anything else is
        # scored the plain way and explained from its own features, at the cost of one extra
        # pass. A hard dependency here would quietly make `supervised` mean `LGBMDetector`.
        reuses_features = hasattr(self.supervised, "design_matrix") and (
            getattr(self.supervised, "model", None) is not None
        )
        values = self.supervised.design_matrix(txns) if reuses_features else None
        p_sup = (
            self.supervised.predict_proba(txns, values=values)
            if reuses_features
            else self.supervised.predict_proba(txns)
        )
        return p_sup, self.unsupervised.predict_proba(txns)

    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        p_sup, p_uns = self.split_proba(txns)
        return self.weight * p_sup + (1 - self.weight) * p_uns

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

    def model_card(self) -> dict[str, Any]:
        from afl.defend.models.lgbm import model_card_of

        return {
            "detector": type(self).__name__,
            "weight": self.weight,
            "supervised": model_card_of(self.supervised),
            "unsupervised": self.unsupervised.model_card(),
            "decision": self.policy.to_dict(),
        }


__all__ = [
    "MIN_FIT_ROWS",
    "AnomalyDetector",
    "AnomalyTraining",
    "EnsembleDetector",
    "ScoreMap",
    "contaminated_control",
]
