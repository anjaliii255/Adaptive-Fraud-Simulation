"""The workhorse. Gradient-boosted trees over the causal feature table.

This is the detector every headline number is measured against. The exotic layers (sequence,
GNN) have to beat *this*, on the same split, before they earn a place in the table — which only
means something if this one is tuned honestly rather than left weak to flatter what comes later.
The tuning lives in `afl/defend/tuning.py`, and it never sees the test window.

Falls back to sklearn's HistGradientBoosting when LightGBM cannot be loaded, so the loop runs
anywhere. The fallback is logged loudly and recorded on the model card, because on macOS the
LightGBM wheel imports cleanly and *then* fails to `dlopen` its own shared library when libomp
is missing — the one failure mode that silently renames the model in the table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from afl.contract.metrics import DetectorScore
from afl.contract.schema import AttackBatch, Transaction
from afl.defend.decision import DecisionPolicy
from afl.defend.features import FeatureBuilder

log = logging.getLogger(__name__)

#: The starting point, before any tuning. Every name here is LightGBM's; the sklearn fallback is
#: handed the same values under its own names by `_sklearn_estimator`.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "n_estimators": 300,
    "subsample": 0.9,
    # LightGBM ignores `subsample` unless bagging is switched on with a frequency. Left at its
    # default of 0 the row-sampling fraction above is inert, so a search over it would be a
    # search over a knob that does nothing — which is worse than not searching it.
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.0,
    # None or "balanced". Not set to "balanced" by default: at a 0.1% base rate it is an obvious
    # thing to reach for and not obviously right, so it is a searched knob and the validation
    # split decides. See afl/defend/tuning.py.
    "class_weight": None,
    "verbosity": -1,
}

#: Same knob, different name. Anything outside this map and `_NOT_A_KNOB` has no
#: HistGradientBoosting equivalent, and is recorded as dropped rather than silently ignored.
_SKLEARN_ALIASES = {
    "learning_rate": "learning_rate",
    "n_estimators": "max_iter",
    "num_leaves": "max_leaf_nodes",
    "min_child_samples": "min_samples_leaf",
    "reg_lambda": "l2_regularization",
    "max_depth": "max_depth",
    "class_weight": "class_weight",
}

#: Configures the backend rather than the model, so its absence from the fallback is not a loss.
_NOT_A_KNOB = frozenset({"objective", "verbosity", "verbose", "random_state", "seed", "n_jobs"})

_FALLBACK_WARNED = False  # the backend is an environment fact; say it once, not once per fit


@dataclass(frozen=True)
class Backend:
    """Which library produced a number, and what was lost getting there.

    Recorded on every model card and written into every run artefact. A backend that is not
    written down is a backend nobody checks, and "LightGBM baseline" then names a table rather
    than a model.
    """

    name: str
    version: str
    reason: str
    dropped_params: tuple[str, ...] = ()

    @property
    def is_lightgbm(self) -> bool:
        return self.name == "lightgbm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "reason": self.reason,
            "dropped_params": list(self.dropped_params),
        }

    def __str__(self) -> str:
        return f"{self.name} {self.version}" if self.version != "-" else self.name


UNTRAINED = Backend(name="untrained", version="-", reason="fit() has not been called")


def _sklearn_estimator(params: dict[str, Any], seed: int) -> tuple[Any, Backend, list[str]]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    kwargs: dict[str, Any] = {}
    dropped: list[str] = []
    for name, value in params.items():
        if name in _NOT_A_KNOB:
            continue
        alias = _SKLEARN_ALIASES.get(name)
        if alias is None:
            dropped.append(name)
            continue
        if name == "max_depth" and value in (-1, 0):  # LightGBM's "no limit" is sklearn's None
            value = None
        kwargs[alias] = value

    import sklearn

    return (
        HistGradientBoostingClassifier(random_state=seed, **kwargs),
        Backend(
            name="sklearn-hgb",
            version=sklearn.__version__,
            reason="",  # filled in by the caller, which knows why LightGBM was unavailable
            dropped_params=tuple(sorted(dropped)),
        ),
        dropped,
    )


def make_estimator(params: dict[str, Any], seed: int) -> tuple[Any, Backend]:
    """The configured backend, or the sklearn fallback, saying loudly which one it got.

    Shared with the tuner so a hyperparameter searched on one backend is never applied to
    another: the search and the reported model are the same estimator, built here.
    """
    global _FALLBACK_WARNED
    # `seed` is the default, not an override: a params dict that already names `random_state`
    # (every LGBMDetector's does) must not collide with it.
    params = {"random_state": seed, **params}
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(**params)
        return model, Backend(
            name="lightgbm",
            version=lgb.__version__,
            reason="libomp present, LightGBM loaded",
        )
    except (ImportError, OSError) as e:
        # ImportError: not installed. OSError: installed but libomp is missing, which is the
        # usual macOS case — the wheel imports and then fails to dlopen its own shared library.
        model, backend, dropped = _sklearn_estimator(params, seed)
        backend = Backend(
            name=backend.name,
            version=backend.version,
            reason=f"LightGBM unavailable ({type(e).__name__}): {str(e).splitlines()[0][:160]}",
            dropped_params=backend.dropped_params,
        )
        if not _FALLBACK_WARNED:
            log.warning(
                "LIGHTGBM UNAVAILABLE (%s) - running on the sklearn HistGradientBoosting "
                "fallback, which is NOT the headline detector. %s has no equivalent there and "
                "is being dropped. On macOS: brew install libomp",
                type(e).__name__,
                ", ".join(dropped) or "nothing",
                stacklevel=2,
            )
            _FALLBACK_WARNED = True
        return model, backend


@dataclass
class TrainingRecord:
    """What one fit actually saw. Goes into the run artefact next to the metrics."""

    n_rows: int = 0
    n_fraud: int = 0
    base_rate: float = 0.0
    n_features: int = 0
    n_weighted_up: int = 0
    replay_weight: float = 1.0
    fitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_fraud": self.n_fraud,
            "base_rate": round(self.base_rate, 8),
            "n_features": self.n_features,
            "n_weighted_up": self.n_weighted_up,
            "replay_weight": self.replay_weight,
            "fitted": self.fitted,
        }


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
        params_source: str = "default",
    ) -> None:
        self.policy = policy or DecisionPolicy()
        self.features = features or FeatureBuilder(stateful=True)
        self.seed = seed
        self.replay_weight = replay_weight  # evasions are the expensive examples; weight them up
        self.explain = explain
        #: Where `params` came from — "default", or the artefact a tuned set was read from.
        #: Written into the model card so a reported number says whether it was tuned.
        self.params_source = params_source
        self.params = {**DEFAULT_PARAMS, "random_state": seed, **(params or {})}
        self.model = None
        self.backend: Backend = UNTRAINED
        self.training = TrainingRecord(replay_weight=replay_weight)
        self.feature_names: list[str] = []
        self._replay: list[Transaction] = []  # evasion memory across rounds
        self._corpus: list[Transaction] = []  # everything trained on so far
        self._explainer = None
        self._warned_unfitted = False

    # ── training ────────────────────────────────────────────────────────────────
    def _new_model(self):
        model, backend = make_estimator(self.params, self.seed)
        self.backend = backend
        return model

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
        self.training = TrainingRecord(
            n_rows=len(txns),
            n_fraud=int(y.sum()),
            base_rate=float(y.mean()) if len(y) else 0.0,
            n_features=X.shape[1],
            n_weighted_up=int((np.asarray(sample_weight) > 1.0).sum())
            if sample_weight is not None
            else 0,
            replay_weight=self.replay_weight,
        )
        if len(set(y.tolist())) < 2:
            log.warning("single-class training set (%d rows) — model not fitted", len(y))
            return self
        self.feature_names = list(X.columns)
        self.model = self._new_model()
        try:
            self.model.fit(X.to_numpy(), y, sample_weight=sample_weight)
        except TypeError:  # backend without sample_weight support
            self.model.fit(X.to_numpy(), y)
        self.training.fitted = True
        self._explainer = None
        return self

    def sample_weights(self, txns: list[Transaction]) -> np.ndarray:
        """`replay_weight` on every row that once evaded, 1.0 on the rest.

        A row that got through is the expensive example — it is the only evidence the detector
        has about its own blind spot — so it counts for more than an ordinary training row. How
        much more is `defend.supervised.replay_weight` in config, never a literal here.
        """
        heavy = {t.txn_id for t in self._replay}
        return np.array(
            [self.replay_weight if t.txn_id in heavy else 1.0 for t in txns], dtype=float
        )

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
        self._fit_model(self._corpus, self.sample_weights(self._corpus))

    # ── scoring ─────────────────────────────────────────────────────────────────
    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        if self.model is None:
            if not self._warned_unfitted:
                log.warning(
                    "scoring with an unfitted detector — every row scores 0.0, which reads in a "
                    "metric exactly like a detector that caught nothing"
                )
                self._warned_unfitted = True
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
    def model_card(self) -> dict[str, Any]:
        """Everything a reported number needs to be traceable back to the model that made it."""
        return {
            "detector": type(self).__name__,
            "backend": self.backend.to_dict(),
            "params": dict(self.params),
            "params_source": self.params_source,
            "seed": self.seed,
            "training": self.training.to_dict(),
            "decision": {
                "mode": self.policy.mode,
                "step_up_at": self.policy.step_up_at,
                "hold_at": self.policy.hold_at,
                "review_at": self.policy.review_at,
                "decline_at": self.policy.decline_at,
            },
        }

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


def backend_of(detector) -> Backend:
    """The backend behind any detector — including one wrapped in an ensemble."""
    inner = getattr(detector, "supervised", detector)
    got = getattr(inner, "backend", None)
    return got if isinstance(got, Backend) else UNTRAINED


def model_card_of(detector) -> dict[str, Any]:
    """The model card of any detector, or a minimal one for a detector that has no card."""
    inner = getattr(detector, "supervised", detector)
    card = getattr(inner, "model_card", None)
    if callable(card):
        out = card()
        if inner is not detector:
            out = {"detector": type(detector).__name__, "supervised": out}
        return out
    return {"detector": type(detector).__name__, "backend": UNTRAINED.to_dict()}


__all__ = [
    "DEFAULT_PARAMS",
    "UNTRAINED",
    "Backend",
    "LGBMDetector",
    "TrainingRecord",
    "backend_of",
    "make_estimator",
    "model_card_of",
]
