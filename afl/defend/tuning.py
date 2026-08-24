"""Honest hyperparameter search for the supervised baseline.

Ticket 08's whole point: *"This is the hard baseline everything else must beat, so it must be
tuned honestly rather than left weak to flatter what comes later. A soft baseline makes every
subsequent result meaningless."* Two rules follow from that, and they pull in opposite
directions, which is why they are both written down here.

1. **Search hard.** An untuned detector at a 0.1% base rate is not a baseline, it is a straw
   man. The search covers depth, regularisation, sampling and the class weighting that is the
   obvious thing to reach for at this base rate and is not obviously right.
2. **Search on validation only.** The tuner is handed two row lists and never sees a third. It
   asserts that the second starts strictly after the first ends, because the failure mode is
   silent: a search that touches the test window produces a number that looks like skill.

The search is a function of `(fit rows, val rows, space, seed)` and nothing else, so the params
it lands on are reproducible and get committed next to the metrics they produced.

Optuna drives the search when it is installed; a plain random sampler stands in when it is not,
so nothing here hard-depends on the backend and CI exercises both paths.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from afl.contract.schema import Transaction
from afl.defend.features import FeatureBuilder
from afl.defend.models.lgbm import DEFAULT_PARAMS, make_estimator

log = logging.getLogger(__name__)

#: What the search maximises. Both are measured on the validation tail, never on test.
METRICS = ("pr_auc", "recall_at_fixed_fpr")

#: The default envelope. Bounds are deliberately wide — a narrow space around the defaults would
#: make the search a formality and the baseline soft. Overridable from
#: `config/defend/lgbm.yaml: tuning.search_space`.
DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "num_leaves": {"type": "int", "low": 15, "high": 255, "log": True},
    "min_child_samples": {"type": "int", "low": 5, "high": 300, "log": True},
    "n_estimators": {"type": "int", "low": 100, "high": 900, "step": 100},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
    "reg_lambda": {"type": "float", "low": 1e-3, "high": 50.0, "log": True},
    "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
}


class LeakingValidationSplit(AssertionError):
    """Raised when the tuning validation set is not strictly after the fitting set."""


@dataclass(frozen=True)
class TuningResult:
    """The params the search landed on, and enough context to believe them."""

    params: dict[str, Any]
    metric: str
    best_score: float
    default_score: float
    n_trials: int
    backend: str  # "optuna" | "random" | "none"
    seed: int
    n_fit_rows: int
    n_val_rows: int
    n_val_positives: int
    fit_end: str = ""
    val_start: str = ""
    seconds: float = 0.0
    skipped: str = ""  # non-empty when the search did not run, and why
    trials: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tuned(self) -> bool:
        return not self.skipped

    @property
    def lift(self) -> float:
        """What the search bought over the defaults, on validation. Can be negative; say so."""
        return round(self.best_score - self.default_score, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tuned": self.tuned,
            "skipped": self.skipped,
            "metric": self.metric,
            "best_score": round(self.best_score, 6),
            "default_score": round(self.default_score, 6),
            "lift_over_defaults": self.lift,
            "n_trials": self.n_trials,
            "backend": self.backend,
            "seed": self.seed,
            "n_fit_rows": self.n_fit_rows,
            "n_val_rows": self.n_val_rows,
            "n_val_positives": self.n_val_positives,
            "fit_end": self.fit_end,
            "val_start": self.val_start,
            "seconds": round(self.seconds, 2),
            "params": dict(self.params),
            "trials": self.trials,
        }


def assert_validation_is_out_of_time(fit: list[Transaction], val: list[Transaction]) -> None:
    """The one guard that matters here: the tuner must not be able to see the future.

    Deliberately an exception rather than a warning. A tuner that quietly searched across the
    boundary produces params that look good for a reason nobody can reconstruct later.
    """
    if not fit or not val:
        return
    fit_end, val_start = max(t.ts for t in fit), min(t.ts for t in val)
    if val_start <= fit_end:
        raise LeakingValidationSplit(
            f"tuning validation starts {val_start} but the fitting rows run to {fit_end} — "
            "the search would be scoring itself on data it fitted on"
        )
    overlap = {t.txn_id for t in fit} & {t.txn_id for t in val}
    if overlap:
        raise LeakingValidationSplit(
            f"{len(overlap)} txn_id(s) are in both the fitting and validation sets, "
            f"e.g. {sorted(overlap)[:3]}"
        )


def _score(y: np.ndarray, p: np.ndarray, metric: str, fixed_fpr: float) -> float:
    """The search objective. Never accuracy, never ROC-AUC — both are noise at this base rate."""
    if y.sum() == 0 or y.sum() == y.size:
        return 0.0
    if metric == "recall_at_fixed_fpr":
        thr = float(np.quantile(p[y == 0], 1.0 - fixed_fpr))
        return float((p[y == 1] > thr).mean())
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


def _sample(space: dict[str, dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """One draw from the envelope, for the no-Optuna path."""
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec.get("type", "float")
        if kind == "categorical":
            choices = list(spec["choices"])
            out[name] = choices[int(rng.integers(len(choices)))]
        elif kind == "int":
            if spec.get("log"):
                lo, hi = np.log(spec["low"]), np.log(spec["high"])
                out[name] = int(round(float(np.exp(rng.uniform(lo, hi)))))
            else:
                step = int(spec.get("step", 1))
                n = (int(spec["high"]) - int(spec["low"])) // step
                out[name] = int(spec["low"]) + step * int(rng.integers(n + 1))
        elif spec.get("log"):
            lo, hi = np.log(spec["low"]), np.log(spec["high"])
            out[name] = float(np.exp(rng.uniform(lo, hi)))
        else:
            out[name] = float(rng.uniform(spec["low"], spec["high"]))
    return out


def _suggest(trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The same envelope, expressed to Optuna."""
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec.get("type", "float")
        if kind == "categorical":
            out[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif kind == "int":
            out[name] = (
                trial.suggest_int(
                    name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False))
                )
                if spec.get("log")
                else trial.suggest_int(
                    name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
                )
            )
        else:
            out[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
    return out


def tune(
    fit_rows: list[Transaction],
    val_rows: list[Transaction],
    *,
    base_params: dict[str, Any] | None = None,
    search_space: dict[str, dict[str, Any]] | None = None,
    n_trials: int = 40,
    seed: int = 1337,
    metric: str = "pr_auc",
    fixed_fpr: float = 0.01,
    backend: str = "auto",  # "optuna" | "random" | "auto"
    features: FeatureBuilder | None = None,
) -> TuningResult:
    """Search the envelope on `val_rows`, which must be strictly after `fit_rows`.

    The feature table is built **once** — the search varies model hyperparameters, not features,
    and rebuilding a 400k-row causal table per trial would make the search cost the run rather
    than the model. `update=True` on the fitting rows and `update=False` on validation is the
    same discipline production scoring uses.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown tuning metric {metric!r}; known: {METRICS}")
    base = {**DEFAULT_PARAMS, **(base_params or {})}
    space = dict(search_space or DEFAULT_SEARCH_SPACE)
    started = time.perf_counter()

    def skipped(reason: str) -> TuningResult:
        log.warning("tuning skipped: %s — falling back to the default params", reason)
        return TuningResult(
            params=base,
            metric=metric,
            best_score=0.0,
            default_score=0.0,
            n_trials=0,
            backend="none",
            seed=seed,
            n_fit_rows=len(fit_rows),
            n_val_rows=len(val_rows),
            n_val_positives=sum(1 for t in val_rows if t.is_fraud),
            skipped=reason,
            seconds=time.perf_counter() - started,
        )

    assert_validation_is_out_of_time(fit_rows, val_rows)
    y_fit = FeatureBuilder.labels(fit_rows)
    y_val = FeatureBuilder.labels(val_rows)
    if not len(fit_rows) or not len(val_rows):
        return skipped("no rows on one side of the tuning split")
    if y_fit.sum() == 0 or y_val.sum() == 0:
        return skipped(
            f"{int(y_fit.sum())} fraud rows to fit on and {int(y_val.sum())} to validate on; "
            "a search with nothing to rank is a random walk"
        )

    builder = features or FeatureBuilder(stateful=True)
    builder.reset()
    X_fit = builder.transform(fit_rows, update=True).to_numpy()
    X_val = builder.transform(val_rows, update=False).to_numpy()

    trials: list[dict[str, Any]] = []

    def objective(params: dict[str, Any]) -> float:
        model, _ = make_estimator({**base, **params}, seed)
        model.fit(X_fit, y_fit)
        return _score(y_val, model.predict_proba(X_val)[:, 1], metric, fixed_fpr)

    default_score = objective({})
    log.info(
        "tuning: defaults score %s=%.6f on %d validation rows", metric, default_score, len(y_val)
    )

    best_params, best_score = dict(base), default_score
    used = "random"
    if backend in ("optuna", "auto"):
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
            )

            def _objective(trial):
                params = _suggest(trial, space)
                score = objective(params)
                trials.append({"trial": trial.number, "score": round(score, 6), "params": params})
                return score

            study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
            if study.best_value > best_score:
                best_params, best_score = {**base, **study.best_params}, float(study.best_value)
            used = "optuna"
        except ImportError:
            log.warning("optuna not installed — searching the same envelope at random")
            used = "random"

    if used == "random":
        rng = np.random.default_rng(seed)
        for i in range(n_trials):
            params = _sample(space, rng)
            score = objective(params)
            trials.append({"trial": i, "score": round(score, 6), "params": params})
            if score > best_score:
                best_params, best_score = {**base, **params}, score

    log.info(
        "tuning: %d trials on %s, best %s=%.6f (defaults %.6f, %+.6f)",
        len(trials),
        used,
        metric,
        best_score,
        default_score,
        best_score - default_score,
    )
    return TuningResult(
        params=best_params,
        metric=metric,
        best_score=best_score,
        default_score=default_score,
        n_trials=len(trials),
        backend=used,
        seed=seed,
        n_fit_rows=len(fit_rows),
        n_val_rows=len(val_rows),
        n_val_positives=int(y_val.sum()),
        fit_end=max(t.ts for t in fit_rows).isoformat(),
        val_start=min(t.ts for t in val_rows).isoformat(),
        seconds=time.perf_counter() - started,
        trials=sorted(trials, key=lambda t: -t["score"]),
    )
