"""The mutation half of the loop.

fitness = evasion_rate − λ · realism_penalty

The λ term is the whole ethic of the project: an optimiser rewarded on evasion alone will
happily discover traffic that evades because it is absurd. Realism is not a nice-to-have here,
it is what keeps the resulting recall lift meaningful.

Optuna drives the search when it is installed; a plain random/hill-climb sampler stands in when
it is not, so the loop never hard-depends on the optimiser backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from afl.attack import realism as realism_lib
from afl.attack.templates import registry
from afl.contract.schema import AttackBatch, AttackParams, Transaction
from afl.utils.seed import rng as make_rng


@dataclass
class Trial:
    """One proposal and how it scored: evasion, realism penalty, and the resulting fitness."""

    vector_id: str
    params: dict[str, Any]
    evasion_rate: float = 0.0
    realism_penalty: float = 0.0
    fitness: float = 0.0
    n_fraud: int = 0
    n_evasions: int = 0


class AttackOptimiser:
    """▲ A. `propose()` → next params; `update(evasions)` → learn from what got through."""

    def __init__(
        self,
        vector_id: str = "S1",
        seed: int = 1337,
        lambda_realism: float = 0.5,
        backend: str = "auto",  # "optuna" | "random" | "auto"
    ) -> None:
        self.vector_id = vector_id
        self.spec = registry.get(vector_id)
        self.lambda_realism = lambda_realism
        self.rng = make_rng(seed)
        self.trials: list[Trial] = []
        self._pending: Trial | None = None
        self._last_batch_stats: dict[str, float] = {}
        self._study = None
        self._optuna_trial = None

        if backend in ("optuna", "auto"):
            try:
                import optuna

                optuna.logging.set_verbosity(optuna.logging.WARNING)
                self._optuna = optuna
                self._study = optuna.create_study(
                    direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
                )
            except ImportError:
                if backend == "optuna":
                    raise
                self._optuna = None
        else:
            self._optuna = None

    # ── proposal ────────────────────────────────────────────────────────────────
    def _sample_random(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, b in self.spec.search_space.items():
            if b["type"] == "int":
                out[k] = int(self.rng.integers(b["low"], b["high"] + 1))
            elif b.get("log"):
                lo, hi = float(b["low"]), float(b["high"])
                out[k] = float(math.exp(self.rng.uniform(math.log(lo), math.log(hi))))
            else:
                out[k] = float(self.rng.uniform(b["low"], b["high"]))
        return out

    def _sample_optuna(self) -> dict[str, Any]:
        self._optuna_trial = self._study.ask()
        out: dict[str, Any] = {}
        for k, b in self.spec.search_space.items():
            if b["type"] == "int":
                out[k] = self._optuna_trial.suggest_int(k, int(b["low"]), int(b["high"]))
            else:
                out[k] = self._optuna_trial.suggest_float(
                    k, float(b["low"]), float(b["high"]), log=bool(b.get("log"))
                )
        return out

    def propose(self) -> AttackParams:
        knobs = self._sample_optuna() if self._study is not None else self._sample_random()
        knobs = registry.clamp(self.vector_id, knobs)
        self._pending = Trial(vector_id=self.vector_id, params=knobs)
        return self.spec.to_attack_params(knobs)

    # ── feedback ────────────────────────────────────────────────────────────────
    def observe_batch(self, batch: AttackBatch) -> None:
        """Called by `bind()`'s wrapper before the detector sees the batch.

        Gives the optimiser the denominator for the evasion rate and the realism penalty,
        without the loop having to know either side exists.
        """
        report = realism_lib.check(batch)
        self._last_batch_stats = {
            "n_fraud": float(len(batch.fraud_transactions)),
            "realism_penalty": report.penalty,
        }

    def update(self, evasions: list[Transaction]) -> None:
        trial = self._pending or Trial(vector_id=self.vector_id, params={})
        n_fraud = self._last_batch_stats.get("n_fraud", 0.0)
        trial.n_fraud = int(n_fraud)
        trial.n_evasions = len(evasions)
        trial.evasion_rate = (len(evasions) / n_fraud) if n_fraud else 0.0
        trial.realism_penalty = self._last_batch_stats.get("realism_penalty", 0.0)
        trial.fitness = trial.evasion_rate - self.lambda_realism * trial.realism_penalty

        if self._study is not None and self._optuna_trial is not None:
            self._study.tell(self._optuna_trial, trial.fitness)
            self._optuna_trial = None

        self.trials.append(trial)
        self._pending = None

    # ── inspection ──────────────────────────────────────────────────────────────
    @property
    def best(self) -> Trial | None:
        return max(self.trials, key=lambda t: t.fitness) if self.trials else None

    def history(self) -> list[dict[str, Any]]:
        return [t.__dict__ for t in self.trials]

    def bind(self, simulator) -> ObservedSimulator:  # noqa: ANN001 - Simulator protocol
        """Wrap a simulator so every generated batch is reported back here."""
        return ObservedSimulator(simulator, self)


@dataclass
class ObservedSimulator:
    """Simulator-protocol pass-through that feeds batch stats to the optimiser.

    Keeps `run_closed_loop` free of any knowledge of realism scoring — the loop still only
    calls `generate`, `score`, `update`, `retrain`, `leave_one_attack_out`.
    """

    inner: Any
    optimiser: AttackOptimiser
    batches: list[AttackBatch] = field(default_factory=list)

    def generate(self, params: AttackParams) -> AttackBatch:
        batch = self.inner.generate(params)
        self.optimiser.observe_batch(batch)
        self.batches.append(batch)
        return batch
