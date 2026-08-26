"""The adaptive half of the loop, searching across the strong vectors at once.

A single-vector optimiser grinds one family against a detector that may already handle it. The
interesting question is which of several surfaces is weakest *now*, and that changes as the
detector learns — so the search covers S1, S2 and S3 together and decides how much of each batch
goes to each.

Two things keep it honest. Fitness is `evasion − λ·realism_penalty`, so evasion bought with
absurd traffic is not worth having. And every candidate batch passes the commensurability audit
*before* it is scored: a batch the anchor can distinguish by one field is rejected outright, so
the optimiser can never be rewarded for finding a provenance leak. Without that gate the fastest
way to a high evasion rate is to drift off-anchor, which is exactly the failure the audit exists
to catch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from afl.attack import realism as realism_lib
from afl.attack.envelope import audit as envelope_audit
from afl.attack.templates import registry
from afl.contract.schema import AttackBatch, AttackParams, Transaction
from afl.utils.seed import rng as make_rng

log = logging.getLogger(__name__)

STRONG_VECTORS = ("S1", "S2", "S3")

#: How the batch is divided between vectors. A stated decision, not an emergent one.
ALLOCATIONS = ("uniform", "search", "fitness")

#: A candidate the anchor can separate by one field is not an attack, it is a tell.
AUDIT_LIFT_LIMIT = 3.0

#: Which rule decides that. Both read the same `envelope.audit` report; they disagree about what
#: counts as separable, and the disagreement is a function of anchor size.
#:
#:   `lift`     — reject at `AUDIT_LIFT_LIMIT x base_rate`, where the base rate is the synthetic
#:                share of anchor + batch. The default, and what ticket 12 shipped. It has no
#:                floor, so it tightens as the anchor grows: a hundred injected rows in a
#:                600k-row anchor put the bar at ~5e-4 PR-AUC, which log-amount alone clears.
#:                On both real anchors it rejects every candidate batch.
#:   `envelope` — reject on `envelope.audit`'s own `trivially_separable` verdict, floor included.
#:                The rule the rest of the repo already applies to the same question: it is what
#:                withholds a leave-one-attack-out fold and what `run_experiment` warns on.
#:
#: Ticket 16 runs on `envelope` and says so in its artefact, because a gate that refuses 100% of
#: candidates on every real anchor makes System C a copy of System A and the hero table vacuous.
#: Ticket 14 owns the leash and is where the two rules should be reconciled; until then both
#: verdicts are recorded on every trial, so no run has to be repeated to find out what the other
#: rule would have done.
AUDIT_RULES = ("lift", "envelope")


@dataclass
class MultiTrial:
    """One composite proposal: what each vector was asked to do, and how the batch scored."""

    allocation: dict[str, float]
    params: dict[str, dict[str, Any]]
    evasion_rate: float = 0.0
    realism_penalty: float = 0.0
    audit_score: float = 0.0
    audit_base_rate: float = 0.0
    audit_worst: str | None = None
    #: What each rule said about this batch, whichever one was in force. Recorded on every trial
    #: so a run never has to be repeated to find out what the other rule would have done.
    audit_rule: str = "lift"
    rejected_by_lift: bool = False
    rejected_by_envelope: bool = False
    rejected: bool = False
    fitness: float = 0.0
    n_fraud: int = 0
    n_evasions: int = 0
    per_vector_evasion: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class MultiVectorOptimiser:
    """Searches parameters *and* budget across the strong vectors, gated on commensurability."""

    def __init__(
        self,
        vectors: tuple[str, ...] = STRONG_VECTORS,
        seed: int = 1337,
        lambda_realism: float = 0.5,
        backend: str = "auto",
        allocation: str = "search",
        episodes_per_round: int = 12,
        anchor: list[Transaction] | None = None,
        audit_rule: str = "lift",
    ) -> None:
        if allocation not in ALLOCATIONS:
            raise ValueError(f"unknown allocation {allocation!r}; expected one of {ALLOCATIONS}")
        if audit_rule not in AUDIT_RULES:
            raise ValueError(f"unknown audit rule {audit_rule!r}; expected one of {AUDIT_RULES}")
        self.vectors = tuple(vectors)
        self.specs = {v: registry.get(v) for v in self.vectors}
        self.lambda_realism = lambda_realism
        self.allocation = allocation
        self.audit_rule = audit_rule
        self.episodes_per_round = episodes_per_round
        self.anchor = anchor or []
        self.rng = make_rng(seed)
        self.trials: list[MultiTrial] = []
        self.rejected = 0
        self._pending: MultiTrial | None = None
        self._batch_stats: dict[str, Any] = {}
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
    def _sample_knobs(self, vector_id: str) -> dict[str, Any]:
        space = self.specs[vector_id].search_space
        out: dict[str, Any] = {}
        for key, bounds in space.items():
            name = f"{vector_id}.{key}"
            if self._optuna_trial is not None:
                out[key] = (
                    self._optuna_trial.suggest_int(name, int(bounds["low"]), int(bounds["high"]))
                    if bounds["type"] == "int"
                    else self._optuna_trial.suggest_float(
                        name,
                        float(bounds["low"]),
                        float(bounds["high"]),
                        log=bool(bounds.get("log")),
                    )
                )
            elif bounds["type"] == "int":
                out[key] = int(self.rng.integers(bounds["low"], bounds["high"] + 1))
            elif bounds.get("log"):
                lo, hi = float(bounds["low"]), float(bounds["high"])
                out[key] = float(math.exp(self.rng.uniform(math.log(lo), math.log(hi))))
            else:
                out[key] = float(self.rng.uniform(bounds["low"], bounds["high"]))
        return registry.clamp(vector_id, out)

    def _sample_allocation(self) -> dict[str, float]:
        """How much of the batch each vector gets. Uniform, searched, or chased by success."""
        if self.allocation == "uniform":
            return {v: 1 / len(self.vectors) for v in self.vectors}

        if self.allocation == "fitness":
            # follow where evasion actually worked, with a floor so no vector is ever abandoned
            recent = [t for t in self.trials if not t.rejected][-5:]
            if not recent:
                return {v: 1 / len(self.vectors) for v in self.vectors}
            scores = {
                v: sum(t.per_vector_evasion.get(v, 0.0) for t in recent) / len(recent) + 0.05
                for v in self.vectors
            }
            total = sum(scores.values())
            return {v: s / total for v, s in scores.items()}

        if self._optuna_trial is not None:
            raw = {
                v: self._optuna_trial.suggest_float(f"alloc.{v}", 0.05, 1.0) for v in self.vectors
            }
        else:
            raw = {v: float(self.rng.uniform(0.05, 1.0)) for v in self.vectors}
        total = sum(raw.values())
        return {v: w / total for v, w in raw.items()}

    def propose(self) -> AttackParams:
        """A composite proposal: per-vector knobs plus the budget split, as one contract object."""
        if self._study is not None:
            self._optuna_trial = self._study.ask()
        allocation = self._sample_allocation()
        params = {v: self._sample_knobs(v) for v in self.vectors}
        self._pending = MultiTrial(allocation=allocation, params=params)
        return AttackParams(
            vector_id="+".join(self.vectors),
            engine="multi",
            params={
                "allocation": allocation,
                "vectors": params,
                "episodes": self.episodes_per_round,
            },
        )

    # ── feedback ────────────────────────────────────────────────────────────────
    def observe_batch(self, batch: AttackBatch) -> bool:
        """Audit-gate the candidate, then record what the loop needs to score it.

        Returns False when the batch is rejected, which is the signal not to train on it.
        """
        report = realism_lib.check(batch)
        fraud = batch.fraud_transactions
        audit = envelope_audit(self.anchor, fraud) if self.anchor else {}
        score = float(audit.get("score", 0.0))
        base = float(audit.get("base_rate", 0.0))
        by_lift = bool(self.anchor) and score >= AUDIT_LIFT_LIMIT * max(base, 1e-9)
        by_envelope = bool(self.anchor) and bool(audit.get("trivially_separable"))
        rejected = by_envelope if self.audit_rule == "envelope" else by_lift

        self._batch_stats = {
            "n_fraud": len(fraud),
            "realism_penalty": report.penalty,
            "audit_score": score,
            "audit_base_rate": base,
            "audit_worst": audit.get("worst"),
            "audit_rule": self.audit_rule,
            "rejected_by_lift": by_lift,
            "rejected_by_envelope": by_envelope,
            "rejected": rejected,
        }
        if rejected:
            self.rejected += 1
            log.warning(
                "audit gate (%s rule) rejected a candidate: %r at %.4f against a %.4f base "
                "rate — scoring it would reward the optimiser for finding a leak",
                self.audit_rule,
                audit.get("worst"),
                score,
                base,
            )
        return not rejected

    def update(self, evasions: list[Transaction]) -> None:
        trial = self._pending or MultiTrial(allocation={}, params={})
        stats = self._batch_stats
        n_fraud = int(stats.get("n_fraud", 0))
        trial.n_fraud = n_fraud
        trial.n_evasions = len(evasions)
        trial.evasion_rate = (len(evasions) / n_fraud) if n_fraud else 0.0
        trial.realism_penalty = float(stats.get("realism_penalty", 0.0))
        trial.audit_score = float(stats.get("audit_score", 0.0))
        trial.audit_base_rate = float(stats.get("audit_base_rate", 0.0))
        trial.audit_worst = stats.get("audit_worst")
        trial.audit_rule = str(stats.get("audit_rule", self.audit_rule))
        trial.rejected_by_lift = bool(stats.get("rejected_by_lift", False))
        trial.rejected_by_envelope = bool(stats.get("rejected_by_envelope", False))
        trial.rejected = bool(stats.get("rejected", False))

        by_vector: dict[str, list[int]] = {v: [0, 0] for v in self.vectors}
        for t in evasions:
            if t.vector_id in by_vector:
                by_vector[t.vector_id][0] += 1
        for v in self.vectors:
            by_vector[v][1] = max(by_vector[v][1], 1)
        trial.per_vector_evasion = {v: c[0] / max(n_fraud, 1) for v, c in by_vector.items()}

        # a rejected candidate scores worse than any honest one, so the search leaves that region
        trial.fitness = (
            -1.0
            if trial.rejected
            else trial.evasion_rate - self.lambda_realism * trial.realism_penalty
        )

        if self._study is not None and self._optuna_trial is not None:
            self._study.tell(self._optuna_trial, trial.fitness)
            self._optuna_trial = None
        self.trials.append(trial)
        self._pending = None

    # ── inspection ──────────────────────────────────────────────────────────────
    @property
    def best(self) -> MultiTrial | None:
        honest = [t for t in self.trials if not t.rejected]
        return max(honest, key=lambda t: t.fitness) if honest else None

    def history(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.trials]

    def bind(self, simulator) -> MultiVectorSimulator:
        return MultiVectorSimulator(simulator, self)


@dataclass
class MultiVectorSimulator:
    """Turns one composite proposal into one batch, and reports it back for the audit gate.

    `run_closed_loop` still only calls `generate`, so the loop stays unaware that several vectors
    are in play — the per-row `vector_id` carries which family each transaction came from.
    """

    inner: Any
    optimiser: MultiVectorOptimiser
    batches: list[AttackBatch] = field(default_factory=list)
    accepted: list[AttackBatch] = field(default_factory=list)

    def generate(self, params: AttackParams) -> AttackBatch:
        allocation = params.params.get("allocation") or {}
        per_vector = params.params.get("vectors") or {}
        episodes = int(params.params.get("episodes", 12))

        base_episodes = self.inner.n_episodes
        rows: list[Transaction] = []
        entities = list(getattr(self.inner, "entities", []))
        for vector_id, share in allocation.items():
            budget = max(1, round(share * episodes))
            self.inner.n_episodes = budget
            spec = registry.get(vector_id)
            batch = self.inner.generate(spec.to_attack_params(per_vector.get(vector_id, {})))
            rows.extend(t for t in batch.transactions if t.is_fraud)
        self.inner.n_episodes = base_episodes

        rows.sort(key=lambda t: t.ts)
        merged = AttackBatch(
            run_id=f"multi-{len(self.batches):03d}",
            params=params,
            transactions=rows,
            seed=self.inner.seed,
            entities=entities,
        )
        self.batches.append(merged)
        if self.optimiser.observe_batch(merged):
            self.accepted.append(merged)
        return merged
