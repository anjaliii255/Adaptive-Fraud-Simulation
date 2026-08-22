"""Dummy halves so the loop can run on day one, before either side exists.

These are deliberately dumb but *contract-valid*: they exercise the schema round-trip, the
evasion seam, and the tracker wiring. Step 4 of the build order swaps them out one at a time.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from afl.contract.metrics import Action, DetectorScore, MetricResult
from afl.contract.schema import AttackBatch, AttackParams, Rail, Transaction
from afl.utils.seed import child_seed, rng

_EPOCH = datetime(2024, 1, 1)


class StubSimulator:
    """Emits a mixed batch of legit + fraud rows whose separability is set by `params`."""

    def __init__(self, seed: int = 1337, n_per_batch: int = 200, fraud_rate: float = 0.1) -> None:
        self.seed = seed
        self.n_per_batch = n_per_batch
        self.fraud_rate = fraud_rate
        self._round = 0

    def generate(self, params: AttackParams) -> AttackBatch:
        seed = child_seed(self.seed, params.vector_id, self._round)
        r = rng(seed)
        run_id = f"stub-{self._round:03d}-{uuid.UUID(int=seed).hex[:8]}"
        # "stealth" shrinks the amount gap between fraud and legit — the knob the optimiser turns
        stealth = float(params.params.get("stealth", 0.0))

        txns: list[Transaction] = []
        n_fraud = max(1, int(self.n_per_batch * self.fraud_rate))
        for i in range(self.n_per_batch):
            is_fraud = i < n_fraud
            amount = float(r.lognormal(3.0 + (1.5 * (1 - stealth) if is_fraud else 0.0), 0.5))
            txns.append(
                Transaction(
                    txn_id=f"{run_id}-{i:05d}",
                    ts=_EPOCH + timedelta(seconds=int(r.integers(0, 86_400))),
                    src=f"e{int(r.integers(0, 50)):03d}",
                    dst=f"e{int(r.integers(50, 100)):03d}",
                    amount=round(amount, 2),
                    rail=Rail.A2A,
                    device_id=f"d{int(r.integers(0, 20)):02d}",
                    is_fraud=is_fraud,
                    vector_id=params.vector_id if is_fraud else None,
                    attack_run_id=run_id if is_fraud else None,
                )
            )
        self._round += 1
        return AttackBatch(run_id=run_id, params=params, transactions=txns, seed=seed)


class StubOptimiser:
    """Hill-climbs one knob: more evasions → push stealth further."""

    def __init__(self, vector_id: str = "S1", engine: str = "graph", step: float = 0.1) -> None:
        self.vector_id = vector_id
        self.engine = engine
        self.step = step
        self.stealth = 0.0
        self.history: list[int] = []

    def propose(self) -> AttackParams:
        return AttackParams(
            vector_id=self.vector_id,
            engine=self.engine,
            params={"stealth": round(self.stealth, 4)},
        )

    def update(self, evasions: list[Transaction]) -> None:
        self.history.append(len(evasions))
        # got caught → get quieter; getting through → keep the recipe
        if not evasions:
            self.stealth = min(1.0, self.stealth + self.step)


class StubDetector:
    """Amount threshold, nudged down whenever fraud slips past."""

    def __init__(self, threshold: float = 100.0, decline_at: float = 0.5) -> None:
        self.threshold = threshold
        self.decline_at = decline_at
        self.retrain_calls = 0

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        out: list[DetectorScore] = []
        for t in batch.transactions:
            p = min(1.0, t.amount / (self.threshold * 4))
            action = Action.DECLINE if p >= self.decline_at else Action.ALLOW
            out.append(
                DetectorScore(
                    txn_id=t.txn_id,
                    score=round(p, 6),
                    action=action,
                    reasons=["amount_vs_threshold"],
                )
            )
        return out

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        self.retrain_calls += 1
        if evasions:
            cheapest = min(t.amount for t in evasions)
            self.threshold = min(self.threshold, max(1.0, cheapest * 2))


class StubEvaluator:
    """Returns a plausible, monotonically-improving MetricResult without touching real data."""

    def __init__(self, held_out_vector: str | None = "M3") -> None:
        self.held_out_vector = held_out_vector
        self._calls = 0

    def leave_one_attack_out(self, detector) -> MetricResult:  # noqa: ANN001 - Detector protocol
        self._calls += 1
        base = min(0.95, 0.40 + 0.05 * self._calls)
        return MetricResult(
            pr_auc=round(base, 4),
            recall_at_fixed_fpr=round(base * 0.9, 4),
            fixed_fpr=0.01,
            precision_at_k=round(base * 0.8, 4),
            held_out_vector=self.held_out_vector,
            k=100,
        )
