"""Where A meets B.

The loop only ever touches contract types: it takes an `AttackBatch` from the red side,
a `list[DetectorScore]` from the blue side, and a `MetricResult` from evaluation. It knows
nothing about how any of the three are produced — which is exactly why both halves can be
swapped from stub to real without touching this file.

Run it hollow first. A hollow loop that runs beats two polished halves that never connect.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from afl.contract.metrics import Action, DetectorScore, MetricResult
from afl.contract.schema import AttackBatch, AttackParams, Transaction


@runtime_checkable
class Simulator(Protocol):
    """▲ A — turns params into a labelled, schema-valid batch."""

    def generate(self, params: AttackParams) -> AttackBatch: ...


@runtime_checkable
class Optimiser(Protocol):
    """▲ A — proposes the next attack, learns from what got through."""

    def propose(self) -> AttackParams: ...

    def update(self, evasions: list[Transaction]) -> None: ...


@runtime_checkable
class Detector(Protocol):
    """■ B — scores a batch, then learns from it."""

    def score(self, batch: AttackBatch) -> list[DetectorScore]: ...

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None: ...


@runtime_checkable
class Evaluator(Protocol):
    """■ B — the honest number, measured on data the loop never trained on."""

    def leave_one_attack_out(self, detector: Detector) -> MetricResult: ...


def find_evasions(batch: AttackBatch, scores: list[DetectorScore]) -> list[Transaction]:
    """⚑ The seam's one piece of logic: fraud that was let through untouched.

    Matched by `txn_id` rather than by position — a detector is free to return scores in any
    order, and a silent zip-misalignment here would corrupt every number downstream.
    """
    by_id = {s.txn_id: s for s in scores}
    missing = [t.txn_id for t in batch.transactions if t.txn_id not in by_id]
    if missing:
        raise ValueError(
            f"detector returned no score for {len(missing)} transaction(s), "
            f"e.g. {missing[:3]} — the seam requires one DetectorScore per transaction"
        )
    return [t for t in batch.transactions if t.is_fraud and by_id[t.txn_id].action == Action.ALLOW]


def run_closed_loop(
    simulator: Simulator,
    optimiser: Optimiser,
    detector: Detector,
    evaluator: Evaluator,
    rounds: int,
    tracker: Any,
) -> list[dict[str, Any]]:
    """attack → detect → log evasions → mutate → retrain → eval, `rounds` times."""
    for r in range(rounds):
        batch = simulator.generate(optimiser.propose())  # ▲ A
        scores = detector.score(batch)  # ■ B
        evasions = find_evasions(batch, scores)  # ⚑ seam

        optimiser.update(evasions)  # ▲ A  (mutate)
        detector.retrain(batch, evasions)  # ■ B  (learn)

        metrics = evaluator.leave_one_attack_out(detector)  # ■ B

        n_fraud = len(batch.fraud_transactions)
        tracker.log(
            round=r,
            vector_id=batch.params.vector_id,
            n_transactions=len(batch.transactions),
            n_fraud=n_fraud,
            n_evasions=len(evasions),
            # rate over fraud rows, not all rows: diluting by legit volume makes the
            # convergence curve a function of batch composition instead of attack success.
            evasion_rate=(len(evasions) / n_fraud) if n_fraud else 0.0,
            **metrics.model_dump(),
        )  # ⚑ convergence curve

    return tracker.history
