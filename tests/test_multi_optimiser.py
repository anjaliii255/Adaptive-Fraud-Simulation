"""▲ A — the adaptive search, and the gate that keeps it honest.

The optimiser is rewarded for evading the detector, so the one thing it must never be able to do
is win by drifting off the anchor. A batch the anchor can separate by a single field is rejected
before it is scored, because otherwise the cheapest route to a high evasion rate is a provenance
leak — the exact failure the commensurability audit exists to catch.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from afl.attack.envelope import AnchorEnvelope
from afl.attack.multi import STRONG_VECTORS, MultiVectorOptimiser
from afl.attack.simulator import Simulator
from afl.contract.schema import Rail, Transaction
from afl.loop.closed_loop import Optimiser, run_closed_loop
from afl.loop.closed_loop import Simulator as SimulatorProtocol
from afl.loop.stubs import StubDetector, StubEvaluator
from afl.tracking import InMemoryTracker
from afl.utils.seed import rng

T0 = datetime(2023, 1, 1)


def anchor(n: int = 2_000, senders: int = 60) -> list[Transaction]:
    """Two-sided traffic at a realistic scale, so an anchored run has somewhere to hide."""
    draw = rng(3).lognormal(9.0, 1.0, size=n)
    return [
        Transaction(
            txn_id=f"real{i}",
            ts=T0 + timedelta(hours=i),
            src=f"acct{i % senders}",
            dst=f"acct{(i * 7) % senders}" if i % 3 else f"M{i % 9}",
            amount=round(float(draw[i]), 2),
            rail=Rail.A2A,
            device_id=None,
        )
        for i in range(n)
        if f"acct{i % senders}" != f"acct{(i * 7) % senders}"
    ]


def loop(optimiser, simulator, rounds: int = 3) -> InMemoryTracker:
    tracker = InMemoryTracker()
    run_closed_loop(
        optimiser.bind(simulator), optimiser, StubDetector(), StubEvaluator(), rounds, tracker
    )
    return tracker


def test_it_satisfies_the_loop_protocol():
    assert isinstance(MultiVectorOptimiser(backend="random"), Optimiser)
    sim = Simulator(seed=1, n_entities=60, n_background=0, n_episodes=2)
    assert isinstance(MultiVectorOptimiser(backend="random").bind(sim), SimulatorProtocol)


def test_one_round_covers_every_strong_vector():
    """The point of searching across vectors is finding the weakest surface, not grinding one."""
    sim = Simulator(seed=2, n_entities=200, n_background=0, n_episodes=6)
    optimiser = MultiVectorOptimiser(seed=2, backend="random", episodes_per_round=6)
    bound = optimiser.bind(sim)
    batch = bound.generate(optimiser.propose())
    assert {t.vector_id for t in batch.transactions} == set(STRONG_VECTORS)


@pytest.mark.parametrize("allocation", ["uniform", "search", "fitness"])
def test_budget_allocation_is_a_stated_choice(allocation):
    sim = Simulator(seed=3, n_entities=200, n_background=0, n_episodes=6)
    optimiser = MultiVectorOptimiser(
        seed=3, backend="random", allocation=allocation, episodes_per_round=6
    )
    loop(optimiser, sim, rounds=3)
    shares = optimiser.trials[-1].allocation
    assert set(shares) == set(STRONG_VECTORS)
    assert sum(shares.values()) == pytest.approx(1.0)
    if allocation == "uniform":
        assert len(set(round(v, 6) for v in shares.values())) == 1


def test_an_unknown_allocation_is_a_loud_error():
    with pytest.raises(ValueError, match="unknown allocation"):
        MultiVectorOptimiser(allocation="vibes")


def test_searched_parameters_stay_inside_the_declared_envelope():
    """The realism envelope belongs to the vector; the search may move inside it and no further."""
    from afl.attack.templates import registry

    optimiser = MultiVectorOptimiser(seed=4, backend="random")
    for _ in range(8):
        proposal = optimiser.propose()
        for vector_id, knobs in proposal.params["vectors"].items():
            space = registry.get(vector_id).search_space
            for key, bounds in space.items():
                assert bounds["low"] <= knobs[key] <= bounds["high"], f"{vector_id}.{key} escaped"
        optimiser.update([])


def test_an_off_anchor_batch_is_rejected_before_it_can_score():
    """The gate, and the reason it is a gate rather than a penalty."""
    real = anchor()
    unanchored = Simulator(seed=5, n_entities=60, n_background=0, n_episodes=4)
    optimiser = MultiVectorOptimiser(seed=5, backend="random", episodes_per_round=4, anchor=real)
    loop(optimiser, unanchored, rounds=2)

    assert optimiser.rejected == 2, "an off-scale, off-namespace batch has to fail the audit"
    assert all(t.rejected for t in optimiser.trials)
    assert all(t.fitness == -1.0 for t in optimiser.trials), "a rejected batch must score worst"
    assert optimiser.best is None, "a rejected batch must never become the best trial"


def test_an_anchored_batch_passes_the_gate_and_can_score():
    real = anchor()
    envelope = AnchorEnvelope.measure(real, "stand-in")
    sim = Simulator(seed=6, n_entities=60, n_background=0, n_episodes=4, envelope=envelope)
    optimiser = MultiVectorOptimiser(seed=6, backend="random", episodes_per_round=4, anchor=real)
    bound = optimiser.bind(sim)

    batch = bound.generate(optimiser.propose())
    assert batch.transactions
    assert optimiser.observe_batch(
        batch
    ), "traffic generated inside the anchor must not be rejected"
    optimiser.update(batch.fraud_transactions[:5])
    assert optimiser.trials[-1].fitness > -1.0
    assert optimiser.best is not None


def test_evasion_is_attributed_back_to_the_vector_that_earned_it():
    """Budget can only follow success if success is per-vector."""
    sim = Simulator(seed=7, n_entities=200, n_background=0, n_episodes=6)
    optimiser = MultiVectorOptimiser(seed=7, backend="random", episodes_per_round=6)
    bound = optimiser.bind(sim)
    batch = bound.generate(optimiser.propose())
    s1_rows = [t for t in batch.transactions if t.vector_id == "S1"]
    optimiser.update(s1_rows)

    per_vector = optimiser.trials[-1].per_vector_evasion
    assert per_vector["S1"] > 0
    assert per_vector["S2"] == 0 and per_vector["S3"] == 0
