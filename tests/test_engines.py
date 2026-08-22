"""▲ A — the engines produce schema-valid, reproducible, labelled traffic.

The bar here is not "looks plausible". It is: every row validates, the same seed gives the same
batch, provenance is on every fraud row and on no legit row, and the realism gate is actually
capable of failing.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from afl.attack import actors, realism
from afl.attack.engines import drift, graph, velocity
from afl.attack.optimiser import AttackOptimiser
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.schema import Transaction
from afl.utils.seed import rng as make_rng

START = datetime(2024, 1, 1)
VICTIMS = [f"v{i:03d}" for i in range(30)]
MULES = [f"m{i:03d}" for i in range(10)]
CASHOUT = [f"c{i:03d}" for i in range(10)]


def assert_valid_attack(rows: list[Transaction], vector_id: str) -> None:
    assert rows, "engine produced an empty episode"
    for t in rows:
        assert t.amount > 0
        assert t.src != t.dst
        if t.is_fraud:
            assert t.vector_id == vector_id and t.attack_run_id
        else:
            assert t.vector_id is None and t.attack_run_id is None
    assert len({t.txn_id for t in rows}) == len(rows)


# ── engines ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("motif", graph.MOTIFS)
def test_graph_engine_every_motif(motif):
    rows = graph.generate(
        rng=make_rng(1),
        run_id="r1",
        vector_id="S1",
        actor=actors.MULE,
        start_ts=START,
        params={
            "motif": motif,
            "n_sources": 5,
            "n_hops": 3,
            "split_ratio": 0.5,
            "hold_time_s": 600,
            "leak": 0.05,
        },
        victim_pool=VICTIMS,
        mule_pool=MULES,
        cashout_pool=CASHOUT,
    )
    assert_valid_attack(rows, "S1")
    assert rows == sorted(rows, key=lambda t: t.ts), "money must move forward in time"


def test_graph_engine_rejects_unknown_motif():
    with pytest.raises(ValueError, match="unknown motif"):
        graph.generate(
            rng=make_rng(1),
            run_id="r1",
            vector_id="S1",
            actor=actors.MULE,
            start_ts=START,
            params={"motif": "teleport"},
            victim_pool=VICTIMS,
            mule_pool=MULES,
            cashout_pool=CASHOUT,
        )


def test_velocity_engine_stays_under_its_threshold():
    rows = velocity.generate(
        rng=make_rng(2),
        run_id="r2",
        vector_id="V2",
        actor=actors.MULE,
        start_ts=START,
        params={
            "n_txns": 20,
            "burst_size": 3,
            "burst_gap_s": 3600,
            "intra_burst_s": 60,
            "threshold": 10_000,
            "device_rotation": 0.5,
            "amount_jitter": 0.05,
        },
        src="m000",
        dst_pool=CASHOUT,
    )
    assert_valid_attack(rows, "V2")
    assert len(rows) == 20
    assert all(t.amount < 10_000 for t in rows), "threshold-aware pacing must respect the ceiling"


def test_velocity_bursts_are_tighter_than_the_gaps_between_them():
    rows = velocity.generate(
        rng=make_rng(3),
        run_id="r3",
        vector_id="V1",
        actor=actors.FRAUDSTER,
        start_ts=START,
        params={
            "n_txns": 12,
            "burst_size": 4,
            "burst_gap_s": 86_400,
            "intra_burst_s": 5,
            "threshold": None,
            "device_rotation": 0.0,
        },
        src="m001",
        dst_pool=CASHOUT,
    )
    gaps = [(b.ts - a.ts).total_seconds() for a, b in zip(rows, rows[1:], strict=False)]
    intra = [g for i, g in enumerate(gaps) if (i + 1) % 4 != 0]
    inter = [g for i, g in enumerate(gaps) if (i + 1) % 4 == 0]
    assert max(intra) < min(inter)


def test_drift_engine_labels_only_the_post_event_tail():
    rows = drift.generate(
        rng=make_rng(4),
        run_id="r4",
        vector_id="M1",
        actor=actors.FRAUDSTER,
        start_ts=START,
        params={
            "n_baseline": 15,
            "n_post": 5,
            "ramp": 0.0,
            "amount_shift": 6.0,
            "new_device": True,
            "dormancy_s": 3600,
        },
        src="v001",
        benign_dst_pool=CASHOUT,
        cashout_pool=CASHOUT,
    )
    assert_valid_attack(rows, "M1")
    assert [t.is_fraud for t in rows] == [False] * 15 + [True] * 5
    # the post-event tail is what makes it detectable at all
    baseline = sum(t.amount for t in rows[:15]) / 15
    post = sum(t.amount for t in rows[15:]) / 5
    assert post > baseline


def test_drift_new_device_only_after_the_event():
    rows = drift.generate(
        rng=make_rng(5),
        run_id="r5",
        vector_id="M1",
        actor=actors.FRAUDSTER,
        start_ts=START,
        params={"n_baseline": 10, "n_post": 4, "new_device": True, "dormancy_s": 0},
        src="v002",
        benign_dst_pool=CASHOUT,
        cashout_pool=CASHOUT,
    )
    assert len({t.device_id for t in rows[:10]}) == 1
    assert rows[10].device_id != rows[9].device_id


# ── registry ────────────────────────────────────────────────────────────────────
def test_all_nine_vectors_load():
    vectors = registry.load_vectors()
    assert len(vectors) == 9
    assert set(vectors) == {"S1", "S2", "S3", "V1", "V2", "V3", "M1", "M2", "M3"}
    for spec in vectors.values():
        assert spec.engine in registry.ENGINES
        assert spec.maturity in registry.MATURITIES
        assert spec.why, f"{spec.vector_id} has no stated reason to exist"


def test_clamp_pulls_params_back_into_the_realism_envelope():
    knobs = registry.clamp("S1", {"n_sources": 9_999, "leak": -1.0})
    space = registry.get("S1").search_space
    assert knobs["n_sources"] == space["n_sources"]["high"]
    assert knobs["leak"] == space["leak"]["low"]


def test_unknown_vector_is_a_loud_error():
    with pytest.raises(KeyError, match="unknown vector"):
        registry.get("Z9")


# ── simulator ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("vector_id", ["S1", "S2", "S3", "V1", "V2", "V3", "M1", "M2", "M3"])
def test_simulator_generates_every_vector(vector_id):
    sim = Simulator(seed=11, n_entities=120, n_background=200, n_episodes=2)
    batch = sim.generate(registry.get(vector_id).to_attack_params())
    assert_valid_attack(batch.transactions, vector_id)
    assert batch.fraud_transactions, f"{vector_id} produced no fraud"
    assert batch.transactions == sorted(batch.transactions, key=lambda t: t.ts)
    assert batch.params.vector_id == vector_id


def test_same_seed_same_batch():
    params = registry.get("S1").to_attack_params()
    a = Simulator(seed=99, n_entities=100, n_background=150, n_episodes=2).generate(params)
    b = Simulator(seed=99, n_entities=100, n_background=150, n_episodes=2).generate(params)
    assert a.seed == b.seed
    assert a.model_dump_json() == b.model_dump_json()


def test_different_seed_different_batch():
    params = registry.get("S1").to_attack_params()
    a = Simulator(seed=1, n_entities=100, n_background=150, n_episodes=2).generate(params)
    b = Simulator(seed=2, n_entities=100, n_background=150, n_episodes=2).generate(params)
    assert a.model_dump_json() != b.model_dump_json()


# ── realism ─────────────────────────────────────────────────────────────────────
def test_realism_passes_a_normal_batch():
    sim = Simulator(seed=3, n_entities=200, n_background=400, n_episodes=3)
    report = realism.check(sim.generate(registry.get("S1").to_attack_params()))
    assert report.ok, report.violations
    assert report.penalty < 1.0


def test_realism_can_actually_fail():
    """A gate that never fails is decoration."""
    sim = Simulator(seed=3, n_entities=200, n_background=400, n_episodes=1)
    batch = sim.generate(registry.get("S1").to_attack_params())
    for t in batch.transactions:
        if t.is_fraud:
            t.dst = t.src  # self-transfer: impossible on any real rail
    report = realism.check(batch)
    assert not report.ok
    assert report.penalty == 1.0
    assert "self_transfer" in report.violations


# ── optimiser ───────────────────────────────────────────────────────────────────
def test_optimiser_proposals_stay_inside_the_search_space():
    opt = AttackOptimiser(vector_id="S1", seed=5, backend="random")
    space = registry.get("S1").search_space
    for _ in range(10):
        params = opt.propose()
        for k, bounds in space.items():
            assert bounds["low"] <= params.params[k] <= bounds["high"]
        opt.update([])


def test_optimiser_scores_evasion_against_realism():
    sim = Simulator(seed=7, n_entities=150, n_background=200, n_episodes=2)
    opt = AttackOptimiser(vector_id="S1", seed=7, backend="random", lambda_realism=0.5)
    bound = opt.bind(sim)
    batch = bound.generate(opt.propose())
    evasions = batch.fraud_transactions[:3]
    opt.update(evasions)

    trial = opt.trials[-1]
    assert trial.n_fraud == len(batch.fraud_transactions)
    assert trial.evasion_rate == pytest.approx(len(evasions) / trial.n_fraud)
    assert trial.fitness == pytest.approx(trial.evasion_rate - 0.5 * trial.realism_penalty)
