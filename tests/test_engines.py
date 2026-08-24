"""▲ A — the engines produce schema-valid, reproducible, labelled traffic.

The bar here is not "looks plausible". It is: every row validates, the same seed gives the same
batch, provenance is on every fraud row and on no legit row, and the realism gate is actually
capable of failing.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from afl.attack import actors, realism
from afl.attack.engines import drift, graph, velocity
from afl.attack.optimiser import AttackOptimiser
from afl.attack.simulator import DEFAULT_START, Simulator
from afl.attack.templates import registry
from afl.contract.schema import Rail, Transaction
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
            assert t.vector_id is None, "a legit row must never carry a family label"
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
        vector_id="M1",
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
    assert_valid_attack(rows, "M1")
    assert len(rows) == 20
    assert all(t.amount < 10_000 for t in rows), "threshold-aware pacing must respect the ceiling"


def test_velocity_bursts_are_tighter_than_the_gaps_between_them():
    rows = velocity.generate(
        rng=make_rng(3),
        run_id="r3",
        vector_id="S2",
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
        vector_id="S3",
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
    assert_valid_attack(rows, "S3")
    assert [t.is_fraud for t in rows] == [False] * 15 + [True] * 5
    # the post-event tail is what makes it detectable at all
    baseline = sum(t.amount for t in rows[:15]) / 15
    post = sum(t.amount for t in rows[15:]) / 5
    assert post > baseline


def test_drift_new_device_only_after_the_event():
    rows = drift.generate(
        rng=make_rng(5),
        run_id="r5",
        vector_id="S3",
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
#: The taxonomy, frozen. Ticket 01 exists so that these ids mean the same thing in a config
#: file, a test, a metric and a conversation — so they are pinned here, not derived.
TAXONOMY = {"S1", "S2", "S3", "C1", "C2", "C3", "M1", "M2", "M3"}


def test_all_nine_vectors_load():
    vectors = registry.load_vectors()
    assert len(vectors) == 9
    assert set(vectors) == TAXONOMY
    for spec in vectors.values():
        assert spec.engine in registry.ENGINES
        assert spec.maturity in registry.MATURITIES
        assert spec.level in registry.LEVELS
        assert spec.tier in registry.TIERS
        assert spec.status in registry.STATUSES
        assert spec.why, f"{spec.vector_id} has no stated reason to exist"


def test_nine_vectors_collapse_onto_three_engines():
    """The design's headline claim: breadth is near-free because the engines are shared."""
    assert {s.engine for s in registry.list_vectors()} == set(registry.ENGINES)


def test_the_three_taxonomy_levels_are_populated_and_never_flattened():
    by_level = {
        lvl: {s.vector_id for s in registry.list_vectors(level=lvl)} for lvl in registry.LEVELS
    }
    assert by_level["model-attack"] == {"M1"}, "the attack against our own model is its own level"
    assert by_level["enabler"] == {"C2", "M2"}
    assert set().union(*by_level.values()) == TAXONOMY
    assert sum(len(v) for v in by_level.values()) == len(TAXONOMY), "a vector sits at one level"


def test_the_strong_tier_is_the_three_the_loop_runs_on():
    assert {s.vector_id for s in registry.list_vectors(tier="strong")} == {"S1", "S2", "S3"}
    assert all(s.status == registry.BUILT for s in registry.list_vectors(tier="strong"))


def test_an_unfinished_vector_must_say_what_is_missing():
    """`status` without `gap` would let a half-built family pass as a finished one."""
    for spec in registry.list_vectors():
        if spec.status != registry.BUILT:
            assert spec.gap, f"{spec.vector_id} is {spec.status} but names no gap"
            assert "icket" in spec.gap, f"{spec.vector_id}'s gap names no ticket to close it"


def test_the_declared_holdout_exists_and_is_never_generated_by_the_red_side():
    from afl.evaluation.leave_one_attack_out import DEFAULT_HOLDOUT

    spec = registry.get(DEFAULT_HOLDOUT)
    assert spec.name.startswith("First-party"), "the holdout is first-party fraud, by decision"
    assert spec.level == "mechanism" and spec.tier == "mid"


def test_clamp_pulls_params_back_into_the_realism_envelope():
    knobs = registry.clamp("S1", {"n_sources": 9_999, "leak": -1.0})
    space = registry.get("S1").search_space
    assert knobs["n_sources"] == space["n_sources"]["high"]
    assert knobs["leak"] == space["leak"]["low"]


def test_unknown_vector_is_a_loud_error():
    with pytest.raises(KeyError, match="unknown vector"):
        registry.get("Z9")


# ── simulator ───────────────────────────────────────────────────────────────────
GENERATABLE = [s.vector_id for s in registry.list_vectors(generatable=True)]
PLANNED = [s.vector_id for s in registry.list_vectors(generatable=False)]


@pytest.mark.parametrize("vector_id", GENERATABLE)
def test_simulator_generates_every_generatable_vector(vector_id):
    sim = Simulator(seed=11, n_entities=120, n_background=200, n_episodes=2)
    batch = sim.generate(registry.get(vector_id).to_attack_params())
    assert_valid_attack(batch.transactions, vector_id)
    assert batch.fraud_transactions, f"{vector_id} produced no fraud"
    assert batch.transactions == sorted(batch.transactions, key=lambda t: t.ts)
    assert batch.params.vector_id == vector_id


def test_a_planned_vector_refuses_loudly_instead_of_generating_nothing(monkeypatch):
    """An attack family that silently emits nothing reads exactly like one the detector caught.

    Driven off a stand-in rather than off PLANNED, so the guard keeps being tested once every
    declared vector is built — an empty parametrize would skip and take the guard with it.
    """
    planned = replace(registry.get("S1"), status="planned", gap="stand-in, ticket 99")
    monkeypatch.setattr(registry, "get", lambda vid: planned)
    sim = Simulator(seed=11, n_entities=60, n_background=50, n_episodes=1)
    with pytest.raises(NotImplementedError, match="declared but not implemented"):
        sim.generate(planned.to_attack_params())


@pytest.mark.parametrize("vector_id", PLANNED)
def test_every_declared_planned_vector_refuses(vector_id):
    sim = Simulator(seed=11, n_entities=60, n_background=50, n_episodes=1)
    with pytest.raises(NotImplementedError, match="declared but not implemented"):
        sim.generate(registry.get(vector_id).to_attack_params())


def test_card_testing_settles_on_the_card_rail():
    """A card-testing vector that settles on UPI is not a card-testing vector."""
    sim = Simulator(seed=12, n_entities=120, n_background=100, n_episodes=1)
    batch = sim.generate(registry.get("S2").to_attack_params())
    assert {t.rail for t in batch.fraud_transactions} == {Rail.CARD}


# ── C1 and M2: bust-out, and the same ending on an account that was never real ──
def drift_batch(vector_id: str, seed: int = 3):
    sim = Simulator(seed=seed, n_entities=300, n_background=1500, n_episodes=4)
    return sim.generate(registry.get(vector_id).to_attack_params())


def history_of(batch, owner):
    return [t for t in batch.transactions if t.src == owner and not t.is_fraud]


def test_bust_out_spikes_visibly_against_its_own_tenure():
    """C1 is must-catch load, not a holdout. If it were subtle it would not be doing its job."""
    import numpy as np

    batch = drift_batch("C1")
    assert_valid_attack(batch.transactions, "C1")
    assert {t.rail for t in batch.fraud_transactions} == {Rail.CARD}
    for owner in sorted({t.src for t in batch.fraud_transactions}):
        spike = [t.amount for t in batch.fraud_transactions if t.src == owner]
        tenure = [t.amount for t in history_of(batch, owner)]
        assert len(tenure) >= 30, "a bust-out needs a long clean tenure behind it"
        assert np.mean(spike) > 3 * np.mean(tenure), "the spike has to be visible"


def test_synthetic_identity_busts_out_of_an_account_that_was_never_real():
    """C1 is a real old account going bad. M2's account never existed before the run minted it."""
    batch = drift_batch("M2")
    assert_valid_attack(batch.transactions, "M2")
    opened = {e.entity_id: e.opened_at for e in batch.entities}
    for owner in sorted({t.src for t in batch.fraud_transactions}):
        assert opened[owner] >= DEFAULT_START, "a fabricated identity is not decades old"
        # thin file: every row this account has is one the run synthesised
        assert all(t.attack_run_id for t in history_of(batch, owner))


def test_synthetic_identity_seasons_before_it_busts():
    batch = drift_batch("M2")
    for owner in sorted({t.src for t in batch.fraud_transactions}):
        rows = sorted((t for t in batch.transactions if t.src == owner), key=lambda t: t.ts)
        first_abuse = next(i for i, t in enumerate(rows) if t.is_fraud)
        assert first_abuse >= 10, "seasoning is what makes the account look usable"
        assert all(not t.is_fraud for t in rows[:first_abuse])


def test_seasoning_is_legit_but_still_traceable_to_its_run():
    """Labelling seasoning as fraud teaches the wrong thing; losing its run id loses the trail."""
    batch = drift_batch("M2")
    seasoning = [t for t in batch.transactions if not t.is_fraud and t.attack_run_id]
    assert seasoning
    assert all(t.vector_id is None for t in seasoning)
    assert {t.attack_run_id for t in seasoning} == {batch.run_id}


@pytest.mark.parametrize("vector_id", ["C1", "M2"])
def test_drift_families_are_realistic(vector_id):
    assert realism.check(drift_batch(vector_id)).violations == []


# ── C3: the instant-A2A relay. One hop, near-zero dwell, in ≈ out ───────────────
def c3_batch(seed: int = 9):
    sim = Simulator(seed=seed, n_entities=300, n_background=1500, n_episodes=4)
    return sim.generate(registry.get("C3").to_attack_params())


def c3_relays(batch):
    """One (inbound, outbound) pair per episode, keyed off the run id inside each txn_id."""
    episodes: dict[str, list] = {}
    for t in batch.fraud_transactions:
        episodes.setdefault(t.txn_id.split("-g")[0], []).append(t)
    for rows in episodes.values():
        relay = ({t.dst for t in rows} & {t.src for t in rows}).pop()
        yield [t for t in rows if t.dst == relay], [t for t in rows if t.src == relay]


def test_pass_through_moves_what_it_receives_and_holds_almost_nothing():
    """In ≈ out with near-zero dwell is the signature; a relay that keeps the money is not one."""
    batch = c3_batch()
    assert_valid_attack(batch.transactions, "C3")
    assert {t.rail for t in batch.fraud_transactions} == {Rail.A2A}
    for inbound, outbound in c3_relays(batch):
        ratio = sum(t.amount for t in outbound) / sum(t.amount for t in inbound)
        dwell = (min(t.ts for t in outbound) - max(t.ts for t in inbound)).total_seconds()
        assert 0.9 < ratio <= 1.0, f"pass-through ratio {ratio:.2f} is not a pass-through"
        assert 0 <= dwell < 600, f"dwell {dwell:.0f}s is a hold, not a relay"


def test_pass_through_is_one_hop_not_a_mule_network():
    """S1 is the multi-hop layering vector. If C3 grows hops the two stop being distinguishable."""
    for _, outbound in c3_relays(c3_batch()):
        assert len(outbound) == 1


def test_pass_through_exits_to_an_account_with_no_prior_inbound():
    batch = c3_batch()
    ever_paid = {t.dst for t in batch.transactions if not t.is_fraud}
    for _, outbound in c3_relays(batch):
        assert not ({t.dst for t in outbound} & ever_paid)


def test_pass_through_batches_are_realistic():
    assert realism.check(c3_batch()).violations == []


# ── C2: the APP scam. Allowed to be catchable — it is training load, not a holdout ──
def c2_batch(seed: int = 5):
    sim = Simulator(seed=seed, n_entities=300, n_background=1500, n_episodes=4)
    return sim.generate(registry.get("C2").to_attack_params())


def test_app_scam_pays_a_payee_the_victim_has_never_paid():
    """Payee novelty is the whole signal: the victim authorised it, so nothing else looks wrong."""
    batch = c2_batch()
    assert_valid_attack(batch.transactions, "C2")
    for victim in sorted({t.src for t in batch.fraud_transactions}):
        scam = {t.dst for t in batch.fraud_transactions if t.src == victim}
        paid_before = {t.dst for t in batch.transactions if t.src == victim and not t.is_fraud}
        assert scam and not (scam & paid_before), f"{victim} had already paid {scam & paid_before}"


def test_app_scam_keeps_the_victims_own_device_and_account():
    """The victim is real. Only the destination is hostile, so no device or operator tell."""
    batch = c2_batch()
    for victim in sorted({t.src for t in batch.fraud_transactions}):
        scam = [t for t in batch.fraud_transactions if t.src == victim]
        assert {t.device_id for t in scam} == {f"dev-{victim}"}


def test_app_scam_drains_in_minutes_on_upi():
    import numpy as np

    batch = c2_batch()
    assert {t.rail for t in batch.fraud_transactions} == {Rail.UPI}
    for victim in sorted({t.src for t in batch.fraud_transactions}):
        scam = sorted((t for t in batch.fraud_transactions if t.src == victim), key=lambda t: t.ts)
        drain_s = (scam[-1].ts - scam[0].ts).total_seconds()
        assert drain_s < 3600, "a scam that takes hours is not a rapid drain"
    # atypical for the payer, still ordinary for the rail
    legit = np.median([t.amount for t in batch.transactions if not t.is_fraud])
    scam_mean = np.mean([t.amount for t in batch.fraud_transactions])
    assert legit < scam_mean < 40 * legit


def test_app_scam_batches_are_realistic():
    assert realism.check(c2_batch()).violations == []


# ── M3: first-party fraud, the holdout family ───────────────────────────────────
# `user == fraudster`, so the tells every other family leaks must all be absent.
M3_SEED = 7


def m3_batch(seed: int = M3_SEED):
    sim = Simulator(seed=seed, n_entities=300, n_background=1500, n_episodes=4)
    return sim.generate(registry.get("M3").to_attack_params())


def owners(batch):
    return sorted({t.src for t in batch.fraud_transactions})


def test_first_party_labels_only_the_abuse_not_the_history():
    """Genuine history stays legit; only the abuse is fraud, as an investigator would call it."""
    batch = m3_batch()
    assert_valid_attack(batch.transactions, "M3")
    for owner in owners(batch):
        rows = sorted((t for t in batch.transactions if t.src == owner), key=lambda t: t.ts)
        first_abuse = next(i for i, t in enumerate(rows) if t.is_fraud)
        # a long genuine history has to precede the abuse, or it is not first-party
        assert sum(not t.is_fraud for t in rows[:first_abuse]) >= 20
        assert all(t.is_fraud for t in rows[first_abuse:] if t.attack_run_id)


def test_first_party_never_changes_device_or_operator():
    """No compromised device and no new operator — the two signals a takeover would leak."""
    batch = m3_batch()
    for owner in owners(batch):
        abuse = [t for t in batch.fraud_transactions if t.src == owner]
        prior = {t.device_id for t in batch.transactions if t.src == owner and not t.is_fraud}
        assert {t.device_id for t in abuse} <= prior, "abuse introduced an unseen device"
        assert {t.src for t in abuse} == {owner}, "abuse came from a second operator"


def test_first_party_pays_nobody_new():
    """A fresh beneficiary is a ring tell. The owner keeps paying the people they already pay."""
    batch = m3_batch()
    for owner in owners(batch):
        abuse = {t.dst for t in batch.fraud_transactions if t.src == owner}
        established = {t.dst for t in batch.transactions if t.src == owner and not t.is_fraud}
        assert abuse <= established, f"{owner} paid {abuse - established} for the first time"


def test_first_party_is_anomalous_only_against_its_own_baseline():
    """Elevated against the account's own history, ordinary against the population."""
    import numpy as np

    batch = m3_batch()
    population = np.median([t.amount for t in batch.transactions if not t.is_fraud])
    for owner in owners(batch):
        abuse = [t.amount for t in batch.fraud_transactions if t.src == owner]
        baseline = [t.amount for t in batch.transactions if t.src == owner and not t.is_fraud]
        assert np.mean(abuse) > np.mean(baseline), "no drift at all"
        assert np.mean(abuse) < 20 * population, "a population-level outlier is not first-party"


def test_first_party_is_not_caught_by_a_hand_rolled_rule():
    """If a one-line amount rule catches it, the holdout measures the rule, not generalisation."""
    import numpy as np

    batch = m3_batch()
    amounts = np.array([t.amount for t in batch.transactions])
    is_fraud = np.array([t.is_fraud for t in batch.transactions])
    threshold = np.quantile(amounts[~is_fraud], 0.99)  # a naive "big amount" rule at 1% FPR
    assert float((amounts[is_fraud] > threshold).mean()) < 0.25


def test_first_party_batches_are_realistic():
    assert realism.check(m3_batch()).violations == []


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
