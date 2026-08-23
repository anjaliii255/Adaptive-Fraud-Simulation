"""■ B — the metrics say what we claim they say, and the splits do not leak.

Most of the ways this project could quietly become dishonest live in this file: a random split,
a held-out family that isn't held out, a metric measured at a threshold nobody agreed to.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.metrics import Action, DetectorScore
from afl.contract.schema import Rail, Transaction
from afl.data.splits import assert_no_leakage, holdout_vector, out_of_time_split
from afl.defend.decision import CostModel, DecisionPolicy
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import protocol, three_system
from afl.evaluation.leave_one_attack_out import LeaveOneAttackOut, make_splits, sweep
from afl.fidelity import level1_statistical, level2_structural, privacy, scorecard

T0 = datetime(2024, 1, 1)


def txns(n: int, fraud_every: int = 10, start: datetime = T0) -> list[Transaction]:
    return [
        Transaction(
            txn_id=f"t{i:05d}",
            ts=start + timedelta(hours=i),
            src=f"s{i % 7}",
            dst=f"d{i % 5}",
            amount=100.0 + i,
            rail=Rail.A2A,
            is_fraud=i % fraud_every == 0,
            vector_id="S1" if i % fraud_every == 0 else None,
        )
        for i in range(n)
    ]


# ── the metrics ─────────────────────────────────────────────────────────────────
def test_perfect_and_inverted_ranking():
    y = np.array([0, 0, 0, 1, 1])
    assert protocol.pr_auc(y, np.array([0.1, 0.2, 0.3, 0.9, 0.95])) == pytest.approx(1.0)
    assert protocol.pr_auc(y, np.array([0.9, 0.8, 0.7, 0.1, 0.05])) < 0.5


def test_pr_auc_is_zero_when_a_class_is_missing():
    # not "1.0 because nothing was wrong" — a single-class set has no ranking to score
    assert protocol.pr_auc(np.zeros(10), np.random.rand(10)) == 0.0


def test_recall_at_fixed_fpr_respects_the_budget():
    y = np.array([0] * 100 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.6, 1.0, 10)])
    assert protocol.recall_at_fixed_fpr(y, scores, fpr=0.01) == pytest.approx(1.0)

    # a detector that ranks fraud below the legit tail cannot buy recall at a 1% budget
    hidden = np.concatenate([np.linspace(0.5, 1.0, 100), np.linspace(0.0, 0.4, 10)])
    assert protocol.recall_at_fixed_fpr(y, hidden, fpr=0.01) == 0.0


def test_precision_at_k_is_the_analyst_queue():
    y = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    assert protocol.precision_at_k(y, scores, k=2) == pytest.approx(1.0)
    assert protocol.precision_at_k(y, scores, k=4) == pytest.approx(0.5)


def test_metric_result_records_its_operating_point():
    y = np.array([0] * 50 + [1] * 5)
    result = protocol.evaluate(y, np.linspace(0, 1, 55), fixed_fpr=0.02, k=10, held_out_vector="M3")
    assert (result.fixed_fpr, result.k, result.held_out_vector) == (0.02, 10, "M3")
    assert result.n_positives == 5


def test_align_refuses_to_guess_at_missing_scores():
    rows = txns(5)
    scores = [DetectorScore(txn_id=t.txn_id, score=0.5, action=Action.ALLOW) for t in rows[:3]]
    with pytest.raises(ValueError, match="unscored"):
        protocol.align(rows, scores)


def test_operational_rates_separate_friction_from_blocking():
    rows = txns(20, fraud_every=2)
    scores = [
        DetectorScore(
            txn_id=t.txn_id,
            score=0.9 if t.is_fraud else 0.1,
            action=Action.ALLOW if t.is_fraud else Action.DECLINE,
        )
        for t in rows
    ]
    rates = protocol.operational_rates(rows, scores)
    assert rates["evasion_rate"] == 1.0
    assert rates["false_decline_rate"] == 1.0
    assert rates["amount_evaded"] == sum(t.amount for t in rows if t.is_fraud)


# ── the splits ──────────────────────────────────────────────────────────────────
def test_out_of_time_split_is_chronological_with_an_embargo():
    rows = txns(200)
    train, test = out_of_time_split(rows, train_frac=0.7, embargo_days=1.0)
    assert train and test
    assert max(t.ts for t in train) < min(t.ts for t in test)
    assert min(t.ts for t in test) - max(t.ts for t in train) > timedelta(hours=23)
    assert len(train) + len(test) < len(rows), "the embargo gap must actually drop rows"
    assert_no_leakage(train, test)


def test_leakage_guard_catches_a_random_split():
    rows = txns(100)
    bad_train, bad_test = rows[::2], rows[1::2]
    with pytest.raises(AssertionError, match="temporal leakage"):
        assert_no_leakage(bad_train, bad_test)


def test_holdout_vector_keeps_the_haystack():
    rows = txns(100, fraud_every=5)
    seen, held = holdout_vector(rows, "S1")
    assert all(t.vector_id != "S1" for t in seen)
    assert any(t.is_fraud for t in held)
    assert any(not t.is_fraud for t in held), "an FPR with no negatives is not an FPR"


def test_make_splits_removes_the_family_and_the_future():
    sim = Simulator(seed=21, n_entities=120, n_background=300, n_episodes=6)
    pool = []
    for vid in ("S1", "S2", "M3"):
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)

    train, holdout = make_splits(pool, held_out_vector="M3")
    assert all(t.vector_id != "M3" for t in train)
    assert {t.vector_id for t in holdout if t.is_fraud} == {"M3"}
    assert max(t.ts for t in train) < min(t.ts for t in holdout)


def test_empty_holdout_is_announced_not_scored_as_zero(caplog):
    """The out-of-time cut can land after every episode of the held-out family."""
    sim = Simulator(seed=21, n_entities=120, n_background=300, n_episodes=2)
    pool = []
    for vid in ("S1", "M3"):
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)

    with caplog.at_level("WARNING"):
        evaluator, _ = LeaveOneAttackOut.from_pool(pool, held_out_vector="M3")
    if not any(t.is_fraud for t in evaluator.holdout):
        assert "measure nothing" in caplog.text


def test_evaluator_never_returns_the_training_number():
    sim = Simulator(seed=22, n_entities=120, n_background=300, n_episodes=2)
    pool = []
    for vid in ("S1", "M3"):
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)

    evaluator, train = LeaveOneAttackOut.from_pool(pool, held_out_vector="M3")
    assert not ({t.txn_id for t in train} & {t.txn_id for t in evaluator.holdout})

    detector = LGBMDetector(seed=22, params={"n_estimators": 30}).fit(train)
    result = evaluator.leave_one_attack_out(detector)
    assert result.held_out_vector == "M3"
    assert 0.0 <= result.pr_auc <= 1.0
    assert len(evaluator.history) == 1


def test_sweep_reports_one_row_per_family():
    sim = Simulator(seed=23, n_entities=120, n_background=300, n_episodes=2)
    pool = []
    for vid in ("S1", "S2"):
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)

    matrix = sweep(pool, lambda: LGBMDetector(seed=23, params={"n_estimators": 20}))
    assert set(matrix) <= {"S1", "S2"}
    for vid, result in matrix.items():
        assert result.held_out_vector == vid


# ── decision policy ─────────────────────────────────────────────────────────────
def test_graded_actions_ladder_up_with_score():
    policy = DecisionPolicy()
    ladder = [policy.act(s) for s in (0.05, 0.30, 0.60, 0.80, 0.95)]
    assert ladder == [Action.ALLOW, Action.STEP_UP, Action.HOLD, Action.REVIEW, Action.DECLINE]


def test_cost_mode_declines_a_large_likely_fraud_and_allows_a_small_unlikely_one():
    policy = DecisionPolicy(mode="cost", costs=CostModel())
    assert policy.act(0.99, amount=50_000) in (Action.DECLINE, Action.REVIEW)
    assert policy.act(0.001, amount=5.0) is Action.ALLOW


def test_calibration_hits_the_target_fpr():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 1_000 + [1] * 50)
    scores = np.concatenate([rng.uniform(0, 0.6, 1_000), rng.uniform(0.4, 1.0, 50)])
    policy = DecisionPolicy().calibrate_to_fpr(scores, labels, target_fpr=0.01)
    realised = float((scores[labels == 0] >= policy.decline_at).mean())
    assert realised == pytest.approx(0.01, abs=0.005)


# ── the detector remembers ──────────────────────────────────────────────────────
def test_retrain_accumulates_rather_than_forgetting():
    """The loop must not reduce the detector to whatever it saw most recently."""
    sim = Simulator(seed=26, n_entities=100, n_background=250, n_episodes=2)
    first = sim.generate(registry.get("S1").to_attack_params())
    second = sim.generate(registry.get("S2").to_attack_params())

    detector = LGBMDetector(seed=26, params={"n_estimators": 20})
    detector.fit(first.transactions)
    detector.retrain(second, evasions=second.fraud_transactions[:3])

    corpus = {t.txn_id for t in detector._corpus}
    assert {t.txn_id for t in first.transactions} <= corpus
    assert {t.txn_id for t in second.transactions} <= corpus


# ── three systems ───────────────────────────────────────────────────────────────
def test_smote_stays_inside_its_own_family():
    rows = txns(120, fraud_every=4)
    synth = three_system.smote_transactions(rows, ratio=1.0, seed=1)
    assert synth
    assert all(t.is_fraud and t.vector_id == "S1" for t in synth)
    assert all(t.attack_run_id == "smote" for t in synth)
    assert len({t.txn_id for t in synth}) == len(synth)


def test_three_systems_share_one_holdout_and_one_operating_point():
    sim = Simulator(seed=24, n_entities=120, n_background=400, n_episodes=2)
    pool = []
    for vid in ("S1", "S2", "M3"):
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)

    from afl.attack.optimiser import AttackOptimiser

    optimiser = AttackOptimiser(vector_id="S1", seed=24, backend="random")
    results = three_system.run_three_systems(
        pool=pool,
        detector_factory=lambda: LGBMDetector(seed=24, params={"n_estimators": 25}),
        simulator=optimiser.bind(sim),
        optimiser=optimiser,
        held_out_vector="M3",
        rounds=2,
    )
    assert [r.name for r in results] == ["A_baseline", "B_smote", "C_adaptive"]
    assert len({r.metrics.held_out_vector for r in results}) == 1
    assert len({r.metrics.fixed_fpr for r in results}) == 1
    assert "vs_baseline" in three_system.lift(results)
    assert three_system.to_markdown(results).startswith("| system")


def test_system_c_refuses_to_generate_the_family_it_is_graded_on():
    from afl.attack.optimiser import AttackOptimiser

    sim = Simulator(seed=25, n_entities=80, n_background=200, n_episodes=1)
    pool = sim.generate(registry.get("M3").to_attack_params()).transactions
    with pytest.raises(ValueError, match="held-out family"):
        three_system.run_three_systems(
            pool=pool,
            detector_factory=lambda: LGBMDetector(seed=25),
            simulator=sim,
            optimiser=AttackOptimiser(vector_id="M3", seed=25, backend="random"),
            held_out_vector="M3",
            rounds=1,
        )


# ── fidelity ────────────────────────────────────────────────────────────────────
def test_level1_ranks_a_copy_above_noise():
    rows = txns(400, fraud_every=8)
    copy = [t.model_copy(update={"txn_id": f"c{t.txn_id}"}) for t in rows]
    rng = np.random.default_rng(0)
    noise = [
        t.model_copy(update={"txn_id": f"n{i}", "amount": float(rng.uniform(1, 9_000))})
        for i, t in enumerate(rows)
    ]
    assert (
        level1_statistical.report(rows, copy)["score"]
        > level1_statistical.report(rows, noise)["score"]
    )


def test_level2_sees_structure_that_level1_cannot():
    rows = txns(300, fraud_every=6)
    rng = np.random.default_rng(1)
    # same amounts and timestamps, edges rewired: level 1 is blind to this, level 2 must not be
    rewired = [
        t.model_copy(
            update={
                "txn_id": f"r{i}",
                "src": f"x{int(rng.integers(0, 50))}",
                "dst": f"y{int(rng.integers(0, 50))}",
            }
        )
        for i, t in enumerate(rows)
    ]
    l1 = level1_statistical.report(rows, rewired)["score"]
    l2 = level2_structural.report(rows, rewired)["score"]
    assert l1 > l2


def test_privacy_flags_a_verbatim_copy():
    rows = txns(300, fraud_every=6)
    train, holdout = out_of_time_split(rows, train_frac=0.7, embargo_days=1.0)
    report = privacy.report(
        train, holdout, [t.model_copy(update={"txn_id": f"c{t.txn_id}"}) for t in train]
    )
    assert report["dcr"]["identical_share"] > 0.0
    assert report["flags"]


def test_scorecard_gates_on_utility_not_on_histograms():
    card = scorecard.Scorecard(
        levels={
            "level1": {"score": 0.95},
            "level2": {"score": 0.90},
            "level3": {
                "score": 0.2,
                "tstr": {"tstr_gap": 0.40},
                "augmentation": {"recall_lift": -0.05},
            },
        }
    )
    judged = scorecard._judge(card)
    assert judged.verdict == scorecard.FAIL
    assert any("TSTR gap" in r for r in judged.reasons)


def test_scorecard_says_when_it_skipped_the_bar():
    rows = txns(200, fraud_every=5)
    card = scorecard.build(real=rows, synth=rows[:100])
    assert card.verdict == scorecard.WARN
    assert any("skipped" in r for r in card.reasons)
    assert "level3" not in card.levels
