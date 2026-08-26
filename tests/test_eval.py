"""■ B — the metrics say what we claim they say, and the splits do not leak.

Most of the ways this project could quietly become dishonest live in this file: a random split,
a held-out family that isn't held out, a metric measured at a threshold nobody agreed to.

The detector's own guarantees — the backend it ran on, honest tuning, accumulating retrains,
the weight on an evasion — live in `tests/test_detector.py`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.metrics import Action, DetectorScore
from afl.contract.schema import Rail, Transaction
from afl.data.splits import assert_no_leakage, holdout_vector, out_of_time_split
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import leave_one_attack_out as loao
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


def attack_pool(vids: tuple[str, ...], seed: int = 21, n_episodes: int = 4) -> list[Transaction]:
    """Background traffic plus one batch per vector — the input every carve-out starts from."""
    sim = Simulator(seed=seed, n_entities=120, n_background=400, n_episodes=n_episodes)
    pool: list[Transaction] = []
    for vid in vids:
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)
    return pool


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
    pool = attack_pool(("S1", "S2"), seed=23)
    matrix = sweep(pool, lambda: LGBMDetector(seed=23, params={"n_estimators": 20}))
    assert {f.held_out_vector for f in matrix} == {"S1", "S2"}
    assert all(f.outcome in loao.OUTCOMES for f in matrix)


# ── ticket 11: the guards, and what a fold is allowed to claim ───────────────────
def test_a_held_out_row_in_the_training_list_is_caught():
    """The obvious half of the carve-out, and the one a filter edit could lose silently."""
    rows = txns(40, fraud_every=4)
    leaked = rows[0].model_copy(update={"txn_id": "leak-1", "is_fraud": True, "vector_id": "M3"})
    assert loao.assert_family_held_out(rows, "M3")["leaked_rows"] == 0
    with pytest.raises(loao.GuardFailed, match="reached training"):
        loao.assert_family_held_out([*rows, leaked], "M3")


def test_a_held_out_row_in_the_replay_buffer_is_caught():
    """The half that actually goes wrong: the split stays clean and training does not.

    A row that once evaded is kept across rounds and weighted up. Nothing in the split changes
    when it comes back, so this is the leak that a split-side assertion cannot see.
    """
    rows = txns(120, fraud_every=6)
    detector = LGBMDetector(seed=11, params={"n_estimators": 15}).fit(rows)
    assert loao.assert_family_held_out(rows, "M3", detector)["leaked_rows"] == 0

    detector._replay = [
        rows[1].model_copy(update={"txn_id": "evaded-1", "is_fraud": True, "vector_id": "M3"})
    ]
    with pytest.raises(loao.GuardFailed, match="reached training"):
        loao.assert_family_held_out(rows, "M3", detector)


def test_a_detector_that_cannot_say_what_it_trained_on_fails_the_guard():
    """Unauditable is a failure, not a pass — the expensive way to lose a fold is quietly."""
    from afl.loop.stubs import StubDetector

    with pytest.raises(loao.GuardFailed, match="training_rows"):
        loao.assert_family_held_out(txns(10), "M3", StubDetector())


def test_the_embargo_has_to_survive_the_carve_out():
    def later(start):
        return [t.model_copy(update={"txn_id": f"h{t.txn_id}"}) for t in txns(20, start=start)]

    train = txns(20)
    intact = loao.assert_embargo_intact(train, later(T0 + timedelta(days=5)), timedelta(days=1))
    assert intact["gap_seconds"] > 86_400

    near = later(max(t.ts for t in train) + timedelta(hours=1))
    with pytest.raises(loao.GuardFailed, match="embargo did not survive"):
        loao.assert_embargo_intact(train, near, timedelta(days=1))


def test_every_legit_row_of_the_window_stays_in_the_holdout():
    window = txns(60, fraud_every=5)
    holdout = [t for t in window if t.is_fraud or t.txn_id != "t00003"]
    with pytest.raises(loao.GuardFailed, match="legit row"):
        loao.assert_haystack_intact(window, holdout)
    assert loao.assert_haystack_intact(window, window)["dropped"] == 0


def test_carving_a_fold_runs_all_three_guards_and_records_them():
    fold = loao.Fold.carve(attack_pool(("S1", "M3"), seed=31), "M3")
    assert set(fold.guards) == {"family", "embargo", "haystack"}
    assert fold.guards["family"]["leaked_rows"] == 0
    assert fold.guards["haystack"]["dropped"] == 0
    assert all(t.vector_id != "M3" for t in fold.train)
    assert {t.vector_id for t in fold.holdout if t.is_fraud} <= {"M3"}


def test_any_vector_can_be_named_as_the_holdout():
    """The config names the family; nothing in the harness is specific to M3."""
    pool = attack_pool(("S1", "S2", "M3"), seed=32)
    for vid in ("S1", "S2", "M3"):
        fold = loao.Fold.carve(pool, vid)
        assert all(t.vector_id != vid for t in fold.train)
        assert {t.vector_id for t in fold.holdout if t.is_fraud} <= {vid}

    cfg = yaml.safe_load(Path("config/eval/leave_one_attack_out.yaml").read_text())
    assert cfg["held_out_vector"] in registry.load_vectors()
    assert "folds" in cfg and "min_meaningful_positives" in cfg


def test_a_thin_fold_is_withheld_rather_than_scored_low():
    """Recall on eight rows moves 12.5 points per row. That is not a low score, it is noise."""
    fold = loao.Fold.carve(attack_pool(("S1", "M3"), seed=33), "M3")
    positives = [t for t in fold.holdout if t.is_fraud][:8]
    thin = loao.Fold(
        held_out_vector="M3",
        train=fold.train,
        holdout=[t for t in fold.holdout if not t.is_fraud] + positives,
        embargo=fold.embargo,
        guards=fold.guards,
    )
    detector = LGBMDetector(seed=33, params={"n_estimators": 20}).fit(thin.train)

    result = loao.run_fold(thin, detector, min_positives=30)
    assert result.outcome == loao.WITHHELD
    assert result.metrics is None, "a withheld number must not sit where a reader quotes it"
    assert result.withheld_metrics is not None
    assert "8 positives against a floor of 30" in result.reason


def test_a_fold_separable_by_one_field_measures_provenance_not_detection():
    """Ticket 07 and ticket 10 both landed here. The fold has to say it itself."""
    fold = loao.Fold.carve(attack_pool(("S1", "M3"), seed=34), "M3")
    detector = LGBMDetector(seed=34, params={"n_estimators": 20}).fit(fold.train)
    result = loao.run_fold(
        fold,
        detector,
        min_positives=1,
        separability={"trivially_separable": True, "worst": "sender_in_anchor", "score": 0.98},
    )
    assert result.outcome == loao.WITHHELD
    assert result.metrics is None
    assert "sender_in_anchor" in result.reason and "provenance" in result.reason


def test_a_fold_a_classifier_can_sort_by_provenance_is_withheld():
    """Ticket 07 measured this by hand at AUC 1.00 and it lived in a carry-out. It lives here now.

    In this fold "caught the fraud" and "spotted the synthetic row" are the same label — the
    carve-out drops the anchor's own fraud from the holdout — so a probe that makes the call
    easily leaves the detector's recall saying nothing about detection.
    """
    fold = loao.Fold.carve(attack_pool(("S1", "M3"), seed=39), "M3")
    detector = LGBMDetector(seed=39, params={"n_estimators": 20}).fit(fold.train)
    result = loao.run_fold(
        fold,
        detector,
        min_positives=1,
        provenance={"checked": True, "pr_auc": 0.97, "base_rate": 0.002, "separable": True},
    )
    assert result.outcome == loao.WITHHELD
    assert result.metrics is None and result.withheld_metrics is not None
    assert "provenance" in result.reason and "0.970" in result.reason


def test_the_provenance_bar_is_a_floor_or_a_multiple_of_the_base_rate():
    """A fold with three positives must not pass by having a base rate small enough to flatter."""
    assert loao.is_provenance_bound(0.97, base_rate=0.002)
    assert not loao.is_provenance_bound(0.30, base_rate=0.002)
    # at a fat base rate the floor is not the binding constraint any more
    assert not loao.is_provenance_bound(0.60, base_rate=0.30)
    assert loao.is_provenance_bound(0.99, base_rate=0.15)


def test_a_template_vector_cannot_claim_a_recall_number_for_its_family():
    fold = loao.Fold.carve(attack_pool(("S1", "M3"), seed=35), "M3")
    detector = LGBMDetector(seed=35, params={"n_estimators": 20}).fit(fold.train)
    result = loao.run_fold(
        fold, detector, min_positives=1, not_reportable="M3 is a `template` vector: no tell yet"
    )
    assert result.outcome == loao.WITHHELD
    assert result.withheld_metrics is not None and result.metrics is None


def test_a_fold_result_cannot_carry_numbers_it_is_not_allowed_to_quote():
    numbers = protocol.evaluate(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9]))
    with pytest.raises(ValueError, match="withheld_metrics"):
        loao.FoldResult(held_out_vector="M3", outcome=loao.WITHHELD, reason="thin", metrics=numbers)
    with pytest.raises(ValueError, match="has to say why"):
        loao.FoldResult(held_out_vector="M3", outcome=loao.SKIPPED)


def test_the_sweep_names_the_folds_it_could_not_run():
    """A fold that vanishes reads as 'not applicable' when it means 'we did not look'."""
    pool = attack_pool(("S1", "S2"), seed=36)
    matrix = sweep(
        pool,
        lambda: LGBMDetector(seed=36, params={"n_estimators": 20}),
        vectors=["S1", "S2", "C2"],
    )
    assert [f.held_out_vector for f in matrix] == ["S1", "S2", "C2"]
    absent = next(f for f in matrix if f.held_out_vector == "C2")
    assert absent.outcome == loao.SKIPPED
    assert "nothing to hold out" in absent.reason


def test_an_empty_holdout_is_skipped_rather_than_scored_as_zero():
    fold = loao.Fold.carve(attack_pool(("S1", "M3"), seed=37), "M3")
    barren = loao.Fold(
        held_out_vector="M3",
        train=fold.train,
        holdout=[t for t in fold.holdout if not t.is_fraud],
        embargo=fold.embargo,
        guards=fold.guards,
    )
    detector = LGBMDetector(seed=37, params={"n_estimators": 20}).fit(barren.train)
    result = loao.run_fold(barren, detector)
    assert result.outcome == loao.SKIPPED
    assert result.metrics is None and result.withheld_metrics is None
    assert "would read 0.0" in result.reason


def test_the_report_carries_the_config_and_seed_that_produced_it(tmp_path):
    fold = loao.Fold.carve(attack_pool(("S1", "M3"), seed=38), "M3")
    detector = LGBMDetector(seed=38, params={"n_estimators": 20}).fit(fold.train)
    report = loao.LeaveOneAttackOutReport(
        dataset="synthetic",
        seed=38,
        config={"held_out_vector": "M3", "folds": ["M3"], "source": "config/eval/x.yaml"},
        operating_point={"fixed_fpr": 0.01, "k": 100, "min_meaningful_positives": 30},
        folds=[loao.run_fold(fold, detector, min_positives=1)],
    )
    path = report.save(tmp_path)
    again = loao.LeaveOneAttackOutReport.load("synthetic", tmp_path)
    assert again.seed == 38 and again.config["held_out_vector"] == "M3"
    assert again.headline is not None and again.headline.held_out_vector == "M3"
    assert [f.to_dict() for f in again.folds] == [f.to_dict() for f in report.folds]

    stale = json.loads(path.read_text())
    stale["version"] = loao.LOAO_ARTEFACT_VERSION + 1
    path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="artefact version"):
        loao.LeaveOneAttackOutReport.load("synthetic", tmp_path)


def test_a_report_with_no_folds_at_all_is_refused():
    with pytest.raises(ValueError, match="no folds"):
        loao.LeaveOneAttackOutReport(
            dataset="synthetic", seed=1, config={}, operating_point={}, folds=[]
        )


# The decision policy's own guarantees — cost-derived bands, calibration, reason codes — live
# in `tests/test_decision.py`, which is where ticket 09's work landed.


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
