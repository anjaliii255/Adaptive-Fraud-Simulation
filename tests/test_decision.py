"""■ B — a score becomes an action an analyst can act on, and the action has a price behind it.

Ticket 09. What could go wrong here is quieter than a wrong metric, which is why it gets its own
file: a threshold nobody chose, a cost model whose numbers came from nowhere, an action taken on
a "probability" that is not one, or a flagged customer with no explanation anyone can argue with.
None of those show up in PR-AUC. All of them show up in the ops queue on Monday.

The four properties this file exists to hold:

  * bands are **derived** from the cost model, and the derivation agrees with pricing each
    transaction individually — one policy, two readings, same answer;
  * the number the cost model acts on is a **probability**, and it never reaches the field the
    detection metrics are computed from, so the decision layer cannot move one of them;
  * a **flagged** transaction always carries reason codes, and a global explanation says so in
    its own text;
  * `evaded` still means `allow` and nothing else.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import yaml

from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.metrics import EVASION_ACTIONS, Action, DetectorScore
from afl.contract.schema import AttackBatch, AttackParams, Rail, Transaction
from afl.defend import explain
from afl.defend.calibration import MIN_POSITIVES, ScoreCalibrator
from afl.defend.decision import (
    SEVERITY,
    UNREACHABLE,
    CostModel,
    DecisionPolicy,
    DominatedAction,
    action_mix,
    assert_one_operating_point,
    cost_model_for,
    median_amount,
    policy_from_config,
    total_cost,
)
from afl.defend.models.anomaly import AnomalyDetector, EnsembleDetector
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import protocol

T0 = datetime(2024, 1, 1)
FAST = {"n_estimators": 25, "num_leaves": 7}
COSTS_CONFIG = yaml.safe_load(open("config/costs/default.yaml"))
LGBM_CONFIG = yaml.safe_load(open("config/defend/lgbm.yaml"))
EVAL_CONFIG = yaml.safe_load(open("config/eval/leave_one_attack_out.yaml"))


def txns(n: int, fraud_every: int = 10, start: datetime = T0) -> list[Transaction]:
    return [
        Transaction(
            txn_id=f"t{i:05d}",
            ts=start + timedelta(hours=i),
            src=f"s{i % 7}",
            dst=f"d{i % 5}",
            amount=100.0 + (i % 50) * (30.0 if i % fraud_every == 0 else 1.0),
            rail=Rail.A2A,
            device_id=f"dev{i % 3}",
            is_fraud=i % fraud_every == 0,
            vector_id="S1" if i % fraud_every == 0 else None,
        )
        for i in range(n)
    ]


def as_batch(rows: list[Transaction]) -> AttackBatch:
    return AttackBatch(
        run_id="decision-test",
        params=AttackParams(vector_id="eval", engine="none"),
        transactions=list(rows),
        seed=0,
    )


# ── bands come from the cost model, not from anybody's judgement ────────────────
def test_the_shipped_policy_has_no_hand_set_bands_left_to_edit():
    """The regression this whole ticket is: four numbers in a config, calibrated to nothing."""
    decision = LGBM_CONFIG["decision"]
    typed = [k for k in ("step_up_at", "hold_at", "review_at", "decline_at") if k in decision]
    assert not typed, f"config/defend/lgbm.yaml still carries hand-set bands: {typed}"
    assert decision["mode"] == "cost"

    policy = policy_from_config(decision, cost_model_for(COSTS_CONFIG, txns(20)))
    assert policy.bands_source == "derived from the cost model"


def test_the_derived_ladder_and_the_per_transaction_price_agree():
    """One policy, two readings. If the bands disagree with the argmin, one of them is a fiction.

    Points within a whisker of a band edge are skipped: there the two actions cost exactly the
    same, the choice is genuinely indifferent, and only floating point decides it.
    """
    costs = CostModel()
    for amount in (5.0, 100.0, 12_345.0, 1e6):
        policy = DecisionPolicy(mode="threshold", costs=costs, reference_amount=amount)
        edges = list(policy.band_edges.values())
        for p in np.linspace(0.0, 1.0, 2_001):
            if any(abs(float(p) - e) < 1e-9 for e in edges):
                continue
            assert policy.act_on_probability(float(p), amount) is costs.action_for(
                float(p), amount
            ), f"band ladder and expected-cost argmin disagree at p={p} on {amount}"


def test_the_ladder_is_monotone_in_the_score():
    """More risk never buys a *softer* action. The graded part of 'graded decisions'."""
    policy = DecisionPolicy(reference_amount=10_000.0)
    seen = [SEVERITY.index(policy.act(float(p), amount=10_000.0)) for p in np.linspace(0, 1, 500)]
    assert seen == sorted(seen)
    assert seen[0] == 0, "a zero-risk transaction has to be allowed"
    assert seen[-1] == len(SEVERITY) - 1, "a certain fraud has to be declined"


def test_a_small_payment_is_not_worth_an_analyst_and_the_policy_says_which_bands_closed():
    """Graded means the ladder can be shorter on a cheap payment, not that a rung was forgotten."""
    costs = CostModel()
    assert costs.dominated_at(5.0) == [Action.HOLD, Action.REVIEW]
    assert costs.dominated_at(50_000.0) == []

    policy = DecisionPolicy(reference_amount=5.0)
    assert policy.review_at == UNREACHABLE
    assert policy.to_dict()["unreachable_bands"] == ["hold", "review"]


def test_a_cost_model_that_prices_a_rung_out_of_existence_is_refused():
    """The house numbers had this defect: a hold cost 3x a step-up and stopped less fraud."""
    with pytest.raises(DominatedAction, match="hold is dominated by step_up"):
        CostModel(step_up_efficacy=0.80, hold_efficacy=0.50)  # costs more, catches less
    with pytest.raises(DominatedAction, match="review_efficacy"):
        CostModel(review_efficacy=1.0)  # nothing left for a decline to add


# ── the parameters live in config, with a rationale that is enforced ────────────
def test_every_cost_parameter_in_the_shipped_config_states_why():
    model = CostModel.from_config(COSTS_CONFIG)
    priced = set(model.to_dict()) - {"source", "rationale", "unit_amount"}
    assert priced <= set(model.rationale)
    assert all(
        len(model.rationale[name]) > 40 for name in model.rationale
    ), "a one-word `why` is not a rationale"


def test_a_cost_parameter_with_no_rationale_will_not_load():
    """A YAML comment can be deleted and nothing notices; a required field cannot."""
    blank = {**COSTS_CONFIG, "review_cost": {"value": 0.05, "why": "  "}}
    with pytest.raises(ValueError, match="review_cost"):
        CostModel.from_config(blank)

    bare = {**COSTS_CONFIG, "false_decline_cost": 0.35}
    with pytest.raises(ValueError, match="false_decline_cost"):
        CostModel.from_config(bare)


def test_flat_costs_are_denominated_in_the_anchor_so_one_config_serves_both_anchors():
    """PaySim's median payment is 74,872 and AMLSim's is 157. An absolute cost cannot serve both."""
    paysim = cost_model_for(COSTS_CONFIG, [Transaction(**_row(74_871.94))])
    amlsim = cost_model_for(COSTS_CONFIG, [Transaction(**_row(156.71))])

    assert paysim.review_cost > amlsim.review_cost * 100, "the costs did not follow the anchor"
    at_median = {a.value: round(p, 6) for a, p in paysim.bands(paysim.unit_amount).items()}
    assert at_median == {a.value: round(p, 6) for a, p in amlsim.bands(amlsim.unit_amount).items()}


def test_the_amount_scale_is_the_median_and_not_the_mean():
    """PaySim's mean payment is 2.4x its median; pricing an analyst off the mean is off by that."""
    heavy = [Transaction(**_row(a)) for a in (10.0, 20.0, 30.0, 40.0, 1e9)]
    assert median_amount(heavy) == 30.0


def _row(amount: float) -> dict:
    return {
        "txn_id": f"x{amount}",
        "ts": T0,
        "src": "a",
        "dst": "b",
        "amount": amount,
        "rail": "a2a",
    }


# ── changing a cost parameter moves the action mix ─────────────────────────────
def test_changing_a_cost_parameter_visibly_moves_the_action_mix():
    """The acceptance criterion, and the reason the cost model is not decoration.

    A cheaper false decline should decline more; a dearer analyst should review less.
    """
    rows = txns(400, fraud_every=5)
    scores = np.linspace(0.0, 1.0, len(rows))

    def mix(costs: CostModel) -> dict[str, float]:
        policy = DecisionPolicy(costs=costs)
        return action_mix(
            [
                policy.decide(t.txn_id, float(s), amount=t.amount)
                for t, s in zip(rows, scores, strict=True)
            ]
        )

    house = mix(CostModel())
    cheap_decline = mix(CostModel(false_decline_cost=0.05))
    dear_analyst = mix(CostModel(review_cost=400.0))

    assert (
        cheap_decline["decline"] > house["decline"]
    ), "a false decline got 7x cheaper and the policy declined no more of the traffic"
    assert (
        dear_analyst["review"] < house["review"]
    ), "an analyst got 100x dearer and the review queue did not shrink"
    assert sum(house.values()) == pytest.approx(1.0)


def test_a_large_likely_fraud_is_declined_and_a_small_unlikely_one_is_allowed():
    """The two ends of the ladder, in the terms the ticket states them in.

    Moved from `tests/test_eval.py` when the decision layer got its own file. It reads as
    obvious, which is the point — a cost model that got this wrong would be wrong everywhere.
    """
    policy = DecisionPolicy()
    assert policy.act(0.99, amount=50_000) is Action.DECLINE
    assert policy.act(0.001, amount=5.0) is Action.ALLOW


def test_a_bigger_amount_earns_more_friction_at_the_same_score():
    """The whole reason the decision is per transaction and not per threshold."""
    policy = DecisionPolicy()
    small = policy.act(0.05, amount=5.0)
    large = policy.act(0.05, amount=500_000.0)
    assert SEVERITY.index(large) > SEVERITY.index(small)


# ── the score reaching the cost model is a probability ─────────────────────────
def test_calibration_does_not_move_a_single_reported_metric():
    """Platt is strictly monotone, so PR-AUC, recall@FPR and precision@k are rank-invariant.

    This is the property that let the decision layer change without disturbing ticket 08's
    committed reference numbers. If it ever fails, every baseline in `artifacts/detector/` is
    measuring something else.
    """
    rng = np.random.default_rng(0)
    labels = np.array([0] * 2_000 + [1] * 60)
    raw = np.concatenate([rng.beta(2, 40, 2_000), rng.beta(6, 12, 60)])

    calibrator = ScoreCalibrator(method="sigmoid").fit(raw, labels)
    assert calibrator.fitted
    calibrated = calibrator(raw)

    before = protocol.evaluate(labels, raw, fixed_fpr=0.01, k=100)
    after = protocol.evaluate(labels, calibrated, fixed_fpr=0.01, k=100)
    assert (before.pr_auc, before.recall_at_fixed_fpr, before.precision_at_k) == (
        after.pr_auc,
        after.recall_at_fixed_fpr,
        after.precision_at_k,
    )
    assert np.all(np.diff(calibrated[np.argsort(raw)]) >= -1e-12), "the map is not monotone"


def test_calibration_actually_improves_the_probability_it_reports():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 4_000 + [1] * 200)
    # a ranking score that is systematically far too confident — what a weighted tree emits
    raw = np.clip(np.concatenate([rng.beta(2, 6, 4_000), rng.beta(9, 3, 200)]), 0, 1)
    calibrator = ScoreCalibrator(method="sigmoid").fit(raw, labels)
    assert calibrator.reliability["after"]["brier"] < calibrator.reliability["before"]["brier"]
    assert calibrator.reliability["after"]["ece"] < calibrator.reliability["before"]["ece"]


def test_too_few_positives_leaves_the_calibrator_an_identity_and_says_so():
    """A curve fitted to nine positives is noise with a slope. The fallback is the identity."""
    labels = np.array([0] * 500 + [1] * (MIN_POSITIVES - 1))
    raw = np.linspace(0, 1, len(labels))
    calibrator = ScoreCalibrator(method="sigmoid").fit(raw, labels)
    assert not calibrator.fitted
    assert "below the floor" in calibrator.note
    assert calibrator.probability(0.42) == pytest.approx(0.42)


def test_recalibrating_does_not_stack_two_maps():
    """A second fit on already-calibrated scores squashes every probability twice, silently."""
    rng = np.random.default_rng(2)
    labels = np.array([0] * 2_000 + [1] * 80)
    raw = np.concatenate([rng.beta(2, 30, 2_000), rng.beta(6, 10, 80)])

    policy = DecisionPolicy()
    policy.fit_calibrator(raw, labels)
    once = policy.probability(0.3)

    policy.reset_calibration()
    assert policy.probability(0.3) == pytest.approx(0.3), "reset did not return to the identity"
    policy.fit_calibrator(raw, labels)
    assert policy.probability(0.3) == pytest.approx(once)


def test_the_reported_score_is_the_detectors_own_and_not_the_calibrated_probability():
    """The seam that makes the monotonicity argument unnecessary.

    Reporting the calibrated probability was tried first and it moved a metric: float64 rounds
    `1/(1+exp(-z))` to exactly 1.0 past z ~ 37, and on PaySim's test window the fitted map
    collapsed 129 distinct top-200 scores into one, taking precision@100 on the stock-params
    control from 0.14 to 0.06. The calibrated probability chooses the action; it does not reach
    the field the metrics read.
    """
    rng = np.random.default_rng(3)
    labels = np.array([0] * 1_500 + [1] * 60)
    raw = np.concatenate([rng.beta(2, 30, 1_500), rng.beta(6, 10, 60)])
    policy = DecisionPolicy().fit_calibrator(raw, labels)

    decided = policy.decide("t1", 0.30, amount=1_000.0)
    assert decided.score == pytest.approx(0.30), "the calibrator leaked into the reported score"
    assert policy.probability(0.30) != pytest.approx(0.30), "the calibrator did nothing at all"
    assert any(
        f"p={policy.probability(0.30):.4f}" in r for r in decided.reasons
    ), "the probability the action was taken on has to be visible somewhere"


def test_the_decision_layer_cannot_move_a_detection_metric():
    """Same fitted model, two decision policies, three metrics: identical to the last bit.

    The strong form of the property. It holds whatever the calibrator does, because the metrics
    never see the calibrator — which is why it is asserted on `DetectorScore.score` end to end
    rather than on the map in isolation.
    """
    rows = txns(900, fraud_every=6)
    train, holdout = rows[:700], rows[700:]
    detector = LGBMDetector(seed=17, params=FAST).fit(train)
    y = [int(t.is_fraud) for t in holdout]

    plain = protocol.evaluate_detector(detector, holdout, fixed_fpr=0.01, k=50)

    cal_y, cal_s = protocol.align(train, protocol.score_transactions(detector, train, "cal"))
    detector.policy.fit_calibrator(cal_s, cal_y)
    calibrated = protocol.evaluate_detector(detector, holdout, fixed_fpr=0.01, k=50)

    # an absurd but *valid* ladder — friction priced at nothing, so nearly everything is
    # flagged. Still a monotone ladder, or `DominatedAction` would refuse it.
    detector.policy = DecisionPolicy(
        costs=CostModel(step_up_cost=1e-9, hold_cost=1e-8, review_cost=1e-7)
    )
    wild = protocol.evaluate_detector(detector, holdout, fixed_fpr=0.01, k=50)
    assert action_mix(protocol.score_transactions(detector, holdout, "wild"))["allow"] < 0.5

    for other in (calibrated, wild):
        assert (other.pr_auc, other.recall_at_fixed_fpr, other.precision_at_k) == (
            plain.pr_auc,
            plain.recall_at_fixed_fpr,
            plain.precision_at_k,
        )
    assert sum(y) > 0, "a holdout with no positives would pass this vacuously"


def test_the_calibrated_probability_does_not_saturate_away_the_ordering():
    """`1/(1+exp(-z))` rounds to 1.0 past z ~ 37, and a fitted Platt slope reaches that easily."""
    rng = np.random.default_rng(9)
    labels = np.array([0] * 3_000 + [1] * 90)
    raw = np.concatenate([rng.beta(1.2, 900, 3_000), rng.beta(9, 1.4, 90)])
    calibrated = ScoreCalibrator(method="sigmoid").fit(raw, labels)(raw)

    top = np.sort(raw)[-200:]
    assert (
        len(set(calibrated[np.argsort(raw)[-200:]])) > len(set(top)) // 2
    ), "the map flattened the top of the distribution into a handful of values"
    assert calibrated.max() < 1.0 and calibrated.min() > 0.0


# ── the operating point is still one operating point ───────────────────────────
def test_cost_mode_refuses_a_second_operating_point():
    assert_one_operating_point(None, 0.01, mode="cost")
    with pytest.raises(ValueError, match="two operating points"):
        assert_one_operating_point(0.01, 0.01, mode="cost")


def test_the_shipped_config_names_exactly_one_operating_point():
    assert_one_operating_point(
        LGBM_CONFIG["decision"]["calibrate_to_fpr"],
        EVAL_CONFIG["fixed_fpr"],
        mode=LGBM_CONFIG["decision"]["mode"],
    )


def test_threshold_mode_still_hits_a_target_fpr_when_asked():
    """The old path is kept, not deleted — a team that wants an FPR target can still have one."""
    rng = np.random.default_rng(0)
    labels = np.array([0] * 1_000 + [1] * 50)
    scores = np.concatenate([rng.uniform(0, 0.6, 1_000), rng.uniform(0.4, 1.0, 50)])
    policy = DecisionPolicy(mode="threshold").calibrate_to_fpr(scores, labels, target_fpr=0.01)
    realised = float((scores[labels == 0] >= policy.decline_at).mean())
    assert realised == pytest.approx(0.01, abs=0.005)
    assert policy.bands_source.startswith("calibrated to")


# ── reason codes ───────────────────────────────────────────────────────────────
def _fitted_detector(seed: int = 11) -> tuple[LGBMDetector, list[Transaction]]:
    rows = txns(900, fraud_every=6)
    detector = LGBMDetector(seed=seed, params=FAST).fit(rows[:700])
    return detector, rows[700:]


def test_every_flagged_transaction_carries_at_least_three_reason_codes():
    """The acceptance criterion, asserted on the emitted scores rather than on the explainer."""
    detector, holdout = _fitted_detector()
    scores = detector.score(as_batch(holdout))
    flagged = [s for s in scores if s.action is not Action.ALLOW]
    assert flagged, "nothing was flagged — this test would pass vacuously"
    explain.assert_flagged_rows_are_explained(scores)
    assert all(len(s.reasons) >= explain.MIN_REASONS for s in flagged)


def test_reason_codes_are_analyst_language_and_name_the_decision():
    detector, holdout = _fitted_detector()
    flagged = [s for s in detector.score(as_batch(holdout)) if s.action is not Action.ALLOW]
    reasons = flagged[0].reasons

    assert any(r.startswith("decision:") for r in reasons), "no reason says why this action"
    feature_codes = [r for r in reasons if r[0] in "↑↓"]
    assert feature_codes, "no local feature driver in the reason list"
    assert not any(
        "_" in r.split("(")[0] for r in feature_codes
    ), f"a raw column name reached an analyst-facing reason code: {feature_codes}"


def test_the_ensemble_explains_flagged_rows_from_both_halves():
    """The detector every anchored run actually uses — it used to emit the string 'ensemble'."""
    rows = txns(900, fraud_every=6)
    ensemble = EnsembleDetector(
        supervised=LGBMDetector(seed=12, params=FAST),
        unsupervised=AnomalyDetector(seed=12),
        weight=0.7,
    ).fit(rows[:700])

    scores = ensemble.score(as_batch(rows[700:]))
    flagged = [s for s in scores if s.action is not Action.ALLOW]
    assert flagged
    explain.assert_flagged_rows_are_explained(scores)
    assert all(
        any(r.startswith("blend:") for r in s.reasons) for s in flagged
    ), "an analyst holding a blended score cannot see which half raised it"
    assert any(any("σ vs legit traffic" in r for r in s.reasons) for s in flagged)


def test_the_shap_unavailable_fallback_is_labelled_in_the_reason_string_itself(monkeypatch):
    """Not in a log line nobody reads, and not in a sibling field somebody drops en route."""
    import builtins

    real_import = builtins.__import__

    def no_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("no shap in this environment")
        return real_import(name, *args, **kwargs)

    detector, holdout = _fitted_detector()
    detector._explainer = None
    monkeypatch.setattr(builtins, "__import__", no_shap)

    codes = explain.reason_codes(detector, holdout[:5])
    monkeypatch.undo()

    assert all(len(row) == explain.MIN_REASONS for row in codes)
    assert all(
        r.startswith(explain.GLOBAL_PREFIX) for row in codes for r in row
    ), "a global explanation was presented as if it were about this transaction"


def test_a_short_shap_explanation_is_padded_rather_than_returned_short():
    """A shallow tree can attribute a score to two features. Two reasons is still not three."""
    detector, holdout = _fitted_detector()
    codes = explain.reason_codes(detector, holdout[:20], top_k=8)
    assert all(len(row) == 8 for row in codes)


def test_the_invariant_catches_an_unexplained_flag():
    """The guard has to fire, or it is decoration."""
    naked = [DetectorScore(txn_id="t1", score=0.9, action=Action.DECLINE, reasons=["only one"])]
    assert explain.unexplained(naked) == ["t1"]
    with pytest.raises(AssertionError, match="fewer than 3 reason codes"):
        explain.assert_flagged_rows_are_explained(naked)


def test_explaining_only_the_flagged_rows_is_the_same_explanation():
    """The optimisation must not change the answer, only what it costs."""
    detector, holdout = _fitted_detector()
    flagged_mode = detector.score(as_batch(holdout))
    detector.explain = "always"
    all_mode = detector.score(as_batch(holdout))

    for a, b in zip(flagged_mode, all_mode, strict=True):
        assert a.action is b.action
        if a.action is not Action.ALLOW:
            assert a.reasons == b.reasons


# ── the evasion definition has not moved ───────────────────────────────────────
def test_only_allow_counts_as_evaded():
    """Both sides agree on this or the loop's fitness function measures something else."""
    assert EVASION_ACTIONS == frozenset({Action.ALLOW})
    for action in SEVERITY:
        score = DetectorScore(txn_id="t", score=0.5, action=action)
        assert score.evaded is (action is Action.ALLOW)


def test_a_graded_action_still_counts_as_caught_in_the_operational_rates():
    """A stepped-up fraud is not an evasion, however graded the ladder became."""
    rows = txns(20, fraud_every=2)
    policy = DecisionPolicy()
    scores = [
        DetectorScore(
            txn_id=t.txn_id,
            score=0.5,
            action=Action.STEP_UP if t.is_fraud else Action.ALLOW,
            reasons=["a", "b", "c"],
        )
        for t in rows
    ]
    rates = protocol.operational_rates(rows, scores)
    assert rates["evasion_rate"] == 0.0
    assert rates["caught_rate"] == 1.0
    assert rates["friction_rate"] == 0.0
    assert policy.mode == "cost"


def test_the_cost_policy_beats_doing_nothing_and_beats_blocking_everything():
    """The floor a decision policy has to clear, and the one thing that *is* guaranteed.

    Two assertions that were tempting here and are both wrong. "Less friction than the ratio
    bands" is wrong because a cost-minimising policy will happily buy more friction when the
    fraud it stops is worth more — on PaySim's own fraud it frictions 3.7% against the ratio
    bands' 0.67% and lets 17 points less fraud through. "Lower realised cost than the ratio
    bands" is wrong too, and only just: on this small synthetic window it loses by 0.2%
    (5,186 against 5,174), because minimising *expected* cost under an imperfect probability
    does not guarantee the lower *realised* cost on any particular sample.

    So that comparison is measured in `artifacts/decisions/<anchor>.json` on real anchors, where
    it is a result rather than an assumption, and what is asserted here is the floor: a policy
    that cannot beat allowing everything and declining everything is not a policy.
    """
    sim = Simulator(seed=31, n_entities=120, n_background=600, n_episodes=2)
    pool: list[Transaction] = []
    for vid in ("S1", "S2", "M3"):
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)
    pool.sort(key=lambda t: t.ts)
    train, holdout = pool[: int(len(pool) * 0.7)], pool[int(len(pool) * 0.7) :]
    assert any(t.is_fraud for t in holdout), "a holdout with no fraud would pass this vacuously"

    costs = cost_model_for(COSTS_CONFIG, train)
    detector = LGBMDetector(
        seed=31, params=FAST, policy=policy_from_config({"mode": "cost"}, costs)
    ).fit(train)
    y, raw = protocol.align(train, protocol.score_transactions(detector, train, "cal"))
    detector.policy.fit_calibrator(raw, y)

    amounts = {t.txn_id: t.amount for t in holdout}
    labels = {t.txn_id: int(t.is_fraud) for t in holdout}
    scores = protocol.score_transactions(detector, holdout, "priced")

    priced = total_cost(scores, amounts, labels, costs)
    allow_all = total_cost(
        [s.model_copy(update={"action": Action.ALLOW}) for s in scores], amounts, labels, costs
    )
    decline_all = total_cost(
        [s.model_copy(update={"action": Action.DECLINE}) for s in scores], amounts, labels, costs
    )

    assert priced < allow_all, (
        f"realised {priced:,.0f} against allow-everything {allow_all:,.0f} — the policy is not "
        "paying for itself"
    )
    assert priced < decline_all, (
        f"realised {priced:,.0f} against decline-everything {decline_all:,.0f} — blanket "
        "blocking would be cheaper than this policy"
    )
    assert action_mix(scores)["allow"] < 1.0, "nothing was actioned at all"


def test_the_uncalibrated_warning_stays_quiet_during_calibration_and_speaks_after_a_refusal(
    caplog,
):
    """A warning that fires on the code path doing the right thing is one nobody reads.

    Scoring the validation rows is *supposed* to happen uncalibrated — that is where the map
    comes from. But a fit that was refused for want of positives leaves the policy genuinely
    pricing a ranking score, and that has to be audible.
    """
    import logging

    def warnings_from(fn) -> list[str]:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="afl.defend.decision"):
            fn()
        return [r.getMessage() for r in caplog.records if "UNCALIBRATED" in r.getMessage()]

    rng = np.random.default_rng(5)
    labels = np.array([0] * 1_000 + [1] * 40)
    raw = np.concatenate([rng.beta(2, 30, 1_000), rng.beta(6, 10, 40)])

    policy = DecisionPolicy()
    policy.reset_calibration()
    assert not warnings_from(lambda: policy.decide("t", 0.5, amount=1_000.0))

    policy.fit_calibrator(raw, labels)
    assert policy.calibrator.fitted
    assert not warnings_from(lambda: policy.decide("t", 0.5, amount=1_000.0))

    thin = DecisionPolicy()
    thin.reset_calibration()
    thin.fit_calibrator(raw[:60], labels[:60])  # all negatives — the fit is refused
    assert not thin.calibrator.fitted
    assert warnings_from(lambda: thin.decide("t", 0.5, amount=1_000.0))
