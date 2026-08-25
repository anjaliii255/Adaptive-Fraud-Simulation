"""■ B — the supervised baseline is strong, honest, and says which model produced it.

Ticket 08's failure modes are all quiet ones, which is why they are tests rather than a review:

  * the detector runs on a fallback backend under a table headed "LightGBM";
  * the search that made it "tuned" scored itself on the window it is reported from;
  * two systems in one table are compared at two different operating points;
  * `retrain` replaces the corpus instead of accumulating, so the loop measures recency;
  * a metric that flatters at a 0.13% base rate creeps back into the headline.

Every one of them looks like a working pipeline right up until someone tries to reproduce it.
"""

from __future__ import annotations

import builtins
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.metrics import Action, MetricResult
from afl.contract.schema import AttackBatch, AttackParams, Rail, Transaction
from afl.data.splits import CommittedSplit
from afl.defend import baseline, tuning
from afl.defend.models import lgbm
from afl.defend.models.lgbm import LGBMDetector, make_estimator
from afl.evaluation import protocol, three_system

T0 = datetime(2023, 1, 1)

CONFIG_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
DETECTOR_DIR = Path("artifacts/detector")
SPLIT_DIR = Path("artifacts/splits")

#: Small enough that the whole file runs in seconds; the reported baseline is built by
#: `scripts/build_baseline.py`, not here.
FAST = {"n_estimators": 20, "num_leaves": 7}


def txns(
    n: int, fraud_every: int = 8, start: datetime = T0, prefix: str = "t"
) -> list[Transaction]:
    """Traffic with a learnable tell: fraud rows are the large ones."""
    return [
        Transaction(
            txn_id=f"{prefix}{i:05d}",
            ts=start + timedelta(minutes=17 * i),
            src=f"s{i % 23}",
            dst=f"d{i % 11}",
            amount=(9_000.0 + i) if i % fraud_every == 0 else (10.0 + i % 90),
            rail=Rail.A2A,
            is_fraud=i % fraud_every == 0,
            vector_id="S1" if i % fraud_every == 0 else None,
        )
        for i in range(n)
    ]


def batch_of(rows: list[Transaction], run_id: str = "r") -> AttackBatch:
    return AttackBatch(
        run_id=run_id,
        params=AttackParams(vector_id="S1", engine="graph"),
        transactions=list(rows),
        seed=0,
    )


def unimportable(monkeypatch, module: str, exc: Exception) -> None:
    """Make one `import` fail, so the fallback path is exercised rather than assumed."""
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == module:
            raise exc
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


def anchors() -> list[dict]:
    configs = [yaml.safe_load(p.read_text()) for p in sorted(CONFIG_DIR.glob("*.yaml"))]
    return [c for c in configs if c.get("loader")]


ANCHORS = anchors()
ANCHOR_IDS = [c["name"] for c in ANCHORS]


# ── the backend is never a guess ────────────────────────────────────────────────
def test_the_backend_is_recorded_on_the_model_card():
    detector = LGBMDetector(seed=1, params=FAST).fit(txns(200))
    card = detector.model_card()

    assert card["backend"]["name"] in ("lightgbm", "sklearn-hgb")
    assert card["backend"]["version"] and card["backend"]["reason"]
    assert card["training"]["fitted"] is True
    assert card["training"]["n_rows"] == 200
    assert card["training"]["n_fraud"] == 25
    assert card["seed"] == 1


def test_an_unfitted_detector_is_not_silently_a_perfect_negative(caplog):
    """Zeros from an unfitted model read in a metric exactly like a detector that caught nothing."""
    detector = LGBMDetector(seed=1, params=FAST)
    with caplog.at_level("WARNING"):
        probs = detector.predict_proba(txns(10))
    assert probs.tolist() == [0.0] * 10
    assert "unfitted" in caplog.text
    assert detector.model_card()["backend"]["name"] == "untrained"


def test_the_sklearn_fallback_is_loud_and_says_what_it_lost(monkeypatch, caplog):
    """The macOS case: the wheel imports, then fails to dlopen its own library."""
    monkeypatch.setattr(lgbm, "_FALLBACK_WARNED", False)
    unimportable(monkeypatch, "lightgbm", OSError("Library not loaded: @rpath/libomp.dylib"))

    with caplog.at_level("WARNING"):
        model, backend = make_estimator(dict(lgbm.DEFAULT_PARAMS), seed=1)

    assert backend.name == "sklearn-hgb"
    assert not backend.is_lightgbm
    assert "OSError" in backend.reason and "libomp" in backend.reason
    assert "LIGHTGBM UNAVAILABLE" in caplog.text
    assert "brew install libomp" in caplog.text
    # knobs with no HistGradientBoosting equivalent are named, not quietly dropped
    assert set(backend.dropped_params) >= {"colsample_bytree", "subsample", "subsample_freq"}
    assert model.max_iter == lgbm.DEFAULT_PARAMS["n_estimators"]


def test_the_fallback_is_handed_the_same_tuning_under_its_own_names(monkeypatch):
    """A fallback given none of the searched params is not a stand-in, it is a different model."""
    monkeypatch.setattr(lgbm, "_FALLBACK_WARNED", True)
    unimportable(monkeypatch, "lightgbm", ImportError("no lightgbm"))

    params = {**lgbm.DEFAULT_PARAMS, "num_leaves": 63, "min_child_samples": 7, "reg_lambda": 2.5}
    model, backend = make_estimator(params, seed=3)

    assert (model.max_leaf_nodes, model.min_samples_leaf, model.l2_regularization) == (63, 7, 2.5)
    assert model.random_state == 3
    assert backend.name == "sklearn-hgb"


def test_a_detector_scores_every_row_through_the_seam():
    rows = txns(200)
    detector = LGBMDetector(seed=2, params=FAST).fit(rows)
    scores = detector.score(batch_of(rows))

    assert [s.txn_id for s in scores] == [t.txn_id for t in rows]
    assert all(0.0 <= s.score <= 1.0 for s in scores)
    assert all(isinstance(s.action, Action) for s in scores)


# ── retrain accumulates, and evasions cost more ─────────────────────────────────
def test_retrain_accumulates_rather_than_forgetting():
    """The loop must not reduce the detector to whatever it saw most recently."""
    sim = Simulator(seed=26, n_entities=100, n_background=250, n_episodes=2)
    first = sim.generate(registry.get("S1").to_attack_params())
    second = sim.generate(registry.get("S2").to_attack_params())

    detector = LGBMDetector(seed=26, params=FAST)
    detector.fit(first.transactions)
    before = detector.training.n_rows
    detector.retrain(second, evasions=second.fraud_transactions[:3])

    corpus = {t.txn_id for t in detector._corpus}
    assert {t.txn_id for t in first.transactions} <= corpus
    assert {t.txn_id for t in second.transactions} <= corpus
    assert detector.training.n_rows > before
    assert detector.training.n_rows == len(corpus)


def test_retrain_twice_keeps_the_first_round(caplog):
    """Two rounds, because a bug that drops all but the last round survives a one-round test."""
    detector = LGBMDetector(seed=4, params=FAST).fit(txns(120, prefix="a"))
    detector.retrain(batch_of(txns(120, start=T0 + timedelta(days=9), prefix="b")), evasions=[])
    detector.retrain(batch_of(txns(120, start=T0 + timedelta(days=18), prefix="c")), evasions=[])

    corpus = {t.txn_id for t in detector._corpus}
    assert len(corpus) == 360
    assert {"a00000", "b00000", "c00000"} <= corpus


@pytest.mark.parametrize("weight", [2.0, 7.5])
def test_evasions_are_weighted_above_ordinary_training_rows(weight):
    rows = txns(160)
    detector = LGBMDetector(seed=5, params=FAST, replay_weight=weight).fit(rows)
    evasions = [t for t in rows if t.is_fraud][:4]
    detector.retrain(batch_of(txns(80, start=T0 + timedelta(days=5), prefix="n")), evasions)

    weights = detector.sample_weights(detector._corpus)
    heavy = {t.txn_id for t in evasions}
    by_id = dict(zip((t.txn_id for t in detector._corpus), weights, strict=True))

    assert all(by_id[i] == weight for i in heavy)
    assert {v for i, v in by_id.items() if i not in heavy} == {1.0}
    assert detector.training.n_weighted_up == len(evasions)
    assert detector.model_card()["training"]["replay_weight"] == weight


def test_the_replay_weight_is_config_and_not_a_literal():
    """Ticket 08 asks for the weight to be a knob. A default is not a knob if nothing sets it."""
    cfg = yaml.safe_load(LGBM_CONFIG.read_text())
    assert "replay_weight" in cfg, "the weight has no home in config"

    # the config value reaches the detector, and a different one produces a different weight —
    # which is what rules out a literal that happens to agree with the config today
    from_config = LGBMDetector(replay_weight=float(cfg["replay_weight"]))
    other = LGBMDetector(replay_weight=float(cfg["replay_weight"]) + 6.0)
    rows = txns(40)
    for detector in (from_config, other):
        detector._replay = [rows[0]]
        assert detector.sample_weights(rows)[0] == detector.replay_weight
        assert detector.model_card()["training"]["replay_weight"] == detector.replay_weight
    assert from_config.replay_weight != other.replay_weight


# ── the search never sees the window it is reported from ────────────────────────
def test_tuning_refuses_a_validation_set_that_is_not_out_of_time():
    rows = txns(400)
    with pytest.raises(tuning.LeakingValidationSplit, match="scoring itself"):
        tuning.tune(rows[:300], rows[200:], n_trials=1, seed=1)


def test_tuning_refuses_a_validation_set_that_shares_a_row():
    rows = sorted(txns(400), key=lambda t: t.ts)
    fit, val = rows[:300], rows[300:]
    # same timestamps, one id smuggled across — the guard must not be satisfied by order alone
    with pytest.raises(tuning.LeakingValidationSplit, match="both"):
        tuning.tune(fit, [fit[0].model_copy(update={"ts": val[0].ts}), *val], n_trials=1, seed=1)


def test_the_search_stays_inside_the_declared_envelope():
    rows = sorted(txns(600), key=lambda t: t.ts)
    space = {
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.2, "log": True},
        "num_leaves": {"type": "int", "low": 4, "high": 16},
        "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
    }
    result = tuning.tune(
        rows[:400], rows[420:], base_params=FAST, search_space=space, n_trials=6, seed=7
    )

    assert result.tuned and result.trials
    for trial in result.trials:
        p = trial["params"]
        assert 0.02 <= p["learning_rate"] <= 0.2
        assert 4 <= p["num_leaves"] <= 16
        assert p["class_weight"] in (None, "balanced")
    assert set(result.params) >= set(FAST)


def test_tuning_is_reproducible_from_its_seed():
    rows = sorted(txns(600), key=lambda t: t.ts)
    kwargs = dict(base_params=FAST, n_trials=5, seed=11, backend="random")
    a = tuning.tune(rows[:400], rows[420:], **kwargs)
    b = tuning.tune(rows[:400], rows[420:], **kwargs)
    assert a.params == b.params and a.best_score == b.best_score


def test_the_search_runs_without_optuna(monkeypatch):
    """The config says CI exercises both paths. This is that."""
    unimportable(monkeypatch, "optuna", ImportError("no optuna"))
    rows = sorted(txns(600), key=lambda t: t.ts)
    result = tuning.tune(rows[:400], rows[420:], base_params=FAST, n_trials=4, seed=3)

    assert result.backend == "random"
    assert result.tuned and len(result.trials) == 4


def test_a_search_with_nothing_to_rank_says_so_rather_than_returning_a_number(caplog):
    rows = sorted(txns(400, fraud_every=10**6), key=lambda t: t.ts)  # no fraud anywhere
    with caplog.at_level("WARNING"):
        result = tuning.tune(rows[:300], rows[320:], base_params=FAST, n_trials=3, seed=1)

    assert not result.tuned
    assert "random walk" in result.skipped
    assert result.params["n_estimators"] == FAST["n_estimators"]  # the defaults, untouched
    assert "tuning skipped" in caplog.text


def test_tuning_reports_the_counterfactual_not_just_the_winner():
    """ "Tuned" is a claim about a comparison, so the artefact carries both sides of it."""
    rows = sorted(txns(600), key=lambda t: t.ts)
    result = tuning.tune(rows[:400], rows[420:], base_params=FAST, n_trials=4, seed=2)
    payload = result.to_dict()

    assert payload["default_score"] == pytest.approx(result.default_score)
    assert payload["lift_over_defaults"] == pytest.approx(result.best_score - result.default_score)
    assert payload["n_val_positives"] > 0
    assert payload["val_start"] > payload["fit_end"]


# ── one operating point, every row ──────────────────────────────────────────────
def _pool(seed: int = 24) -> list[Transaction]:
    sim = Simulator(seed=seed, n_entities=120, n_background=400, n_episodes=2)
    pool: list[Transaction] = []
    for vid in ("S1", "S2", "M3"):
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)
    return pool


def test_every_system_is_reported_at_the_configured_operating_point():
    """Two systems compared at two thresholds is not a comparison."""
    results = three_system.run_three_systems(
        pool=_pool(),
        detector_factory=lambda: LGBMDetector(seed=24, params=FAST),
        held_out_vector="M3",
        fixed_fpr=0.02,
        k=25,
    )
    assert results
    for r in results:
        assert r.metrics.fixed_fpr == 0.02
        assert r.metrics.k == 25
        assert "recall@2%fpr" in r.row()


def test_one_fit_function_is_applied_to_every_system():
    """The bands are part of the operating point, so they cannot be set per system."""
    seen: list[int] = []

    def fit(detector, rows):
        seen.append(len(rows))
        detector.fit(rows)
        detector.policy.decline_at = 0.42

    from afl.attack.optimiser import AttackOptimiser

    sim = Simulator(seed=24, n_entities=120, n_background=400, n_episodes=2)
    optimiser = AttackOptimiser(vector_id="S1", seed=24, backend="random")
    results = three_system.run_three_systems(
        pool=_pool(),
        detector_factory=lambda: LGBMDetector(seed=24, params=FAST),
        simulator=optimiser.bind(sim),
        optimiser=optimiser,
        held_out_vector="M3",
        rounds=1,
        fit_detector=fit,
    )
    assert len(seen) == 3, "every system is trained the same way or none of them are comparable"
    assert {r.model_card["decision"]["decline_at"] for r in results} == {0.42}


def test_two_operating_points_in_one_config_are_refused():
    """The bands and the metric are the same decision; a config naming two is a bug, not a mode."""
    from afl.defend.decision import assert_one_operating_point

    assert_one_operating_point(0.01, 0.01)
    assert_one_operating_point(None, 0.05)  # calibration off: the fixed bands stand
    with pytest.raises(ValueError, match="two operating points"):
        assert_one_operating_point(0.05, 0.01)


# The shipped config's operating point, the cost model behind the bands, and the reason codes
# are asserted in `tests/test_decision.py` — ticket 09's file.


def test_the_table_names_the_backend_that_produced_it():
    # `real_vectors` names the families a team already has labels for; without it System A trains
    # on legit rows alone, no model is fitted, and the backend is honestly "untrained"
    results = three_system.run_three_systems(
        pool=_pool(),
        detector_factory=lambda: LGBMDetector(seed=24, params=FAST),
        held_out_vector="M3",
        real_vectors=("S1", "S2"),
    )
    markdown = three_system.to_markdown(results)
    assert "detector backend:" in markdown
    assert results[0].backend.split()[0] in ("lightgbm", "sklearn-hgb")
    assert results[0].model_card["backend"]["reason"]


# ── metrics we do not report ────────────────────────────────────────────────────
def test_the_reported_metric_types_have_no_accuracy_or_roc_auc():
    """At a 0.13% base rate both are dominated by the negatives, so neither is a headline."""
    banned = ("accuracy", "roc_auc", "auroc")
    assert not [f for f in MetricResult.model_fields if any(b in f for b in banned)]

    row = three_system.SystemResult(name="x", metrics=protocol.evaluate([0, 1], [0.1, 0.9])).row()
    assert not [c for c in row if any(b in str(c).lower() for b in banned)]


@pytest.mark.parametrize(
    "payload",
    [
        {"tuned": {"accuracy": 0.99}},
        {"tuned": {"roc_auc": 0.98}},
        {"tuned": {"nested": [{"balanced_accuracy": 0.5}]}},
    ],
)
def test_a_baseline_artefact_refuses_to_carry_a_flattering_metric(payload):
    with pytest.raises(baseline.ForbiddenMetric):
        baseline.assert_no_forbidden_metrics(payload)


def test_the_three_metrics_we_do_report_pass_the_guard():
    baseline.assert_no_forbidden_metrics(
        {"tuned": {"pr_auc": 0.5, "recall_at_fixed_fpr": 0.3, "precision_at_k": 0.2}}
    )


# ── the committed reference ─────────────────────────────────────────────────────
def _reference(**overrides) -> baseline.Baseline:
    kwargs = dict(
        dataset="demo",
        operating_point={"fixed_fpr": 0.01, "k": 100},
        backend={"name": "lightgbm", "version": "4.5.0", "reason": "loaded"},
        params={"num_leaves": 31},
        metrics={
            "tuned": {"pr_auc": 0.5, "recall_at_fixed_fpr": 0.4, "precision_at_k": 0.3},
            "default": {"pr_auc": 0.2, "recall_at_fixed_fpr": 0.1, "precision_at_k": 0.05},
        },
    )
    kwargs.update(overrides)
    return baseline.Baseline(**kwargs)


def test_the_committed_baseline_round_trips(tmp_path):
    ref = _reference()
    path = ref.save(tmp_path)
    assert baseline.Baseline.load("demo", tmp_path).to_dict() == ref.to_dict()
    assert json.loads(path.read_text())["version"] == baseline.BASELINE_ARTEFACT_VERSION
    assert "PR-AUC 0.500" in ref.summary() and "lightgbm" in ref.summary()


def test_a_reference_missing_one_of_the_three_metrics_is_refused():
    with pytest.raises(ValueError, match="missing"):
        _reference(metrics={"tuned": {"pr_auc": 0.5}})


def test_an_artefact_from_an_older_shape_fails_loudly(tmp_path):
    (tmp_path / "demo.json").write_text(json.dumps({"version": 0, "dataset": "demo"}))
    with pytest.raises(ValueError, match="version"):
        baseline.Baseline.load("demo", tmp_path)


def test_a_run_with_no_committed_baseline_falls_back_to_config(tmp_path):
    """A fresh clone, and the synthetic default, have no baseline by design."""
    params, source = baseline.tuned_params("nothing-here", tmp_path)
    assert params == {}
    assert "no usable baseline" in source


def test_the_committed_params_are_what_a_later_run_picks_up(tmp_path):
    _reference().save(tmp_path)
    params, source = baseline.tuned_params("demo", tmp_path)
    assert params == {"num_leaves": 31}
    assert source.endswith("demo.json")


@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_every_anchor_baseline_matches_the_boundary_and_operating_point_in_force(cfg):
    """The drift this catches: a reference measured against a boundary that has since moved.

    Cheap on purpose — it reads artefacts, not the raw files — because the check that matters is
    whether the committed numbers still describe the split and the operating point the rest of
    the project runs at, not whether they can be recomputed on this machine right now.
    """
    name = cfg["name"]
    path = DETECTOR_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"no committed baseline for {name} — run `make baseline`")

    ref = baseline.Baseline.load(name, DETECTOR_DIR)
    split = CommittedSplit.load(name, SPLIT_DIR)
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())

    assert ref.split["digest"] == split.digest, "the baseline predates the committed boundary"
    assert ref.operating_point["fixed_fpr"] == float(eval_cfg["fixed_fpr"])
    assert ref.operating_point["k"] == int(eval_cfg["k"])
    assert ref.backend_name in ("lightgbm", "sklearn-hgb")
    assert ref.params, "a reference with no params cannot be reproduced"
    assert ref.data["test"]["fraud"] > 0, "a reference measured on a window with no fraud"
    assert set(ref.metrics) >= {"tuned", "default", "amount_only"}, (
        "a reference needs both controls: the stock params it was tuned against, and the "
        "amount-only floor that says how hard the anchor was to begin with"
    )


@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_the_committed_reference_beats_the_floor_and_the_stock_params(cfg):
    """The guard against a soft baseline, asserted on the committed numbers.

    A detector that cannot beat sorting the rows by amount is not a baseline whatever it is
    called, and one that cannot beat its own stock params did not need the search.
    """
    name = cfg["name"]
    if not (DETECTOR_DIR / f"{name}.json").exists():
        pytest.skip(f"no committed baseline for {name} — run `make baseline`")

    ref = baseline.Baseline.load(name, DETECTOR_DIR)
    tuned, stock, floor = (ref.metrics[v] for v in ("tuned", "default", "amount_only"))

    assert tuned["pr_auc"] > floor["pr_auc"], (
        f"{name}: the reference ranks no better than amount alone "
        f"({tuned['pr_auc']} vs {floor['pr_auc']})"
    )
    assert tuned["pr_auc"] >= stock["pr_auc"], f"{name}: the search made the detector worse"
    assert ref.tuning["lift_over_defaults"] > 0, f"{name}: the search bought nothing on validation"


@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_the_floor_direction_was_chosen_on_the_training_window(cfg):
    """Picking the direction on the window it is reported from is tuning on test in miniature."""
    name = cfg["name"]
    if not (DETECTOR_DIR / f"{name}.json").exists():
        pytest.skip(f"no committed baseline for {name} — run `make baseline`")

    floor = baseline.Baseline.load(name, DETECTOR_DIR).metrics["amount_only"]
    assert floor["direction_chosen_on"] == "train"
    assert floor["direction"] in ("largest amount first", "smallest amount first")
    assert 0.0 <= floor["legit_share_inside_fraud_amount_range"] <= 1.0


def test_the_reference_beats_a_ranking_that_knows_nothing():
    """A baseline that cannot beat chance is not a baseline, whatever it is called."""
    rows = sorted(txns(1_200), key=lambda t: t.ts)
    train, test = rows[:800], rows[820:]
    detector = LGBMDetector(seed=8, params=FAST).fit(train)
    result = protocol.evaluate_detector(detector, test)

    rng = np.random.default_rng(0)
    chance = protocol.evaluate(
        [int(t.is_fraud) for t in test], rng.random(len(test)), fixed_fpr=0.01, k=100
    )
    assert result.pr_auc > chance.pr_auc
    assert result.recall_at_fixed_fpr >= chance.recall_at_fixed_fpr
