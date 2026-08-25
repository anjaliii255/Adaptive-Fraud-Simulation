"""■ B — the zero-day layer is fitted on legit traffic, and says the same thing twice.

Ticket 10's failure modes are all quiet, which is why they are tests rather than a review:

  * a fraud row reaches the outlier model's training set and teaches it that fraud is normal;
  * a retrain widens "normal" with the very rows that evaded;
  * the score is a statement about the batch instead of about the transaction, so an ensemble
    blends a rank statistic against a probability and every metric still looks fine;
  * "the ensemble" is a name rather than a defined function of its two halves;
  * a held-out-family comparison is run against a fold one contract field already separates.

Every one of them leaves a plausible-looking table behind.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from afl.attack.envelope import AnchorEnvelope, audit
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.metrics import Action
from afl.contract.schema import AttackBatch, AttackParams, Rail, Transaction
from afl.defend import explain
from afl.defend.models.anomaly import (
    AnomalyDetector,
    EnsembleDetector,
    ScoreMap,
    contaminated_control,
)
from afl.defend.models.lgbm import LGBMDetector, model_card_of
from afl.evaluation import protocol
from afl.evaluation.leave_one_attack_out import make_splits

T0 = datetime(2024, 1, 1)

#: Small enough that the whole file runs in seconds; the reported numbers are built by
#: `scripts/build_anomaly.py`, not here.
FAST = {"n_estimators": 20, "num_leaves": 7}


def txns(
    n: int, fraud_every: int = 8, start: datetime = T0, prefix: str = "t", vector: str = "S1"
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
            vector_id=vector if i % fraud_every == 0 else None,
        )
        for i in range(n)
    ]


def as_batch(rows: list[Transaction], run_id: str = "r") -> AttackBatch:
    return AttackBatch(
        run_id=run_id,
        params=AttackParams(vector_id="eval", engine="none"),
        transactions=list(rows),
        seed=0,
    )


# ── fits on legit rows only ─────────────────────────────────────────────────────
def test_no_fraud_row_enters_training():
    """The acceptance criterion, asserted on what the fit recorded rather than on the filter."""
    rows = txns(400)
    fraud = [t for t in rows if t.is_fraud]
    assert fraud, "this test would pass vacuously on a fraud-free corpus"

    detector = AnomalyDetector(seed=7).fit(rows)

    assert detector.training.n_fraud_seen == 0
    assert detector.training.n_fraud_excluded == len(fraud)
    assert detector.training.n_rows == len(rows) - len(fraud)
    assert not any(t.is_fraud for t in detector._corpus)
    assert detector.model_card()["training"]["legit_only"] is True


def test_fit_has_no_argument_that_lets_a_fraud_row_in():
    """`legit_only=True` was a default, which made the guarantee a setting.

    The contaminated variant still exists — ticket 10 measures it — but it is reached through a
    function whose name appears in the artefact, not through a keyword nobody reads.
    """
    with pytest.raises(TypeError):
        AnomalyDetector(seed=7).fit(txns(200), legit_only=False)


def test_the_contaminated_control_is_labelled_as_one():
    rows = txns(400)
    control = contaminated_control(rows, AnomalyDetector(seed=7))
    card = control.model_card()

    assert card["training"]["legit_only"] is False
    assert card["training"]["n_fraud_seen"] == sum(1 for t in rows if t.is_fraud)
    assert card["training"]["n_rows"] == len(rows)


def test_retrain_widens_normal_without_swallowing_the_evasions():
    """A round's fraud rows are what the layer is supposed to find, not what defines normal."""
    detector = AnomalyDetector(seed=7).fit(txns(400))
    before = detector.training.n_rows

    later = txns(200, start=T0 + timedelta(days=30), prefix="u")
    detector.retrain(as_batch(later), evasions=[t for t in later if t.is_fraud])

    assert detector.training.n_fraud_seen == 0
    assert not any(t.is_fraud for t in detector._corpus)
    assert detector.training.n_rows > before, "the corpus narrowed to the latest round"


def test_an_unfittably_small_corpus_says_so_rather_than_shipping_a_model(caplog):
    detector = AnomalyDetector(seed=7)
    with caplog.at_level("WARNING"):
        detector.fit(txns(8, fraud_every=100))
    assert detector.model is None
    assert not detector.training.fitted
    assert "notion of normal" in caplog.text


# ── one row, one score ──────────────────────────────────────────────────────────
def test_the_score_map_does_not_depend_on_the_batch():
    """It used to min-max the batch, so a row's score was a statement about its company.

    Measured with the features held fixed, because `FeatureBuilder.transform(update=False)` is
    batch-dependent on purpose — a row sees the rows before it in the same call — and that
    residual belongs to the feature contract, not to this layer.
    """
    rows = txns(600)
    detector = AnomalyDetector(seed=7).fit(rows[:400])
    scaled = detector.scaled(rows[400:])
    raw = detector.raw_scores(scaled=scaled)

    half = len(raw) // 2
    whole = detector.score_map.apply(raw)
    halves = np.concatenate(
        [detector.score_map.apply(raw[:half]), detector.score_map.apply(raw[half:])]
    )
    assert np.array_equal(whole, halves)

    def min_max(v):
        return (v - v.min()) / (v.max() - v.min())

    legacy = np.concatenate([min_max(raw[:half]), min_max(raw[half:])])
    assert np.max(np.abs(min_max(raw) - legacy)) > 0.01, "the bug this test guards has no bite"


def test_the_isolation_forest_score_is_already_a_bounded_quantity():
    """`-score_samples` is 2 ** (-E[h(x)] / c(n)), which is why the map is the identity."""
    rows = txns(600)
    detector = AnomalyDetector(seed=7).fit(rows[:400])
    raw = detector.raw_scores(rows[400:])

    assert detector.score_map.kind == "identity"
    assert 0.0 < float(raw.min()) and float(raw.max()) < 1.0


def test_the_autoencoder_score_is_mapped_against_a_fit_time_reference():
    """Reconstruction error is unbounded, so it needs a map — one fixed at fit time."""
    rows = txns(400)
    detector = AnomalyDetector(kind="ae", seed=7).fit(rows)

    assert detector.score_map.kind == "saturating"
    assert detector.score_map.scale > 0.0
    probs = detector.predict_proba(rows[:50])
    assert np.all((probs >= 0.0) & (probs < 1.0))


def test_a_score_map_never_returns_a_value_a_detector_score_would_refuse():
    """`DetectorScore` validates its score into [0, 1], so a map that leaves it is a crash."""
    wild = np.array([-3.0, 0.0, 1.0, 1e6, 1e300, np.inf])
    for m in (ScoreMap(kind="saturating", scale=2.0), ScoreMap(kind="identity")):
        mapped = m.apply(wild)
        assert np.all((mapped >= 0.0) & (mapped <= 1.0)), m


def test_the_saturating_map_saturates_where_it_says_it_does_and_not_before():
    """Ticket 09's lesson: name the float64 limit rather than assert there is not one."""
    m = ScoreMap(kind="saturating", scale=2.0)
    realistic = m.apply(np.array([0.0, 0.5, 2.0, 20.0, 2e3, 2e9, 2e14]))
    assert np.all(np.diff(realistic) > 0.0), "the map lost its ordering inside the useful range"
    assert realistic.max() < 1.0
    assert m.apply(np.array([2.0 * 2**53]))[0] == 1.0  # the documented limit, reached exactly


# ── through the same seam as every other detector ───────────────────────────────
def test_it_scores_through_the_seam_and_returns_graded_actions():
    rows = txns(900, fraud_every=6)
    detector = AnomalyDetector(seed=7).fit(rows[:700])

    scores = detector.score(as_batch(rows[700:]))

    assert [s.txn_id for s in scores] == [t.txn_id for t in rows[700:]]
    assert all(0.0 <= s.score <= 1.0 for s in scores)
    assert all(isinstance(s.action, Action) for s in scores)
    explain.assert_flagged_rows_are_explained(scores)
    assert any(any("σ vs legit traffic" in r for r in s.reasons) for s in scores)


def test_an_unfitted_layer_scores_zero_rather_than_raising():
    """The loop must survive a first round with no legit history to learn from."""
    rows = txns(20, fraud_every=100)
    scores = AnomalyDetector(seed=7).score(as_batch(rows))
    assert all(s.score == 0.0 and s.action is Action.ALLOW for s in scores)


# ── the ensemble is a defined function of its halves ────────────────────────────
def test_the_blend_is_exactly_the_weighted_sum_of_the_two_halves():
    """Not approximately, and not "the ensemble" as a name for whatever the wrapper does.

    `scripts/build_anomaly.py` sweeps the weight over one pair of probability vectors instead of
    re-scoring eleven times, and that reuse is only valid if this holds.
    """
    rows = txns(900, fraud_every=6)
    supervised = LGBMDetector(seed=12, params=FAST).fit(rows[:700])
    unsupervised = AnomalyDetector(seed=12).fit(rows[:700])
    holdout = rows[700:]

    for weight in (0.0, 0.3, 0.7, 1.0):
        ensemble = EnsembleDetector(supervised, unsupervised, weight=weight)
        _, blended = protocol.align(holdout, ensemble.score(as_batch(holdout)))
        expected = weight * supervised.predict_proba(holdout) + (
            1 - weight
        ) * unsupervised.predict_proba(holdout)
        assert np.allclose(blended, expected, atol=1e-12)


def test_the_blend_endpoints_are_the_halves_themselves():
    rows = txns(900, fraud_every=6)
    supervised = LGBMDetector(seed=12, params=FAST).fit(rows[:700])
    unsupervised = AnomalyDetector(seed=12).fit(rows[:700])
    holdout = rows[700:]

    _, sup_only = protocol.align(
        holdout, EnsembleDetector(supervised, unsupervised, weight=1.0).score(as_batch(holdout))
    )
    _, uns_only = protocol.align(
        holdout, EnsembleDetector(supervised, unsupervised, weight=0.0).score(as_batch(holdout))
    )
    assert np.allclose(sup_only, supervised.predict_proba(holdout))
    assert np.allclose(uns_only, unsupervised.predict_proba(holdout))


def test_a_blend_weight_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        EnsembleDetector(LGBMDetector(seed=1), AnomalyDetector(seed=1), weight=1.5)


def test_the_ensembles_model_card_carries_the_half_that_produced_a_third_of_the_score():
    """It used to report its supervised half's card under the ensemble's name."""
    rows = txns(600, fraud_every=6)
    ensemble = EnsembleDetector(
        LGBMDetector(seed=12, params=FAST), AnomalyDetector(seed=12), weight=0.7
    ).fit(rows)

    card = model_card_of(ensemble)
    assert card["detector"] == "EnsembleDetector"
    assert card["weight"] == 0.7
    assert card["unsupervised"]["training"]["legit_only"] is True
    assert card["unsupervised"]["training"]["n_fraud_excluded"] > 0
    assert card["supervised"]["backend"]["name"] != "untrained"


# ── the comparison the ticket is actually about ─────────────────────────────────
def test_both_layers_are_scored_on_the_same_unseen_family_at_one_operating_point():
    """The shape of ticket 10's table: one fold, two detectors, neither trained on the family."""
    seen = txns(700, fraud_every=6, vector="S1")
    unseen = txns(120, fraud_every=2, start=T0 + timedelta(days=9), prefix="m", vector="M3")
    train, holdout = make_splits(seen + unseen, held_out_vector="M3", train_frac=0.7)

    assert not any(t.vector_id == "M3" for t in train)
    assert any(t.is_fraud and t.vector_id == "M3" for t in holdout)
    assert all(t.vector_id == "M3" for t in holdout if t.is_fraud)

    supervised = LGBMDetector(seed=12, params=FAST).fit(train)
    unsupervised = AnomalyDetector(seed=12).fit(train)
    assert unsupervised.training.n_fraud_excluded == sum(1 for t in train if t.is_fraud)

    results = {
        name: protocol.evaluate_detector(d, holdout, held_out_vector="M3")
        for name, d in (("supervised", supervised), ("anomaly", unsupervised))
    }
    for name, result in results.items():
        assert result.held_out_vector == "M3", name
        assert result.fixed_fpr == protocol.DEFAULT_FPR, name
        assert result.n_positives == sum(1 for t in holdout if t.is_fraud), name


def test_the_injected_family_is_not_separable_from_its_anchor_by_account_id():
    """The fold ticket 10 reports on has to be commensurable, or the table measures provenance.

    PaySim's senders are effectively unique per row, so the envelope's "seasoned accounts" filter
    returned one of them and the simulator minted `e00042`-style ids for the rest of the
    population. `sender_in_anchor` then separated the held-out family from the anchor at PR-AUC
    1.000 — a perfect label, in a fold whose whole purpose is to measure generalisation.
    """
    anchor = [
        Transaction(
            txn_id=f"real{i:05d}",
            ts=T0 + timedelta(minutes=7 * i),
            src=f"C{i:07d}",  # every sender distinct, as on PaySim
            dst=f"M{i % 200:05d}",
            amount=100.0 + (i % 500),
            rail=Rail.A2A,
        )
        for i in range(4_000)
    ]
    envelope = AnchorEnvelope.measure(anchor, "unique-senders")
    simulator = Simulator(seed=3, n_entities=400, n_background=0, envelope=envelope)
    batch = simulator.generate(registry.get("M3").to_attack_params())
    fraud = [t for t in batch.transactions if t.is_fraud]

    assert fraud, "no episode generated — this test would pass vacuously"
    senders = {t.src for t in anchor}
    assert all(t.src in senders for t in fraud), "an attack ran on an account the anchor never saw"
    assert audit(anchor, fraud)["signals"]["sender_in_anchor"] < 0.5
