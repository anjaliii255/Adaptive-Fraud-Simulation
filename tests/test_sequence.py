"""■ B — the sequence model reads history, not the answer key, and it earns its seat or it does not.

Ticket 17's failure modes are the quiet kind, which is why they are tests rather than a review:

  * a window peeks forward, or reaches into another entity's history, and the offline table is
    beautiful while the deployment is worthless;
  * the label is per *entity* rather than per row — the window containing the fraud predicts that
    the window contains fraud, and the score lands on the account's clean baseline rows too;
  * the scoring-time window has no history in it, so the one arc this model exists for is the one
    it cannot see;
  * torch is missing and the layer degrades to zeros, which reads in a metric exactly like a
    detector that caught nothing;
  * the two ends of the drift axis are averaged into one number, so nobody can tell which of them
    paid for the result;
  * the gate promotes on the easy end, or promotes on a fold that cannot carry a claim at all;
  * `defend.sequence.enabled` drifts away from the evidence that was supposed to decide it.

The reported numbers are built by `scripts/build_sequence.py`, never here.
"""

from __future__ import annotations

import json
import zlib
from datetime import datetime, timedelta

import numpy as np
import pytest
import yaml

from afl.contract.metrics import Action, MetricResult
from afl.contract.schema import AttackBatch, AttackParams, Rail, Transaction
from afl.defend import explain
from afl.defend.models import sequence as seq
from afl.defend.models.sequence import SequenceDetector
from afl.evaluation import drift_arc, protocol
from afl.evaluation.leave_one_attack_out import MEASURED, SKIPPED, WITHHELD

#: Only the tests that actually *build* a network need the extra. The window arithmetic, the arc
#: breakdown, the gate and the artefact are pure numpy and are the parts most likely to break
#: quietly, so they run on the default torch-free install rather than being skipped with the rest.
needs_torch = pytest.mark.skipif(
    not seq.available(), reason="the sequence model needs the `deep` extra: uv sync --extra deep"
)

T0 = datetime(2024, 1, 1)


def row(
    i: int,
    src: str,
    minutes: float,
    amount: float = 100.0,
    dst: str | None = None,
    device: str | None = None,
    fraud: bool = False,
    vector: str | None = None,
    run: str | None = None,
) -> Transaction:
    return Transaction(
        txn_id=f"t{i:05d}",
        ts=T0 + timedelta(minutes=minutes),
        src=src,
        dst=dst or f"m{i % 4}",
        amount=amount,
        rail=Rail.A2A,
        device_id=device,
        is_fraud=fraud,
        vector_id=vector,
        attack_run_id=run,
    )


def drift_rows(
    n_accounts: int = 40, n_baseline: int = 20, n_post: int = 10, seed: int = 5
) -> list[Transaction]:
    """Accounts with a quiet baseline; the back half then escalates and is labelled fraud.

    The tell lives only in the trajectory: every fraud row's amount is inside the population's
    ordinary range, and it is anomalous only against its own account's recent normal.
    """
    rng = np.random.default_rng(seed)
    rows, i = [], 0
    for a in range(n_accounts):
        drifts = a >= n_accounts // 2
        base = float(rng.uniform(60, 140))
        ts = float(rng.uniform(0, 600))
        for j in range(n_baseline + n_post):
            ts += float(rng.exponential(45))
            shift = 1.0 + 2.0 * max(0, j - n_baseline + 1) / n_post if drifts else 1.0
            rows.append(
                row(
                    i,
                    f"a{a:03d}",
                    ts,
                    amount=round(base * shift * float(rng.lognormal(0, 0.2)), 2),
                    fraud=bool(drifts and j >= n_baseline),
                    vector="S3" if (drifts and j >= n_baseline) else None,
                )
            )
            i += 1
    rows.sort(key=lambda t: t.ts)
    return rows


def as_batch(rows: list[Transaction], run_id: str = "r") -> AttackBatch:
    return AttackBatch(
        run_id=run_id,
        params=AttackParams(vector_id="eval", engine="none"),
        transactions=list(rows),
        seed=0,
    )


FAST = {"epochs": 3, "hidden": 16, "max_len": 12, "batch_size": 64}


# ── the window is causal, per entity, and ends at the row being judged ───────────
def test_a_window_holds_this_entity_and_only_this_entity():
    rows = [row(0, "a", 0), row(1, "b", 1), row(2, "a", 5), row(3, "b", 6), row(4, "a", 9)]
    raw = seq.raw_rows(rows)
    keys = np.array([zlib.crc32(t.src.encode()) for t in rows], dtype=np.int64)
    idx = seq.window_index(keys, raw[:, seq.RAW_TS], max_len=4)

    for i, t in enumerate(rows):
        steps = [j for j in idx[i] if j >= 0]
        assert steps[-1] == i, "the row being judged has to be the last step of its own window"
        assert all(rows[j].src == t.src for j in steps), "a window reached into another entity"
        assert all(rows[j].ts <= t.ts for j in steps), "a window peeked forward"
    assert [j for j in idx[4] if j >= 0] == [0, 2, 4]


def test_a_window_is_capped_and_keeps_the_most_recent_steps():
    rows = [row(i, "a", i) for i in range(10)]
    raw = seq.raw_rows(rows)
    keys = np.array([zlib.crc32(t.src.encode()) for t in rows], dtype=np.int64)
    idx = seq.window_index(keys, raw[:, seq.RAW_TS], max_len=3)
    assert [j for j in idx[9] if j >= 0] == [7, 8, 9]


def test_a_later_row_cannot_change_an_earlier_row_s_window():
    """The property that makes the offline number mean anything: no lookahead, at all."""
    before = [row(i, "a", i * 10, amount=100.0) for i in range(5)]
    after = [*before, row(99, "a", 100, amount=99_000.0, fraud=True)]

    def window_of(rows, i):
        raw = seq.raw_rows(rows)
        keys = np.array([zlib.crc32(t.src.encode()) for t in rows], dtype=np.int64)
        idx = seq.window_index(keys, raw[:, seq.RAW_TS], max_len=8)
        X, mask = seq.step_tensor(raw, idx[i : i + 1])
        return X[0][mask[0]]

    assert np.allclose(window_of(before, 4), window_of(after, 4))


def test_step_features_are_computed_against_earlier_steps_only():
    rows = [
        row(0, "a", 0, amount=100.0, dst="m1", device="d1"),
        row(1, "a", 10, amount=100.0, dst="m1", device="d1"),
        row(2, "a", 20, amount=800.0, dst="m9", device="d2"),
    ]
    raw = seq.raw_rows(rows)
    keys = np.array([zlib.crc32(t.src.encode()) for t in rows], dtype=np.int64)
    idx = seq.window_index(keys, raw[:, seq.RAW_TS], max_len=4)
    X, mask = seq.step_tensor(raw, idx[2:3])
    names = list(seq.STEP_FEATURES)
    steps = X[0][mask[0]]

    assert steps[0][names.index("amount_vs_window_running_mean")] == pytest.approx(
        0.0
    ), "the first step of a window has nothing before it to deviate from"
    assert steps[-1][names.index("amount_vs_window_running_mean")] > 1.0
    assert steps[-1][names.index("beneficiary_new_in_window")] == 1.0
    assert steps[1][names.index("beneficiary_new_in_window")] == 0.0, "m1 was already seen"
    assert steps[-1][names.index("device_changed")] == 1.0
    assert steps[1][names.index("log_gap_since_previous")] > 0.0


def test_an_anchor_with_no_device_column_says_so_rather_than_saying_unchanged():
    rows = [row(i, "a", i * 10) for i in range(3)]  # device_id is None throughout
    raw = seq.raw_rows(rows)
    keys = np.array([zlib.crc32(t.src.encode()) for t in rows], dtype=np.int64)
    idx = seq.window_index(keys, raw[:, seq.RAW_TS], max_len=4)
    X, mask = seq.step_tensor(raw, idx[2:3])
    changed = X[0][mask[0]][:, list(seq.STEP_FEATURES).index("device_changed")]
    assert np.all(changed == seq.NEVER), "'no device to change' is not 'the device did not change'"


def test_entity_ids_hash_the_same_way_in_any_process():
    """`hash()` is salted per interpreter; a window built from it would not reproduce."""
    assert seq._key("beneficiary-42") == float(zlib.crc32(b"beneficiary-42"))
    assert seq._key(None) == seq.NEVER


# ── the label is per row, not per entity ────────────────────────────────────────
@needs_torch
def test_a_clean_row_on_a_drifting_account_is_not_scored_as_fraud():
    """The bug the entity-level `sequence_tensor` had: one label per window, broadcast back.

    An account's pre-takeover rows are legitimate and an investigator would call them so. A model
    that scores them like the post-takeover tail is not detecting anything; it is answering a
    question about the account.
    """
    rows = drift_rows()
    cut = int(len(rows) * 0.6)
    train, holdout = rows[:cut], rows[cut:]
    detector = SequenceDetector(seed=3, **FAST).fit(train)

    probs = detector.predict_proba(holdout)
    drifting = {t.src for t in holdout if t.is_fraud}
    clean = np.array(
        [p for t, p in zip(holdout, probs, strict=False) if t.src in drifting and not t.is_fraud]
    )
    dirty = np.array([p for t, p in zip(holdout, probs, strict=False) if t.is_fraud])
    assert clean.size and dirty.size, "this test would pass vacuously without both populations"
    assert (
        dirty.mean() > clean.mean()
    ), "the same account's clean and fraudulent rows scored the same — the label is per entity"


@needs_torch
def test_the_arc_is_learnable_at_all():
    """A guard on every other test in this file: a model that learns nothing proves nothing."""
    rows = drift_rows()
    cut = int(len(rows) * 0.6)
    detector = SequenceDetector(seed=3, **FAST).fit(rows[:cut])
    y, s = protocol.align(rows[cut:], detector.score(as_batch(rows[cut:])))
    assert protocol.pr_auc(y, s) > 2 * float(y.mean()), "no signal recovered from a planted drift"


# ── history crosses the fit/score boundary ──────────────────────────────────────
@needs_torch
def test_scoring_carries_the_history_the_fit_saw():
    """Without this the arc is invisible: the baseline is in train, the escalation is in test."""
    train = [row(i, "a", i * 30, amount=100.0) for i in range(10)]
    train.append(row(50, "b", 5, amount=100.0, fraud=True, vector="S3"))
    later = [row(20, "a", 400, amount=100.0)]

    detector = SequenceDetector(seed=3, **FAST).fit(train)
    _, idx, lengths = detector._layout(later, with_history=True)
    assert lengths[0] > 1, "a holdout row was judged with no history at all"
    _, _, bare = detector._layout(later, with_history=False)
    assert bare[0] == 1


@needs_torch
def test_history_is_capped_at_max_len():
    train = [row(i, "a", i * 30) for i in range(50)]
    train.append(row(90, "b", 5, fraud=True, vector="S3"))
    detector = SequenceDetector(seed=3, **FAST).fit(train)
    assert len(detector._tails["a"]) == FAST["max_len"]


@needs_torch
def test_chunking_the_forward_pass_cannot_change_what_a_row_sees(monkeypatch):
    """Windows are laid out over the whole call; only the network is chunked."""
    rows = drift_rows(n_accounts=12)
    detector = SequenceDetector(seed=3, **FAST).fit(rows[: int(len(rows) * 0.6)])
    holdout = rows[int(len(rows) * 0.6) :]
    whole = detector.predict_proba(holdout)
    monkeypatch.setattr(seq, "SCORE_CHUNK", 7)
    assert np.allclose(whole, detector.predict_proba(holdout), atol=1e-6)


# ── the seam ────────────────────────────────────────────────────────────────────
@needs_torch
def test_it_scores_through_the_standard_seam_and_explains_what_it_flags():
    rows = drift_rows()
    cut = int(len(rows) * 0.6)
    detector = SequenceDetector(seed=3, **FAST).fit(rows[:cut])
    scores = detector.score(as_batch(rows[cut:]))

    assert [s.txn_id for s in scores] == [t.txn_id for t in rows[cut:]]
    assert all(0.0 <= s.score <= 1.0 for s in scores)
    assert all(isinstance(s.action, Action) for s in scores)
    explain.assert_flagged_rows_are_explained(scores)
    flagged = [s for s in scores if s.action is not Action.ALLOW]
    assert flagged, "this test would pass vacuously if nothing were flagged"
    assert any("sequence:gru" in r for r in flagged[0].reasons)


@needs_torch
def test_the_leave_one_attack_out_guard_can_audit_what_it_trained_on():
    from afl.evaluation.leave_one_attack_out import assert_family_held_out, training_rows

    rows = drift_rows(n_accounts=12)
    detector = SequenceDetector(seed=3, **FAST).fit(rows)
    assert len(training_rows(detector)) == len(rows)
    with pytest.raises(AssertionError):
        assert_family_held_out([], "S3", detector)


@needs_torch
def test_a_retrain_accumulates_rather_than_forgetting():
    rows = drift_rows(n_accounts=12)
    first, second = rows[: len(rows) // 2], rows[len(rows) // 2 :]
    detector = SequenceDetector(seed=3, **FAST).fit(first)
    evaded = [t for t in second if t.is_fraud][:2]
    detector.retrain(as_batch(second), evaded)
    assert len(detector.training_rows) >= len(rows)
    assert all(t.txn_id in {r.txn_id for r in detector.training_rows} for t in evaded)


# ── it raises rather than degrading ─────────────────────────────────────────────
def test_without_torch_the_constructor_refuses(monkeypatch):
    """A detector that silently scores zeros reads in a metric exactly like one that caught none."""

    def no_torch():
        raise ImportError(seq.TORCH_HINT)

    monkeypatch.setattr(seq, "require_torch", no_torch)
    with pytest.raises(ImportError, match="deep"):
        SequenceDetector()


@needs_torch
def test_scoring_before_fitting_raises_rather_than_returning_zeros():
    detector = SequenceDetector(seed=3, **FAST)
    with pytest.raises(RuntimeError, match="caught nothing"):
        detector.score(as_batch(drift_rows(n_accounts=4)))


@needs_torch
def test_a_single_class_training_set_is_refused():
    rows = [row(i, "a", i * 10) for i in range(20)]
    with pytest.raises(ValueError, match="single-class"):
        SequenceDetector(seed=3, **FAST).fit(rows)


def test_an_unknown_architecture_is_refused():
    with pytest.raises(ValueError, match="unknown arch"):
        SequenceDetector(arch="lstm")


# ── the compute cost travels with the lift ──────────────────────────────────────
@needs_torch
def test_the_model_card_prices_the_model():
    rows = drift_rows(n_accounts=12)
    detector = SequenceDetector(seed=3, **FAST).fit(rows)
    detector.score(as_batch(rows))
    card = detector.model_card()
    assert card["compute"]["fit_seconds"] > 0
    assert card["compute"]["n_parameters"] > 0
    assert card["compute"]["scored_rows"] == len(rows)
    assert card["training"]["n_negatives_sampled"] > 0
    assert card["history_coverage"]["mean_window_length"] > 1


@needs_torch
def test_history_coverage_notices_an_anchor_with_no_per_entity_history():
    """PaySim's shape: every sender appears once, so every window is one step long."""
    rows = [
        row(i, f"s{i}", i, fraud=i % 9 == 0, vector="S3" if i % 9 == 0 else None) for i in range(60)
    ]
    detector = SequenceDetector(seed=3, **FAST).fit(rows)
    assert detector.coverage.share_length_one == 1.0
    assert detector.coverage.mean_length == 1.0


# ── the axis ────────────────────────────────────────────────────────────────────
def test_the_two_arcs_come_from_the_vector_s_own_envelope():
    """An arc generated outside the declared search space is a family the vector is not."""
    for family in drift_arc.DRIFT_ARC_FAMILIES:
        space = drift_arc.registry.get(family).search_space["ramp"]
        assert drift_arc.arc_ramp(family, drift_arc.SUDDEN) == space["low"]
        assert drift_arc.arc_ramp(family, drift_arc.GRADUAL) == space["high"]
    assert drift_arc.arc_ramp("C1", drift_arc.GRADUAL) < drift_arc.arc_ramp("S3", drift_arc.GRADUAL)


def test_a_vector_with_no_ramp_has_no_axis():
    with pytest.raises(ValueError, match="sudden-vs-gradual"):
        drift_arc.arc_ramp("S1", drift_arc.SUDDEN)


def test_arcs_are_tagged_by_the_run_that_generated_them():
    rows = [
        row(0, "a", 0, fraud=True, vector="S3", run="S3-000-aaaa"),
        row(1, "a", 1, fraud=True, vector="S3", run="S3-001-bbbb"),
        row(2, "a", 2),  # legit, no run id
    ]
    tags = drift_arc.tag_arcs(rows, {"S3-000-aaaa": "sudden", "S3-001-bbbb": "gradual"})
    assert tags == {"t00000": "sudden", "t00001": "gradual"}


def _arc_pool(n_legit: int = 400, n_sudden: int = 40, n_gradual: int = 40):
    rows = [row(i, f"s{i % 20}", i) for i in range(n_legit)]
    rows += [
        row(1000 + i, "x", i, fraud=True, vector="S3", run="run-sudden") for i in range(n_sudden)
    ]
    rows += [
        row(2000 + i, "y", i, fraud=True, vector="S3", run="run-gradual") for i in range(n_gradual)
    ]
    return rows, {"run-sudden": drift_arc.SUDDEN, "run-gradual": drift_arc.GRADUAL}


def test_the_two_arcs_are_reported_separately_against_the_same_haystack():
    """The whole point of the ticket: an average over the axis hides which end paid."""
    rows, runs = _arc_pool()
    arcs = drift_arc.tag_arcs(rows, runs)
    # sudden is caught, gradual is missed — the shape a per-row feature table produces
    probs = np.array(
        [0.9 if arcs.get(t.txn_id) == drift_arc.SUDDEN else 0.1 for t in rows], dtype=float
    )
    out = drift_arc.arc_breakdown(rows, probs, arcs, {"sudden": 0.0, "gradual": 1.0})

    assert out[drift_arc.SUDDEN].outcome == MEASURED
    assert out[drift_arc.GRADUAL].outcome == MEASURED
    assert out[drift_arc.SUDDEN].metrics.pr_auc > out[drift_arc.GRADUAL].metrics.pr_auc
    assert out[drift_arc.SUDDEN].recall_at_shared_threshold == 1.0
    assert out[drift_arc.GRADUAL].recall_at_shared_threshold == 0.0


def test_both_arcs_are_ranked_against_every_legit_row_of_the_fold():
    rows, runs = _arc_pool(n_legit=400, n_sudden=40, n_gradual=60)
    arcs = drift_arc.tag_arcs(rows, runs)
    probs = np.linspace(0, 1, len(rows))
    out = drift_arc.arc_breakdown(rows, probs, arcs, {"sudden": 0.0, "gradual": 1.0})
    legit = sum(1 for t in rows if not t.is_fraud)
    for arc, result in out.items():
        assert result.metrics.n_positives == sum(1 for v in arcs.values() if v == arc)
        assert result.metrics.k == min(protocol.DEFAULT_K, legit + result.n_positives)


def test_one_threshold_for_both_arcs():
    rows, _ = _arc_pool()
    probs = np.array([0.5 if t.is_fraud else 0.1 for t in rows], dtype=float)
    cut = drift_arc.shared_threshold(rows, probs, fixed_fpr=0.01)
    assert 0.1 <= cut < 0.5, "the cut is set by the legit rows of the whole fold, not by an arc"


def test_the_two_arcs_are_read_at_one_operating_point():
    """The invariant behind the `one-threshold recall` column: the same cut for both arcs.

    Each arc keeps every legit row of the fold, so the 1%-FPR quantile is the same score in both
    rows and the two recall columns coincide. A divergence would mean the haystack stopped being
    shared — which is how one comparison quietly becomes two.
    """
    rows, runs = _arc_pool(n_legit=800, n_sudden=60, n_gradual=60)
    arcs = drift_arc.tag_arcs(rows, runs)
    rng = np.random.default_rng(11)
    probs = np.array([rng.uniform(0.4, 1.0) if t.is_fraud else rng.uniform(0.0, 0.6) for t in rows])
    out = drift_arc.arc_breakdown(rows, probs, arcs, {})
    for result in out.values():
        assert result.metrics.recall_at_fixed_fpr == pytest.approx(
            result.recall_at_shared_threshold
        )


def test_a_thin_arc_is_withheld_rather_than_scored_low():
    rows, runs = _arc_pool(n_sudden=40, n_gradual=5)
    arcs = drift_arc.tag_arcs(rows, runs)
    out = drift_arc.arc_breakdown(rows, np.linspace(0, 1, len(rows)), arcs, {}, min_positives=30)
    assert out[drift_arc.GRADUAL].outcome == WITHHELD
    assert out[drift_arc.GRADUAL].metrics is None
    assert out[drift_arc.GRADUAL].withheld_metrics is not None
    assert "floor of 30" in out[drift_arc.GRADUAL].reason


def test_an_arc_with_nothing_in_the_holdout_is_skipped_not_scored_as_zero():
    rows, runs = _arc_pool(n_gradual=0)
    arcs = drift_arc.tag_arcs(rows, runs)
    out = drift_arc.arc_breakdown(rows, np.linspace(0, 1, len(rows)), arcs, {})
    assert out[drift_arc.GRADUAL].outcome == SKIPPED
    assert out[drift_arc.GRADUAL].any_metrics is None


def test_an_arc_cannot_carry_numbers_it_is_not_allowed_to_quote():
    with pytest.raises(ValueError, match="withheld_metrics"):
        drift_arc.ArcResult(
            arc="gradual",
            ramp=1.0,
            outcome=WITHHELD,
            reason="thin",
            metrics=MetricResult(
                pr_auc=0.9, recall_at_fixed_fpr=0.9, fixed_fpr=0.01, precision_at_k=0.9
            ),
        )


# ── the precondition ────────────────────────────────────────────────────────────
def _measured(arc: str, pr_auc: float) -> drift_arc.ArcResult:
    return drift_arc.ArcResult(
        arc=arc,
        ramp=1.0,
        outcome=MEASURED,
        metrics=MetricResult(
            pr_auc=pr_auc, recall_at_fixed_fpr=pr_auc, fixed_fpr=0.01, precision_at_k=pr_auc
        ),
        n_positives=100,
    )


def test_window_length_alone_can_be_the_label_and_the_audit_says_so():
    """PaySim's shape: real senders appear once, injected episodes carry a whole arc.

    The injected share is kept near the real thing (~1%), because `is_provenance_bound`'s bar is
    a floor *or* a multiple of the base rate — a fixture with 40% positives would be judged at a
    bar no score can reach, which is the opposite of what this is checking.
    """
    real = [row(i, f"s{i}", i) for i in range(2_000)]
    injected = [
        row(10_000 + a * 12 + j, f"x{a}", 3_000 + j, fraud=True, vector="S3")
        for a in range(2)
        for j in range(12)
    ]
    audit = drift_arc.history_audit(real, injected, max_len=32)
    assert audit["separable"] is True
    assert audit["anchor_mean_window"] == pytest.approx(1.0)
    assert audit["anchor_share_with_no_history"] == pytest.approx(1.0)
    assert audit["injected_mean_window"] > 5


def test_an_anchor_with_real_histories_passes_the_audit():
    """AMLSim's shape: the anchor's accounts are already deeper than the window, so an injected
    episode is not visible in the window length at all."""
    real = [row(a * 50 + j, f"s{a:02d}", j, amount=100.0) for a in range(40) for j in range(50)]
    injected = [
        row(10_000 + a * 12 + j, f"s{a:02d}", 3_000 + j, fraud=True, vector="S3")
        for a in range(2)
        for j in range(12)
    ]
    audit = drift_arc.history_audit(real, injected, max_len=32)
    assert audit["separable"] is False
    assert audit["anchor_share_with_no_history"] < 0.05, "the anchor has real history to read"


def test_the_audit_says_so_rather_than_guessing_when_one_side_is_empty():
    assert drift_arc.history_audit([], [row(0, "a", 0)], max_len=8)["checked"] is False


# ── the gate ────────────────────────────────────────────────────────────────────
def test_a_win_on_gradual_drift_promotes_the_model():
    verdict = drift_arc.decide_promotion(
        challenger={drift_arc.GRADUAL: _measured("gradual", 0.80)},
        champion={drift_arc.GRADUAL: _measured("gradual", 0.60)},
        floor=0.10,
    )
    assert verdict.promoted is True
    assert verdict.margin == pytest.approx(0.20)


def test_a_win_on_sudden_drift_alone_does_not_promote_it():
    """Sudden takeover is an event a per-row table already sees. That end is not the argument."""
    verdict = drift_arc.decide_promotion(
        challenger={
            drift_arc.SUDDEN: _measured("sudden", 0.95),
            drift_arc.GRADUAL: _measured("gradual", 0.40),
        },
        champion={
            drift_arc.SUDDEN: _measured("sudden", 0.60),
            drift_arc.GRADUAL: _measured("gradual", 0.60),
        },
        floor=0.10,
    )
    assert verdict.promoted is False
    assert "loses to" in verdict.reason


def test_a_tie_is_not_a_win():
    verdict = drift_arc.decide_promotion(
        challenger={drift_arc.GRADUAL: _measured("gradual", 0.605)},
        champion={drift_arc.GRADUAL: _measured("gradual", 0.600)},
        floor=0.10,
    )
    assert verdict.promoted is False
    assert "is level with" in verdict.reason


def test_a_model_the_amount_floor_beats_is_not_promoted():
    verdict = drift_arc.decide_promotion(
        challenger={drift_arc.GRADUAL: _measured("gradual", 0.30)},
        champion={drift_arc.GRADUAL: _measured("gradual", 0.10)},
        floor=0.40,
    )
    assert verdict.promoted is False
    assert "sorting by amount" in verdict.reason


def test_a_fold_that_cannot_carry_a_claim_cannot_promote_anything():
    verdict = drift_arc.decide_promotion(
        challenger={drift_arc.GRADUAL: _measured("gradual", 0.99)},
        champion={drift_arc.GRADUAL: _measured("gradual", 0.10)},
        floor=0.05,
        blocked="window length alone sorts the injected rows from the anchor's own",
    )
    assert verdict.promoted is False
    assert "window length" in verdict.reason


def test_a_thin_arc_cannot_promote_anything_either():
    thin = drift_arc.ArcResult(
        arc="gradual", ramp=1.0, outcome=WITHHELD, reason="4 positives against a floor of 30"
    )
    verdict = drift_arc.decide_promotion(
        challenger={drift_arc.GRADUAL: thin},
        champion={drift_arc.GRADUAL: _measured("gradual", 0.10)},
    )
    assert verdict.promoted is False
    assert "cannot carry a claim" in verdict.reason


# ── the artefact, and the config it is answerable to ────────────────────────────
def _report(dataset: str, promoted: bool) -> drift_arc.SequenceReport:
    fold = drift_arc.ArcFold(
        held_out_vector="S3",
        outcome=MEASURED if promoted else WITHHELD,
        reason="" if promoted else "the fold cannot carry a claim",
        promotion=drift_arc.Promotion(
            promoted=promoted, reason="measured, and it " + ("won" if promoted else "lost")
        ),
        systems={
            "sequence": drift_arc.SystemResult(
                name="sequence",
                overall=MetricResult(
                    pr_auc=0.5, recall_at_fixed_fpr=0.5, fixed_fpr=0.01, precision_at_k=0.5
                ),
                arcs={drift_arc.GRADUAL: _measured("gradual", 0.5)},
                compute={"fit_seconds": 12.0},
            )
        },
        ramps={"sudden": 0.0, "gradual": 1.0},
    )
    return drift_arc.SequenceReport(
        dataset=dataset,
        seed=1337,
        config={"arch": "gru"},
        operating_point={"fixed_fpr": 0.01, "k": 100},
        folds=[fold],
    )


def test_the_report_round_trips_through_disk(tmp_path):
    path = _report("amlsim", promoted=False).save(tmp_path)
    back = drift_arc.SequenceReport.load("amlsim", tmp_path)
    assert back.folds[0].systems["sequence"].compute["fit_seconds"] == 12.0
    assert back.folds[0].systems["sequence"].arcs[drift_arc.GRADUAL].metrics.pr_auc == 0.5
    assert back.promoted is False
    assert json.loads(path.read_text())["version"] == drift_arc.SEQUENCE_ARTEFACT_VERSION


def test_an_artefact_from_another_shape_fails_loudly(tmp_path):
    raw = json.loads(_report("amlsim", promoted=False).save(tmp_path).read_text())
    (tmp_path / "amlsim.json").write_text(json.dumps({**raw, "version": 99}))
    with pytest.raises(ValueError, match="rebuild it"):
        drift_arc.SequenceReport.load("amlsim", tmp_path)


def test_a_report_with_no_folds_is_refused():
    with pytest.raises(ValueError, match="not a result"):
        drift_arc.SequenceReport(dataset="amlsim", seed=1, config={}, operating_point={}, folds=[])


def test_enabling_the_layer_against_the_committed_evidence_is_refused():
    refused = {"amlsim": _report("amlsim", promoted=False)}
    drift_arc.assert_config_matches_promotion(enabled=False, reports=refused)
    with pytest.raises(AssertionError, match="does not support it"):
        drift_arc.assert_config_matches_promotion(enabled=True, reports=refused)
    drift_arc.assert_config_matches_promotion(
        enabled=True, reports={"amlsim": _report("amlsim", promoted=True)}
    )


def test_enabling_the_layer_with_nothing_measured_is_refused():
    with pytest.raises(AssertionError, match="no committed sequence-model report"):
        drift_arc.assert_config_matches_promotion(enabled=True, reports={})


def test_the_shipped_config_agrees_with_the_committed_evidence():
    """The ticket's fourth criterion, checked against what is actually on disk right now."""
    cfg = yaml.safe_load(open("config/defend/sequence.yaml"))
    drift_arc.assert_config_matches_promotion(enabled=bool(cfg["enabled"]))
