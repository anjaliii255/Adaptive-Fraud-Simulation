"""■ B — the temporal GNN reads the graph that was there before, and earns its seat or does not.

Ticket 18's failure modes are the quiet kind, which is why they are tests rather than a review:

  * the snapshot contains the payment it is scoring, or one after it, and the offline table is
    beautiful while the deployment is worthless;
  * the label is per *node* rather than per row — the beneficiary of a fraud row is marked
    positive and the score lands on that account's legitimate inbound payments too, which is the
    exact bug the previous version of `afl/defend/models/gnn.py` shipped;
  * the scoring-time graph is rebuilt from the batch alone, so a holdout row's beneficiary
    arrives with no neighbourhood on the one family where the neighbourhood is the signal;
  * the deep extra is missing and the layer degrades to zeros, which reads in a metric exactly
    like a detector that caught nothing;
  * a lift is quoted from one seed, or from a margin smaller than its own seed-to-seed spread;
  * the fold cannot see the motif form at all, or the injected ring is its own synthetic island,
    and a number gets reported from it anyway;
  * `defend.gnn.enabled` — or the README — drifts away from the evidence that decided it.

The reported numbers are built by `scripts/build_gnn.py`, never here.
"""

from __future__ import annotations

import json
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from afl.contract.metrics import Action, MetricResult
from afl.contract.schema import AttackBatch, AttackParams, Rail, Transaction
from afl.defend import explain
from afl.defend.features import (
    FeatureBuilder,
    GraphFeatureBuilder,
    feature_names,
    graph_feature_names,
)
from afl.defend.models import gnn as gnn_mod
from afl.defend.models.gnn import TemporalGNNDetector
from afl.evaluation import mule_graph, protocol
from afl.evaluation.leave_one_attack_out import MEASURED, SKIPPED, WITHHELD
from afl.evaluation.three_system import Comparison, Spread

#: Only the tests that actually *build* a network need the extra. The window arithmetic, the two
#: audits, the seed aggregation, the gate and the artefact are pure numpy and are the parts most
#: likely to break quietly, so they run on the default install rather than being skipped with the
#: rest.
needs_deep = pytest.mark.skipif(
    not gnn_mod.available(),
    reason="the temporal GNN needs the `deep` extra: uv sync --extra deep",
)

T0 = datetime(2024, 1, 1)
HOUR = 3_600.0
DAY = 86_400.0

FAST = {"epochs": 3, "hidden": 8, "heads": 2, "window_hours": 72, "stride_hours": 24}


def row(
    i: int,
    src: str,
    dst: str,
    hours: float,
    amount: float = 100.0,
    fraud: bool = False,
    vector: str | None = None,
    run: str | None = None,
) -> Transaction:
    return Transaction(
        txn_id=f"t{i:05d}",
        ts=T0 + timedelta(hours=hours),
        src=src,
        dst=dst,
        amount=amount,
        rail=Rail.A2A,
        is_fraud=fraud,
        vector_id=vector,
        attack_run_id=run,
    )


def ring_rows(
    n_background: int = 2_000, n_rings: int = 24, seed: int = 5
) -> list[Transaction]:
    """Ordinary traffic, plus mule rings: several payers into a collector, then a hop out.

    The tell is topology alone. Every amount is drawn from the same distribution as the
    background, so a row taken on its own is unremarkable and only the fan-in is wrong.
    """
    rng = np.random.default_rng(seed)
    rows, i = [], 0
    for _ in range(n_background):
        i += 1
        rows.append(
            row(
                i,
                f"a{rng.integers(0, 200):03d}",
                f"m{rng.integers(0, 30):02d}",
                float(rng.uniform(0, 24 * 20)),
                amount=float(rng.lognormal(4, 0.6)),
            )
        )
    for r in range(n_rings):
        start = float(rng.uniform(0, 24 * 18))
        collector = f"a{rng.integers(0, 200):03d}"
        for s in range(6):
            i += 1
            rows.append(
                row(
                    i,
                    f"a{rng.integers(0, 200):03d}",
                    collector,
                    start + 6 * s,  # spread over strides, so the ring is visible as it forms
                    amount=float(rng.lognormal(4, 0.5)),
                    fraud=True,
                    vector="S1",
                    run=f"r{r}",
                )
            )
        i += 1
        rows.append(
            row(
                i,
                collector,
                f"exit{r:02d}",
                start + 40,
                amount=float(rng.lognormal(6, 0.3)),
                fraud=True,
                vector="S1",
                run=f"r{r}",
            )
        )
    rows.sort(key=lambda t: t.ts)
    return rows


def as_batch(rows: list[Transaction], run_id: str = "r") -> AttackBatch:
    return AttackBatch(
        run_id=run_id,
        params=AttackParams(vector_id="eval", engine="none"),
        transactions=list(rows),
        seed=0,
    )


def snapshot_of(rows: list[Transaction], end_ts: float, window_s: float) -> gnn_mod.Snapshot:
    raw = gnn_mod.raw_rows(rows)
    order = np.argsort(raw[:, gnn_mod.RAW_TS], kind="stable")
    return gnn_mod.build_snapshot(
        raw,
        order,
        raw[order, gnn_mod.RAW_TS],
        end_ts=end_ts,
        window_s=window_s,
        extra_nodes=np.zeros(0, dtype=np.int64),
    )


# ── the temporal window: edges older than it are dropped, and none are in the future ─
def test_edges_older_than_the_window_are_dropped():
    """Ticket 18's first criterion, and the thing that makes the graph temporal at all."""
    rows = [row(0, "a", "b", 0.0), row(1, "c", "d", 100.0), row(2, "e", "f", 101.0)]
    end = (T0 + timedelta(hours=102)).timestamp()
    assert snapshot_of(rows, end, 3 * HOUR).n_edges == 2
    assert snapshot_of(rows, end, 200 * HOUR).n_edges == 3


def test_a_snapshot_holds_nothing_at_or_after_the_boundary_it_scores():
    rows = [row(i, f"a{i}", f"b{i}", float(i)) for i in range(10)]
    end = (T0 + timedelta(hours=5)).timestamp()
    snap = snapshot_of(rows, end, 100 * HOUR)
    raw = gnn_mod.raw_rows(rows)
    assert snap.n_edges == 5
    assert (raw[snap.edge_rows, gnn_mod.RAW_TS] < end).all()


def test_the_accounts_a_stride_is_about_to_score_join_as_isolated_nodes():
    """A beneficiary nobody has paid in a week is a real state, not a row to drop."""
    rows = [row(0, "a", "b", 0.0)]
    raw = gnn_mod.raw_rows(rows)
    order = np.argsort(raw[:, gnn_mod.RAW_TS])
    snap = gnn_mod.build_snapshot(
        raw,
        order,
        raw[order, gnn_mod.RAW_TS],
        end_ts=raw[0, gnn_mod.RAW_TS],  # nothing is strictly before the first row
        window_s=DAY,
        extra_nodes=np.array([zlib.crc32(b"a"), zlib.crc32(b"b")], dtype=np.int64),
    )
    assert snap.n_edges == 0
    assert snap.n_nodes == 2
    assert snap.x.shape == (2, gnn_mod.N_NODE_FEATURES)
    recency = gnn_mod.NODE_FEATURES.index("recency_in_window")
    counters = [i for i in range(gnn_mod.N_NODE_FEATURES) if i != recency]
    assert not snap.x[:, counters].any()  # no edges, so every counter is zero
    assert (snap.x[:, recency] == 1.0).all()  # "nothing, for the whole window", not "just now"


def test_a_self_payment_is_dropped_from_the_graph_and_counted():
    """`GATConv` strips self-loops before adding its own, which would shift the attention."""
    rows = [row(0, "a", "a", 0.0), row(1, "a", "b", 1.0)]
    snap = snapshot_of(rows, (T0 + timedelta(hours=5)).timestamp(), DAY)
    assert snap.n_edges == 1
    assert snap.dropped_self_loops == 1


def test_every_payment_is_carried_both_ways_and_the_reverse_is_flagged():
    rows = [row(0, "a", "b", 0.0), row(1, "c", "b", 1.0)]
    snap = snapshot_of(rows, (T0 + timedelta(hours=5)).timestamp(), DAY)
    index = snap.edge_index()
    assert index.shape == (2, 2 * snap.n_edges)
    assert (index[0][: snap.n_edges] == index[1][snap.n_edges :]).all()
    attr = snap.edge_attr(gnn_mod.raw_rows(rows))
    flag = gnn_mod.EDGE_FEATURES.index("is_reverse")
    assert (attr[: snap.n_edges, flag] == 0).all()
    assert (attr[snap.n_edges :, flag] == 1).all()


def test_node_features_count_the_snapshot_and_nothing_else():
    rows = [row(i, f"p{i}", "collector", float(i)) for i in range(4)]
    snap = snapshot_of(rows, (T0 + timedelta(hours=10)).timestamp(), DAY)
    node = int(snap.local([zlib.crc32(b"collector")])[0])
    in_degree = gnn_mod.NODE_FEATURES.index("log_in_degree")
    payers = gnn_mod.NODE_FEATURES.index("log_uniq_payers")
    assert snap.x[node, in_degree] == pytest.approx(np.log1p(4))
    assert snap.x[node, payers] == pytest.approx(np.log1p(4))
    assert snap.in_degree()[node] == 4


def test_account_ids_hash_the_same_way_in_any_process():
    """`hash()` is salted per interpreter; a graph that would not rebuild is not reproducible."""
    assert gnn_mod._key("acct-1") == zlib.crc32(b"acct-1")


def test_a_stride_wider_than_its_window_is_refused():
    with pytest.raises(ValueError, match="at least one full step"):
        TemporalGNNDetector(window_hours=1, stride_hours=24)


def test_a_window_with_no_time_in_it_is_refused():
    with pytest.raises(ValueError, match="positive window"):
        TemporalGNNDetector(window_hours=0, stride_hours=1)


def test_a_model_with_no_message_passing_layer_is_refused():
    with pytest.raises(ValueError, match="MLP over the degree block"):
        TemporalGNNDetector(layers=0)


# ── the label is per row, and no later row can move an earlier score ────────────
@needs_deep
def test_a_later_payment_cannot_change_an_earlier_row_s_score():
    """The causality guarantee, asserted rather than argued."""
    rows = ring_rows(n_background=600, n_rings=10)
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows)
    head = rows[:200]
    later = [row(90_000 + i, "zz", "m0", 24 * 40 + i) for i in range(40)]
    assert np.allclose(detector.predict_proba(head), detector.predict_proba(head + later)[:200])


@needs_deep
def test_two_payments_into_one_collector_are_scored_apart():
    """The bug the node-labelled version shipped: one score per *account*, broadcast to its rows.

    Two payments into the same beneficiary in the same stride share every graph feature there is.
    If the model still separates them it is reading the transaction; if it cannot, it is reading
    the node and every legitimate payment into a mule inherits the mule's score.
    """
    rows = ring_rows(n_background=800, n_rings=12)
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows)
    pair = [
        row(80_001, "a001", "collector-x", 24 * 30, amount=40.0),
        row(80_002, "a002", "collector-x", 24 * 30, amount=900_000.0),
    ]
    a, b = detector.predict_proba(pair)
    assert a != b


@needs_deep
def test_the_ring_is_learnable_at_all():
    """If the model cannot find a fan-in it built the graph from, nothing below means anything."""
    rows = ring_rows()
    cut = int(len(rows) * 0.7)
    detector = TemporalGNNDetector(seed=3, epochs=8, hidden=16, heads=2, window_hours=72).fit(
        rows[:cut]
    )
    y = np.array([int(t.is_fraud) for t in rows[cut:]])
    assert protocol.pr_auc(y, detector.predict_proba(rows[cut:])) > 5 * y.mean()


# ── the graph crosses the fit/score boundary ───────────────────────────────────
@needs_deep
def test_scoring_carries_the_graph_the_fit_saw():
    rows = ring_rows(n_background=800, n_rings=12)
    cut = int(len(rows) * 0.7)
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows[:cut])
    assert detector._tail.size

    with_history = detector.predict_proba(rows[cut:])
    detector._tail = np.zeros((0, gnn_mod.N_RAW), dtype=np.float64)
    without = detector.predict_proba(rows[cut:])
    assert not np.allclose(with_history, without), (
        "the holdout scored identically with and without the training-window graph, so the "
        "fit/score boundary is not being crossed and a holdout beneficiary arrives with no "
        "neighbourhood at all"
    )


@needs_deep
def test_the_remembered_graph_is_capped_at_the_window():
    rows = ring_rows(n_background=600, n_rings=8)
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows)
    tail = detector._tail[:, gnn_mod.RAW_TS]
    assert tail.size and (tail.max() - tail.min()) <= detector.window_s


# ── the seam ───────────────────────────────────────────────────────────────────
@needs_deep
def test_it_scores_through_the_standard_seam_and_explains_what_it_flags():
    rows = ring_rows(n_background=800, n_rings=12)
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows)
    scores = detector.score(as_batch(rows[-400:]))
    assert len(scores) == 400
    assert {s.txn_id for s in scores} == {t.txn_id for t in rows[-400:]}
    explain.assert_flagged_rows_are_explained(scores)
    flagged = [s for s in scores if s.action is not Action.ALLOW]
    assert flagged, "nothing was flagged, so the explanation floor was never exercised"
    assert any("gnn:gat" in r for r in flagged[0].reasons)
    assert not any(explain.GLOBAL_PREFIX in r for r in flagged[0].reasons)


@needs_deep
def test_an_isolated_beneficiary_says_so_rather_than_narrating_a_ring():
    detector = TemporalGNNDetector(seed=3, **FAST).fit(ring_rows(n_background=600, n_rings=8))
    codes = detector.reason_codes(row(1, "a", "b", 0.0), None)
    assert len(codes) >= explain.MIN_REASONS
    assert any("isolated node" in c for c in codes)


@needs_deep
def test_the_leave_one_attack_out_guard_can_audit_what_it_trained_on():
    from afl.evaluation.leave_one_attack_out import assert_family_held_out, training_rows

    rows = ring_rows(n_background=600, n_rings=8)
    clean = [t for t in rows if not t.is_fraud]
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows)
    assert len(training_rows(detector)) == len(rows)
    with pytest.raises(AssertionError, match="reached training"):
        assert_family_held_out(clean, "S1", detector)


@needs_deep
def test_a_retrain_accumulates_rather_than_forgetting():
    rows = ring_rows(n_background=600, n_rings=8)
    cut = int(len(rows) * 0.6)
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows[:cut])
    before = len(detector.training_rows)
    evaded = [t for t in rows[cut:] if t.is_fraud][:3]
    assert evaded, "the tail carries no fraud, so the replay buffer is never exercised"
    detector.retrain(as_batch(rows[cut:]), evaded)
    assert len(detector.training_rows) > before
    assert {t.txn_id for t in evaded} <= {t.txn_id for t in detector.training_rows}


# ── it raises rather than degrading ────────────────────────────────────────────
def test_without_the_deep_extra_the_constructor_refuses(monkeypatch):
    """Ticket 18's sixth criterion. A layer that degrades to zeros scores like one that missed."""

    def no_torch():
        raise ImportError(gnn_mod.TORCH_HINT)

    monkeypatch.setattr(gnn_mod, "require_deep", no_torch)
    with pytest.raises(ImportError, match="uv sync --extra deep"):
        TemporalGNNDetector()


@needs_deep
def test_scoring_before_fitting_raises_rather_than_returning_zeros():
    with pytest.raises(RuntimeError, match="caught nothing"):
        TemporalGNNDetector(**FAST).predict_proba([row(0, "a", "b", 0.0)])


@needs_deep
def test_a_single_class_training_set_is_refused():
    with pytest.raises(ValueError, match="single-class"):
        TemporalGNNDetector(**FAST).fit([row(i, f"a{i}", "m", float(i)) for i in range(20)])


# ── the fallback is named, and it is the shipped detector ──────────────────────
def test_the_stated_fallback_is_the_hand_rolled_detector():
    """Ticket 18's fifth criterion, at the level of code rather than of prose."""
    from afl.defend.models.lgbm import LGBMDetector

    assert isinstance(TemporalGNNDetector.fallback(), LGBMDetector)


def test_the_graph_feature_baseline_is_a_real_subset_of_the_hand_rolled_table():
    names, whole = graph_feature_names(), feature_names()
    assert set(names) < set(whole), "the narrower baseline is not narrower"
    assert "dst_in_degree" in names and "pair_is_first_payment" in names
    assert "src_out_amount_mean" not in names  # velocity and RFM are the non-graph half


def test_the_graph_feature_builder_emits_exactly_those_columns():
    rows = [row(i, f"a{i % 5}", f"m{i % 3}", float(i)) for i in range(30)]
    assert list(GraphFeatureBuilder().transform(rows).columns) == graph_feature_names()
    assert list(FeatureBuilder().transform(rows).columns) == feature_names()


# ── the compute cost travels with the lift ─────────────────────────────────────
@needs_deep
def test_the_model_card_prices_the_model():
    rows = ring_rows(n_background=600, n_rings=8)
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows)
    detector.predict_proba(rows[-200:])
    card = detector.model_card()
    assert card["hyperparameters"]["window_hours"] == FAST["window_hours"]
    assert card["compute"]["fit_seconds"] > 0
    assert card["compute"]["scored_rows"] == 200
    assert card["compute"]["n_parameters"] > 0
    assert card["fallback"].startswith("LGBMDetector")
    assert card["training"]["n_snapshots"] > 1


@needs_deep
def test_graph_coverage_notices_an_anchor_with_no_graph_to_pass_over():
    """PaySim's sender side, in miniature: every account appears once, so nothing connects."""
    rows = [row(i, f"a{i}", f"b{i}", float(i)) for i in range(60)]
    rows += [row(500 + i, f"x{i}", f"y{i}", float(i), fraud=True, vector="S1") for i in range(20)]
    detector = TemporalGNNDetector(seed=3, **FAST).fit(rows)
    # every account appears once, so no row's own endpoints are ever in the graph it is scored
    # against and no node is ever paid twice — message passing has nothing to pass
    assert detector.coverage.share_rows_with_isolated_endpoints == 1.0
    assert detector.coverage.max_in_degree == 1
    assert detector.coverage.mean_degree < 1.0


# ── the precondition: can a causal graph see the motif form at all? ────────────
def _instantaneous(n_rings: int = 20) -> tuple[list[Transaction], list[Transaction]]:
    """An anchor on a daily clock, and rings whose every payment carries one timestamp.

    Each ring is staged on its own slice of the anchor's accounts, so what the audit measures is
    the *clock* rather than two rings happening to share a payer.
    """
    real = [row(i, f"a{i % 400:03d}", f"m{i % 9}", 24.0 * (i % 30)) for i in range(2_000)]
    injected = []
    for r in range(n_rings):
        for s in range(6):
            injected.append(
                row(
                    5_000 + r * 10 + s,
                    f"a{(r * 7 + s) % 400:03d}",
                    f"a{(r * 7 + 6) % 400:03d}",
                    24.0 * (r % 25),  # every row of the ring on the same day boundary
                    fraud=True,
                    vector="S1",
                )
            )
    return real, injected


def test_a_motif_that_is_instantaneous_on_the_anchor_s_clock_is_reported_as_blind():
    real, injected = _instantaneous()
    audit = mule_graph.resolution_audit(real, injected, 168.0, 24.0, granularity_s=86_400)
    assert audit["checked"]
    assert audit["blind"] is True
    assert audit["injected_share_seeing_an_earlier_family_edge"] < audit["floor"]
    assert audit["anchor_time_granularity_s"] == 86_400


def test_a_motif_spread_across_strides_is_visible():
    real = [row(i, f"a{i % 40:02d}", f"m{i % 9}", 6.0 * i) for i in range(400)]
    injected = []
    for r in range(20):
        for s in range(6):
            injected.append(
                row(
                    5_000 + r * 10 + s,
                    f"p{(r * 3 + s) % 40:02d}",
                    f"c{r % 5}",
                    24.0 * (r % 20) + 30.0 * s,  # one payment per stride, so the ring forms
                    fraud=True,
                    vector="S1",
                )
            )
    audit = mule_graph.resolution_audit(real, injected, 168.0, 24.0, granularity_s=3_600)
    assert audit["blind"] is False
    assert audit["injected_share_seeing_an_earlier_family_edge"] > audit["floor"]


def test_the_resolution_audit_says_so_rather_than_guessing_when_one_side_is_empty():
    assert mule_graph.resolution_audit([], [row(0, "a", "b", 0.0)], 24.0, 1.0, 1) == {
        "checked": False,
        "reason": "one side of the comparison is empty",
    }


# ── the precondition: is the injected ring its own synthetic island? ───────────
def test_a_synthetic_island_is_caught_by_the_neighbourhood_audit():
    """The graph twin of the sequence model's history audit, and the reason it exists."""
    real = [row(i, f"a{i % 40:02d}", f"m{i % 9}", 6.0 * i) for i in range(600)]
    injected = []
    for r in range(20):
        for s in range(6):
            injected.append(
                row(
                    5_000 + r * 10 + s,
                    f"ghost-p{r}-{s}",  # accounts the anchor has never carried
                    f"ghost-c{r}",
                    24.0 * (r % 20) + 30.0 * s,
                    fraud=True,
                    vector="S1",
                )
            )
    audit = mule_graph.neighbourhood_audit(real, injected, 168.0, 24.0)
    assert audit["separable"] is True
    assert audit["injected_mean_synthetic_neighbour_share"] > 0.5
    assert audit["anchor_mean_synthetic_neighbour_share"] < 0.1


def test_a_ring_staged_on_the_anchor_s_own_busy_accounts_passes_the_audit():
    real = [row(i, f"a{i % 40:02d}", f"a{(i * 7) % 40:02d}", 1.5 * i) for i in range(2_000)]
    injected = []
    for r in range(20):
        for s in range(6):
            injected.append(
                row(
                    50_000 + r * 10 + s,
                    f"a{(r * 3 + s) % 40:02d}",
                    f"a{(r + 11) % 40:02d}",
                    24.0 * (r % 20) + 30.0 * s,
                    fraud=True,
                    vector="S1",
                )
            )
    audit = mule_graph.neighbourhood_audit(real, injected, 168.0, 24.0)
    assert audit["separable"] is False


def test_the_neighbourhood_audit_says_so_rather_than_guessing_when_one_side_is_empty():
    assert mule_graph.neighbourhood_audit([], [], 24.0, 1.0)["checked"] is False


# ── the variance across seeds ──────────────────────────────────────────────────
def _metrics(value: float) -> MetricResult:
    return MetricResult(
        pr_auc=value, recall_at_fixed_fpr=value, fixed_fpr=0.01, precision_at_k=value
    )


def _seed(
    seed: int,
    gnn: float,
    lgbm: float,
    floor: float = 0.01,
    outcome: str = MEASURED,
    reason: str = "",
) -> mule_graph.SeedRun:
    def system(name: str, value: float) -> mule_graph.SystemResult:
        quotable = outcome == MEASURED
        return mule_graph.SystemResult(
            name=name,
            outcome=outcome,
            reason="" if quotable else reason,
            metrics=_metrics(value) if quotable else None,
            withheld_metrics=None if quotable else _metrics(value),
        )

    return mule_graph.SeedRun(
        seed=seed,
        outcome=outcome,
        reason=reason,
        systems={
            mule_graph.GNN: system(mule_graph.GNN, gnn),
            mule_graph.LGBM: system(mule_graph.LGBM, lgbm),
            mule_graph.FLOOR: system(mule_graph.FLOOR, floor),
        },
    )


def test_the_lift_is_paired_by_seed_and_carries_its_spread():
    """Ticket 18's fourth criterion: a single-seed GNN result is not a result."""
    runs = [_seed(1, 0.60, 0.50), _seed(2, 0.70, 0.62), _seed(3, 0.66, 0.55)]
    comparison = mule_graph.compare_across_seeds(runs, "S1")
    assert comparison.n == 3
    assert [d["seed"] for d in comparison.per_seed] == [1, 2, 3]
    assert comparison.mean_delta == pytest.approx((0.10 + 0.08 + 0.11) / 3, abs=1e-6)
    assert comparison.wins == 3
    assert comparison.sd_delta > 0
    assert not comparison.inside_noise


def test_a_spread_reports_the_mean_and_the_sd_of_the_seeds_that_ran():
    spread = mule_graph.spread_across_seeds(
        [_seed(1, 0.6, 0.5), _seed(2, 0.4, 0.5)], mule_graph.GNN, "S1"
    )
    assert spread.n == 2
    assert spread.mean == pytest.approx(0.5)
    assert spread.reported is True
    assert spread.text() == "0.500 ± 0.141"


def test_one_withheld_seed_withholds_the_whole_spread():
    runs = [_seed(1, 0.6, 0.5), _seed(2, 0.6, 0.5, outcome=WITHHELD, reason="the fold is unsound")]
    spread = mule_graph.spread_across_seeds(runs, mule_graph.GNN, "S1")
    assert spread.outcome == WITHHELD
    assert spread.reported is False
    assert spread.text().startswith("[")


# ── the gate ───────────────────────────────────────────────────────────────────
def _compare(
    deltas: list[tuple[float, float]], outcome: str = MEASURED, reason: str = ""
) -> Comparison:
    return Comparison(
        challenger=mule_graph.GNN,
        incumbent=mule_graph.LGBM,
        column="S1",
        metric="pr_auc",
        per_seed=[
            {"seed": 1_000 + i, "incumbent": b, "challenger": a, "delta": round(a - b, 6)}
            for i, (a, b) in enumerate(deltas)
        ],
        outcome=outcome,
        reason=reason,
    )


def _floor(value: float = 0.02) -> Spread:
    return Spread(
        system=mule_graph.FLOOR,
        column="S1",
        metric="pr_auc",
        values=[value, value, value],
        outcome=MEASURED,
    )


def test_a_clear_win_across_seeds_promotes_the_model():
    promotion = mule_graph.decide_promotion(
        _compare([(0.80, 0.60), (0.82, 0.61), (0.79, 0.60)]), floor=_floor()
    )
    assert promotion.promoted is True
    assert promotion.shipped == "the temporal GNN"
    assert "sign-test" in promotion.reason


def test_a_loss_ships_the_stated_fallback():
    promotion = mule_graph.decide_promotion(
        _compare([(0.30, 0.60), (0.31, 0.62), (0.29, 0.59)]), floor=_floor()
    )
    assert promotion.promoted is False
    assert promotion.shipped == mule_graph.FALLBACK
    assert "loses to" in promotion.reason


def test_a_margin_inside_its_own_seed_to_seed_spread_is_not_a_lift():
    promotion = mule_graph.decide_promotion(
        _compare([(0.90, 0.60), (0.50, 0.62), (0.61, 0.60)]), floor=_floor()
    )
    assert promotion.promoted is False
    assert "inside its own seed-to-seed spread" in promotion.reason


def test_one_seed_cannot_promote_anything():
    promotion = mule_graph.decide_promotion(_compare([(0.90, 0.20)]), floor=_floor())
    assert promotion.promoted is False
    assert "single-seed GNN result" in promotion.reason


def test_a_model_the_amount_floor_beats_is_not_promoted():
    promotion = mule_graph.decide_promotion(
        _compare([(0.20, 0.05), (0.21, 0.06), (0.19, 0.04)]), floor=_floor(0.5)
    )
    assert promotion.promoted is False
    assert "sorting by amount alone" in promotion.reason


def test_a_fold_that_cannot_carry_a_claim_cannot_promote_anything():
    winning = _compare([(0.90, 0.20), (0.91, 0.21), (0.89, 0.19)])
    promotion = mule_graph.decide_promotion(
        winning, floor=_floor(), blocked="the ring is an island"
    )
    assert promotion.promoted is False
    assert promotion.reason == "the ring is an island"

    withheld = _compare(
        [(0.90, 0.20), (0.91, 0.21), (0.89, 0.19)], outcome=WITHHELD, reason="the probe says no"
    )
    assert mule_graph.decide_promotion(withheld, floor=_floor()).promoted is False


def test_a_comparison_that_never_ran_ships_the_fallback_by_decision():
    promotion = mule_graph.decide_promotion(None)
    assert promotion.promoted is False
    assert mule_graph.FALLBACK in promotion.reason


def test_every_refusal_names_what_ships_instead():
    """Ticket 18's fifth criterion is a claim about deployment; every branch has to make it."""
    for comparison, floor, blocked in (
        (None, None, ""),
        (_compare([(0.9, 0.2)]), _floor(), ""),
        (_compare([(0.3, 0.6), (0.31, 0.62), (0.29, 0.59)]), _floor(), ""),
        (_compare([(0.9, 0.6), (0.5, 0.62), (0.61, 0.6)]), _floor(), ""),
        (_compare([(0.2, 0.05), (0.21, 0.06), (0.19, 0.04)]), _floor(0.5), ""),
        (_compare([(0.9, 0.2), (0.91, 0.21), (0.89, 0.19)]), _floor(), "blocked"),
    ):
        promotion = mule_graph.decide_promotion(comparison, floor=floor, blocked=blocked)
        assert promotion.promoted is False
        assert promotion.shipped == mule_graph.FALLBACK


# ── the artefact, and the config and README it is answerable to ────────────────
def _report(dataset: str, promoted: bool) -> mule_graph.GNNReport:
    runs = [_seed(1, 0.6, 0.5), _seed(2, 0.62, 0.5), _seed(3, 0.61, 0.5)]
    comparison = mule_graph.compare_across_seeds(runs, "S1")
    fold = mule_graph.MuleFold(
        held_out_vector="S1",
        outcome=MEASURED if promoted else WITHHELD,
        reason="" if promoted else "the fold cannot carry a claim",
        promotion=mule_graph.Promotion(
            promoted=promoted,
            reason="measured, and it " + ("won" if promoted else "lost"),
            shipped="the temporal GNN" if promoted else mule_graph.FALLBACK,
        ),
        seeds=runs,
        spreads={
            s: mule_graph.spread_across_seeds(runs, s, "S1")
            for s in (mule_graph.GNN, mule_graph.LGBM, mule_graph.FLOOR)
        },
        comparison=comparison,
    )
    return mule_graph.GNNReport(
        dataset=dataset,
        seeds=[1, 2, 3],
        config={"window_hours": 168},
        operating_point={"fixed_fpr": 0.01, "k": 100},
        folds=[fold],
    )


def test_the_report_round_trips_through_disk(tmp_path):
    path = _report("amlsim", promoted=False).save(tmp_path)
    back = mule_graph.GNNReport.load("amlsim", tmp_path)
    assert back.promoted is False
    assert back.shipped == mule_graph.FALLBACK
    assert back.folds[0].comparison.n == 3
    assert back.folds[0].comparison.mean_delta == pytest.approx(0.11, abs=1e-6)
    assert back.folds[0].spreads[mule_graph.GNN].mean == pytest.approx(0.61)
    assert json.loads(path.read_text())["version"] == mule_graph.GNN_ARTEFACT_VERSION


def test_an_artefact_from_another_shape_fails_loudly(tmp_path):
    raw = json.loads(_report("amlsim", promoted=False).save(tmp_path).read_text())
    (tmp_path / "amlsim.json").write_text(json.dumps({**raw, "version": 99}))
    with pytest.raises(ValueError, match="rebuild it"):
        mule_graph.GNNReport.load("amlsim", tmp_path)


def test_a_report_with_no_folds_is_refused():
    with pytest.raises(ValueError, match="not a result"):
        mule_graph.GNNReport(
            dataset="amlsim", seeds=[1], config={}, operating_point={}, folds=[]
        )


def test_a_result_that_is_not_measured_cannot_carry_quotable_numbers():
    with pytest.raises(ValueError, match="withheld_metrics"):
        mule_graph.SystemResult(
            name="gnn", outcome=WITHHELD, reason="unsound", metrics=_metrics(0.9)
        )
    with pytest.raises(ValueError, match="has to say why"):
        mule_graph.SystemResult(name="gnn", outcome=SKIPPED)


def test_enabling_the_layer_against_the_committed_evidence_is_refused():
    refused = {"amlsim": _report("amlsim", promoted=False)}
    mule_graph.assert_config_matches_promotion(enabled=False, reports=refused)
    with pytest.raises(AssertionError, match="does not support it"):
        mule_graph.assert_config_matches_promotion(enabled=True, reports=refused)
    mule_graph.assert_config_matches_promotion(
        enabled=True, reports={"amlsim": _report("amlsim", promoted=True)}
    )


def test_enabling_the_layer_with_nothing_measured_is_refused():
    with pytest.raises(AssertionError, match="no committed GNN report"):
        mule_graph.assert_config_matches_promotion(enabled=True, reports={})


def test_the_shipped_config_agrees_with_the_committed_evidence():
    """The ticket's fifth criterion, checked against what is actually on disk right now."""
    cfg = yaml.safe_load(Path("config/defend/gnn.yaml").read_text())
    mule_graph.assert_config_matches_promotion(enabled=bool(cfg["enabled"]))


def test_the_readme_says_which_one_shipped():
    """The other half of that criterion: the claim lives in the README, so the README is tested."""
    reports = mule_graph.load_all()
    if not reports:
        pytest.skip("no committed GNN report yet — `make gnn` writes one")
    readme = Path("README.md").read_text().lower()
    assert "artifacts/gnn/" in readme and "docs/gnn.md" in readme
    promoted = any(r.promoted for r in reports.values())
    claim = (
        "the temporal gnn is what ships"
        if promoted
        else "the hand-rolled graph features are what ship"
    )
    assert claim in readme, (
        f"the committed evidence says {'the GNN' if promoted else 'the fallback'} ships and the "
        f"README does not say so — ticket 18's fifth criterion is that sentence"
    )
