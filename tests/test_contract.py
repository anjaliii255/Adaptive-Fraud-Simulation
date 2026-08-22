"""⚑ Protects the seam.

If these break, A and B have silently stopped speaking the same language, and every number
produced after that point is measuring two different things.
"""

from __future__ import annotations

from datetime import datetime

import pydantic
import pytest

from afl.contract.metrics import EVASION_ACTIONS, Action, DetectorScore, MetricResult
from afl.contract.schema import AttackBatch, AttackParams, Entity, EntityRole, Rail, Transaction

TS = datetime(2024, 1, 1, 12, 0)


def txn(**kw) -> Transaction:
    base = dict(txn_id="t1", ts=TS, src="a", dst="b", amount=100.0, rail=Rail.A2A)
    return Transaction(**{**base, **kw})


# ── field names are frozen ──────────────────────────────────────────────────────
def test_transaction_field_names_are_frozen():
    expected = {
        "txn_id",
        "ts",
        "src",
        "dst",
        "amount",
        "rail",
        "device_id",
        "is_fraud",
        "vector_id",
        "attack_run_id",
    }
    assert set(Transaction.model_fields) == expected


def test_detector_score_field_names_are_frozen():
    assert set(DetectorScore.model_fields) == {"txn_id", "score", "action", "reasons"}


def test_metric_result_carries_its_operating_point():
    # a metric without the threshold it was measured at is not comparable to anything
    for field in ("fixed_fpr", "k", "held_out_vector"):
        assert field in MetricResult.model_fields


# ── round trips ─────────────────────────────────────────────────────────────────
def test_transaction_round_trip():
    t = txn(device_id="d1", is_fraud=True, vector_id="S1", attack_run_id="r1")
    assert Transaction(**t.model_dump()) == t
    assert Transaction.model_validate_json(t.model_dump_json()) == t


def test_batch_round_trip_and_helpers():
    rows = [txn(txn_id="t1", is_fraud=True, vector_id="S1"), txn(txn_id="t2")]
    batch = AttackBatch(
        run_id="r1",
        params=AttackParams(vector_id="S1", engine="graph", params={"n_sources": 4}),
        transactions=rows,
        seed=7,
        entities=[Entity(entity_id="a", role=EntityRole.MULE)],
    )
    assert AttackBatch.model_validate_json(batch.model_dump_json()) == batch
    assert len(batch) == 2
    assert [t.txn_id for t in batch.fraud_transactions] == ["t1"]


def test_enums_serialise_as_their_string_values():
    # the API and any on-disk artefact depend on these being plain strings
    assert txn().model_dump(mode="json")["rail"] == "a2a"
    assert Action.STEP_UP.value == "step_up"
    assert Rail("upi") is Rail.UPI


# ── validators ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("amount", [0.0, -1.0])
def test_non_positive_amount_rejected(amount):
    with pytest.raises(pydantic.ValidationError):
        txn(amount=amount)


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_score_must_be_a_probability(score):
    with pytest.raises(pydantic.ValidationError):
        DetectorScore(txn_id="t1", score=score, action=Action.ALLOW)


def test_default_transaction_is_legit_with_no_provenance():
    t = txn()
    assert (t.is_fraud, t.vector_id, t.attack_run_id) == (False, None, None)


# ── the one shared piece of logic ───────────────────────────────────────────────
def test_only_allow_counts_as_evasion():
    assert EVASION_ACTIONS == {Action.ALLOW}
    assert DetectorScore(txn_id="t", score=0.1, action=Action.ALLOW).evaded
    for action in (Action.STEP_UP, Action.HOLD, Action.REVIEW, Action.DECLINE):
        assert not DetectorScore(txn_id="t", score=0.9, action=action).evaded
