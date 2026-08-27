"""▲ A — the leash, and the ticket-14 finding that it was not holding anything.

The optimiser's fitness is `evasion − λ·realism_penalty`. That expression is only a leash if the
penalty *moves*: subtracting an equal constant from every candidate leaves the argmax alone, so a
constant penalty is a λ that does nothing, however large λ is.

The committed A/B/C/D run scored 0.065 ± 0.001 in 41 of 42 rounds. These tests pin why, so that the
diagnosis is a property of the code rather than a paragraph in a document, and so that a future fix
is visible as these tests changing.

The `check()` defaults are deliberately not changed here. `docs/realism-leash.md` says why.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from afl.attack import realism
from afl.contract.schema import AttackBatch, AttackParams, Rail, Transaction

TS = datetime(2024, 5, 1, 9, 0, 0)
PARAMS = AttackParams(vector_id="S1", engine="graph", params={})


def txn(i: int, *, src="a", dst=None, amount=1234.56, fraud=True, ts=None) -> Transaction:
    return Transaction(
        txn_id=f"t{i}",
        ts=ts or (TS + timedelta(minutes=i)),
        src=src,
        dst=dst if dst is not None else f"m{i}",
        amount=amount,
        rail=Rail.A2A,
        is_fraud=fraud,
        vector_id="S1" if fraud else None,
    )


def batch(txns: list[Transaction]) -> AttackBatch:
    return AttackBatch(run_id="r", params=PARAMS, transactions=txns, seed=7)


# ── the finding: the shipped bounds cannot bind ──────────────────────────────────


def test_the_shipped_penalty_is_a_constant_for_any_well_formed_batch():
    """41 of 42 committed rounds scored 0.065. This is why: only one term is ever non-zero."""
    a = realism.check(batch([txn(i, amount=100.5 + i) for i in range(40)]))
    b = realism.check(batch([txn(i, amount=9000.25 + 7 * i, dst=f"z{i}") for i in range(40)]))

    assert a.terms["degree"] == 0.0 and a.terms["round"] == 0.0
    assert b.terms["degree"] == 0.0 and b.terms["round"] == 0.0
    # two wholly different batches, one number
    assert a.penalty == pytest.approx(b.penalty, abs=1e-3)
    assert a.penalty == pytest.approx(0.065, abs=0.005)
    assert not a.binding


def test_the_precision_target_is_further_from_every_real_anchor_than_from_nothing():
    """The 0.6 target is the miscalibration. Real anchors measure ~0.99, so a generator that

    matches reality is penalised for it, and the penalty it is charged is a constant.
    """
    measured_anchor_share = 0.9875  # amlworld; amlsim 0.9826, paysim 0.9893
    default = realism.DEFAULT_BOUNDS.target_precision_share
    assert abs(measured_anchor_share - default) > 0.35
    # and that gap, halved and divided across three terms, is the constant we observed
    assert abs(measured_anchor_share - default) * 0.5 / 3 == pytest.approx(0.065, abs=0.005)


@pytest.mark.parametrize(
    "field,shipped,real_anchor_max",
    [("max_degree_concentration", 0.6, 0.008368), ("max_round_share", 0.5, 0.0007)],
)
def test_the_ceiling_terms_sit_far_above_anything_real_data_does(field, shipped, real_anchor_max):
    """A ceiling 70x above the busiest real anchor is a ceiling nothing reaches."""
    assert getattr(realism.DEFAULT_BOUNDS, field) == shipped
    assert shipped > real_anchor_max * 50


# ── λ: a no-op against a constant, a leash against a measurement ─────────────────


def _argmax_under(lam: float, candidates: list[tuple[float, float]]) -> int:
    """The optimiser's own rule: argmax of evasion − λ·penalty."""
    return max(range(len(candidates)), key=lambda i: candidates[i][0] - lam * candidates[i][1])


def test_lambda_at_two_values_changes_nothing_when_the_penalty_is_constant():
    """The ticket asks for λ demonstrated at two values. Against the shipped leash, this is it."""
    constant = [(0.9, 0.065), (0.7, 0.065), (0.5, 0.065)]
    assert _argmax_under(0.5, constant) == _argmax_under(50.0, constant) == 0


def test_lambda_at_two_values_changes_the_winner_once_the_penalty_moves():
    """And this is what a leash that binds looks like: the same λ knob, now with an effect."""
    varying = [(0.9, 0.80), (0.7, 0.05), (0.5, 0.01)]
    assert _argmax_under(0.0, varying) == 0  # λ off: the most evasive candidate wins
    assert _argmax_under(0.5, varying) == 1  # λ on: the absurd one is priced out


def test_measured_bounds_make_a_realistic_batch_cheap_and_leave_an_absurd_one_dear():
    """Bounds from the anchor, which is what the ticket asked for instead of three guesses."""
    anchor = [
        txn(i, src=f"s{i}", dst=f"d{i}", amount=100.0 + i / 4, fraud=False) for i in range(60)
    ]
    bounds = realism.RealismBounds.from_anchor(anchor, "stand-in")

    realistic = realism.check(
        batch([txn(i, amount=250.0 + i / 3) for i in range(40)]), bounds=bounds
    )
    absurd = realism.check(
        batch([txn(i, amount=500.0, dst="mule") for i in range(40)]), bounds=bounds
    )

    assert realistic.penalty < absurd.penalty
    assert absurd.binding and not realistic.binding
    assert bounds.source == "stand-in"


# ── the deliberately absurd param set the ticket asks for ────────────────────────


def test_a_deliberately_absurd_batch_is_caught_by_name():
    """Every fraud edge on one beneficiary, every amount round, all at one instant."""
    anchor = [
        txn(i, src=f"s{i}", dst=f"d{i}", amount=100.0 + i / 4, fraud=False) for i in range(60)
    ]
    rows = [txn(i, dst="one_mule", amount=1000.0, ts=TS) for i in range(50)]
    report = realism.check(
        batch(rows), bounds=realism.RealismBounds.from_anchor(anchor, "stand-in")
    )
    assert report.terms["degree"] > 0
    assert report.terms["round"] > 0
    assert report.binding


@pytest.mark.parametrize(
    "rows,violation",
    [
        ([txn(0, src="a", dst="a")], "self_transfer"),
        ([txn(0), txn(0)], "duplicate_txn_id"),
        (
            [
                Transaction(
                    txn_id="x", ts=TS, src="a", dst="b", amount=5.0, rail=Rail.A2A, is_fraud=True
                )
            ],
            "unlabelled_fraud_row",
        ),
        (
            [
                txn(0),
                Transaction(
                    txn_id="legit",
                    ts=TS,
                    src="c",
                    dst="d",
                    amount=9.5,
                    rail=Rail.A2A,
                    is_fraud=False,
                    vector_id="S1",
                ),
            ],
            "provenance_leak_on_legit_row",
        ),
        ([], "empty_attack"),
    ],
)
def test_each_cross_row_rule_is_enforced_and_named(rows, violation):
    """A violation the report cannot name is a violation nobody can act on."""
    report = realism.check(batch(rows))
    assert violation in report.violations
    assert report.penalty == 1.0
    assert not report.ok


def test_a_non_positive_amount_is_refused_twice_over():
    """Pydantic rejects it at construction, so reaching the leash's own check needs a bypass.

    Kept rather than deleted: the two layers guard different things, and the schema is the one that
    can be turned off by a `model_construct` in a hurry.
    """
    with pytest.raises(Exception, match="amount"):
        txn(0, amount=-1.0)

    smuggled = Transaction.model_construct(
        txn_id="x",
        ts=TS,
        src="a",
        dst="b",
        amount=-1.0,
        rail=Rail.A2A,
        is_fraud=True,
        vector_id="S1",
    )
    assert "non_positive_amount" in realism.check(batch([smuggled])).violations


def test_a_hard_violation_outranks_every_soft_term():
    """The cliff is the only part of the shipped leash that ever fired — once, in 42 rounds."""
    clean = realism.check(batch([txn(i) for i in range(20)]))
    broken = realism.check(batch([txn(i) for i in range(20)] + [txn(0, src="q", dst="q")]))
    assert clean.penalty < 0.1 and broken.penalty == 1.0
