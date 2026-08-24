"""▲ A — an attack is only a detection problem if it is commensurate with the traffic it hides in.

Three separate versions of the same bug shipped before these tests existed: amounts a thousand
times too small for the anchor, a rail the anchor never carries, and timestamps spread across a
clock whose real rows all sit at midnight. Each one let a single field separate synthetic from
real, and each one made an unseen-attack score look like skill.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from afl.attack import actors
from afl.attack.envelope import (
    TRIVIAL_SEPARATION,
    AnchorEnvelope,
    audit,
    weighted_pool,
)
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.schema import Rail, Transaction
from afl.utils.seed import rng

T0 = datetime(2023, 1, 1)


def anchor_rows(
    n: int = 600,
    amount: float = 70_000.0,
    rail: Rail = Rail.A2A,
    senders: int = 20,
    hourly: bool = True,
) -> list[Transaction]:
    """Stand-in for a real anchor: big amounts, one rail, repeat senders, no devices.

    Amounts are lognormal rather than a tidy band. Real traffic is heavy-tailed, and an anchor
    with almost no dispersion makes any shifted vector look separable for the wrong reason.
    """
    draw = rng(7).lognormal(math.log(amount), 1.0, size=n)
    return [
        Transaction(
            txn_id=f"real{i}",
            ts=T0 + timedelta(hours=i if hourly else 0, days=0 if hourly else i),
            src=f"acct{i % senders}",
            dst=f"merch{i % 7}",
            amount=round(float(draw[i]), 2),
            rail=rail,
            device_id=None,
        )
        for i in range(n)
    ]


def test_envelope_measures_the_anchor_rather_than_assuming_it():
    e = AnchorEnvelope.measure(anchor_rows(), "stand-in")
    assert e.rail_mix == {"a2a": 1.0}
    assert e.carries_devices is False
    assert e.supports_behavioural_vectors is True
    assert 40_000 < math.exp(e.amount_log_mu) < 130_000


def test_an_anchor_where_no_account_repeats_cannot_host_behavioural_vectors():
    """PaySim: 636,323 distinct senders over 636,409 rows. There is no history to drift from."""
    e = AnchorEnvelope.measure(anchor_rows(n=200, senders=200), "unique-senders")
    assert e.sender_reuse_rate < 0.5
    assert e.supports_behavioural_vectors is False


def test_daily_anchors_are_recognised_as_daily():
    assert AnchorEnvelope.measure(anchor_rows(hourly=False), "daily").time_granularity_s == 86_400
    assert AnchorEnvelope.measure(anchor_rows(hourly=True), "hourly").time_granularity_s == 3_600


def test_rescale_moves_actors_onto_the_anchor_without_flattening_them():
    e = AnchorEnvelope.measure(anchor_rows(), "stand-in")
    normal, fraudster = e.rescale(actors.NORMAL), e.rescale(actors.FRAUDSTER)
    assert normal.amount_mu > actors.NORMAL.amount_mu + 4, "did not move onto the anchor's scale"
    # the population still has structure: a fraudster out-spends a normal user by the same factor
    assert fraudster.amount_mu - normal.amount_mu == pytest.approx(
        actors.FRAUDSTER.amount_mu - actors.NORMAL.amount_mu
    )
    assert normal.rails == (Rail.A2A,), "kept a rail the anchor never carries"


def test_the_audit_catches_a_generator_that_is_off_the_anchor():
    real = anchor_rows()
    off_scale = [
        t.model_copy(
            update={"txn_id": f"s{i}", "amount": 40.0, "is_fraud": True, "vector_id": "M3"}
        )
        for i, t in enumerate(real[:100])
    ]
    report = audit(real, off_scale)
    assert report["trivially_separable"]
    assert report["worst"] == "log_amount"
    assert report["score"] > TRIVIAL_SEPARATION


def test_the_audit_passes_traffic_that_sits_inside_the_anchor():
    """New rows on the anchor's own accounts, at its own scale, are not separable by one field."""
    real = anchor_rows()
    inside = [
        t.model_copy(
            update={
                "txn_id": f"s{i}",
                "ts": t.ts + timedelta(minutes=30),
                "is_fraud": True,
                "vector_id": "M3",
            }
        )
        for i, t in enumerate(real[::7])
    ]
    assert not audit(real, inside)["trivially_separable"]


def test_an_anchored_simulator_lands_inside_its_anchor():
    """The end-to-end guard: generate against an anchor and let the audit judge the result."""
    real = anchor_rows(n=2_000, senders=60)
    envelope = AnchorEnvelope.measure(real, "stand-in")
    sim = Simulator(seed=5, n_entities=50, n_background=0, n_episodes=6, envelope=envelope)
    m3 = [t for t in sim.generate(registry.get("M3").to_attack_params()).transactions if t.is_fraud]

    assert m3
    assert {t.rail for t in m3} == {Rail.A2A}, "generated a rail the anchor never carries"
    assert all(t.device_id is None for t in m3), "invented a device the anchor cannot have"
    anchor_accounts = {t.src for t in real}
    assert {t.src for t in m3} <= anchor_accounts, "staged the attack on invented accounts"
    assert not audit(real, m3)["trivially_separable"]


def test_an_unanchored_simulator_is_left_exactly_as_it_was():
    """Anchoring is opt-in: the synthetic default must not move because real data exists."""
    plain = Simulator(seed=11, n_entities=120, n_background=200, n_episodes=2)
    again = Simulator(seed=11, n_entities=120, n_background=200, n_episodes=2)
    params = registry.get("S1").to_attack_params()
    assert plain.generate(params).model_dump_json() == again.generate(params).model_dump_json()


# ── the payee namespace: the bug this section exists for ────────────────────────
def customer_to_merchant_anchor(n: int = 600) -> list[Transaction]:
    """An anchor where the two sides of a payment are separate namespaces, as BankSim is."""
    rows = anchor_rows(n=n, senders=40)
    return [t.model_copy(update={"dst": f"M{i % 12}"}) for i, t in enumerate(rows)]


def test_a_customer_as_payee_fails_the_payee_check():
    """The false pass that shipped: pooling senders and payees hid a payee no anchor row uses.

    Every real payment goes customer -> merchant. A synthetic row paying a *customer* is a row
    no real payment resembles, and it carries none of the merchant's category — but a combined
    membership set contains customers, so it passed.
    """
    real = customer_to_merchant_anchor()
    paying_a_customer = [
        t.model_copy(update={"txn_id": f"s{i}", "dst": t.src, "is_fraud": True, "vector_id": "M3"})
        for i, t in enumerate(real[::7])
    ]
    report = audit(real, paying_a_customer)
    assert report["signals"]["payee_in_anchor"] > TRIVIAL_SEPARATION
    assert report["trivially_separable"], "a customer payee has to fail the payee check"
    # and the sender side is genuinely fine, so the two must not be conflated
    assert report["signals"]["sender_in_anchor"] < report["signals"]["payee_in_anchor"]


def test_an_anchored_simulator_pays_the_anchors_own_merchants():
    real = customer_to_merchant_anchor(n=2_000)
    envelope = AnchorEnvelope.measure(real, "two-sided")
    assert set(envelope.active_payees) <= {t.dst for t in real}
    assert not (set(envelope.active_payees) & {t.src for t in real}), "payee pool leaked senders"

    sim = Simulator(seed=5, n_entities=60, n_background=0, n_episodes=4, envelope=envelope)
    m3 = [t for t in sim.generate(registry.get("M3").to_attack_params()).transactions if t.is_fraud]
    assert m3
    assert {t.dst for t in m3} <= {t.dst for t in real}, "paid a beneficiary the anchor never uses"


def test_the_merchant_pool_carries_the_anchors_own_proportions():
    """BankSim sends 85% of payments to one merchant; drawing evenly gets the category mix wrong."""
    weights = {"M0": 0.85, "M1": 0.10, "M2": 0.05}
    pool = weighted_pool(weights, 400)
    assert set(pool) == set(weights), "a real beneficiary dropped out of the pool entirely"
    share = pool.count("M0") / len(pool)
    assert 0.75 < share < 0.95, f"dominant beneficiary got {share:.0%} of the pool"
