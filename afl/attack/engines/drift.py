"""Drift engine — baseline behaviour, then a deviation after a takeover event.

This is the account-takeover / dormant-mule family: the account looks ordinary for a while,
so a model that only sees the fraud window learns the wrong thing. The engine emits *both*
sides of the event, and only the post-event tail is labelled fraud.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from afl.attack.actors import ActorParams
from afl.contract.schema import Rail, Transaction


def generate(
    *,
    rng,
    run_id: str,
    vector_id: str,
    actor: ActorParams,
    start_ts: datetime,
    params: dict,
    src: str,
    benign_dst_pool: list[str],
    cashout_pool: list[str],
) -> list[Transaction]:
    """Emit a baseline tail plus a post-event deviation.

    params:
      n_baseline      ordinary transactions before the event
      n_post          transactions after the takeover
      ramp            0 = hard switch on the event, 1 = fully gradual escalation
      amount_shift    multiplicative jump in typical amount after the event
      new_device      whether the post-event tail uses an unseen device
      dormancy_s      quiet gap between baseline and event (dormant mule wakes up)
      beneficiary_reuse  share of post-event payments going back to a beneficiary the account
                      already used. 1.0 = never a new counterparty, which is the first-party shape
      pace_factor     how much faster the tail transacts at full escalation (0 = no speed-up)
      label_baseline  if True the pre-event rows are labelled fraud too (usually wrong —
                      keep False so the label matches what an investigator would call it)
    """
    n_baseline = int(params.get("n_baseline", 20))
    n_post = int(params.get("n_post", 10))
    ramp = float(params.get("ramp", 0.0))
    amount_shift = float(params.get("amount_shift", 4.0))
    new_device = bool(params.get("new_device", True))
    dormancy_s = float(params.get("dormancy_s", 7 * 86_400.0))
    label_baseline = bool(params.get("label_baseline", False))
    beneficiary_reuse = float(params.get("beneficiary_reuse", 0.0))
    pace_factor = float(params.get("pace_factor", 4.0))
    rail = actor.rails[0] if actor.rails else Rail.CARD

    txns: list[Transaction] = []
    known_dsts: list[str] = []  # the account's established counterparties
    ts = start_ts
    device = f"dev-{run_id[-4:]}-base"

    for i in range(n_baseline):
        ts += timedelta(seconds=float(max(1.0, rng.exponential(actor.interarrival_mean_s))))
        dst = str(rng.choice(benign_dst_pool))
        known_dsts.append(dst)
        txns.append(
            Transaction(
                txn_id=f"{run_id}-d{i:05d}",
                ts=ts,
                src=src,
                dst=dst,
                amount=round(float(rng.lognormal(actor.amount_mu, actor.amount_sigma)), 2),
                rail=rail,
                device_id=device,
                is_fraud=label_baseline,
                vector_id=vector_id if label_baseline else None,
                attack_run_id=run_id if label_baseline else None,
            )
        )

    ts += timedelta(seconds=dormancy_s)
    if new_device:
        device = f"dev-{run_id[-4:]}-ato"

    for j in range(n_post):
        # ramp=0 → full shift immediately; ramp=1 → shift arrives linearly across the tail
        progress = 1.0 if ramp <= 0 else min(1.0, (j + 1) / max(1, n_post) / ramp)
        shift = 1.0 + (amount_shift - 1.0) * progress
        pace = actor.interarrival_mean_s / (1.0 + pace_factor * progress)
        ts += timedelta(seconds=float(max(1.0, rng.exponential(pace))))
        # cash-out to a fresh counterparty is a ring tell; reuse keeps the account looking itself
        reuse = beneficiary_reuse > 0 and known_dsts and rng.random() < beneficiary_reuse
        dst = str(rng.choice(known_dsts)) if reuse else str(rng.choice(cashout_pool))
        txns.append(
            Transaction(
                txn_id=f"{run_id}-d{n_baseline + j:05d}",
                ts=ts,
                src=src,
                dst=dst,
                amount=round(float(rng.lognormal(actor.amount_mu, actor.amount_sigma)) * shift, 2),
                rail=rail,
                device_id=device,
                is_fraud=True,
                vector_id=vector_id,
                attack_run_id=run_id,
            )
        )
    return txns
