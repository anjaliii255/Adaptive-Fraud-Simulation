"""Velocity engine — inter-arrival, bursts, threshold-aware pacing.

The interesting behaviour is not "fast". It is *paced*: an attacker who knows a rule fires
under its window and under its amount ceiling, which is what makes velocity rules brittle.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from afl.attack.actors import ActorParams
from afl.attack.engines import choose_other
from afl.contract.schema import Rail, Transaction

#: Round-number reporting ceilings an attacker paces under.
THRESHOLDS = (10_000.0, 5_000.0, 2_000.0, 1_000.0)


def _just_under(threshold: float, rng, jitter: float = 0.06) -> float:
    """An amount below a ceiling, but not suspiciously exactly below it."""
    return threshold * (1.0 - float(rng.uniform(0.005, jitter)))


def generate(
    *,
    rng,
    run_id: str,
    vector_id: str,
    actor: ActorParams,
    start_ts: datetime,
    params: dict,
    src: str,
    dst_pool: list[str],
) -> list[Transaction]:
    """Emit one velocity-shaped attack episode.

    params:
      n_txns           how many attempts in the episode
      burst_size       attempts per burst (1 = evenly paced, no burst)
      burst_gap_s      quiet time between bursts — the pacing knob
      intra_burst_s    mean seconds between attempts inside a burst
      threshold        reporting ceiling to stay under (None = ignore)
      device_rotation  probability of a fresh device per attempt
      amount_jitter    lognormal sigma around the target amount
    """
    n_txns = int(params.get("n_txns", 12))
    burst_size = max(1, int(params.get("burst_size", 4)))
    burst_gap_s = float(params.get("burst_gap_s", 3 * 3_600.0))
    intra_burst_s = float(params.get("intra_burst_s", 45.0))
    threshold = params.get("threshold", THRESHOLDS[-1])
    device_rotation = float(params.get("device_rotation", 1.0 - actor.device_stickiness))
    amount_jitter = float(params.get("amount_jitter", 0.15))
    rail = actor.rails[0] if actor.rails else Rail.UPI

    txns: list[Transaction] = []
    ts = start_ts
    device = f"dev-{run_id[-4:]}-0"
    for i in range(n_txns):
        if i and i % burst_size == 0:
            ts += timedelta(seconds=float(max(1.0, rng.exponential(burst_gap_s))))
        elif i:
            ts += timedelta(seconds=float(max(1.0, rng.exponential(intra_burst_s))))

        if rng.random() < device_rotation:
            device = f"dev-{run_id[-4:]}-{i}"

        if threshold:
            amount = _just_under(float(threshold), rng) * float(rng.lognormal(0.0, amount_jitter))
            amount = min(amount, float(threshold) * 0.999)
        else:
            amount = float(rng.lognormal(actor.amount_mu, actor.amount_sigma))

        txns.append(
            Transaction(
                txn_id=f"{run_id}-v{i:05d}",
                ts=ts,
                src=src,
                dst=choose_other(rng, dst_pool, src),
                amount=round(max(0.01, amount), 2),
                rail=rail,
                device_id=device,
                is_fraud=True,
                vector_id=vector_id,
                attack_run_id=run_id,
            )
        )
    return txns
