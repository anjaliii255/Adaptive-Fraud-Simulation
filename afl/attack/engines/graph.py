"""Graph engine — fan-in / fan-out, layering, cycles, pass-through.

Everything here is *topology over time*: who pays whom, in what shape, and how fast the money
leaves. Amounts and pacing come from the actor bundle; the shape comes from `params`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from afl.attack.actors import ActorParams
from afl.attack.engines import choose_other
from afl.contract.schema import Rail, Transaction

MOTIFS = ("fan_in", "fan_out", "layering", "cycle", "pass_through")


def _txn(
    idx: int,
    run_id: str,
    ts: datetime,
    src: str,
    dst: str,
    amount: float,
    rail: Rail,
    device: str,
    vector_id: str,
) -> Transaction:
    return Transaction(
        txn_id=f"{run_id}-g{idx:05d}",
        ts=ts,
        src=src,
        dst=dst,
        amount=round(max(0.01, amount), 2),
        rail=rail,
        device_id=device,
        is_fraud=True,
        vector_id=vector_id,
        attack_run_id=run_id,
    )


def generate(
    *,
    rng,
    run_id: str,
    vector_id: str,
    actor: ActorParams,
    start_ts: datetime,
    params: dict,
    victim_pool: list[str],
    mule_pool: list[str],
    cashout_pool: list[str],
) -> list[Transaction]:
    """Emit one graph-shaped attack episode.

    params:
      motif           one of MOTIFS
      n_sources       victims paying in (fan-in width)
      n_hops          layering depth
      split_ratio     how evenly a hop splits its balance (1.0 = even)
      hold_time_s     mean dwell time before money moves on — the knob that trades
                      detectability for realism (instant pass-through is loud)
      leak            fraction skimmed at each hop, so amounts are not a perfect chain
      fresh_beneficiary  exit into the mule pool instead of established cash-out points, so the
                      final payee has no prior inbound at all — the instant-relay signature
    """
    motif = params.get("motif", "fan_in")
    n_sources = int(params.get("n_sources", 6))
    n_hops = int(params.get("n_hops", 2))
    split_ratio = float(params.get("split_ratio", 1.0))
    hold_time_s = float(params.get("hold_time_s", 900.0))
    leak = float(params.get("leak", 0.05))
    fresh_beneficiary = bool(params.get("fresh_beneficiary", False))
    exit_pool = mule_pool if fresh_beneficiary else cashout_pool
    rail = actor.rails[0] if actor.rails else Rail.A2A

    txns: list[Transaction] = []
    ts = start_ts
    device = f"dev-{run_id[-4:]}"
    idx = 0

    def step(mean_s: float) -> datetime:
        return ts + timedelta(seconds=float(max(1.0, rng.exponential(mean_s))))

    if motif in ("fan_in", "layering", "cycle", "pass_through"):
        collector = str(rng.choice(mule_pool))
        payers = [v for v in victim_pool if v != collector]
        sources = list(rng.choice(payers, size=min(n_sources, len(payers)), replace=False))
        pot = 0.0
        for s in sources:
            amt = float(rng.lognormal(actor.amount_mu, actor.amount_sigma))
            ts = step(actor.interarrival_mean_s)
            txns.append(_txn(idx, run_id, ts, str(s), collector, amt, rail, device, vector_id))
            idx += 1
            pot += amt

        if motif == "pass_through":
            n_hops = 1
        if motif in ("layering", "cycle", "pass_through"):
            hop_src = collector
            for hop in range(n_hops):
                ts = step(hold_time_s)
                nxt = choose_other(rng, mule_pool if hop < n_hops - 1 else exit_pool, hop_src)
                pot *= 1.0 - leak
                if split_ratio < 1.0:  # break the pot into uneven legs
                    legs = max(2, int(round(1.0 / max(split_ratio, 1e-3))))
                    weights = rng.dirichlet([split_ratio * 5] * legs)
                    for w in weights:
                        txns.append(
                            _txn(
                                idx,
                                run_id,
                                ts,
                                hop_src,
                                nxt,
                                pot * float(w),
                                rail,
                                device,
                                vector_id,
                            )
                        )
                        idx += 1
                        ts = step(hold_time_s / legs)
                else:
                    txns.append(_txn(idx, run_id, ts, hop_src, nxt, pot, rail, device, vector_id))
                    idx += 1
                hop_src = nxt
            if motif == "cycle" and hop_src != collector:  # money returns to where it started
                ts = step(hold_time_s)
                txns.append(
                    _txn(
                        idx,
                        run_id,
                        ts,
                        hop_src,
                        collector,
                        pot * (1 - leak),
                        rail,
                        device,
                        vector_id,
                    )
                )
                idx += 1

    elif motif == "fan_out":
        source = str(rng.choice(mule_pool))
        pot = float(rng.lognormal(actor.amount_mu + 1.5, actor.amount_sigma))
        pool = [d for d in exit_pool if d != source]
        dests = list(rng.choice(pool, size=min(n_sources, len(pool)), replace=False))
        weights = rng.dirichlet([split_ratio * 5] * len(dests))
        for d, w in zip(dests, weights, strict=False):
            ts = step(hold_time_s / max(len(dests), 1))
            txns.append(
                _txn(idx, run_id, ts, source, str(d), pot * float(w), rail, device, vector_id)
            )
            idx += 1
    else:
        raise ValueError(f"unknown motif {motif!r}; expected one of {MOTIFS}")

    return txns
