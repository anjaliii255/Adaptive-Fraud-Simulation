"""Attack engines: graph (topology), velocity (pacing), drift (behaviour change).

Each owns one axis of attack shape and nothing else. Amounts and rails come from the actor
bundle; which engine runs comes from the vector registry.
"""

from __future__ import annotations


def choose_other(rng, pool: list[str], exclude: str) -> str:
    """Pick a counterparty from `pool` that is not `exclude`.

    No rail settles a payment from an account to itself, and `realism.check` treats one as a
    hard violation — so a single accidental self-transfer pins the batch's penalty at 1.0 and
    silently flattens the optimiser's fitness signal for the whole round.
    """
    options = [p for p in pool if p != exclude]
    if not options:
        raise ValueError(f"no counterparty available for {exclude!r} in a pool of {len(pool)}")
    return str(rng.choice(options))
