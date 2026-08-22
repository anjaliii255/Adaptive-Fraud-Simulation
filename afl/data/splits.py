"""Out-of-time splits. NEVER random.

A random split lets the model see tomorrow's fraud ring while scoring today's, which inflates
every number in the table. Fraud is non-stationary; the only honest split is chronological, with
an embargo gap so a slow-moving ring does not straddle the boundary.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from afl.contract.schema import Transaction


def out_of_time_split(
    txns: list[Transaction],
    train_frac: float = 0.7,
    embargo_days: float = 1.0,
    val_frac: float = 0.0,
) -> (
    tuple[list[Transaction], list[Transaction]]
    | tuple[list[Transaction], list[Transaction], list[Transaction]]
):
    """Chronological split with an embargo gap; rows inside the gap are dropped, not assigned.

    Returns (train, test), or (train, val, test) when `val_frac > 0`.
    """
    if not txns:
        return ([], []) if val_frac <= 0 else ([], [], [])
    rows = sorted(txns, key=lambda t: t.ts)
    t0, t1 = rows[0].ts, rows[-1].ts
    span = (t1 - t0) or timedelta(seconds=1)
    embargo = timedelta(days=embargo_days)

    cut = t0 + span * train_frac
    if val_frac <= 0:
        train = [t for t in rows if t.ts <= cut]
        test = [t for t in rows if t.ts > cut + embargo]
        return train, test

    val_cut = t0 + span * (train_frac + val_frac)
    train = [t for t in rows if t.ts <= cut]
    val = [t for t in rows if cut + embargo < t.ts <= val_cut]
    test = [t for t in rows if t.ts > val_cut + embargo]
    return train, val, test


def split_at(
    txns: list[Transaction], cutoff: datetime, embargo_days: float = 1.0
) -> tuple[list[Transaction], list[Transaction]]:
    """Split at an explicit timestamp — use when the cutoff is a business fact, not a fraction."""
    embargo = timedelta(days=embargo_days)
    return (
        [t for t in txns if t.ts <= cutoff],
        [t for t in txns if t.ts > cutoff + embargo],
    )


def holdout_vector(
    txns: list[Transaction], vector_id: str
) -> tuple[list[Transaction], list[Transaction]]:
    """Leave-one-attack-out: pull one synthetic family out entirely.

    Legit rows stay on both sides — the held-out set still needs a haystack for FPR to mean
    anything.
    """
    seen = [t for t in txns if t.vector_id != vector_id]
    held = [t for t in txns if t.vector_id == vector_id or not t.is_fraud]
    return seen, held


def assert_no_leakage(train: list[Transaction], test: list[Transaction]) -> None:
    """Cheap guard worth running before every reported number."""
    if not train or not test:
        return
    latest_train = max(t.ts for t in train)
    earliest_test = min(t.ts for t in test)
    if earliest_test <= latest_train:
        raise AssertionError(
            f"temporal leakage: test starts {earliest_test} but train runs to {latest_train}"
        )
    overlap = {t.txn_id for t in train} & {t.txn_id for t in test}
    if overlap:
        raise AssertionError(
            f"{len(overlap)} txn_id(s) appear in both splits, e.g. {list(overlap)[:3]}"
        )
