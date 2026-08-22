"""Level 1 — do the marginals and the joints look right?

The cheapest, weakest evidence: matching one-dimensional distributions is necessary, never
sufficient. A generator can nail every marginal and still produce traffic with no structure at
all, which is why passing here only buys the right to be measured at level 2.

All distances are reported so that **0 = identical, 1 = unrelated**, and the level score is
`1 - mean(distance)`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from afl.contract.schema import Transaction

#: Columns compared marginally. Nothing here is provenance — comparing `is_fraud` would be circular.
MARGINAL_COLUMNS = ("log_amount", "hour_of_day", "day_of_week", "log_gap_s", "amount_cents")


def marginals(txns: list[Transaction]) -> pd.DataFrame:
    """A dataset-agnostic numeric view of traffic — amount, timing, and pacing only."""
    rows = sorted(txns, key=lambda t: t.ts)
    last_seen: dict[str, float] = {}
    out = []
    for t in rows:
        ts = t.ts.timestamp()
        gap = ts - last_seen.get(t.src, ts)
        last_seen[t.src] = ts
        out.append(
            {
                "log_amount": float(np.log1p(t.amount)),
                "hour_of_day": t.ts.hour + t.ts.minute / 60.0,
                "day_of_week": float(t.ts.weekday()),
                "log_gap_s": float(np.log1p(max(gap, 0.0))),
                "amount_cents": float(round(t.amount % 1, 2)),
            }
        )
    return pd.DataFrame(out, columns=list(MARGINAL_COLUMNS))


def ks_distances(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    """Kolmogorov-Smirnov per column. Already in [0, 1]."""
    from scipy.stats import ks_2samp

    out = {}
    for c in real.columns:
        if len(real[c]) < 2 or len(synth[c]) < 2:
            out[c] = 1.0
            continue
        out[c] = float(ks_2samp(real[c].to_numpy(), synth[c].to_numpy()).statistic)
    return out


def wasserstein_distances(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    """Earth-mover distance, scaled by the real column's spread so it is comparable across units."""
    from scipy.stats import wasserstein_distance

    out = {}
    for c in real.columns:
        a, b = real[c].to_numpy(), synth[c].to_numpy()
        if a.size == 0 or b.size == 0:
            out[c] = 1.0
            continue
        scale = float(a.std()) or 1.0
        out[c] = float(min(1.0, wasserstein_distance(a, b) / scale))
    return out


def correlation_delta(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean absolute difference between the two correlation matrices — the first joint check."""
    a = real.corr().to_numpy()
    b = synth.reindex(columns=real.columns).corr().to_numpy()
    mask = ~np.eye(a.shape[0], dtype=bool)
    diff = np.abs(np.nan_to_num(a) - np.nan_to_num(b))[mask]
    return float(min(1.0, diff.mean() / 2.0))  # corr differences span [0, 2]


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 12) -> float:
    """Binned MI in nats. Catches the non-linear dependence correlation misses."""
    if x.size < bins or y.size < bins:
        return 0.0
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    joint = joint / max(joint.sum(), 1.0)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def mutual_information_delta(real: pd.DataFrame, synth: pd.DataFrame, bins: int = 12) -> float:
    """Mean relative error in pairwise MI, catching the dependence correlation misses."""
    cols = list(real.columns)
    diffs = []
    for i, ci in enumerate(cols):
        for cj in cols[i + 1 :]:
            mi_r = _mutual_information(real[ci].to_numpy(), real[cj].to_numpy(), bins)
            mi_s = _mutual_information(synth[ci].to_numpy(), synth[cj].to_numpy(), bins)
            denom = max(mi_r, mi_s, 1e-6)
            diffs.append(abs(mi_r - mi_s) / denom)
    return float(min(1.0, np.mean(diffs))) if diffs else 0.0


def categorical_deltas(real: list[Transaction], synth: list[Transaction]) -> dict[str, float]:
    """Total-variation distance on the categorical fields the marginal frame cannot hold."""

    def share(txns, key) -> dict[str, float]:
        counts: dict[str, int] = defaultdict(int)
        for t in txns:
            counts[str(key(t))] += 1
        n = max(len(txns), 1)
        return {k: v / n for k, v in counts.items()}

    out = {}
    for name, key in (
        ("rail", lambda t: t.rail.value),
        ("has_device", lambda t: t.device_id is not None),
    ):
        a, b = share(real, key), share(synth, key)
        keys = set(a) | set(b)
        out[f"tv_{name}"] = float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / 2)
    return out


def report(real: list[Transaction], synth: list[Transaction]) -> dict[str, object]:
    """Level 1 verdict. `score` in [0, 1], higher is better."""
    rf, sf = marginals(real), marginals(synth)
    ks = ks_distances(rf, sf)
    wass = wasserstein_distances(rf, sf)
    corr = correlation_delta(rf, sf)
    mi = mutual_information_delta(rf, sf)
    cats = categorical_deltas(real, synth)

    distances = list(ks.values()) + list(wass.values()) + list(cats.values()) + [corr, mi]
    return {
        "level": 1,
        "n_real": len(real),
        "n_synth": len(synth),
        "ks": {k: round(v, 4) for k, v in ks.items()},
        "wasserstein": {k: round(v, 4) for k, v in wass.items()},
        "categorical": {k: round(v, 4) for k, v in cats.items()},
        "corr_delta": round(corr, 4),
        "mi_delta": round(mi, 4),
        "worst_column": max(ks, key=ks.get) if ks else None,
        "score": round(1.0 - float(np.mean(distances)), 4),
    }
