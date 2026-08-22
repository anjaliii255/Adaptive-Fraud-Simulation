"""Feature construction — velocity windows, RFM, graph features, sequence tensors.

Two rules, both non-negotiable:

1. **Causal only.** Every feature for a row is computed from rows strictly before it. A feature
   that peeks forward will make the offline table beautiful and the deployment worthless.
2. **No provenance.** `is_fraud`, `vector_id`, and `attack_run_id` never enter X. They are the
   answer key; a model that reads them scores 1.0 and has learnt nothing.
"""

from __future__ import annotations

import copy
from bisect import bisect_right, insort
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from afl.contract.schema import Transaction

#: Columns that must never reach a model.
FORBIDDEN = ("is_fraud", "vector_id", "attack_run_id", "txn_id")

VELOCITY_WINDOWS_S = (3_600, 86_400, 604_800)  # 1h, 1d, 7d


@dataclass
class FeatureBuilder:
    """Stateless across calls by default; `stateful=True` carries entity history between batches.

    In the loop we keep state, because a mule's history does not reset because a new round began.
    """

    windows_s: tuple[int, ...] = VELOCITY_WINDOWS_S
    stateful: bool = False
    _hist: dict[str, deque] = field(default_factory=lambda: defaultdict(deque), repr=False)
    # counterparties and devices map to the timestamp they were *first* seen, and in-degree keeps
    # its timestamps, so every cumulative count can still be asked "as of when" — a plain counter
    # would answer with everything it has ever seen, including rows from after the row scored
    _counterparties: dict[str, dict] = field(default_factory=lambda: defaultdict(dict), repr=False)
    _devices: dict[str, dict] = field(default_factory=lambda: defaultdict(dict), repr=False)
    _in_degree: dict[str, list] = field(default_factory=lambda: defaultdict(list), repr=False)

    feature_names: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self._hist.clear()
        self._counterparties.clear()
        self._devices.clear()
        self._in_degree.clear()

    # ── the public surface ──────────────────────────────────────────────────────
    def transform(self, txns: list[Transaction], update: bool = True) -> pd.DataFrame:
        """One row of features per transaction, in the order given (index = txn_id).

        Rows are *computed* in timestamp order — features are causal — but written back to their
        input position. Selecting by label instead would silently multiply rows whenever a
        txn_id appears twice, which is exactly what a replay buffer does.

        `update=False` computes against the running history without committing to it. Scoring
        must use it: a detector whose feature state changes every time it scores answers the
        same question differently depending on how often it has been asked, and re-scoring the
        same holdout each round would quietly inflate that entity's velocity counts.
        """
        if not self.stateful:
            self.reset()
        snapshot = None if update else self._snapshot()

        order = sorted(range(len(txns)), key=lambda i: txns[i].ts)
        out: list[dict[str, float] | None] = [None] * len(txns)
        for i in order:
            out[i] = self._features_for(txns[i])

        if snapshot is not None:
            self._restore(snapshot)
        df = pd.DataFrame(out, index=[t.txn_id for t in txns]).astype("float64")
        df.index.name = "txn_id"
        self.feature_names = list(df.columns)
        return df

    def _snapshot(self):
        return copy.deepcopy((self._hist, self._counterparties, self._devices, self._in_degree))

    def _restore(self, snapshot) -> None:
        self._hist, self._counterparties, self._devices, self._in_degree = snapshot

    @staticmethod
    def labels(txns: list[Transaction]) -> np.ndarray:
        return np.array([int(t.is_fraud) for t in txns], dtype=int)

    # ── per-row, causal ─────────────────────────────────────────────────────────
    def _features_for(self, t: Transaction) -> dict[str, float]:
        ts = t.ts.timestamp()
        src_hist = self._hist[t.src]  # (ts, amount, dst) of *earlier* rows only
        dst_hist = self._hist[t.dst]

        f: dict[str, float] = {
            "amount": t.amount,
            "log_amount": float(np.log1p(t.amount)),
            "hour_of_day": t.ts.hour + t.ts.minute / 60.0,
            "is_night": float(t.ts.hour < 6),
            "day_of_week": float(t.ts.weekday()),
            "rail_card": float(t.rail.value == "card"),
            "rail_upi": float(t.rail.value == "upi"),
            "rail_a2a": float(t.rail.value == "a2a"),
        }

        # ── velocity: counts and sums in trailing windows (source side)
        # the lower bound is not decoration: `ts - h[0] <= w` alone also matches rows from the
        # future, so scoring against a history that contains later transactions would peek ahead
        for w in self.windows_s:
            recent = [h for h in src_hist if 0.0 <= ts - h[0] <= w]
            f[f"src_cnt_{w}s"] = float(len(recent))
            f[f"src_sum_{w}s"] = float(sum(h[1] for h in recent))
            f[f"src_uniq_dst_{w}s"] = float(len({h[2] for h in recent}))
        for w in self.windows_s[:2]:
            recent = [h for h in dst_hist if 0.0 <= ts - h[0] <= w]
            f[f"dst_cnt_{w}s"] = float(len(recent))  # fan-in pressure on the beneficiary
            f[f"dst_sum_{w}s"] = float(sum(h[1] for h in recent))

        # ── recency / frequency / monetary
        past = [h for h in src_hist if h[0] <= ts]
        f["src_seconds_since_last"] = float(ts - max(h[0] for h in past)) if past else -1.0
        f["src_txn_count"] = float(len(past))
        amounts = [h[1] for h in past]
        f["src_amount_mean"] = float(np.mean(amounts)) if amounts else 0.0
        f["src_amount_std"] = float(np.std(amounts)) if len(amounts) > 1 else 0.0
        # amount z-score against the account's own history — the ATO signal
        f["src_amount_z"] = (
            (t.amount - f["src_amount_mean"]) / f["src_amount_std"]
            if f["src_amount_std"] > 0
            else 0.0
        )
        f["src_amount_ratio_to_mean"] = (
            t.amount / f["src_amount_mean"] if f["src_amount_mean"] else 1.0
        )

        # ── graph, counted as of `ts`
        seen_dsts = self._counterparties[t.src]
        f["src_uniq_counterparties"] = float(sum(1 for first in seen_dsts.values() if first <= ts))
        f["dst_in_degree"] = float(bisect_right(self._in_degree[t.dst], ts))
        f["dst_is_new_counterparty"] = float(seen_dsts.get(t.dst, float("inf")) > ts)

        # ── device
        seen_devices = self._devices[t.src]
        f["src_uniq_devices"] = float(sum(1 for first in seen_devices.values() if first <= ts))
        f["device_is_new"] = float(
            t.device_id is not None and seen_devices.get(t.device_id, float("inf")) > ts
        )

        # ── structuring tells
        f["amount_is_round"] = float(abs(t.amount - round(t.amount, -2)) < 1e-9)
        f["amount_under_10k"] = float(9_000 <= t.amount < 10_000)
        f["amount_under_1k"] = float(900 <= t.amount < 1_000)

        self._observe(t, ts)
        return f

    def _observe(self, t: Transaction, ts: float) -> None:
        """Commit the row to history — strictly after its own features were computed."""
        self._hist[t.src].append((ts, t.amount, t.dst))
        self._hist[t.dst].append((ts, t.amount, t.src))
        firsts = self._counterparties[t.src]
        firsts[t.dst] = min(firsts.get(t.dst, ts), ts)
        insort(self._in_degree[t.dst], ts)
        if t.device_id:
            devices = self._devices[t.src]
            devices[t.device_id] = min(devices.get(t.device_id, ts), ts)
        max_w = max(self.windows_s)
        for key in (t.src, t.dst):  # bound memory: nothing older than the widest window matters
            h = self._hist[key]
            while h and ts - h[0][0] > max_w:
                h.popleft()


def sequence_tensor(
    txns: list[Transaction], max_len: int = 32
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-entity padded sequences for the GRU/transformer layer.

    Returns (X[n_entities, max_len, n_feats], y[n_entities], entity_ids). Right-aligned so the
    most recent activity is always at the end of the window.
    """
    by_src: dict[str, list[Transaction]] = defaultdict(list)
    for t in sorted(txns, key=lambda t: t.ts):
        by_src[t.src].append(t)

    ids, seqs, ys = [], [], []
    for eid, rows in by_src.items():
        rows = rows[-max_len:]
        feats = []
        prev_ts = None
        for t in rows:
            gap = (t.ts.timestamp() - prev_ts) if prev_ts else 0.0
            prev_ts = t.ts.timestamp()
            feats.append(
                [np.log1p(t.amount), np.log1p(gap), float(t.ts.hour), float(t.rail == "card")]
            )
        pad = [[0.0] * 4] * (max_len - len(feats))
        seqs.append(pad + feats)
        ys.append(int(any(t.is_fraud for t in rows)))
        ids.append(eid)
    return np.array(seqs, dtype="float32"), np.array(ys, dtype=int), ids
