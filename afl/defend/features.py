"""Feature construction — velocity, RFM, graph, relationship and device features.

Three rules, all non-negotiable:

1. **Causal only.** Every feature for a row is computed from events strictly before it, and
   "before" is judged against the row's own timestamp, never against whatever the builder happens
   to have observed. A feature that peeks forward makes the offline table beautiful and the
   deployment worthless.
2. **No provenance.** `is_fraud`, `vector_id` and `attack_run_id` never enter X. They are the
   answer key; a model that reads them scores 1.0 and has learnt nothing. `transform` checks.
3. **Say what you count.** Every column name states the entity, the direction and the window, and
   every column has a one-line rationale in `feature_specs()`. `docs/features.md` is generated
   from that registry, so the documentation cannot drift from the table.

**Direction is the whole design.** The first version of this module kept one history per entity
and appended each transaction to *both* the sender's and the beneficiary's — so "payments sent in
the last hour" silently counted payments received as well, and fan-in and fan-out were the same
number. Here every entity has two streams, `out` (what it sent) and `in` (what it received), and
some of the most useful features are the ones that cross them: money arriving and leaving inside
the hour is a pass-through, and that is invisible if the two are added together.

**The beneficiary side is not an afterthought.** PaySim, one of the two real anchors, has no
sender history at all — `nameOrig` is effectively unique per row, so every `src_*` feature below
is structurally empty on it and the beneficiary is the only entity with a past. See
`docs/data-cards/paysim.md`, and `docs/features.md` for how much of this table each anchor
actually fills in — measured, not assumed.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pandas as pd

from afl.contract.schema import Transaction

log = logging.getLogger(__name__)

#: Columns that must never reach a model.
FORBIDDEN = ("is_fraud", "vector_id", "attack_run_id", "txn_id")

#: 1h / 1d / 7d. Every windowed feature is emitted once per entry.
VELOCITY_WINDOWS_S = (3_600, 86_400, 604_800)

#: The window the pass-through features look back over. Deliberately the shortest one: an
#: instant-A2A mule forwards in minutes, and a day-wide window buries that in ordinary traffic.
DWELL_WINDOW_S = 3_600

#: "This has never happened." A zero would read as "it happened just now", which is the opposite.
NEVER = -1.0

_WINDOW_LABELS = {3_600: "1h", 86_400: "24h", 604_800: "7d"}


def window_label(w: int) -> str:
    """`3600 -> '1h'` for prose; falls back to seconds for a non-standard window."""
    return _WINDOW_LABELS.get(int(w), f"{int(w)}s")


# ── the feature registry ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeatureSpec:
    """One column: what it is called, which block it belongs to, and why it exists.

    The `why` is a deliverable, not decoration. A feature nobody can justify in one line is a
    feature nobody can defend when it turns out to be the top SHAP driver on a declined payment.
    """

    name: str
    group: str
    why: str


#: The blocks, in emission order, and what each one is for.
FEATURE_GROUPS: dict[str, str] = {
    "row": "The transaction on its own — no history needed, so never empty on any anchor.",
    "src-outbound": (
        "What the paying account has sent before. Velocity and RFM: the card-testing (S2) and "
        "fan-out (S1) block. Structurally empty on PaySim, which has no sender history."
    ),
    "beneficiary-inbound": (
        "What the beneficiary has received before. Fan-in pressure and payee novelty: the mule "
        "collector (S1) and new-payee (C2, C3) block, and the only side PaySim can fill."
    ),
    "pass-through": (
        "Money in versus money out at the same account inside the hour. The dwell-time block "
        "that instant-A2A pass-through (C3) and layering (S1) trade against."
    ),
    "relationship": (
        "This payer to this payee. Whether the pair has ever transacted, and how recently."
    ),
    "device": (
        "Devices seen on the paying account. The takeover (S3) block. Neither real anchor "
        "carries a device id, so these are empty on both and only move on synthetic traffic."
    ),
}


#: Columns that need no history — name -> why, in emission order.
_ROW_FEATURES: dict[str, str] = {
    "amount": "The amount. Every cost model and every action band is a function of it.",
    "log_amount": "log1p(amount): payment amounts are lognormal, so the log splits evenly.",
    "hour_of_day": "Hour with minutes as a fraction. Constant on AMLSim, whose steps are days.",
    "is_night": "00:00-05:59. The cheapest out-of-pattern-timing proxy there is.",
    "day_of_week": "Monday=0. Payroll, rent and settlement traffic all have a weekday shape.",
    "is_weekend": "Saturday or Sunday: less legit volume, so the same burst is louder.",
    "rail_card": "Card rail. Constant on both real anchors, which are account-to-account only.",
    "rail_upi": "UPI rail — the C2 collect-request surface. Synthetic traffic only, today.",
    "rail_a2a": "A2A rail: irrevocable, so a pass-through here has no chargeback to fear.",
    "amount_is_round": "Round to the nearest 100. Hand-typed amounts and structuring cluster here.",
    "amount_under_10k": "9,000-9,999.99: parked under a reporting ceiling. The M1 boundary tell.",
    "amount_under_1k": "900-999.99: the same trick at the smaller limit.",
}

#: Windowed velocity on the payer's outbound stream. `{w}` is seconds, `{lab}` is prose.
_SRC_OUT_WINDOWED: dict[str, str] = {
    "src_out_cnt_{w}s": "Payments this account sent in the last {lab}. The classic velocity count.",
    "src_out_sum_{w}s": "Value it sent in the last {lab}. Catches a large drain a count misses.",
    "src_out_uniq_dst_{w}s": (
        "Distinct beneficiaries paid in the last {lab}. Fan-out breadth: card testing and mule "
        "spraying both widen it."
    ),
}

_SRC_OUT_SCALARS: dict[str, str] = {
    "src_out_txn_count": (
        "Payments this account has ever sent, as of now. Tenure, and the denominator the ratios "
        "need."
    ),
    "src_seconds_since_last_out": (
        "Seconds since its last payment; -1 if it has never paid. Dormancy then activity is the "
        "bust-out arc."
    ),
    "src_out_amount_mean": (
        "Its own average payment. 'Normal' has to mean normal for this account, not for the "
        "population."
    ),
    "src_out_amount_std": (
        "Spread of its own payments. A steady account and an erratic one need different z-scores."
    ),
    "src_amount_z": (
        "Amount in standard deviations of the account's own history. The drift tell behind S3, "
        "C1 and M3."
    ),
    "src_amount_ratio_to_mean": (
        "Amount over its own mean. Scale-free, and it survives a zero standard deviation."
    ),
    "src_out_uniq_beneficiaries": (
        "Distinct beneficiaries ever paid, counted as of this row. Graph out-degree with no "
        "future in it."
    ),
    "src_account_age_s": (
        "Seconds since the account was first seen at all; -1 if this is its first row. A "
        "fabricated identity (M2) has no past to show."
    ),
}

#: Windowed fan-in on the beneficiary's inbound stream — the block PaySim can actually fill.
_DST_IN_WINDOWED: dict[str, str] = {
    "dst_in_cnt_{w}s": "Payments this beneficiary received in the last {lab}. Fan-in pressure.",
    "dst_in_sum_{w}s": "Value it received in the last {lab}. What a mule account is holding.",
    "dst_in_uniq_src_{w}s": (
        "Distinct payers in the last {lab}. Fourteen accounts paying one beneficiary in an hour "
        "is the S1 tell; fourteen payments from one account is a subscription."
    ),
}

_DST_IN_SCALARS: dict[str, str] = {
    "dst_in_txn_count": "Payments this beneficiary has ever received, as of now.",
    "dst_in_degree": (
        "Distinct accounts that have ever paid it, as of now. Graph in-degree computed at the "
        "row's timestamp, never over the finished graph."
    ),
    "dst_seconds_since_last_in": "Seconds since it last received anything; -1 if never.",
    "dst_in_amount_mean": (
        "What it usually receives. A merchant taking 300 a hundred times is not a mule taking "
        "300,000 once."
    ),
    "dst_amount_z": (
        "This amount in standard deviations of what the beneficiary usually receives."
    ),
    "dst_is_first_ever_inbound": (
        "Nobody has ever paid this beneficiary before. A brand-new payee is the C2 and C3 "
        "opening move."
    ),
    "dst_account_age_s": (
        "Seconds since the beneficiary was first seen at all; -1 if this is its first row."
    ),
}

#: Money in versus money out at the same account, over `DWELL_WINDOW_S`.
_PASS_THROUGH: dict[str, str] = {
    "src_in_cnt_{w}s": (
        "Payments the *sender* received in the last {lab}. A mule forwards what has just arrived."
    ),
    "src_in_sum_{w}s": (
        "Value the sender received in the last {lab}. The denominator of the ratio below."
    ),
    "src_seconds_since_last_in": (
        "Dwell time: seconds between money arriving at this account and this payment leaving it; "
        "-1 if it has never received. The single knob C3 trades detectability against."
    ),
    "src_passthrough_ratio_{w}s": (
        "Sent over received in the last {lab}. Near 1.0 is a pass-through; a real account keeps "
        "some of it."
    ),
    "dst_out_cnt_{w}s": (
        "Payments the *beneficiary* sent in the last {lab}. Paying into an account that is "
        "already forwarding is layering, not commerce."
    ),
}

_PAIR_FEATURES: dict[str, str] = {
    "pair_is_first_payment": (
        "First-ever payment from this account to this beneficiary. The APP-scam (C2) tell, and "
        "informative even where there is no other history at all."
    ),
    "pair_txn_count": "How many times this account has paid this beneficiary before.",
    "pair_seconds_since_last": "Seconds since the last payment on this pair; -1 if first.",
}

_DEVICE_FEATURES: dict[str, str] = {
    "src_uniq_devices": (
        "Distinct devices ever seen on this account, as of now. Device churn is the takeover "
        "(S3) signature."
    ),
    "device_is_new": "This device has not been seen on this account before.",
    "device_seconds_since_first": (
        "How long this device has been on the account; -1 if it is new or absent. A device "
        "minted an hour ago is not a trusted one."
    ),
}


def feature_specs(windows_s: tuple[int, ...] = VELOCITY_WINDOWS_S) -> list[FeatureSpec]:
    """Every column the builder emits, in emission order, each with its rationale.

    `FeatureBuilder.transform` fills a positional buffer, so this list *is* the column order.
    `tests/test_features.py` asserts the two agree — a spec added without a value, or a value
    added without a spec, fails the suite rather than silently shifting every column by one.
    """

    def fixed(block: dict[str, str], group: str) -> list[FeatureSpec]:
        return [FeatureSpec(name, group, why) for name, why in block.items()]

    def windowed(block: dict[str, str], group: str, windows: tuple[int, ...]) -> list[FeatureSpec]:
        out = []
        for w in windows:
            lab = window_label(w)
            for name, why in block.items():
                out.append(FeatureSpec(name.format(w=w), group, why.format(w=w, lab=lab)))
        return out

    dwell = (DWELL_WINDOW_S,)
    return [
        *fixed(_ROW_FEATURES, "row"),
        *windowed(_SRC_OUT_WINDOWED, "src-outbound", windows_s),
        *fixed(_SRC_OUT_SCALARS, "src-outbound"),
        *windowed(_DST_IN_WINDOWED, "beneficiary-inbound", windows_s),
        *fixed(_DST_IN_SCALARS, "beneficiary-inbound"),
        *windowed(_PASS_THROUGH, "pass-through", dwell),
        *fixed(_PAIR_FEATURES, "relationship"),
        *fixed(_DEVICE_FEATURES, "device"),
    ]


def feature_names(windows_s: tuple[int, ...] = VELOCITY_WINDOWS_S) -> list[str]:
    """Column order, without the prose."""
    return [spec.name for spec in feature_specs(windows_s)]


# ── per-entity state ────────────────────────────────────────────────────────────
class _Stream:
    """One direction of one entity's history — what it sent, or what it received.

    Parallel sorted arrays with prefix sums rather than a list of events, so a window query is
    two bisects and a subtraction instead of a scan. That is not premature: the scanning version
    re-read the whole retained history once per window per row, which is what made a 1.3M-row
    anchor take half a minute for a third of the features.

    Amounts are summed **shifted by the account's first amount** rather than raw. Summing raw and
    then taking `E[x^2] - E[x]^2` is the textbook unstable variance, and an account that pays
    nearly the same amount every time — a subscription, an EMI, a salary — is exactly the case it
    fails on: the two terms agree to fifteen digits and the difference is rounding error. That
    matters here because `src_amount_z` is the drift tell behind S3, C1 and M3, so a spurious
    variance is a spurious "this account is behaving unusually". Shifting costs one float per
    stream and makes the variance exact when the amounts are constant.
    """

    __slots__ = ("ts", "other", "cum", "cum2", "firsts", "shift")

    def __init__(self) -> None:
        self.ts: list[float] = []  # event times, non-decreasing
        self.other: list[int] = []  # counterparty code per event
        self.cum: list[float] = [0.0]  # prefix sums of (amount - shift); len == len(ts) + 1
        self.cum2: list[float] = [0.0]  # prefix sums of (amount - shift)**2, for the deviation
        self.firsts: list[float] = []  # sorted first-contact time per distinct counterparty
        self.shift: float = 0.0  # the first amount this stream saw; see the class docstring

    def before(self, t: float) -> int:
        """How many events happened at or before `t`. Every as-of-t count hangs off this."""
        return bisect_right(self.ts, t)

    def total(self, lo: int, hi: int) -> float:
        """Amount summed over events `[lo, hi)`, with the shift added back in."""
        return self.cum[hi] - self.cum[lo] + self.shift * (hi - lo)

    def window(self, t: float, w: float) -> tuple[int, int]:
        """Index range `[lo, hi)` of events in `[t - w, t]`.

        Both bounds matter. Without the upper one a stateful builder that has already seen later
        traffic would answer with rows from the future — which is exactly the bug this module
        exists to make impossible.
        """
        return bisect_left(self.ts, t - w), bisect_right(self.ts, t)

    def rewind(self, n: int) -> None:
        """Drop back to `n` events — how the journal unwinds a scoring pass."""
        del self.ts[n:]
        del self.other[n:]
        del self.cum[n + 1 :]
        del self.cum2[n + 1 :]


class _Node:
    """One entity: both directions, its devices, and when it was first seen at all."""

    __slots__ = ("out", "inn", "devices", "device_firsts", "first_ts")

    def __init__(self) -> None:
        self.out: _Stream | None = None  # lazily built: most PaySim entities only ever pay once
        self.inn: _Stream | None = None
        self.devices: dict[int, float] | None = None  # device code -> first seen on this account
        self.device_firsts: list[float] = []
        self.first_ts: float = float("inf")


# Journal opcodes. Scoring (`update=False`) observes each row and then rolls the state back, so
# the undo record has to be cheap: one tuple per mutated container rather than a deep copy of a
# state that, on the real anchors, is hundreds of megabytes.
_REWIND = 0  # (stream, n)                    -> stream.rewind(n)
_DEL_AT = 1  # (list, index)                  -> del list[index]
_POP_KEY = 2  # (dict, key)                   -> dict.pop(key)
_SET_KEY = 3  # (dict, key, old)              -> dict[key] = old
_CLEAR_STREAM = 4  # (node, attr)             -> setattr(node, attr, None)
_NODE_FIRST = 5  # (node, old)                -> node.first_ts = old
_RESTORE_LIST = 6  # (list, copy)             -> list[:] = copy
_RESTORE_TAIL = 7  # (stream, i, 4 suffixes)  -> every array back as it was, from index i


@dataclass
class FeatureBuilder:
    """Transactions in, one causal feature row out.

    `stateful=True` carries entity history between calls, because a mule's history does not reset
    because a new round began. `stateful=False` (the default) rebuilds from the batch alone.
    """

    windows_s: tuple[int, ...] = VELOCITY_WINDOWS_S
    stateful: bool = False

    _codes: dict[str, int] = field(default_factory=dict, repr=False)
    _nodes: dict[int, _Node] = field(default_factory=dict, repr=False)
    # (src, dst) -> the times that pair transacted. Held once, here, rather than on both
    # entities: the same list answers "has this payer ever paid this payee" from either side.
    _pairs: dict[tuple[int, int], list[float]] = field(default_factory=dict, repr=False)
    _journal: list | None = field(default=None, repr=False)

    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.windows_s = tuple(int(w) for w in self.windows_s)
        self._specs = feature_specs(self.windows_s)
        self.feature_names = [s.name for s in self._specs]

    @property
    def specs(self) -> list[FeatureSpec]:
        """The registry for exactly the windows this builder was configured with."""
        return list(self._specs)

    def reset(self) -> None:
        self._codes.clear()
        self._nodes.clear()
        self._pairs.clear()
        self._journal = None

    # ── the public surface ──────────────────────────────────────────────────────
    def transform(self, txns: list[Transaction], update: bool = True) -> pd.DataFrame:
        """One row of features per transaction, in the order given (index = txn_id).

        Rows are *computed* in timestamp order — features are causal — but written back to their
        input position. Selecting by label instead would silently multiply rows whenever a
        txn_id appears twice, which is exactly what a replay buffer does.

        `update=False` computes against the running history without committing to it. Scoring
        must use it: a detector whose feature state changes every time it scores answers the
        same question differently depending on how often it has been asked, and re-scoring the
        same holdout each round would quietly inflate that entity's velocity counts. Rows inside
        the batch still see the rows before them — by the time the second payment of a burst is
        scored in production, the first one has happened.
        """
        if not self.stateful:
            self.reset()

        journal = [] if not update else None
        self._journal = journal
        buf = np.empty((len(txns), len(self._specs)), dtype="float64")
        for i in sorted(range(len(txns)), key=lambda i: txns[i].ts):
            buf[i] = self._row(txns[i])
        if journal is not None:
            _undo(journal)
        self._journal = None

        df = pd.DataFrame(buf, columns=self.feature_names, index=[t.txn_id for t in txns])
        df.index.name = "txn_id"
        assert_no_forbidden_columns(df)
        return df

    @staticmethod
    def labels(txns: list[Transaction]) -> np.ndarray:
        return np.array([int(t.is_fraud) for t in txns], dtype=int)

    # ── per-row, causal ─────────────────────────────────────────────────────────
    def _row(self, t: Transaction) -> list[float]:
        """Features for one transaction from events strictly before it — then observe it.

        Reads the state, then writes to it, in that order, always. The causality guarantee is
        the one `self._observe(...)` line at the bottom of this method.
        """
        ts = t.ts.timestamp()
        amount = t.amount
        codes = self._codes
        s_code = codes.get(t.src)
        d_code = codes.get(t.dst)
        src = self._nodes.get(s_code) if s_code is not None else None
        dst = self._nodes.get(d_code) if d_code is not None else None
        s_out = src.out if src is not None else None
        s_in = src.inn if src is not None else None
        d_in = dst.inn if dst is not None else None
        d_out = dst.out if dst is not None else None

        hour = t.ts.hour
        weekday = t.ts.weekday()
        rail = t.rail.value
        f: list[float] = [
            amount,
            float(np.log1p(amount)),
            hour + t.ts.minute / 60.0,
            float(hour < 6),
            float(weekday),
            float(weekday >= 5),
            float(rail == "card"),
            float(rail == "upi"),
            float(rail == "a2a"),
            float(abs(amount - round(amount, -2)) < 1e-9),
            float(9_000 <= amount < 10_000),
            float(900 <= amount < 1_000),
        ]

        # ── the sender's outbound history: velocity, then RFM
        for w in self.windows_s:
            lo, hi = s_out.window(ts, w) if s_out is not None else (0, 0)
            f.append(float(hi - lo))
            f.append(s_out.total(lo, hi) if hi > lo else 0.0)
            f.append(float(len(set(s_out.other[lo:hi]))) if hi > lo else 0.0)

        n_out = s_out.before(ts) if s_out is not None else 0
        mean_out, std_out = _mean_std(s_out, n_out)
        f.append(float(n_out))
        f.append(ts - s_out.ts[n_out - 1] if n_out else NEVER)
        f.append(mean_out)
        f.append(std_out)
        f.append((amount - mean_out) / std_out if std_out > 0 else 0.0)
        f.append(amount / mean_out if mean_out else 1.0)
        f.append(float(bisect_right(s_out.firsts, ts)) if s_out is not None else 0.0)
        f.append(_age(src, ts))

        # ── the beneficiary's inbound history: fan-in, then novelty
        for w in self.windows_s:
            lo, hi = d_in.window(ts, w) if d_in is not None else (0, 0)
            f.append(float(hi - lo))
            f.append(d_in.total(lo, hi) if hi > lo else 0.0)
            f.append(float(len(set(d_in.other[lo:hi]))) if hi > lo else 0.0)

        n_in_dst = d_in.before(ts) if d_in is not None else 0
        mean_in, std_in = _mean_std(d_in, n_in_dst)
        f.append(float(n_in_dst))
        f.append(float(bisect_right(d_in.firsts, ts)) if d_in is not None else 0.0)
        f.append(ts - d_in.ts[n_in_dst - 1] if n_in_dst else NEVER)
        f.append(mean_in)
        f.append((amount - mean_in) / std_in if std_in > 0 else 0.0)
        f.append(float(n_in_dst == 0))
        f.append(_age(dst, ts))

        # ── pass-through: what arrived here, and how fast it is leaving again
        lo, hi = s_in.window(ts, DWELL_WINDOW_S) if s_in is not None else (0, 0)
        in_sum = s_in.total(lo, hi) if hi > lo else 0.0
        n_in_src = s_in.before(ts) if s_in is not None else 0
        o_lo, o_hi = s_out.window(ts, DWELL_WINDOW_S) if s_out is not None else (0, 0)
        out_sum = s_out.total(o_lo, o_hi) if o_hi > o_lo else 0.0
        f.append(float(hi - lo))
        f.append(in_sum)
        f.append(ts - s_in.ts[n_in_src - 1] if n_in_src else NEVER)
        f.append(out_sum / in_sum if in_sum > 0 else 0.0)
        d_lo, d_hi = d_out.window(ts, DWELL_WINDOW_S) if d_out is not None else (0, 0)
        f.append(float(d_hi - d_lo))

        # ── the pair: has this payer ever paid this payee
        pair = (
            self._pairs.get((s_code, d_code)) if s_code is not None and d_code is not None else None
        )
        k = bisect_right(pair, ts) if pair else 0
        f.append(float(k == 0))
        f.append(float(k))
        f.append(ts - pair[k - 1] if k else NEVER)

        # ── device, attributed to the paying account
        dev_code = codes.get(t.device_id) if t.device_id is not None else None
        first_seen = (
            src.devices.get(dev_code)
            if src is not None and src.devices is not None and dev_code is not None
            else None
        )
        f.append(float(bisect_right(src.device_firsts, ts)) if src is not None else 0.0)
        f.append(float(t.device_id is not None and (first_seen is None or first_seen > ts)))
        f.append(ts - first_seen if first_seen is not None and first_seen <= ts else NEVER)

        self._observe(t, ts)
        return f

    # ── writing to history ──────────────────────────────────────────────────────
    def _observe(self, t: Transaction, ts: float) -> None:
        """Commit the row — strictly after its own features were computed."""
        j = self._journal
        s_code = self._code(t.src, j)
        d_code = self._code(t.dst, j)
        src = self._node(s_code, j)
        dst = self._node(d_code, j)

        _set_first_ts(src, ts, j)
        _set_first_ts(dst, ts, j)
        out = _stream(src, "out", j)
        inn = _stream(dst, "inn", j)

        key = (s_code, d_code)
        pair = self._pairs.get(key)
        if pair is None:
            self._pairs[key] = [ts]
            if j is not None:
                j.append((_POP_KEY, self._pairs, key))
            _insert(out.firsts, ts, j)
            _insert(inn.firsts, ts, j)
        else:
            was_first = pair[0]
            _insert(pair, ts, j)
            if ts < was_first:  # an out-of-order row that predates the pair's first contact
                _move(out.firsts, was_first, ts, j)
                _move(inn.firsts, was_first, ts, j)

        _add_event(out, ts, t.amount, d_code, j)
        _add_event(inn, ts, t.amount, s_code, j)

        if t.device_id is not None:
            dev = self._code(t.device_id, j)
            if src.devices is None:
                src.devices = {}
            seen = src.devices.get(dev)
            if seen is None:
                src.devices[dev] = ts
                if j is not None:
                    j.append((_POP_KEY, src.devices, dev))
                _insert(src.device_firsts, ts, j)
            elif ts < seen:
                if j is not None:
                    j.append((_SET_KEY, src.devices, dev, seen))
                src.devices[dev] = ts
                _move(src.device_firsts, seen, ts, j)

    def _code(self, name: str, j: list | None) -> int:
        code = self._codes.get(name)
        if code is None:
            code = len(self._codes)
            self._codes[name] = code
            if j is not None:
                j.append((_POP_KEY, self._codes, name))
        return code

    def _node(self, code: int, j: list | None) -> _Node:
        node = self._nodes.get(code)
        if node is None:
            node = self._nodes[code] = _Node()
            if j is not None:
                j.append((_POP_KEY, self._nodes, code))
        return node

    def state_size(self) -> dict[str, int]:
        """How much history the builder is holding — the memory story without a profiler.

        Retention is unbounded on purpose: `src_out_txn_count`, `dst_in_degree` and the two
        account ages are exact as-of-the-row *because* nothing is ever dropped. The cost is
        linear in events (two per transaction, one on each side), and this is how to see it.
        """
        events = 0
        for node in self._nodes.values():
            if node.out is not None:
                events += len(node.out.ts)
            if node.inn is not None:
                events += len(node.inn.ts)
        return {
            "entities": len(self._nodes),
            "events": events,
            "pairs": len(self._pairs),
            "devices": sum(len(n.devices or ()) for n in self._nodes.values()),
        }

    # ── what the table actually contains ────────────────────────────────────────
    def coverage(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per column: how much of it is actually there, on this data.

        Ticket 02 found that PaySim has no sender history, which makes a whole block of this
        table structurally empty on that anchor. A dead column is not a bug; a dead column
        nobody noticed is, because it reads as a feature the model has and it is not one. So the
        emptiness is measured and published rather than inferred from the data card.
        """
        by_name = {s.name: s for s in self._specs}
        rows = []
        for name in X.columns:
            v = X[name].to_numpy()
            distinct = int(np.unique(v).size) if v.size else 0
            modal = float(pd.Series(v).mode().iloc[0]) if v.size else 0.0
            spec = by_name.get(name)
            rows.append(
                {
                    "feature": name,
                    "group": spec.group if spec else "?",
                    "distinct_values": distinct,
                    "share_informative": float((v != modal).mean()) if v.size else 0.0,
                    "share_never": float((v == NEVER).mean()) if v.size else 0.0,
                    "mean": float(np.mean(v)) if v.size else 0.0,
                    "dead": bool(distinct <= 1),
                }
            )
        return pd.DataFrame(rows).set_index("feature")


# ── small helpers, kept out of the hot loop's namespace ─────────────────────────
def _mean_std(stream: _Stream | None, n: int) -> tuple[float, float]:
    """Mean and population standard deviation of the first `n` events; `n` is already as-of-t."""
    if stream is None or n == 0:
        return 0.0, 0.0
    # the moments are of `amount - shift`, so the variance is exact where it matters and the
    # mean just needs the shift added back
    centred = stream.cum[n] / n
    mean = stream.shift + centred
    if n < 2:
        return mean, 0.0
    var = stream.cum2[n] / n - centred * centred
    return mean, sqrt(var) if var > 0 else 0.0


def _age(node: _Node | None, ts: float) -> float:
    """Seconds since the entity was first seen anywhere, or `NEVER` if this is its first row."""
    if node is None or node.first_ts > ts:
        return NEVER
    return ts - node.first_ts


def _stream(node: _Node, attr: str, j: list | None) -> _Stream:
    stream = getattr(node, attr)
    if stream is None:
        stream = _Stream()
        setattr(node, attr, stream)
        if j is not None:
            j.append((_CLEAR_STREAM, node, attr))
    return stream


def _set_first_ts(node: _Node, ts: float, j: list | None) -> None:
    if ts < node.first_ts:
        if j is not None:
            j.append((_NODE_FIRST, node, node.first_ts))
        node.first_ts = ts


def _add_event(stream: _Stream, ts: float, amount: float, other: int, j: list | None) -> None:
    """Append in time order; fall back to an insertion when a row predates the history.

    The fast path is the only one a normal run takes: `transform` feeds rows in timestamp order,
    so each event lands at the end. The slow path exists because "stateful" cannot quietly mean
    "assumes every future batch starts after every past one" — a replay buffer breaks that on
    its first round, and the wrong answer there would be a silently non-causal feature.
    """
    n = len(stream.ts)
    if n and ts < stream.ts[-1]:
        i = bisect_right(stream.ts, ts)
        cum, cum2 = stream.cum, stream.cum2
        if j is not None:
            # only the suffix moves, so only the suffix is worth copying — the undo record for a
            # late insertion costs nothing, and it is a late insertion nearly every time
            j.append(
                (
                    _RESTORE_TAIL,
                    stream,
                    i,
                    stream.ts[i:],
                    stream.other[i:],
                    cum[i + 1 :],
                    cum2[i + 1 :],
                )
            )
        a = amount - stream.shift
        a2 = a * a
        stream.ts.insert(i, ts)
        stream.other.insert(i, other)
        cum.insert(i + 1, cum[i] + a)
        cum2.insert(i + 1, cum2[i] + a2)
        for k in range(i + 2, n + 2):  # everything after the insertion shifts by one event
            cum[k] += a
            cum2[k] += a2
        return
    if j is not None:
        j.append((_REWIND, stream, n))
    if n == 0:
        # the shift is the stream's first amount, re-taken after a rewind empties it
        stream.shift = amount
    a = amount - stream.shift
    stream.ts.append(ts)
    stream.other.append(other)
    stream.cum.append(stream.cum[-1] + a)
    stream.cum2.append(stream.cum2[-1] + a * a)


def _insert(values: list[float], ts: float, j: list | None) -> None:
    """Keep a sorted list sorted. Appends at the end on the fast path, which is nearly always."""
    i = bisect_right(values, ts)
    values.insert(i, ts)
    if j is not None:
        j.append((_DEL_AT, values, i))


def _move(firsts: list[float], old: float, new: float, j: list | None) -> None:
    """A counterparty's first contact moved earlier — keep the sorted first-contact list honest."""
    if j is not None:
        j.append((_RESTORE_LIST, firsts, list(firsts)))
    i = bisect_left(firsts, old)
    if i < len(firsts) and firsts[i] == old:
        del firsts[i]
    firsts.insert(bisect_right(firsts, new), new)


def _undo(journal: list) -> None:
    """Roll the state back to where the scoring pass found it, newest mutation first."""
    for entry in reversed(journal):
        op = entry[0]
        if op == _REWIND:
            entry[1].rewind(entry[2])
        elif op == _DEL_AT:
            del entry[1][entry[2]]
        elif op == _POP_KEY:
            entry[1].pop(entry[2], None)
        elif op == _SET_KEY:
            entry[1][entry[2]] = entry[3]
        elif op == _CLEAR_STREAM:
            setattr(entry[1], entry[2], None)
        elif op == _NODE_FIRST:
            entry[1].first_ts = entry[2]
        elif op == _RESTORE_LIST:
            entry[1][:] = entry[2]
        elif op == _RESTORE_TAIL:
            stream, i, ts_tail, other_tail, cum_tail, cum2_tail = entry[1:]
            stream.ts[i:] = ts_tail
            stream.other[i:] = other_tail
            stream.cum[i + 1 :] = cum_tail
            stream.cum2[i + 1 :] = cum2_tail


def assert_no_forbidden_columns(X: pd.DataFrame) -> None:
    """The answer key never reaches the model, and the check is cheap enough to always run."""
    leaked = [c for c in FORBIDDEN if c in X.columns]
    if leaked:
        raise AssertionError(
            f"forbidden column(s) {leaked} in the feature table — these are the label and its "
            "provenance, and a model that reads them scores 1.0 having learnt nothing"
        )


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
