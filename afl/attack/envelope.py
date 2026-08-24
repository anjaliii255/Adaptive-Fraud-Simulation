"""The anchor envelope: the shape of the real traffic an attack is injected into.

Synthetic attacks are only a detection problem if they are commensurate with the data around
them. Left uncalibrated they are not: the simulator's own defaults put PaySim's traffic in 2023
and its attacks in 2024, at a thousandth of the amount, so an "unseen attack family" is separable
from the anchor by timestamp or by order of magnitude before any behaviour is considered. A
detector that scores well on that has learned which generator made the row.

So the anchor gets measured first, and the simulator is placed inside it: same window, same
amount scale, same pacing order. What is left for the detector to find is behaviour, which is the
only thing the claim is about.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from afl.attack.actors import ActorParams
from afl.contract.schema import Rail, Transaction

ENVELOPE_VERSION = 1
DEFAULT_ENVELOPE_DIR = Path("artifacts/envelopes")


@dataclass(frozen=True)
class AnchorEnvelope:
    """Measured shape of one anchor dataset. Everything here comes from the file, not the docs."""

    dataset: str
    n_rows: int
    start: datetime
    end: datetime
    amount_log_mu: float
    amount_log_sigma: float
    amount_p01: float
    amount_p99: float
    median_gap_s: float
    fraud_base_rate: float
    rail_mix: dict[str, float]
    sender_reuse_rate: float
    device_coverage: float
    time_granularity_s: int
    active_senders: list[str]
    version: int = ENVELOPE_VERSION

    @property
    def window_days(self) -> int:
        return max(1, int((self.end - self.start).total_seconds() // 86_400))

    def contains_amount(self, amount: float) -> bool:
        """Inside the anchor's own bulk, so an attack cannot be found by order of magnitude."""
        return self.amount_p01 <= amount <= self.amount_p99

    @property
    def carries_devices(self) -> bool:
        """Neither real anchor has a device column; inventing one is a perfect label."""
        return self.device_coverage > 0.01

    @property
    def supports_behavioural_vectors(self) -> bool:
        """Whether accounts here transact more than once.

        PaySim has 636,323 distinct senders across 636,409 rows, so no account has a history to
        drift away from and every `src_*` feature is structurally zero. Velocity and drift
        families cannot be posed as a detection problem on an anchor like that.
        """
        return self.sender_reuse_rate > 0.5

    @property
    def rails(self) -> tuple[Rail, ...]:
        """The rails the anchor actually carries, commonest first."""
        ordered = sorted(self.rail_mix.items(), key=lambda kv: kv[1], reverse=True)
        return tuple(Rail(name) for name, share in ordered if share > 0)

    def rescale(self, actor: ActorParams) -> ActorParams:
        """Move an actor onto the anchor's amount scale and rails.

        The amount shift is in log space rather than a rewrite: a fraudster still pays more than a
        normal user by the same factor it always did, the whole population just lands where the
        real money is.

        Rails are narrowed to what the anchor carries, because a rail the anchor never uses is a
        perfect label. On PaySim every real row is A2A while an unanchored drift vector settles on
        CARD, and `rail` alone then separates synthetic from real at PR-AUC 1.0 — no model needed.
        A vector whose own rails are absent from the anchor keeps its preference, and the
        commensurability audit reports it rather than the run pretending it fits.
        """
        shift = self.amount_log_mu - _REFERENCE_LOG_MU
        spread = self.amount_log_sigma / _REFERENCE_LOG_SIGMA
        shared = tuple(r for r in actor.rails if r in self.rails) or self.rails or actor.rails
        return ActorParams(
            **{
                **actor.__dict__,
                "amount_mu": actor.amount_mu + shift,
                "amount_sigma": max(0.05, actor.amount_sigma * spread),
                "rails": shared,
            }
        )

    # ── persistence ─────────────────────────────────────────────────────────────
    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> AnchorEnvelope:
        raw = json.loads(Path(path).read_text())
        raw["start"] = datetime.fromisoformat(raw["start"])
        raw["end"] = datetime.fromisoformat(raw["end"])
        return cls(**raw)

    @classmethod
    def measure(cls, txns: list[Transaction], dataset: str) -> AnchorEnvelope:
        """Fit to real rows. Legit only: fraud is the thing being imitated, not the baseline."""
        legit = [t for t in txns if not t.is_fraud] or list(txns)
        amounts = np.array([t.amount for t in legit], dtype=float)
        amounts = amounts[amounts > 0]
        logs = np.log(amounts)
        stamps = np.sort([t.ts.timestamp() for t in legit])
        gaps = np.diff(stamps)
        return cls(
            dataset=dataset,
            n_rows=len(txns),
            start=min(t.ts for t in txns),
            end=max(t.ts for t in txns),
            amount_log_mu=float(logs.mean()),
            amount_log_sigma=float(logs.std() or 1.0),
            amount_p01=float(np.percentile(amounts, 1)),
            amount_p99=float(np.percentile(amounts, 99)),
            median_gap_s=float(np.median(gaps)) if gaps.size else 0.0,
            fraud_base_rate=sum(t.is_fraud for t in txns) / max(len(txns), 1),
            rail_mix=_shares([t.rail.value for t in legit]),
            sender_reuse_rate=1.0 - len({t.src for t in legit}) / max(len(legit), 1),
            device_coverage=sum(t.device_id is not None for t in legit) / max(len(legit), 1),
            time_granularity_s=_granularity([t.ts for t in legit]),
            active_senders=_busiest_senders(legit),
        )


#: Accounts carried into the simulated population so attacks happen on real, seasoned accounts.
MAX_ACTIVE_SENDERS = 500


def _busiest_senders(txns: list[Transaction], limit: int = MAX_ACTIVE_SENDERS) -> list[str]:
    """The anchor's most active accounts, so a synthetic attack has a real history behind it.

    Without this the simulator invents its own entity namespace and every attack sender has no
    anchor history at all, which is a label the moment the anchor's accounts do.
    """
    counts: dict[str, int] = {}
    for t in txns:
        counts[t.src] = counts.get(t.src, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [src for src, n in ranked[:limit] if n > 1]


def _granularity(stamps: list[datetime]) -> int:
    """The coarsest unit every real timestamp lands on.

    AMLSim's steps are whole days, so every row sits at midnight. Synthetic traffic spread across
    the clock is then separable on hour-of-day alone at PR-AUC 0.93.
    """
    # read the calendar fields, not the POSIX timestamp: a naive datetime converts through the
    # local zone, and a half-hour offset makes every day look like it lands mid-hour
    if all((ts.hour, ts.minute, ts.second, ts.microsecond) == (0, 0, 0, 0) for ts in stamps):
        return 86_400
    if all((ts.minute, ts.second, ts.microsecond) == (0, 0, 0) for ts in stamps):
        return 3_600
    return 1


def _shares(values: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return {k: c / max(len(values), 1) for k, c in counts.items()}


#: The actor bundle amounts were authored against, so a rescale is expressed relative to it
#: rather than overwriting the actor definitions.
_REFERENCE_LOG_MU = 3.2
_REFERENCE_LOG_SIGMA = 0.9


def overlap_days(envelope: AnchorEnvelope, txns: list[Transaction]) -> float:
    """Days of calendar overlap between generated traffic and the anchor. Zero is a broken split."""
    if not txns:
        return 0.0
    start, end = min(t.ts for t in txns), max(t.ts for t in txns)
    latest_start, earliest_end = max(start, envelope.start), min(end, envelope.end)
    return max(0.0, (earliest_end - latest_start).total_seconds() / 86_400)


def scale_gap(envelope: AnchorEnvelope, amounts: list[float]) -> float:
    """How many orders of magnitude the generated amounts sit away from the anchor's centre."""
    if not amounts:
        return 0.0
    positive = [a for a in amounts if a > 0]
    if not positive:
        return 0.0
    return abs(float(np.median(np.log10(positive))) - envelope.amount_log_mu / math.log(10))


# ── the guard ───────────────────────────────────────────────────────────────────
#: Above this, one signal alone separates synthetic from real well enough that any model score on
#: that holdout is measuring provenance. PR-AUC has to clear both an absolute bar and a multiple
#: of the base rate: at a 0.1% base rate 0.25 is enormous, at 14% it is barely above chance.
TRIVIAL_SEPARATION = 0.25
TRIVIAL_LIFT = 3.0


def _pr_auc(labels: np.ndarray, score: np.ndarray) -> float:
    """Average precision, taking the better of the signal and its negation."""
    from sklearn.metrics import average_precision_score

    if labels.sum() in (0, labels.size):
        return 0.0
    return float(
        max(average_precision_score(labels, score), average_precision_score(labels, -score))
    )


def audit(real: list[Transaction], synth: list[Transaction]) -> dict[str, object]:
    """Can one contract field alone tell the generated rows from the real ones?

    Four versions of this bug shipped before the check existed: amounts a thousand times too
    small, a rail the anchor never carries, a device column the anchor does not have, and
    accounts invented in a namespace of their own. Each made an unseen-attack score look like
    skill. Anything a single field can do here, a model will do first.

    Whether the anchor can host behavioural attacks at all is a separate question, answered by
    `supports_behavioural_vectors`, not by this.

    Contract fields only, so this stays on the red side of the seam and never learns the
    detector's feature space.
    """
    rows = list(real) + list(synth)
    if not real or not synth:
        return {"signals": {}, "worst": None, "score": 0.0, "trivially_separable": False}
    labels = np.array([0] * len(real) + [1] * len(synth))

    # membership, not activity: an attack is *supposed* to concentrate transactions on an
    # account, so counting them would penalise the behaviour we are trying to generate. What
    # must not happen is the attack running on accounts the anchor has never seen.
    anchor_accounts = {t.src for t in real} | {t.dst for t in real}

    signals = {
        "log_amount": np.log([max(t.amount, 1e-9) for t in rows]),
        "hour_of_day": np.array([t.ts.hour for t in rows], dtype=float),
        "rail": np.array([hash(t.rail.value) % 997 for t in rows], dtype=float),
        "sender_in_anchor": np.array([t.src in anchor_accounts for t in rows], dtype=float),
        "payee_in_anchor": np.array([t.dst in anchor_accounts for t in rows], dtype=float),
        "has_device": np.array([t.device_id is not None for t in rows], dtype=float),
    }
    scored = {name: round(_pr_auc(labels, v), 4) for name, v in signals.items()}
    worst = max(scored, key=scored.get)
    base_rate = float(labels.mean())
    return {
        "signals": scored,
        "worst": worst,
        "score": scored[worst],
        "base_rate": round(base_rate, 4),
        "trivially_separable": scored[worst] > max(TRIVIAL_SEPARATION, TRIVIAL_LIFT * base_rate),
    }
