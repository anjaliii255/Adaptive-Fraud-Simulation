"""The sudden-vs-gradual axis, and the gate a sequence model has to pass to be reported.

Ticket 17. A sequence model earns its seat or it does not, and either way the comparison is
published. This module holds the three things that decision needs and `scripts/build_sequence.py`
orchestrates:

**The axis.** `ramp` is the drift engine's only shape knob: 0 is a hard switch at the takeover
event, 1 is escalation spread across the whole tail. A sudden takeover is an *event* — the
amount jumps, the beneficiary is new, the device changes — and a per-row feature table sees all
three on the row itself. Gradual drift has no event to anchor on: every individual row is
unremarkable and only the trajectory is wrong. Averaging the two into one number hides which of
them paid for the result, so `arc_breakdown` reports them separately against the same haystack at
the same threshold. The two ramp values come from each vector's own declared search space, so an
arc cannot be generated outside the realism envelope ticket 14 set.

**The precondition.** A model over per-entity history has nothing to read on an anchor whose
entities appear once. PaySim is exactly that — `nameOrig` is effectively unique per row — so its
real windows are one step long while every injected drift episode carries a full arc, and window
length alone separates the injected rows from the real ones. `history_audit` measures that in the
same shape `envelope.audit` and the leave-one-attack-out provenance probe use, and a fold it
flags is withheld rather than reported.

**The gate.** `decide_promotion` is the ticket's rule written down: the sequence model enters the
reported table only if it beats LightGBM on the **gradual** arc, on a fold that can carry a claim,
by more than the margin this project treats as a difference — and only if it also clears the
amount floor there. A win on sudden drift does not promote it; that is the easy end, and a model
that only wins there is not doing anything the per-row table was not already doing.

`assert_config_matches_promotion` closes the loop back onto config: `defend.sequence.enabled`
cannot be true while a committed artefact says the gate refused it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from afl.attack.templates import registry
from afl.contract.metrics import MetricResult
from afl.contract.schema import Transaction
from afl.evaluation import protocol
from afl.evaluation.leave_one_attack_out import (
    MEASURED,
    MIN_MEANINGFUL_POSITIVES,
    OUTCOMES,
    SKIPPED,
    WITHHELD,
    is_provenance_bound,
)
from afl.utils.runcard import with_provenance

# ── the axis ────────────────────────────────────────────────────────────────────
SUDDEN = "sudden"
GRADUAL = "gradual"
ARCS = (SUDDEN, GRADUAL)

#: The two families the drift engine puts on this axis. S3 is a takeover of a real account; C1 is
#: a genuinely old account busting out. M2 and M3 also ride the drift engine and are deliberately
#: not here: M3 is the leave-one-attack-out holdout and is always gradual by declaration, so it
#: has no sudden end to compare against, and M2's tell is the thinness of the file rather than
#: the shape of the tail.
DRIFT_ARC_FAMILIES = ("S3", "C1")

#: A PR-AUC difference this project treats as a difference. Same bar `docs/anomaly.md` reads its
#: tables at: below it, two systems are reported as level rather than as a win.
MATERIAL_GAP = 0.01

SEQUENCE_ARTEFACT_VERSION = 1

DEFAULT_SEQUENCE_DIR = Path(os.getenv("AFL_SEQUENCE_DIR", "artifacts/sequence"))


def arc_ramp(vector_id: str, arc: str) -> float:
    """The `ramp` value for one end of one vector's axis, taken from its own search space.

    Read out of `vectors.yaml` rather than written here, so an arc is always inside the realism
    envelope the vector declares. The two families do not share an envelope and are not supposed
    to: C1's gradual end is 0.6 because a bust-out spike is an event, not a drift, and quoting
    S3's 1.0 for it would be generating a family C1 is not.
    """
    if arc not in ARCS:
        raise ValueError(f"unknown arc {arc!r}; expected one of {ARCS}")
    space = registry.get(vector_id).search_space
    bounds = space.get("ramp")
    if not bounds:
        raise ValueError(
            f"{vector_id} does not declare `ramp` in its search space, so it has no "
            f"sudden-vs-gradual axis — {DRIFT_ARC_FAMILIES} are the families that do"
        )
    return float(bounds["low"] if arc == SUDDEN else bounds["high"])


def arc_params(vector_id: str, arc: str) -> dict[str, Any]:
    """The vector's own params with `ramp` moved to one end. Nothing else changes.

    One knob differs between the two batches, which is what makes the breakdown an axis rather
    than two unrelated configurations that happen to be printed next to each other.
    """
    return registry.clamp(vector_id, {"ramp": arc_ramp(vector_id, arc)})


def tag_arcs(rows: list[Transaction], arc_of_run: dict[str, str]) -> dict[str, str]:
    """txn_id -> arc, for the fraud rows an arc batch produced.

    Keyed off `attack_run_id`, which `Simulator.generate` stamps on every episode row and which
    already differs between the two batches because the knobs it hashes differ.
    """
    return {
        t.txn_id: arc_of_run[t.attack_run_id]
        for t in rows
        if t.is_fraud and t.attack_run_id in arc_of_run
    }


# ── the precondition ────────────────────────────────────────────────────────────
def window_lengths(rows: list[Transaction], max_len: int, entity: str = "src") -> np.ndarray:
    """How many steps of its own entity's history each row carries, capped at `max_len`."""
    from afl.defend.models.sequence import RAW_TS, raw_rows, window_index

    if not rows:
        return np.zeros(0, dtype=int)
    import zlib

    keys = np.array(
        [zlib.crc32((t.src if entity == "src" else t.dst).encode()) for t in rows], dtype=np.int64
    )
    idx = window_index(keys, raw_rows(rows)[:, RAW_TS], max_len)
    return (idx >= 0).sum(axis=1)


def history_audit(
    real: list[Transaction], injected: list[Transaction], max_len: int, entity: str = "src"
) -> dict[str, Any]:
    """Does window length alone sort the injected rows from the anchor's own?

    The sequence model's version of the provenance probe, asked about the one thing this model
    reads that the per-row table does not. It is measured over the two populations *together*,
    because that is the pool the fold scores: a real row's window is however much history the
    anchor gives it, and an injected episode carries its own arc with it.

    The verdict uses the same rule the leave-one-attack-out harness applies to its probe
    (`is_provenance_bound`), so a fold refused here is refused for the same reason and at the
    same bar as a fold refused there.
    """
    if not real or not injected:
        return {"checked": False, "reason": "one side of the comparison is empty"}
    rows = list(real) + list(injected)
    lengths = window_lengths(rows, max_len, entity).astype(float)
    labels = np.array([0] * len(real) + [1] * len(injected), dtype=int)
    base_rate = float(labels.mean())
    score = protocol.pr_auc(labels, lengths)
    real_len, inj_len = lengths[: len(real)], lengths[len(real) :]
    return {
        "checked": True,
        "entity": entity,
        "max_len": max_len,
        "pr_auc": round(float(score), 6),
        "base_rate": round(base_rate, 8),
        "separable": bool(is_provenance_bound(score, base_rate)),
        "anchor_mean_window": round(float(real_len.mean()), 4),
        "anchor_share_with_no_history": round(float((real_len == 1).mean()), 6),
        "injected_mean_window": round(float(inj_len.mean()), 4),
        "injected_share_with_no_history": round(float((inj_len == 1).mean()), 6),
        "why": (
            "a model over per-entity history reads window length whether or not it is asked to; "
            "where the anchor's entities appear once and the injected episodes carry a full arc, "
            "that length is the label"
        ),
    }


# ── the breakdown ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ArcResult:
    """One end of the axis for one system: an outcome, a reason, and numbers when quotable.

    Same discipline as `FoldResult`: `metrics` is `None` unless the arc carries enough positives
    to mean something, and the numbers that exist anyway live under `withheld_metrics` so quoting
    one is a decision somebody made rather than the default a reader falls into.
    """

    arc: str
    ramp: float
    outcome: str
    reason: str = ""
    metrics: MetricResult | None = None
    withheld_metrics: MetricResult | None = None
    n_positives: int = 0
    #: Recall on this arc's positives at the threshold set by the *whole* fold's legit traffic.
    #:
    #: It equals `metrics.recall_at_fixed_fpr` — and that equality is the point rather than a
    #: redundancy. Each arc is scored against every legit row of the fold, so the quantile that
    #: spends 1% of the negatives is literally the same score for both arcs, and the two columns
    #: agreeing is the evidence that the axis is read at one operating point. Computed from the
    #: whole fold rather than from the arc's own selection, so if the two ever diverge it means
    #: the haystack stopped being shared, which is the one way this comparison could quietly
    #: become two comparisons. `tests/test_sequence.py` asserts they agree.
    recall_at_shared_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown arc outcome {self.outcome!r}; expected one of {OUTCOMES}")
        if self.outcome != MEASURED and self.metrics is not None:
            raise ValueError(
                f"{self.arc}: a {self.outcome} arc carries numbers under `withheld_metrics`, "
                "never under `metrics` — that field is what a reader quotes"
            )
        if self.outcome != MEASURED and not self.reason:
            raise ValueError(f"{self.arc}: an arc that is not measured has to say why")

    @property
    def any_metrics(self) -> MetricResult | None:
        return self.metrics or self.withheld_metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "arc": self.arc,
            "ramp": self.ramp,
            "outcome": self.outcome,
            "reason": self.reason,
            "metrics": self.metrics.model_dump() if self.metrics else None,
            "withheld_metrics": self.withheld_metrics.model_dump()
            if self.withheld_metrics
            else None,
            "n_positives": self.n_positives,
            "recall_at_shared_threshold": round(self.recall_at_shared_threshold, 6),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ArcResult:
        return cls(
            arc=raw["arc"],
            ramp=float(raw["ramp"]),
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
            metrics=MetricResult(**raw["metrics"]) if raw.get("metrics") else None,
            withheld_metrics=MetricResult(**raw["withheld_metrics"])
            if raw.get("withheld_metrics")
            else None,
            n_positives=int(raw.get("n_positives", 0)),
            recall_at_shared_threshold=float(raw.get("recall_at_shared_threshold", 0.0)),
        )


def shared_threshold(
    rows: list[Transaction], probs: np.ndarray, fixed_fpr: float = protocol.DEFAULT_FPR
) -> float:
    """The score that spends exactly `fixed_fpr` of the fold's legit rows — one cut, both arcs."""
    legit = np.asarray(probs, dtype=float)[[not t.is_fraud for t in rows]]
    return float(np.quantile(legit, 1.0 - fixed_fpr)) if legit.size else float("inf")


def arc_breakdown(
    rows: list[Transaction],
    probs: np.ndarray,
    arcs: dict[str, str],
    ramps: dict[str, float],
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
    min_positives: int = MIN_MEANINGFUL_POSITIVES,
) -> dict[str, ArcResult]:
    """One arc at a time, against the same legit rows and at the same operating point.

    Each arc's ranking metrics are computed on (that arc's positives + *all* of the fold's legit
    traffic), so the haystack is identical and only the needles change. Anything else — carving
    the legit rows up per arc, or comparing an arc against its own episodes' background — would
    move the base rate between the two columns and make the comparison arithmetic rather than
    evidence.
    """
    probs = np.asarray(probs, dtype=float)
    legit = np.flatnonzero([not t.is_fraud for t in rows])
    cut = shared_threshold(rows, probs, fixed_fpr)

    out: dict[str, ArcResult] = {}
    for arc in ARCS:
        idx = np.array(
            [i for i, t in enumerate(rows) if t.is_fraud and arcs.get(t.txn_id) == arc], dtype=int
        )
        ramp = float(ramps.get(arc, float("nan")))
        recall = float((probs[idx] > cut).mean()) if idx.size else 0.0
        if not idx.size:
            out[arc] = ArcResult(
                arc=arc,
                ramp=ramp,
                outcome=SKIPPED,
                reason=(
                    f"no {arc}-drift rows landed in the holdout — the out-of-time cut can fall "
                    "after every episode of one end of the axis. Raise the arc episode count"
                ),
            )
            continue

        take = np.concatenate([idx, legit])
        y = np.array([int(rows[i].is_fraud) for i in take], dtype=int)
        result = protocol.evaluate(y, probs[take], fixed_fpr=fixed_fpr, k=k)
        if idx.size < min_positives:
            out[arc] = ArcResult(
                arc=arc,
                ramp=ramp,
                outcome=WITHHELD,
                reason=(
                    f"{idx.size} {arc}-drift positives against a floor of {min_positives} — "
                    f"recall moves {1 / idx.size:.1%} per row here, so this end of the axis is "
                    "reported as missing rather than as a low score"
                ),
                withheld_metrics=result,
                n_positives=int(idx.size),
                recall_at_shared_threshold=recall,
            )
            continue
        out[arc] = ArcResult(
            arc=arc,
            ramp=ramp,
            outcome=MEASURED,
            metrics=result,
            n_positives=int(idx.size),
            recall_at_shared_threshold=recall,
        )
    return out


# ── the gate ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Promotion:
    """Whether the sequence model may appear in a reported table, and why.

    A refusal is a result. `reason` is written to be quotable as-is, because the thing this
    ticket is most likely to lose is not the experiment — it is the record that the experiment
    was run and came back negative.
    """

    promoted: bool
    reason: str
    arc: str = GRADUAL
    metric: str = "pr_auc"
    challenger: float | None = None
    champion: float | None = None
    floor: float | None = None
    margin: float | None = None
    material_gap: float = MATERIAL_GAP

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "decided_on_arc": self.arc,
            "metric": self.metric,
            "challenger": self.challenger,
            "champion": self.champion,
            "floor": self.floor,
            "margin": self.margin,
            "material_gap": self.material_gap,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Promotion:
        return cls(
            promoted=bool(raw["promoted"]),
            reason=raw["reason"],
            arc=raw.get("decided_on_arc", GRADUAL),
            metric=raw.get("metric", "pr_auc"),
            challenger=raw.get("challenger"),
            champion=raw.get("champion"),
            floor=raw.get("floor"),
            margin=raw.get("margin"),
            material_gap=float(raw.get("material_gap", MATERIAL_GAP)),
        )


def decide_promotion(
    challenger: dict[str, ArcResult],
    champion: dict[str, ArcResult],
    floor: float | None = None,
    arc: str = GRADUAL,
    metric: str = "pr_auc",
    material_gap: float = MATERIAL_GAP,
    blocked: str = "",
) -> Promotion:
    """The ticket's rule, in the order a reader can check it.

    `blocked` carries a reason the whole fold cannot carry a claim — the family guard failed, or
    the history audit says window length is the label. It is checked first because a comparison
    inside an invalid fold is not a comparison, however it comes out.
    """
    common = {"arc": arc, "metric": metric, "material_gap": material_gap, "floor": floor}
    if blocked:
        return Promotion(promoted=False, reason=blocked, **common)

    mine, theirs = challenger.get(arc), champion.get(arc)
    if mine is None or theirs is None:
        return Promotion(
            promoted=False,
            reason=f"the {arc} end of the axis was not measured for both systems",
            **common,
        )
    if mine.outcome != MEASURED or theirs.outcome != MEASURED:
        unmeasured = mine if mine.outcome != MEASURED else theirs
        return Promotion(
            promoted=False,
            reason=(
                f"the {arc} arc cannot carry a claim — {unmeasured.reason}. The comparison is "
                "published under `arcs`; it is not evidence for a promotion"
            ),
            **common,
        )

    a = float(getattr(mine.metrics, metric))
    b = float(getattr(theirs.metrics, metric))
    common = {
        **common,
        "challenger": round(a, 6),
        "champion": round(b, 6),
        "margin": round(a - b, 6),
    }
    if floor is not None and a <= floor:
        return Promotion(
            promoted=False,
            reason=(
                f"on {arc} drift the sequence model reaches {metric} {a:.3f} against {floor:.3f} "
                "for sorting by amount alone — a detector a sort beats has not found anything"
            ),
            **common,
        )
    if a - b < material_gap:
        verdict = "loses to" if a < b else "is level with"
        return Promotion(
            promoted=False,
            reason=(
                f"on {arc} drift — the end this model exists for — it {verdict} LightGBM: "
                f"{metric} {a:.3f} against {b:.3f}, a margin of {a - b:+.3f} against a bar of "
                f"{material_gap:.2f}. It does not enter the reported table, and this line is the "
                "result"
            ),
            **common,
        )
    return Promotion(
        promoted=True,
        reason=(
            f"on {arc} drift the sequence model beats LightGBM on the same split at the same "
            f"operating point: {metric} {a:.3f} against {b:.3f}, a margin of {a - b:+.3f}"
        ),
        **common,
    )


# ── the artefact ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SystemResult:
    """One system on one fold: the whole holdout, then each end of the axis, then the bill."""

    name: str
    overall: MetricResult
    arcs: dict[str, ArcResult] = field(default_factory=dict)
    operational: dict[str, float] = field(default_factory=dict)
    compute: dict[str, Any] = field(default_factory=dict)
    model_card: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "overall": self.overall.model_dump(),
            "arcs": {a: r.to_dict() for a, r in self.arcs.items()},
            "operational": self.operational,
            "compute": self.compute,
            "model_card": self.model_card,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SystemResult:
        return cls(
            name=raw["name"],
            overall=MetricResult(**raw["overall"]),
            arcs={a: ArcResult.from_dict(r) for a, r in (raw.get("arcs") or {}).items()},
            operational=raw.get("operational", {}),
            compute=raw.get("compute", {}),
            model_card=raw.get("model_card", {}),
        )


@dataclass(frozen=True)
class ArcFold:
    """One drift family held out, both systems measured on it, split by arc, with the verdict."""

    held_out_vector: str
    outcome: str
    promotion: Promotion
    reason: str = ""
    systems: dict[str, SystemResult] = field(default_factory=dict)
    floor: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    guards: dict[str, Any] = field(default_factory=dict)
    ramps: dict[str, float] = field(default_factory=dict)
    separability: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    history: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown fold outcome {self.outcome!r}; expected one of {OUTCOMES}")
        if self.outcome != MEASURED and not self.reason:
            raise ValueError(
                f"{self.held_out_vector}: a fold that is not measured has to say why — a "
                f"{self.outcome} fold with no reason is indistinguishable from a forgotten one"
            )

    @classmethod
    def skipped(cls, held_out_vector: str, reason: str, **kwargs) -> ArcFold:
        return cls(
            held_out_vector=held_out_vector,
            outcome=SKIPPED,
            reason=reason,
            promotion=Promotion(promoted=False, reason=reason),
            **kwargs,
        )

    def summary(self) -> str:
        """One line, in the form the run prints it."""
        if self.outcome != MEASURED:
            return f"{self.held_out_vector}: {self.outcome} — {self.reason}"
        parts = []
        for name, system in self.systems.items():
            arc = system.arcs.get(GRADUAL)
            m = arc.any_metrics if arc else None
            parts.append(f"{name} {m.pr_auc:.3f}" if m else f"{name} —")
        verdict = "PROMOTED" if self.promotion.promoted else "not promoted"
        return f"{self.held_out_vector}: gradual PR-AUC {' vs '.join(parts)} — {verdict}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_vector": self.held_out_vector,
            "outcome": self.outcome,
            "reason": self.reason,
            "promotion": self.promotion.to_dict(),
            "ramps": self.ramps,
            "systems": {n: s.to_dict() for n, s in self.systems.items()},
            "floor": self.floor,
            "counts": self.counts,
            "guards": self.guards,
            "separability": self.separability,
            "provenance": self.provenance,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ArcFold:
        return cls(
            held_out_vector=raw["held_out_vector"],
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
            promotion=Promotion.from_dict(raw["promotion"]),
            systems={n: SystemResult.from_dict(s) for n, s in (raw.get("systems") or {}).items()},
            floor=raw.get("floor", {}),
            counts=raw.get("counts", {}),
            guards=raw.get("guards", {}),
            ramps=raw.get("ramps", {}),
            separability=raw.get("separability"),
            provenance=raw.get("provenance"),
            history=raw.get("history"),
        )


@dataclass(frozen=True)
class SequenceReport:
    """One anchor's whole comparison, with the config and the seed that produced it.

    Same division the rest of the project keeps: the config holds the inputs, the artefact holds
    the decision. `make sequence` regenerates it; nothing hand-edits it.
    """

    dataset: str
    seed: int
    config: dict[str, Any]
    operating_point: dict[str, Any]
    folds: list[ArcFold]
    split: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    version: int = SEQUENCE_ARTEFACT_VERSION

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError(
                f"{self.dataset}: a sequence-model report with no folds is not a result — even a "
                "run where every family was skipped has to name them and say why"
            )

    @property
    def promoted(self) -> bool:
        """Whether any fold on this anchor earned the model a place in a reported table."""
        return any(f.promotion.promoted for f in self.folds)

    def fold(self, vector_id: str) -> ArcFold | None:
        return next((f for f in self.folds if f.held_out_vector == vector_id), None)

    def summary(self) -> str:
        won = sum(1 for f in self.folds if f.promotion.promoted)
        return f"{self.dataset}: {won}/{len(self.folds)} drift families promote the sequence model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset": self.dataset,
            "seed": self.seed,
            "config": self.config,
            "operating_point": self.operating_point,
            "split": self.split,
            "data": self.data,
            "promoted": self.promoted,
            "folds": [f.to_dict() for f in self.folds],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SequenceReport:
        if int(raw.get("version", 0)) != SEQUENCE_ARTEFACT_VERSION:
            raise ValueError(
                f"sequence artefact version {raw.get('version')} != {SEQUENCE_ARTEFACT_VERSION}; "
                "rebuild it with scripts/build_sequence.py rather than reading it as-is"
            )
        return cls(
            dataset=raw["dataset"],
            seed=int(raw["seed"]),
            config=raw["config"],
            operating_point=raw["operating_point"],
            folds=[ArcFold.from_dict(f) for f in raw["folds"]],
            split=raw.get("split", {}),
            data=raw.get("data", {}),
            meta=raw.get("meta", {}),
        )

    def save(self, directory: str | Path = DEFAULT_SEQUENCE_DIR) -> Path:
        path = Path(directory) / f"{self.dataset}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(with_provenance(self.to_dict()), indent=2, default=str) + "\n")
        return path

    @classmethod
    def load(cls, dataset: str, directory: str | Path = DEFAULT_SEQUENCE_DIR) -> SequenceReport:
        path = Path(directory) / f"{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no sequence-model report for {dataset!r} at {path} — run `make sequence` once "
                "and commit the result"
            )
        return cls.from_dict(json.loads(path.read_text()))


def load_all(directory: str | Path = DEFAULT_SEQUENCE_DIR) -> dict[str, SequenceReport]:
    """Every committed report on disk, keyed by dataset."""
    directory = Path(directory)
    if not directory.exists():
        return {}
    return {
        path.stem: SequenceReport.from_dict(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
    }


def assert_config_matches_promotion(
    enabled: bool, reports: dict[str, SequenceReport] | None = None
) -> None:
    """`defend.sequence.enabled` may not be true while the committed evidence refuses it.

    The ticket's fourth criterion is a claim about what ships, and a claim about what ships that
    lives only in a doc is one config edit away from being false. With no committed reports at
    all this passes for `enabled: false` and refuses `enabled: true`: turning on a layer nothing
    has measured is the case this exists for.
    """
    if not enabled:
        return
    reports = load_all() if reports is None else reports
    if not reports:
        raise AssertionError(
            "defend.sequence.enabled is true and there is no committed sequence-model report — "
            "run `make sequence` and let the gate decide, or set it back to false"
        )
    refused = {name: r for name, r in reports.items() if not r.promoted}
    if refused:
        raise AssertionError(
            "defend.sequence.enabled is true but the committed evidence does not support it on "
            + ", ".join(sorted(refused))
            + " — "
            + "; ".join(
                f"{name}: {f.promotion.reason}"
                for name, r in sorted(refused.items())
                for f in r.folds[:1]
            )
        )


__all__ = [
    "ARCS",
    "DRIFT_ARC_FAMILIES",
    "GRADUAL",
    "MATERIAL_GAP",
    "SUDDEN",
    "ArcFold",
    "ArcResult",
    "Promotion",
    "SequenceReport",
    "SystemResult",
    "arc_breakdown",
    "arc_params",
    "arc_ramp",
    "assert_config_matches_promotion",
    "decide_promotion",
    "history_audit",
    "load_all",
    "shared_threshold",
    "tag_arcs",
    "window_lengths",
]
