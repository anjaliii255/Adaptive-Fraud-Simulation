"""The mule-family comparison, and the gate a temporal GNN has to pass to be reported.

Ticket 18. A GNN earns its seat or the hand-rolled graph features keep it, and either way the
comparison is published. This module holds the four things that decision needs and
`scripts/build_gnn.py` orchestrates:

**The families.** The graph engine's two: S1, fan-in to a collector then layering hops, and C3,
a single relay hop with no dwell. They are the families whose whole tell is topology — every
individual payment in either one looks ordinary, and only who-paid-whom is wrong. The drift and
velocity families are deliberately absent: a model that reads the graph has nothing to say about
an account whose amounts changed, and asking it anyway would produce a number about the fold
rather than about the model.

**The precondition.** A message-passing model has nothing to pass on an anchor whose accounts
appear once, and worse than nothing on a fold where the injected episodes form their own island.
If a row's neighbourhood is made of other injected rows, then "the neighbourhood looks synthetic"
and "the neighbourhood looks fraudulent" are the same statement, and the GNN reads provenance
before it reads behaviour. `neighbourhood_audit` measures exactly that, in the model's own units
— the snapshot the model is scored on — and a fold it flags is withheld rather than reported.
This is the graph twin of the sequence model's history audit, and it exists for the same reason:
the leave-one-attack-out provenance probe is built out of the *supervised* feature space and
cannot see a leak that only a graph model would read.

**The variance.** A single-seed GNN result is not a result — the ticket says so in as many
words, and it is the criterion this whole comparison is shaped around. Every fold runs at several
seeds, each seed regenerating its own pool and refitting every system, and the lift is reported
as a paired per-seed difference with its spread and a sign test. The machinery is
`three_system.Spread` and `three_system.Comparison` rather than a second copy of them, so ticket
16's hero table and ticket 18's experiment are read at exactly the same bar.

**The gate.** `decide_promotion` is the ticket's rule written down: the GNN enters a reported
table only if it beats the hand-rolled baseline on a fold that can carry a claim, at enough seeds
to have a spread, by a margin larger than that spread and larger than the difference this project
treats as a difference — and only if it also clears the amount floor. Anything else and the
stated fallback ships, which is `TemporalGNNDetector.fallback()`: LightGBM over the hand-rolled
feature table.

**Which champion.** The ticket names "graph-features + LightGBM". Two columns are reported —
`lgbm` over the whole hand-rolled table, which is what actually ships, and `graph_lgbm` over the
graph blocks alone (`features.graph_feature_names`) — and the gate is decided on the *former*.
A challenger promoted over a deliberately narrowed champion is a number that does not survive
contact with the deployed system, and this is the one place in the repo where narrowing the
baseline would be the easiest way to manufacture a lift. The narrower column is still published,
because "it loses to graph features" and "it loses to the velocity block sitting next to them"
are different findings.

`assert_config_matches_promotion` closes the loop back onto config: `defend.gnn.enabled` cannot
be true while a committed artefact says the gate refused it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
from afl.evaluation.three_system import Comparison, Spread

# ── the families ────────────────────────────────────────────────────────────────
#: The two families the graph engine puts on the table. S1 is fan-in to a collector then layering
#: hops; C3 is one relay hop with no dwell. The other seven ride the velocity or drift engines,
#: where the tell is pacing or trajectory rather than topology.
MULE_FAMILIES = ("S1", "C3")

#: The systems, in the order the tables print them.
GNN = "gnn"
LGBM = "lgbm"
GRAPH_LGBM = "graph_lgbm"
FLOOR = "amount_only"
SYSTEMS = (GNN, LGBM, GRAPH_LGBM, FLOOR)

#: A PR-AUC difference this project treats as a difference. Same bar `docs/anomaly.md` and
#: `docs/sequence.md` read their tables at: below it, two systems are reported as level.
MATERIAL_GAP = 0.01

#: Below this share of injected rows able to see any earlier edge of their own family, a fold is
#: not measuring a temporal graph model at all — it is asking one to detect a ring it structurally
#: cannot watch form. Four positives in five scored on their endpoints' unrelated history is not a
#: graph experiment, whichever way the number falls.
MOTIF_VISIBILITY_FLOOR = 0.2

#: Seeds below which a lift has no spread to be outside of. Two is the arithmetic minimum for a
#: standard deviation; three is what the three-system table runs and what the config asks for.
MIN_SEEDS = 2

GNN_ARTEFACT_VERSION = 1

DEFAULT_GNN_DIR = Path(os.getenv("AFL_GNN_DIR", "artifacts/gnn"))


# ── the precondition ────────────────────────────────────────────────────────────
def _share_for(
    nodes: np.ndarray, touched: np.ndarray, synthetic: np.ndarray, codes: np.ndarray
) -> np.ndarray:
    """Per account code: what share of the edges touching it inside the window were injected.

    A code the window has never seen scores 0.0 rather than raising — an account with no
    neighbourhood has no synthetic neighbourhood either, and that is the honest reading of a
    cold node rather than a missing value to impute.
    """
    codes = np.asarray(codes, dtype=np.int64)
    if not nodes.size:
        return np.zeros(codes.size, dtype=float)
    idx = np.clip(np.searchsorted(nodes, codes), 0, nodes.size - 1)
    hit = nodes[idx] == codes
    share = np.where(touched[idx] > 0, synthetic[idx] / np.maximum(touched[idx], 1), 0.0)
    return np.where(hit, share, 0.0)


def neighbour_provenance(
    rows: list[Transaction], injected: np.ndarray, window_hours: float, stride_hours: float
) -> np.ndarray:
    """Per row: the share of its endpoints' in-window neighbours that are injected rows.

    Computed on the *same snapshots the detector is scored on* — edges strictly before the stride
    the row lands in, none older than the window — so this is not a proxy for what the GNN reads,
    it is the thing the GNN reads with the labels put back on.

    A row whose beneficiary has only ever been paid by other injected episodes sits in a
    synthetic island. Message passing over that island returns "synthetic", and a fold where the
    injected rows can be found that way cannot tell detection from provenance for a graph model,
    however clean the per-row feature space looks.
    """
    from afl.defend.models.gnn import RAW_DST, RAW_SRC, RAW_TS, raw_rows, stride_of

    if not rows:
        return np.zeros(0, dtype=float)
    raw = raw_rows(rows)
    order = np.argsort(raw[:, RAW_TS], kind="stable")
    ts_sorted = raw[order, RAW_TS]
    origin = float(raw[:, RAW_TS].min())
    window_s, stride_s = window_hours * 3_600.0, stride_hours * 3_600.0
    strides = stride_of(raw[:, RAW_TS], origin, stride_s)

    out = np.zeros(len(rows), dtype=float)
    for b in np.unique(strides):
        end_ts = origin + float(b) * stride_s
        lo = int(np.searchsorted(ts_sorted, end_ts - window_s, side="left"))
        hi = int(np.searchsorted(ts_sorted, end_ts, side="left"))
        edges = order[lo:hi]
        here = np.flatnonzero(strides == b)
        if not edges.size:
            continue  # an empty neighbourhood carries no provenance either way
        ends = np.concatenate([raw[edges, RAW_SRC], raw[edges, RAW_DST]]).astype(np.int64)
        flags = np.tile(injected[edges].astype(float), 2)
        nodes, inverse = np.unique(ends, return_inverse=True)
        touched = np.bincount(inverse, minlength=nodes.size)
        synthetic = np.bincount(inverse, weights=flags, minlength=nodes.size)
        out[here] = np.maximum(
            _share_for(nodes, touched, synthetic, raw[here, RAW_SRC]),
            _share_for(nodes, touched, synthetic, raw[here, RAW_DST]),
        )
    return out


def motif_visibility(
    real: list[Transaction],
    injected: list[Transaction],
    window_hours: float,
    stride_hours: float,
) -> tuple[float, float]:
    """(share seeing an earlier edge of their own family, share seeing any earlier edge).

    Per injected row, over the snapshot it is actually scored on. "Sees" means one of its two
    endpoints appears on an edge that is inside the window and strictly before the row's stride —
    which is the whole of what message passing has to work with for that row.
    """
    from afl.defend.models.gnn import RAW_DST, RAW_SRC, RAW_TS, raw_rows, stride_of

    rows = list(real) + list(injected)
    raw = raw_rows(rows)
    flags = np.array([0] * len(real) + [1] * len(injected), dtype=bool)
    order = np.argsort(raw[:, RAW_TS], kind="stable")
    ts_sorted = raw[order, RAW_TS]
    origin = float(raw[:, RAW_TS].min())
    window_s, stride_s = window_hours * 3_600.0, stride_hours * 3_600.0
    strides = stride_of(raw[:, RAW_TS], origin, stride_s)

    saw_family = np.zeros(len(rows), dtype=bool)
    saw_anything = np.zeros(len(rows), dtype=bool)
    for b in np.unique(strides[flags]):
        end_ts = origin + float(b) * stride_s
        lo = int(np.searchsorted(ts_sorted, end_ts - window_s, side="left"))
        hi = int(np.searchsorted(ts_sorted, end_ts, side="left"))
        edges = order[lo:hi]
        here = np.flatnonzero((strides == b) & flags)
        if not edges.size or not here.size:
            continue
        family = edges[flags[edges]]
        for target, source in ((saw_family, family), (saw_anything, edges)):
            if not source.size:
                continue
            nodes = np.unique(
                np.concatenate([raw[source, RAW_SRC], raw[source, RAW_DST]]).astype(np.int64)
            )
            target[here] = np.isin(
                raw[here, RAW_SRC].astype(np.int64), nodes
            ) | np.isin(raw[here, RAW_DST].astype(np.int64), nodes)
    n = max(int(flags.sum()), 1)
    return float(saw_family[flags].sum()) / n, float(saw_anything[flags].sum()) / n


def resolution_audit(
    real: list[Transaction],
    injected: list[Transaction],
    window_hours: float,
    stride_hours: float,
    granularity_s: int,
) -> dict[str, Any]:
    """Can a causal temporal graph watch this family's motif form on this anchor's clock?

    The precondition nobody expects to fail, and the one that decides this whole ticket on one of
    the two anchors. A temporal GNN may only read edges strictly *earlier* than the payment it is
    scoring. A mule ring whose eight fan-in payments and layering hop all carry the **same
    timestamp** therefore never exists in any snapshot that scores one of its own rows: the model
    is handed the endpoints' unrelated history and asked about a shape it cannot see.

    That is not a hypothetical. AMLSim's rows are whole days, so `Simulator.generate` quantises
    every injected fraud row to midnight to keep it commensurable with the anchor — and a ring
    that took twenty minutes becomes a ring that took no time at all. PaySim's clock is hourly and
    its episodes are spread wide enough that the motif survives.

    `granularity_s` comes from the anchor's own `AnchorEnvelope`, not re-derived here: the rule
    that decides how synthetic timestamps are rounded and the rule that reports the consequence
    have to be the same rule.
    """
    if not real or not injected:
        return {"checked": False, "reason": "one side of the comparison is empty"}
    family, anything = motif_visibility(real, injected, window_hours, stride_hours)
    return {
        "checked": True,
        "window_hours": window_hours,
        "stride_hours": stride_hours,
        "anchor_time_granularity_s": int(granularity_s),
        "injected_share_seeing_an_earlier_family_edge": round(family, 6),
        "injected_share_seeing_any_earlier_edge": round(anything, 6),
        "floor": MOTIF_VISIBILITY_FLOOR,
        "blind": bool(family < MOTIF_VISIBILITY_FLOOR),
        "why": (
            "a causal temporal graph may only read edges strictly earlier than the payment it "
            "scores, so a ring whose payments all carry one timestamp is a ring no snapshot ever "
            "contains — the model is asked about a shape it structurally cannot see, and the "
            "answer says nothing about message passing either way"
        ),
    }


def neighbourhood_audit(
    real: list[Transaction],
    injected: list[Transaction],
    window_hours: float,
    stride_hours: float,
) -> dict[str, Any]:
    """Does the shape of a row's neighbourhood alone sort the injected rows from the anchor's own?

    The graph model's version of the provenance probe, asked about the one thing this model reads
    that the per-row table does not. Measured over the two populations *together*, because that
    is the pool the fold scores.

    The verdict uses the same rule the leave-one-attack-out harness applies to its own probe
    (`is_provenance_bound`), so a fold refused here is refused for the same reason and at the same
    bar as a fold refused there.

    Note what this deliberately does **not** check: fan-in degree. A collector with fourteen
    payers inside an hour is the S1 tell, and a fold that was refused for containing it would be
    a fold refused for containing the fraud. Only the *provenance* of the neighbours is the
    artefact, and only the artefact decides the verdict.
    """
    if not real or not injected:
        return {"checked": False, "reason": "one side of the comparison is empty"}
    rows = list(real) + list(injected)
    flags = np.array([0] * len(real) + [1] * len(injected), dtype=int)
    share = neighbour_provenance(rows, flags, window_hours, stride_hours)
    base_rate = float(flags.mean())
    score = protocol.pr_auc(flags, share)
    real_share, inj_share = share[: len(real)], share[len(real) :]
    return {
        "checked": True,
        "window_hours": window_hours,
        "stride_hours": stride_hours,
        "pr_auc": round(float(score), 6),
        "base_rate": round(base_rate, 8),
        "separable": bool(is_provenance_bound(score, base_rate)),
        "anchor_mean_synthetic_neighbour_share": round(float(real_share.mean()), 6),
        "injected_mean_synthetic_neighbour_share": round(float(inj_share.mean()), 6),
        "injected_share_in_a_pure_island": round(float((inj_share >= 0.999).mean()), 6),
        "why": (
            "a message-passing model reads its neighbours' company whether or not it is asked "
            "to; where an injected episode's neighbours are other injected episodes, 'this "
            "neighbourhood looks synthetic' and 'this neighbourhood looks fraudulent' are the "
            "same statement"
        ),
    }


# ── one system, one seed ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SystemResult:
    """One system on one seed of one fold: an outcome, and numbers only when quotable.

    Same discipline as `FoldResult`: `metrics` is `None` unless the seed's fold can carry a
    claim, and the numbers that exist anyway live under `withheld_metrics` so quoting one is a
    decision somebody made rather than the default a reader falls into.
    """

    name: str
    outcome: str
    reason: str = ""
    metrics: MetricResult | None = None
    withheld_metrics: MetricResult | None = None
    operational: dict[str, float] = field(default_factory=dict)
    compute: dict[str, Any] = field(default_factory=dict)
    model_card: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}; expected one of {OUTCOMES}")
        if self.outcome != MEASURED and self.metrics is not None:
            raise ValueError(
                f"{self.name}: a {self.outcome} result carries numbers under `withheld_metrics`, "
                "never under `metrics` — that field is what a reader quotes"
            )
        if self.outcome != MEASURED and not self.reason:
            raise ValueError(f"{self.name}: a result that is not measured has to say why")

    @property
    def any_metrics(self) -> MetricResult | None:
        return self.metrics or self.withheld_metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outcome": self.outcome,
            "reason": self.reason,
            "metrics": self.metrics.model_dump() if self.metrics else None,
            "withheld_metrics": self.withheld_metrics.model_dump()
            if self.withheld_metrics
            else None,
            "operational": self.operational,
            "compute": self.compute,
            "model_card": self.model_card,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SystemResult:
        return cls(
            name=raw["name"],
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
            metrics=MetricResult(**raw["metrics"]) if raw.get("metrics") else None,
            withheld_metrics=MetricResult(**raw["withheld_metrics"])
            if raw.get("withheld_metrics")
            else None,
            operational=raw.get("operational", {}),
            compute=raw.get("compute", {}),
            model_card=raw.get("model_card", {}),
        )


@dataclass(frozen=True)
class SeedRun:
    """One seed of one fold: its own pool, its own carve, its own audits, every system on it.

    The seed moves the generated pool as well as the model init, which is the honest version of
    "does this replicate": a GNN whose lift survives only one draw of the attacker's episodes has
    not shown anything about graphs.
    """

    seed: int
    outcome: str
    reason: str = ""
    systems: dict[str, SystemResult] = field(default_factory=dict)
    floor: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    guards: dict[str, Any] = field(default_factory=dict)
    separability: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    neighbourhood: dict[str, Any] | None = None
    resolution: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}; expected one of {OUTCOMES}")
        if self.outcome != MEASURED and not self.reason:
            raise ValueError(
                f"seed {self.seed}: a run that is not measured has to say why — one with no "
                "reason is indistinguishable from a forgotten one"
            )

    def metric(self, system: str, metric: str = "pr_auc") -> float | None:
        got = self.systems.get(system)
        m = got.any_metrics if got else None
        return round(float(getattr(m, metric)), 6) if m else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "outcome": self.outcome,
            "reason": self.reason,
            "systems": {n: s.to_dict() for n, s in self.systems.items()},
            "floor": self.floor,
            "counts": self.counts,
            "guards": self.guards,
            "separability": self.separability,
            "provenance": self.provenance,
            "neighbourhood": self.neighbourhood,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SeedRun:
        return cls(
            seed=int(raw["seed"]),
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
            systems={n: SystemResult.from_dict(s) for n, s in (raw.get("systems") or {}).items()},
            floor=raw.get("floor", {}),
            counts=raw.get("counts", {}),
            guards=raw.get("guards", {}),
            separability=raw.get("separability"),
            provenance=raw.get("provenance"),
            neighbourhood=raw.get("neighbourhood"),
            resolution=raw.get("resolution"),
        )


# ── across seeds ────────────────────────────────────────────────────────────────
def spread_across_seeds(
    seeds: list[SeedRun], system: str, column: str, metric: str = "pr_auc"
) -> Spread:
    """One system's number across every seed that ran it, with the spread that qualifies it.

    Withheld numbers are aggregated too — they are the evidence for the verdict — but the
    aggregate inherits the withholding: a mean of two quotable seeds and one withheld one is not
    a quotable number. Same rule `three_system.spread_of` applies to the hero table.
    """
    values, outcomes, reasons = [], [], []
    for run in seeds:
        got = run.systems.get(system)
        m = got.any_metrics if got else None
        if m is None:
            outcomes.append(SKIPPED)
            reasons.append(run.reason or f"{system} did not run on seed {run.seed}")
            continue
        values.append(round(float(getattr(m, metric)), 6))
        outcomes.append(got.outcome)
        reasons.append(got.reason or run.reason)

    if not outcomes:
        outcome, reason = SKIPPED, "this system did not run on any seed"
    elif all(o == MEASURED for o in outcomes):
        outcome, reason = MEASURED, ""
    elif any(o == WITHHELD for o in outcomes):
        outcome = WITHHELD
        reason = next(
            (r for o, r in zip(outcomes, reasons, strict=False) if r and o != MEASURED), ""
        )
    else:
        outcome, reason = SKIPPED, next((r for r in reasons if r), "")
    return Spread(
        system=system, column=column, metric=metric, values=values, outcome=outcome, reason=reason
    )


def compare_across_seeds(
    seeds: list[SeedRun],
    column: str,
    challenger: str = GNN,
    incumbent: str = LGBM,
    metric: str = "pr_auc",
) -> Comparison | None:
    """The paired difference the claim stands or falls on. Paired by seed, never pooled.

    Pairing matters more here than in the hero table: both systems see the same generated pool,
    the same carve and the same calibration on a given seed, so the difference between them is
    the one thing the seed does not also move. Two pooled means would put the seed's whole
    variance into the gap and turn a draw into a result.
    """
    per_seed, outcomes, reasons = [], [], []
    for run in seeds:
        a, b = run.systems.get(incumbent), run.systems.get(challenger)
        if a is None or b is None:
            continue
        ma, mb = a.any_metrics, b.any_metrics
        if ma is None or mb is None:
            continue
        va, vb = float(getattr(ma, metric)), float(getattr(mb, metric))
        per_seed.append(
            {
                "seed": run.seed,
                "incumbent": round(va, 6),
                "challenger": round(vb, 6),
                "delta": round(vb - va, 6),
            }
        )
        for cell in (a, b):
            outcomes.append(cell.outcome)
            reasons.append(cell.reason or run.reason)
    if not per_seed:
        return None
    if all(o == MEASURED for o in outcomes):
        outcome, reason = MEASURED, ""
    else:
        outcome = WITHHELD
        reason = next(
            (r for o, r in zip(outcomes, reasons, strict=False) if r and o != MEASURED), ""
        )
    return Comparison(
        challenger=challenger,
        incumbent=incumbent,
        column=column,
        metric=metric,
        per_seed=per_seed,
        outcome=outcome,
        reason=reason,
    )


# ── the gate ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Promotion:
    """Whether the temporal GNN may appear in a reported table, and why.

    A refusal is a result. `reason` is written to be quotable as-is, because the thing this
    ticket is most likely to lose is not the experiment — it is the record that the experiment
    was run and came back negative, and the fallback that shipped instead.
    """

    promoted: bool
    reason: str
    shipped: str
    champion: str = LGBM
    metric: str = "pr_auc"
    n_seeds: int = 0
    mean_delta: float | None = None
    sd_delta: float | None = None
    wins: int | None = None
    p_value: float | None = None
    challenger: float | None = None
    incumbent: float | None = None
    floor: float | None = None
    material_gap: float = MATERIAL_GAP
    min_seeds: int = MIN_SEEDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "shipped": self.shipped,
            "champion": self.champion,
            "metric": self.metric,
            "n_seeds": self.n_seeds,
            "mean_delta": self.mean_delta,
            "sd_delta": self.sd_delta,
            "wins": self.wins,
            "p_value": self.p_value,
            "challenger": self.challenger,
            "incumbent": self.incumbent,
            "floor": self.floor,
            "material_gap": self.material_gap,
            "min_seeds": self.min_seeds,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Promotion:
        return cls(
            promoted=bool(raw["promoted"]),
            reason=raw["reason"],
            shipped=raw.get("shipped", ""),
            champion=raw.get("champion", LGBM),
            metric=raw.get("metric", "pr_auc"),
            n_seeds=int(raw.get("n_seeds", 0)),
            mean_delta=raw.get("mean_delta"),
            sd_delta=raw.get("sd_delta"),
            wins=raw.get("wins"),
            p_value=raw.get("p_value"),
            challenger=raw.get("challenger"),
            incumbent=raw.get("incumbent"),
            floor=raw.get("floor"),
            material_gap=float(raw.get("material_gap", MATERIAL_GAP)),
            min_seeds=int(raw.get("min_seeds", MIN_SEEDS)),
        )


#: What ships when the gate says no. Named here rather than in prose, because ticket 18's fifth
#: criterion is a claim about what is deployed and a claim about deployment that lives only in a
#: document is one config edit away from being false.
FALLBACK = "hand-rolled graph features + LightGBM (`TemporalGNNDetector.fallback()`)"


def decide_promotion(
    comparison: Comparison | None,
    challenger: Spread | None = None,
    floor: Spread | None = None,
    material_gap: float = MATERIAL_GAP,
    min_seeds: int = MIN_SEEDS,
    blocked: str = "",
) -> Promotion:
    """The ticket's rule, in the order a reader can check it.

    `blocked` carries a reason the whole fold cannot carry a claim — the family guard failed, the
    injected episodes are separable from the anchor, or the neighbourhood audit says the graph is
    an island. It is checked first because a comparison inside an invalid fold is not a
    comparison, however it comes out.
    """
    common: dict[str, Any] = {
        "shipped": FALLBACK,
        "material_gap": material_gap,
        "min_seeds": min_seeds,
        "champion": comparison.incumbent if comparison else LGBM,
        "metric": comparison.metric if comparison else "pr_auc",
    }
    if blocked:
        return Promotion(promoted=False, reason=blocked, **common)
    if comparison is None or not comparison.per_seed:
        return Promotion(
            promoted=False,
            reason=(
                "the comparison never ran on a single seed, so there is nothing to promote and "
                f"{FALLBACK} ships by default rather than by decision"
            ),
            **common,
        )

    common = {
        **common,
        "n_seeds": comparison.n,
        "mean_delta": comparison.mean_delta,
        "sd_delta": comparison.sd_delta,
        "wins": comparison.wins,
        "p_value": comparison.p_value,
        "challenger": round(float(np.mean([d["challenger"] for d in comparison.per_seed])), 6),
        "incumbent": round(float(np.mean([d["incumbent"] for d in comparison.per_seed])), 6),
        "floor": floor.mean if floor and floor.values else None,
    }
    if comparison.outcome != MEASURED:
        return Promotion(
            promoted=False,
            reason=(
                f"this fold cannot carry a claim — {comparison.reason}. The comparison is "
                f"published under `seeds`; it is not evidence for a promotion, and {FALLBACK} "
                "ships"
            ),
            **common,
        )
    if comparison.n < min_seeds:
        return Promotion(
            promoted=False,
            reason=(
                f"the comparison ran at {comparison.n} seed(s) against a floor of {min_seeds} — a "
                "single-seed GNN result has no spread to be outside of, so nothing follows from "
                f"its sign. {FALLBACK} ships"
            ),
            **common,
        )

    mine = common["challenger"]
    theirs = common["incumbent"]
    bar = common["floor"]
    if bar is not None and mine <= bar:
        return Promotion(
            promoted=False,
            reason=(
                f"the GNN reaches {comparison.metric} {mine:.3f} against {bar:.3f} for sorting by "
                f"amount alone — a detector a sort beats has not found anything, and {FALLBACK} "
                "ships"
            ),
            **common,
        )
    if comparison.mean_delta < material_gap:
        verdict = "loses to" if comparison.mean_delta < 0 else "is level with"
        return Promotion(
            promoted=False,
            reason=(
                f"the temporal GNN {verdict} the hand-rolled baseline: {comparison.metric} "
                f"{mine:.3f} against {theirs:.3f}, a margin of {comparison.mean_delta:+.3f} "
                f"± {comparison.sd_delta:.3f} over {comparison.n} seeds against a bar of "
                f"{material_gap:.2f}. It does not enter the reported table, {FALLBACK} ships, "
                "and this line is the result"
            ),
            **common,
        )
    if comparison.inside_noise:
        return Promotion(
            promoted=False,
            reason=(
                f"the margin is inside its own seed-to-seed spread: {comparison.mean_delta:+.3f} "
                f"± {comparison.sd_delta:.3f} over {comparison.n} seeds, {comparison.wins}/"
                f"{comparison.n} in that direction, sign-test p = {comparison.p_value:.3f}. A "
                f"difference smaller than the noise it sits in is not a lift, so {FALLBACK} ships"
            ),
            **common,
        )
    return Promotion(
        promoted=True,
        reason=(
            f"the temporal GNN beats the hand-rolled baseline on the same split at the same "
            f"operating point: {comparison.metric} {mine:.3f} against {theirs:.3f}, a margin of "
            f"{comparison.mean_delta:+.3f} ± {comparison.sd_delta:.3f} over {comparison.n} "
            f"seeds, {comparison.wins}/{comparison.n} in that direction, sign-test "
            f"p = {comparison.p_value:.3f}"
        ),
        **{**common, "shipped": "the temporal GNN"},
    )


# ── the artefact ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MuleFold:
    """One mule family held out, every seed measured on it, with the spread and the verdict."""

    held_out_vector: str
    outcome: str
    promotion: Promotion
    reason: str = ""
    seeds: list[SeedRun] = field(default_factory=list)
    spreads: dict[str, Spread] = field(default_factory=dict)
    comparison: Comparison | None = None
    graph_comparison: Comparison | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown fold outcome {self.outcome!r}; expected one of {OUTCOMES}")
        if self.outcome != MEASURED and not self.reason:
            raise ValueError(
                f"{self.held_out_vector}: a fold that is not measured has to say why — a "
                f"{self.outcome} fold with no reason is indistinguishable from a forgotten one"
            )

    @classmethod
    def skipped(cls, held_out_vector: str, reason: str, **kwargs) -> MuleFold:
        return cls(
            held_out_vector=held_out_vector,
            outcome=SKIPPED,
            reason=reason,
            promotion=Promotion(promoted=False, reason=reason, shipped=FALLBACK),
            **kwargs,
        )

    def summary(self) -> str:
        """One line, in the form the run prints it."""
        if self.outcome != MEASURED and not self.seeds:
            return f"{self.held_out_vector}: {self.outcome} — {self.reason}"
        parts = [
            f"{name} {self.spreads[name].text()}" for name in SYSTEMS if name in self.spreads
        ]
        verdict = "PROMOTED" if self.promotion.promoted else "not promoted"
        return f"{self.held_out_vector}: PR-AUC {' vs '.join(parts)} — {verdict}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_vector": self.held_out_vector,
            "outcome": self.outcome,
            "reason": self.reason,
            "promotion": self.promotion.to_dict(),
            "spreads": {n: s.to_dict() for n, s in self.spreads.items()},
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "graph_comparison": self.graph_comparison.to_dict()
            if self.graph_comparison
            else None,
            "seeds": [s.to_dict() for s in self.seeds],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MuleFold:
        return cls(
            held_out_vector=raw["held_out_vector"],
            outcome=raw["outcome"],
            reason=raw.get("reason", ""),
            promotion=Promotion.from_dict(raw["promotion"]),
            seeds=[SeedRun.from_dict(s) for s in raw.get("seeds", [])],
            spreads={n: Spread.from_dict(s) for n, s in (raw.get("spreads") or {}).items()},
            comparison=Comparison.from_dict(raw["comparison"]) if raw.get("comparison") else None,
            graph_comparison=Comparison.from_dict(raw["graph_comparison"])
            if raw.get("graph_comparison")
            else None,
        )


@dataclass(frozen=True)
class GNNReport:
    """One anchor's whole comparison, with the config and the seeds that produced it.

    Same division the rest of the project keeps: the config holds the inputs, the artefact holds
    the decision. `make gnn` regenerates it; nothing hand-edits it.
    """

    dataset: str
    seeds: list[int]
    config: dict[str, Any]
    operating_point: dict[str, Any]
    folds: list[MuleFold]
    split: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    version: int = GNN_ARTEFACT_VERSION

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError(
                f"{self.dataset}: a GNN report with no folds is not a result — even a run where "
                "every family was skipped has to name them and say why"
            )

    @property
    def promoted(self) -> bool:
        """Whether any fold on this anchor earned the GNN a place in a reported table."""
        return any(f.promotion.promoted for f in self.folds)

    @property
    def shipped(self) -> str:
        """What ships on this anchor. The answer ticket 18's fifth criterion asks for."""
        return "the temporal GNN" if self.promoted else FALLBACK

    def fold(self, vector_id: str) -> MuleFold | None:
        return next((f for f in self.folds if f.held_out_vector == vector_id), None)

    def summary(self) -> str:
        won = sum(1 for f in self.folds if f.promotion.promoted)
        return f"{self.dataset}: {won}/{len(self.folds)} mule families promote the temporal GNN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset": self.dataset,
            "seeds": self.seeds,
            "config": self.config,
            "operating_point": self.operating_point,
            "split": self.split,
            "data": self.data,
            "promoted": self.promoted,
            "shipped": self.shipped,
            "folds": [f.to_dict() for f in self.folds],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GNNReport:
        if int(raw.get("version", 0)) != GNN_ARTEFACT_VERSION:
            raise ValueError(
                f"gnn artefact version {raw.get('version')} != {GNN_ARTEFACT_VERSION}; rebuild "
                "it with scripts/build_gnn.py rather than reading it as-is"
            )
        return cls(
            dataset=raw["dataset"],
            seeds=[int(s) for s in raw.get("seeds", [])],
            config=raw["config"],
            operating_point=raw["operating_point"],
            folds=[MuleFold.from_dict(f) for f in raw["folds"]],
            split=raw.get("split", {}),
            data=raw.get("data", {}),
            meta=raw.get("meta", {}),
        )

    def save(self, directory: str | Path = DEFAULT_GNN_DIR) -> Path:
        path = Path(directory) / f"{self.dataset}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        return path

    @classmethod
    def load(cls, dataset: str, directory: str | Path = DEFAULT_GNN_DIR) -> GNNReport:
        path = Path(directory) / f"{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no GNN report for {dataset!r} at {path} — run `make gnn` once and commit the "
                "result"
            )
        return cls.from_dict(json.loads(path.read_text()))


def load_all(directory: str | Path = DEFAULT_GNN_DIR) -> dict[str, GNNReport]:
    """Every committed report on disk, keyed by dataset."""
    directory = Path(directory)
    if not directory.exists():
        return {}
    return {
        path.stem: GNNReport.from_dict(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
    }


def assert_config_matches_promotion(
    enabled: bool, reports: dict[str, GNNReport] | None = None
) -> None:
    """`defend.gnn.enabled` may not be true while the committed evidence refuses it.

    The ticket's fifth criterion — "the documented fallback is what ships if the lift is not
    there" — is a claim about deployment, and a claim about deployment that lives only in a doc
    is one config edit away from being false. With no committed reports at all this passes for
    `enabled: false` and refuses `enabled: true`: turning on a layer nothing has measured is the
    case this exists for.
    """
    if not enabled:
        return
    reports = load_all() if reports is None else reports
    if not reports:
        raise AssertionError(
            "defend.gnn.enabled is true and there is no committed GNN report — run `make gnn` "
            "and let the gate decide, or set it back to false"
        )
    refused = {name: r for name, r in reports.items() if not r.promoted}
    if refused:
        raise AssertionError(
            "defend.gnn.enabled is true but the committed evidence does not support it on "
            + ", ".join(sorted(refused))
            + " — "
            + "; ".join(
                f"{name}: {f.promotion.reason}"
                for name, r in sorted(refused.items())
                for f in r.folds[:1]
            )
        )


def min_meaningful_positives() -> int:
    """Re-exported so a caller does not have to import two evaluation modules to read one bar."""
    return MIN_MEANINGFUL_POSITIVES


__all__ = [
    "FALLBACK",
    "FLOOR",
    "GNN",
    "GNN_ARTEFACT_VERSION",
    "GRAPH_LGBM",
    "LGBM",
    "MATERIAL_GAP",
    "MIN_SEEDS",
    "MOTIF_VISIBILITY_FLOOR",
    "MULE_FAMILIES",
    "SYSTEMS",
    "GNNReport",
    "MuleFold",
    "Promotion",
    "SeedRun",
    "SystemResult",
    "assert_config_matches_promotion",
    "compare_across_seeds",
    "decide_promotion",
    "load_all",
    "min_meaningful_positives",
    "neighbour_provenance",
    "motif_visibility",
    "neighbourhood_audit",
    "resolution_audit",
    "spread_across_seeds",
]
