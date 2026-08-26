"""Does a temporal GNN beat hand-rolled graph features on the mule families? Publish it either way.

    python scripts/build_gnn.py                 # every real anchor, both mule families
    python scripts/build_gnn.py amlsim          # just one
    python scripts/build_gnn.py --families S1   # just one family
    python scripts/build_gnn.py --seeds 1337    # a quick pass — and the gate will refuse it
    python scripts/build_gnn.py --sample 0.02   # a quick pass over less data
    python scripts/build_gnn.py --doc-only      # rewrite the doc from committed artefacts

Ticket 18's deliverable, and like ticket 17's it is a comparison rather than a model. Graph
attention over the account-beneficiary graph is the obvious thing to reach for on a mule family,
and it is also the easiest place in this repo to produce a number that does not replicate: it
reads structure the per-row table sees only as counters, and on a fold where the injected ring is
its own island that structure spells "synthetic".

So the comparison is set up to be losable.

**One split, one operating point, four systems.** `lgbm` is LightGBM over the whole hand-rolled
feature table — what actually ships, and the champion the gate is decided on. `graph_lgbm` is the
same model over the graph blocks alone, which is the narrower baseline the ticket names by
"graph-features + LightGBM"; it is reported so a reader can tell "loses to graph features" apart
from "loses to the velocity block sitting next to them", and it is deliberately *not* the gate,
because a challenger promoted over a narrowed champion is a number that does not survive contact
with the deployed system. `gnn` is the challenger. `amount_only` is under all three: direction
chosen on the training window, no model, no training. Two results in this repo were walked back
for want of that column.

**Two families, several seeds.** S1 and C3 are the graph engine's families. Each is held out of
training in turn, so the number is generalisation to an unseen family rather than memorisation of
a seen one — and each fold is run at every seed in `defend.gnn.mule.seeds`, with the pool
regenerated per seed. The lift is then a paired per-seed difference with its spread and a sign
test, because the ticket's own words are that a single-seed GNN result is not a result.

**Four ways a fold can fail to mean anything, all checked.** Whether one contract field separates
the injected family from the anchor (`envelope.audit`); whether a classifier can sort injected
rows from real ones in the detector's feature space (`build_loao.provenance_probe`); whether the
family is a `template` vector whose defining tell is not modelled yet (`template_gate`); and —
the one specific to this model — whether a row's *neighbourhood provenance* does it by itself
(`mule_graph.neighbourhood_audit`). A fold blocked at any seed is blocked for the family: a fold
that measures provenance at one seed is not a fold that measures detection at the other two.

Everything lands in `artifacts/gnn/<anchor>.json` and in `docs/gnn.md`, which is generated from
those files and never hand-typed. The gate is `afl.evaluation.mule_graph.decide_promotion`, and
`assert_config_matches_promotion` keeps `config/defend/gnn.yaml` answerable to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

# One definition of the fold, the pool and the probes. This script differs from `build_loao.py`
# in exactly two places — several seeds per fold, and three detectors on each — and every other
# piece is imported rather than re-derived, because a second copy of a split rule drifts the
# first time either is edited.
from build_loao import fold_config, provenance_probe, template_gate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from run_experiment import build_pool, build_simulator, detector_params  # noqa: E402

from afl.attack import envelope as envelope_lib  # noqa: E402
from afl.attack.envelope import AnchorEnvelope  # noqa: E402
from afl.attack.templates import registry  # noqa: E402
from afl.contract.schema import AttackBatch, AttackParams, Transaction  # noqa: E402
from afl.data import loaders  # noqa: E402
from afl.data.splits import CommittedSplit, committed_split_for  # noqa: E402
from afl.defend import baseline, explain  # noqa: E402
from afl.defend.decision import (  # noqa: E402
    action_mix,
    assert_one_operating_point,
    cost_model_for,
    policy_from_config,
)
from afl.defend.features import (  # noqa: E402
    FeatureBuilder,
    GraphFeatureBuilder,
    graph_feature_names,
)
from afl.defend.models.gnn import TemporalGNNDetector, available  # noqa: E402
from afl.defend.models.lgbm import LGBMDetector, model_card_of  # noqa: E402
from afl.evaluation import leave_one_attack_out as loao  # noqa: E402
from afl.evaluation import mule_graph, protocol  # noqa: E402
from afl.evaluation.mule_graph import FLOOR, GNN, GRAPH_LGBM, LGBM, SYSTEMS  # noqa: E402
from afl.utils.seed import set_all_seeds  # noqa: E402

log = logging.getLogger("build_gnn")

DATA_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
ANOMALY_CONFIG = Path("config/defend/anomaly.yaml")
GNN_CONFIG = Path("config/defend/gnn.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
ENGINES_CONFIG = Path("config/attack/engines.yaml")
COSTS_CONFIG = Path("config/costs/default.yaml")
ARTIFACT_DIR = mule_graph.DEFAULT_GNN_DIR
DOC_PATH = Path("docs/gnn.md")


def anchors(selected: list[str]) -> list[dict]:
    """Every data config that names a loader — the real anchors, not the synthetic default."""
    out = []
    for path in sorted(DATA_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("loader") and (not selected or cfg["name"] in selected):
            out.append(cfg)
    return out


def stats(rows: list[Transaction]) -> dict:
    fraud = sum(1 for t in rows if t.is_fraud)
    return {
        "rows": len(rows),
        "fraud": fraud,
        "base_rate": round(fraud / len(rows), 8) if rows else 0.0,
        "first_ts": min((t.ts for t in rows), default=None),
        "last_ts": max((t.ts for t in rows), default=None),
    }


def as_batch(rows: list[Transaction], run_id: str) -> AttackBatch:
    return AttackBatch(
        run_id=run_id,
        params=AttackParams(vector_id="eval", engine="none"),
        transactions=list(rows),
        seed=0,
    )


def compose(cfg, sup, uns, gnn_cfg, eval_cfg, engines, costs, seed: int):
    """The same config tree `config/config.yaml` composes, built here from the same files."""
    return OmegaConf.create(
        {
            "seed": seed,
            "data": cfg,
            "attack": {"engines": engines},
            "defend": {"supervised": sup, "unsupervised": uns, "gnn": gnn_cfg},
            "eval": eval_cfg,
            "costs": costs,
        }
    )


def amount_signal(train: list[Transaction], holdout: list[Transaction]) -> tuple[np.ndarray, str]:
    """Rank by amount alone, direction chosen on the training window. The floor under every model.

    Returned as a score vector rather than as a metric block so the same `measure` runs over it,
    which is what keeps the floor on the same row as the models rather than in a footnote.
    """
    y = np.array([int(t.is_fraud) for t in train], dtype=int)
    a = np.array([t.amount for t in train], dtype=float)
    high_first = protocol.pr_auc(y, a) >= protocol.pr_auc(y, -a)
    signal = np.array([t.amount for t in holdout], dtype=float)
    return (
        signal if high_first else -signal,
        "largest amount first" if high_first else "smallest amount first",
    )


def calibrate_and_fit(detector, train: list[Transaction], eval_cfg: dict, fit_frac: float) -> None:
    """Fit the score → probability map on a validation tail of training, then refit on all of it.

    The same two-step `run_experiment.calibrate` runs, applied identically to every system. It
    matters here because the action bands are priced in calibrated probability: a system that
    skipped this would read its evasion and friction rates at a different operating point from
    the one it is being compared at.
    """
    from afl.data.splits import out_of_time_split

    fit_rows, val_rows = out_of_time_split(
        train, train_frac=fit_frac, embargo_days=float(eval_cfg["embargo_days"])
    )
    if len(val_rows) < 50 or not any(t.is_fraud for t in fit_rows):
        log.warning("not enough validation rows to calibrate — fitting on the whole window")
        detector.fit(train)
        return
    detector.fit(fit_rows)
    detector.policy.reset_calibration()
    y, s = protocol.align(val_rows, protocol.score_transactions(detector, val_rows, "calibration"))
    detector.policy.fit_calibrator(s, y)
    detector.fit(train)


def measure(
    name: str,
    rows: list[Transaction],
    probs: np.ndarray,
    scores,
    outcome: str,
    reason: str,
    fixed_fpr: float,
    k: int,
    compute: dict | None = None,
    card: dict | None = None,
) -> mule_graph.SystemResult:
    """One system's numbers on one seed's holdout, what the policy did, and what it cost.

    `outcome` is the *fold's* verdict, not this system's: a fold that cannot tell detection from
    provenance cannot tell it for any model in it, so every system in a withheld seed carries its
    numbers under `withheld_metrics`.
    """
    y = np.array([int(t.is_fraud) for t in rows], dtype=int)
    result = protocol.evaluate(y, probs, fixed_fpr=fixed_fpr, k=k)
    operational: dict[str, float] = {}
    if scores is not None:
        operational = {
            **{key: round(v, 6) for key, v in protocol.operational_rates(rows, scores).items()},
            **{f"action_{a}": v for a, v in action_mix(scores).items()},
        }
    quotable = outcome == loao.MEASURED
    return mule_graph.SystemResult(
        name=name,
        outcome=outcome,
        reason="" if quotable else reason,
        metrics=result if quotable else None,
        withheld_metrics=None if quotable else result,
        operational=operational,
        compute=compute or {},
        model_card=card or {},
    )


def lgbm_compute(detector, n_rows: int, fit_seconds: float, score_seconds: float) -> dict:
    """The baseline's bill, in the same shape the GNN reports its own."""
    return {
        "fit_seconds": round(fit_seconds, 2),
        "score_seconds": round(score_seconds, 2),
        "scored_rows": n_rows,
        "rows_per_second": round(n_rows / score_seconds, 1) if score_seconds else None,
        "n_parameters": None,
        "backend": str(detector.backend),
    }


def fit_and_score(detector, fold, eval_cfg: dict, fit_frac: float, run_id: str):
    """Calibrate, fit, score the holdout, and time both halves. Identical for every system."""
    started = time.perf_counter()
    calibrate_and_fit(detector, fold.train, eval_cfg, fit_frac)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = detector.score(as_batch(fold.holdout, run_id))
    score_seconds = time.perf_counter() - started
    _, probs = protocol.align(fold.holdout, scores)
    explain.assert_flagged_rows_are_explained(scores)
    return scores, probs, fit_seconds, score_seconds


def run_seed(
    composed,
    family: str,
    real: list[Transaction],
    envelope: AnchorEnvelope,
    split: CommittedSplit,
    generatable: list[str],
    sup: dict,
    gnn_cfg: dict,
    seed: int,
) -> mule_graph.SeedRun:
    """One family held out at one seed, end to end: generate, carve, guard, fit three, judge."""
    eval_cfg = OmegaConf.to_container(composed.eval)
    fixed_fpr, k = float(eval_cfg["fixed_fpr"]), int(eval_cfg["k"])
    min_positives = int(eval_cfg["min_meaningful_positives"])
    name = str(composed.data.name)
    window_hours = float(gnn_cfg["window_hours"])
    stride_hours = float(gnn_cfg["stride_hours"])

    set_all_seeds(seed)
    cfg = fold_config(composed, family, generatable)
    # the seed moves the *pool* as well as the model init. Without this line every seed would
    # draw the same episodes from the same simulator and the spread below would measure network
    # initialisation alone — which is the narrower question, and not the one the ticket asks.
    cfg.seed = seed
    cfg.eval.holdout_episodes = int(gnn_cfg["mule"]["episodes"])
    simulator = build_simulator(cfg, anchor=real, envelope=envelope)
    pool = build_pool(cfg, simulator, real)

    fold = loao.Fold.carve(pool, family, split=split)
    counts = fold.counts()
    injected = [t for t in pool if t.vector_id == family]
    log.info(
        "%s / %s / seed %d: pool %d -> train %d (%d fraud, no %s) / holdout %d (%d positives)",
        name,
        family,
        seed,
        len(pool),
        counts["train_rows"],
        counts["train_fraud"],
        family,
        counts["holdout_rows"],
        counts["holdout_positives"],
    )
    if not fold.n_train_fraud:
        return mule_graph.SeedRun(
            seed=seed,
            outcome=loao.SKIPPED,
            reason=(
                "the carve-out left no fraud in the training window — a single-class fit is not "
                "a detector, so there is nothing to measure this family against"
            ),
            counts=counts,
            guards=fold.guards,
        )
    if not fold.n_positives:
        return mule_graph.SeedRun(
            seed=seed,
            outcome=loao.SKIPPED,
            reason=(
                f"the holdout carries no {family} rows — every metric would read 0.0 without "
                "having measured anything. Widen the window or raise defend.gnn.mule.episodes"
            ),
            counts=counts,
            guards=fold.guards,
        )

    # ── the five audits that decide whether this seed's fold means anything ────
    separability = envelope_lib.audit(real, injected)
    provenance = provenance_probe(fold, sup, seed)
    neighbourhood = mule_graph.neighbourhood_audit(real, injected, window_hours, stride_hours)
    resolution = mule_graph.resolution_audit(
        real, injected, window_hours, stride_hours, envelope.time_granularity_s
    )

    spec = registry.get(family)
    blocked = template_gate(spec)
    if not blocked and fold.n_positives < min_positives:
        blocked = (
            f"{fold.n_positives} {family} positives against a floor of {min_positives} — recall "
            f"moves {1 / fold.n_positives:.1%} per row here, so this fold is reported as too "
            "thin to carry a claim rather than as a low score"
        )
    if not blocked and separability.get("trivially_separable"):
        blocked = (
            f"`{separability.get('worst')}` alone separates the injected {family} rows from the "
            f"anchor at PR-AUC {float(separability.get('score', 0.0)):.3f} — this fold measures "
            "provenance, not detection, and every number in it inherits that"
        )
    if not blocked and resolution.get("blind"):
        blocked = (
            f"this anchor's clock has {resolution['anchor_time_granularity_s']:,}s resolution, so "
            f"an injected {family} episode is very nearly instantaneous on it: only "
            f"{resolution['injected_share_seeing_an_earlier_family_edge']:.1%} of the injected "
            f"rows can see any earlier edge of their own episode inside the window, against a "
            f"floor of {mule_graph.MOTIF_VISIBILITY_FLOOR:.0%}. A causal temporal graph reads "
            "only what is strictly earlier, so the ring never exists in a snapshot that scores "
            "one of its own rows — this fold cannot measure a temporal graph model, whichever "
            "way its numbers fall"
        )
    if not blocked and neighbourhood.get("separable"):
        blocked = (
            f"a row's neighbourhood provenance alone sorts the injected {family} rows from this "
            f"anchor's own traffic at PR-AUC {float(neighbourhood['pr_auc']):.3f} — "
            f"{neighbourhood['injected_share_in_a_pure_island']:.0%} of the injected rows sit in "
            "a neighbourhood made only of other injected rows, so a model that passes messages "
            "between neighbours reads provenance here before it reads topology"
        )
    if not blocked and provenance.get("separable"):
        blocked = (
            f"a classifier sorts the injected {family} rows from this anchor's own traffic at "
            f"PR-AUC {float(provenance.get('pr_auc', 0.0)):.3f} — every positive in this fold is "
            "injected and every negative is real, so it cannot tell detection apart from "
            "provenance for any system in it"
        )
    outcome = loao.WITHHELD if blocked else loao.MEASURED

    costs = cost_model_for(OmegaConf.to_container(composed.costs), real)
    params, params_source = detector_params(composed)
    fit_frac = float(sup["tuning"]["fit_frac"])
    windows = tuple(int(w) for w in sup["features"]["windows_s"])
    stateful = bool(sup["features"]["stateful"])

    def supervised(builder) -> LGBMDetector:
        return LGBMDetector(
            policy=policy_from_config(sup["decision"], costs),
            features=builder(stateful=stateful, windows_s=windows),
            params=params,
            seed=seed,
            replay_weight=float(sup["replay_weight"]),
            params_source=params_source,
        )

    systems: dict[str, mule_graph.SystemResult] = {}
    common = dict(outcome=outcome, reason=blocked, fixed_fpr=fixed_fpr, k=k)

    # ── the champion: LightGBM over the whole hand-rolled table — what ships ────
    set_all_seeds(seed)
    champion = supervised(FeatureBuilder)
    scores, probs, fit_s, score_s = fit_and_score(champion, fold, eval_cfg, fit_frac, LGBM)
    systems[LGBM] = measure(
        LGBM,
        fold.holdout,
        probs,
        scores,
        compute=lgbm_compute(champion, len(fold.holdout), fit_s, score_s),
        card=model_card_of(champion),
        **common,
    )

    # ── the narrower baseline the ticket names: the graph blocks alone ──────────
    set_all_seeds(seed)
    graph_only = supervised(GraphFeatureBuilder)
    scores, probs, fit_s, score_s = fit_and_score(graph_only, fold, eval_cfg, fit_frac, GRAPH_LGBM)
    systems[GRAPH_LGBM] = measure(
        GRAPH_LGBM,
        fold.holdout,
        probs,
        scores,
        compute=lgbm_compute(graph_only, len(fold.holdout), fit_s, score_s),
        card={**model_card_of(graph_only), "features": graph_feature_names(windows)},
        **common,
    )

    # ── the challenger ─────────────────────────────────────────────────────────
    set_all_seeds(seed)
    challenger = TemporalGNNDetector(
        hidden=int(gnn_cfg["hidden"]),
        heads=int(gnn_cfg["heads"]),
        layers=int(gnn_cfg["layers"]),
        dropout=float(gnn_cfg["dropout"]),
        window_hours=window_hours,
        stride_hours=stride_hours,
        epochs=int(gnn_cfg["epochs"]),
        lr=float(gnn_cfg["learning_rate"]),
        negative_ratio=float(gnn_cfg["negative_ratio"]),
        max_edges=int(gnn_cfg["max_edges"]),
        policy=policy_from_config(sup["decision"], costs),
        seed=seed,
    )
    scores, probs, _, _ = fit_and_score(challenger, fold, eval_cfg, fit_frac, GNN)
    systems[GNN] = measure(
        GNN,
        fold.holdout,
        probs,
        scores,
        compute=challenger.compute_cost(),
        card=challenger.model_card(),
        **common,
    )

    # ── the floor ──────────────────────────────────────────────────────────────
    signal, direction = amount_signal(fold.train, fold.holdout)
    systems[FLOOR] = measure(FLOOR, fold.holdout, signal, None, **common)

    # the family guard again, against each *detector* rather than against the split: fitting is
    # where a replay buffer gets a say, and every one of these three has one
    guards = {
        **fold.guards,
        "family": loao.assert_family_held_out(fold.train, family, champion),
        "family_graph_lgbm": loao.assert_family_held_out(fold.train, family, graph_only),
        "family_gnn": loao.assert_family_held_out(fold.train, family, challenger),
    }
    return mule_graph.SeedRun(
        seed=seed,
        outcome=outcome,
        reason=blocked,
        systems=systems,
        floor={"direction": direction, "direction_chosen_on": "train"},
        counts=counts,
        guards=guards,
        separability=separability,
        provenance=provenance,
        neighbourhood=neighbourhood,
        resolution=resolution,
    )


def run_family(
    composed,
    family: str,
    real: list[Transaction],
    envelope: AnchorEnvelope,
    split: CommittedSplit,
    generatable: list[str],
    sup: dict,
    gnn_cfg: dict,
    seeds: list[int],
) -> mule_graph.MuleFold:
    """Every seed of one family, then the spread, the paired lift and the verdict."""
    spec = registry.get(family)
    if not spec.generatable:
        return mule_graph.MuleFold.skipped(
            family, f"{family} ({spec.name}) is declared but not implemented — {spec.gap}"
        )

    runs = [
        run_seed(composed, family, real, envelope, split, generatable, sup, gnn_cfg, seed)
        for seed in seeds
    ]
    metric = str(gnn_cfg["mule"]["metric"])
    spreads = {
        system: mule_graph.spread_across_seeds(runs, system, family, metric)
        for system in SYSTEMS
    }
    comparison = mule_graph.compare_across_seeds(runs, family, GNN, LGBM, metric)
    graph_comparison = mule_graph.compare_across_seeds(runs, family, GNN, GRAPH_LGBM, metric)

    # a fold blocked at any seed is blocked for the family: a fold that measures provenance at
    # one seed is not a fold that measures detection at the other two
    blocked = next((r.reason for r in runs if r.outcome != loao.MEASURED), "")
    outcome = loao.MEASURED if all(r.outcome == loao.MEASURED for r in runs) else loao.WITHHELD
    if not any(r.systems for r in runs):
        outcome = loao.SKIPPED
    promotion = mule_graph.decide_promotion(
        comparison,
        challenger=spreads.get(GNN),
        floor=spreads.get(FLOOR),
        material_gap=float(gnn_cfg["mule"]["material_gap"]),
        blocked=blocked,
    )
    return mule_graph.MuleFold(
        held_out_vector=family,
        outcome=outcome,
        reason=blocked,
        promotion=promotion,
        seeds=runs,
        spreads=spreads,
        comparison=comparison,
        graph_comparison=graph_comparison,
    )


def run_anchor(
    cfg: dict, args, sup, uns, gnn_cfg, eval_cfg, engines, costs
) -> mule_graph.GNNReport:
    """One anchor, both mule families, one operating point fixed before any of it was measured."""
    name = cfg["name"]
    seeds = args.seeds or [int(s) for s in gnn_cfg["mule"]["seeds"]]
    set_all_seeds(seeds[0])
    assert_one_operating_point(
        sup["decision"].get("calibrate_to_fpr"),
        float(eval_cfg["fixed_fpr"]),
        mode=str(sup["decision"]["mode"]),
    )

    if args.sample is not None:
        cfg = {**cfg, "sample": {**(cfg.get("sample") or {}), "sample_fraction": args.sample}}
    real = loaders.load_from_config(cfg)

    split = committed_split_for(cfg)
    if split is None:
        raise SystemExit(f"{name}: no committed split — run `make splits` first")

    composed = compose(cfg, sup, uns, gnn_cfg, eval_cfg, engines, costs, seeds[0])
    envelope = AnchorEnvelope.measure(real, name)
    generatable = [s.vector_id for s in registry.list_vectors(generatable=True)]

    families = args.families or list(gnn_cfg["mule"]["families"])
    folds = [
        run_family(composed, family, real, envelope, split, generatable, sup, gnn_cfg, seeds)
        for family in families
    ]

    return mule_graph.GNNReport(
        dataset=name,
        seeds=seeds,
        config={
            **gnn_cfg,
            "families": families,
            "source": str(GNN_CONFIG),
            "sample_fraction": (cfg.get("sample") or {}).get("sample_fraction"),
            "seeds": seeds,
        },
        operating_point={
            "fixed_fpr": float(eval_cfg["fixed_fpr"]),
            "k": int(eval_cfg["k"]),
            "source": str(EVAL_CONFIG),
            "min_meaningful_positives": int(eval_cfg["min_meaningful_positives"]),
            "band_units": "calibrated probability",
            "champion": "LGBMDetector over the whole hand-rolled table — what ships. The narrower "
            "graph-blocks-only column is reported next to it and does not decide the gate",
        },
        folds=folds,
        split=split.to_dict(),
        data={"config": f"config/data/{name}.yaml", "real": stats(real)},
        meta={
            "generated_by": "scripts/build_gnn.py",
            "note": (
                "Every positive in these folds is an injected synthetic row and every negative "
                "is a real one. Read each seed's `neighbourhood`, `separability` and "
                "`provenance` before reading its `systems` — the first of those is specific to "
                "this model, because a neighbourhood's provenance is a signal a message-passing "
                "model reads whether or not it is asked to."
            ),
        },
    )


# ── the document ────────────────────────────────────────────────────────────────
def _wrap(text: str, width: int = 98) -> str:
    """One paragraph, hard-wrapped, so the generated file reads like the hand-written ones."""
    words, lines, line = " ".join(str(text).split()).split(" "), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "\n".join(lines)


def _period(text: str) -> str:
    """A generated verdict as prose, ending in exactly one full stop. Never recapitalised.

    `Comparison.verdict` opens with a system name — `gnn`, `lgbm` — and sentence-casing it would
    print a column that does not exist.
    """
    text = " ".join(str(text).split())
    return text + ("" if not text or text.endswith(".") else ".")


def _sentence(text: str) -> str:
    """A generated reason as prose: capitalised, and ending in exactly one full stop."""
    text = " ".join(str(text).split())
    if not text:
        return ""
    return text[0].upper() + text[1:] + ("" if text.endswith(".") else ".")


def _compute(compute: dict) -> str:
    fit, score = compute.get("fit_seconds"), compute.get("score_seconds")
    if fit is None:
        return "—"
    bits = [f"{fit:,.0f}s fit", f"{score:,.0f}s score"]
    if compute.get("rows_per_second"):
        bits.append(f"{compute['rows_per_second']:,.0f} rows/s")
    if compute.get("n_parameters"):
        bits.append(f"{compute['n_parameters']:,} params")
    if compute.get("backend"):
        bits.append(str(compute["backend"]))
    return ", ".join(bits)


def _mean_compute(fold: mule_graph.MuleFold, system: str) -> str:
    """The bill averaged over the seeds that paid it."""
    bills = [r.systems[system].compute for r in fold.seeds if system in r.systems]
    bills = [b for b in bills if b.get("fit_seconds") is not None]
    if not bills:
        return "—"
    return _compute(
        {
            "fit_seconds": float(np.mean([b["fit_seconds"] for b in bills])),
            "score_seconds": float(np.mean([b["score_seconds"] for b in bills])),
            "rows_per_second": float(
                np.mean([b["rows_per_second"] or 0.0 for b in bills])
            ),
            "n_parameters": bills[0].get("n_parameters"),
            "backend": bills[0].get("backend"),
        }
    )


def _operational(fold: mule_graph.MuleFold, system: str, key: str) -> str:
    values = [
        r.systems[system].operational[key]
        for r in fold.seeds
        if system in r.systems and key in r.systems[system].operational
    ]
    return f"{float(np.mean(values)):.1%}" if values else "—"


def _fold_section(fold: mule_graph.MuleFold) -> str:
    spec = registry.get(fold.held_out_vector)
    out = [f"### {fold.held_out_vector} — {spec.name}\n"]

    if fold.outcome == loao.SKIPPED and not fold.seeds:
        out.append(_wrap(f"**Skipped.** {_sentence(fold.reason)}") + "\n")
        return "\n".join(out)

    seeds = [r.seed for r in fold.seeds]
    out.append(
        _wrap(
            f"Held out of training and measured at {len(seeds)} seed(s) — "
            f"{', '.join(map(str, seeds))} — each regenerating its own pool and refitting every "
            "system, so the lift below is a "
            "paired per-seed difference rather than one draw of the attacker's episodes."
        )
        + "\n"
    )
    if fold.reason:
        out.append(_wrap(f"**Withheld.** {_sentence(fold.reason)}") + "\n")
        out.append(
            _wrap(
                "The numbers below exist and are printed in brackets. They are evidence about the "
                "pipeline; they are not evidence about this family, and nothing in this repo "
                "quotes them."
            )
            + "\n"
        )
    else:
        out.append(_wrap(f"**Measured.** {_sentence(fold.promotion.reason)}") + "\n")

    first = fold.seeds[0]
    counts = first.counts
    out.append(
        "- **Fold** — "
        + _wrap(
            f"{counts.get('holdout_rows', 0):,} holdout rows carrying "
            f"{counts.get('holdout_positives', 0):,} {fold.held_out_vector} positives at the "
            f"first seed, against {counts.get('train_rows', 0):,} training rows with "
            f"{counts.get('train_fraud', 0):,} fraud and not one {fold.held_out_vector} row.",
            width=400,
        )
    )
    res = first.resolution or {}
    if res.get("checked"):
        out.append(
            "- **Resolution** — "
            + _wrap(
                f"this anchor's clock lands on {res['anchor_time_granularity_s']:,}s, and "
                f"{res['injected_share_seeing_an_earlier_family_edge']:.1%} of the injected rows "
                f"can see an earlier edge of their own episode inside the "
                f"{res['window_hours']:.0f}h window "
                f"({res['injected_share_seeing_any_earlier_edge']:.1%} can see any earlier edge "
                f"at all), against a floor of {res['floor']:.0%}: "
                f"**{'blind' if res['blind'] else 'ok'}**.",
                width=400,
            )
        )
    hood = first.neighbourhood or {}
    if hood.get("checked"):
        out.append(
            "- **Neighbourhood** — "
            + _wrap(
                f"the anchor's own rows sit in neighbourhoods that are "
                f"{hood['anchor_mean_synthetic_neighbour_share']:.1%} synthetic on average; the "
                f"injected ones, {hood['injected_mean_synthetic_neighbour_share']:.1%}, and "
                f"{hood['injected_share_in_a_pure_island']:.1%} of them sit in a neighbourhood "
                f"made only of other injected rows. That share alone separates the two at PR-AUC "
                f"{hood['pr_auc']:.3f}: **{'separable' if hood['separable'] else 'ok'}**.",
                width=400,
            )
        )
    sep = first.separability or {}
    if sep.get("worst"):
        out.append(
            "- **Commensurability** — "
            + _wrap(
                f"worst single contract field `{sep['worst']}` at PR-AUC "
                f"{float(sep.get('score', 0.0)):.4f}: "
                f"**{'separable' if sep.get('trivially_separable') else 'ok'}**.",
                width=400,
            )
        )
    prov = first.provenance or {}
    if prov.get("checked"):
        out.append(
            "- **Provenance** — "
            + _wrap(
                f"a classifier over the detector's own features sorts injected rows from real "
                f"ones at PR-AUC {float(prov['pr_auc']):.3f} against a base rate of "
                f"{float(prov['base_rate']):.4f}: "
                f"**{'separable' if prov.get('separable') else 'ok'}**.",
                width=400,
            )
        )
    out.append("")

    out.append("| system | PR-AUC | rec@FPR | p@k | evasion | friction | compute |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for name in SYSTEMS:
        spread = fold.spreads.get(name)
        if spread is None or not spread.values:
            continue
        recall = mule_graph.spread_across_seeds(
            fold.seeds, name, fold.held_out_vector, "recall_at_fixed_fpr"
        )
        precision = mule_graph.spread_across_seeds(
            fold.seeds, name, fold.held_out_vector, "precision_at_k"
        )
        out.append(
            f"| {name} | {spread.text()} | {recall.text()} | {precision.text()} | "
            f"{_operational(fold, name, 'evasion_rate')} | "
            f"{_operational(fold, name, 'friction_rate')} | {_mean_compute(fold, name)} |"
        )
    out.append("")
    out.append(
        _wrap(
            "Every cell is mean ± sd over the seeds. A number in brackets is one this file will "
            "not quote. `lgbm` is the whole hand-rolled table and is the champion the gate is "
            "decided on; `graph_lgbm` is the same model over the graph blocks alone, which is the "
            "narrower baseline the ticket names. `evasion` is the share of the held-out family "
            "the policy let through untouched and `friction` the share of legit traffic that "
            "carried any action; the floor row has neither because it has no policy."
        )
        + "\n"
    )

    for comparison, label in (
        (fold.comparison, "against what ships"),
        (fold.graph_comparison, "against graph features alone"),
    ):
        if comparison is None:
            continue
        out.append(_wrap(f"**Lift {label}** — {_period(comparison.verdict)}") + "\n")
        out.append("| seed | " + comparison.incumbent + " | gnn | delta |")
        out.append("| --- | ---: | ---: | ---: |")
        for row in comparison.per_seed:
            out.append(
                f"| {row['seed']} | {row['incumbent']:.3f} | {row['challenger']:.3f} | "
                f"{row['delta']:+.3f} |"
            )
        out.append("")

    if fold.reason:
        out.append(
            _wrap(
                "**Gate.** Never reached. The fold was refused before the comparison could carry "
                "anything, so the bracketed rows above are what the gate *would* have read and "
                "not what it did. What ships is the stated fallback: "
                f"{mule_graph.FALLBACK}."
            )
            + "\n"
        )
    else:
        out.append(_wrap(f"**Gate.** {_sentence(fold.promotion.reason)}") + "\n")
    return "\n".join(out)


def _headline(reports: dict[str, mule_graph.GNNReport]) -> str:
    """The cross-anchor reading, derived from the artefacts rather than typed over them."""
    rows = [
        "| anchor | family | motif visible | connected rows | gnn | lgbm | graph_lgbm | "
        "amount | fold |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, report in sorted(reports.items()):
        for fold in report.folds:
            card = next(
                (r.systems[GNN].model_card for r in fold.seeds if GNN in r.systems), {}
            )
            coverage = card.get("graph_coverage", {})
            isolated = coverage.get("share_of_rows_whose_endpoints_are_isolated")
            cells = " | ".join(
                fold.spreads[s].text() if s in fold.spreads and fold.spreads[s].values else "—"
                for s in SYSTEMS
            )
            res = next((r.resolution or {} for r in fold.seeds if r.resolution), {})
            visible = res.get("injected_share_seeing_an_earlier_family_edge")
            rows.append(
                f"| {name} | {fold.held_out_vector} | "
                f"{f'{visible:.0%}' if visible is not None else '—'} | "
                f"{f'{1 - isolated:.0%}' if isolated is not None else '—'} | "
                f"{cells} | "
                f"{'measured' if fold.outcome == loao.MEASURED else fold.outcome} |"
            )

    promoted = [name for name, r in reports.items() if r.promoted]
    if promoted:
        head = _wrap(
            "**The temporal GNN earned its seat on "
            + ", ".join(sorted(promoted))
            + ".** It beats the hand-rolled baseline on the same out-of-time split at the same "
            "operating point, at every seed's own pool, by a margin larger than the seed-to-seed "
            "spread it sits in."
        )
    else:
        head = _wrap(
            "**The temporal GNN did not earn its seat on any anchor measured, and that is the "
            "result.** It is not in the headline table, `config/defend/gnn.yaml` stays "
            f"`enabled: false`, and the stated fallback is what ships: {mule_graph.FALLBACK}. "
            "The comparison is published because a negative result that lives on one laptop is "
            "not a result, and because the next person to reach for message passing on a mule "
            "family should be able to read what happened when we did."
        )
    return "\n".join(
        [
            head,
            "",
            *rows,
            "",
            _wrap(
                "Every cell is mean ± sd of PR-AUC over the seeds, and every one in brackets is a "
                "number this file will not quote; the per-fold sections say which audit refused "
                "each one. `motif visible` is the share of injected rows that can see any earlier "
                "edge of their own episode — a causal temporal graph reads only what is strictly "
                "earlier, so below the floor the ring never exists in a snapshot that scores one "
                "of its own rows. `connected rows` is the share whose endpoints are not both "
                "isolated inside the window, which is what message passing has to pass over at "
                "all."
            ),
        ]
    )


def _probe_power(reports: dict[str, mule_graph.GNNReport]) -> str:
    """What this run says about the committed leave-one-attack-out matrix, if anything.

    These folds carry many more injected rows than the matrix's do, because
    `defend.gnn.mule.episodes` is well above `eval.holdout_episodes`. That makes the provenance
    probe better powered — and `build_loao`'s own docstring is explicit that a low probe score on
    a thin fold is weak evidence of soundness while a high one is strong evidence of the
    opposite. Where the same family comes back separable at this size, the matrix row it
    disagrees with is named rather than left for somebody to notice. `docs/sequence.md` reported
    the same effect on AMLSim C1; this is the second time it has happened, which makes it a
    property of the episode count rather than of one family.
    """
    lines = []
    for name, report in sorted(reports.items()):
        try:
            matrix = loao.LeaveOneAttackOutReport.load(name)
        except (FileNotFoundError, ValueError):
            continue
        for fold in report.folds:
            here = next((r.provenance or {} for r in fold.seeds if r.provenance), {})
            there = matrix.fold(fold.held_out_vector)
            was = (there.provenance or {}) if there else {}
            if not (here.get("separable") and there and there.outcome == loao.MEASURED):
                continue
            lines.append(
                _wrap(
                    f"**{name} / {fold.held_out_vector}.** `artifacts/loao/{name}.json` reports "
                    f"this family as *measured*, on a probe that saw {was.get('n_injected', '?')} "
                    f"injected rows and scored PR-AUC {float(was.get('pr_auc', 0.0)):.3f}. The "
                    f"same probe here sees {here.get('n_injected')} and scores "
                    f"{float(here['pr_auc']):.3f}, over the bar. Nothing about the generator "
                    "changed between the two runs — the episode count did. Read the matrix row "
                    "as underpowered rather than as a contradiction, and `make loao` at this "
                    "episode count would be the way to settle it."
                )
            )
    if not lines:
        return ""
    return "## What this says about the leave-one-attack-out matrix\n\n" + "\n\n".join(lines) + "\n"


def gnn_doc(reports: dict[str, mule_graph.GNNReport], missing: list[str]) -> str:
    out = [
        "# The temporal GNN, and which model shipped\n",
        _wrap(
            "_Generated by `scripts/build_gnn.py` from the anchors on disk. Every number below "
            "traces to `artifacts/gnn/<anchor>.json`, which carries the split digest, the seeds, "
            "the model cards, the compute cost and the four audits that decide whether a fold "
            "means anything. Do not edit this file — re-run `make gnn`._"
        )
        + "\n",
        _wrap(
            "Ticket 18. Graph attention over the account-beneficiary graph inside an explicit "
            "temporal window is measured against the hand-rolled graph features + LightGBM "
            "baseline on the same out-of-time split at the same operating point, on the mule "
            "families — S1 fan-in and layering, C3 instant relay — at several seeds, with the "
            "lift reported as a paired per-seed difference and its spread. The model enters a "
            "reported table only if it beats the baseline by more than that spread. The "
            "comparison is published either way, and the fallback was named before the "
            "experiment ran."
        )
        + "\n",
        "## The result\n",
        _headline(reports) + "\n",
        _wrap(
            "**Read the audits before the tables.** Every positive in these folds is an injected "
            "synthetic row and every negative is a real one, so a fold where one contract field "
            "— or, for this model specifically, one neighbourhood's provenance — separates the "
            "two is measuring which generator wrote the row rather than whether anything detects "
            "fraud."
        )
        + "\n",
    ]
    for name in sorted(reports):
        report = reports[name]
        out.append(f"## {name}\n")
        out.append(
            _wrap(
                f"Committed split digest `{report.split.get('digest', '?')}`, seeds "
                f"{', '.join(str(s) for s in report.seeds)}, operating point "
                f"{float(report.operating_point['fixed_fpr']):.0%} FPR and "
                f"k={report.operating_point['k']} from `{report.operating_point['source']}`. "
                f"**Shipped here: {report.shipped}.**"
            )
            + "\n"
        )
        for fold in report.folds:
            out.append(_fold_section(fold))
    probe = _probe_power(reports)
    if probe:
        out.append(probe)
    if missing:
        out.append("## Anchors not measured\n")
        out.append(
            _wrap(
                ", ".join(sorted(missing))
                + (" is" if len(missing) == 1 else " are")
                + " not on disk, so "
                + ("it is" if len(missing) == 1 else "they are")
                + " named as missing rather than reported as a zero. See the data card for how "
                "to fetch " + ("it" if len(missing) == 1 else "them") + "."
            )
            + "\n"
        )
    out.append("## What this does not settle\n")
    out.append(
        _wrap(
            "One architecture, one window, one set of hyperparameters read from "
            "`config/defend/gnn.yaml`, and three seeds — which cannot reach p < 0.05 by "
            "construction and says so in every sign test above. A negative result here is "
            "evidence that *this* temporal GNN, on *these* anchors, does not beat the per-row "
            "table on the families it was built for — not that no graph model could. What would "
            "change the answer is an anchor whose real fraud is itself a ring, so the comparison "
            "could be run on real mule topology rather than on injected episodes; that is the "
            "precondition the neighbourhood audit measures and the one neither anchor meets on "
            "its own labels."
        )
        + "\n"
    )
    return "\n".join(out)


# ── the CLI ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("anchors", nargs="*", help="anchor names; default every real anchor")
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(v) for v in s.split(",") if v.strip()],
        default=None,
        help="comma-separated seeds; default defend.gnn.mule.seeds",
    )
    parser.add_argument(
        "--families",
        type=lambda s: [v.strip() for v in s.split(",") if v.strip()],
        default=None,
        help="comma-separated mule families; default defend.gnn.mule.families",
    )
    parser.add_argument(
        "--sample", type=float, default=None, help="override sample_fraction for a quick pass"
    )
    parser.add_argument(
        "--doc-only",
        action="store_true",
        help="rewrite docs/gnn.md from the committed artefacts without re-running anything",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_DIR,
        help="where the artefacts go; point it elsewhere so a quick pass cannot overwrite the "
        "committed comparison",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.doc_only:
        reports = mule_graph.load_all(args.out)
        if not reports:
            print(f"no committed comparison in {args.out} — run `make gnn` first")
            return 1
        missing = [c["name"] for c in anchors(args.anchors) if c["name"] not in reports]
        DOC_PATH.write_text(gnn_doc(reports, missing))
        print(f"→ {DOC_PATH} (from {len(reports)} committed artefact(s))")
        return 0

    if not available():
        print(
            "the temporal GNN needs the `deep` extra: uv sync --extra deep. It is not installed, "
            "and this script refuses to report a number from a model it could not build rather "
            "than degrading to something that scores like a detector that caught nothing.",
            file=sys.stderr,
        )
        return 2

    sup = yaml.safe_load(LGBM_CONFIG.read_text())
    uns = yaml.safe_load(ANOMALY_CONFIG.read_text())
    gnn_cfg = yaml.safe_load(GNN_CONFIG.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    engines = yaml.safe_load(ENGINES_CONFIG.read_text())
    costs = yaml.safe_load(COSTS_CONFIG.read_text())

    reports: dict[str, mule_graph.GNNReport] = {}
    missing: list[str] = []
    for cfg in anchors(args.anchors):
        name = cfg["name"]
        try:
            reports[name] = run_anchor(cfg, args, sup, uns, gnn_cfg, eval_cfg, engines, costs)
        except loaders.DatasetNotDownloaded as e:
            # the rule every build script here follows: an anchor that is not downloaded is named
            # as skipped, never reported as a zero
            log.warning("%s: skipped — %s", name, e)
            missing.append(name)

    if not reports:
        print("no anchor on disk — nothing to compare", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        baseline.assert_no_forbidden_metrics(report.to_dict()["folds"])
        path = report.save(args.out)
        print(f"\n── {name} " + "─" * (70 - len(name)))
        for fold in report.folds:
            print(f"  {'  ' if fold.promotion.promoted else '! '}{fold.summary()}")
        print(f"  shipped: {report.shipped}")
        print(f"  → {path}")

    if args.out != ARTIFACT_DIR:
        print(f"\n(--out is not {ARTIFACT_DIR}; leaving {DOC_PATH} alone)")
        return 0
    for path in sorted(ARTIFACT_DIR.glob("*.json")):
        # an anchor this run did not touch, but whose artefact is committed, still belongs in the
        # document
        raw = json.loads(path.read_text())
        reports.setdefault(path.stem, mule_graph.GNNReport.from_dict(raw))
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(gnn_doc(reports, missing))
    print(f"\n→ {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
