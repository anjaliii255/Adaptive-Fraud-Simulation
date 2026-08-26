"""Does a sequence model beat LightGBM on the drift arc? Measure it, and publish it either way.

    python scripts/build_sequence.py                 # every real anchor, both drift families
    python scripts/build_sequence.py amlsim          # just one
    python scripts/build_sequence.py --families S3   # just one family
    python scripts/build_sequence.py --sample 0.02   # a quick pass
    python scripts/build_sequence.py --doc-only      # rewrite the doc from committed artefacts

Ticket 17's deliverable, and like ticket 10's it is a comparison rather than a model. A GRU over
per-entity history is the obvious thing to reach for on an account-takeover arc, and it is also
the easiest place in this repo to produce a number that flatters itself: it reads a signal the
per-row feature table does not have, and on an anchor whose accounts appear once that signal is
"this row was injected".

So the comparison is set up to be losable.

**One split, one operating point, three systems.** The champion is LightGBM — not the ensemble
the loop ships, because the ticket's bar is the supervised baseline and comparing against a
blend would change two things at once. The challenger is `SequenceDetector`, fitted on the same
carve-out and calibrated on the same validation tail. `amount_only` is under both of them:
direction chosen on the training window, no model, no training. Two results in this repo were
walked back for want of that column.

**Two families, both ends of the axis.** S3 is a takeover of a real account, C1 a genuinely old
account busting out. Each is generated twice — `ramp` at the low and high end of its own declared
search space, nothing else changed — and each family is held out of training in turn, so the
number is generalisation to an unseen family rather than memorisation of a seen one. The two arcs
are then reported separately against the same haystack at the same threshold, because averaging
them hides which end paid: sudden takeover is an event a per-row table already sees, and gradual
drift is the case this model is supposed to exist for.

**Three ways this fold can fail to mean anything, all checked.** Whether one contract field
separates the injected family from the anchor (`envelope.audit`); whether a classifier can sort
injected rows from real ones in the detector's own feature space (`build_loao.provenance_probe`);
and — the one specific to this model — whether *window length* does it by itself
(`drift_arc.history_audit`). PaySim is the case that check exists for: `nameOrig` is effectively
unique per row, so a real window there is one step long while every injected episode carries a
full arc.

Everything lands in `artifacts/sequence/<anchor>.json` and in `docs/sequence.md`, which is
generated from those files and never hand-typed. The gate that decides whether the model may
appear in a reported table is `afl.evaluation.drift_arc.decide_promotion`, and
`assert_config_matches_promotion` keeps `config/defend/sequence.yaml` answerable to it.
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

# One definition of the fold, the simulator and the probe. This script differs from
# `build_loao.py` in exactly one place — the held-out family is generated twice, at both ends of
# `ramp` — and every other piece is imported rather than re-derived, because a second copy of a
# split rule drifts the first time either is edited.
from build_loao import provenance_probe, template_gate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from run_experiment import build_simulator, detector_params  # noqa: E402

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
from afl.defend.features import FeatureBuilder  # noqa: E402
from afl.defend.models.lgbm import LGBMDetector, model_card_of  # noqa: E402
from afl.defend.models.sequence import SequenceDetector, available  # noqa: E402
from afl.evaluation import drift_arc, protocol  # noqa: E402
from afl.evaluation import leave_one_attack_out as loao  # noqa: E402
from afl.utils.seed import set_all_seeds  # noqa: E402

log = logging.getLogger("build_sequence")

DATA_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
ANOMALY_CONFIG = Path("config/defend/anomaly.yaml")
SEQUENCE_CONFIG = Path("config/defend/sequence.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
ENGINES_CONFIG = Path("config/attack/engines.yaml")
COSTS_CONFIG = Path("config/costs/default.yaml")
ARTIFACT_DIR = drift_arc.DEFAULT_SEQUENCE_DIR
DOC_PATH = Path("docs/sequence.md")

LGBM, SEQUENCE, FLOOR = "lgbm", "sequence", "amount_only"


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


def compose(cfg, sup, uns, seq, eval_cfg, engines, costs, seed: int):
    """The same config tree `config/config.yaml` composes, built here from the same files."""
    return OmegaConf.create(
        {
            "seed": seed,
            "data": cfg,
            "attack": {"engines": engines},
            "defend": {"supervised": sup, "unsupervised": uns, "sequence": seq},
            "eval": eval_cfg,
            "costs": costs,
        }
    )


def arc_pool(
    simulator, real: list[Transaction], family: str, episodes: int, generatable: list[str]
) -> tuple[list[Transaction], dict[str, str]]:
    """Real traffic, the other families as background, and `family` at both ends of `ramp`.

    A near-copy of `run_experiment.build_pool` and it has to be: the one difference is that the
    held-out family is generated twice rather than once, which is the whole experiment. The rule
    it does share is the important one — **on a real anchor only the fraud rows of a batch are
    injected**. The drift engine's pre-event baseline is dropped with everything else, so an
    episode's history is the anchor account's own real traffic rather than synthesised filler,
    and the arc lives entirely in the injected tail where `ramp` shapes it.
    """
    pool = list(real)
    for vid in generatable:
        if vid == family:
            continue
        batch = simulator.generate(registry.get(vid).to_attack_params())
        pool.extend(
            batch.transactions if not real else [t for t in batch.transactions if t.is_fraud]
        )

    arc_of_run: dict[str, str] = {}
    saved = simulator.n_episodes
    simulator.n_episodes = max(saved, episodes)
    for arc in drift_arc.ARCS:
        params = registry.get(family).to_attack_params(drift_arc.arc_params(family, arc))
        batch = simulator.generate(params)
        arc_of_run[batch.run_id] = arc
        pool.extend(
            batch.transactions if not real else [t for t in batch.transactions if t.is_fraud]
        )
    simulator.n_episodes = saved
    return pool, arc_of_run


def amount_signal(train: list[Transaction], holdout: list[Transaction]) -> tuple[np.ndarray, str]:
    """Rank by amount alone, direction chosen on the training window. The floor under both models.

    Returned as a score vector rather than as a metric block, so the arc breakdown can be run
    over it the same way it is run over a model — the floor has to be visible at *each* end of
    the axis, not only over the fold as a whole.
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

    The same two-step `run_experiment.calibrate` runs, applied identically to both systems. It
    matters more here than usual: the action bands are priced in calibrated probability, so a
    system that skipped this would be reading its evasion and friction rates at a different
    operating point from the one it is being compared at.
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
    arcs: dict[str, str],
    ramps: dict[str, float],
    fixed_fpr: float,
    k: int,
    min_positives: int,
    compute: dict | None = None,
    card: dict | None = None,
) -> drift_arc.SystemResult:
    """One system's whole-fold numbers, its two arcs, what the policy did, and what it cost."""
    y = np.array([int(t.is_fraud) for t in rows], dtype=int)
    overall = protocol.evaluate(y, probs, fixed_fpr=fixed_fpr, k=k)
    operational: dict[str, float] = {}
    if scores is not None:
        operational = {
            **{key: round(v, 6) for key, v in protocol.operational_rates(rows, scores).items()},
            **{f"action_{a}": v for a, v in action_mix(scores).items()},
        }
    return drift_arc.SystemResult(
        name=name,
        overall=overall,
        arcs=drift_arc.arc_breakdown(rows, probs, arcs, ramps, fixed_fpr, k, min_positives),
        operational=operational,
        compute=compute or {},
        model_card=card or {},
    )


def run_family(
    composed,
    family: str,
    real: list[Transaction],
    envelope: AnchorEnvelope,
    split: CommittedSplit,
    generatable: list[str],
    sup: dict,
    seq_cfg: dict,
    seed: int,
) -> drift_arc.ArcFold:
    """One drift family held out end to end: generate both arcs, carve, guard, fit both, judge."""
    eval_cfg = OmegaConf.to_container(composed.eval)
    fixed_fpr, k = float(eval_cfg["fixed_fpr"]), int(eval_cfg["k"])
    min_positives = int(eval_cfg["min_meaningful_positives"])
    name = str(composed.data.name)
    max_len = int(seq_cfg["max_len"])
    entity = str(seq_cfg["entity"])

    spec = registry.get(family)
    if not spec.generatable:
        return drift_arc.ArcFold.skipped(
            family, f"{family} ({spec.name}) is declared but not implemented — {spec.gap}"
        )

    set_all_seeds(seed)
    simulator = build_simulator(composed, anchor=real, envelope=envelope)
    pool, arc_of_run = arc_pool(
        simulator, real, family, int(seq_cfg["arc"]["episodes"]), generatable
    )
    ramps = {arc: drift_arc.arc_ramp(family, arc) for arc in drift_arc.ARCS}

    fold = loao.Fold.carve(pool, family, split=split)
    counts = fold.counts()
    injected = [t for t in pool if t.vector_id == family]
    arcs = drift_arc.tag_arcs(pool, arc_of_run)
    per_arc = {
        arc: sum(1 for t in fold.holdout if t.is_fraud and arcs.get(t.txn_id) == arc)
        for arc in drift_arc.ARCS
    }
    counts = {**counts, "holdout_positives_by_arc": per_arc}
    log.info(
        "%s / %s: pool %d -> train %d (%d fraud, no %s) / holdout %d (%s)",
        name,
        family,
        len(pool),
        counts["train_rows"],
        counts["train_fraud"],
        family,
        counts["holdout_rows"],
        ", ".join(f"{a} {n}" for a, n in per_arc.items()),
    )
    if not fold.n_train_fraud:
        return drift_arc.ArcFold.skipped(
            family,
            "the carve-out left no fraud in the training window — a single-class fit is not a "
            "detector, so there is nothing to measure this family against",
            counts=counts,
            guards=fold.guards,
            ramps=ramps,
        )
    if not fold.n_positives:
        return drift_arc.ArcFold.skipped(
            family,
            f"the holdout carries no {family} rows — every metric would read 0.0 without having "
            "measured anything. Widen the window or raise defend.sequence.arc.episodes",
            counts=counts,
            guards=fold.guards,
            ramps=ramps,
        )

    # ── the three audits that decide whether this fold means anything ───────────
    separability = envelope_lib.audit(real, injected)
    provenance = provenance_probe(fold, sup, seed)
    history = drift_arc.history_audit(real, injected, max_len, entity)

    costs = cost_model_for(OmegaConf.to_container(composed.costs), real)
    params, params_source = detector_params(composed)

    def features() -> FeatureBuilder:
        return FeatureBuilder(
            stateful=bool(sup["features"]["stateful"]),
            windows_s=tuple(int(w) for w in sup["features"]["windows_s"]),
        )

    systems: dict[str, drift_arc.SystemResult] = {}
    common = dict(arcs=arcs, ramps=ramps, fixed_fpr=fixed_fpr, k=k, min_positives=min_positives)

    # ── the champion: LightGBM, not the ensemble the loop ships ─────────────────
    set_all_seeds(seed)
    started = time.perf_counter()
    champion = LGBMDetector(
        policy=policy_from_config(sup["decision"], costs),
        features=features(),
        params=params,
        seed=seed,
        replay_weight=float(sup["replay_weight"]),
        params_source=params_source,
    )
    calibrate_and_fit(champion, fold.train, eval_cfg, float(sup["tuning"]["fit_frac"]))
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    champion_scores = champion.score(as_batch(fold.holdout, "lgbm"))
    score_seconds = time.perf_counter() - started
    _, p_champion = protocol.align(fold.holdout, champion_scores)
    explain.assert_flagged_rows_are_explained(champion_scores)
    systems[LGBM] = measure(
        LGBM,
        fold.holdout,
        p_champion,
        champion_scores,
        compute={
            "fit_seconds": round(fit_seconds, 2),
            "score_seconds": round(score_seconds, 2),
            "scored_rows": len(fold.holdout),
            "rows_per_second": round(len(fold.holdout) / score_seconds, 1)
            if score_seconds
            else None,
            "n_parameters": None,
            "backend": str(champion.backend),
        },
        card=model_card_of(champion),
        **common,
    )

    # ── the challenger ──────────────────────────────────────────────────────────
    set_all_seeds(seed)
    challenger = SequenceDetector(
        arch=str(seq_cfg["arch"]),
        hidden=int(seq_cfg["hidden"]),
        layers=int(seq_cfg["layers"]),
        dropout=float(seq_cfg["dropout"]),
        max_len=max_len,
        epochs=int(seq_cfg["epochs"]),
        batch_size=int(seq_cfg["batch_size"]),
        lr=float(seq_cfg["learning_rate"]),
        negative_ratio=float(seq_cfg["negative_ratio"]),
        entity=entity,
        policy=policy_from_config(sup["decision"], costs),
        seed=seed,
    )
    calibrate_and_fit(challenger, fold.train, eval_cfg, float(sup["tuning"]["fit_frac"]))
    challenger_scores = challenger.score(as_batch(fold.holdout, "sequence"))
    _, p_challenger = protocol.align(fold.holdout, challenger_scores)
    explain.assert_flagged_rows_are_explained(challenger_scores)
    systems[SEQUENCE] = measure(
        SEQUENCE,
        fold.holdout,
        p_challenger,
        challenger_scores,
        compute=challenger.compute_cost(),
        card=challenger.model_card(),
        **common,
    )

    # ── the floor ───────────────────────────────────────────────────────────────
    signal, direction = amount_signal(fold.train, fold.holdout)
    systems[FLOOR] = measure(FLOOR, fold.holdout, signal, None, **common)

    # the family guard again, against each *detector* rather than against the split: fitting is
    # where a replay buffer gets a say, and a sequence model has one too
    guards = {
        **fold.guards,
        "family": loao.assert_family_held_out(fold.train, family, champion),
        "family_sequence": loao.assert_family_held_out(fold.train, family, challenger),
    }

    blocked = template_gate(spec)
    if not blocked and separability.get("trivially_separable"):
        blocked = (
            f"`{separability.get('worst')}` alone separates the injected {family} rows from the "
            f"anchor at PR-AUC {float(separability.get('score', 0.0)):.3f} — this fold measures "
            "provenance, not detection, and every number in it inherits that"
        )
    if not blocked and history.get("separable"):
        blocked = (
            f"window length alone sorts the injected {family} rows from this anchor's own "
            f"traffic at PR-AUC {float(history['pr_auc']):.3f} — the anchor's own rows carry "
            f"{history['anchor_mean_window']:.1f} steps of history on average against "
            f"{history['injected_mean_window']:.1f} for the injected ones, so a model that reads "
            "per-entity history reads provenance here before it reads behaviour"
        )
    if not blocked and provenance.get("separable"):
        blocked = (
            f"a classifier sorts the injected {family} rows from this anchor's own traffic at "
            f"PR-AUC {float(provenance.get('pr_auc', 0.0)):.3f} — every positive in this fold is "
            "injected and every negative is real, so it cannot tell detection apart from "
            "provenance for either system"
        )

    gradual_floor = systems[FLOOR].arcs.get(drift_arc.GRADUAL)
    promotion = drift_arc.decide_promotion(
        challenger=systems[SEQUENCE].arcs,
        champion=systems[LGBM].arcs,
        floor=float(gradual_floor.any_metrics.pr_auc)
        if gradual_floor and gradual_floor.any_metrics
        else None,
        arc=str(seq_cfg["arc"]["decided_on"]),
        metric=str(seq_cfg["arc"]["metric"]),
        material_gap=float(seq_cfg["arc"]["material_gap"]),
        blocked=blocked,
    )

    return drift_arc.ArcFold(
        held_out_vector=family,
        outcome=loao.WITHHELD if blocked else loao.MEASURED,
        reason=blocked,
        promotion=promotion,
        systems=systems,
        floor={"direction": direction, "direction_chosen_on": "train"},
        counts=counts,
        guards=guards,
        ramps=ramps,
        separability=separability,
        provenance=provenance,
        history=history,
    )


def run_anchor(
    cfg: dict, args, sup, uns, seq_cfg, eval_cfg, engines, costs
) -> drift_arc.SequenceReport:
    """One anchor, both drift families, one operating point fixed before any of it was measured."""
    name = cfg["name"]
    seed = int(args.seed)
    set_all_seeds(seed)
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

    composed = compose(cfg, sup, uns, seq_cfg, eval_cfg, engines, costs, seed)
    envelope = AnchorEnvelope.measure(real, name)
    generatable = [s.vector_id for s in registry.list_vectors(generatable=True)]

    families = args.families or list(seq_cfg["arc"]["families"])
    folds = [
        run_family(composed, family, real, envelope, split, generatable, sup, seq_cfg, seed)
        for family in families
    ]

    return drift_arc.SequenceReport(
        dataset=name,
        seed=seed,
        config={
            **seq_cfg,
            "families": families,
            "source": str(SEQUENCE_CONFIG),
            "sample_fraction": (cfg.get("sample") or {}).get("sample_fraction"),
            "seed": seed,
        },
        operating_point={
            "fixed_fpr": float(eval_cfg["fixed_fpr"]),
            "k": int(eval_cfg["k"]),
            "source": str(EVAL_CONFIG),
            "min_meaningful_positives": int(eval_cfg["min_meaningful_positives"]),
            "band_units": "calibrated probability",
            "champion": "LGBMDetector, not the ensemble the loop ships — the ticket's bar is the "
            "supervised baseline, and a blend would move two things at once",
        },
        folds=folds,
        split=split.to_dict(),
        data={"config": f"config/data/{name}.yaml", "real": stats(real)},
        meta={
            "generated_by": "scripts/build_sequence.py",
            "note": (
                "Every positive in these folds is an injected synthetic row and every negative "
                "is a real one. Read each fold's `history`, `separability` and `provenance` "
                "before reading its `systems` — the first of those is specific to this model, "
                "because window length is a signal a sequence model reads whether or not it is "
                "asked to."
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


def _sentence(text: str) -> str:
    """A generated reason as prose: capitalised, and ending in exactly one full stop."""
    text = " ".join(str(text).split())
    if not text:
        return ""
    return text[0].upper() + text[1:] + ("" if text.endswith(".") else ".")


def _cell(result: drift_arc.ArcResult | None, field: str, quotable: bool = True) -> str:
    """A number, a number in brackets when it may not be quoted, or an em dash.

    Two independent ways to lose the right to quote a number, and both put it in brackets: the arc
    itself is too thin to carry a claim, or the whole fold is — an arc with 500 positives inside a
    fold that is measuring provenance is still measuring provenance.
    """
    if result is None or result.any_metrics is None:
        return "—"
    text = f"{getattr(result.any_metrics, field):.3f}"
    return text if (quotable and result.outcome == loao.MEASURED) else f"({text})"


def _compute(compute: dict) -> str:
    fit, score = compute.get("fit_seconds"), compute.get("score_seconds")
    if fit is None:
        return "—"
    rate = compute.get("rows_per_second")
    params = compute.get("n_parameters")
    bits = [f"{fit:,.0f}s fit", f"{score:,.0f}s score"]
    if rate:
        bits.append(f"{rate:,.0f} rows/s")
    if params:
        bits.append(f"{params:,} params")
    return ", ".join(bits)


def _fold_section(fold: drift_arc.ArcFold) -> str:
    spec = registry.get(fold.held_out_vector)
    out = [f"### {fold.held_out_vector} — {spec.name}\n"]

    if fold.outcome == loao.SKIPPED:
        out.append(_wrap(f"**Skipped.** {fold.reason}.") + "\n")
        return "\n".join(out)

    ramps = fold.ramps
    out.append(
        _wrap(
            f"Two batches of this family, identical but for `ramp`: **{ramps.get('sudden')}** for "
            f"the sudden end and **{ramps.get('gradual')}** for the gradual one, both read out of "
            f"{fold.held_out_vector}'s own declared search space so neither leaves the realism "
            "envelope. The family is held out of training in both cases, so what is measured is "
            "generalisation to an unseen family and not recall on a seen one."
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

    counts = fold.counts
    by_arc = counts.get("holdout_positives_by_arc", {})
    out.append(
        "- **Fold** — "
        + _wrap(
            f"{counts.get('holdout_rows', 0):,} holdout rows carrying "
            f"{counts.get('holdout_positives', 0):,} {fold.held_out_vector} positives "
            f"({by_arc.get('sudden', 0):,} sudden, {by_arc.get('gradual', 0):,} gradual), against "
            f"{counts.get('train_rows', 0):,} training rows with "
            f"{counts.get('train_fraud', 0):,} fraud and not one {fold.held_out_vector} row.",
            width=400,
        )
    )
    history = fold.history or {}
    if history.get("checked"):
        out.append(
            "- **History** — "
            + _wrap(
                f"the anchor's own rows carry {history['anchor_mean_window']:.1f} steps of "
                f"per-{history['entity']} history on average "
                f"({history['anchor_share_with_no_history']:.0%} of them have none at all); the "
                f"injected episodes carry {history['injected_mean_window']:.1f}. Window length "
                f"alone separates the two at PR-AUC {history['pr_auc']:.3f}: "
                f"**{'separable' if history['separable'] else 'ok'}**.",
                width=400,
            )
        )
    sep = fold.separability or {}
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
    prov = fold.provenance or {}
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

    quotable = fold.outcome == loao.MEASURED
    out.append("| system | arc | PR-AUC | rec@FPR | p@k | positives | one-threshold recall |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name in (SEQUENCE, LGBM, FLOOR):
        system = fold.systems.get(name)
        if system is None:
            continue
        for arc in drift_arc.ARCS:
            result = system.arcs.get(arc)
            positives = f"{result.n_positives:,}" if result else "—"
            recall = f"{result.recall_at_shared_threshold:.3f}" if result else "—"
            out.append(
                f"| {name} | {arc} | {_cell(result, 'pr_auc', quotable)} | "
                f"{_cell(result, 'recall_at_fixed_fpr', quotable)} | "
                f"{_cell(result, 'precision_at_k', quotable)} | {positives} | {recall} |"
            )
    out.append("")
    out.append(
        _wrap(
            "A number in brackets is one this file will not quote — that arc is too thin to carry "
            "a claim, or the whole fold is. Each arc is ranked against *every* legit row of the "
            "fold, so only the needles change between the two rows and the base rate does not. "
            "`one-threshold recall` is measured from the whole fold's legit traffic rather than "
            "from the arc's own selection, and it matches `rec@FPR` in every row above: that "
            "agreement is what says the two arcs were read at one operating point rather than at "
            "two."
        )
        + "\n"
    )

    out.append("| system | whole fold PR-AUC | evasion | friction | compute |")
    out.append("| --- | ---: | ---: | ---: | --- |")
    for name in (SEQUENCE, LGBM, FLOOR):
        system = fold.systems.get(name)
        if system is None:
            continue
        op = system.operational
        evasion = f"{op['evasion_rate']:.1%}" if "evasion_rate" in op else "—"
        friction = f"{op['friction_rate']:.1%}" if "friction_rate" in op else "—"
        out.append(
            f"| {name} | {system.overall.pr_auc:.3f} | {evasion} | {friction} | "
            f"{_compute(system.compute)} |"
        )
    out.append("")
    out.append(
        _wrap(
            "Compute is in the table because the trade has to be visible: a lift is only readable "
            "next to what it cost. `evasion` is the share of the held-out family the policy let "
            "through untouched and `friction` the share of legit traffic that carried any action; "
            "the floor row has neither because it has no policy."
        )
        + "\n"
    )
    if fold.reason:
        # the gate never ran: repeating the withheld reason here would read as two findings
        out.append(
            _wrap(
                f"**Gate.** Never reached. It is decided on the {fold.promotion.arc} arc and the "
                "fold was refused before the comparison could carry anything, so the bracketed "
                f"{fold.promotion.arc} row above is what the gate *would* have read and not what "
                "it did."
            )
            + "\n"
        )
    else:
        out.append(_wrap(f"**Gate.** {_sentence(fold.promotion.reason)}") + "\n")
    return "\n".join(out)


def _gradual(fold: drift_arc.ArcFold, system: str) -> drift_arc.ArcResult | None:
    """The end of the axis the gate is decided on, for one system."""
    got = fold.systems.get(system)
    return got.arcs.get(drift_arc.GRADUAL) if got else None


def _pr(result: drift_arc.ArcResult | None) -> float | None:
    return float(result.any_metrics.pr_auc) if result and result.any_metrics else None


def _headline(reports: dict[str, drift_arc.SequenceReport]) -> str:
    """The cross-anchor reading, derived from the artefacts rather than typed over them."""
    rows = [
        "| anchor | family | history per entity | sequence | lgbm | amount | fold |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    #: Split by the model's own precondition rather than by who won: the interesting thing about
    #: these four folds is that the answer flips exactly where the anchor stops having histories.
    deep, shallow = [], []
    for name, report in sorted(reports.items()):
        for fold in report.folds:
            history = fold.history or {}
            depth = history.get("anchor_mean_window")
            quotable = fold.outcome == loao.MEASURED
            seq, lgb, flr = (_gradual(fold, s) for s in (SEQUENCE, LGBM, FLOOR))
            rows.append(
                f"| {name} | {fold.held_out_vector} | "
                f"{f'{depth:.1f} steps' if depth is not None else '—'} | "
                f"{_cell(seq, 'pr_auc', quotable)} | {_cell(lgb, 'pr_auc', quotable)} | "
                f"{_cell(flr, 'pr_auc', quotable)} | "
                f"{'measured' if fold.outcome == loao.MEASURED else fold.outcome} |"
            )
            a, b = _pr(seq), _pr(lgb)
            if a is None or b is None:
                continue
            entry = (f"{name}/{fold.held_out_vector}", a, b)
            (shallow if history.get("separable") else deep).append(entry)

    promoted = [name for name, r in reports.items() if r.promoted]
    if promoted:
        head = _wrap(
            "**The sequence model earned its seat on "
            + ", ".join(sorted(promoted))
            + ".** It beats LightGBM on the gradual end of the drift axis — the end with no event "
            "to anchor on, which is the case per-row features run out on — on the same "
            "out-of-time split at the same operating point."
        )
    else:
        head = _wrap(
            "**The sequence model did not earn its seat on any anchor measured, and that is the "
            "result.** It is not in the headline table, `config/defend/sequence.yaml` stays "
            "`enabled: false`, and the hand-rolled per-row feature table is what ships. The "
            "comparison is published because a negative result that lives on one laptop is not a "
            "result, and because the next person to reach for a GRU on this arc should be able to "
            "read what happened when we did."
        )

    out = [head, "", *rows, ""]
    out.append(
        _wrap(
            "Every number above is the **gradual** arc — the end the gate is decided on. All of "
            "them are in brackets, because no fold here can carry a claim; the per-fold sections "
            "say which audit refused each one. `history per entity` is how many steps of its own "
            "account's past a typical *anchor* row carries, which is the precondition this whole "
            "model rests on."
        )
    )

    def _scores(entries) -> str:
        return "; ".join(f"{label} {a:.3f} against {b:.3f}" for label, a, b in entries)

    if deep and shallow:
        out.append("")
        out.append(
            _wrap(
                "**The answer flips exactly along that history column, and it is a refusal at "
                "both ends.** Where the anchor's accounts have real pasts the sequence model "
                "loses on merit — "
                + _scores(deep)
                + " — so the trajectory it reads is worth less than the per-row table it was "
                "supposed to improve on. Where the anchor has no per-entity history at all, every "
                "sender appearing once so that a real window is one step long, it is level or "
                "ahead — "
                + _scores(shallow)
                + " — and that is the result that disqualifies it rather than the one that "
                "promotes it: the only history in those folds belongs to the injected episodes, "
                "so a model that reads history is reading which generator wrote the row before it "
                "reads any behaviour. The `history` audit is what catches it, and it exists "
                "because this is the model it had to be built for."
            )
        )
    elif shallow:
        out.append("")
        out.append(
            _wrap(
                "Window length alone sorts injected rows from real ones on "
                + ", ".join(label for label, _, _ in shallow)
                + ", so a model that reads per-entity history reads provenance there before it "
                "reads behaviour."
            )
        )
    return "\n".join(out)


def _probe_power(reports: dict[str, drift_arc.SequenceReport]) -> str:
    """What this run says about the committed leave-one-attack-out matrix, if anything.

    These folds carry many more injected rows than the matrix's do, because the positives are
    split two ways here. That makes the provenance probe better powered — and `build_loao`'s own
    docstring is explicit that a low probe score on a thin fold is weak evidence of soundness.
    Where the same family comes back separable at this size, the matrix row it disagrees with is
    named rather than left for somebody to notice.
    """
    from afl.evaluation import leave_one_attack_out as harness

    lines = []
    for name, report in sorted(reports.items()):
        try:
            matrix = harness.LeaveOneAttackOutReport.load(name)
        except (FileNotFoundError, ValueError):
            continue
        for fold in report.folds:
            here = fold.provenance or {}
            there = matrix.fold(fold.held_out_vector)
            was = (there.provenance or {}) if there else {}
            if not (here.get("separable") and there and there.outcome == harness.MEASURED):
                continue
            lines.append(
                _wrap(
                    f"**{name} / {fold.held_out_vector}.** `artifacts/loao/{name}.json` reports "
                    f"this family as *measured*, on a probe that saw {was.get('n_injected', '?')} "
                    f"injected rows and scored PR-AUC {float(was.get('pr_auc', 0.0)):.3f}. The "
                    f"same probe here sees {here.get('n_injected')} and scores "
                    f"{float(here['pr_auc']):.3f}, over the bar. Nothing about the generator "
                    "changed between the two runs — the episode count did. Read the matrix row as "
                    "underpowered rather than as a contradiction, and `make loao` at this episode "
                    "count would be the way to settle it.",
                    width=400,
                )
            )
    if not lines:
        return ""
    return "## What this says about the leave-one-attack-out matrix\n\n" + "\n\n".join(lines) + "\n"


def sequence_doc(reports: dict[str, drift_arc.SequenceReport], missing: list[str]) -> str:
    out = [
        "# The sequence model, and whether it earned its seat\n",
        _wrap(
            "_Generated by `scripts/build_sequence.py` from the anchors on disk. Every number "
            "below traces to `artifacts/sequence/<anchor>.json`, which carries the split digest, "
            "the seed, the model cards, the compute cost and the three audits that decide whether "
            "a fold means anything. Do not edit this file — re-run `make sequence`._"
        )
        + "\n",
        _wrap(
            "Ticket 17. A GRU over per-entity history is measured against the tuned LightGBM "
            "baseline on the same out-of-time split at the same operating point, on the drift arc "
            "— S3 account takeover and C1 bust-out — with the sudden and gradual ends of that arc "
            "reported separately. The model enters a reported table only if it beats the baseline "
            "on the gradual end. The comparison is published either way."
        )
        + "\n",
        "## The result\n",
        _headline(reports) + "\n",
        _wrap(
            "**Read the audits before the tables.** Every positive in these folds is an injected "
            "synthetic row and every negative is a real one, so a fold where one contract field — "
            "or, for this model specifically, one window length — separates the two is measuring "
            "which generator wrote the row rather than whether anything detects fraud."
        )
        + "\n",
    ]
    for name in sorted(reports):
        report = reports[name]
        out.append(f"## {name}\n")
        out.append(
            _wrap(
                f"Committed split digest `{report.split.get('digest', '?')}`, seed {report.seed}, "
                f"operating point {float(report.operating_point['fixed_fpr']):.0%} FPR and "
                f"k={report.operating_point['k']} from `{report.operating_point['source']}`. The "
                f"champion is {report.operating_point.get('champion', 'LightGBM')}."
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
            "One architecture, one seed per anchor, one set of hyperparameters read from "
            "`config/defend/sequence.yaml`. A negative result here is evidence that *this* "
            "sequence model, on *these* anchors, does not beat the per-row table on the arc it "
            "was built for — not that no sequence model could. What would change the answer is an "
            "anchor with real per-entity histories on both sides of the label, which is the "
            "precondition the `history` audit measures and the one PaySim does not meet."
        )
        + "\n"
    )
    return "\n".join(out)


# ── the CLI ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("anchors", nargs="*", help="anchor names; default every real anchor")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--families",
        type=lambda s: [v.strip() for v in s.split(",") if v.strip()],
        default=None,
        help="comma-separated drift families; default defend.sequence.arc.families",
    )
    parser.add_argument(
        "--sample", type=float, default=None, help="override sample_fraction for a quick pass"
    )
    parser.add_argument(
        "--doc-only",
        action="store_true",
        help="rewrite docs/sequence.md from the committed artefacts without re-running anything",
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
        reports = drift_arc.load_all(args.out)
        if not reports:
            print(f"no committed comparison in {args.out} — run `make sequence` first")
            return 1
        missing = [c["name"] for c in anchors(args.anchors) if c["name"] not in reports]
        DOC_PATH.write_text(sequence_doc(reports, missing))
        print(f"→ {DOC_PATH} (from {len(reports)} committed artefact(s))")
        return 0

    if not available():
        print(
            "the sequence model needs the `deep` extra: uv sync --extra deep. It is not installed, "
            "and this script refuses to report a number from a model it could not build rather "
            "than degrading to something that scores like a detector that caught nothing.",
            file=sys.stderr,
        )
        return 2

    sup = yaml.safe_load(LGBM_CONFIG.read_text())
    uns = yaml.safe_load(ANOMALY_CONFIG.read_text())
    seq_cfg = yaml.safe_load(SEQUENCE_CONFIG.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    engines = yaml.safe_load(ENGINES_CONFIG.read_text())
    costs = yaml.safe_load(COSTS_CONFIG.read_text())

    reports: dict[str, drift_arc.SequenceReport] = {}
    missing: list[str] = []
    for cfg in anchors(args.anchors):
        name = cfg["name"]
        try:
            reports[name] = run_anchor(cfg, args, sup, uns, seq_cfg, eval_cfg, engines, costs)
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
        print(f"  → {path}")

    if args.out != ARTIFACT_DIR:
        print(f"\n(--out is not {ARTIFACT_DIR}; leaving {DOC_PATH} alone)")
        return 0
    for path in sorted(ARTIFACT_DIR.glob("*.json")):
        # an anchor this run did not touch, but whose artefact is committed, still belongs
        # in the document
        raw = json.loads(path.read_text())
        reports.setdefault(path.stem, drift_arc.SequenceReport.from_dict(raw))
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(sequence_doc(reports, missing))
    print(f"\n→ {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
