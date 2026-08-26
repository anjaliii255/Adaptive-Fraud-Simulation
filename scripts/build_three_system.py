"""The three-system table: real-only vs SMOTE vs the adaptive loop, over seeds, in two columns.

    python scripts/build_three_system.py                    # every real anchor, every seed
    python scripts/build_three_system.py paysim             # just one anchor
    python scripts/build_three_system.py --seeds 1337 7     # two seeds instead of three
    python scripts/build_three_system.py --sample 0.02      # a quick pass, --out somewhere else
    python scripts/build_three_system.py --doc-only         # rewrite the doc from the artefacts

Ticket 16's deliverable, and the slide the project is judged on.

  A  baseline  real rows only                       — what a team has today
  B  smote     real + off-the-shelf oversampling    — what a team tries first
  C  adaptive  real + the attacker-defender loop    — the claim

**System B is why this table can say no.** Row-level oversampling moves an amount and a
timestamp; it cannot invent a new fan-in shape, a new pacing strategy or a beneficiary that never
existed, which is exactly the gap System C claims to fill. If C does not beat B on the held-out
column, the project reduces to an expensive way of duplicating rows — and this file is built so
that outcome is reported rather than re-run until it goes away.

**Two columns, because one is not a result.** `unseen` is the held-out family nobody trained on;
`known` is the fraud every system did train on — the anchor's own labelled rows — scored on the
same window, against the same haystack, at the same operating point. A system that buys the
first by giving up the second has traded rather than improved, and a single-column table cannot
see the trade.

**Several seeds, because one is not a measurement.** Every cell is a mean with its seed-to-seed
spread beside it, every comparison is paired by seed, and a gap smaller than that spread is
reported as inside the noise. Three seeds cannot reach p < 0.05 on a sign test by construction;
that is stated rather than worked around.

**The held-out column inherits ticket 11's verdicts.** The fold here is the fold `make loao`
builds — same pool, same committed boundary, same three guards, same commensurability audit and
provenance probe — so a column that cannot carry a claim there cannot carry one here either. Its
numbers move to `withheld_metrics` and the table prints them in brackets next to the reason.

Everything lands in `artifacts/three_system/<anchor>.json` and in `docs/three_system.md`, which
is generated from those files and never hand-typed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import yaml

# One fold definition, not three. This table is only comparable to the leave-one-attack-out
# matrix if it is measured on the same fold, so the pool, the simulator, the detector factory and
# the fit all come from the files that already own them rather than from copies here.
from build_loao import (  # noqa: E402
    PROBE_FOLDS,
    anchors,
    compose,
    fold_config,
    provenance_probe,
    stats,
    supervised_reference,
    template_gate,
)
from omegaconf import OmegaConf
from run_experiment import (  # noqa: E402
    build_detector_factory,
    build_fit,
    build_pool,
    build_simulator,
)

from afl.attack import envelope as envelope_lib
from afl.attack import multi
from afl.attack.envelope import AnchorEnvelope
from afl.attack.multi import MultiVectorOptimiser
from afl.attack.templates import registry
from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import CommittedSplit, committed_split_for
from afl.defend.decision import assert_one_operating_point
from afl.defend.features import FeatureBuilder
from afl.defend.models.lgbm import DEFAULT_PARAMS, make_estimator, model_card_of
from afl.evaluation import leave_one_attack_out as loao
from afl.evaluation import protocol, three_system
from afl.utils.seed import set_all_seeds

log = logging.getLogger("build_three_system")

DATA_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
ANOMALY_CONFIG = Path("config/defend/anomaly.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
ENGINES_CONFIG = Path("config/attack/engines.yaml")
OPTIMISER_CONFIG = Path("config/attack/optimiser.yaml")
COSTS_CONFIG = Path("config/costs/default.yaml")
EXPERIMENT_CONFIG = Path("config/experiment/adaptive.yaml")
ARTIFACT_DIR = Path("artifacts/three_system")
DOC_PATH = Path("docs/three_system.md")

#: Three seeds, stated rather than tuned. Enough to show the spread; not enough for a sign test
#: to reach significance, which the report says out loud instead of implying otherwise.
DEFAULT_SEEDS = (1337, 7, 21)


def strong_vectors(held_out: str) -> tuple[str, ...]:
    """What System C's optimiser searches: the strong tier, minus the family it is graded on.

    The subtraction is belt-and-braces — `build_pool` already refuses a config whose holdout is
    on the generate list — but System C is the row with a motive, so the guard is stated twice.
    """
    return tuple(
        s.vector_id
        for s in registry.list_vectors(tier="strong", generatable=True)
        if s.vector_id != held_out
    )


def historical_rows(train: list[Transaction], known_vectors: tuple[str, ...]) -> list[Transaction]:
    """What a team already had: real rows, plus any family they are stated to have labels for.

    The same rule that builds the `known` column, applied to the training side — so the column
    every system is measured on is the column every system was trained for.
    """
    return [t for t in train if t.vector_id is None or t.vector_id in known_vectors]


def loop_provenance_probe(
    train_rows: list[Transaction], holdout: list[Transaction], sup: dict, seed: int
) -> dict:
    """The provenance probe, asked with System C's advantage.

    `build_loao`'s probe cross-validates on the holdout's own rows, so it learns "injected" from
    the hundred-odd positives the fold contains. System C learns the same thing from every row
    the loop generated — thousands of them, in the training window — and ticket 11's carry-out
    is explicit that a low probe on a thin fold is weak evidence of anything. This one closes
    that gap: it trains on exactly System C's training set with provenance as the label, and is
    then asked the question System C is scored on.

    It never sees the held-out family. If it still reaches System C's score on that family, the
    fingerprint transfers between families and System C's held-out number is a statement about
    the generator — a model that was told only *who wrote the row* did as well as one that was
    told what fraud is.

    Returns `checked: False` when there is nothing to learn from: a training set with no
    injected rows (Systems A and B) has no fingerprint to find, which is why this is only ever
    applied to the adaptive row.
    """
    injected = [t for t in train_rows if t.vector_id is not None]
    y_test = np.array([int(t.is_fraud) for t in holdout], dtype=int)
    if len(injected) < PROBE_FOLDS or not y_test.sum() or y_test.sum() == y_test.size:
        return {
            "checked": False,
            "reason": "the training set carries no generated rows, or the holdout is single-class",
        }

    features = FeatureBuilder(
        stateful=bool(sup["features"]["stateful"]),
        windows_s=tuple(int(w) for w in sup["features"]["windows_s"]),
    )
    rows = sorted(train_rows, key=lambda t: t.ts)
    # the detector's own order of operations: accumulate state over training, then score the
    # holdout against it without committing. A probe fitted any other way is measuring
    # something the detector never had.
    X_train = features.transform(rows, update=True).to_numpy()
    y_train = np.array([int(t.vector_id is not None) for t in rows], dtype=int)
    scored_rows = sorted(holdout, key=lambda t: t.ts)
    X_test = features.transform(scored_rows, update=False).to_numpy()
    y_test = np.array([int(t.is_fraud) for t in scored_rows], dtype=int)

    model, backend = make_estimator(dict(DEFAULT_PARAMS), seed)
    model.fit(X_train, y_train)
    scored = model.predict_proba(X_test)[:, 1]

    pr_auc = protocol.pr_auc(y_test, scored)
    base_rate = float(y_test.mean())
    return {
        "checked": True,
        "question": "the generator wrote this row — learned from other families, "
        "applied to this one",
        "trained_on": "the loop's own output, which is System C's training set",
        "pr_auc": round(pr_auc, 6),
        "base_rate": round(base_rate, 8),
        "n_rows": len(holdout),
        "n_injected": int(y_test.sum()),
        "n_train_injected": len(injected),
        "n_train_rows": len(rows),
        "backend": str(backend),
        "floor": loao.PROVENANCE_FLOOR,
        "lift": loao.PROVENANCE_LIFT,
        "separable": loao.is_provenance_bound(pr_auc, base_rate),
    }


def _more_damning(*probes: dict) -> dict:
    """The probe that explains more of the number. A checked one always beats an unchecked one.

    Two probes ask the same question with different power, and the honest reading of "can
    provenance explain this row" is the strongest evidence available, not the average of it.
    """
    checked = [p for p in probes if p and p.get("checked")]
    if not checked:
        return next((p for p in probes if p), {})
    return max(checked, key=lambda p: float(p.get("pr_auc", 0.0)))


def measure_system(
    name: str,
    detector,
    fold: loao.Fold,
    known: list[Transaction],
    train: list[Transaction],
    fixed_fpr: float,
    k: int,
    min_positives: int,
    separability: dict | None,
    provenance: dict | None,
    not_reportable: str,
    **extra,
) -> three_system.SystemRow:
    """One system, both columns, judged by the same rules the matrix judges every fold by.

    `loao.run_fold` re-runs the family guard against the *fitted* detector, replay buffer
    included. For System C that is the guard that matters: the loop retrains round after round,
    and a held-out row that walked into the replay buffer would not show up in the split.
    """
    unseen = loao.run_fold(
        fold,
        detector,
        fixed_fpr=fixed_fpr,
        k=k,
        min_positives=min_positives,
        separability=separability,
        provenance=provenance,
        not_reportable=not_reportable,
    )
    known_result = three_system.measure_known_column(
        detector, fold, known, fixed_fpr=fixed_fpr, k=k, min_positives=min_positives
    )
    row = three_system.SystemRow(
        name=name,
        known=known_result,
        unseen=unseen,
        n_train=len(train),
        n_train_fraud=sum(1 for t in train if t.is_fraud),
        model_card=model_card_of(detector),
        **extra,
    )
    log.info(
        "  %-11s unseen %s | known %s",
        name,
        unseen.summary().split(": ", 1)[1],
        known_result.reason or f"PR-AUC {known_result.metrics.pr_auc:.3f}",
    )
    return row


def run_system_c(
    cfg,
    args,
    fold: loao.Fold,
    historical: list[Transaction],
    train_envelope: AnchorEnvelope,
    factory,
    fit,
    seed: int,
) -> tuple[three_system.AdaptiveRun, list[Transaction]]:
    """Run the loop, and hand back the rows the audit gate let through.

    The loop drives a detector of its own — the attacker has to be probing a model that learns,
    or "adaptive" means nothing — but that model is *not* the one the table reports. System C is
    refitted from scratch on `historical + kept`, through the same `fit_detector` as A and B, so
    the only difference between the three rows is which rows they saw. A row whose calibration
    came from a different path is not on the same operating point as its neighbours, whatever
    its metrics say.

    **The attacker only sees the training window.** Its envelope is measured on `historical`
    rather than on the whole anchor, and the audit gate compares candidates against the same
    rows. Both halves of that matter. Generating from a whole-anchor envelope hands the attacker
    accounts that only exist after the split boundary, which is knowledge of the future wearing a
    realism setting; and auditing those rows against the training window alone then rejects them
    for `sender_in_anchor` — on PaySim, where no account transacts twice, that rejected *every*
    batch and made System C a copy of System A for a reason that was an artefact of the harness
    rather than a fact about the loop.
    """
    vectors = strong_vectors(str(cfg.eval.held_out_vector))
    optimiser = MultiVectorOptimiser(
        vectors=vectors,
        seed=seed,
        lambda_realism=float(cfg.attack.optimiser.lambda_realism),
        backend=str(cfg.attack.optimiser.backend),
        allocation=str(args.allocation),
        episodes_per_round=int(args.episodes),
        anchor=historical,
        # `envelope`, not the optimiser's default `lift`: the lift rule has no floor, so on an
        # anchor this size it puts the bar at ~5e-4 PR-AUC and rejects every candidate batch —
        # which would make System C a copy of System A for a reason that is about the gate
        # rather than about the loop. `envelope` is the same verdict the leave-one-attack-out
        # harness applies to the same question. Both verdicts are recorded per round.
        audit_rule=str(args.audit_rule),
    )
    simulator = build_simulator(cfg, anchor=historical, envelope=train_envelope)
    bound = optimiser.bind(simulator)
    three_system.confine_to_training_window(bound, historical)

    # the same factory and the same fit as the three reported rows, so the model the attacker
    # is probing declines exactly what System A would decline. A loop detector on its own cost
    # scale would make "evasion" mean something the table does not report
    loop_detector = factory()
    fit(loop_detector, historical)
    evaluator = None
    if args.curve:
        evaluator = loao.LeaveOneAttackOut(
            holdout=fold.holdout,
            held_out_vector=str(cfg.eval.held_out_vector),
            fixed_fpr=float(cfg.eval.fixed_fpr),
            k=int(cfg.eval.k),
        )
    run = three_system.run_adaptive_loop(
        bound, optimiser, loop_detector, rounds=int(args.rounds), evaluator=evaluator
    )
    log.info(
        "  loop: %d rounds, %d rows kept, %d rejected by the audit gate, evasion %.3f -> %.3f",
        run.rounds,
        len(run.rows),
        run.rejected,
        run.evasion_trajectory[0] if run.history else float("nan"),
        run.evasion_trajectory[-1] if run.history else float("nan"),
    )
    return run, historical + run.rows


def run_seed(
    cfg_data: dict,
    args,
    sup: dict,
    uns: dict,
    eval_cfg: dict,
    engines: dict,
    costs: dict,
    real: list[Transaction],
    envelope: AnchorEnvelope,
    split: CommittedSplit,
    generatable: list[str],
    seed: int,
) -> three_system.SeedRun:
    """One seed: one pool, one carve-out, three systems fitted the same way on it."""
    started = time.perf_counter()
    set_all_seeds(seed)
    held_out = str(args.held_out or eval_cfg["held_out_vector"])
    composed = compose(cfg_data, sup, uns, eval_cfg, engines, costs, seed)
    composed.attack.optimiser = OmegaConf.create(yaml.safe_load(OPTIMISER_CONFIG.read_text()))
    cfg = fold_config(composed, held_out, generatable)

    simulator = build_simulator(cfg, anchor=real, envelope=envelope)
    pool = build_pool(cfg, simulator, real)
    fold = loao.Fold.carve(pool, held_out, split=split)
    known_vectors = tuple(cfg_data.get("known_fraud_vectors") or ())
    known = three_system.known_column(fold, known_vectors)
    haystack = three_system.assert_same_haystack(known, fold.holdout)
    historical = historical_rows(fold.train, known_vectors)

    counts = {
        "pool": stats(pool),
        "train": {
            "rows": len(fold.train),
            "fraud": fold.n_train_fraud,
            "historical_rows": len(historical),
            "historical_fraud": sum(1 for t in historical if t.is_fraud),
        },
        three_system.UNSEEN: three_system.column_counts(fold.holdout),
        three_system.KNOWN: three_system.column_counts(known),
        "columns_share_haystack": haystack,
    }
    log.info(
        "seed %d — pool %d, train %d (%d fraud, %d of them the systems train on), "
        "unseen %d positives, known %d positives",
        seed,
        len(pool),
        len(fold.train),
        fold.n_train_fraud,
        counts["train"]["historical_fraud"],
        counts[three_system.UNSEEN]["positives"],
        counts[three_system.KNOWN]["positives"],
    )
    if not counts["train"]["historical_fraud"]:
        log.warning(
            "System A has no fraud to train on — every row of this table would score 0.0 for "
            "lack of a model rather than for lack of skill. Use a labelled anchor, or name "
            "known families in `known_fraud_vectors`"
        )

    # The red side's audit of the held-out family against this anchor, and the blue side's
    # version of the same question in the feature space the detector sees. Computed once per
    # seed: they are properties of the fold, not of any system in it.
    separability = envelope_lib.audit(real, [t for t in pool if t.vector_id == held_out])
    provenance = provenance_probe(fold, OmegaConf.to_container(cfg.defend.supervised), seed)
    not_reportable = template_gate(registry.get(held_out))

    fixed_fpr, k = float(cfg.eval.fixed_fpr), int(cfg.eval.k)
    min_positives = int(cfg.eval.min_meaningful_positives)
    factory = build_detector_factory(cfg, real)
    fit = build_fit(cfg)
    judged = dict(
        fixed_fpr=fixed_fpr,
        k=k,
        min_positives=min_positives,
        separability=separability,
        provenance=provenance,
        not_reportable=not_reportable,
    )

    # the row names are the module's, not this script's: the artefact, the document and the
    # figure have to agree about what System B is called
    a_name, b_name, c_name = three_system.SYSTEMS
    systems: list[three_system.SystemRow] = []

    # A — what a team has today.
    a = factory()
    fit(a, historical)
    systems.append(measure_system(a_name, a, fold, known, historical, **judged))

    # B — the same rows, oversampled. It can only oversample what System A actually had.
    smote_rows = three_system.smote_transactions(
        historical, ratio=float(args.smote_ratio), seed=seed
    )
    b_train = historical + smote_rows
    b = factory()
    fit(b, b_train)
    systems.append(
        measure_system(b_name, b, fold, known, b_train, n_generated=len(smote_rows), **judged)
    )

    # C — the same starting rows, plus whatever the loop generated and the audit accepted. The
    # attacker's envelope is the training window's, not the anchor's: see `run_system_c`.
    train_envelope = AnchorEnvelope.measure(historical, f"{cfg.data.name}-train")
    run, c_train = run_system_c(cfg, args, fold, historical, train_envelope, factory, fit, seed)
    # The check the fold-level probe is too underpowered to make. System C is the only row that
    # trains on generated rows, so it is the only row whose held-out score can be explained by
    # the generator's fingerprint — and the model that would explain it is fitted here, on
    # exactly System C's training set, with provenance as the label. Whichever of the two probes
    # is more damning is the one that judges this row; both are kept in the artefact.
    loop_probe = loop_provenance_probe(
        c_train, fold.holdout, OmegaConf.to_container(cfg.defend.supervised), seed
    )
    if loop_probe.get("checked"):
        log.info(
            "  provenance-only model on the held-out family: PR-AUC %.4f (%s), separable=%s",
            loop_probe["pr_auc"],
            f"{loop_probe['n_train_injected']:,} generated rows to learn from",
            loop_probe["separable"],
        )
    c_judged = {**judged, "provenance": _more_damning(provenance, loop_probe)}
    c = factory()
    fit(c, c_train)
    systems.append(
        measure_system(
            c_name,
            c,
            fold,
            known,
            c_train,
            n_generated=len(run.rows),
            rounds=run.rounds,
            rejected_rounds=run.rejected,
            loop=run.history,
            **c_judged,
        )
    )

    return three_system.SeedRun(
        seed=seed,
        systems=systems,
        counts=counts,
        guards=fold.guards,
        separability=separability,
        provenance=provenance,
        loop_provenance=loop_probe,
        seconds=time.perf_counter() - started,
    )


def run_anchor(
    cfg_data: dict, args, sup: dict, uns: dict, eval_cfg: dict, engines: dict, costs: dict
) -> three_system.ThreeSystemReport:
    """One anchor's whole table, at an operating point fixed before any of it was measured."""
    name = cfg_data["name"]
    fixed_fpr, k = float(eval_cfg["fixed_fpr"]), int(eval_cfg["k"])
    assert_one_operating_point(
        sup["decision"].get("calibrate_to_fpr"), fixed_fpr, mode=str(sup["decision"]["mode"])
    )
    if args.sample is not None:
        cfg_data = {
            **cfg_data,
            "sample": {**(cfg_data.get("sample") or {}), "sample_fraction": args.sample},
        }

    real = loaders.load_from_config(cfg_data)
    split = committed_split_for(cfg_data)
    if split is None:
        raise SystemExit(f"{name}: no committed split — run `make splits` first")
    # measured once and handed to every seed's simulator: re-measuring the same rows per seed
    # says the same thing three times, slowly, and would make the envelope a function of the seed
    envelope = AnchorEnvelope.measure(real, name)
    generatable = [s.vector_id for s in registry.list_vectors(generatable=True)]
    held_out = str(args.held_out or eval_cfg["held_out_vector"])

    runs = [
        run_seed(
            cfg_data,
            args,
            sup,
            uns,
            eval_cfg,
            engines,
            costs,
            real,
            envelope,
            split,
            generatable,
            seed,
        )
        for seed in args.seeds
    ]

    return three_system.ThreeSystemReport(
        dataset=name,
        held_out_vector=held_out,
        config={
            **eval_cfg,
            "held_out_vector": held_out,
            "source": str(EVAL_CONFIG),
            "seeds": list(args.seeds),
            "rounds": int(args.rounds),
            "episodes_per_round": int(args.episodes),
            "allocation": str(args.allocation),
            "audit_rule": str(args.audit_rule),
            "smote_ratio": float(args.smote_ratio),
            "searched_vectors": list(strong_vectors(held_out)),
            "known_fraud_vectors": list(cfg_data.get("known_fraud_vectors") or ()),
            "sample_fraction": (cfg_data.get("sample") or {}).get("sample_fraction"),
            "convergence_curve": bool(args.curve),
        },
        operating_point={
            "fixed_fpr": fixed_fpr,
            "k": k,
            "source": str(EVAL_CONFIG),
            "min_meaningful_positives": int(eval_cfg["min_meaningful_positives"]),
            "band_units": "calibrated probability",
        },
        runs=runs,
        split=split.to_dict(),
        data={
            "config": f"config/data/{name}.yaml",
            "real": stats(real),
            "supervised_reference": supervised_reference(name),
            "seconds": {r.seed: round(r.seconds, 1) for r in runs},
        },
        meta={
            "generated_by": "scripts/build_three_system.py",
            "note": (
                "The unseen column's positives are all injected synthetic rows and its negatives "
                "are all real, so its numbers report the distance between two distributions as "
                "well as detection — read `outcome` before reading `metrics`. The known column "
                "is the mirror: every positive there is a real labelled fraud row."
            ),
        },
    )


# ── the document ────────────────────────────────────────────────────────────────
def _sentence(text: str) -> str:
    """A reason is written as a clause; promoted to a sentence, it gets its capital back."""
    text = str(text).strip()
    return text[:1].upper() + text[1:] if text else text


def _wrap(text: str, indent: str = "") -> str:
    """Reflow a generated sentence to the width the hand-written prose around it uses."""
    return textwrap.fill(
        " ".join(str(text).split()),
        width=98,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _comparison_rows(report: three_system.ThreeSystemReport) -> str:
    """Every comparison the table owes a reader, in both columns and both metrics."""
    lines = []
    for c in three_system.comparisons(report):
        deltas = ", ".join(f"{d['delta']:+.3f}" for d in c.per_seed)
        reading = "inside the spread" if c.inside_noise else ("beats" if c.beats else "loses")
        if c.outcome != loao.MEASURED:
            reading += ", column withheld"
        lines.append(
            f"| {c.challenger} − {c.incumbent} | {c.column} | {c.metric} | {deltas} | "
            f"{c.mean_delta:+.4f} ± {c.sd_delta:.4f} | {c.wins}/{c.n} | {c.p_value:.3f} | "
            f"{reading} |"
        )
    return "\n".join(lines)


def _per_seed_rows(report: three_system.ThreeSystemReport) -> str:
    lines = []
    for run in report.runs:
        for row in run.systems:
            u, k = row.unseen.any_metrics, row.known.any_metrics
            lines.append(
                f"| {run.seed} | {row.name} | "
                f"{u.pr_auc:.3f} | {u.recall_at_fixed_fpr:.3f} | "
                f"{(f'{k.pr_auc:.3f}' if k else '—')} | "
                f"{(f'{k.recall_at_fixed_fpr:.3f}' if k else '—')} | "
                f"{row.n_train:,} | {row.n_train_fraud:,} | {row.n_generated:,} |"
            )
    return "\n".join(lines)


def _loop_block(report: three_system.ThreeSystemReport) -> str:
    """What System C's attacker actually did, per seed. The row's provenance, not its metrics."""
    rows = [r for r in report.rows_of("C_adaptive") if r.loop]
    if not rows:
        return "_System C did not run on this anchor._"
    lines = []
    for run, row in zip(report.runs, rows, strict=False):
        traj = [float(h.get("evasion_rate", 0.0)) for h in row.loop]
        # what the rule that was NOT in force would have done, from the same audit report
        by_lift = sum(1 for h in row.loop if h.get("rejected_by_lift"))
        by_envelope = sum(1 for h in row.loop if h.get("rejected_by_envelope"))
        lines.append(
            f"| {run.seed} | {row.rounds} | {row.rejected_rounds} | {by_envelope} / {by_lift} | "
            f"{row.n_generated:,} | {traj[0]:.3f} → {traj[-1]:.3f} | "
            f"{min(traj):.3f} / {max(traj):.3f} |"
        )
    return (
        "| seed | rounds | rejected | would reject: envelope / lift | rows kept | "
        "evasion first → last | min / max |\n"
        "| --- | ---: | ---: | ---: | ---: | --- | --- |\n" + "\n".join(lines)
    )


def _provenance_only_block(report: three_system.ThreeSystemReport) -> str:
    """What a model that knows only *who wrote the row* scores on the same column.

    System C is the only row trained on generated rows, so it is the only row whose held-out
    score can be the generator's fingerprint rather than detection. The fold's own probe cannot
    settle that — it learns "injected" from the holdout's handful of positives — so the
    counterfactual is fitted on System C's training set instead, and printed here whatever it
    says.
    """
    probes = [r.loop_provenance for r in report.runs if (r.loop_provenance or {}).get("checked")]
    if not probes:
        return (
            "**Does provenance explain System C?** The probe did not run: "
            f"{(report.runs[0].loop_provenance or {}).get('reason', 'not attempted')}."
        )
    rows = report.rows_of("C_adaptive")
    scores = [float(p["pr_auc"]) for p in probes]
    detector = [
        float(r.unseen.any_metrics.pr_auc) for r in rows if r.unseen.any_metrics is not None
    ]
    mean_probe = sum(scores) / len(scores)
    mean_detector = sum(detector) / len(detector) if detector else float("nan")
    verdict = (
        "provenance alone matches or beats System C, so System C's held-out number is the "
        "generator's fingerprint transferring between families rather than a detector "
        "generalising to an unseen attack"
        if mean_probe >= mean_detector - three_system.PROVENANCE_MARGIN
        else "provenance alone falls short of System C, so it does not by itself account for "
        "the gap between System C and the two controls"
    )
    return _wrap(
        f"""**Does provenance explain System C?** A model given the same training rows as System C
and told only *which rows the generator wrote* — never which rows are fraud, and never a row of
the held-out family — scores PR-AUC **{mean_probe:.3f}** on the held-out column against System
C's **{mean_detector:.3f}** ({probes[0]["n_train_injected"]:,} generated rows to learn from,
separable: **{probes[0]["separable"]}**). On this anchor {verdict}."""
    )


def _fold_block(report: three_system.ThreeSystemReport) -> str:
    """The carve-out these three systems share, and the guards it passed."""
    run = report.runs[0]
    counts, guards = run.counts, run.guards
    unseen, known = counts.get(three_system.UNSEEN, {}), counts.get(three_system.KNOWN, {})
    family = guards.get("family", {})
    embargo = guards.get("embargo", {})
    haystack = guards.get("haystack", {})
    sep = run.separability or {}
    probe = run.provenance or {}

    probe_line = (
        _wrap(
            f"""**The provenance probe.** A classifier given the fold's own features sorts the
injected `{report.held_out_vector}` rows from this anchor's own traffic at PR-AUC
**{float(probe["pr_auc"]):.3f}** ({probe["cv_folds"]}-fold cross-validated,
{probe["n_injected"]:,} injected rows in {probe["n_rows"]:,}) — separable:
**{probe["separable"]}**. Every positive in the unseen column is injected and every negative is
real, so a probe at or above the detector's own score means that column cannot tell detection
apart from provenance."""
        )
        if probe.get("checked")
        else f"**The provenance probe.** Did not run: {probe.get('reason', 'not attempted')}."
    )

    train = counts.get("train", {})
    fold_line = _wrap(
        f"""**The fold.** All three systems train on {train.get("historical_rows", 0):,}
rows ({train.get("historical_fraud", 0):,} fraud) out of a
{counts.get("pool", {}).get("rows", 0):,}-row pool, and are scored on one test window:
**unseen** {unseen.get("rows", 0):,} rows / {unseen.get("positives", 0):,} positives
(base rate {float(unseen.get("base_rate", 0.0)):.4%}, all injected), **known**
{known.get("rows", 0):,} rows / {known.get("positives", 0):,} positives
({known.get("positives_anchor_own", 0):,} of them the anchor's own labelled fraud).
Committed boundary `{report.split.get("digest")}`, embargo
{report.split.get("embargo_seconds", 0):,}s, sample fraction
{report.config.get("sample_fraction")}."""
    )
    return f"""{fold_line}

| guard | what it checked | result |
| --- | --- | ---: |
| family carve-out | {family.get("audited", "—")} | {family.get("rows_checked", 0):,} rows, \
{family.get("leaked_rows", 0)} leaked |
| out-of-time embargo | committed gap {embargo.get("embargo_seconds", 0):,}s | \
actual gap {embargo.get("gap_seconds", 0):,}s |
| haystack | legit rows in the test window | \
{haystack.get("legit_rows_kept", 0):,} of {haystack.get("legit_rows_in_window", 0):,} kept |
| one haystack, two columns | legit rows shared by both columns | \
{counts.get("columns_share_haystack", {}).get("legit_rows", 0):,} shared |

{_wrap(f'''**One contract field.** The worst single field on this anchor's
`{report.held_out_vector}` rows is `{sep.get("worst")}`, at PR-AUC
{float(sep.get("score", 0.0)):.4f} against a base rate of {float(sep.get("base_rate", 0.0)):.4f}
— trivially separable: **{sep.get("trivially_separable")}**.''') if sep else ""}

{probe_line}
"""


def _anchor_section(report: three_system.ThreeSystemReport) -> str:
    nl = chr(10)
    head = three_system.compare(report)
    head_auc = three_system.compare(report, column=three_system.UNSEEN, metric="pr_auc")
    verdict = _wrap(f"**{_sentence(head.verdict)}.**") if head else "**No comparison ran.**"
    if head_auc is not None:
        verdict += nl + nl + _wrap(f"On PR-AUC in the same column: {head_auc.verdict}.")

    findings = three_system.diagnose(report)
    findings_block = (
        nl.join(
            _wrap(
                f"- **{_sentence(f['finding'])}.** "
                f"Evidence: `{json.dumps(f['evidence'], ensure_ascii=False)}`."
            )
            .replace(nl, nl + "  ")
            .replace(nl + "  - ", nl + "- ")
            for f in findings
        )
        if findings
        else "_Nothing in the run's own logs argues against the table above._"
    )

    searched = ", ".join(f"`{v}`" for v in report.config.get("searched_vectors", []))
    loop_intro = _wrap(
        f"""**System C's loop.** {report.config.get("rounds")} rounds of
{report.config.get("episodes_per_round")} episodes, searching {searched} with the
`{report.config.get("allocation")}` allocation. Every candidate batch is audit-gated on the
`{report.config.get("audit_rule")}` rule before the detector is allowed to train on it, and a
rejected batch is discarded rather than learned from — see `afl/attack/multi.py:AUDIT_RULES`
for what the two rules disagree about."""
    )
    runtime = ", ".join(
        f"seed {seed} {float(value):,.0f}s"
        for seed, value in (report.data.get("seconds") or {}).items()
    )
    ref = report.data.get("supervised_reference") or {}
    ref_line = (
        _wrap(
            f"""**Against this anchor's own fraud.** The committed baseline reaches PR-AUC
{float(ref["pr_auc"]):.3f} on {report.dataset}'s own labelled fraud in the same test window
({ref["n_positives"]:,} real positives, `{ref["source"]}`), where sorting by amount alone reaches
{float(ref["amount_floor_pr_auc"]):.3f}. Any row above that line on the unseen column is easier
unseen than seen, which is a statement about the injected rows rather than a generalisation
result."""
        )
        if ref
        else ""
    )

    return f"""### {report.dataset}

{verdict}

{three_system.table_markdown(report)}

{_fold_block(report)}
{ref_line}

{loop_intro}

{_loop_block(report)}

{_provenance_only_block(report)}

**Every comparison, paired by seed.**

| comparison | column | metric | per-seed Δ | mean ± sd | wins | sign-test p | reading |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
{_comparison_rows(report)}

**Why it landed there.**

{findings_block}

**Every seed.**

| seed | system | unseen PR-AUC | unseen rec@FPR | known PR-AUC | known rec@FPR | train rows | \
train fraud | generated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{_per_seed_rows(report)}

_Runtime: {runtime}._
"""


def three_system_doc(reports: dict[str, three_system.ThreeSystemReport], missing: list[str]) -> str:
    """The table, generated from the artefacts that produced it."""
    nl = chr(10)
    first = next(iter(reports.values()))
    op = first.operating_point
    sections = nl.join(_anchor_section(r) for r in reports.values())

    verdicts = []
    for report in reports.values():
        head = three_system.compare(report)
        if head is None:
            continue
        verdicts.append(_wrap(f"- **{report.dataset}** — {head.verdict}.", indent="  ").lstrip())
    absent = (
        _wrap(
            "**Not measured.** "
            + " ".join(f"`{n}` is not downloaded, so it has no table here." for n in missing)
        )
        if missing
        else "_Every configured anchor was on disk when this ran._"
    )
    quotable = sum(
        1
        for r in reports.values()
        for s in r.systems
        if three_system.spread_of(r, s, three_system.UNSEEN).reported
    )
    total = sum(len(r.systems) for r in reports.values())

    return f"""# The three-system table

_Generated by `scripts/build_three_system.py` from the anchors on disk. Every number below
traces to `artifacts/three_system/<anchor>.json`, which carries the eval config, the committed
split digest and every seed that produced it. Do not edit this file — re-run `make table`._

Three systems, one holdout, one operating point:

| | system | trained on | what it stands for |
| --- | --- | --- | --- |
| **A** | `A_baseline` | the anchor's real rows and real labels | what a team has today |
| **B** | `B_smote` | the same rows plus row-level oversampling | what a team tries first |
| **C** | `C_adaptive` | the same rows plus the attacker–defender loop | the claim |

**System B is here to make System C falsifiable.** Row-level oversampling can move an amount and
a timestamp; it cannot invent a new fan-in shape, a new pacing strategy, or a beneficiary that
never existed — which is precisely the gap the adaptive system claims to fill. If C does not beat
B on the held-out column, the project reduces to an expensive way of duplicating rows, and this
is the table where that shows up. **It is built to be able to say so.**

**Operating point: recall at {float(op["fixed_fpr"]):.0%} FPR, precision@{int(op["k"])}**, fixed
in `config/eval/leave_one_attack_out.yaml` before any of these numbers existed, and applied
identically to all three rows — including where the action bands sit, because `evasion_rate` in
the same table is a function of them.

## Two columns, because one is not a result

- **unseen** — the held-out family (`{first.held_out_vector}`), which no system trained on. This
  is the claim. It is the same carve-out `make loao` builds: same pool, same committed boundary,
  same three guards, same commensurability audit and provenance probe.
- **known** — the fraud every system *did* train on: the anchor's own labelled rows, scored on
  the same window against the same legit haystack. This is the price of the claim. A system that
  buys the unseen column by giving up the known one has traded rather than improved.

**These are not `docs/loao.md`'s numbers, and they are not meant to be.** The fold is the same
one — same pool, same boundary, same guards — but the *training set* is the whole point of this
table: System A sees the anchor's real rows alone, where the matrix's detector trains on the
entire training side, injected families included. A row here and a row there are answers to
different questions and do not belong in one comparison.

The two columns share their negatives, asserted rather than assumed
(`three_system.assert_same_haystack`): recall at a fixed FPR is a quantile of the negatives, so
two haystacks would be two operating points wearing one table. Families the pool carries but
nobody trained on appear in neither column — they are neither the claim nor the control, and
counting them as negatives would label real fraud as legit traffic.

## How a difference becomes a result

Every cell is a mean over seeds with its seed-to-seed spread beside it, and every comparison is
**paired by seed**: both systems see the same fold, the same pool and the same fitted procedure
on a given seed, so the difference between them is the only thing the seed does not also move.

The seed moves the whole pipeline, not just the fit — the attack episodes that go into the pool,
the SMOTE draw, the optimiser's search and the model's own randomness all turn with it. That is
deliberate: the question a reader has is whether the *system* is better, and a spread measured
over refits alone would answer a narrower one.

A gap smaller than its own spread is reported as **inside the noise**, whichever way it points.
The sign test counts which way each seed fell; over
{len(first.config.get("seeds", []))} seed(s) it cannot reach p < 0.05 by construction, and that
is stated rather than worked around.

**The unseen column inherits ticket 11's verdicts.** Where that harness withholds a fold — too
few positives, separable from the anchor by one contract field, separable by a whole model, or a
`template` vector whose defining tell is not modelled yet — the numbers move out of `metrics`
into `withheld_metrics` and are printed **in brackets** next to the reason. {quotable} of {total}
system-columns across {len(reports)} anchor(s) carry a quotable unseen number.

## The verdict

{nl.join(verdicts) if verdicts else "_No anchor produced a comparison._"}

{absent}

## Per anchor

{sections}
## Where the numbers came from

Every cell here is read out of `artifacts/three_system/<anchor>.json` — the eval config as it was
read, the committed split digest, the seeds, per-column row counts, all four guards, the
commensurability audit, the provenance probe, the loop's per-round history and each system's
model card. Nothing in this document is typed by hand, and nothing in it can be refreshed without
re-running the command that produced the artefacts:

```bash
make table                                     # every anchor, every seed
python scripts/build_three_system.py paysim --seeds 1337 7
python scripts/build_three_system.py --doc-only    # rebuild this file from the artefacts
```
"""


# ── the run ─────────────────────────────────────────────────────────────────────
def print_report(report: three_system.ThreeSystemReport, path: Path) -> None:
    name = report.dataset
    print(f"\n── {name} " + "─" * (70 - len(name)))
    print(three_system.table_markdown(report))
    for c in (
        three_system.compare(report),
        three_system.compare(report, challenger="C_adaptive", incumbent="A_baseline"),
    ):
        if c is not None:
            print(f"\n  {c.verdict}")
    for finding in three_system.diagnose(report):
        print("  - " + _wrap(f"{finding['finding']}.", indent="    ").strip())
    print(f"\n  → {path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="dataset names; default: all of them")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="seeds to repeat the whole table on; the spread across them is reported",
    )
    parser.add_argument(
        "--held-out", default=None, help="the family to hold out; default: eval.held_out_vector"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="System C's loop rounds; default: the committed `rounds` in "
        "config/attack/optimiser.yaml, so the table runs the loop the config declares",
    )
    parser.add_argument(
        "--episodes", type=int, default=12, help="attack episodes per loop round, across vectors"
    )
    parser.add_argument(
        "--allocation",
        default="search",
        help="how the loop splits its budget: uniform|search|fitness",
    )
    parser.add_argument("--smote-ratio", type=float, default=1.0, help="System B's oversampling")
    parser.add_argument(
        "--audit-rule",
        default="envelope",
        choices=list(multi.AUDIT_RULES),
        help="which commensurability verdict gates the loop's batches; both are recorded either "
        "way. `lift` is the optimiser's own default and rejects 100%% of candidates on a real "
        "anchor, so this table runs on `envelope` — see afl/attack/multi.py:AUDIT_RULES",
    )
    parser.add_argument("--sample", type=float, default=None, help="override the entity sample")
    parser.add_argument(
        "--curve",
        action="store_true",
        help="score the held-out column every round, for the convergence curve. Off by default: "
        "it is a full scoring pass per round, and ticket 19 owns that artefact",
    )
    parser.add_argument(
        "--doc-only",
        action="store_true",
        help="rewrite docs/three_system.md from the committed artefacts without re-running the "
        "table; the document is a pure function of them, so this cannot disagree with a run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_DIR,
        help="where the artefacts go; point it elsewhere for a trial run so a quick pass cannot "
        "overwrite the committed table",
    )
    args = parser.parse_args()

    sup = yaml.safe_load(LGBM_CONFIG.read_text())
    uns = yaml.safe_load(ANOMALY_CONFIG.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    engines = yaml.safe_load(ENGINES_CONFIG.read_text())
    costs = yaml.safe_load(COSTS_CONFIG.read_text())
    if args.rounds is None:
        args.rounds = int(yaml.safe_load(OPTIMISER_CONFIG.read_text())["rounds"])

    reports: dict[str, three_system.ThreeSystemReport] = {}
    missing: list[str] = []
    if args.doc_only:
        reports = three_system.load_all(args.out)
        missing = [c["name"] for c in anchors(args.datasets) if c["name"] not in reports]
        if not reports:
            print(f"no committed table in {args.out} — run `make table` first", file=sys.stderr)
            return 1
        # a trial run's artefacts write a trial run's document; only the committed directory
        # is allowed to rewrite the committed one
        path = DOC_PATH if args.out == ARTIFACT_DIR else args.out / DOC_PATH.name
        path.write_text(three_system_doc(reports, missing))
        print(f"→ {path} (from {len(reports)} committed artefact(s))")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for cfg_data in anchors(args.datasets):
        try:
            report = run_anchor(cfg_data, args, sup, uns, eval_cfg, engines, costs)
        except loaders.DatasetNotDownloaded as exc:
            missing.append(cfg_data["name"])
            print(f"SKIPPED {cfg_data['name']}: {exc}", file=sys.stderr)
            continue
        # saved as soon as it exists, not after every anchor has finished: an anchor costs
        # tens of minutes, and a failure on the next one must not throw away the last one
        reports[cfg_data["name"]] = report
        print_report(report, report.save(args.out))

    if not reports:
        print("no anchor on disk — there is nothing to build a table from", file=sys.stderr)
        return 1

    if args.out != ARTIFACT_DIR:
        print(f"\n(--out is not {ARTIFACT_DIR}; leaving {DOC_PATH} alone)")
        return 0
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(three_system_doc(reports, missing))
    print(f"\n→ {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
