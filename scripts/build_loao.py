"""The leave-one-attack-out matrix: every family held out in turn, under guards that raise.

    python scripts/build_loao.py                     # every real anchor, every fold
    python scripts/build_loao.py paysim              # just one anchor
    python scripts/build_loao.py --folds M3,C2       # just two folds
    python scripts/build_loao.py --sample 0.02       # a quick pass
    python scripts/build_loao.py --doc-only          # rewrite the doc from committed artefacts

Ticket 11's deliverable, and it is the evaluation everything after this is measured through.
Reporting recall on a family the model trained on measures memorisation; the claim is
generalisation to an unseen attack, so the carve-out has to be airtight and the fold has to say
when it is not carrying a claim at all.

**Three guards, all of them assertions rather than intentions.** Not one row of the held-out
family reaches training — the detector's own replay buffer included, audited on the fitted model
rather than on the list handed to `fit`. The split is still out-of-time with the committed
embargo intact after the carve-out. Every legit row of the test window is still in the holdout,
because an FPR measured without negatives is not an FPR. Each one raises, and each one has a
test that deliberately tries to leak a row past it.

**A fold that runs is not a fold that means something,** and this file is where that stops being
a caveat in a doc. Three earlier results — ticket 07's carry-out, the transfer test and ticket
10 — all landed on the same finding from different directions: on a real anchor every positive
in this fold is an injected synthetic row and every negative is a real one, so a fold can report
a confident number while measuring which generator wrote the row. So a fold's numbers are
`measured` only if it clears all of:

  * enough positives to move a metric by less than a rounding error per row;
  * not separable from the anchor by a single contract field (`afl.attack.envelope.audit`);
  * not separable from the anchor by a *model* either — a classifier two-sample test on the
    fold's own feature space, which is the check ticket 07's carry-out asked this ticket for;
  * a family this repo can actually generate today, tell and all (`registry.reportable`).

Anything else is `withheld` — the numbers exist, under `withheld_metrics`, never under
`metrics` — or `skipped`, with the reason in the row where the number would have been. Every
requested fold gets a row either way: a fold that vanishes from the table reads as "not
applicable" when it means "we did not look".

The amount floor rides along on every fold. Two results in this repo were walked back for want
of it, and a held-out-family recall that loses to sorting by amount has not detected anything.

Everything lands in `artifacts/loao/<anchor>.json` and in `docs/loao.md`, which is generated
from those files and never hand-typed.
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
from omegaconf import OmegaConf

# One fold definition, not two. The matrix is only comparable to the loop's headline number if
# it is the same fold the loop runs on, so the simulator and the pool come from `run_experiment`
# itself rather than from a re-implementation here that would drift the first time either was
# edited.
from run_experiment import (  # noqa: E402
    build_detector_factory,
    build_fit,
    build_pool,
    build_simulator,
)

from afl.attack import envelope as envelope_lib
from afl.attack.envelope import AnchorEnvelope
from afl.attack.templates import registry
from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import CommittedSplit, committed_split_for
from afl.defend.decision import assert_one_operating_point
from afl.defend.features import FeatureBuilder
from afl.defend.models.lgbm import DEFAULT_PARAMS, make_estimator, model_card_of
from afl.evaluation import leave_one_attack_out as loao
from afl.evaluation import protocol
from afl.utils.seed import set_all_seeds

log = logging.getLogger("build_loao")

DATA_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
ANOMALY_CONFIG = Path("config/defend/anomaly.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
ENGINES_CONFIG = Path("config/attack/engines.yaml")
COSTS_CONFIG = Path("config/costs/default.yaml")
DETECTOR_DIR = Path("artifacts/detector")
ARTIFACT_DIR = Path("artifacts/loao")
DOC_PATH = Path("docs/loao.md")

#: Folds for the provenance probe's own cross-validation. Every row gets an out-of-sample score,
#: so the probe never reports a number it fitted on and never spends half the injected rows on a
#: split. Three is enough at these counts and costs three fits.
PROBE_FOLDS = 3


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


def supervised_reference(name: str) -> dict | None:
    """What the same detector scores on this anchor's OWN labelled fraud, in the same window.

    The number a held-out-family score has to be read against. Ticket 10 put the reading
    plainly: an unseen family that is *easier* than the seen one is not a generalisation result,
    it is a statement about the injected rows. A fold that clears its anchor's own fraud by a
    wide margin is saying something about the generator, whatever the fold's outcome says.

    Read from `make baseline`'s committed artefact rather than recomputed, so the two numbers
    cannot drift apart between runs.
    """
    path = DETECTOR_DIR / f"{name}.json"
    if not path.exists():
        return None
    tuned = json.loads(path.read_text())["metrics"]["tuned"]
    return {
        "source": str(path),
        "measured_on": "this anchor's OWN labelled fraud, out of time at the same boundary",
        "pr_auc": tuned["pr_auc"],
        "recall_at_fixed_fpr": tuned["recall_at_fixed_fpr"],
        "precision_at_k": tuned["precision_at_k"],
        "n_positives": tuned["n_positives"],
        "amount_floor_pr_auc": json.loads(path.read_text())["metrics"]["amount_only"]["pr_auc"],
    }


def requested_folds(eval_cfg: dict, override: str | None) -> list[str]:
    """Which families to hold out, one at a time.

    `auto` is every vector the registry knows about — including the ones it cannot generate yet,
    because a fold that is absent from the matrix and a fold that is impossible read identically
    once the table is printed. The impossible ones become `skipped` rows naming their ticket.
    """
    if override:
        return [v.strip() for v in override.split(",") if v.strip()]
    configured = eval_cfg.get("folds", "auto")
    if isinstance(configured, str) and configured.lower() == "auto":
        return [spec.vector_id for spec in registry.list_vectors()]
    return [str(v) for v in configured]


def compose(cfg: dict, sup: dict, uns: dict, eval_cfg: dict, engines: dict, costs: dict, seed: int):
    """The same config tree `config/config.yaml` composes, built here from the same files."""
    return OmegaConf.create(
        {
            "seed": seed,
            "data": cfg,
            "attack": {"engines": engines},
            "defend": {"supervised": sup, "unsupervised": uns},
            "eval": eval_cfg,
            "costs": costs,
        }
    )


def fold_config(composed, held_out: str, generatable: list[str]):
    """The composed config as this fold sees it: one family held out, the rest generated.

    `build_pool` refuses a config whose holdout is also on the generate list, which is the guard
    that stops the red side handing the blue side the answer. Every fold therefore gets its own
    config rather than a shared one with a mutated field.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(composed, resolve=True))
    cfg.eval.held_out_vector = held_out
    cfg.attack.engines.vectors = [v for v in generatable if v != held_out]
    return cfg


def provenance_probe(fold: loao.Fold, sup: dict, seed: int) -> dict:
    """Can a classifier sort the injected rows from the anchor's own, on the fold's own features?

    Ticket 07 measured this ad hoc and got AUC 1.00 on PaySim, and it has been quoted from a
    carry-out ever since. It belongs in the harness, because the thing it measures is the fold's
    own validity: the carve-out drops the anchor's real fraud from the holdout, so in this fold
    **every positive is an injected row and every negative is a real one** — "caught the fraud"
    and "spotted the synthetic row" are literally the same label. If a model separates them
    easily, the detector's recall on this fold is a statement about the generator, and the only
    honest thing to do with the number is withhold it.

    Deliberately independent of the detector: its own `FeatureBuilder`, its own fit, the stock
    params. A probe that borrowed the detector's fitted feature state would be measuring the
    detector, which is the number this is supposed to be a check on.

    **Cross-validated, not split out of time,** and it is the one measurement in this repo where
    that is the right call. Everywhere else the question is whether a model generalises forward,
    and a random split answers it dishonestly. Here the question is whether two populations are
    distinguishable at all — a classifier two-sample test — which has no time direction in it.
    A chronological split also needs the injected episodes to span the holdout window, and on
    PaySim they do not: the first attempt at this returned "one side is single-class", which is
    an answer of nothing wearing the shape of a pass.

    **The probe is underpowered on a thin fold, and the asymmetry runs the wrong way.** It learns
    "injected" from the fold's own positives — 27 of them per cross-validation fold when the
    holdout carries 80 — while the detector it is checking learned from a whole training window.
    So a low probe score on a thin fold is weak evidence that the fold is sound, where a high one
    is strong evidence that it is not. `n_injected` travels with the score for that reason, and
    the thin-fold floor is applied first so that the weakest probes belong to folds that were
    already withheld.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    rows = sorted(fold.holdout, key=lambda t: t.ts)
    y = np.array([int(t.is_fraud) for t in rows], dtype=int)
    injected = np.array([int(t.attack_run_id is not None) for t in rows], dtype=int)
    if y.sum() < PROBE_FOLDS or (y == 0).sum() < PROBE_FOLDS:
        return {"checked": False, "reason": "too few rows on one side to cross-validate a probe"}

    features = FeatureBuilder(
        stateful=bool(sup["features"]["stateful"]),
        windows_s=tuple(int(w) for w in sup["features"]["windows_s"]),
    )
    X = features.transform(rows, update=True).to_numpy()
    model, backend = make_estimator(dict(DEFAULT_PARAMS), seed)
    scored = cross_val_predict(
        model,
        X,
        y,
        cv=StratifiedKFold(PROBE_FOLDS, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]

    pr_auc = protocol.pr_auc(y, scored)
    base_rate = float(y.mean())
    return {
        "checked": True,
        "question": "injected row, or the anchor's own?",
        # the fold drops the anchor's own fraud from the holdout, so these two labels are the
        # same one. If they ever stop being, this fold's whole argument changes and says so here.
        "label_is_exactly_is_fraud": bool(np.array_equal(y, injected)),
        "pr_auc": round(pr_auc, 6),
        "base_rate": round(base_rate, 8),
        "n_rows": len(rows),
        "n_injected": int(y.sum()),
        "cv_folds": PROBE_FOLDS,
        "backend": str(backend),
        "floor": loao.PROVENANCE_FLOOR,
        "lift": loao.PROVENANCE_LIFT,
        "separable": loao.is_provenance_bound(pr_auc, base_rate),
    }


def template_gate(spec) -> str:
    """Why a `template` vector's numbers cannot be claimed for its family, in one sentence.

    Shared with `scripts/build_three_system.py`: the same gate decides the same thing in the
    matrix and in the hero table, and two copies of this sentence would be two gates the first
    time either was edited.
    """
    if spec.reportable:
        return ""
    return (
        f"{spec.vector_id} is a `{spec.status}` vector: it emits schema-valid traffic of the "
        f"right shape, but its defining tell is not modelled yet — "
        f"{' '.join(spec.gap.split()).rstrip('.')}. The number in this row measures the "
        "pipeline, not the family"
    )


def run_fold(
    composed,
    held_out: str,
    real: list[Transaction],
    envelope: AnchorEnvelope,
    split: CommittedSplit,
    generatable: list[str],
    seed: int,
    min_positives: int,
) -> tuple[loao.FoldResult, dict]:
    """One family held out end to end: generate, carve, guard, fit, score, judge.

    The pool is rebuilt per fold rather than carved out of one shared pool. It costs a
    regeneration, and it buys the thing the shared pool cannot give: this fold's family gets
    `eval.holdout_episodes` episodes instead of the five a background family gets, so the
    out-of-time cut has something to land on. A holdout that is empty by chance reports 0.0
    everywhere and looks exactly like a detector that caught nothing.
    """
    spec = registry.get(held_out)
    if not spec.generatable:
        return (
            loao.FoldResult.skipped(
                held_out,
                f"{held_out} ({spec.name}) is declared but not implemented — {spec.gap}",
            ),
            {},
        )

    set_all_seeds(seed)
    cfg = fold_config(composed, held_out, generatable)
    simulator = build_simulator(cfg, anchor=real, envelope=envelope)
    pool = build_pool(cfg, simulator, real)

    fold = loao.Fold.carve(pool, held_out, split=split)
    counts = fold.counts()
    log.info(
        "%s / %s: pool %d -> train %d (%d fraud, no %s) / holdout %d (%d positives)",
        cfg.data.name,
        held_out,
        len(pool),
        counts["train_rows"],
        counts["train_fraud"],
        held_out,
        counts["holdout_rows"],
        counts["holdout_positives"],
    )
    if not fold.n_train_fraud:
        return (
            loao.FoldResult.skipped(
                held_out,
                "the carve-out left no fraud in the training window — a single-class fit is not "
                "a detector, so there is nothing to measure this family against",
                counts=counts,
                guards=fold.guards,
            ),
            {},
        )

    started = time.perf_counter()
    detector = build_detector_factory(cfg, real)()
    build_fit(cfg)(detector, fold.train)
    fit_seconds = time.perf_counter() - started

    # The red side's own audit of this family against this anchor. Passed in rather than
    # computed inside the harness: whether the generator is distinguishable from the traffic it
    # was injected into is a question about the generator, and the blue side never learns it.
    separability = envelope_lib.audit(real, [t for t in pool if t.vector_id == held_out])
    # and the blue side's version of the same question, asked in the feature space the detector
    # actually sees. One contract field is the cheap check; this is the one ticket 07 asked for.
    provenance = provenance_probe(fold, OmegaConf.to_container(cfg.defend.supervised), seed)

    result = loao.run_fold(
        fold,
        detector,
        fixed_fpr=float(cfg.eval.fixed_fpr),
        k=int(cfg.eval.k),
        min_positives=min_positives,
        separability=separability,
        provenance=provenance,
        not_reportable=template_gate(spec),
    )
    log.info("%s / %s: %s", cfg.data.name, held_out, result.summary())
    return result, {
        "model_card": model_card_of(detector),
        "pool": stats(pool),
        "fit_seconds": round(fit_seconds, 2),
    }


def run_anchor(cfg: dict, args, sup: dict, uns: dict, eval_cfg: dict, engines: dict, costs: dict):
    """One anchor's whole matrix, at one operating point fixed before any of it was measured."""
    name = cfg["name"]
    seed = int(args.seed)
    set_all_seeds(seed)
    fixed_fpr, k = float(eval_cfg["fixed_fpr"]), int(eval_cfg["k"])
    min_positives = int(eval_cfg["min_meaningful_positives"])
    assert_one_operating_point(
        sup["decision"].get("calibrate_to_fpr"), fixed_fpr, mode=str(sup["decision"]["mode"])
    )

    if args.sample is not None:
        cfg = {**cfg, "sample": {**(cfg.get("sample") or {}), "sample_fraction": args.sample}}
    real = loaders.load_from_config(cfg)

    split = committed_split_for(cfg)
    if split is None:
        raise SystemExit(f"{name}: no committed split — run `make splits` first")

    composed = compose(cfg, sup, uns, eval_cfg, engines, costs, seed)
    # measured once and handed to every fold's simulator: nine folds re-measuring the same 600k
    # rows would say the same thing nine times, slowly
    envelope = AnchorEnvelope.measure(real, name)
    generatable = [s.vector_id for s in registry.list_vectors(generatable=True)]

    folds: list[loao.FoldResult] = []
    extras: dict[str, dict] = {}
    for held_out in requested_folds(eval_cfg, args.folds):
        result, extra = run_fold(
            composed, held_out, real, envelope, split, generatable, seed, min_positives
        )
        folds.append(result)
        if extra:
            extras[held_out] = extra

    headline = str(args.held_out or eval_cfg["held_out_vector"])
    if not any(f.held_out_vector == headline for f in folds):
        log.warning(
            "the headline fold %r was not among the folds run (%s) — the artefact will have no "
            "headline row",
            headline,
            ", ".join(f.held_out_vector for f in folds) or "none",
        )

    return loao.LeaveOneAttackOutReport(
        dataset=name,
        seed=seed,
        config={
            **eval_cfg,
            "held_out_vector": headline,
            "source": str(EVAL_CONFIG),
            "folds_requested": requested_folds(eval_cfg, args.folds),
            "sample_fraction": (cfg.get("sample") or {}).get("sample_fraction"),
            "seed": seed,
        },
        operating_point={
            "fixed_fpr": fixed_fpr,
            "k": k,
            "source": str(EVAL_CONFIG),
            "min_meaningful_positives": min_positives,
            "band_units": "calibrated probability",
        },
        folds=folds,
        split=split.to_dict(),
        data={
            "config": f"config/data/{name}.yaml",
            "supervised_reference": supervised_reference(name),
            "real": stats(real),
            "pools": {v: e["pool"] for v, e in extras.items()},
            "fit_seconds": {v: e["fit_seconds"] for v, e in extras.items()},
        },
        # the headline fold's detector when there is one. Every fold builds the same detector
        # from the same config and differs only in what it fitted on, so one card describes the
        # configuration; which fold's it is matters only for the row counts inside it.
        model_card=(extras.get(headline) or next(iter(extras.values()), {})).get("model_card", {}),
        meta={
            "generated_by": "scripts/build_loao.py",
            "note": (
                "Every positive in these folds is an injected synthetic row and every negative "
                "is a real one, so a fold's recall reports the distance between two "
                "distributions as well as detection. Read each fold's `separability` and "
                "`outcome` before reading its `metrics`."
            ),
        },
    )


# ── the document ────────────────────────────────────────────────────────────────
def _sentence(text: str) -> str:
    """A reason is written as a clause, because a table cell reads better without a capital.

    When one is promoted to a sentence of its own, it gets the capital back.
    """
    text = str(text).strip()
    return text[:1].upper() + text[1:] if text else text


def _wrap(text: str, indent: str = "") -> str:
    """Reflow a generated sentence to the width the hand-written prose around it uses.

    The reasons in this document are composed at run time from numbers, so they arrive as one
    long line. A generated paragraph that a reader has to scroll sideways for is a generated
    paragraph nobody reads, which defeats the point of generating it.
    """
    return textwrap.fill(
        " ".join(str(text).split()),
        width=98,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _fold_row(fold: loao.FoldResult, own_fraud: float | None = None) -> str:
    """One line of the matrix.

    A withheld fold still shows its numbers, in brackets, next to the reason they are withheld.
    Hiding them would be its own kind of dishonesty — the reader cannot check a claim about a
    number they cannot see — but a bracketed number beside "withheld" is not one anybody quotes
    by accident.
    """
    spec = registry.load_vectors().get(fold.held_out_vector)
    label = f"`{fold.held_out_vector}`" + (f" {spec.name}" if spec else "")
    m = fold.any_metrics
    probe = fold.provenance or {}
    positives = fold.counts.get("holdout_positives")
    probe_cell = (
        f"{float(probe['pr_auc']):.3f}" if probe.get("checked") else probe.get("reason", "—")[:28]
    )
    if m is None:
        return (
            f"| {label} | **{fold.outcome}** | — | — | — | "
            f"{positives if positives is not None else '—'} | — | {probe_cell} | {fold.reason} |"
        )

    def cell(value: str) -> str:
        return value if fold.reported else f"_({value})_"

    floor = float((fold.floor or {}).get("pr_auc", 0.0))
    if fold.reason:
        note = fold.reason
    elif m.pr_auc <= floor:
        note = f"**loses to the amount floor ({floor:.3f})** — it has not detected anything"
    elif own_fraud is not None and m.pr_auc > own_fraud:
        # ticket 10's reading: an unseen family that is easier than the seen one is a statement
        # about the injected rows, not a generalisation result
        note = (
            f"clears the floor ({floor:.3f}), but **beats this anchor's own fraud "
            f"({own_fraud:.3f})** — easier unseen than seen"
        )
    else:
        note = f"clears the floor ({floor:.3f}); the anchor's own fraud scores {own_fraud:.3f}"
    return (
        f"| {label} | {'measured' if fold.reported else '**' + fold.outcome + '**'} | "
        f"{cell(f'{m.pr_auc:.3f}')} | {cell(f'{m.recall_at_fixed_fpr:.3f}')} | "
        f"{cell(f'{m.precision_at_k:.2f}')} | {m.n_positives:,} | {floor:.3f} | "
        f"{probe_cell} | {note} |"
    )


def _anchor_section(report: loao.LeaveOneAttackOutReport) -> str:
    nl = chr(10)
    own_fraud = (report.data.get("supervised_reference") or {}).get("pr_auc")
    rows = nl.join(_fold_row(f, own_fraud) for f in report.folds)
    head = report.headline
    guards = (head.guards if head else {}) or {}
    family, embargo, haystack = (
        guards.get("family", {}),
        guards.get("embargo", {}),
        guards.get("haystack", {}),
    )
    counts = (head.counts if head else {}) or {}
    sep = (head.separability if head else None) or {}

    if head is None:
        verdict = "**There is no headline fold in this run.**"
    elif head.outcome != loao.MEASURED:
        verdict = _wrap(f"**The headline fold is `{head.outcome}`.** {_sentence(head.reason)}.")
    else:
        verdict = _wrap(f"**The headline fold is measured.** {head.summary()}")
    guard_block = (
        f"""
| guard | what it checked | result |
| --- | --- | ---: |
| family carve-out | {family.get("audited", "—")} | {family.get("rows_checked", 0):,} rows, \
{family.get("leaked_rows", 0)} leaked |
| out-of-time embargo | committed gap {embargo.get("embargo_seconds", 0):,}s | \
actual gap {embargo.get("gap_seconds", 0):,}s |
| haystack | legit rows in the test window | \
{haystack.get("legit_rows_kept", 0):,} of {haystack.get("legit_rows_in_window", 0):,} kept |
"""
        if head
        else ""
    )
    worst, headline_vector = sep.get("worst"), report.config.get("held_out_vector")
    sep_line = (
        "\n"
        + _wrap(
            f"""**One contract field.** The worst single field on this anchor's
`{headline_vector}` rows is `{worst}`, at PR-AUC {float(sep.get("score", 0.0)):.4f} against a
base rate of {float(sep.get("base_rate", 0.0)):.4f} — trivially separable:
**{sep.get("trivially_separable")}**. Anything one field can do here, a model does first."""
        )
        + "\n"
        if sep
        else ""
    )

    probe = (head.provenance if head else None) or {}
    detector_pr = head.any_metrics.pr_auc if head and head.any_metrics else 0.0
    if probe.get("checked"):
        probe_line = _wrap(
            f"""**A whole model.** A classifier given the fold's own features sorts the injected
rows from the anchor's own at PR-AUC **{float(probe["pr_auc"]):.3f}**
({probe["cv_folds"]}-fold cross-validated, base rate {float(probe["base_rate"]):.4f},
{probe["n_injected"]:,} injected rows in {probe["n_rows"]:,}) — separable:
**{probe["separable"]}**. The detector reaches {float(detector_pr):.3f} on the same rows. Every
positive in this fold is injected and every negative is real, so a probe that scores at or above
the detector means the fold cannot tell detection apart from provenance."""
        )
        probe_line = f"\n{probe_line}\n"
    elif probe:
        probe_line = f"""
**A whole model.** The provenance probe did not run: {probe.get("reason")}.
"""
    else:
        probe_line = ""

    ref = report.data.get("supervised_reference") or {}
    ref_line = (
        "\n"
        + _wrap(
            f"""**Against this anchor's own fraud.** The same detector reaches PR-AUC
{float(ref["pr_auc"]):.3f} on {report.dataset}'s own labelled fraud in the same test window
({ref["n_positives"]:,} real positives, `{ref["source"]}`), where sorting by amount alone reaches
{float(ref["amount_floor_pr_auc"]):.3f}. A held-out family that scores *above* that line is not a
generalisation result — it is a statement about the injected rows, which is the reading ticket 10
arrived at from the other side."""
        )
        + "\n"
        if ref
        else ""
    )

    return f"""### {report.dataset}

{verdict}

| held out | outcome | PR-AUC | rec@FPR | p@k | positives | floor | probe | why |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{rows}

_floor_ is `amount_only`: rank by amount, no model, direction chosen on the training window.
_probe_ is the provenance classifier's PR-AUC at telling the injected rows from the anchor's own.
Numbers in brackets are withheld — they exist, and nothing may be concluded from them.

Committed boundary `{report.split.get("digest")}`, embargo \
{report.split.get("embargo_seconds", 0):,}s, seed {report.seed}, \
sample fraction {report.config.get("sample_fraction")}.
The headline fold trains on {counts.get("train_rows", 0):,} rows \
({counts.get("train_fraud", 0):,} fraud) and is measured on {counts.get("holdout_rows", 0):,} \
({counts.get("holdout_positives", 0):,} positives, base rate \
{float(counts.get("holdout_base_rate", 0.0)):.4%}).
{ref_line}{guard_block}{sep_line}{probe_line}"""


def _anchor_readings(reports: dict[str, loao.LeaveOneAttackOutReport]) -> str:
    """What the measured rows say once each is put next to its anchor's own labelled fraud.

    A fold's PR-AUC on its own is not a claim. Ticket 08 established that a near-perfect number
    on AMLSim says the simulator is legible; ticket 10 established that an unseen family the
    detector finds *easier* than the seen one is a statement about the injected rows. Both
    readings are arithmetic over numbers this artefact already carries, so the document does the
    arithmetic rather than leaving it to a reader who may not know to.
    """
    lines: list[str] = []
    for report in reports.values():
        measured = [f for f in report.measured if f.metrics]
        if not measured:
            lines.append(
                _wrap(
                    f"- **{report.dataset}** — no fold carries a quotable number.", indent="  "
                ).lstrip()
            )
            continue
        ref = report.data.get("supervised_reference") or {}
        own = float(ref.get("pr_auc", 0.0))
        own_floor = float(ref.get("amount_floor_pr_auc", 0.0))
        scores = ", ".join(f"`{f.held_out_vector}` {f.metrics.pr_auc:.3f}" for f in measured)
        if own >= 0.99:
            reading = (
                f"The same detector scores {own:.3f} on {report.dataset}'s own labelled fraud in "
                f"the same window, where sorting by amount alone already reaches {own_floor:.3f}. "
                "A near-perfect fold on an anchor like that says the generator is legible, not "
                "that anything generalised."
            )
        elif all(f.metrics.pr_auc > own for f in measured):
            subject = "It sits" if len(measured) == 1 else "Every one of them sits"
            reading = (
                f"{subject} above {report.dataset}'s own labelled fraud ({own:.3f}) in the same "
                "window. An unseen family the detector finds *easier* than the ones it trains on "
                "is a statement about the injected rows."
            )
        else:
            reading = (
                f"Against {own:.3f} on {report.dataset}'s own labelled fraud in the same window "
                f"and a {own_floor:.3f} amount floor on it."
            )
        lines.append(_wrap(f"- **{report.dataset}** — {scores}. {reading}", indent="  ").lstrip())
    return "**What the measured rows say next to their anchor's own fraud.**\n\n" + "\n".join(lines)


def loao_doc(reports: dict[str, loao.LeaveOneAttackOutReport], missing: list[str]) -> str:
    """The matrix, generated from the artefacts that produced it."""
    nl = chr(10)
    first = next(iter(reports.values()))
    op = first.operating_point
    sections = nl.join(_anchor_section(r) for r in reports.values())

    not_measured = [
        _wrap(f"- **{r.dataset} / `{f.held_out_vector}`** — {f.outcome}: {f.reason}.").replace(
            "\n", "\n  "
        )
        for r in reports.values()
        for f in r.not_measured
    ]
    skipped_block = (
        nl.join(not_measured)
        if not_measured
        else "_Every requested fold on every anchor carried a quotable number._"
    )
    absent = (
        "**Not measured.** "
        + " ".join(f"`{n}` is not downloaded, so it has no matrix here." for n in missing)
        if missing
        else "_Every configured anchor was on disk when this ran._"
    )
    n_measured = sum(len(r.measured) for r in reports.values())
    n_folds = sum(len(r.folds) for r in reports.values())
    survivors = _anchor_readings(reports)
    verdict = _wrap(
        f"**{n_measured} of {n_folds} folds across {len(reports)} anchor(s) carry a quotable "
        "number.** Read a fold's provenance probe before quoting its recall: the two are the "
        "same measurement pointed at different questions."
        if n_measured
        else (
            "**Not one fold on any anchor produced a quotable number, and that is the result.** "
            "The harness works — the guards pass, the folds run, the matrix is complete — and "
            "every row of it is withheld or skipped for a reason stated in the row. A "
            "leave-one-attack-out headline built on this generator would be a number about "
            "provenance wearing the name of a number about detection."
        )
    )

    return f"""# Leave-one-attack-out

_Generated by `scripts/build_loao.py` from the anchors on disk. Every number below traces to
`artifacts/loao/<anchor>.json`, which carries the eval config, the committed split digest and the
seed that produced it. Do not edit this file — re-run `make loao`._

Train without one attack family, then measure recall on that family alone. Reporting recall on a
family the model trained on measures memorisation; this is the only fold that asks about
generalisation to an unseen attack, and everything downstream of ticket 11 is measured through
it.

**Operating point: recall at {float(op["fixed_fpr"]):.0%} FPR, precision@{int(op["k"])}**, fixed
in `config/eval/leave_one_attack_out.yaml` before any of these numbers existed.

## The three guards

They are assertions in `afl/evaluation/leave_one_attack_out.py`, not intentions, and each has a
test that deliberately tries to leak a row past it.

1. **Not one row of the held-out family reaches training — replay buffer included.** The audit
   runs against the *fitted detector*, via its `training_rows`, not against the list handed to
   `fit`. The replay buffer is the place a carved-out family walks back into training four
   rounds later without the split changing. A detector that cannot say what it trained on fails
   the guard rather than passing it.
2. **The split is still out-of-time, with the committed embargo intact after the carve-out.**
   Removing a family shortens both sides and the arithmetic says the gap can only widen — but an
   argument is exactly what an assertion replaces.
3. **Every legit row of the test window stays in the holdout.** An FPR measured without
   negatives is not an FPR, and precision@100 on a holdout of 100 positives is 1.0 by
   construction.

A fourth check is not a guard but a verdict, and it is the one that decides most of this table.
The carve-out drops the anchor's own fraud from the holdout, so **every positive in a fold is an
injected row and every negative is a real one** — "caught the fraud" and "spotted the synthetic
row" are the same label. The provenance probe asks a classifier to make exactly that call on the
fold's own features. Where it succeeds, the fold's recall is a statement about the generator, and
the number is withheld. Ticket 07 measured this by hand and got AUC 1.00 on PaySim; it lives in
the harness now so it cannot be quoted around.

Read the probe asymmetrically. It learns "injected" from the fold's own positives, so a **high**
probe score is strong evidence the fold is measuring provenance, and a **low** one on a fold with
few positives is weak evidence of anything — the probe simply had less to learn from than the
detector it is checking. The positive count sits beside every probe score for that reason.

## What a fold is allowed to claim

A fold that runs is not a fold that means something. Each row is one of three outcomes, and only
the first carries a claim:

- **measured** — the numbers stand.
- **withheld** — the fold ran and the numbers exist, under `withheld_metrics` rather than
  `metrics`, but nothing may be concluded from them. Four ways to land here: fewer than
  {int(op["min_meaningful_positives"])} positives, so recall moves further per row than the
  differences anyone would read into it; the family is separable from the anchor by a single
  contract field; **a classifier can sort the injected rows from the anchor's own**, so the fold
  is measuring provenance rather than detection; or the vector is a `template` whose defining
  tell is not modelled yet.
- **skipped** — the fold never ran, and the reason sits in the cell where the number would be.

{verdict}

{survivors}

## The matrix

{sections}
## Folds that carry no number

{skipped_block}

{absent}

## Where the numbers came from

```bash
make loao        # or: python scripts/build_loao.py
```
"""


# ── the run ─────────────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="dataset names; default: all of them")
    parser.add_argument(
        "--folds",
        default=None,
        help="comma-separated vectors to hold out, one at a time; default: eval.folds",
    )
    parser.add_argument(
        "--held-out",
        default=None,
        help="which fold is the headline row; default: eval.held_out_vector",
    )
    parser.add_argument("--sample", type=float, default=None, help="override the entity sample")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--doc-only",
        action="store_true",
        help="rewrite docs/loao.md from the committed artefacts without re-running the matrix; "
        "the document is a pure function of them, so this cannot disagree with a run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_DIR,
        help="where the artefacts go; point it elsewhere for a trial run so a quick pass cannot "
        "overwrite the committed matrix",
    )
    args = parser.parse_args()

    sup = yaml.safe_load(LGBM_CONFIG.read_text())
    uns = yaml.safe_load(ANOMALY_CONFIG.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    engines = yaml.safe_load(ENGINES_CONFIG.read_text())
    costs = yaml.safe_load(COSTS_CONFIG.read_text())

    reports: dict[str, loao.LeaveOneAttackOutReport] = {}
    missing: list[str] = []
    if args.doc_only:
        reports = loao.load_all(args.out)
        missing = [c["name"] for c in anchors(args.datasets) if c["name"] not in reports]
        if not reports:
            print(f"no committed matrix in {args.out} — run `make loao` first", file=sys.stderr)
            return 1
        DOC_PATH.write_text(loao_doc(reports, missing))
        print(f"→ {DOC_PATH} (from {len(reports)} committed artefact(s))")
        return 0

    for cfg in anchors(args.datasets):
        try:
            reports[cfg["name"]] = run_anchor(cfg, args, sup, uns, eval_cfg, engines, costs)
        except loaders.DatasetNotDownloaded as exc:
            missing.append(cfg["name"])
            print(f"SKIPPED {cfg['name']}: {exc}", file=sys.stderr)

    if not reports:
        print("no anchor on disk — nothing to hold out", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        path = report.save(args.out)
        print(f"\n── {name} " + "─" * (70 - len(name)))
        for fold in report.folds:
            marker = "  " if fold.reported else "! "
            print(f"  {marker}{fold.summary()}")
        print(f"  → {path}")

    if args.out != ARTIFACT_DIR:
        print(f"\n(--out is not {ARTIFACT_DIR}; leaving {DOC_PATH} alone)")
        return 0
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(loao_doc(reports, missing))
    print(f"\n→ {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
