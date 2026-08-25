"""Tune the supervised detector honestly on each real anchor and commit the result.

    python scripts/build_baseline.py                  # every real anchor
    python scripts/build_baseline.py paysim           # just one
    python scripts/build_baseline.py --trials 8 --sample 0.02   # a quick pass

Ticket 08's deliverable. This is the hard baseline everything later has to beat, so the one
thing that would ruin it is leaving it weak: a soft baseline makes every subsequent result
meaningless, and it is the easiest kind of dishonesty to commit by accident.

Three properties, all of them enforced rather than promised:

**The search never sees the test window.** The committed out-of-time boundary splits the anchor
once. The training side is split again, chronologically, and only that inner tail is scored
during the search — `afl.defend.tuning` raises if it is not strictly after the fitting rows.
The action bands are calibrated on the same inner tail, for the same reason.

**The number says which model produced it.** On macOS the LightGBM wheel imports cleanly and
then fails to `dlopen` its own shared library when libomp is missing, so the code runs on the
sklearn fallback under a table headed "LightGBM". The backend, its version and the reason it was
chosen are recorded on every artefact.

**Tuning is shown to have earned its keep.** Both the tuned and the stock detector are scored on
the same test window, so the artefact carries the counterfactual instead of an assurance.

Everything lands in `artifacts/detector/<anchor>.json` and in `docs/detector.md`, which is
generated from those files and not hand-typed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import (
    CommittedSplit,
    assert_no_leakage,
    committed_split_for,
    out_of_time_split,
)
from afl.defend import tuning
from afl.defend.baseline import Baseline, load_all
from afl.defend.decision import (
    CostModel,
    assert_one_operating_point,
    cost_model_for,
    policy_from_config,
)
from afl.defend.features import FeatureBuilder
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import protocol
from afl.utils.seed import set_all_seeds

log = logging.getLogger("build_baseline")

DATA_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
COSTS_CONFIG = Path("config/costs/default.yaml")
ARTIFACT_DIR = Path("artifacts/detector")
DOC_PATH = Path("docs/detector.md")


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


def build_detector(
    params: dict, sup: dict, seed: int, source: str, costs: CostModel
) -> LGBMDetector:
    """The reference detector, on the same decision policy the rest of the project runs.

    `costs` is denominated in this anchor's own median payment, so the bands the policy derives
    are comparable across anchors whose amounts differ by two orders of magnitude. It is passed
    in rather than built here because the anchor's rows are what measures it.
    """
    return LGBMDetector(
        policy=policy_from_config(sup["decision"], costs),
        features=FeatureBuilder(
            stateful=bool(sup["features"]["stateful"]),
            windows_s=tuple(int(w) for w in sup["features"]["windows_s"]),
        ),
        params=params,
        seed=seed,
        replay_weight=float(sup["replay_weight"]),
        params_source=source,
    )


def fit_calibrate_score(
    detector: LGBMDetector,
    fit_rows: list[Transaction],
    val_rows: list[Transaction],
    train: list[Transaction],
    test: list[Transaction],
    fixed_fpr: float,
    k: int,
    fpr_bands: bool,
) -> dict:
    """Fit, calibrate on validation, refit on the full training window, score test.

    Two different things are set on that inner validation tail and neither is ever set on
    `test`, for the same reason the hyperparameters are not: an operating point chosen on the
    window it is reported from is not an operating point, it is a result.

      * the score → probability map, always, because the cost model needs a probability;
      * the FPR-pinned bands, only when the config asked for them, which in cost mode it cannot.
    """
    started = time.perf_counter()
    if val_rows and any(t.is_fraud for t in fit_rows):
        detector.fit(fit_rows)
        detector.policy.reset_calibration()
        y, s = protocol.align(
            val_rows, protocol.score_transactions(detector, val_rows, "calibration")
        )
        detector.policy.fit_calibrator(s, y)
        if fpr_bands:
            detector.policy.calibrate_to_fpr(s, y, target_fpr=fixed_fpr)
    detector.fit(train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scores = protocol.score_transactions(detector, test, run_id="baseline")
    score_seconds = time.perf_counter() - started

    y, s = protocol.align(test, scores)
    result = protocol.evaluate(y, s, fixed_fpr=fixed_fpr, k=k)
    return {
        **result.model_dump(exclude={"held_out_vector"}),
        **{k2: round(v, 6) for k2, v in protocol.operational_rates(test, scores).items()},
        "fit_seconds": round(fit_seconds, 2),
        "score_seconds": round(score_seconds, 2),
        "score_rows_per_second": round(len(test) / score_seconds) if score_seconds else 0,
    }


def amount_only_reference(
    train: list[Transaction], test: list[Transaction], fixed_fpr: float, k: int
) -> dict:
    """The floor: rank by amount alone, no model, no features, no training.

    A baseline is only "strong" relative to how hard the anchor is, and this is the cheapest
    honest way to say how hard it is. The direction — large amounts first or small amounts
    first — is chosen on the **training** window and applied to test, because picking it on the
    window it is reported from is exactly the tuning-on-test this whole file is arranged to
    avoid.

    It also carries the diagnostic that explains the direction: how much of the legit traffic
    the fraud rows' amount range already excludes, before anything is fitted at all.
    """
    y_tr = np.array([int(t.is_fraud) for t in train])
    a_tr = np.array([t.amount for t in train], dtype=float)
    high_first = protocol.pr_auc(y_tr, a_tr) >= protocol.pr_auc(y_tr, -a_tr)

    y_te = np.array([int(t.is_fraud) for t in test])
    a_te = np.array([t.amount for t in test], dtype=float)
    scores = a_te if high_first else -a_te
    result = protocol.evaluate(y_te, scores, fixed_fpr=fixed_fpr, k=k)

    fraud = a_te[y_te == 1]
    legit = a_te[y_te == 0]
    inside = float(((legit >= fraud.min()) & (legit <= fraud.max())).mean()) if fraud.size else 1.0
    return {
        **result.model_dump(exclude={"held_out_vector"}),
        "direction": "largest amount first" if high_first else "smallest amount first",
        "direction_chosen_on": "train",
        "fraud_amount_min": round(float(fraud.min()), 2) if fraud.size else None,
        "fraud_amount_max": round(float(fraud.max()), 2) if fraud.size else None,
        "legit_share_inside_fraud_amount_range": round(inside, 6),
    }


def run_anchor(cfg: dict, args, sup: dict, eval_cfg: dict, costs_cfg: dict) -> Baseline:
    """One anchor: split, tune, calibrate, score, and package the whole thing as a reference."""
    name = cfg["name"]
    seed = int(args.seed)
    set_all_seeds(seed)
    fixed_fpr, k = float(eval_cfg["fixed_fpr"]), int(eval_cfg["k"])
    tune_cfg = sup["tuning"]
    assert_one_operating_point(
        sup["decision"].get("calibrate_to_fpr"), fixed_fpr, mode=str(sup["decision"]["mode"])
    )

    if args.sample is not None:
        cfg = {**cfg, "sample": {**(cfg.get("sample") or {}), "sample_fraction": args.sample}}
    rows = loaders.load_from_config(cfg)
    # The flat costs are quoted against this anchor's own median payment — 74,872 on PaySim and
    # 157 on AMLSim — so one cost config places comparable bands on both. See config/costs/.
    costs = cost_model_for(costs_cfg, rows)

    split: CommittedSplit | None = committed_split_for(cfg)
    if split is None:
        raise SystemExit(f"{name}: no committed split — run `make splits` first")
    train, test = split.apply(rows)
    assert_no_leakage(train, test)

    # the inner split: the search and the action bands see this tail and nothing later
    fit_rows, val_rows = out_of_time_split(
        train,
        train_frac=float(tune_cfg["fit_frac"]),
        embargo_days=float(eval_cfg["embargo_days"]),
    )

    log.info(
        "%s: %d rows -> train %d (%d fraud) / test %d (%d fraud); tuning on a %d-row tail",
        name,
        len(rows),
        len(train),
        sum(1 for t in train if t.is_fraud),
        len(test),
        sum(1 for t in test if t.is_fraud),
        len(val_rows),
    )
    if not any(t.is_fraud for t in test):
        raise SystemExit(f"{name}: the test window has no fraud — there is nothing to measure")

    result = tuning.TuningResult(
        params=dict(sup["params"]),
        metric=str(tune_cfg["metric"]),
        best_score=0.0,
        default_score=0.0,
        n_trials=0,
        backend="none",
        seed=seed,
        n_fit_rows=len(fit_rows),
        n_val_rows=len(val_rows),
        n_val_positives=sum(1 for t in val_rows if t.is_fraud),
        skipped="tuning.enabled is false",
    )
    if bool(tune_cfg["enabled"]) and args.trials != 0:
        result = tuning.tune(
            fit_rows,
            val_rows,
            base_params=sup["params"],
            search_space=tune_cfg["search_space"],
            n_trials=int(args.trials if args.trials is not None else tune_cfg["n_trials"]),
            seed=seed,
            metric=str(tune_cfg["metric"]),
            fixed_fpr=fixed_fpr,
            backend=str(args.backend or tune_cfg["backend"]),
            features=FeatureBuilder(
                stateful=bool(sup["features"]["stateful"]),
                windows_s=tuple(int(w) for w in sup["features"]["windows_s"]),
            ),
        )

    # Both variants on the same window: "tuned" is a claim about a comparison, so the artefact
    # carries the counterfactual rather than an assurance that the search helped.
    fpr_bands = sup["decision"].get("calibrate_to_fpr") is not None
    metrics: dict[str, dict] = {}
    tuned_detector: LGBMDetector | None = None
    for variant, params, source in (
        ("tuned", result.params, f"tuning ({result.backend}, {result.n_trials} trials)"),
        ("default", sup["params"], "config/defend/lgbm.yaml"),
    ):
        set_all_seeds(seed)
        detector = build_detector(params, sup, seed, source, costs)
        metrics[variant] = fit_calibrate_score(
            detector, fit_rows, val_rows, train, test, fixed_fpr, k, fpr_bands
        )
        metrics[variant]["backend"] = str(detector.backend)
        tuned_detector = tuned_detector or detector
        log.info("%s [%s]: %s", name, variant, json.dumps(metrics[variant], default=str))

    metrics["amount_only"] = amount_only_reference(train, test, fixed_fpr, k)
    log.info("%s [amount_only]: %s", name, json.dumps(metrics["amount_only"], default=str))

    assert tuned_detector is not None
    card = tuned_detector.model_card()

    return Baseline(
        dataset=name,
        seed=seed,
        operating_point={
            "fixed_fpr": fixed_fpr,
            "k": k,
            "source": str(EVAL_CONFIG),
            # The whole decision layer, not four band numbers: the cost model that placed
            # them, its stated rationale per parameter, and how well the score → probability
            # map actually calibrated. A band without those behind it is unauditable.
            "decision": card["decision"],
            "calibrated_on": "the training window's validation tail",
        },
        backend=card["backend"],
        params=result.params,
        split=split.to_dict(),
        data={
            "config": f"config/data/{name}.yaml",
            "sample_fraction": (cfg.get("sample") or {}).get("sample_fraction"),
            "all": stats(rows),
            "train": stats(train),
            "test": stats(test),
            "tuning_fit": stats(fit_rows),
            "tuning_validation": stats(val_rows),
        },
        tuning=result.to_dict(),
        metrics=metrics,
        meta={
            "generated_by": "scripts/build_baseline.py",
            "note": (
                "Measured on this anchor's OWN labelled fraud, out of time at the committed "
                "boundary. Not the leave-one-attack-out fold: that fold's positives are injected "
                "synthetic rows, so it measures the distance between two distributions as much "
                "as it measures detection. See docs/detector.md."
            ),
        },
    )


# ── the document ────────────────────────────────────────────────────────────────
def _pct(x: float) -> str:
    """One decimal, except near the ends, where rounding would erase the point being made.

    "100.0%" for 99.996% would say amount excludes nothing when it excludes six rows in a
    hundred thousand — a difference nobody needs, stated as a certainty nobody checked.
    """
    if 0.999 < x < 1.0:
        return ">99.9%"
    if 0.0 < x < 0.001:
        return "<0.1%"
    return f"{x:.1%}"


def detector_doc(cards: dict[str, Baseline], missing: list[str]) -> str:
    """The reference numbers, generated from the artefacts that produced them."""

    def row(b: Baseline) -> str:
        t, d, a = b.metrics["tuned"], b.metrics["default"], b.metrics["amount_only"]
        return (
            f"| {b.dataset} | {t['pr_auc']:.3f} | {d['pr_auc']:.3f} | {a['pr_auc']:.3f} | "
            f"{t['recall_at_fixed_fpr']:.3f} | {d['recall_at_fixed_fpr']:.3f} | "
            f"{a['recall_at_fixed_fpr']:.3f} | "
            f"{t['precision_at_k']:.2f} | {d['precision_at_k']:.2f} | {a['precision_at_k']:.2f} | "
            f"{t['n_positives']:,} |"
        )

    def window_row(label: str, d: dict) -> str:
        return (
            f"| {label} | {d['rows']:,} | {d['fraud']:,} | {d['base_rate']:.4%} "
            f"| {d['first_ts']} | {d['last_ts']} |"
        )

    def detail(b: Baseline) -> str:
        tr, te = b.data["train"], b.data["test"]
        tn = b.tuning
        params = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(b.params.items()))
        ops = b.metrics["tuned"]
        dropped = b.backend.get("dropped_params") or []
        dropped_line = (
            f"\n{len(dropped)} param(s) have no equivalent on this backend and were dropped: "
            f"{', '.join('`' + d + '`' for d in dropped)}.\n"
            if dropped
            else ""
        )
        return f"""### {b.dataset}

**{b.summary()}**

| | rows | fraud | base rate | from | to |
| --- | ---: | ---: | ---: | --- | --- |
{window_row("train", tr)}
{window_row("test", te)}

Committed boundary `{b.split["digest"]}`, embargo {b.split["embargo_seconds"]:,}s. The search and
the score → probability calibration saw a {b.data["tuning_validation"]["rows"]:,}-row validation
tail inside train ({b.data["tuning_validation"]["fraud"]:,} fraud) and nothing after it.

**Backend: {b.backend["name"]} {b.backend["version"]}** — {b.backend["reason"]}
{dropped_line}
**Search:** {tn["n_trials"]} trials on {tn["backend"]}, maximising `{tn["metric"]}` on validation.
Best {tn["best_score"]:.4f} against {tn["default_score"]:.4f} for the stock params
({tn["lift_over_defaults"]:+.4f}), in {tn["seconds"]:.0f}s.

**What the policy did on the test window:** friction on {_pct(ops["friction_rate"])} of legit
traffic, {_pct(ops["false_decline_rate"])} declined outright, and
{_pct(ops["evasion_rate"])} of fraud allowed through untouched. Those three are a property of
the *decision layer*, not of the ranking above them — the bands come from the cost model in
`config/costs/default.yaml`, and `docs/decisions.md` is where they are priced and compared
against the alternatives.

| param | committed value |
| --- | --- |
{params}
"""

    absent = (
        "**Not measured.** "
        + " ".join(f"`{n}` is not downloaded, so it has no baseline here." for n in missing)
        if missing
        else "_Every configured anchor was on disk when this ran._"
    )

    def floor_note(b: Baseline) -> str:
        a, t = b.metrics["amount_only"], b.metrics["tuned"]
        return (
            f"- **{b.dataset}** — every fraud row in the test window sits between "
            f"{a['fraud_amount_min']:,.2f} and {a['fraud_amount_max']:,.2f}, a band holding "
            f"{_pct(a['legit_share_inside_fraud_amount_range'])} of the legit traffic. Sorting on "
            f"amount alone ({a['direction']}) already reaches PR-AUC {a['pr_auc']:.3f}; the "
            f"detector reaches {t['pr_auc']:.3f}."
        )

    nl = chr(10)
    floors = nl.join(floor_note(b) for b in cards.values())
    first = next(iter(cards.values())).operating_point
    operating_point = f"recall at {float(first['fixed_fpr']):.0%} FPR, precision@{int(first['k'])}"
    columns = ["anchor", "PR-AUC", "_stock_", "_floor_"]
    columns += ["rec@FPR", "_stock_", "_floor_", "p@k", "_stock_", "_floor_", "fraud"]
    header = "| " + " | ".join(columns) + " |" + nl + "| --- |" + " ---: |" * (len(columns) - 1)
    rows = nl.join(row(b) for b in cards.values())
    details = nl.join(detail(b) for b in cards.values())
    backends = sorted({f"{b.backend['name']} {b.backend['version']}" for b in cards.values()})

    return f"""# The supervised baseline

_Generated by `scripts/build_baseline.py` from the anchors on disk. Every number below traces to
`artifacts/detector/<anchor>.json`, which carries the params, the backend, the split digest and
the seed that produced it. Do not edit this file — re-run `make baseline`._

This is the detector the rest of the project is measured against: gradient-boosted trees over the
56 causal features in `docs/features.md`, trained on the training side of the committed
out-of-time boundary and scored on the test side, at one operating point fixed in
`config/eval/leave_one_attack_out.yaml` before any of these numbers existed.

The three metrics below are **rank statistics**, so the graded decision layer added in ticket 09
cannot move them: calibrating the score to a probability is a monotone map. What that layer does
move — the action mix, the friction, the realised cost — lives in `docs/decisions.md`.

**Backend: {", ".join(backends)}.** LightGBM needs libomp on macOS. Without it the wheel imports
and then fails to load its own shared library, and the code falls back to sklearn's
HistGradientBoosting — a different model under a table headed "LightGBM". Every artefact records
which one ran and why, and `make baseline` prints it.

## The reference

Each anchor's **own labelled fraud**, out of time, tuned versus stock params on the same window.

{header}
{rows}

At the operating point in `config/eval/leave_one_attack_out.yaml`: **{operating_point}**.
_stock_ is the same detector at the params in `config/defend/lgbm.yaml`, so the gap between the
first two columns is what the search bought. _floor_ is `amount_only`: rank by amount alone, no
model, no features, no training, with the direction chosen on the training window.

## Read the floor column first

A baseline is only "strong" relative to how hard the anchor is, and the floor column is the
cheapest honest way to say how hard it is.

{floors}

Accuracy and ROC-AUC are absent on purpose, and `afl/defend/baseline.py` refuses to save an
artefact containing either: at these base rates both are dominated by the negatives, so a
detector that catches nothing still scores well on them.

## What this number is, and what it is not

**It is** the honest ceiling a later ticket has to clear on the same anchor, the same boundary
and the same operating point.

**It is not the leave-one-attack-out number.** In that fold every positive is an injected
synthetic M3 row and every negative is a real one, so its recall partly reports how far the
injected family sits from the real distribution rather than how well anything detects
first-party fraud — see the ticket 07 carry-out and the committed fidelity scorecards. The
hero table's held-out column is measured by `scripts/run_experiment.py`; this file is the
detector's own reference, and the two are not interchangeable.

**AMLSim is not a production number, and the floor column above is the proof rather than the
assertion.** It is a simulator, and its alerted rows are separable before any model runs: they
occupy a narrow amount band that most of the background traffic sits outside, on top of the
deliberately distinctive fan-in / cycle topology the graph features then finish the job with.
A near-perfect figure there says the generator is legible, not that the detector is good.

**PaySim is the anchor to read.** Its fraud spans the whole amount range, so the floor column
excludes nothing and the detector has to earn every point. It is also the harder anchor for a
different reason: `nameOrig` is effectively unique per row, so there is no sender history at all
and a third of the feature table is structurally empty on it (`docs/features.md`).

{absent}

## Per anchor

{details}
## Where the numbers came from

```bash
make baseline        # or: python scripts/build_baseline.py
```
"""


# ── the run ─────────────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="dataset names; default: all of them")
    parser.add_argument("--trials", type=int, default=None, help="override tuning.n_trials")
    parser.add_argument("--backend", default=None, help="optuna | random | auto")
    parser.add_argument("--sample", type=float, default=None, help="override the entity sample")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--doc-only",
        action="store_true",
        help="rewrite docs/detector.md from the committed artefacts without re-running the "
        "search; the document is a pure function of them, so this cannot disagree with a run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_DIR,
        help="where the artefacts go; point it elsewhere for a trial run so a quick pass "
        "cannot overwrite the committed reference",
    )
    args = parser.parse_args()

    sup = yaml.safe_load(LGBM_CONFIG.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    costs_cfg = yaml.safe_load(COSTS_CONFIG.read_text())

    cards: dict[str, Baseline] = {}
    missing: list[str] = []
    if args.doc_only:
        cards = load_all(args.out)
        missing = [c["name"] for c in anchors(args.datasets) if c["name"] not in cards]
        if not cards:
            print(
                f"no committed baseline in {args.out} — run `make baseline` first", file=sys.stderr
            )
            return 1
        DOC_PATH.write_text(detector_doc(cards, missing))
        print(f"→ {DOC_PATH} (from {len(cards)} committed artefact(s))")
        return 0

    for cfg in anchors(args.datasets):
        try:
            cards[cfg["name"]] = run_anchor(cfg, args, sup, eval_cfg, costs_cfg)
        except loaders.DatasetNotDownloaded as exc:
            missing.append(cfg["name"])
            print(f"SKIPPED {cfg['name']}: {exc}", file=sys.stderr)

    if not cards:
        print("no anchor on disk — nothing to baseline", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name, card in cards.items():
        path = card.save(args.out)
        print(f"\n── {name} " + "─" * (70 - len(name)))
        print(f"  {card.summary()}")
        b = card.backend
        print(f"  backend: {b['name']} {b['version']} — {b['reason']}")
        print(
            f"  tuning: {card.tuning['n_trials']} trials on {card.tuning['backend']}, "
            f"{card.tuning['metric']} {card.tuning['default_score']:.4f} → "
            f"{card.tuning['best_score']:.4f} ({card.tuning['lift_over_defaults']:+.4f}) on "
            f"validation"
        )
        print(f"  → {path}")

    if args.out != ARTIFACT_DIR:
        print(f"\n(--out is not {ARTIFACT_DIR}; leaving {DOC_PATH} alone)")
        return 0
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(detector_doc(cards, missing))
    print(f"\n→ {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
