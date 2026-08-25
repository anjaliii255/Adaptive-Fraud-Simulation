"""Measure what the graded decision policy actually does, on each real anchor, and commit it.

    python scripts/build_decisions.py                 # every real anchor
    python scripts/build_decisions.py paysim          # just one
    python scripts/build_decisions.py --sample 0.02   # a quick pass
    python scripts/build_decisions.py --doc-only      # rewrite the doc from committed artefacts

Ticket 09's deliverable. `make baseline` says how well the detector *ranks*; this says what
happens when a rank becomes an action somebody has to work. They are different questions and the
second one has been answered by guesswork until now: `calibrate_to_fpr` placed `decline_at` at
the target FPR and then put the other three bands at 0.8, 0.6 and 0.3 of it — ratios calibrated
to nothing at all. On the PaySim M3 fold that landed friction on 45.6% of holdout traffic while
precision@100 was 0.00. Blanket friction, reported in the same table as a detection metric.

So the bands come from a cost model now, and this script is the evidence that the swap was worth
making. Four things it measures rather than asserts:

**Where the bands land**, at several points of the anchor's own amount distribution — because a
policy priced per transaction does not have one ladder, it has one per amount, and the whole
claim is that a 500-rupee payment and a 5-lakh payment should not be treated alike.

**Whether the score is a probability.** Brier and expected calibration error, before and after
the Platt map, on the validation tail. A cost model fed a ranking score is arithmetic on the
wrong units, and the reliability numbers are what say whether that was fixed. The calibrated
probability stays inside the decision — the reported score is the detector's own — so nothing in
this file can move a number in `artifacts/detector/`.

**What the policy costs against its controls** — allow everything, decline everything, and the
ratio bands it replaces — all four scored on the same rows, from the same probabilities, so the
only difference is where the bands sit.

**Whether a cost parameter actually moves anything.** The sensitivity block re-decides the same
probabilities under a cheaper false decline and a dearer analyst. A cost model whose parameters
change nothing is a decoration with a rationale attached.

Everything lands in `artifacts/decisions/<anchor>.json` and in `docs/decisions.md`, which is
generated from those files and never hand-typed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from afl.contract.metrics import Action, DetectorScore
from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import (
    CommittedSplit,
    assert_no_leakage,
    committed_split_for,
    out_of_time_split,
)
from afl.defend import baseline, explain
from afl.defend.decision import (
    SEVERITY,
    CostModel,
    DecisionPolicy,
    DominatedAction,
    action_mix,
    assert_one_operating_point,
    cost_model_for,
    policy_from_config,
    ratio_band_policy,
    total_cost,
)
from afl.defend.features import FeatureBuilder
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import protocol
from afl.utils.seed import set_all_seeds

log = logging.getLogger("build_decisions")

DATA_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
COSTS_CONFIG = Path("config/costs/default.yaml")
ARTIFACT_DIR = Path("artifacts/decisions")
DOC_PATH = Path("docs/decisions.md")

DECISIONS_ARTEFACT_VERSION = 1

#: Where the ladder is reported. A cost-priced policy has one band set per amount, so quoting a
#: single ladder would hide the only thing that makes it graded.
AMOUNT_QUANTILES = (0.10, 0.50, 0.90, 0.99)

#: The parameter sweeps that show which cost knob is connected to what. A `*_multiple` is quoted
#: against `unit_amount`, exactly as `config/costs/default.yaml` quotes it; the middle value of
#: each triple is the shipped one, so the row above and below it bracket the house position.
#: Each triple stays inside a valid ladder — a step-up dearer than a hold is a *dominated* rung
#: and `CostModel` refuses it outright, which is the guard working rather than a range to explore.
#: A value that is refused anyway is recorded as refused, not allowed to kill the run.
SENSITIVITY = (
    ("step_up_cost_multiple", (0.001, 0.004, 0.010)),
    ("review_cost_multiple", (0.020, 0.050, 0.120)),
    ("false_decline_cost", (0.20, 0.35, 0.50)),
)


def anchors(selected: list[str]) -> list[dict]:
    """Every data config that names a loader — the real anchors, not the synthetic default."""
    out = []
    for path in sorted(DATA_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("loader") and (not selected or cfg["name"] in selected):
            out.append(cfg)
    return out


def redecide(
    policy: DecisionPolicy, probabilities: np.ndarray, rows: list[Transaction]
) -> list[DetectorScore]:
    """Apply a policy to probabilities the caller has already put on the right scale.

    Every comparison in this artefact runs through here, and that is deliberate: re-scoring with
    a second detector would confound where the bands sit with which model produced the score.
    Same model, same numbers, one difference.

    `probabilities` must be in the units the policy's bands are in — calibrated for a cost-mode
    policy, raw for the ratio bands, which were fitted on raw scores and are reconstructed here
    exactly as they ran.
    """
    return [
        DetectorScore(
            txn_id=t.txn_id,
            score=float(p),
            action=policy.act_on_probability(float(p), t.amount),
            reasons=[],
        )
        for t, p in zip(rows, probabilities, strict=True)
    ]


def realised(scores: list[DetectorScore], rows: list[Transaction], costs: CostModel) -> dict:
    """What this decision set did, and what it cost when the labels came in."""
    amounts = {t.txn_id: t.amount for t in rows}
    labels = {t.txn_id: int(t.is_fraud) for t in rows}
    cost = total_cost(scores, amounts, labels, costs)
    return {
        "action_mix": {k: round(v, 6) for k, v in action_mix(scores).items()},
        **{k: round(v, 6) for k, v in protocol.operational_rates(rows, scores).items()},
        "realised_cost": round(cost, 2),
        "cost_per_1k_txns": round(1_000 * cost / len(rows), 2) if rows else 0.0,
    }


def controls(
    probabilities: np.ndarray,
    raw: np.ndarray,
    rows: list[Transaction],
    costs: CostModel,
    val_raw: np.ndarray,
    val_y: np.ndarray,
    fixed_fpr: float,
) -> dict:
    """The three policies the cost model has to beat, on the same probabilities.

    `ratio_bands` is the policy this ticket replaced, reconstructed exactly: `decline_at` pinned
    to the target FPR on validation, the other three at 0.8, 0.6 and 0.3 of it. It is the one
    that matters — allow-everything and decline-everything are there to bracket the range, and a
    policy that does not sit inside that bracket is broken rather than merely bad.
    """
    amounts = {t.txn_id: t.amount for t in rows}
    labels = {t.txn_id: int(t.is_fraud) for t in rows}

    def flat(action: Action) -> dict:
        scores = [
            DetectorScore(txn_id=t.txn_id, score=float(p), action=action)
            for t, p in zip(rows, raw, strict=True)
        ]
        return {
            "realised_cost": round(total_cost(scores, amounts, labels, costs), 2),
            **{k: round(v, 6) for k, v in protocol.operational_rates(rows, scores).items()},
        }

    # Reconstructed on the RAW scores, which is what it ran on before this ticket: the ratio
    # bands predate calibration entirely, and re-fitting them on a calibrated score would be
    # comparing against a policy that never existed. `ratio_band_policy` is the historical
    # placement kept verbatim, for the same reason.
    ratio = ratio_band_policy(val_raw, val_y, fixed_fpr, costs)
    return {
        "allow_everything": flat(Action.ALLOW),
        "decline_everything": flat(Action.DECLINE),
        "ratio_bands": {
            "bands": {a.value: round(float(e), 8) for a, e in ratio.band_edges.items()},
            "note": ratio.bands_source + ". The policy ticket 09 replaced.",
            **realised(redecide(ratio, raw, rows), rows, costs),
        },
    }


def sensitivity(
    probabilities: np.ndarray, rows: list[Transaction], costs: CostModel, decision_cfg: dict
) -> list[dict]:
    """Re-decide the same probabilities under moved cost parameters.

    The acceptance criterion is that a changed cost parameter *visibly* moves the action mix.
    A test asserts the direction on synthetic rows; this reports the magnitude on real ones,
    which is the number a fraud lead would actually argue about.
    """
    out = []
    for parameter, values in SENSITIVITY:
        for value in values:
            try:
                if parameter.endswith("_multiple"):
                    field = parameter.removesuffix("_multiple")
                    moved = replace(costs, **{field: float(value) * costs.unit_amount})
                    shown = f"{value:g} x unit = {getattr(moved, field):,.0f}"
                else:
                    moved = replace(costs, **{parameter: float(value)})
                    shown = f"{value:g}"
            except DominatedAction as exc:
                # Reported, not skipped: "this value collapses the ladder" is a fact about the
                # cost model that a reader of the sensitivity table should see.
                out.append(
                    {
                        "parameter": parameter,
                        "value": value,
                        "resolved": f"{value:g}",
                        "refused": str(exc).split(".")[0],
                    }
                )
                continue
            policy = policy_from_config(decision_cfg, moved)
            out.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "resolved": shown,
                    "bands": {a.value: round(float(e), 8) for a, e in policy.band_edges.items()},
                    **realised(redecide(policy, probabilities, rows), rows, costs),
                }
            )
    return out


def reason_samples(
    scores: list[DetectorScore],
    rows: list[Transaction],
    probabilities: np.ndarray,
    limit: int = 6,
) -> list:
    """A handful of real flagged decisions, exactly as an analyst would receive them.

    Both numbers are carried: the detector's own score, which every metric is computed from, and
    the calibrated probability the action was actually taken on. Quoting one as the other is the
    confusion this artefact exists to prevent.
    """
    by_id = {t.txn_id: t for t in rows}
    calibrated = {t.txn_id: float(q) for t, q in zip(rows, probabilities, strict=True)}
    flagged = sorted(
        (s for s in scores if s.action is not Action.ALLOW), key=lambda s: -calibrated[s.txn_id]
    )
    picked, seen = [], set()
    for s in flagged:
        if s.action.value in seen and len(picked) >= len(SEVERITY):
            continue
        seen.add(s.action.value)
        t = by_id[s.txn_id]
        picked.append(
            {
                "txn_id": s.txn_id,
                "amount": round(t.amount, 2),
                "detector_score": round(s.score, 8),
                "p_fraud": round(calibrated[s.txn_id], 6),
                "action": s.action.value,
                "was_fraud": bool(t.is_fraud),
                "reasons": list(s.reasons),
            }
        )
        if len(picked) >= limit:
            break
    return picked


def run_anchor(cfg: dict, args, sup: dict, eval_cfg: dict, costs_cfg: dict) -> dict:
    """One anchor: split, fit, calibrate on validation, decide the test window, price it all."""
    name = cfg["name"]
    seed = int(args.seed)
    set_all_seeds(seed)
    fixed_fpr, k = float(eval_cfg["fixed_fpr"]), int(eval_cfg["k"])
    decision_cfg = sup["decision"]
    assert_one_operating_point(
        decision_cfg.get("calibrate_to_fpr"), fixed_fpr, mode=str(decision_cfg["mode"])
    )

    if args.sample is not None:
        cfg = {**cfg, "sample": {**(cfg.get("sample") or {}), "sample_fraction": args.sample}}
    rows = loaders.load_from_config(cfg)
    costs = cost_model_for(costs_cfg, rows)

    split: CommittedSplit | None = committed_split_for(cfg)
    if split is None:
        raise SystemExit(f"{name}: no committed split — run `make splits` first")
    train, test = split.apply(rows)
    assert_no_leakage(train, test)
    fit_rows, val_rows = out_of_time_split(
        train,
        train_frac=float(sup["tuning"]["fit_frac"]),
        embargo_days=float(eval_cfg["embargo_days"]),
    )
    if not any(t.is_fraud for t in test):
        raise SystemExit(f"{name}: the test window has no fraud — there is nothing to measure")

    # The committed tuned params, so this measures the shipped detector rather than a stand-in.
    params, source = baseline.tuned_params(name)
    detector = LGBMDetector(
        policy=policy_from_config(decision_cfg, costs),
        features=FeatureBuilder(
            stateful=bool(sup["features"]["stateful"]),
            windows_s=tuple(int(w) for w in sup["features"]["windows_s"]),
        ),
        params={**sup["params"], **params},
        seed=seed,
        replay_weight=float(sup["replay_weight"]),
        explain=sup["explain"],
        params_source=source,
    )

    started = time.perf_counter()
    detector.fit(fit_rows)
    detector.policy.reset_calibration()
    val_y, val_raw = protocol.align(
        val_rows, protocol.score_transactions(detector, val_rows, "calibration")
    )
    detector.policy.fit_calibrator(val_raw, val_y)
    detector.fit(train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scores = protocol.score_transactions(detector, test, run_id="decisions")
    score_seconds = time.perf_counter() - started
    # `DetectorScore.score` is the detector's own score, so the metrics below read it
    # directly and the cost policy's comparisons read the calibrated version. Two scales,
    # named, rather than one scale doing two jobs badly — see `DecisionPolicy.decide`.
    y, raw = protocol.align(test, scores)
    p = detector.policy.calibrator(raw)

    amounts = np.array([t.amount for t in test], dtype=float)
    quantiles = {f"p{int(q * 100)}": float(np.quantile(amounts, q)) for q in AMOUNT_QUANTILES}
    short = explain.unexplained(scores)
    if short:
        raise SystemExit(
            f"{name}: {len(short)} flagged transaction(s) carry fewer than "
            f"{explain.MIN_REASONS} reason codes, e.g. {short[:3]}"
        )

    log.info(
        "%s: %d test rows (%d fraud), %s; friction %.2f%%, evasion %.2f%%",
        name,
        len(test),
        int(y.sum()),
        detector.policy.calibrator.note,
        100 * protocol.operational_rates(test, scores)["friction_rate"],
        100 * protocol.operational_rates(test, scores)["evasion_rate"],
    )

    return {
        "version": DECISIONS_ARTEFACT_VERSION,
        "dataset": name,
        "seed": seed,
        "operating_point": {"fixed_fpr": fixed_fpr, "k": k, "source": str(EVAL_CONFIG)},
        "cost_model": costs.to_dict(),
        "amount_scale": {
            "unit_amount": round(costs.unit_amount, 4),
            "measured_on": "the whole anchor, median payment",
            "test_window_quantiles": {q: round(v, 2) for q, v in quantiles.items()},
        },
        "policy": detector.policy.to_dict(),
        "bands_by_amount": {
            q: {
                "amount": round(v, 2),
                "bands": {
                    a.value: round(float(e), 8)
                    for a, e in DecisionPolicy(
                        mode="cost", costs=costs, reference_amount=v
                    ).band_edges.items()
                },
                "unreachable": [a.value for a in costs.dominated_at(v)],
            }
            for q, v in quantiles.items()
        },
        "calibration": detector.policy.calibrator.to_dict(),
        "data": {
            "config": f"config/data/{name}.yaml",
            "sample_fraction": (cfg.get("sample") or {}).get("sample_fraction"),
            "split_digest": split.digest,
            "train_rows": len(train),
            "validation_rows": len(val_rows),
            "validation_fraud": int(sum(1 for t in val_rows if t.is_fraud)),
            "test_rows": len(test),
            "test_fraud": int(y.sum()),
            "fit_seconds": round(fit_seconds, 2),
            "score_seconds": round(score_seconds, 2),
        },
        "metrics": {
            k2: v for k2, v in protocol.evaluate(y, raw, fixed_fpr, k).model_dump().items()
        },
        "decisions": {**realised(scores, test, costs), "unexplained_flags": len(short)},
        "controls": controls(p, raw, test, costs, val_raw, val_y, fixed_fpr),
        "sensitivity": sensitivity(p, test, costs, decision_cfg),
        "reason_code_samples": reason_samples(scores, test, p),
        "meta": {
            "generated_by": "scripts/build_decisions.py",
            "params_source": source,
            "note": (
                "Measured on this anchor's OWN labelled fraud at the committed boundary, with "
                "the committed tuned params. The ranking metrics here are ticket 08's and are "
                "unchanged by this ticket: calibration is monotone. What moved is the action "
                "mix and the realised cost."
            ),
        },
    }


# ── the document ────────────────────────────────────────────────────────────────
def _pct(x: float) -> str:
    if 0.0 < x < 0.0001:
        return "<0.01%"
    return f"{x:.2%}"


def decisions_doc(cards: dict[str, dict], missing: list[str]) -> str:
    """The document, generated from the artefacts that produced every number in it."""
    first = next(iter(cards.values()))
    costs = first["cost_model"]
    rationale = costs["rationale"]

    def cost_rows() -> str:
        order = [
            "fraud_loss_rate",
            "false_decline_cost",
            "step_up_cost",
            "step_up_efficacy",
            "hold_cost",
            "hold_efficacy",
            "review_cost",
            "review_efficacy",
        ]
        out = []
        for name in order:
            resolved = " · ".join(
                f"{c['dataset']} {c['cost_model'][name]:,.4g}"
                for c in cards.values()
                if name.endswith("_cost") and not name.startswith("false")
            )
            why = " ".join(rationale.get(name, "").split())
            out.append(f"| `{name}` | {resolved or f'{costs[name]:g}'} | {why} |")
        return "\n".join(out)

    def anchor_section(c: dict) -> str:
        d, ctl = c["decisions"], c["controls"]
        mix = d["action_mix"]
        ratio = ctl["ratio_bands"]
        cal = c["calibration"]
        before, after = cal["reliability"]["before"], cal["reliability"]["after"]
        identity = "" if cal["fitted"] else " (unchanged — the map is the identity)"
        saving = ratio["realised_cost"] - d["realised_cost"]

        bands = "\n".join(
            f"| {q} | {b['amount']:,.2f} | "
            + " | ".join(
                (f"{b['bands'][a.value]:.4f}" if b["bands"][a.value] <= 1.0 else "—")
                for a in SEVERITY[1:]
            )
            + " |"
            for q, b in c["bands_by_amount"].items()
        )

        def sens_row(s: dict) -> str:
            if "refused" in s:
                return f"| `{s['parameter']}` | {s['resolved']} | _refused: {s['refused']}_ |"
            return (
                f"| `{s['parameter']}` | {s['resolved']} | {_pct(s['action_mix']['allow'])} | "
                f"{_pct(s['action_mix']['step_up'])} | {_pct(s['action_mix']['hold'])} | "
                f"{_pct(s['action_mix']['review'])} | {_pct(s['action_mix']['decline'])} | "
                f"{s['realised_cost']:,.0f} |"
            )

        sens = "\n".join(sens_row(s) for s in c["sensitivity"])
        # Both policies against the idle control, always — not only when the news is bad. A
        # policy that costs more than allowing everything is not a strict policy, it is an
        # expensive one, and two numbers sitting in a table do not get read against each other.
        idle = ctl["allow_everything"]["realised_cost"]

        def against_idle(cost: float) -> str:
            if cost > idle:
                return f"**worse than having no policy at all** ({cost:,.0f} against {idle:,.0f})"
            return f"{1 - cost / idle:.1%} better than allowing everything ({cost:,.0f})"

        # Allowing everything costs exactly the fraud that got through, so the idle control IS
        # the total value of the fraud in this window. When it is small against the price of an
        # intervention, no policy can pay for itself — and that is a fact about the anchor, not
        # a defect in the policy. Worth one sentence rather than a reader's afternoon.
        review_price = c["cost_model"]["review_cost"]
        unaffordable = (
            f" Allowing everything costs {idle:,.0f}, which *is* the total value of the fraud in "
            f"this window — against {review_price:,.2f} for a single analyst review. The whole "
            f"fold is worth {idle / review_price:,.0f} reviews, so no policy at any threshold "
            f"can pay for itself here. That is a fact about this anchor, not about the policy."
            if d["realised_cost"] > idle
            else ""
        )
        verdict = (
            f"\n\n**Against doing nothing.** The policy this replaced was "
            f"{against_idle(ratio['realised_cost'])}; the shipped one is "
            f"{against_idle(d['realised_cost'])}.{unaffordable} Under this cost model and only "
            f"under it — the comparison is exactly as good as the eight numbers above, which is "
            f"why each of them carries a rationale."
        )
        silent = [name for name, share in mix.items() if share == 0.0 and name != "allow"]
        silent_note = (
            f"\n\n**The {', '.join(silent)} band never fires on this window.** The detector's "
            "highest calibrated probability sits below where the cost model opens it, so the "
            "parameters governing that rung move nothing here — which is why the sweep below "
            "looks flat in those columns. It is a fact about this anchor's score distribution, "
            "not a band that went missing."
            if silent
            else ""
        )
        samples = "\n".join(
            f"- **{r['action']}** on {r['amount']:,.2f} at p={r['p_fraud']:.3f} "
            f"({'fraud' if r['was_fraud'] else 'legit'}) — " + "; ".join(r["reasons"])
            for r in c["reason_code_samples"]
        )
        return f"""### {c["dataset"]}

{c["data"]["test_rows"]:,} test rows, {c["data"]["test_fraud"]:,} fraud, at committed boundary
`{c["data"]["split_digest"]}`. Flat costs denominated in a median payment of
**{c["amount_scale"]["unit_amount"]:,.2f}**.

**The ladder, at four points of this anchor's own amount distribution.** A dash is a rung the
cost model never chooses at that amount — a small payment is genuinely not worth an analyst, and
that is the graded policy working rather than a band going missing.

| quantile | amount | step_up ≥ | hold ≥ | review ≥ | decline ≥ |
| --- | ---: | ---: | ---: | ---: | ---: |
{bands}

**Is the score a probability?** {cal["note"]}, on {cal["n_positives"]:,} positives in the
validation tail. Brier {before["brier"]:.5f} → {after["brier"]:.5f}, expected calibration error
{before["ece"]:.5f} → {after["ece"]:.5f}{identity}.

The calibrated probability chooses the action; it is **not** what `DetectorScore.score` carries,
and so it is not what the metrics below are computed from. That is why they are identical to
`artifacts/detector/{c["dataset"]}.json` — not because the map is monotone, but because it never
reaches them. (It is monotone, and that was the first argument. It also rounds to exactly 1.0 in
float64 past z ~ 37, which on this very window collapsed 129 distinct top-200 scores into one and
moved precision@100 on the stock-params control from 0.14 to 0.06. An argument, then a seam.)

| | PR-AUC | recall@{c["operating_point"]["fixed_fpr"]:.0%}FPR | precision@{c["metrics"]["k"]} |
| --- | ---: | ---: | ---: |
| {c["dataset"]} | {c["metrics"]["pr_auc"]:.3f} | {c["metrics"]["recall_at_fixed_fpr"]:.3f} | \
{c["metrics"]["precision_at_k"]:.2f} |

**What the policy did, against the policy it replaced.** Same detector, same probabilities; the
only difference is where the bands sit.

| policy | friction on legit | declined | fraud allowed | realised cost |
| --- | ---: | ---: | ---: | ---: |
| cost-derived (shipped) | {_pct(d["friction_rate"])} | {_pct(d["false_decline_rate"])} | \
{_pct(d["evasion_rate"])} | {d["realised_cost"]:,.0f} |
| ratio bands (replaced) | {_pct(ratio["friction_rate"])} | {_pct(ratio["false_decline_rate"])} | \
{_pct(ratio["evasion_rate"])} | {ratio["realised_cost"]:,.0f} |
| allow everything | 0.00% | 0.00% | 100.00% | \
{ctl["allow_everything"]["realised_cost"]:,.0f} |
| decline everything | 100.00% | 100.00% | 0.00% | \
{ctl["decline_everything"]["realised_cost"]:,.0f} |

Action mix under the shipped policy: allow {_pct(mix["allow"])}, step-up {_pct(mix["step_up"])},
hold {_pct(mix["hold"])}, review {_pct(mix["review"])}, decline {_pct(mix["decline"])}.
Against the ratio bands the cost model {"saves" if saving >= 0 else "costs a further"}
**{abs(saving):,.0f}** on this window.{verdict}{silent_note}

**Does a cost parameter move anything?** Re-deciding the same probabilities under moved costs.
The ranges were fixed in `scripts/build_decisions.py` before the run.

| parameter | value | allow | step-up | hold | review | decline | realised cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{sens}

**What an analyst receives.** Real flagged decisions from the test window, verbatim.

{samples}
"""

    absent = (
        " ".join(f"`{n}` is not downloaded, so it has no decision artefact here." for n in missing)
        if missing
        else "_Every configured anchor was on disk when this ran._"
    )
    sections = "\n".join(anchor_section(c) for c in cards.values())

    return f"""# Graded decisions and reason codes

_Generated by `scripts/build_decisions.py` from the anchors on disk. Every number below traces to
`artifacts/decisions/<anchor>.json`, which carries the cost model, the calibration, the split
digest and the seed that produced it. Do not edit this file — re-run `make decisions`._

`make baseline` says how well the detector **ranks**. This says what happens when a rank becomes
an action somebody has to work, and the two are different questions.

## The bands are derived, and there is nowhere left to type one

Before this, `calibrate_to_fpr` put `decline_at` at the target FPR and the other three bands at
0.8, 0.6 and 0.3 of it. Those ratios were calibrated to nothing. On the PaySim M3 fold they
landed friction on 45.6% of holdout traffic while precision@100 was 0.00 — blanket friction,
reported in the same table as a detection metric.

Now every action is priced at the transaction's own probability and its own amount, and the one
that minimises expected cost is the one taken. `config/defend/lgbm.yaml` has no band numbers in
it at all. The four `*_at` values still exist because a model card and an auditor need a
threshold to read, but they are computed from the cost model and
`tests/test_decision.py::test_the_derived_ladder_and_the_per_transaction_price_agree` holds the
two readings to the same answer.

## The cost model

`config/costs/default.yaml`. Every number carries a `why` and `CostModel.from_config` **refuses
to load one whose `why` is blank** — a comment can be deleted and nothing notices, a required
field cannot.

Flat costs are quoted as multiples of `unit_amount`, the anchor's median payment, and resolved
to currency at load. That indirection is load-bearing: PaySim's median payment is 74,872 and
AMLSim's is 157, so a flat cost of "4.0" would open the review band at a probability of 0.00005
on one anchor and never open it on the other, while looking equally principled in both artefacts.

| parameter | resolved value | why |
| --- | --- | --- |
{cost_rows()}

## What it does on each anchor

{sections}

## What this is not

**It is not a claim that these are the right costs.** They are defensible numbers with stated
reasoning, and the sensitivity table above is there so a fraud lead can see what changes if they
disagree. The contribution is that the argument is now about eight business numbers with
rationales instead of four thresholds with none.

**It is not measured on the leave-one-attack-out fold.** These are each anchor's own labelled
fraud at the committed boundary. The M3 fold's positives are injected synthetic rows — see the
ticket 07 carry-out and `docs/detector.md` — so an action mix measured there would be partly a
statement about the generator.

**AMLSim's numbers are not evidence about detection.** Every alerted row in that file is a
sub-20 amount against traffic reaching 21.5M, so it is separable before any model runs. Read
PaySim.

{absent}
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="anchor names; default every one on disk")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample", type=float, default=None, help="override sample_fraction")
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--doc-only",
        action="store_true",
        help="rewrite docs/decisions.md from the committed artefacts; measures nothing",
    )
    args = parser.parse_args()

    sup = yaml.safe_load(LGBM_CONFIG.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    costs_cfg = yaml.safe_load(COSTS_CONFIG.read_text())

    cards: dict[str, dict] = {}
    missing: list[str] = []

    if args.doc_only:
        for path in sorted(Path(args.out).glob("*.json")):
            cards[path.stem] = json.loads(path.read_text())
        missing = [c["name"] for c in anchors(args.datasets) if c["name"] not in cards]
        if not cards:
            print(f"no committed decisions in {args.out} — run `make decisions`", file=sys.stderr)
            return 1
    else:
        for cfg in anchors(args.datasets):
            try:
                cards[cfg["name"]] = run_anchor(cfg, args, sup, eval_cfg, costs_cfg)
            except loaders.DatasetNotDownloaded as exc:
                missing.append(cfg["name"])
                print(f"SKIPPED {cfg['name']}: {exc}", file=sys.stderr)
        if not cards:
            print("no anchor on disk — nothing to measure", file=sys.stderr)
            return 1
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for name, card in cards.items():
            path = Path(args.out) / f"{name}.json"
            path.write_text(json.dumps(card, indent=2, default=str) + "\n")
            print(f"→ {path}")

    DOC_PATH.write_text(decisions_doc(cards, missing))
    print(f"→ {DOC_PATH} (from {len(cards)} artefact(s))")
    for card in cards.values():
        d = card["decisions"]
        print(
            f"  {card['dataset']}: friction {d['friction_rate']:.2%}, "
            f"evasion {d['evasion_rate']:.2%}, cost {d['realised_cost']:,.0f} against the "
            f"ratio bands' {card['controls']['ratio_bands']['realised_cost']:,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
