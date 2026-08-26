"""The fidelity scorecard, on the real anchor, in one command.

    python scripts/build_fidelity.py                 # every real anchor, all three levels
    python scripts/build_fidelity.py paysim          # just one anchor
    python scripts/build_fidelity.py --sample 0.02   # a quick pass
    python scripts/build_fidelity.py --selftest      # the day-one discrimination check
    python scripts/build_fidelity.py --doc-only      # rewrite the doc from committed artefacts

Ticket 15's deliverable. Three levels and a privacy panel, measured against the anchor the rest
of the project is measured on, written to `artifacts/fidelity/<anchor>.json` and to
`docs/fidelity.md`, which is generated from those files and never hand-typed.

**The levels are not equal, and the scorecard does not pretend they are.** Level 3 — does
training on this data teach a detector anything real — is the gate. Levels 1 and 2 are
diagnostics that explain *why* level 3 landed where it did. A generator that resembles real
traffic and teaches a model nothing has failed, however pretty its histograms, and neither the
verdict nor the headline score can be argued upwards from them.

**The bars are checked, not asserted.** They live in `config/fidelity/thresholds.yaml`, each
with a stated reason and an origin commit, and `afl/fidelity/provenance.py` reads that commit
back out of git on every run to confirm the values have not moved since. The check is in the
artefact next to the verdict it produced, so "thresholds were set in advance" is a claim a
reader can audit rather than one they have to take.

**A failing card is a result.** This script writes the artefact and the doc first and exits
non-zero afterwards, so a FAIL is committed rather than re-run at a friendlier setting. If a
threshold ever does move, the provenance block names it, says which direction it moved, and
prints the commit that moved it.

Two comparisons run at levels 1 and 2, because on a real anchor "the generated traffic" is two
different things. The **headline** is generated fraud against the anchor's own labelled fraud —
like against like, and the only part of the batch anything downstream uses, since an anchored
run discards the simulator's background and injects the attacks into real traffic. The whole
batch against the whole anchor is measured too and reported underneath, because it is what the
phrase usually means and a reader is owed both.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import yaml
from omegaconf import OmegaConf

# The same assembly the loop and the leave-one-attack-out matrix use. A fidelity scorecard that
# judged a differently-built simulator, or a differently-parameterised detector, would be
# judging something nobody ships.
from run_experiment import (  # noqa: E402
    build_detector_factory,
    build_simulator,
)

from afl.attack.envelope import AnchorEnvelope
from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import committed_split_for, out_of_time_split
from afl.defend.models.lgbm import model_card_of
from afl.fidelity import level1_statistical, level2_structural, provenance, scorecard
from afl.utils.seed import rng as make_rng
from afl.utils.seed import set_all_seeds

log = logging.getLogger("build_fidelity")

DATA_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
ANOMALY_CONFIG = Path("config/defend/anomaly.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
ENGINES_CONFIG = Path("config/attack/engines.yaml")
COSTS_CONFIG = Path("config/costs/default.yaml")
THRESHOLDS_CONFIG = Path("config/fidelity/thresholds.yaml")
ARTIFACT_DIR = Path("artifacts/fidelity")
DOC_PATH = Path("docs/fidelity.md")

#: Rows per side for levels 1, 2 and privacy. Every distance there is distributional, so more
#: rows buy precision rather than a different answer — and the privacy checks are O(n²) while
#: level 2 enumerates short cycles, which on 600k rows is not a wait, it is a different project.
#: Recorded in the artefact, because a sample size that is not written down is a sample size
#: somebody will assume was the whole file.
COMPARISON_ROWS = 20_000


def anchors(selected: list[str]) -> list[dict]:
    """Every data config that names a loader — the real anchors, not the synthetic default."""
    out = []
    for path in sorted(DATA_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("loader") and (not selected or cfg["name"] in selected):
            out.append(cfg)
    return out


def compose(cfg: dict, seed: int):
    """The same config tree `config/config.yaml` composes, built here from the same files."""
    return OmegaConf.create(
        {
            "seed": seed,
            "data": cfg,
            "attack": {"engines": yaml.safe_load(ENGINES_CONFIG.read_text())},
            "defend": {
                "supervised": yaml.safe_load(LGBM_CONFIG.read_text()),
                "unsupervised": yaml.safe_load(ANOMALY_CONFIG.read_text()),
            },
            "eval": yaml.safe_load(EVAL_CONFIG.read_text()),
            "costs": yaml.safe_load(COSTS_CONFIG.read_text()),
        }
    )


def subsample(rows: list[Transaction], n: int, seed: int) -> list[Transaction]:
    """A seeded sample that keeps the time order — the pacing metrics read gaps between rows."""
    if len(rows) <= n:
        return rows
    idx = make_rng(seed).choice(len(rows), size=n, replace=False)
    return [rows[i] for i in sorted(idx)]


def stats(rows: list[Transaction]) -> dict:
    fraud = sum(1 for t in rows if t.is_fraud)
    return {
        "rows": len(rows),
        "fraud": fraud,
        "base_rate": round(fraud / len(rows), 8) if rows else 0.0,
        "first_ts": min((t.ts for t in rows), default=None),
        "last_ts": max((t.ts for t in rows), default=None),
    }


def generate(cfg, real: list[Transaction], envelope: AnchorEnvelope) -> dict[str, list]:
    """Every vector the registry can generate today, anchored to this anchor's own envelope.

    Not the holdout alone. The scorecard is about the generator, and a generator judged on one
    family is a generator judged on its easiest one.
    """
    from afl.attack.templates import registry

    simulator = build_simulator(cfg, anchor=real, envelope=envelope)
    batches: dict[str, list[Transaction]] = {}
    for spec in registry.list_vectors(generatable=True):
        batch = simulator.generate(spec.to_attack_params())
        batches[spec.vector_id] = batch.transactions
        log.info(
            "  %s: %d rows, %d fraud",
            spec.vector_id,
            len(batch.transactions),
            sum(1 for t in batch.transactions if t.is_fraud),
        )
    return batches


def secondary_comparison(real: list[Transaction], synth: list[Transaction], seed: int) -> dict:
    """Levels 1 and 2 on the whole batch against the whole anchor.

    Reported rather than gated. An anchored run throws the simulator's background away and
    injects the attacks into real traffic, so most of what this comparison measures is a
    placeholder haystack nothing downstream ever sees — but it is what "generated traffic
    against the real anchor" usually means, so it is measured and shown.
    """
    r = subsample(real, COMPARISON_ROWS, seed)
    s = subsample(synth, COMPARISON_ROWS, seed + 1)
    l1 = level1_statistical.report(r, s)
    l2 = level2_structural.report(r, s)
    return {
        "what": "the generator's whole output against the anchor's whole traffic",
        "why_not_the_headline": (
            "an anchored run discards the simulator's background and injects the attacks into "
            "real traffic, so this comparison is mostly about a haystack nothing downstream uses"
        ),
        "n_real": len(r),
        "n_synth": len(s),
        "level1": {"score": l1["score"], "worst_column": l1["worst_column"], "ks": l1["ks"]},
        "level2": {
            "score": l2["score"],
            "worst_motif": l2["worst_motif"],
            "support": l2["support"],
        },
    }


def run_anchor(cfg: dict, args, values: dict, prov) -> tuple[scorecard.Scorecard, dict]:
    """One anchor's whole scorecard, at the operating point everything else here is measured at."""
    name = cfg["name"]
    seed = int(args.seed)
    set_all_seeds(seed)
    started = time.perf_counter()

    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    fixed_fpr, k = float(eval_cfg["fixed_fpr"]), int(eval_cfg["k"])
    min_positives = int(eval_cfg["min_meaningful_positives"])

    if args.sample is not None:
        cfg = {**cfg, "sample": {**(cfg.get("sample") or {}), "sample_fraction": args.sample}}
    real = loaders.load_from_config(cfg)
    log.info("%s: %d real rows, %d fraud", name, len(real), sum(1 for t in real if t.is_fraud))

    split = committed_split_for(cfg)
    if split is None:
        raise SystemExit(f"{name}: no committed split — run `make splits` first")
    real_train, real_test = split.apply(real)
    log.info(
        "  committed split %s: train %d (%d fraud) / test %d (%d fraud)",
        split.digest,
        len(real_train),
        sum(1 for t in real_train if t.is_fraud),
        len(real_test),
        sum(1 for t in real_test if t.is_fraud),
    )

    composed = compose(cfg, seed)
    envelope = AnchorEnvelope.measure(real, name)
    batches = generate(composed, real, envelope)
    synth_all = [t for b in batches.values() for t in b]
    synth_fraud = [t for t in synth_all if t.is_fraud]
    real_fraud = [t for t in real if t.is_fraud]

    # The headline levels 1 and 2 comparison: generated fraud against the anchor's own fraud.
    # Like against like — this is the part of the batch an anchored run actually injects.
    detector_factory = build_detector_factory(composed, real)
    card = scorecard.build(
        real=subsample(real_fraud, COMPARISON_ROWS, seed),
        synth=subsample(synth_fraud, COMPARISON_ROWS, seed + 1),
        real_train=real_train,
        real_test=real_test,
        detector_factory=detector_factory,
        thresholds=scorecard.Thresholds.from_values(values),
        seed=seed,
        fixed_fpr=fixed_fpr,
        k=k,
        synth_fraud=synth_fraud,
        standalone=synth_all,
        min_positives=min_positives,
        provenance=prov.to_dict(),
        meta={
            "title": f"Fidelity scorecard — {name}",
            "generated_by": "scripts/build_fidelity.py",
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "anchor": name,
            "data_config": f"config/data/{name}.yaml",
            "seed": seed,
            "sample_fraction": (cfg.get("sample") or {}).get("sample_fraction"),
            "operating_point": {"fixed_fpr": fixed_fpr, "k": k, "source": str(EVAL_CONFIG)},
            "split": split.to_dict(),
            "comparison": {
                "headline": "the generated fraud against the anchor's own labelled fraud",
                "why": (
                    "like against like: an anchored run injects the generated fraud into real "
                    "traffic and discards the simulator's background, so this is the part of "
                    "the batch anything downstream sees"
                ),
                "rows_per_side": COMPARISON_ROWS,
            },
            "real": stats(real),
            "real_fraud": stats(real_fraud),
            "generated": {
                "vectors": sorted(batches),
                "total": stats(synth_all),
                "fraud": stats(synth_fraud),
                "per_vector": {v: stats(b) for v, b in batches.items()},
            },
            "envelope": {
                "start": str(envelope.start),
                "window_days": envelope.window_days,
                "sender_reuse_rate": round(envelope.sender_reuse_rate, 6),
                "supports_behavioural_vectors": envelope.supports_behavioural_vectors,
            },
            "detector": model_card_of(detector_factory()),
            "note": (
                "Levels 1 and 2 are diagnostics; level 3 is the gate. The decision layer is not "
                "fitted for level 3 — every number on it is a ranking metric, which the decision "
                "layer cannot move by construction (docs/decisions.md)."
            ),
        },
    )
    card.meta["also_measured"] = secondary_comparison(real, synth_all, seed)
    card.meta["seconds"] = round(time.perf_counter() - started, 1)
    return card, {"split": split, "real": real}


# ── the document ────────────────────────────────────────────────────────────────
def _wrap(text: str, indent: str = "") -> str:
    return "\n".join(
        textwrap.fill(p.strip(), width=96, initial_indent=indent, subsequent_indent=indent)
        for p in text.strip().split("\n\n")
    )


def _score_cell(card: dict, key: str) -> str:
    """A level's score, or the word for what happened instead. Never a blank cell."""
    body = (card.get("levels") or {}).get(key) or {}
    if not body:
        return "not run"
    score = body.get("score")
    return "withheld" if score is None else str(score)


def _verdict_line(card: dict) -> str:
    marks = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    return f"{marks.get(card['verdict'], '')} **{card['verdict'].upper()}** — score {card['score']}"


def fidelity_doc(cards: dict[str, dict], missing: list[str]) -> str:
    """`docs/fidelity.md`, generated from the committed artefacts and never hand-typed."""
    out = [
        "# Fidelity, on the real anchor",
        "",
        _wrap(
            "Generated by `make fidelity` from `artifacts/fidelity/*.json`. Every number below "
            "traces to one of those files; nothing here is typed by hand."
        ),
        "",
        _wrap(
            "Three levels and a privacy panel. **They are not equal.** Level 3 asks whether "
            "training on the generated data teaches a detector anything about real fraud, and it "
            "is the gate. Levels 1 and 2 ask whether the traffic resembles the anchor, and they "
            "are diagnostics: they explain why level 3 landed where it did, and they can neither "
            "fail a generator nor rescue one. The headline score is capped at the level-3 score "
            "for that reason — a reader who quotes the number instead of the verdict still "
            "cannot be handed a pass two sets of histograms averaged into existence."
        ),
        "",
        "| anchor | verdict | level 1 | level 2 | level 3 (**gate**) | privacy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, card in sorted(cards.items()):
        cells = [_score_cell(card, key) for key in ("level1", "level2", "level3", "privacy")]
        out.append(f"| {name} | {_verdict_line(card)} | " + " | ".join(cells) + " |")

    for name, card in sorted(cards.items()):
        out += _anchor_section(name, card)

    if missing:
        out += [
            "",
            "## Not measured",
            "",
            _wrap(
                "These anchors have a data config but no scorecard in `artifacts/fidelity/`: "
                + ", ".join(sorted(missing))
                + ". An anchor that is absent from a table reads as one that passed."
            ),
        ]

    out += [
        "",
        "## What this does not claim",
        "",
        _wrap(
            "Levels 1 and 2 are diagnostics, not proofs of realism. A distance near zero means "
            "the statistics we chose to measure agree; it does not mean a fraud analyst would be "
            "fooled, and it never means the data is safe to publish."
        ),
        "",
        _wrap(
            "DCR and MIA are evidence against memorisation, not a privacy guarantee. They say "
            "the two leakage paths we tested for are not present. A formal claim needs "
            "differential privacy, and this project does not make one — see the identifier-reuse "
            "line in each panel for the disclosure path neither of them can see."
        ),
        "",
        _wrap(
            "A failing scorecard stays failed. The artefacts are written before this script exits "
            "non-zero, and the thresholds carry an origin commit that every run reads back out "
            "of git, so a bar that moves after a result exists is visible in the artefact rather "
            "than in a blame trail."
        ),
        "",
    ]
    return "\n".join(out)


def _anchor_section(name: str, card: dict) -> list[str]:
    meta, lv = card.get("meta", {}), card.get("levels", {})
    l3 = lv.get("level3") or {}
    gen = meta.get("generated", {})
    out = [
        "",
        f"## {name}",
        "",
        _verdict_line(card),
        "",
        _wrap(
            f"{meta.get('real', {}).get('rows', 0):,} real rows "
            f"({meta.get('real', {}).get('fraud', 0):,} fraud) against "
            f"{gen.get('fraud', {}).get('rows', 0):,} generated fraud rows over "
            f"{len(gen.get('vectors') or [])} vectors, at fixed FPR "
            f"{meta.get('operating_point', {}).get('fixed_fpr')} and k="
            f"{meta.get('operating_point', {}).get('k')}."
        ),
    ]
    if card.get("reasons"):
        out += ["", "**Why:**", ""] + [f"- {r}" for r in card["reasons"]]

    if l3.get("outcome") == "measured":
        out += [
            "",
            "| system | trained on | rows | PR-AUC | recall@FPR | p@k | beats the floor |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        beats = l3.get("beats_amount_floor", {})
        for sys_name, body in (l3.get("systems") or {}).items():
            clears = beats.get(sys_name)
            out.append(
                f"| {sys_name} | {body.get('trained_on')} | {body.get('n_train'):,} | "
                f"{body.get('pr_auc')} | {body.get('recall_at_fixed_fpr')} | "
                f"{body.get('precision_at_k')} | "
                f"{'—' if clears is None else ('yes' if clears else '**no**')} |"
            )
    elif l3:
        out += ["", _wrap(f"Level 3 withheld — {l3.get('why')}.")]

    also = meta.get("also_measured") or {}
    if also:
        out += [
            "",
            _wrap(
                f"Levels 1 and 2 above compare the generated fraud with the anchor's own fraud. "
                f"The same two levels on the whole batch against the whole anchor land at "
                f"{also.get('level1', {}).get('score')} and "
                f"{also.get('level2', {}).get('score')}; "
                f"{also.get('why_not_the_headline')}."
            ),
        ]
    p = lv.get("privacy") or {}
    if p:
        ids = p.get("identifier_reuse", {})
        out += [
            "",
            _wrap(
                f"Privacy, as evidence rather than proof: DCR ratio "
                f"{p.get('dcr', {}).get('dcr_ratio')}, exact duplicates of training rows "
                f"{p.get('dcr', {}).get('identical_share')}, membership-inference advantage "
                f"{p.get('mia', {}).get('advantage')}. "
                f"{ids.get('either_in_anchor')} of generated rows name an account that exists in "
                f"the anchor, which is the envelope working as designed and is reported rather "
                f"than flagged."
            ),
        ]
    prov = card.get("threshold_provenance") or {}
    if prov:
        out += ["", _wrap(f"Thresholds: {prov.get('verdict')}.")]
    out += ["", f"Artefact: `artifacts/fidelity/{name}.json` · `artifacts/fidelity/{name}.md`"]
    return out


# ── the day-one self-test ───────────────────────────────────────────────────────
def _copy(txns: list[Transaction]) -> list[Transaction]:
    return [t.model_copy(update={"txn_id": f"copy-{t.txn_id}"}) for t in txns]


def _shuffled(txns: list[Transaction], seed: int) -> list[Transaction]:
    """Every marginal preserved exactly; every association between them destroyed."""
    r = make_rng(seed)
    amounts = r.permutation([t.amount for t in txns])
    srcs = r.permutation([t.src for t in txns])
    dsts = r.permutation([t.dst for t in txns])
    return [
        t.model_copy(
            update={
                "txn_id": f"shuf-{i}",
                "amount": float(amounts[i]),
                "src": str(srcs[i]),
                "dst": str(dsts[i]) if str(dsts[i]) != str(srcs[i]) else t.dst,
            }
        )
        for i, t in enumerate(txns)
    ]


def _noise(txns: list[Transaction], seed: int) -> list[Transaction]:
    r = make_rng(seed)
    t0 = min(t.ts for t in txns)
    return [
        t.model_copy(
            update={
                "txn_id": f"noise-{i}",
                "amount": round(float(r.uniform(1, 50_000)), 2),
                "ts": t0 + timedelta(seconds=float(r.uniform(0, 30 * 86_400))),
                "src": f"n{int(r.integers(0, 400)):05d}",
                "dst": f"n{int(r.integers(400, 800)):05d}",
            }
        )
        for i, t in enumerate(txns)
    ]


def selftest(args, values: dict, prov) -> int:
    """Prove the harness discriminates, on three cases whose answers are known in advance.

    A harness written after the generator gets its thresholds chosen to fit the results, so this
    ran before there was a generator to judge and it still runs:

        copy      real data duplicated          -> scores high, and trips the privacy check
        shuffled  marginals kept, joins broken  -> passes level 1, fails level 2
        noise     nothing preserved             -> fails everything

    If the scorecard cannot separate those three it cannot be trusted on a real generator, and
    no anchored number it produces is worth reading.
    """
    from afl.attack.simulator import Simulator
    from afl.attack.templates import registry
    from afl.defend.models.lgbm import LGBMDetector

    set_all_seeds(args.seed)
    sim = Simulator(seed=args.seed, n_background=args.n, n_episodes=3)
    batch = sim.generate(registry.get("S1").to_attack_params())
    real = batch.transactions
    real_train, real_test = out_of_time_split(real, train_frac=0.7, embargo_days=1.0)

    def detector_factory():
        return LGBMDetector(seed=args.seed, params={"n_estimators": 60})

    cases = {
        "copy": _copy(real),
        "shuffled": _shuffled(real, args.seed),
        "noise": _noise(real, args.seed),
    }

    out = args.out / "selftest"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case, synth in cases.items():
        card = scorecard.build(
            real=real,
            synth=synth,
            real_train=real_train,
            real_test=real_test,
            detector_factory=detector_factory,
            thresholds=scorecard.Thresholds.from_values(values),
            seed=args.seed,
            provenance=prov.to_dict(),
            meta={"title": f"Fidelity harness self-test — {case}", "case": case},
        )
        card.save(out, stem=case)
        summary[case] = {
            "verdict": card.verdict,
            "score": card.score,
            "level1": card.levels["level1"]["score"],
            "level2": card.levels["level2"]["score"],
            "level3": (card.levels.get("level3") or {}).get("score"),
            "privacy": (card.levels.get("privacy") or {}).get("score"),
            "reasons": card.reasons,
        }
        print(
            f"{case:9s} verdict={card.verdict:5s} score={card.score:6.3f} "
            f"L1={summary[case]['level1']:6.3f} L2={summary[case]['level2']:6.3f}"
        )

    # the discrimination the harness exists to provide
    checks = {
        "copy scores above noise": summary["copy"]["score"] > summary["noise"]["score"],
        "shuffled scores above noise": summary["shuffled"]["score"] > summary["noise"]["score"],
        "shuffled loses more structure than marginals": summary["shuffled"]["level2"]
        < summary["shuffled"]["level1"],
        "copying is flagged by privacy": bool(
            np.isclose(summary["copy"]["privacy"] or 0.0, 0.0, atol=0.35)
        ),
    }
    (out / "selftest.json").write_text(json.dumps({"cases": summary, "checks": checks}, indent=2))
    print()
    for label, ok in checks.items():
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}")

    failed = [k for k, v in checks.items() if not v]
    if failed:
        print(f"\nharness is not discriminating: {failed}")
        return 1
    print(f"\nharness ok -> {out}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("anchors", nargs="*", help="anchors to score (default: every real one)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--sample", type=float, default=None, help="override the anchor's sample_fraction"
    )
    p.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    p.add_argument("--doc", type=Path, default=DOC_PATH)
    p.add_argument("--selftest", action="store_true", help="the day-one discrimination check")
    p.add_argument("--n", type=int, default=3_000, help="self-test background rows")
    p.add_argument("--doc-only", action="store_true", help="rewrite the doc from the artefacts")
    args = p.parse_args()

    # The bars first, and the evidence about when they were set — before a single number exists.
    values, _why, prov = provenance.load(THRESHOLDS_CONFIG)
    log.info("thresholds: %s", prov.verdict)
    if not prov.predates_results:
        log.warning(
            "the provenance check cannot show these bars predate this run; the artefact will say "
            "so rather than claim otherwise"
        )

    if args.selftest:
        return selftest(args, values, prov)

    if not args.doc_only:
        selected = anchors(args.anchors)
        if not selected:
            raise SystemExit(f"no real anchor matched {args.anchors or 'any data config'}")
        for cfg in selected:
            card, _ = run_anchor(cfg, args, values, prov)
            paths = card.save(args.out, stem=cfg["name"])
            print(f"\n{cfg['name']}: {card.verdict.upper()} (score {card.score})")
            for reason in card.reasons:
                print(f"  - {reason}")
            print(f"  -> {paths['json']}")

    cards = {
        path.stem: json.loads(path.read_text())
        for path in sorted(args.out.glob("*.json"))
        if path.stem != "selftest"
    }
    if not cards:
        log.warning("no scorecards in %s — nothing to write a doc from", args.out)
        return 1
    missing = [c["name"] for c in anchors([]) if c["name"] not in cards]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(fidelity_doc(cards, missing))
    print(f"\ndoc -> {args.doc}")

    # The artefacts are written and the doc is rewritten *before* this line. A failing scorecard
    # is a committed result, never a run that quietly did not happen.
    failed = sorted(name for name, card in cards.items() if card["verdict"] == "fail")
    if failed:
        print(f"\nFIDELITY FAILED on: {', '.join(failed)} — reported, not re-run at a lower bar")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
