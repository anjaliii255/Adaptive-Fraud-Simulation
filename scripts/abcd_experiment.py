"""A/B/C/D on a real held-out laundering typology: does *adaptive* augmentation beat fixed?

`C > A` is already established on this anchor — template augmentation beat real-only on the
held-out GATHER-SCATTER fold. What has never been measured is `D > C`: whether searching the
attack space against the current detector is worth more than generating the same families from
fixed parameters. That is the loop's whole reason to exist, so this is the experiment.

    A  real-only     the floor
    B  SMOTE         naive oversampling of the real fraud
    C  template      real + the strong vectors at their declared parameters
    D  adaptive      real + whatever the loop searched out, audit-gated every round

All four are fitted the same way on the same window and scored at the same operating point, so
the only thing that differs is which synthetic rows they saw. C and D get the same episode budget,
because otherwise "adaptive" would just mean "more data".

    python scripts/abcd_experiment.py --data amlworld --typology GATHER-SCATTER --seeds 1337 7
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from afl.attack import realism as realism_lib
from afl.attack.envelope import AnchorEnvelope
from afl.attack.envelope import audit as envelope_audit
from afl.attack.multi import MultiVectorOptimiser
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import committed_split_for
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import protocol
from afl.evaluation.three_system import smote_transactions
from afl.loop.closed_loop import find_evasions
from afl.utils.provenance import git_provenance
from afl.utils.runcard import environment, write_run_card
from afl.utils.seed import set_all_seeds

log = logging.getLogger("abcd")
ARTIFACT_DIR = Path("artifacts/abcd")
STRONG = ("S1", "S2", "S3")


def fit(rows: list[Transaction], seed: int, params: dict) -> LGBMDetector:
    detector = LGBMDetector(seed=seed, params=params)
    detector.fit(rows)
    return detector


def score(detector, rows: list[Transaction], fpr: float, k: int) -> dict:
    r = protocol.evaluate_detector(detector, rows, fixed_fpr=fpr, k=k)
    return {
        "pr_auc": r.pr_auc,
        "recall_at_fixed_fpr": r.recall_at_fixed_fpr,
        "precision_at_k": r.precision_at_k,
        "n_positives": r.n_positives,
    }


def amount_floor(rows: list[Transaction], fpr: float, k: int) -> dict:
    """No model at all. Every trained number has to clear this to have earned anything."""
    y = np.array([int(t.is_fraud) for t in rows])
    amounts = np.array([t.amount for t in rows], dtype=float)
    best = max(
        (protocol.evaluate(y, s, fpr, k) for s in (amounts, -amounts)), key=lambda r: r.pr_auc
    )
    return {
        "pr_auc": best.pr_auc,
        "recall_at_fixed_fpr": best.recall_at_fixed_fpr,
        "precision_at_k": best.precision_at_k,
        "n_positives": best.n_positives,
    }


def template_attacks(sim: Simulator, episodes: int) -> list[Transaction]:
    """The strong vectors at their declared parameters: augmentation with no search in it."""
    rows: list[Transaction] = []
    base = sim.n_episodes
    sim.n_episodes = max(1, round(episodes / len(STRONG)))
    for vector_id in STRONG:
        batch = sim.generate(registry.get(vector_id).to_attack_params())
        rows.extend(t for t in batch.transactions if t.is_fraud)
    sim.n_episodes = base
    return rows


def adaptive_attacks(
    sim: Simulator,
    anchor: list[Transaction],
    real_train: list[Transaction],
    real_test: list[Transaction],
    seed: int,
    rounds: int,
    episodes: int,
    args,
) -> tuple[list[Transaction], list[dict]]:
    """Run the loop, keeping every batch the audit gate let through.

    The detector inside the loop is what the search is evading; it is retrained each round so the
    attacker is always probing the current model rather than the one it started against.
    """
    # "default" reproduces the v1.0 artefact: guessed bounds, separability reported not vetoed.
    # "binding" measures the bounds off the anchor and vetoes on either audit rule.
    binding = args.leash == "binding"
    optimiser = MultiVectorOptimiser(
        vectors=STRONG,
        seed=seed,
        lambda_realism=args.lambda_realism,
        allocation=args.allocation,
        episodes_per_round=episodes,
        anchor=anchor,
        audit_rule="both" if binding else "lift",
        bounds=None if binding else realism_lib.DEFAULT_BOUNDS,
    )
    bound = optimiser.bind(sim)
    detector = fit(real_train, seed, args.params)
    kept: list[Transaction] = []
    history: list[dict] = []

    for round_index in range(rounds):
        started = time.time()
        batch = bound.generate(optimiser.propose())
        scores = detector.score(batch)
        evasions = find_evasions(batch, scores)
        optimiser.update(evasions)
        trial = optimiser.trials[-1]

        if not trial.rejected:
            kept.extend(batch.transactions)
            detector.retrain(batch, evasions)
        held_out = score(detector, real_test, args.fixed_fpr, args.k)

        history.append(
            {
                "round": round_index,
                "rejected_by_audit": trial.rejected,
                "audit_score": trial.audit_score,
                "audit_base_rate": trial.audit_base_rate,
                "allocation": trial.allocation,
                "n_fraud": trial.n_fraud,
                "n_evasions": trial.n_evasions,
                "evasion_rate": trial.evasion_rate,
                "realism_penalty": trial.realism_penalty,
                "fitness": trial.fitness,
                "seconds": round(time.time() - started, 1),
                **held_out,
            }
        )
        log.info(
            "  round %d  evasion %.3f  fitness %+.3f  held-out PR-AUC %.4f  recall %.4f%s",
            round_index,
            trial.evasion_rate,
            trial.fitness,
            held_out["pr_auc"],
            held_out["recall_at_fixed_fpr"],
            "  [AUDIT REJECTED]" if trial.rejected else "",
        )
    log.info(
        "  loop kept %d rows over %d rounds, %d rejected by the audit gate",
        len(kept),
        rounds,
        optimiser.rejected,
    )
    return kept, history


def run_seed(cfg: dict, args, seed: int) -> dict:
    set_all_seeds(seed)
    rows = loaders.load_from_config(cfg)
    split = committed_split_for(cfg)
    real_train, real_test = split.apply(rows)

    typology = loaders.amlworld_typology_by_txn()
    held = {k for k, v in typology.items() if v == args.typology}
    if not held:
        raise SystemExit(f"no rows carry typology {args.typology!r}")
    real_train = [t for t in real_train if t.txn_id not in held]
    real_test = [t for t in real_test if not t.is_fraud or t.txn_id in held]
    positives = sum(t.is_fraud for t in real_test)
    log.info(
        "seed %d — held out %s: train %d rows (%d fraud), test %d rows (%d positives, base %.5f%%)",
        seed,
        args.typology,
        len(real_train),
        sum(t.is_fraud for t in real_train),
        len(real_test),
        positives,
        100 * positives / len(real_test),
    )

    # measured on the training window only, so generated traffic cannot land in the test window
    envelope = AnchorEnvelope.measure(real_train, str(cfg["name"]))
    sim = Simulator(
        seed=seed,
        n_entities=args.n_entities,
        n_background=0,
        n_episodes=args.episodes,
        envelope=envelope,
    )

    # the same episode budget the loop will spend across all its rounds, so the only thing that
    # differs between C and D is whether the parameters were searched or declared
    templates = template_attacks(sim, args.episodes * args.rounds)
    adaptive, history = adaptive_attacks(
        sim, real_train, real_train, real_test, seed, args.rounds, args.episodes, args
    )
    audit_template = envelope_audit(real_train, [t for t in templates if t.is_fraud])
    audit_adaptive = envelope_audit(real_train, [t for t in adaptive if t.is_fraud])

    training_sets = {
        "A_real": real_train,
        "B_smote": real_train + smote_transactions(real_train, ratio=1.0, seed=seed),
        "C_template": real_train + templates,
        "D_adaptive": real_train + adaptive,
    }
    results = {}
    for name, train_rows in training_sets.items():
        started = time.time()
        detector = fit(train_rows, seed, args.params)
        results[name] = score(detector, real_test, args.fixed_fpr, args.k)
        results[name]["n_train"] = len(train_rows)
        results[name]["n_train_fraud"] = sum(t.is_fraud for t in train_rows)
        log.info(
            "  %-11s PR-AUC %.4f  recall %.4f  P@%d %.2f  (%.0fs)",
            name,
            results[name]["pr_auc"],
            results[name]["recall_at_fixed_fpr"],
            args.k,
            results[name]["precision_at_k"],
            time.time() - started,
        )
    results["amount_floor"] = amount_floor(real_test, args.fixed_fpr, args.k)

    return {
        "seed": seed,
        "positives": positives,
        "base_rate": positives / len(real_test),
        "results": results,
        "convergence": history,
        "audit": {"template": audit_template, "adaptive": audit_adaptive},
        "n_template_rows": len(templates),
        "n_adaptive_rows": len(adaptive),
    }


def sign_test(wins: int, trials: int) -> float:
    """One-sided binomial p: how surprising is `wins` of `trials` if the direction were a coin?

    Seed-to-seed spread here is larger than the gap between systems, which makes the mean the
    wrong summary — it is dominated by whichever seed swung furthest. Counting which way each
    seed fell, and asking whether that count could be chance, is what the data can support.
    """
    if trials == 0:
        return 1.0
    return sum(math.comb(trials, i) for i in range(wins, trials + 1)) / 2**trials


def report(runs: list[dict], args) -> None:
    print(f"\n{'=' * 78}\nA/B/C/D on real held-out {args.typology}, anchor {args.data}")
    base = runs[0]["base_rate"]
    print(
        f"positives in fold: {runs[0]['positives']}   base rate {base:.5%}   "
        f"operating point: recall@{args.fixed_fpr:.0%}FPR, P@{args.k}\n"
    )
    print(f"{'system':12} " + "  ".join(f"seed {r['seed']:<6}" for r in runs) + "   mean PR-AUC")
    for name in ("A_real", "B_smote", "C_template", "D_adaptive", "amount_floor"):
        cells, aucs = [], []
        for r in runs:
            m = r["results"][name]
            aucs.append(m["pr_auc"])
            cells.append(f"{m['pr_auc']:.4f}/{m['recall_at_fixed_fpr']:.3f}")
        print(f"  {name:10} " + "  ".join(f"{c:<13}" for c in cells) + f"   {np.mean(aucs):.4f}")
    print("  (PR-AUC / recall@fixed-FPR)")

    print("\nmean +/- std across seeds (the std is the point: it dwarfs the gaps)")
    for name in ("A_real", "B_smote", "C_template", "D_adaptive", "amount_floor"):
        aucs = np.array([r["results"][name]["pr_auc"] for r in runs])
        rec = np.array([r["results"][name]["recall_at_fixed_fpr"] for r in runs])
        spread = aucs.std(ddof=1) if len(aucs) > 1 else 0.0
        print(
            f"  {name:12} PR-AUC {aucs.mean():.4f} +/- {spread:.4f}   "
            f"recall {rec.mean():.4f} +/- {rec.std(ddof=1) if len(rec) > 1 else 0.0:.4f}"
        )

    for challenger, incumbent, question in (
        ("D_adaptive", "C_template", "does adaptive beat non-adaptive"),
        ("D_adaptive", "A_real", "does adaptive beat real-only"),
    ):
        print(f"\n{challenger} vs {incumbent} — {question}")
        wins = {"pr_auc": 0, "recall_at_fixed_fpr": 0, "both": 0}
        for r in runs:
            a, b = r["results"][incumbent], r["results"][challenger]
            up_auc = b["pr_auc"] > a["pr_auc"]
            up_rec = b["recall_at_fixed_fpr"] > a["recall_at_fixed_fpr"]
            wins["pr_auc"] += up_auc
            wins["recall_at_fixed_fpr"] += up_rec
            wins["both"] += up_auc and up_rec
            verdict = (
                "win"
                if up_auc and up_rec
                else "PR-AUC only"
                if up_auc
                else "recall only"
                if up_rec
                else "loss"
            )
            print(
                f"  seed {r['seed']:<6} PR-AUC {a['pr_auc']:.4f} -> {b['pr_auc']:.4f} "
                f"({b['pr_auc'] - a['pr_auc']:+.4f})   "
                f"recall {a['recall_at_fixed_fpr']:.3f} -> {b['recall_at_fixed_fpr']:.3f} "
                f"({b['recall_at_fixed_fpr'] - a['recall_at_fixed_fpr']:+.3f})   {verdict}"
            )
        n = len(runs)
        for metric in ("pr_auc", "recall_at_fixed_fpr", "both"):
            k, pval = wins[metric], sign_test(wins[metric], n)
            tail = "" if pval < 0.05 else "   (not distinguishable from a coin)"
            print(f"  sign test on {metric:19} {k}/{n} seeds   p = {pval:.3f}{tail}")


def artifact_header(cfg: dict, args) -> dict:
    """Everything a reader needs to regenerate this run, including the code that produced it.

    `git_commit` is here because a split digest and a seed are not enough: they pin the data and
    the draw, not the program. See `afl/utils/provenance.py` for what this cost to learn.
    """
    return {
        "anchor": args.data,
        "typology": args.typology,
        "split_digest": committed_split_for(cfg).digest,
        **git_provenance(),
        # the commit pins the program; the environment pins the arithmetic. LightGBM's version
        # is the first thing to compare when two machines disagree on a number.
        "environment": environment(),
        "operating_point": {"fixed_fpr": args.fixed_fpr, "k": args.k},
        "allocation": args.allocation,
        "leash": args.leash,
        "lambda_realism": args.lambda_realism,
        "audit_rule": "both" if args.leash == "binding" else "lift",
        "rounds": args.rounds,
        "episodes_per_round": args.episodes,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="amlworld")
    p.add_argument("--typology", default="GATHER-SCATTER")
    p.add_argument("--seeds", type=int, nargs="+", default=[1337, 7])
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--n-entities", type=int, default=400)
    p.add_argument("--allocation", default="search", help="uniform | search | fitness")
    p.add_argument("--lambda-realism", type=float, default=0.5)
    p.add_argument("--fixed-fpr", type=float, default=0.01)
    p.add_argument("-k", type=int, default=100)
    p.add_argument(
        "--append",
        action="store_true",
        help="keep runs already in the artefact whose seeds are not being re-run, so adding "
        "seeds does not mean paying again for the ones already measured",
    )
    p.add_argument(
        "--leash",
        default="default",
        choices=("default", "binding"),
        help="'default' reproduces the v1.0 artefact. 'binding' measures the realism bounds off "
        "the anchor and makes separability a veto rather than a note.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="artefact path; defaults to <anchor>_<typology>.json. Set this when a run is not "
        "meant to replace the committed one.",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(Path(f"config/data/{args.data}.yaml").read_text())
    args.params = yaml.safe_load(Path("config/defend/lgbm.yaml").read_text()).get("params") or {}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (ARTIFACT_DIR / f"{args.data}_{args.typology.lower()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = [run_seed(cfg, args, seed) for seed in args.seeds]
    if args.append and out.exists():
        previous = json.loads(out.read_text()).get("runs", [])
        fresh = {r["seed"] for r in runs}
        runs = sorted(
            [*(r for r in previous if r["seed"] not in fresh), *runs], key=lambda r: r["seed"]
        )
        log.info("appended: %d seeds now in the artefact", len(runs))
    report(runs, args)
    out.write_text(
        json.dumps(
            {
                **artifact_header(cfg, args),
                "runs": runs,
            },
            indent=2,
            default=str,
        )
    )
    write_run_card(
        out.parent,
        seed=args.seeds[0] if len(args.seeds) == 1 else None,
        config={"data": cfg, "args": {k: str(v) for k, v in vars(args).items()}},
        attack_params={"vectors": list(STRONG), "detector_params": args.params},
        metrics={r["seed"]: r["results"] for r in runs},
        name=f"{out.stem}_run_card.json",
        seeds=args.seeds,
    )
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
