"""Does a detector trained on our synthetic attacks catch REAL fraud?

The adaptive system scores PR-AUC 0.819 on the held-out synthetic family. That is either the
project's result or its most expensive mistake, and the held-out number alone cannot tell the
two apart: a model that has learned "the simulator wrote this row" scores exactly the same as
one that has learned what fraud looks like.

So train a detector that has never seen a real fraud label — real legit traffic plus synthetic
attacks only — and score it against the anchor's own SAR rows. Fraud behaviour transfers.
Provenance does not.

    python scripts/transfer_test.py --data amlsim

Four systems, all fitted on the training window and all scored on the same test window:

    real         real rows, real labels               the ceiling: what is learnable here
    synthetic    real legit + synthetic attacks       the transfer test: no real fraud label
    both         real rows + synthetic attacks        does synthetic add to real?
    amount       no model at all                      the floor, as in build_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from afl.attack.envelope import AnchorEnvelope
from afl.attack.envelope import audit as envelope_audit
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import committed_split_for
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import protocol
from afl.utils.seed import set_all_seeds

log = logging.getLogger("transfer")
ARTIFACT_DIR = Path("artifacts/transfer")


def synthetic_attacks(
    sim: Simulator, vectors: list[str], episodes: int
) -> dict[str, list[Transaction]]:
    """One batch per vector, fraud rows only: the real anchor already supplies the haystack."""
    out = {}
    sim.n_episodes = episodes
    for vid in vectors:
        batch = sim.generate(registry.get(vid).to_attack_params())
        out[vid] = [t for t in batch.transactions if t.is_fraud]
    return out


def fit(rows: list[Transaction], seed: int, params: dict) -> LGBMDetector:
    detector = LGBMDetector(seed=seed, params=params)
    detector.fit(rows)
    return detector


def score(detector: LGBMDetector, rows: list[Transaction], fixed_fpr: float, k: int) -> dict:
    result = protocol.evaluate_detector(detector, rows, fixed_fpr=fixed_fpr, k=k)
    return {
        "pr_auc": result.pr_auc,
        "recall_at_fixed_fpr": result.recall_at_fixed_fpr,
        "precision_at_k": result.precision_at_k,
        "n_positives": result.n_positives,
    }


def amount_floor(rows: list[Transaction], fixed_fpr: float, k: int) -> dict:
    """No model, no features, no training: the reference every trained number must clear."""
    y = np.array([int(t.is_fraud) for t in rows])
    amounts = np.array([t.amount for t in rows], dtype=float)
    best = max(
        (protocol.evaluate(y, s, fixed_fpr, k) for s in (amounts, -amounts)),
        key=lambda r: r.pr_auc,
    )
    return {
        "pr_auc": best.pr_auc,
        "recall_at_fixed_fpr": best.recall_at_fixed_fpr,
        "precision_at_k": best.precision_at_k,
        "n_positives": best.n_positives,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="amlsim", help="anchor to test transfer against")
    p.add_argument("--holdout", default="M3")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--fixed-fpr", type=float, default=0.01)
    p.add_argument("-k", type=int, default=100)
    args = p.parse_args()
    set_all_seeds(args.seed)

    data_cfg = yaml.safe_load(Path(f"config/data/{args.data}.yaml").read_text())
    engines = yaml.safe_load(Path("config/attack/engines.yaml").read_text())
    lgbm_cfg = yaml.safe_load(Path("config/defend/lgbm.yaml").read_text())

    real = loaders.load_from_config(data_cfg)
    envelope = AnchorEnvelope.measure(real, args.data)
    log.info(
        "anchor %s: %d rows, %.4f%% fraud, sender reuse %.4f, behavioural-capable=%s",
        args.data,
        len(real),
        100 * envelope.fraud_base_rate,
        envelope.sender_reuse_rate,
        envelope.supports_behavioural_vectors,
    )
    if not envelope.supports_behavioural_vectors:
        log.warning("this anchor cannot host behavioural vectors; transfer is not measurable here")

    split = committed_split_for(data_cfg)
    if split is None:
        raise SystemExit(f"{args.data} has no committed split; run `make splits` first")
    real_train, real_test = split.apply(real)
    log.info(
        "committed split: train %d rows, test %d rows (%d real fraud in test)",
        len(real_train),
        len(real_test),
        sum(t.is_fraud for t in real_test),
    )

    sim = Simulator(
        seed=args.seed,
        n_entities=int(engines["n_entities"]),
        n_background=0,  # the anchor is the haystack
        n_episodes=int(engines["n_episodes"]),
        envelope=envelope,
    )
    trainable = [v for v in engines["vectors"] if v != args.holdout]
    attacks = synthetic_attacks(sim, trainable, args.episodes)
    holdout_rows = synthetic_attacks(sim, [args.holdout], args.episodes)[args.holdout]
    synth_train = [t for rows in attacks.values() for t in rows]
    log.info(
        "synthetic: %d attack rows over %s, %d held-out %s rows",
        len(synth_train),
        ",".join(trainable),
        len(holdout_rows),
        args.holdout,
    )

    # keep the synthetic rows on the training side of the boundary, so nothing is trained on
    # traffic contemporaneous with the test window
    synth_train = [t for t in synth_train if t.ts <= split.train_end]
    real_legit_train = [t for t in real_train if not t.is_fraud]

    audit = envelope_audit(real, holdout_rows)
    log.info(
        "commensurability: worst field %r at %.4f (base %.4f) separable=%s",
        audit["worst"],
        audit["score"],
        audit["base_rate"],
        audit["trivially_separable"],
    )

    params = dict(lgbm_cfg.get("params") or {})
    training_sets = {
        "real": real_train,
        "synthetic": real_legit_train + synth_train,
        "both": real_train + synth_train,
    }
    # the two questions, on the same models: real fraud in the test window, and our own holdout
    test_sets = {
        "real_fraud": real_test,
        f"synthetic_{args.holdout}": [t for t in real_test if not t.is_fraud] + holdout_rows,
    }

    results: dict[str, dict[str, dict]] = {}
    for train_name, rows in training_sets.items():
        n_fraud = sum(t.is_fraud for t in rows)
        log.info("fitting %-10s %d rows, %d fraud", train_name, len(rows), n_fraud)
        detector = fit(rows, args.seed, params)
        results[train_name] = {
            test_name: score(detector, test_rows, args.fixed_fpr, args.k)
            for test_name, test_rows in test_sets.items()
        }
    results["amount_floor"] = {
        name: amount_floor(rows, args.fixed_fpr, args.k) for name, rows in test_sets.items()
    }

    report(results, test_sets, args)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / f"{args.data}.json"
    out.write_text(
        json.dumps(
            {
                "anchor": args.data,
                "holdout": args.holdout,
                "seed": args.seed,
                "operating_point": {"fixed_fpr": args.fixed_fpr, "k": args.k},
                "split_digest": split.digest,
                "commensurability": audit,
                "results": results,
            },
            indent=2,
            default=str,
        )
    )
    log.info("\nwritten to %s", out)
    return 0


def report(results: dict, test_sets: dict, args) -> None:
    real_key, synth_key = list(test_sets)
    log.info("\n%-14s %-34s %-34s", "", f"tested on {real_key}", f"tested on {synth_key}")
    log.info(
        "%-14s %-34s %-34s", "trained on", "PR-AUC  recall@fpr  P@k", "PR-AUC  recall@fpr  P@k"
    )
    for name, by_test in results.items():
        a, b = by_test[real_key], by_test[synth_key]
        log.info(
            "%-14s %6.4f  %10.4f  %5.2f          %6.4f  %10.4f  %5.2f",
            name,
            a["pr_auc"],
            a["recall_at_fixed_fpr"],
            a["precision_at_k"],
            b["pr_auc"],
            b["recall_at_fixed_fpr"],
            b["precision_at_k"],
        )

    synthetic_on_real = results["synthetic"][real_key]["pr_auc"]
    synthetic_on_synth = results["synthetic"][synth_key]["pr_auc"]
    real_on_real = results["real"][real_key]["pr_auc"]
    log.info(
        "\nVERDICT: a detector trained without one real fraud label scores %.4f on real fraud, "
        "against %.4f for the same model on our own holdout and %.4f for a detector trained on "
        "real labels.",
        synthetic_on_real,
        synthetic_on_synth,
        real_on_real,
    )
    # the decisive comparison: transfer has to clear the reference that needs no model at all
    real_on_synth = results["real"][synth_key]["pr_auc"]
    floor_on_real = results["amount_floor"][real_key]["pr_auc"]
    if synthetic_on_real <= floor_on_real:
        log.warning(
            "Transfer does not clear the floor: %.4f on real fraud against %.4f for sorting on "
            "amount alone. Whatever the synthetic attacks taught is not worth a model.",
            synthetic_on_real,
            floor_on_real,
        )
    if synthetic_on_synth > 2 * max(synthetic_on_real, 1e-9):
        log.warning(
            "The synthetic holdout number is %.1fx the real-fraud number from the same model. "
            "That gap is provenance and family shape, not transferable fraud behaviour.",
            synthetic_on_synth / max(synthetic_on_real, 1e-9),
        )
    if real_on_synth < 0.25 * real_on_real:
        log.warning(
            "Reverse transfer fails too: a detector trained on REAL fraud scores %.4f on the "
            "held-out synthetic family against %.4f on real fraud. The family does not look like "
            "the fraud this anchor actually contains.",
            real_on_synth,
            real_on_real,
        )


if __name__ == "__main__":
    raise SystemExit(main())
