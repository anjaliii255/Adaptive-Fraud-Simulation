"""The anchor gates, committed rather than read off a terminal.

Before an anchor is allowed to host the loop it has to answer two questions, and the answers
decided which dataset this project runs on:

  Gate 1  does it host behaviour?      A `src_*` feature is worth nothing against a sender with no
                                       past. Measured as the share of rows whose sender has an
                                       earlier row, plus the shape of the per-account distribution.
  Gate 2  is its own fraud non-trivial? Amount-only floor against a trained ceiling, on the anchor's
                                        OWN real fraud. A ratio near 1 means sorting by amount is
                                        already the answer and there is no room left to measure.

Gate 3 is the transfer test and lives in `scripts/transfer_test.py`; it is not recomputed here.

    python scripts/spike_gates.py --data amlworld
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import committed_split_for
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import protocol
from afl.utils.runcard import stamp
from afl.utils.seed import set_all_seeds

log = logging.getLogger("spike")
ARTIFACT_DIR = Path("artifacts/spike")

#: A sender with no past cannot be anomalous against it, so more than half the rows have to have
#: one before any behavioural feature means anything.
MIN_HISTORY_COVERAGE = 0.5
#: And a typical account needs enough of a past for a velocity window to see anything in it.
MIN_MEDIAN_TXNS = 10
#: Floor/ceiling above this and the amount column has already answered the question.
MAX_FLOOR_CEILING_RATIO = 0.6


def gate_1_hosts_behaviour(txns: list[Transaction]) -> dict:
    """Does a typical row have a sender past for the `src_*` features to read?"""
    counts = Counter(t.src for t in txns)
    seen: set[str] = set()
    with_history = 0
    for t in sorted(txns, key=lambda t: t.ts):
        if t.src in seen:
            with_history += 1
        seen.add(t.src)

    per_account = list(counts.values())
    coverage = with_history / len(txns)
    median = statistics.median(per_account)
    # the share of ROWS on a repeat account, which is not the share of ACCOUNTS that repeat: a
    # hub-and-spoke anchor fails the median while nearly every row still sits on a busy account
    repeat_rows = sum(c for c in per_account if c > 1) / len(txns)

    return {
        "rows": len(txns),
        "accounts": len(counts),
        "history_coverage": round(coverage, 4),
        "median_txns_per_account": median,
        "mean_txns_per_account": round(statistics.fmean(per_account), 4),
        "max_txns_per_account": max(per_account),
        "share_of_rows_on_a_repeat_account": round(repeat_rows, 4),
        "thresholds": {
            "history_coverage_gt": MIN_HISTORY_COVERAGE,
            "median_txns_per_account_gte": MIN_MEDIAN_TXNS,
        },
        "passes_coverage": coverage > MIN_HISTORY_COVERAGE,
        "passes_median": median >= MIN_MEDIAN_TXNS,
        "verdict": (
            "PASS"
            if coverage > MIN_HISTORY_COVERAGE and median >= MIN_MEDIAN_TXNS
            else "SPLIT"
            if coverage > MIN_HISTORY_COVERAGE
            else "FAIL"
        ),
    }


def _amount_floor(rows: list[Transaction], fixed_fpr: float, k: int) -> dict:
    """No model, no features, no training: the reference every trained number must clear."""
    y = np.array([int(t.is_fraud) for t in rows])
    amounts = np.array([t.amount for t in rows], dtype=float)
    best = max(
        (protocol.evaluate(y, s, fixed_fpr, k) for s in (amounts, -amounts)),
        key=lambda r: r.pr_auc,
    )
    return {"pr_auc": best.pr_auc, "recall_at_fixed_fpr": best.recall_at_fixed_fpr}


def gate_2_fraud_non_trivial(
    train: list[Transaction],
    test: list[Transaction],
    seed: int,
    params: dict,
    fixed_fpr: float,
    k: int,
) -> dict:
    """Amount alone against a trained model, on the anchor's own fraud. Near-parity kills a fold."""
    floor = _amount_floor(test, fixed_fpr, k)
    detector = LGBMDetector(seed=seed, params=params)
    detector.fit(train)
    scored = protocol.evaluate_detector(detector, test, fixed_fpr=fixed_fpr, k=k)
    ratio = floor["pr_auc"] / max(scored.pr_auc, 1e-12)

    top = sorted((detector.feature_importance() or {}).items(), key=lambda kv: kv[1], reverse=True)[
        :6
    ]
    return {
        "amount_floor_pr_auc": round(floor["pr_auc"], 6),
        "trained_ceiling_pr_auc": round(scored.pr_auc, 6),
        "ratio": round(ratio, 4),
        "threshold_ratio_lte": MAX_FLOOR_CEILING_RATIO,
        "verdict": "PASS" if ratio <= MAX_FLOOR_CEILING_RATIO else "FAIL",
        "n_positives_in_test": scored.n_positives,
        "recall_at_fixed_fpr": round(scored.recall_at_fixed_fpr, 6),
        # names the lift, so "the gain is relational, not a richer amount distribution" is checkable
        "top_features": [name for name, _ in top],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="amlworld", help="anchor to gate")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--fixed-fpr", type=float, default=0.01)
    p.add_argument("-k", type=int, default=100)
    args = p.parse_args()
    set_all_seeds(args.seed)

    data_cfg = yaml.safe_load(Path(f"config/data/{args.data}.yaml").read_text())
    lgbm_cfg = yaml.safe_load(Path("config/defend/lgbm.yaml").read_text())

    real = loaders.load_from_config(data_cfg)
    log.info("anchor %s: %d rows, %d fraud", args.data, len(real), sum(t.is_fraud for t in real))

    gate1 = gate_1_hosts_behaviour(real)
    log.info(
        "gate 1 %s: coverage %.4f (>%.2f), median %.1f (>=%d), mean %.1f, max %d, "
        "%.4f of rows on a repeat account",
        gate1["verdict"],
        gate1["history_coverage"],
        MIN_HISTORY_COVERAGE,
        gate1["median_txns_per_account"],
        MIN_MEDIAN_TXNS,
        gate1["mean_txns_per_account"],
        gate1["max_txns_per_account"],
        gate1["share_of_rows_on_a_repeat_account"],
    )

    split = committed_split_for(data_cfg)
    if split is None:
        raise SystemExit(f"{args.data} has no committed split; run `make splits` first")
    train, test = split.apply(real)

    gate2 = gate_2_fraud_non_trivial(
        train, test, args.seed, dict(lgbm_cfg.get("params") or {}), args.fixed_fpr, args.k
    )
    log.info(
        "gate 2 %s: floor %.4f vs ceiling %.4f = ratio %.4f (<=%.2f), top features %s",
        gate2["verdict"],
        gate2["amount_floor_pr_auc"],
        gate2["trained_ceiling_pr_auc"],
        gate2["ratio"],
        MAX_FLOOR_CEILING_RATIO,
        ", ".join(gate2["top_features"]),
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / f"{args.data}.json"
    out.write_text(
        json.dumps(
            {
                "anchor": args.data,
                "seed": args.seed,
                "provenance": stamp(args.seed),
                "split_digest": split.digest,
                "operating_point": {"fixed_fpr": args.fixed_fpr, "k": args.k},
                "gate_1_hosts_behaviour": gate1,
                "gate_2_fraud_non_trivial": gate2,
                # gate 3 is the transfer test and is not recomputed here
                "gate_3_transfer": f"artifacts/transfer/{args.data}.json",
            },
            indent=2,
        )
        + "\n"
    )
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
