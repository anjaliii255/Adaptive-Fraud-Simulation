"""Day one: prove the fidelity harness discriminates, before there is a generator to judge.

A harness written after the generator gets its thresholds chosen to fit the results. So this
script builds the harness against three known cases and checks it ranks them correctly:

    copy      real data duplicated        -> should score high (and trip the privacy check)
    shuffled  marginals kept, joins broken-> should pass level 1, fail level 2
    noise     nothing preserved           -> should fail everything

If the scorecard cannot separate those three, it cannot be trusted on a real generator, and
nothing downstream is worth building yet.

    python scripts/build_fidelity.py [--n 3000] [--out artifacts/fidelity_selftest]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.schema import Transaction
from afl.data.splits import out_of_time_split
from afl.defend.models.lgbm import LGBMDetector
from afl.fidelity import scorecard
from afl.utils.seed import rng as make_rng
from afl.utils.seed import set_all_seeds


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3_000, help="background rows to stand in for real data")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=Path, default=Path("artifacts/fidelity_selftest"))
    args = p.parse_args()
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

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, synth in cases.items():
        card = scorecard.build(
            real=real,
            synth=synth,
            real_train=real_train,
            real_test=real_test,
            detector_factory=detector_factory,
            seed=args.seed,
            meta={"case": name},
        )
        card.save(args.out / name)
        summary[name] = {
            "verdict": card.verdict,
            "score": card.score,
            "level1": card.levels["level1"]["score"],
            "level2": card.levels["level2"]["score"],
            "level3": card.levels.get("level3", {}).get("score"),
            "privacy": card.levels.get("privacy", {}).get("score"),
            "reasons": card.reasons,
        }
        print(
            f"{name:9s} verdict={card.verdict:5s} score={card.score:6.3f} "
            f"L1={summary[name]['level1']:6.3f} L2={summary[name]['level2']:6.3f}"
        )

    (args.out / "selftest.json").write_text(json.dumps(summary, indent=2))

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
    print()
    for label, ok in checks.items():
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}")

    failed = [k for k, v in checks.items() if not v]
    if failed:
        print(f"\nharness is not discriminating: {failed}")
        return 1
    print(f"\nharness ok -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
