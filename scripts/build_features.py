"""Build the feature table over every anchor, record what it cost, and publish what it contains.

    python scripts/build_features.py                    # every real anchor, plus synthetic
    python scripts/build_features.py paysim             # just one
    python scripts/build_features.py --sample 0.02      # a quick pass over a smaller sample

Two deliverables, and the second is the one that matters.

**Timing** — ticket 07 asks that the table build over the real anchors at a workable speed and
that the speed be recorded rather than believed. Every number in `docs/features.md` comes out of
this script on the files on disk.

**Coverage** — ticket 02 found that PaySim has no sender history, which makes a whole block of
this table structurally empty on that anchor. A dead column is not a bug; a dead column nobody
noticed is, because in a feature list it reads exactly like a feature the model has. So the
emptiness is measured per anchor and written into the dictionary next to the rationale.

The dictionary itself is generated from `afl.defend.features.feature_specs()`, never hand-typed,
for the same reason the data cards are generated from the raw files: a document that is written
by hand is a document that goes stale on the first rename.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from afl.contract.schema import Transaction
from afl.data import loaders
from afl.defend.features import FEATURE_GROUPS, FeatureBuilder, feature_specs

log = logging.getLogger("build_features")

CONFIG_DIR = Path("config/data")
ARTIFACT_DIR = Path("artifacts/features")
DOC_PATH = Path("docs/features.md")

#: The synthetic default is measured too, and it is not decoration: the rail and device blocks
#: are dead on both real anchors, so without this column they would read as dead code rather
#: than as features only synthetic traffic exercises today.
SYNTHETIC = "synthetic"


def synthetic_pool(seed: int = 1337) -> list[Transaction]:
    """The pool `run_experiment.py` builds on `data=synthetic` — the same rows, not a mock."""
    from afl.attack.simulator import Simulator
    from afl.attack.templates import registry

    engines = yaml.safe_load(Path("config/attack/engines.yaml").read_text())
    eval_cfg = yaml.safe_load(Path("config/eval/leave_one_attack_out.yaml").read_text())
    sim = Simulator(
        seed=seed,
        n_entities=int(engines["n_entities"]),
        n_background=int(engines["n_background"]),
        start_ts=datetime.fromisoformat(str(engines["start_ts"])),
        window_days=int(engines["window_days"]),
        n_episodes=int(engines["n_episodes"]),
    )
    pool: list[Transaction] = []
    for vid in engines["vectors"]:
        pool.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)
    sim.n_episodes = max(int(engines["n_episodes"]), int(eval_cfg["holdout_episodes"]))
    pool.extend(
        sim.generate(registry.get(str(eval_cfg["held_out_vector"])).to_attack_params()).transactions
    )
    return pool


def measure(name: str, rows: list[Transaction]) -> dict:
    """Build the table once, time it, and describe what came out.

    Scoring is timed separately and on the *tail* of the data, because that is the shape a real
    run has: a detector fitted on the training window scores an out-of-time test window, so its
    rows arrive after everything the builder has already seen. The other shape — a batch that
    predates the committed history, which the closed loop produces every round — costs more, and
    `docs/features.md` says so rather than quoting the flattering number alone.
    """
    builder = FeatureBuilder(stateful=True)
    start = time.perf_counter()
    X = builder.transform(rows, update=True)
    build_s = time.perf_counter() - start

    ordered = sorted(rows, key=lambda t: t.ts)
    tail = ordered[-min(50_000, len(ordered)) :]
    start = time.perf_counter()
    builder.transform(tail, update=False)
    score_s = time.perf_counter() - start

    coverage = builder.coverage(X)
    return {
        "dataset": name,
        "rows": len(rows),
        "n_features": X.shape[1],
        "build_seconds": round(build_s, 3),
        "build_rows_per_second": round(len(rows) / build_s) if build_s else 0,
        "score_rows": len(tail),
        "score_seconds": round(score_s, 3),
        "score_rows_per_second": round(len(tail) / score_s) if score_s else 0,
        "state": builder.state_size(),
        "dead_features": sorted(coverage[coverage["dead"]].index),
        "coverage": {
            name: {
                # bracketed, not dotted: `row.mean` on a Series resolves to the *method*
                "group": row["group"],
                "distinct_values": int(row["distinct_values"]),
                "share_informative": round(float(row["share_informative"]), 6),
                "share_never": round(float(row["share_never"]), 6),
                "mean": round(float(row["mean"]), 6),
                "dead": bool(row["dead"]),
            }
            for name, row in coverage.iterrows()
        },
    }


# ── the dictionary ──────────────────────────────────────────────────────────────
def _cell(report: dict | None, feature: str) -> str:
    if report is None:
        return "_not measured_"
    entry = report["coverage"].get(feature)
    if entry is None:
        return "—"
    if entry["dead"]:
        return "**dead**"
    share = entry["share_informative"]
    # a rare-but-present column is not the same thing as an absent one, and `0%` reads as absent
    return "&lt;1%" if share < 0.005 else f"{share:.0%}"


def feature_doc(reports: dict[str, dict], missing: list[str]) -> str:
    """The feature list with a one-line rationale each, and how much of it each anchor fills."""
    names = list(reports)
    header = " | ".join(names) if names else ""
    divider = " | ".join(["---:"] * len(names))

    def table(group: str) -> str:
        specs = [s for s in feature_specs() if s.group == group]
        head = f"| feature | why it is here |{' ' + header + ' |' if names else ''}"
        rule = f"| --- | --- |{' ' + divider + ' |' if names else ''}"
        lines = [head, rule]
        for spec in specs:
            cells = "".join(f" {_cell(reports.get(n), spec.name)} |" for n in names)
            lines.append(f"| `{spec.name}` | {spec.why} |{cells}")
        return "\n".join(lines)

    def cost_row(name: str) -> str:
        r = reports[name]
        state = r["state"]
        return (
            f"| {name} | {r['rows']:,} | {r['build_seconds']:.1f}s | "
            f"{r['build_rows_per_second']:,}/s | "
            f"{r['score_rows']:,} in {r['score_seconds']:.1f}s | "
            f"{state['entities']:,} | {state['events']:,} | {len(r['dead_features'])} |"
        )

    dead_notes = []
    for name, r in reports.items():
        dead = r["dead_features"]
        listed = ", ".join(f"`{d}`" for d in dead) if dead else "_none_"
        dead_notes.append(
            f"**{name}** — {len(dead)} of {r['n_features']} columns never take a second value: "
            f"{listed}"
        )

    absent = (
        "\n".join(f"- `{name}` is not downloaded, so it has no column above." for name in missing)
        if missing
        else "- Every configured anchor was on disk when this ran."
    )

    nl = chr(10)
    cost_rows = nl.join(cost_row(name) for name in names)
    dead_lines = nl.join(f"- {note}" for note in dead_notes)
    sections = nl.join(
        f"### {group}{nl}{nl}{why}{nl}{nl}{table(group)}{nl}"
        for group, why in FEATURE_GROUPS.items()
    )

    return f"""# Feature dictionary

_Generated by `scripts/build_features.py` from `afl.defend.features.feature_specs()` and from the
files on disk. Every rationale below is the one in the code, and every percentage was measured,
not estimated. Do not edit this file — edit the registry and re-run `make features`._

The feature table is the blue side's view of a transaction, and it obeys two rules:

1. **Causal only.** Every value is computed from events strictly *before* the row it belongs to,
   judged against the row's own timestamp rather than against whatever the builder happens to
   have observed. `tests/test_features.py` proves it against a brute-force reference and against
   the property directly: appending later traffic never changes an earlier row's features.
2. **No provenance.** `is_fraud`, `vector_id`, `attack_run_id` and `txn_id` never enter X. The
   first three are the answer key; the fourth identifies the row rather than describing it.

## What it costs

| anchor | rows | build | throughput | score (tail) | entities | events | dead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{cost_rows}

Timings are wall-clock on the machine that last ran `make features`, so treat them as an
order of magnitude rather than a benchmark. The shape is the part that transfers: the builder is
linear in rows, and `tests/test_features.py` holds a throughput floor that catches a regression
to quadratic without failing on a busy laptop.

History is retained in full rather than trimmed to the widest window, which is what makes the
lifetime features (`src_out_txn_count`, `dst_in_degree`, `*_account_age_s`) exact as of the row
being scored. The cost is linear in events — two per transaction, one on each side — and the
column above is there so it is visible rather than surprising.

**Scoring an out-of-order batch costs more than the table above suggests.** The timed figure is
the shape a reported run has: a detector fitted on the training window scoring an out-of-time
test window, where every scored row arrives after everything already observed. The closed loop
produces the other shape — an attack batch generated *inside* the traffic it hides in — and each
such row is inserted into the middle of an entity's history rather than appended to the end. That
path is roughly an order of magnitude slower per row on a deep-history anchor like AMLSim. It is
bounded by how much history one entity has, not by how many rows exist in total, so it stays
workable at the batch sizes the loop actually generates.

## Reading the coverage columns

A percentage is the share of rows where the feature differs from its most common value — how much
of the column is actually saying something on that anchor. **dead** means it never takes a second
value at all.

Dead columns are not dead code, and this is the distinction the table exists to make. PaySim has
no sender history — `nameOrig` is effectively unique per row — so every `src_*` feature is empty
there by construction, while the same features carry the mule signal on AMLSim. Neither real
anchor carries a device id, so the device block only moves on synthetic traffic. What would be a
bug is a column that is dead everywhere, or a *beneficiary* column dead on an anchor we report
detection numbers from; `tests/test_features.py` asserts the second one on the real files.

{dead_lines}

{absent}

## The columns

{sections}
## Where the numbers came from

```bash
make features        # or: python scripts/build_features.py
```

Per-anchor artefacts, including the full per-column coverage, are in `artifacts/features/`.
"""


# ── the run ─────────────────────────────────────────────────────────────────────
def anchors(selected: list[str]) -> list[dict]:
    out = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("loader") and (not selected or cfg["name"] in selected):
            out.append(cfg)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="dataset names; default: all of them")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="override the entity sample fraction; default is each config's own",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    reports: dict[str, dict] = {}
    missing: list[str] = []
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in anchors(args.datasets):
        name = cfg["name"]
        if args.sample is not None:
            cfg = {**cfg, "sample": {**(cfg.get("sample") or {}), "sample_fraction": args.sample}}
        try:
            rows = loaders.load_from_config(cfg)
        except loaders.DatasetNotDownloaded as exc:
            missing.append(name)
            print(f"SKIPPED {name}: {exc}", file=sys.stderr)
            continue
        reports[name] = measure(name, rows)

    if not args.datasets or SYNTHETIC in args.datasets:
        reports[SYNTHETIC] = measure(SYNTHETIC, synthetic_pool(args.seed))

    for name, report in reports.items():
        path = ARTIFACT_DIR / f"{name}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        state = report["state"]
        print(f"\n── {name} " + "─" * (70 - len(name)))
        print(
            f"  {report['rows']:,} rows x {report['n_features']} features in "
            f"{report['build_seconds']:.1f}s ({report['build_rows_per_second']:,} rows/s)"
        )
        print(
            f"  scoring {report['score_rows']:,} out-of-time rows: {report['score_seconds']:.1f}s "
            f"({report['score_rows_per_second']:,} rows/s)"
        )
        print(
            f"  state: {state['entities']:,} entities, {state['events']:,} events, "
            f"{state['pairs']:,} pairs, {state['devices']:,} devices"
        )
        dead = report["dead_features"]
        print(f"  dead columns: {len(dead)}/{report['n_features']}")
        for feature in dead:
            print(f"      {feature}")
        print(f"  → {path}")

    if not reports:
        print("nothing measured", file=sys.stderr)
        return 1

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(feature_doc(reports, missing))
    print(f"\n→ {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
