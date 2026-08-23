"""Compute the out-of-time boundary once, commit it, and write the data card beside it.

    python scripts/build_splits.py                 # every real anchor in config/data/
    python scripts/build_splits.py paysim          # just one

The boundary is a *decision*, not a run-time derivation. `out_of_time_split` picks its cut from
a fraction of whatever rows it was handed, so the partition moves when the row set moves — a
different sample fraction, one more vector in the pool, a re-download, and this week's number is
measured against a different split than last week's. This script does that arithmetic once,
against the full raw file, and writes two timestamps to `artifacts/splits/`. Every run after
that reads them.

Everything here runs in pandas over the raw columns. Nothing is loaded into contract types, so
the full 6.3M-row PaySim file is a few seconds rather than 7 GB of pydantic objects.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from afl.data.loaders import DATA_DIR, entity_bucket, steps_to_timestamps
from afl.data.splits import CommittedSplit

log = logging.getLogger("build_splits")

CONFIG_DIR = Path("config/data")
DATA_CARD_DIR = Path("docs/data-cards")

#: Below this the two regimes are not comparable and every operating point has to be re-set.
#: An order of magnitude is the line because recall@1%FPR and precision@k stop measuring the
#: same thing long before the rates differ by 130x — see docs/adr/0002-dataset-anchors.md.
BASE_RATE_ORDER_OF_MAGNITUDE = 10.0

#: Measured, not quoted: see `synthetic_base_rate` below.
SYNTHETIC_CONFIG = Path("config/data/synthetic.yaml")


# ── reading the raw file ────────────────────────────────────────────────────────
#: (time, fraud, sampling entity, amount, src, dst) per loader — the columns the split and the
#: integrity checks need, and nothing else. The leak columns are absent here for the same reason
#: they are absent from the loader: a column that is never read cannot reach a model.
RAW_COLUMNS = {
    "amlsim": (
        "TIMESTAMP",
        "IS_FRAUD",
        "SENDER_ACCOUNT_ID",
        "TX_AMOUNT",
        "SENDER_ACCOUNT_ID",
        "RECEIVER_ACCOUNT_ID",
    ),
    "paysim": ("step", "isFraud", "nameDest", "amount", "nameOrig", "nameDest"),
}


def read_raw(cfg: dict) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """(frame, column roles) — only the columns the split and the integrity checks need."""
    source, loader = cfg["source"], cfg["loader"]
    root = Path(source.get("place_in", DATA_DIR))
    if loader not in RAW_COLUMNS:
        raise KeyError(f"{cfg['name']}: no raw reader for loader {loader!r}")

    roles = RAW_COLUMNS[loader]
    file = source.get("transactions_file") or source["file"]
    df = pd.read_csv(root / file, usecols=sorted(set(roles)))
    fraud_col = roles[1]
    # IS_FRAUD is a real bool in transactions.csv and the strings true/false elsewhere
    df[fraud_col] = (
        df[fraud_col].astype(bool)
        if df[fraud_col].dtype == bool
        else df[fraud_col].astype(str).str.lower().isin(("true", "1"))
    )
    return df, roles


def integrity_stats(df: pd.DataFrame, roles: tuple[str, ...]) -> dict:
    """Rows the contract would refuse or the realism leash calls impossible.

    Worth measuring rather than assuming: `afl/attack/realism.py` scores generated traffic
    against rules like "no self-transfers", and if the real anchor breaks a rule then the rule
    is a modelling choice rather than a fact about payments. Ticket 14 derives its empirical
    bounds from here, so the violations have to be counted, not hoped away.
    """
    _, _, _, amount_col, src_col, dst_col = roles
    non_positive = int((pd.to_numeric(df[amount_col], errors="coerce").fillna(0) <= 0).sum())
    self_transfers = int((df[src_col].astype(str) == df[dst_col].astype(str)).sum())
    return {
        "non_positive_amounts": non_positive,
        "non_positive_amount_share": round(non_positive / len(df), 9),
        "self_transfers": self_transfers,
        "self_transfer_share": round(self_transfers / len(df), 9),
    }


# ── the boundary ────────────────────────────────────────────────────────────────
def derive_boundary(steps: pd.Series, train_frac: float, embargo_steps: int) -> tuple[int, int]:
    """The step at which `train_frac` of the *rows* have happened, plus the embargo.

    Row quantile, not step fraction. PaySim's traffic is violently non-uniform — 341 of its 743
    steps carry fewer than 100 rows, while step 19 alone carries 51,352 — so cutting at 67% of
    the *steps* puts 95% of the rows on the training side and leaves a 4% test tail. A split
    described as 70/30 has to be 70/30 in the thing being counted.
    """
    per_step = steps.value_counts().sort_index()
    cumulative = per_step.cumsum() / per_step.sum()
    reached = cumulative[cumulative >= train_frac]
    if reached.empty:
        raise ValueError(f"no step reaches train_frac={train_frac}")
    train_end = int(reached.index[0])
    return train_end, train_end + embargo_steps + 1


def partition_stats(
    df: pd.DataFrame, time_col: str, fraud_col: str, train_end: int, test_start: int
) -> dict:
    """What the committed boundary actually does to the rows — the number, not the intention."""
    steps, fraud = df[time_col], df[fraud_col]
    train, test = steps <= train_end, steps >= test_start
    embargoed = ~train & ~test

    def side(mask: pd.Series) -> dict:
        n = int(mask.sum())
        n_fraud = int(fraud[mask].sum())
        return {
            "rows": n,
            "row_share": round(n / len(df), 4),
            "fraud": n_fraud,
            "base_rate": round(n_fraud / n, 8) if n else 0.0,
        }

    return {"train": side(train), "test": side(test), "embargoed": side(embargoed)}


def sample_stats(df: pd.DataFrame, fraud_col: str, entity_col: str, fraction: float) -> dict:
    """What the run-time entity sample does to the row count and the base rate."""
    if fraction >= 1.0:
        return {"fraction": 1.0, "note": "no sampling; the loader reads every row"}
    keep = entity_bucket(df[entity_col]) < fraction * 1_000
    sub = df[keep]
    full_rate = float(df[fraud_col].mean())
    rate = float(sub[fraud_col].mean())
    return {
        "fraction": fraction,
        "sampled_by": entity_col,
        "rows": int(len(sub)),
        "fraud": int(sub[fraud_col].sum()),
        "base_rate": round(rate, 8),
        "base_rate_full": round(full_rate, 8),
        "base_rate_drift_relative": round((rate - full_rate) / full_rate, 4) if full_rate else 0.0,
    }


# ── the synthetic comparison ────────────────────────────────────────────────────
def _rate(rows) -> float:
    return sum(1 for t in rows if t.is_fraud) / len(rows) if rows else 0.0


def synthetic_base_rates() -> dict[str, float]:
    """Measure the zero-download default rather than quoting a remembered number.

    There are two synthetic compositions and they are not the same rate, which is exactly why
    this is measured and not asserted:

    * `pipeline` — the pool `scripts/run_experiment.py` actually builds on `data=synthetic`,
      from `config/attack/engines.yaml` plus the holdout episodes. Every synthetic number in the
      README came from this one, so it is the comparison that matters.
    * `data_config` — what `config/data/synthetic.yaml` declares on its own terms.

    The gap between either and a real anchor is the most expensive fact in the project: a
    threshold set at one rate means something else entirely at the other.
    """
    from afl.attack.simulator import Simulator
    from afl.attack.templates import registry

    data_cfg = yaml.safe_load(SYNTHETIC_CONFIG.read_text())
    sim = Simulator(
        seed=1337,
        n_entities=int(data_cfg["n_entities"]),
        n_background=int(data_cfg["n_background"]),
        window_days=int(data_cfg["window_days"]),
    )
    rows = []
    for vid in data_cfg["known_fraud_vectors"]:
        rows.extend(sim.generate(registry.get(vid).to_attack_params()).transactions)

    engines = yaml.safe_load(Path("config/attack/engines.yaml").read_text())
    eval_cfg = yaml.safe_load(Path("config/eval/leave_one_attack_out.yaml").read_text())
    pipeline_sim = Simulator(
        seed=1337,
        n_entities=int(engines["n_entities"]),
        n_background=int(engines["n_background"]),
        start_ts=datetime.fromisoformat(str(engines["start_ts"])),
        window_days=int(engines["window_days"]),
        n_episodes=int(engines["n_episodes"]),
    )
    pool = []
    for vid in engines["vectors"]:
        pool.extend(pipeline_sim.generate(registry.get(vid).to_attack_params()).transactions)
    pipeline_sim.n_episodes = max(int(engines["n_episodes"]), int(eval_cfg["holdout_episodes"]))
    pool.extend(
        pipeline_sim.generate(
            registry.get(str(eval_cfg["held_out_vector"])).to_attack_params()
        ).transactions
    )
    return {"pipeline": _rate(pool), "data_config": _rate(rows)}


# ── the data card ───────────────────────────────────────────────────────────────
def data_card(cfg: dict, split: CommittedSplit, synth_rate: float) -> str:
    """Source, licence, size, base rate, span, quirks, and what it cannot tell us."""
    card, stats = cfg.get("data_card", {}), split.stats
    source, sample = cfg["source"], stats["sample"]
    full, part, integrity = stats["full"], stats["partition"], stats["integrity"]
    nonpos_share = f"{integrity['non_positive_amount_share']:.6%}"
    selfxfer_share = f"{integrity['self_transfer_share']:.6%}"
    # how many times BELOW the synthetic default this anchor sits
    ratio = synth_rate / full["base_rate"] if full["base_rate"] else float("inf")
    shift = (
        part["test"]["base_rate"] / part["train"]["base_rate"]
        if part["train"]["base_rate"]
        else 0.0
    )

    def bullets(key: str) -> str:
        return "\n".join(f"- {line}" for line in card.get(key, [])) or "- _none recorded_"

    def row(label: str, s: dict) -> str:
        return (
            f"| {label} | {s['rows']:,} | {s['row_share']:.1%} | {s['fraud']:,} | "
            f"{s['base_rate']:.5%} |"
        )

    sampled = (
        "The loader reads every row."
        if sample["fraction"] >= 1.0
        else (
            f"The loader keeps a deterministic **{sample['fraction']:.0%} hash-sample of "
            f"`{sample['sampled_by']}`** — {sample['rows']:,} rows, {sample['fraud']:,} fraud, "
            f"base rate {sample['base_rate']:.5%} against the full file's "
            f"{sample['base_rate_full']:.5%} ({sample['base_rate_drift_relative']:+.1%} relative). "
            "Whole entities are kept or dropped, never individual rows, so every kept entity "
            "still has its complete history and the time span is unchanged. Set "
            "`data.sample.sample_fraction=1.0` to read the lot — it costs about "
            f"{sample['rows'] and full['rows'] / sample['rows']:.0f}x the memory."
        )
    )

    return f"""# Data card — {cfg["name"]}

_Generated by `scripts/build_splits.py`. Every number below was measured from the file on disk,
not copied from the source's documentation._

## What it is

{cfg.get("purpose", "A real anchor dataset.")}

| | |
| --- | --- |
| source | {source.get("url", "—")} |
| licence | {source.get("license", "—")} |
| files | `{source.get("place_in", "—")}` |
| download | {source.get("download", "manual")} — not committed (`data/**` is gitignored) |
| rows | {full["rows"]:,} |
| fraud rows | {full["fraud"]:,} |
| **base rate** | **{full["base_rate"]:.5%}** |
| time span | {card.get("time_span", "—")} |
| first / last | {split.stats["span"]["start"]} → {split.stats["span"]["end"]} |

## The base rate, and why it breaks comparability

At **{full["base_rate"]:.5%}** this anchor sits **{ratio:.0f}x below** the synthetic default's
measured **{synth_rate:.2%}**.

{
        "That is more than an order of magnitude, so the two are not comparable and no operating "
        "point carries across. A 1% FPR here buys "
        f"{int(full['rows'] * 0.01):,} false positives against {full['fraud']:,} fraud rows, so "
        "precision@k collapses in a way it never does on synthetic traffic. Thresholds must be "
        "set at this rate, and synthetic numbers never share a table with these."
        if ratio >= BASE_RATE_ORDER_OF_MAGNITUDE
        else "The two rates are within an order of magnitude of each other."
    }

{
        f"**The split shifts the rate again.** Fraud is {shift:.1f}x denser in the test half "
        f"({part['test']['base_rate']:.5%}) than in the train half "
        f"({part['train']['base_rate']:.5%}). This is a property of the data, not of the cut: "
        "the label is spread near-uniformly across steps while the legit volume is not, so any "
        "chronological boundary lands on two different base rates. A threshold calibrated on a "
        "held-out slice of train does not transfer unchanged to test, and a recall figure has to "
        "name which side it was measured on."
        if shift >= 2.0 or shift and shift <= 0.5
        else "Train and test sit at comparable base rates."
    }

## The committed split

Computed once by `scripts/build_splits.py` and read from
`artifacts/splits/{cfg["name"]}_oot.json` on every run. The boundary is two timestamps, so the
partition is identical whatever subset of rows is loaded.

| | |
| --- | --- |
| digest | `{split.digest}` |
| train ends | {split.train_end} (step {split.train_end_step}, inclusive) |
| test starts | {split.test_start} (step {split.test_start_step}, inclusive) |
| embargo | {split.embargo} |
| target train share | {split.train_frac:.0%} of rows |

**Embargo rationale.** {split.embargo_rationale}

| side | rows | share | fraud | base rate |
| --- | ---: | ---: | ---: | ---: |
{row("train", part["train"])}
{row("embargoed", part["embargoed"])}
{row("test", part["test"])}

## Sampling

{sampled}

## Data integrity

Counted from the file, not assumed. The loader clamps a non-positive amount to 0.01 and logs the
count rather than dropping the row, because dropping rows silently moves the base rate.

| check | rows | share |
| --- | ---: | ---: |
| amount <= 0 | {integrity["non_positive_amounts"]:,} | {nonpos_share} |
| self-transfer (src == dst) | {integrity["self_transfers"]:,} | {selfxfer_share} |

{
        f"**{integrity['self_transfers']:,} self-transfers are present in real traffic.** "
        "`afl/attack/realism.py` treats a self-transfer as an impossible row and penalises the "
        "generator for emitting one. On this anchor that rule is a modelling choice, not a fact "
        "about payments — ticket 14 derives its bounds from here and has to decide which."
        if integrity["self_transfers"]
        else "No self-transfers, so the realism leash's rule against them costs nothing here."
    }

## Quirks

{bullets("quirks")}

## What it cannot tell us

{bullets("cannot_tell_us")}
"""


# ── the run ─────────────────────────────────────────────────────────────────────
def build(cfg: dict, synth_rates: dict[str, float]) -> CommittedSplit:
    synth_rate = synth_rates["pipeline"]
    name = cfg["name"]
    df, roles = read_raw(cfg)
    time_col, fraud_col, entity_col = roles[0], roles[1], roles[2]
    split_cfg, time_cfg = cfg["split"], cfg["time"]

    train_frac = float(split_cfg.get("train_frac", 0.70))
    embargo_steps = int(split_cfg["embargo_steps"])
    if embargo_steps < 1:
        raise ValueError(f"{name}: embargo_steps must be >= 1; a zero gap is not an embargo")

    train_end_step, test_start_step = derive_boundary(df[time_col], train_frac, embargo_steps)
    epoch = datetime.fromisoformat(str(time_cfg["epoch"]))
    unit = str(time_cfg["unit"])
    first_step, last_step = int(df[time_col].min()), int(df[time_col].max())
    marks = pd.Series([train_end_step, test_start_step, first_step, last_step]).to_numpy()
    bounds = steps_to_timestamps(marks, unit, epoch)
    train_end, test_start, first, last = pd.DatetimeIndex(bounds).to_pydatetime()

    fraction = float((cfg.get("sample") or {}).get("sample_fraction", 1.0))
    n_fraud = int(df[fraud_col].sum())
    split = CommittedSplit(
        dataset=name,
        train_end=train_end,
        test_start=test_start,
        embargo_rationale=split_cfg["embargo_rationale"],
        train_frac=train_frac,
        time_unit=unit,
        epoch=epoch,
        train_end_step=train_end_step,
        test_start_step=test_start_step,
        stats={
            "full": {
                "rows": len(df),
                "row_share": 1.0,
                "fraud": n_fraud,
                "base_rate": round(n_fraud / len(df), 8),
            },
            "span": {
                "start": first.isoformat(),
                "end": last.isoformat(),
                "first_step": first_step,
                "last_step": last_step,
            },
            "partition": partition_stats(df, time_col, fraud_col, train_end_step, test_start_step),
            "sample": sample_stats(df, fraud_col, entity_col, fraction),
            "integrity": integrity_stats(df, roles),
            "synthetic_base_rate": round(synth_rate, 6),
            "synthetic_base_rate_data_config": round(synth_rates["data_config"], 6),
            "base_rate_ratio_vs_synthetic": round(n_fraud / len(df) / synth_rate, 4)
            if synth_rate
            else None,
        },
    )

    directory = Path(split_cfg.get("commit_to", "artifacts/splits/x")).parent
    path = split.save(directory)
    card_path = DATA_CARD_DIR / f"{name}.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(data_card(cfg, split, synth_rate))

    part = split.stats["partition"]
    print(f"\n── {name} " + "─" * (70 - len(name)))
    print(f"  rows {len(df):,}  fraud {n_fraud:,}  base rate {n_fraud / len(df):.5%}")
    print(f"  boundary  train <= step {train_end_step} ({train_end})")
    print(f"            test  >= step {test_start_step} ({test_start})")
    print(f"            embargo {split.embargo}  digest {split.digest}")
    for side in ("train", "embargoed", "test"):
        s = part[side]
        print(
            f"  {side:<10} {s['rows']:>10,} rows ({s['row_share']:>6.1%})  "
            f"{s['fraud']:>6,} fraud  rate {s['base_rate']:.5%}"
        )
    ratio = split.stats["base_rate_ratio_vs_synthetic"]
    if ratio and ratio <= 1 / BASE_RATE_ORDER_OF_MAGNITUDE:
        print(
            f"  ⚠ base rate is {1 / ratio:.0f}x BELOW the synthetic default ({synth_rate:.2%}). "
            "More than an order of magnitude:\n"
            "    operating points do not carry across, and these numbers never share a table\n"
            "    with synthetic ones. See docs/adr/0002-dataset-anchors.md."
        )
    print(f"  → {path}\n  → {card_path}")
    return split


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="dataset names; default: every real anchor")
    args = parser.parse_args()

    configs = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("loader") and (not args.datasets or cfg["name"] in args.datasets):
            configs.append(cfg)
    if not args.datasets and not configs:
        print("no real anchors configured", file=sys.stderr)
        return 1

    synth_rates = synthetic_base_rates()
    print(
        "synthetic default base rate (measured now, not quoted): "
        f"{synth_rates['pipeline']:.4%} for the pool run_experiment builds, "
        f"{synth_rates['data_config']:.4%} for config/data/synthetic.yaml on its own"
    )

    missing = []
    for cfg in configs:
        try:
            build(cfg, synth_rates)
        except FileNotFoundError as exc:
            missing.append(f"{cfg['name']}: {exc}")
    for line in missing:
        print(f"\nSKIPPED {line}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
