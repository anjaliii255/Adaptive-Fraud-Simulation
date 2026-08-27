"""Figures from logged runs — never from a re-run.

From the A/B/C/D artefact, one pair per anchor/typology:

  <anchor>_<typology>_convergence.png  evasion against held-out recall, round by round, all seeds
  <anchor>_<typology>_realism.png      the realism leash beside the evasion it is meant to constrain
  <anchor>_<typology>_convergence.md   the numbers behind both, so the figures can be checked

From a single `make loop` run, the older shape:

  convergence.png   one run's evasion and recall
  three_system.png  A vs B vs C on the held-out family

Every figure traces back to the artefact that produced it, and regenerating from the same
artefact produces the same bytes.

    python scripts/make_figures.py [--artifacts artifacts] [--run adaptive]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANNER = "PIPELINE CHECK - NOT A RESULT"

EVASION = "#c0392b"
RECALL = "#2471a3"
PENALTY = "#b7950b"
RETRAIN = "#566573"

# matplotlib otherwise stamps its own version into the PNG, which breaks byte-identical reruns
PNG_META = {"Software": "afl-make-figures"}


def _load(path: Path):
    """Parse a run artefact, or None if the run never wrote it."""
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, out: Path) -> Path:
    fig.savefig(out, dpi=150, metadata=PNG_META)
    plt.close(fig)
    return out


def read_metrics(blob) -> tuple[list[dict], bool, str]:
    """(systems, pipeline_check, data_name) from metrics.json, old list shape included."""
    if isinstance(blob, dict):
        return blob.get("systems", []), bool(blob.get("pipeline_check")), str(blob.get("data", "?"))
    return blob or [], False, "?"


def stamp(fig, pipeline_check: bool, data_name: str) -> None:
    """Print the banner across any figure built from a run with no real anchor dataset.

    A figure outlives the terminal that produced it, so the caveat has to live in the image.
    """
    if not pipeline_check:
        return
    # faint enough to read a value label through, dark enough to survive a screenshot
    fig.text(
        0.5,
        0.5,
        BANNER,
        fontsize=19,
        color="#c0392b",
        alpha=0.09,
        ha="center",
        va="center",
        rotation=18,
        weight="bold",
        zorder=0,
    )
    fig.text(
        0.5,
        0.005,
        f"{BANNER} - data={data_name}, not reportable",
        fontsize=8,
        color="#c0392b",
        ha="center",
        va="bottom",
        weight="bold",
    )


# --------------------------------------------------------------------------------------
# the A/B/C/D artefact: many seeds, each carrying a per-round convergence trace
# --------------------------------------------------------------------------------------


def _traces(runs: list[dict], key: str) -> list[list[float]]:
    """Per-seed round traces for one field, clipped to the shortest run so the mean stays honest."""
    n = min(len(r["convergence"]) for r in runs)
    return [[c[key] for c in r["convergence"][:n]] for r in runs]


def _mean(traces: list[list[float]]) -> list[float]:
    return [statistics.fmean(col) for col in zip(*traces, strict=True)]


def _sd(traces: list[list[float]]) -> list[float]:
    return [statistics.pstdev(col) for col in zip(*traces, strict=True)]


def _fold_caption(blob: dict) -> str:
    """The fold's own facts, in the image — a figure gets screenshotted away from its README."""
    run = blob["runs"][0]
    op = blob.get("operating_point", {})
    return (
        f"anchor {blob['anchor']} · held out {blob['typology']} · "
        f"{run['positives']} positives in fold · base rate {run['base_rate']:.4%} · "
        f"{len(blob['runs'])} seeds × {blob['rounds']} rounds · "
        f"recall @ {op.get('fixed_fpr', 0.01):.0%} FPR · split {blob['split_digest']}"
    )


def _mark_retrains(ax, rounds: list[int]) -> None:
    """The detector refits at the end of every round; the marks say where."""
    for i, r in enumerate(rounds):
        ax.axvline(
            r + 0.5,
            color=RETRAIN,
            linestyle=":",
            linewidth=1,
            alpha=0.55,
            label="detector retrain" if i == 0 else None,
        )


def abcd_convergence_figure(blob: dict, out: Path) -> Path:
    """The hero figure: what the attacker got past, and what the detector held, round by round."""
    runs = blob["runs"]
    evasion = _traces(runs, "evasion_rate")
    recall = _traces(runs, "recall_at_fixed_fpr")
    x = list(range(len(evasion[0])))
    fpr = blob.get("operating_point", {}).get("fixed_fpr", 0.01)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    # every seed drawn, so a bad round stays visible instead of being averaged away
    for trace in evasion:
        ax.plot(x, trace, color=EVASION, alpha=0.16, linewidth=1)
    for trace in recall:
        ax.plot(x, trace, color=RECALL, alpha=0.16, linewidth=1)
    _mark_retrains(ax, x[:-1])

    n = len(runs)
    ax.plot(
        x,
        _mean(evasion),
        marker="o",
        color=EVASION,
        linewidth=2.4,
        label=f"attacker evasion rate (mean of {n} seeds)",
    )
    ax.plot(
        x,
        _mean(recall),
        marker="s",
        color=RECALL,
        linewidth=2.4,
        label=f"held-out recall @ {fpr:.0%} FPR (mean of {n} seeds)",
    )

    ax.set_xlabel("loop round")
    ax.set_ylabel("rate")
    ax.set_xticks(x)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Adaptive loop on {blob['anchor']}, holding out {blob['typology']}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.text(0.5, 0.015, _fold_caption(blob), fontsize=7.5, color="#566573", ha="center")
    return _save(fig, out)


def abcd_realism_figure(blob: dict, out: Path) -> Path:
    """Ticket 14's question in one image: is the optimiser buying evasion with absurd traffic?"""
    runs = blob["runs"]
    evasion = _traces(runs, "evasion_rate")
    penalty = _traces(runs, "realism_penalty")
    x = list(range(len(evasion[0])))
    lam = blob.get("lambda_realism")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for trace in penalty:
        ax.plot(x, trace, color=PENALTY, alpha=0.16, linewidth=1)
    ax.plot(
        x, _mean(evasion), marker="o", color=EVASION, linewidth=2.4, label="attacker evasion rate"
    )
    ax.plot(
        x,
        _mean(penalty),
        marker="D",
        color=PENALTY,
        linewidth=2.4,
        label="realism penalty" + (f" (λ={lam})" if lam is not None else ""),
    )
    _mark_retrains(ax, x[:-1])

    ax.set_xlabel("loop round")
    ax.set_ylabel("rate")
    ax.set_xticks(x)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Is it cheating? — evasion against the realism leash, {blob['anchor']}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.text(0.5, 0.015, _fold_caption(blob), fontsize=7.5, color="#566573", ha="center")
    return _save(fig, out)


def abcd_numbers(blob: dict, out: Path) -> Path:
    """The per-round table beside the figures, so a reader can check a curve against its values."""
    runs = blob["runs"]
    fields = ("evasion_rate", "recall_at_fixed_fpr", "pr_auc", "realism_penalty", "fitness")
    stats = {f: (_mean(_traces(runs, f)), _sd(_traces(runs, f))) for f in fields}
    rounds = range(len(stats["evasion_rate"][0]))
    rejected = sum(1 for r in runs for c in r["convergence"] if c["rejected_by_audit"])
    total = sum(len(r["convergence"]) for r in runs)

    lines = [
        f"# Convergence — {blob['anchor']}, held-out {blob['typology']}",
        "",
        f"{_fold_caption(blob)}",
        "",
        f"Seeds: {', '.join(str(r['seed']) for r in runs)}",
        f"Audit gate rejected {rejected} of {total} rounds.",
        "",
        "Mean over seeds, ± population sd. No smoothing.",
        "",
        "| round | evasion | held-out recall | held-out PR-AUC | realism penalty | fitness |",
        "|---|---|---|---|---|---|",
    ]
    for i in rounds:
        cells = " | ".join(f"{stats[f][0][i]:.3f} ± {stats[f][1][i]:.3f}" for f in fields)
        lines.append(f"| {i} | {cells} |")

    header = "| seed | " + " | ".join(f"r{i}" for i in rounds) + " |"
    lines += [
        "",
        "## Per-seed evasion rate",
        "",
        header,
        "|---" * (len(header.split("|")) - 2) + "|",
    ]
    for run, trace in zip(runs, _traces(runs, "evasion_rate"), strict=True):
        lines.append(f"| {run['seed']} | " + " | ".join(f"{v:.3f}" for v in trace) + " |")

    out.write_text("\n".join(lines) + "\n")
    return out


def build_abcd_figures(abcd_dir: Path) -> list[Path]:
    """One convergence figure, one realism figure and one number table per committed A/B/C/D run."""
    made: list[Path] = []
    for path in sorted(abcd_dir.glob("*.json")):
        blob = _load(path)
        if not blob or not blob.get("runs"):
            continue
        if not all(r.get("convergence") for r in blob["runs"]):
            print(f"{path.name}: no per-round convergence trace — skipping")
            continue
        stem = path.stem
        made += [
            abcd_convergence_figure(blob, abcd_dir / f"{stem}_convergence.png"),
            abcd_realism_figure(blob, abcd_dir / f"{stem}_realism.png"),
            abcd_numbers(blob, abcd_dir / f"{stem}_convergence.md"),
        ]
    return made


# --------------------------------------------------------------------------------------
# the older single-run shape, from `make loop`
# --------------------------------------------------------------------------------------


def convergence_figure(
    history: list[dict], out: Path, title: str, pipeline_check: bool = False, data_name: str = "?"
) -> Path | None:
    """Evasion against held-out recall, round by round."""
    rounds = [h for h in history if "round" in h]
    if not rounds:
        print("no per-round records in history.json — skipping convergence figure")
        return None

    x = [h["round"] for h in rounds]
    evasion = [h.get("evasion_rate") for h in rounds]
    recall = [h.get("recall_at_fixed_fpr") for h in rounds]
    pr_auc = [h.get("pr_auc") for h in rounds]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, evasion, marker="o", label="attacker evasion rate", color=EVASION)
    ax.plot(x, recall, marker="s", label="held-out recall @ fixed FPR", color=RECALL)
    ax.plot(
        x, pr_auc, marker="^", linestyle="--", label="held-out PR-AUC", color="#7d3c98", alpha=0.7
    )
    _mark_retrains(ax, x[:-1])
    ax.set_xlabel("loop round")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    stamp(fig, pipeline_check, data_name)
    return _save(fig, out)


def three_system_figure(
    metrics: list[dict], out: Path, title: str, pipeline_check: bool = False, data_name: str = "?"
) -> Path | None:
    """Real-only vs SMOTE vs adaptive, at the one agreed operating point."""
    if not metrics:
        return None
    names = [m["system"] for m in metrics]
    recall_key = next((k for k in metrics[0] if k.startswith("recall@")), None)
    prec_key = next((k for k in metrics[0] if k.startswith("precision@")), None)

    series = [("PR-AUC", [m.get("pr_auc", 0.0) for m in metrics])]
    if recall_key:
        series.append((recall_key, [m.get(recall_key, 0.0) for m in metrics]))
    if prec_key:
        series.append((prec_key, [m.get(prec_key, 0.0) for m in metrics]))

    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, values) in enumerate(series):
        xs = [j + i * width for j in range(len(names))]
        bars = ax.bar(xs, values, width=width, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(names))])
    ax.set_xticklabels(names)
    # headroom so the legend sits above the bars instead of over their value labels
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("score on the held-out attack family")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    stamp(fig, pipeline_check, data_name)
    return _save(fig, out)


def main() -> int:
    """Regenerate every figure from committed artefacts, never from a fresh run."""
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--run", default="adaptive", help="subdirectory under artifacts/")
    args = p.parse_args()

    made: list[Path | None] = []

    abcd_dir = args.artifacts / "abcd"
    if abcd_dir.exists():
        made += build_abcd_figures(abcd_dir)

    run_dir = args.artifacts / args.run
    if run_dir.exists():
        history = _load(run_dir / "history.json") or []
        metrics, pipeline_check, data_name = read_metrics(_load(run_dir / "metrics.json"))
        if pipeline_check:
            print(
                f"\n{'=' * 78}\n{BANNER}\ndata={data_name}: these figures verify that the pipeline "
                f"runs, nothing more.\n{'=' * 78}\n"
            )
        made += [
            convergence_figure(
                history,
                run_dir / "convergence.png",
                f"Adaptive loop — {args.run}",
                pipeline_check,
                data_name,
            ),
            three_system_figure(
                metrics,
                run_dir / "three_system.png",
                "Real-only vs SMOTE vs adaptive",
                pipeline_check,
                data_name,
            ),
        ]

    for path in filter(None, made):
        print(f"wrote {path}")
    if not any(made):
        print(f"nothing to plot under {args.artifacts} — run `make loop` or the A/B/C/D experiment")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
