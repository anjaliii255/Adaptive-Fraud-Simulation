"""Figures from logged runs — never from a re-run.

Two artefacts carry the whole story:

  convergence.png   evasion rate falling while held-out recall rises, round by round. If the
                    two curves do not move together, the loop is not doing what we claim.
  three_system.png  A vs B vs C on the held-out family, at one operating point.

Reads artifacts/<run>/history.json and metrics.json, so a figure can always be traced back to
the run that produced it.

    python scripts/make_figures.py [--artifacts artifacts] [--run adaptive]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANNER = "PIPELINE CHECK - NOT A RESULT"


def _load(path: Path):
    """Parse a run artefact, or None if the run never wrote it."""
    return json.loads(path.read_text()) if path.exists() else None


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
    ax.plot(x, evasion, marker="o", label="attacker evasion rate", color="#c0392b")
    ax.plot(x, recall, marker="s", label="held-out recall @ fixed FPR", color="#2471a3")
    ax.plot(
        x, pr_auc, marker="^", linestyle="--", label="held-out PR-AUC", color="#7d3c98", alpha=0.7
    )
    ax.set_xlabel("loop round")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    stamp(fig, pipeline_check, data_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


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
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    """Regenerate both figures from a run's logged artefacts."""
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--run", default="adaptive", help="subdirectory under artifacts/")
    args = p.parse_args()

    run_dir = args.artifacts / args.run
    if not run_dir.exists():
        print(f"no run at {run_dir} — run `make loop` first")
        return 1

    history = _load(run_dir / "history.json") or []
    metrics, pipeline_check, data_name = read_metrics(_load(run_dir / "metrics.json"))
    if pipeline_check:
        print(
            f"\n{'=' * 78}\n{BANNER}\ndata={data_name}: these figures verify that the pipeline "
            f"runs, nothing more.\n{'=' * 78}\n"
        )

    made = [
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
    return 0 if any(made) else 1


if __name__ == "__main__":
    raise SystemExit(main())
