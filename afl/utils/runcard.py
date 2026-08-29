"""Which code, which seed, which machine — stamped onto the artefact rather than remembered.

`afl/utils/provenance.py` answers "which commit". That is necessary and not sufficient: the same
commit and the same seed can still produce different numbers on a different LightGBM build, and
when that happens the difference has to be readable from the two files rather than reconstructed
from memory. So every artefact carries a `provenance` block, and it records the four things that
have actually explained a divergence in this project: the commit, whether that tree was dirty, the
seed, and the versions of the libraries that do the arithmetic.

**Nothing here is a clock.** A timestamp in an artefact makes two identical runs produce two
different files, which destroys the cheapest reproducibility check there is — run it twice, diff
the bytes. The wall-clock facts (when, how long, what the command line was) live in a separate
`run_card.json`, which is written beside the artefact and excluded from that diff.

Never raises: on a machine with no git, no metadata, or a package that will not import, the field
is `None` and the reader is told the question could not be answered.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afl.utils.provenance import git_provenance

# The libraries whose version has changed a number in this repository, or plausibly could:
# LightGBM decides the model, numpy/scipy/sklearn decide the arithmetic around it, pandas decides
# how the anchor was parsed. `torch` is optional and absent from the default install.
KEY_PACKAGES = ("lightgbm", "numpy", "scipy", "scikit-learn", "pandas", "shap", "optuna", "torch")


def package_versions(names: tuple[str, ...] = KEY_PACKAGES) -> dict[str, str | None]:
    """`{name: version}` for the libraries that do the arithmetic. `None` when not installed."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except (PackageNotFoundError, ValueError):
            out[name] = None
    return out


def environment() -> dict[str, Any]:
    """The machine, in the detail that has ever mattered: interpreter, platform, libraries.

    `cpu_count` is here because thread count is the one plausible source of run-to-run drift in
    LightGBM's histogram building — if a number ever fails to reproduce across two machines, this
    is the first field to compare.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }


def stamp(seed: int | None = None, *, root: Path | None = None, **extra: Any) -> dict[str, Any]:
    """The provenance block an artefact carries: commit, dirty, seed, environment.

    Deterministic by construction — no clock, no path, nothing that changes between two runs of
    the same code on the same machine, so `run it twice and diff` stays a valid check.
    """
    return {**git_provenance(root), "seed": seed, "environment": environment(), **extra}


def with_provenance(payload: dict, seed: int | None = None, **extra: Any) -> dict:
    """`payload` with a `provenance` block attached, unless it already carries one.

    Called from the `save()` of every artefact type rather than from each script, so a new script
    cannot forget it. An artefact that already stamped itself keeps what it wrote.
    """
    if "provenance" in payload:
        return payload
    return {**payload, "provenance": stamp(seed, **extra)}


def write_run_card(
    directory: str | Path,
    *,
    seed: int | None = None,
    config: Any = None,
    attack_params: Any = None,
    metrics: Any = None,
    command: list[str] | None = None,
    name: str = "run_card.json",
    **extra: Any,
) -> Path:
    """Everything needed to re-run this, in one file beside the artefacts it explains.

    Config, seed, attack parameters and metrics in one place, as ticket 20 asks — plus the clock
    facts that must not go into the artefacts themselves. A reader who has this file and the
    commit it names can reconstruct the command; a reader who has only the metrics cannot.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    card = {
        "run": {
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": command if command is not None else [Path(sys.argv[0]).name, *sys.argv[1:]],
            "cwd": str(Path.cwd()),
        },
        **stamp(seed),
        "config": config,
        "attack_params": attack_params,
        "metrics": metrics,
        **extra,
    }
    path = directory / name
    path.write_text(json.dumps(card, indent=2, default=str) + "\n")
    return path
