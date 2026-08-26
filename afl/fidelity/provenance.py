"""Did the bars exist before the numbers they judge?

"Thresholds are set before any result exists" is the kind of claim that is worth exactly nothing
when it is asserted in a docstring, because the failure mode it guards against — seeing a FAIL
and nudging a number until it passes — leaves a repo that asserts it just as loudly.

So it is checked instead. The values live in `config/fidelity/thresholds.yaml`; that file names
the commit each one was first committed in; and this module reads that commit **out of git**,
compares the value that was committed there against the value being applied now, and hands the
comparison to the scorecard, which writes it next to the verdict.

Three things end up in the artefact, and only the first is a claim about intent:

  origin        the commit the bars were committed in, its date, and whether every value still
                matches what that commit contains — a bar that has moved says so, and says by
                how much and in which direction
  working_copy  whether the thresholds file has uncommitted edits right now. If it does, nothing
                here can prove anything about when the values were chosen, and the record says
                that in place of a verdict
  history       every commit that has ever changed a bar. A run whose thresholds changed after
                an earlier result exists is visible in one list rather than in a blame trail

None of it stops anyone loosening a threshold. It stops them doing it quietly.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

#: Bars that are `{value, why}` pairs. Ordered as the scorecard reads them: diagnostics, gate,
#: privacy. `enabled` is a switch rather than a bar and is not required to carry a rationale.
BARS = (
    "level1_min",
    "level2_min",
    "max_tstr_gap",
    "min_recall_lift",
    "require_tstr_beats_amount_floor",
    "min_dcr_ratio",
    "max_mia_advantage",
)

LOOSER_WHEN_HIGHER = ("max_tstr_gap", "max_mia_advantage")
LOOSER_WHEN_LOWER = ("level1_min", "level2_min", "min_recall_lift", "min_dcr_ratio")


class ThresholdError(ValueError):
    """A bar with no value, or no stated reason for being where it is."""


@dataclass
class ThresholdProvenance:
    """The evidence that the bars predate the results, or the reason there is none."""

    source: str
    sha256: str
    values: dict[str, Any] = field(default_factory=dict)
    why: dict[str, str] = field(default_factory=dict)
    origin: dict[str, Any] = field(default_factory=dict)
    working_copy: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    checked_at: str = ""
    predates_results: bool = False
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return self.verdict


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Read-only git, or `None` when git is unavailable, absent, or the question has no answer.

    Every caller treats `None` as "cannot prove it", never as "fine" — an unprovable record is
    reported as unprovable rather than silently scored as clean.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _parse_bars(raw: dict[str, Any], source: str) -> tuple[dict[str, Any], dict[str, str]]:
    """`{value, why}` pairs in, values and rationales out. A blank `why` is refused.

    Same rule as `CostModel.from_config`, for the same reason: a comment can be deleted without
    anything noticing, a required field cannot. A bar nobody can justify in a sentence is a bar
    somebody tuned.
    """
    values: dict[str, Any] = {}
    why: dict[str, str] = {}
    missing: list[str] = []
    for name in BARS:
        entry = raw.get(name)
        if not isinstance(entry, dict) or "value" not in entry:
            missing.append(f"{name} (expected a {{value, why}} pair)")
            continue
        reason = str(entry.get("why") or "").strip()
        if not reason:
            missing.append(f"{name} (no `why`)")
            continue
        values[name] = entry["value"]
        why[name] = reason
    if missing:
        raise ThresholdError(
            f"every fidelity bar in {source} needs a value and a stated rationale; missing: "
            + ", ".join(missing)
        )
    return values, why


def _origin_values(raw: dict[str, Any], repo: Path) -> dict[str, Any]:
    """The bars as the origin commit actually contains them, read back out of git."""
    commit = str(raw.get("origin_commit") or "").strip()
    path = str(raw.get("origin_file") or "").strip()
    block = str(raw.get("origin_block") or "").strip()
    if not (commit and path):
        return {}
    blob = _git("show", f"{commit}:{path}", cwd=repo)
    if blob is None:
        return {}
    try:
        parsed = yaml.safe_load(blob) or {}
    except yaml.YAMLError:
        return {}
    section = parsed.get(block, parsed) if block else parsed
    if not isinstance(section, dict):
        return {}
    # the origin file predates the `{value, why}` shape, so both forms are read
    return {
        k: (v.get("value") if isinstance(v, dict) else v) for k, v in section.items() if k in BARS
    }


def _direction(name: str, then: Any, now: Any) -> str:
    """Did the bar move towards passing more things, or fewer?"""
    try:
        then_f, now_f = float(then), float(now)
    except (TypeError, ValueError):
        return "changed" if then != now else "unchanged"
    if then_f == now_f:
        return "unchanged"
    if name in LOOSER_WHEN_HIGHER:
        return "LOOSENED" if now_f > then_f else "tightened"
    if name in LOOSER_WHEN_LOWER:
        return "LOOSENED" if now_f < then_f else "tightened"
    return "changed"


def _history(path: Path, raw: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    """Every commit that has ever changed a bar, newest first, across both homes.

    Two different questions, so two different queries. The thresholds file holds nothing but
    bars, so *any* commit touching it changed one. Its previous home held the whole run config,
    so there only the commits that added or removed the block itself count — `-S` on the key,
    which is what finds the skeleton commit and nothing else.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    queries = [(str(path), ["log", "--format=%H%x1f%cI%x1f%s", "--", str(path)])]
    origin_file = str(raw.get("origin_file") or "").strip()
    if origin_file:
        argv = ["log", "--format=%H%x1f%cI%x1f%s", "-S", "max_tstr_gap", "--", origin_file]
        queries.append((origin_file, argv))
    for f, argv in queries:
        log = _git(*argv, cwd=repo)
        if not log:
            continue
        for line in log.splitlines():
            parts = line.split("\x1f")
            if len(parts) != 3 or parts[0] in seen:
                continue
            seen.add(parts[0])
            out.append(
                {
                    "commit": parts[0][:12],
                    "committed_at": parts[1],
                    "subject": parts[2],
                    "file": f,
                }
            )
    return sorted(out, key=lambda r: r["committed_at"], reverse=True)


def load(
    path: str | Path = "config/fidelity/thresholds.yaml",
    repo: str | Path | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, str], ThresholdProvenance]:
    """The bars, their rationales, and the evidence about when they were set.

    Raises `ThresholdError` if a bar has no value or no `why` — a scorecard is not built from
    numbers nobody wrote a reason for. Everything else is *reported*, never raised: a repo with
    no git history still gets a scorecard, it just gets one whose provenance says "unprovable"
    rather than one that quietly claims the bars are old.
    """
    path = Path(path)
    repo = Path(repo) if repo is not None else path.resolve().parent.parent.parent
    text = path.read_text()
    raw = yaml.safe_load(text) or {}
    values, why = _parse_bars(raw, str(path))
    checked_at = (now or datetime.now(UTC)).astimezone(UTC)

    # ── origin: what the naming commit actually contains ────────────────────────
    origin_commit = str(raw.get("origin_commit") or "").strip()
    stamp = _git("log", "-1", "--format=%cI", origin_commit, cwd=repo) if origin_commit else None
    then = _origin_values(raw, repo)
    moved = {
        name: {
            "then": then[name],
            "now": values[name],
            "direction": _direction(name, then[name], values[name]),
        }
        for name in then
        if then[name] != values[name]
    }
    inherited = sorted(then)
    origin = {
        "commit": origin_commit or None,
        "file": raw.get("origin_file"),
        "committed_at": stamp,
        "readable": bool(then),
        "bars_inherited": inherited,
        "bars_introduced_later": sorted(set(values) - set(then)),
        "unchanged_since_origin": bool(then) and not moved,
        "moved_since_origin": moved,
    }

    # ── working copy: can anything here be trusted right now? ───────────────────
    porcelain = _git("status", "--porcelain", "--", str(path), cwd=repo)
    last = _git("log", "-1", "--format=%H%x1f%cI%x1f%s", "--", str(path), cwd=repo)
    last_parts = last.split("\x1f") if last else []
    working_copy = {
        "clean": porcelain == "" if porcelain is not None else None,
        "uncommitted_changes": porcelain or None,
        "last_commit": last_parts[0][:12] if len(last_parts) == 3 else None,
        "last_committed_at": last_parts[1] if len(last_parts) == 3 else None,
        "last_subject": last_parts[2] if len(last_parts) == 3 else None,
    }

    history = _history(path, raw, repo)

    # ── the verdict, which is about evidence rather than about fidelity ─────────
    predates = False
    if working_copy["clean"] is False:
        verdict = (
            "UNPROVEN — the thresholds file has uncommitted edits, so nothing can show these "
            "values predate the numbers below"
        )
    elif not stamp:
        verdict = (
            "UNPROVEN — no readable git history for the thresholds, so the run cannot show when "
            "they were set"
        )
    elif moved:
        loosened = [k for k, v in moved.items() if v["direction"] == "LOOSENED"]
        verdict = f"MOVED since {origin_commit[:12]} ({', '.join(sorted(moved))})" + (
            f" — {len(loosened)} LOOSENED: {', '.join(sorted(loosened))}" if loosened else ""
        )
    else:
        origin_dt = datetime.fromisoformat(stamp).astimezone(UTC)
        predates = origin_dt < checked_at
        verdict = (
            f"the bars were committed in {origin_commit[:12]} on {origin_dt.date()} and every "
            f"value still matches that commit; this run started "
            f"{checked_at.isoformat(timespec='seconds')}"
        )

    return (
        values,
        why,
        ThresholdProvenance(
            source=str(path),
            sha256=hashlib.sha256(text.encode()).hexdigest()[:16],
            values=values,
            why=why,
            origin=origin,
            working_copy=working_copy,
            history=history,
            checked_at=checked_at.isoformat(timespec="seconds"),
            predates_results=predates,
            verdict=verdict,
        ),
    )
