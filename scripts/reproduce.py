"""One command. Either the numbers in this repository reproduce, or you find out why not.

    python scripts/reproduce.py              # everything a fresh clone can check, ~2 minutes
    python scripts/reproduce.py --claims-only  # documents against artefacts only, ~1 second
    python scripts/reproduce.py --once         # skip the run-it-twice half
    python scripts/reproduce.py --record       # re-record the committed expectation
    python scripts/reproduce.py --anchor amlworld --anchor-seed 7   # re-run a real seed

Four stages, each of which can fail on its own:

  1. **environment** — the interpreter, the platform and the libraries that do the arithmetic.
     Never fails; it is the context every other stage is read in.
  2. **claims** — every number quoted in the documents, recomputed from the committed artefacts.
     This is how the *real* headline is checked on a machine with no anchor downloaded: the
     A/B/C/D result is re-derived from `artifacts/abcd/`, not re-run.
  3. **synthetic headline** — the whole loop, end to end, on the zero-download default, compared
     against a committed expectation. Then run again, and the two runs compared to each other:
     the same seed twice has to give the same number.
  4. **anchor** (optional) — re-run one seed of the real A/B/C/D and diff it against the committed
     artefact. Needs the anchor in `data/raw/`, so it is skipped rather than failed without it.

**Stage 3 is a pipeline check, not a result.** `data=synthetic` has no real anchor, and every
number it produces is stamped as such. What it proves is that the pipeline runs from a clean
clone and that it is deterministic — not that anything is true about fraud.

**Where a mismatch is not automatically a defect.** LightGBM's arithmetic is a function of its
build and its thread count, so a number produced on another platform may legitimately differ from
the recorded one. When the numbers differ *and* the environment differs, this prints UNCONFIRMED
and exits 2 — the difference is real and unexplained, and pretending it is a pass would be worse
than either. When the environment matches and the numbers do not, that is a defect: exit 1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afl.repro.claims import REGISTRY_PATH, Registry
from afl.utils.runcard import environment, stamp

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = Path("artifacts/reproduce/synthetic_headline.json")
REPORT = Path("artifacts/reproduce/report.json")
DEFAULT_SEED = 1337

PASS, FAIL, SKIP, UNCONFIRMED = "pass", "fail", "skip", "unconfirmed"

# What the synthetic run has to reproduce. Everything else in `metrics.json` is either a count
# that follows from these or a model card that describes how they were made.
HEADLINE_FIELDS = ("pr_auc", "recall@1%fpr", "precision@100", "evasion_rate", "train_rows")


@dataclass
class Stage:
    """One check, its verdict, and the evidence for it."""

    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        mark = {PASS: "PASS", FAIL: "FAIL", SKIP: "skip", UNCONFIRMED: "????"}[self.status]
        secs = f"{self.seconds:5.1f}s" if self.seconds else "      "
        return f"[{mark}] {self.name:<28} {secs}  {self.detail}"


# ── stage 2: the documents against the committed artefacts ──────────────────────
def stage_claims(root: Path, registry_path: Path = REGISTRY_PATH) -> Stage:
    started = time.time()
    registry = Registry.load(root / registry_path)
    checks = registry.check(root)
    gaps = registry.coverage(root)
    failed = [c for c in checks if not c.ok]
    status = PASS if not failed and not gaps else FAIL
    detail = f"{len(checks) - len(failed)}/{len(checks)} claims verified"
    if failed:
        detail += "; stale: " + ", ".join(c.claim.id for c in failed[:4])
    if gaps:
        detail += f"; {len(gaps)} number(s) with no artefact behind them"
    return Stage(
        "claims (documents)",
        status,
        detail,
        time.time() - started,
        {
            "checked": len(checks),
            "failed": [{"id": c.claim.id, "why": c.reason} for c in failed],
            "coverage_gaps": [
                {"file": g.file, "section": g.section, "token": g.token} for g in gaps
            ],
        },
    )


# ── stage 3: the loop itself, on the zero-download default ──────────────────────
def run_loop(seed: int, out_dir: Path, run_name: str = "adaptive") -> dict:
    """One synthetic end-to-end run, into a directory of its own. Returns its headline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "experiment=adaptive",
        f"seed={seed}",
        f"artifact_dir={out_dir}",
        f"hydra.run.dir={out_dir}/hydra",
        "tracker=memory",
    ]
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-12:])
        raise RuntimeError(f"the loop did not finish (exit {proc.returncode}):\n{tail}")
    return headline(out_dir / run_name, seed)


def headline(run_dir: Path, seed: int) -> dict:
    """The numbers a run has to reproduce, pulled out of the artefacts it just wrote."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    systems = {
        s["system"]: {k: s[k] for k in HEADLINE_FIELDS if k in s} for s in metrics["systems"]
    }
    card_path = run_dir / "fidelity_scorecard.json"
    fidelity = {}
    if card_path.exists():
        card = json.loads(card_path.read_text())
        fidelity = {"verdict": card.get("verdict"), "score": card.get("score")}
    return {
        "seed": seed,
        "data": metrics.get("data"),
        "pipeline_check": metrics.get("pipeline_check"),
        "operating_point": metrics.get("operating_point"),
        "backend": metrics.get("backend"),
        "systems": systems,
        "fidelity": fidelity,
    }


def differences(expected: dict, actual: dict, path: str = "") -> list[str]:
    """Every leaf on which two headlines disagree, named by its path."""
    out: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            out += differences(expected.get(key), actual.get(key), f"{path}.{key}".lstrip("."))
        return out
    if expected != actual:
        out.append(f"{path}: expected {expected!r}, got {actual!r}")
    return out


def environment_delta(expected: dict, actual: dict) -> list[str]:
    """How the machine differs from the one that recorded the expectation."""
    return differences(expected, actual, "environment")


def stage_synthetic(root: Path, seed: int, twice: bool, expected_path: Path) -> list[Stage]:
    """The loop, against the committed expectation, and then against itself."""
    stages: list[Stage] = []
    workdir = Path(tempfile.mkdtemp(prefix="afl-reproduce-"))
    try:
        started = time.time()
        first = run_loop(seed, workdir / "run1")
        elapsed = time.time() - started

        expected_blob = (
            json.loads((root / expected_path).read_text())
            if (root / expected_path).exists()
            else None
        )
        if expected_blob is None:
            stages.append(
                Stage(
                    "synthetic headline",
                    SKIP,
                    f"no committed expectation at {expected_path} — record one with --record",
                    elapsed,
                    {"actual": first},
                )
            )
        else:
            diffs = differences(expected_blob["headline"], first)
            env_diffs = environment_delta(
                expected_blob.get("recorded_on", {}).get("environment", {}), environment()
            )
            if not diffs:
                status, detail = PASS, "every headline number matches the committed expectation"
            elif env_diffs:
                status = UNCONFIRMED
                detail = (
                    f"{len(diffs)} number(s) differ on a machine that differs in "
                    f"{len(env_diffs)} way(s) — see the report, and docs/reproducibility.md"
                )
            else:
                status = FAIL
                detail = f"{len(diffs)} number(s) differ on an identical environment"
            # The expectation is only as traceable as the tree it was taken on. This is the same
            # defect that made the first A/B/C/D artefact irreproducible, so it is said out loud
            # rather than left in a field nobody opens.
            if expected_blob.get("recorded_on", {}).get("git_dirty"):
                detail += " (the expectation was recorded on a dirty tree — re-record it with "
                detail += "--record on the commit that ships it)"
            stages.append(
                Stage(
                    "synthetic headline",
                    status,
                    detail,
                    elapsed,
                    {"differences": diffs, "environment_differences": env_diffs, "actual": first},
                )
            )

        if twice:
            started = time.time()
            second = run_loop(seed, workdir / "run2")
            diffs = differences(first, second)
            verdict = "agree on every number" if not diffs else f"differ on {len(diffs)} number(s)"
            stages.append(
                Stage(
                    "determinism (same seed twice)",
                    PASS if not diffs else FAIL,
                    f"two runs of seed {seed} {verdict}",
                    time.time() - started,
                    {"differences": diffs},
                )
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return stages


# ── stage 4: the real anchor, when it is on the machine ─────────────────────────
# Two real experiments live in this repository and they are re-run by different scripts, so the
# anchor decides which one this stage means: amlworld is the A/B/C/D headline (ticket 12), the
# other anchors are the three-system table (ticket 16). `docs/results.md` sets them side by side
# and warns that their system labels collide.
def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _not_downloaded(output: str) -> str:
    """The loader's own message when the anchor is absent, or empty for any other failure.

    An anchor nobody downloaded is not a failed reproduction — the whole point of the synthetic
    default is that this stage is optional. Anything else is a real failure.
    """
    marker = "DatasetNotDownloaded: "
    line = next((ln for ln in reversed(output.splitlines()) if marker in ln), "")
    return line.split(marker, 1)[1].strip() if line else ""


def _table_headline(run: dict) -> dict:
    """Every measured cell of one seed of the three-system table, withheld ones included."""
    return {
        s["system"]: {
            column: (s[column].get("metrics") or s[column].get("withheld_metrics"))
            for column in ("known", "unseen")
            if column in s
        }
        for s in run["systems"]
    }


def stage_anchor(root: Path, anchor: str, typology: str, seed: int) -> Stage:
    """Re-run one seed of the real experiment for this anchor and diff it against the artefact."""
    if anchor == "amlworld":
        committed = root / f"artifacts/abcd/{anchor}_{typology.lower()}.json"
        script = [
            "scripts/abcd_experiment.py",
            "--data",
            anchor,
            "--typology",
            typology,
            "--seeds",
            str(seed),
        ]
        extract = lambda run: run["results"]  # noqa: E731
        label = f"A/B/C/D re-run ({anchor} seed {seed})"
    else:
        committed = root / f"artifacts/three_system/{anchor}.json"
        script = ["scripts/build_three_system.py", anchor, "--seeds", str(seed)]
        extract = _table_headline
        label = f"three-system re-run ({anchor} seed {seed})"

    if not committed.exists():
        return Stage(label, SKIP, f"no committed artefact at {committed.relative_to(root)}")
    blob = json.loads(committed.read_text())
    reference = next((r for r in blob["runs"] if r["seed"] == seed), None)
    if reference is None:
        seeds = ", ".join(str(r["seed"]) for r in blob["runs"])
        return Stage(label, SKIP, f"seed {seed} is not in the artefact (it has {seeds})")

    workdir = Path(tempfile.mkdtemp(prefix="afl-anchor-"))
    # `--out` on both scripts exists precisely so a trial run cannot overwrite the committed one
    out = workdir / "rerun.json" if anchor == "amlworld" else workdir
    started = time.time()
    try:
        code, output = _run([sys.executable, *script, "--out", str(out)])
        if code != 0:
            missing = _not_downloaded(output)
            if missing:
                return Stage(label, SKIP, missing, time.time() - started)
            tail = " / ".join(output.strip().splitlines()[-3:])
            return Stage(label, FAIL, tail, time.time() - started)
        path = out if anchor == "amlworld" else out / f"{anchor}.json"
        fresh = next(r for r in json.loads(path.read_text())["runs"] if r["seed"] == seed)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    diffs = differences(extract(reference), extract(fresh))
    same_commit = blob.get("git_commit") == stamp().get("git_commit")
    detail = (
        "every number matches the committed artefact"
        if not diffs
        else f"{len(diffs)} value(s) differ from the committed run"
    )
    if diffs and blob.get("git_commit") and not same_commit:
        produced_on = str(blob["git_commit"])[:8]
        detail += f" (artefact produced on {produced_on}; this tree is not that commit)"
    return Stage(
        label,
        PASS if not diffs else FAIL,
        detail,
        time.time() - started,
        {"differences": diffs, "same_commit": same_commit},
    )


# ── the command ────────────────────────────────────────────────────────────────
def record(root: Path, seed: int, expected_path: Path) -> int:
    """Re-record the committed expectation from a fresh run on this machine."""
    workdir = Path(tempfile.mkdtemp(prefix="afl-record-"))
    started = time.time()
    try:
        result = run_loop(seed, workdir / "run")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    blob = {
        "what": (
            "The headline of `python scripts/run_experiment.py experiment=adaptive` on the "
            "zero-download synthetic default. A PIPELINE CHECK, not a result: data=synthetic has "
            "no real anchor, so these numbers say the pipeline runs and is deterministic, and "
            "nothing about fraud. `scripts/reproduce.py` compares a fresh run against this."
        ),
        "command": f"python scripts/run_experiment.py experiment=adaptive seed={seed}",
        "seconds_when_recorded": round(time.time() - started, 1),
        "recorded_on": stamp(seed),
        "headline": result,
    }
    path = root / expected_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"recorded {path} in {blob['seconds_when_recorded']}s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--claims-only", action="store_true", help="documents vs artefacts, no models")
    p.add_argument("--once", action="store_true", help="skip the second run of the loop")
    p.add_argument("--record", action="store_true", help="re-record the committed expectation")
    p.add_argument("--expected", type=Path, default=EXPECTED)
    p.add_argument("--anchor", default=None, help="also re-run one seed of the real A/B/C/D")
    p.add_argument("--typology", default="GATHER-SCATTER")
    p.add_argument("--anchor-seed", type=int, default=7)
    p.add_argument("--report", type=Path, default=REPORT)
    args = p.parse_args()

    root = ROOT
    if args.record:
        return record(root, args.seed, args.expected)

    print("=" * 78)
    print("REPRODUCE — the documents, then the pipeline, then (optionally) the anchor")
    print("=" * 78)

    env = environment()
    stages = [
        Stage(
            "environment",
            PASS,
            f"python {env['python']} · {env['machine']} · lightgbm "
            f"{env['packages'].get('lightgbm')} · {env['cpu_count']} cpus",
            data=env,
        ),
        stage_claims(root),
    ]
    if not args.claims_only:
        stages += stage_synthetic(root, args.seed, twice=not args.once, expected_path=args.expected)
    if args.anchor:
        stages.append(stage_anchor(root, args.anchor, args.typology, args.anchor_seed))

    print()
    for stage in stages:
        print(stage.line())
    for stage in stages:
        for diff in stage.data.get("differences", [])[:10]:
            print(f"       {stage.name}: {diff}")

    report = {
        "provenance": stamp(args.seed),
        "stages": [
            {"name": s.name, "status": s.status, "detail": s.detail, "seconds": round(s.seconds, 1)}
            for s in stages
        ],
        "detail": {s.name: s.data for s in stages},
    }
    path = root / args.report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwritten to {args.report}")

    failed = [s for s in stages if s.status == FAIL]
    unconfirmed = [s for s in stages if s.status == UNCONFIRMED]
    if failed:
        print("\nFAIL: " + "; ".join(f"{s.name} — {s.detail}" for s in failed))
        return 1
    if unconfirmed:
        print(
            "\nUNCONFIRMED: "
            + "; ".join(f"{s.name} — {s.detail}" for s in unconfirmed)
            + "\nThe environment differs from the one that recorded the expectation, so this is"
            "\nnot automatically a defect — and it is not a pass either. docs/reproducibility.md"
            "\nsays what is known to be stable across machines and what is not."
        )
        return 2
    print("\nOK — everything this machine could check, reproduced.")
    if not args.anchor:
        print(
            "The real headline (AMLworld A/B/C/D) was re-derived from its committed artefact, not\n"
            "re-run. With the anchor in data/raw/, `--anchor amlworld` re-runs a seed of it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
