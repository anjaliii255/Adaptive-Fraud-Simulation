"""Every number quoted in a document, and the artefact it is recomputed from.

A document is a cache of a computation. This repository has already been bitten twice by that:
an evasion curve that read `0.915 -> 0.201` in three documents after the artefact behind it had
been regenerated to `0.836 -> 0.054`, and a headline table whose `± sd` was population sd while
the script that produced the table printed sample sd. Both were true when written. Neither was
true when read, and nothing in the repository could tell the difference.

So a claim is not prose here. It is a row: an artefact, an expression evaluated over it, the
string that expression must format to, and the documents that string has to appear in. Change
the artefact and the expression stops matching. Change the sentence and the string stops being
found. Either way `make claims` goes red, which is the whole point — the failure lands on the
person editing, not on the reader six weeks later.

**Coverage is the second half.** Checking the registered numbers only proves the registered
numbers. So a `region` names a section of a document and asserts the opposite direction: every
numeric literal inside it is either a registered claim or an explicitly allowed constant with a
stated reason. An unregistered number appearing in a covered section is a failure, which is what
stops the registry silently falling behind the prose it governs.

The expression language is deliberately tiny and evaluated with no builtins: the helpers below
are the whole vocabulary, and each one is the reduction some document actually performs.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path("docs/claims.yaml")


# ── the reductions a document is allowed to perform ─────────────────────────────
def _namespace(blob: Any) -> dict[str, Any]:
    """The vocabulary an expression may use, bound to one artefact.

    Every helper here exists because some sentence in `docs/` performs exactly this reduction.
    Nothing takes a path outside the artefact, so a claim cannot quietly source a number from
    somewhere the reader was not told about.
    """

    def at(path: str, default: Any = None) -> Any:
        """`at('operating_point.fixed_fpr')` — dotted lookup into the artefact."""
        node = blob
        for part in path.split("."):
            if isinstance(node, list):
                node = node[int(part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def runs() -> list[dict]:
        return blob.get("runs", []) if isinstance(blob, dict) else []

    # --- the A/B/C/D shape: one entry per seed, each carrying results and a round trace ---
    def metric(system: str, key: str = "pr_auc") -> list[float]:
        """Per-seed values of one metric for one system."""
        return [r["results"][system][key] for r in runs()]

    def trace(field_name: str) -> list[list[float]]:
        """Per-seed round traces, clipped to the shortest run — the same rule make_figures uses."""
        n = min(len(r["convergence"]) for r in runs())
        return [[c[field_name] for c in r["convergence"][:n]] for r in runs()]

    def round_mean(field_name: str, index: int) -> float:
        """Mean over seeds of one round's value. `-1` is the last round."""
        return statistics.fmean(t[index] for t in trace(field_name))

    def total_rounds() -> int:
        return sum(len(r["convergence"]) for r in runs())

    def rounds_where(field_name: str) -> int:
        """How many (seed, round) pairs had this flag set — counted, rather than recalled."""
        return sum(1 for r in runs() for c in r["convergence"] if c[field_name])

    def distinct(field_name: str) -> int:
        """Distinct values a per-round field took across the whole run.

        The question ticket 14 asked of the realism penalty: is this number responding to
        anything, or is it the same constant reported forty-two times?
        """
        return len({c[field_name] for r in runs() for c in r["convergence"]})

    def flat(field_name: str) -> list[float]:
        """Every per-round value, all seeds, in one list."""
        return [c[field_name] for r in runs() for c in r["convergence"]]

    def wins(challenger: str, incumbent: str, key: str = "pr_auc") -> int:
        """Seeds on which `challenger` strictly beats `incumbent`. Ties count for the incumbent."""
        return sum(
            1 for r in runs() if r["results"][challenger][key] > r["results"][incumbent][key]
        )

    def sign_test(k: int, n: int) -> float:
        """One-sided binomial p — the same function `scripts/abcd_experiment.py` reports with."""
        if n == 0:
            return 1.0
        return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n

    # --- the three-system shape: runs -> systems -> column -> metrics|withheld_metrics ---
    def cell(system: str, column: str, key: str = "pr_auc") -> list[float]:
        """Per-seed values for one cell, withheld or not: brackets are a presentation choice."""
        out = []
        for r in runs():
            block = next(s for s in r["systems"] if s["system"] == system)[column]
            values = block.get("metrics") or block.get("withheld_metrics") or {}
            out.append(values[key])
        return out

    def floor(column: str, key: str = "pr_auc") -> list[float]:
        """The no-model amount floor for a column, per seed."""
        out = []
        for r in runs():
            block = r["systems"][0][column]
            out.append(block["floor"][key])
        return out

    def counts(path: str) -> Any:
        """A count from the first seed's `counts` block — the fold is the same on every seed."""
        node = runs()[0]["counts"]
        for part in path.split("."):
            node = node[part]
        return node

    return {
        # the artefact itself, for anything the helpers do not cover
        "blob": blob,
        "at": at,
        "runs": runs,
        "n_seeds": lambda: len(runs()),
        "seeds": lambda: [r["seed"] for r in runs()],
        # A/B/C/D
        "metric": metric,
        "trace": trace,
        "round_mean": round_mean,
        "total_rounds": total_rounds,
        "rounds_where": rounds_where,
        "distinct": distinct,
        "flat": flat,
        "wins": wins,
        "sign_test": sign_test,
        # three-system
        "cell": cell,
        "floor": floor,
        "counts": counts,
        # statistics — `sd` is the sample sd a 7-seed table should quote, `psd` the population
        # one `make figures` prints. Which is which has already gone wrong once, so both are
        # named rather than implied.
        "mean": statistics.fmean,
        "sd": lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "psd": statistics.pstdev,
        "median": statistics.median,
        # plain arithmetic
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "sorted": sorted,
    }


# ── the registry ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Claim:
    """One number, one derivation, one set of documents that must still be quoting it."""

    id: str
    what: str
    artifact: str
    expr: str
    quoted: str
    docs: tuple[str, ...]
    fmt: str = "{}"

    @classmethod
    def from_dict(cls, raw: dict) -> Claim:
        return cls(
            id=str(raw["id"]),
            what=str(raw.get("what", "")),
            artifact=str(raw["artifact"]),
            expr=str(raw["expr"]),
            quoted=str(raw["quoted"]),
            docs=tuple(raw.get("docs", ())),
            fmt=str(raw.get("fmt", "{}")),
        )


@dataclass(frozen=True)
class Region:
    """A section of a document in which every number has to be accounted for.

    `until` ends the region on a literal line instead of on the next heading — a README's H1
    section is the whole file, and the four-question table under it still has to be policed.
    """

    file: str
    section: str
    until: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> Region:
        return cls(
            file=str(raw["file"]), section=str(raw["section"]), until=str(raw.get("until", ""))
        )


@dataclass
class Check:
    """What became of one claim: the value recomputed, and whether the documents still say it."""

    claim: Claim
    computed: str | None
    ok: bool
    reason: str
    doc_hits: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        mark = "ok " if self.ok else "FAIL"
        where = " ".join(f"{Path(d).name}×{n}" for d, n in self.doc_hits.items())
        got = "" if self.computed == self.claim.quoted else f" (computed {self.computed})"
        return f"  [{mark}] {self.claim.id:<44} {self.claim.quoted:>10}{got}  {where}"


@dataclass
class Gap:
    """A number found in a covered section that no claim explains."""

    file: str
    section: str
    token: str
    context: str

    def line(self) -> str:
        return f"  [FAIL] {self.file}#{self.section}: {self.token!r} — {self.context}"


# A number as a document writes one: 1, 42, 0.0806, 83.6%, 446,214. Thousands separators only
# between digits, so `round 0, before` yields `0` rather than `0,`.
NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?%?")


def flatten(text: str) -> str:
    """Collapse whitespace, so a phrase that a document line-wrapped is still one phrase.

    `rejected 0 of\n42 rounds` is the same claim as `rejected 0 of 42 rounds`, and a checker that
    cannot see that teaches people to write around it.
    """
    return re.sub(r"\s+", " ", text)


# A quoted value ends where a longer number would continue, not where a word does: `recall@1%FPR`
# quotes `1%`, and `0.053%.` at the end of a sentence still quotes `0.053%`. What must never match
# is a prefix of a longer number — `0.065` inside `0.0652` — which is why both lookaheads are here.
_BEFORE = r"(?<![\w.])"
_AFTER = r"(?!\d)(?!\.\d)"


def _mask(text: str, token: str) -> str:
    """Blank out occurrences of a multi-token claim, so its parts are not read as loose numbers."""
    pattern = re.compile(_BEFORE + re.escape(token) + _AFTER)
    return pattern.sub(lambda m: " " * len(m.group()), text)


def occurrences(text: str, token: str) -> int:
    """How many times a document quotes this exact value, whitespace-insensitively."""
    return len(re.findall(_BEFORE + re.escape(token) + _AFTER, flatten(text)))


def section_text(doc: str, section: str, until: str = "") -> str:
    """The lines under a heading, up to `until` or the next heading of the same or higher level.

    Sections rather than whole files: the parts of a document that carry measurements are the
    parts worth policing, and demanding that every `python3.11` in a quickstart be registered as
    a claim would make the check noise rather than evidence.
    """
    level = len(section) - len(section.lstrip("#"))
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == section.strip())
    except StopIteration:
        return ""
    out: list[str] = []
    for line in lines[start + 1 :]:
        if until and line.strip() == until.strip():
            break
        stripped = line.lstrip("#")
        depth = len(line) - len(stripped)
        if not until and line.startswith("#") and depth <= level and stripped.startswith(" "):
            break
        out.append(line)
    return "\n".join(out)


@dataclass
class Registry:
    """The whole registry: the artefacts, the claims over them, and the covered regions."""

    artifacts: dict[str, str]
    claims: list[Claim]
    regions: list[Region]
    allow: dict[str, str]
    path: Path = REGISTRY_PATH

    @classmethod
    def load(cls, path: Path | str = REGISTRY_PATH) -> Registry:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        return cls(
            artifacts={str(k): str(v) for k, v in (raw.get("artifacts") or {}).items()},
            claims=[Claim.from_dict(c) for c in raw.get("claims") or ()],
            regions=[Region.from_dict(r) for r in raw.get("regions") or ()],
            allow={str(k): str(v) for k, v in (raw.get("allow") or {}).items()},
            path=path,
        )

    # -- evaluation ------------------------------------------------------------
    def blob(self, root: Path, key: str) -> Any:
        return json.loads((root / self.artifacts[key]).read_text())

    def evaluate(self, claim: Claim, blob: Any) -> str:
        """The claim's expression, formatted exactly as the document has to quote it.

        A tuple spreads into the format string, so a phrase that carries two derived numbers —
        `0 of 42` — stays one claim rather than two halves nobody reads together.
        """
        value = eval(claim.expr, {"__builtins__": {}}, _namespace(blob))  # noqa: S307
        return claim.fmt.format(*value) if isinstance(value, tuple) else claim.fmt.format(value)

    def check(self, root: Path | str = Path(".")) -> list[Check]:
        """Recompute every claim and confirm each document still quotes it."""
        root = Path(root)
        blobs: dict[str, Any] = {}
        checks: list[Check] = []
        for claim in self.claims:
            if claim.artifact not in self.artifacts:
                checks.append(Check(claim, None, False, f"unknown artefact {claim.artifact!r}"))
                continue
            try:
                if claim.artifact not in blobs:
                    blobs[claim.artifact] = self.blob(root, claim.artifact)
                computed = self.evaluate(claim, blobs[claim.artifact])
            except FileNotFoundError:
                checks.append(
                    Check(claim, None, False, f"missing artefact {self.artifacts[claim.artifact]}")
                )
                continue
            except Exception as exc:  # a broken expression is a failed claim, not a crash
                checks.append(Check(claim, None, False, f"{type(exc).__name__}: {exc}"))
                continue

            hits = {}
            for doc in claim.docs:
                text = (root / doc).read_text() if (root / doc).exists() else ""
                hits[doc] = occurrences(text, claim.quoted)
            missing = [d for d, n in hits.items() if n == 0]
            ok = computed == claim.quoted and not missing
            reason = ""
            if computed != claim.quoted:
                reason = f"artefact says {computed}, document says {claim.quoted}"
            elif missing:
                reason = "not quoted in " + ", ".join(missing)
            checks.append(Check(claim, computed, ok, reason, hits))
        return checks

    def coverage(self, root: Path | str = Path(".")) -> list[Gap]:
        """Every number in a covered section must be a claim or an allowed constant.

        Two passes, because a claim is not always one number. Phrases — `0 of 42`, `1/7`, a commit
        sha — are masked out of the text first; what remains is tokenised and each token has to be
        a registered or allowed value on its own.
        """
        root = Path(root)
        known = {c.quoted for c in self.claims} | set(self.allow)
        phrases = sorted((t for t in known if not NUMBER.fullmatch(t)), key=len, reverse=True)
        singles = {t for t in known if NUMBER.fullmatch(t)}

        gaps: list[Gap] = []
        for region in self.regions:
            path = root / region.file
            if not path.exists():
                gaps.append(Gap(region.file, region.section, "-", "document not found"))
                continue
            text = flatten(section_text(path.read_text(), region.section, region.until))
            if not text.strip():
                gaps.append(Gap(region.file, region.section, "-", "section not found or empty"))
                continue
            for phrase in phrases:
                text = _mask(text, phrase)
            for match in NUMBER.finditer(text):
                if match.group() in singles:
                    continue
                start = max(0, match.start() - 40)
                context = text[start : start + 96].strip()
                gaps.append(Gap(region.file, region.section, match.group(), context))
        return gaps


def report(checks: list[Check], gaps: list[Gap]) -> str:
    """The human-readable verdict — every claim, then every unexplained number."""
    lines = [f"claims: {sum(c.ok for c in checks)}/{len(checks)} verified"]
    lines += [c.line() for c in checks]
    failed = [c for c in checks if not c.ok]
    if failed:
        lines.append("")
        lines.append("failures:")
        lines += [f"  {c.claim.id}: {c.reason}" for c in failed]
    lines.append("")
    if gaps:
        lines.append(f"coverage: {len(gaps)} number(s) in covered sections trace to nothing")
        lines += [g.line() for g in gaps]
    else:
        lines.append("coverage: every number in every covered section traces to a claim")
    return "\n".join(lines)
