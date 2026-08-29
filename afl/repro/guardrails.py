"""The honesty guardrails, as something a command can fail on rather than a list nobody re-reads.

`claims.py` polices the numbers. This polices the sentences around them, which is where a result
is actually oversold. The seven guardrails ticket 22 names are all of the same shape — *this
instrument measures X, and X is not the thing a reader will hear* — and every one of them survives
a careful edit and dies in the next one:

    the provenance probe is a diagnostic       ->  "our attacks are proven realistic"
    DCR and MIA are evidence, not proof        ->  "the synthetic data is private"
    synthetic-only lowers exposure             ->  "DPDP compliant"
    latency depends on the decision point      ->  "sub-50ms scoring"
    AI-risk class depends on deployment        ->  "not a high-risk system"
    projections are projections                ->  "prevented Rs 4 crore of fraud"
    a demonstrated vector is a capability      ->  "the attack sweeping Indian payments"

So a guardrail here is not prose either. It is a rule with two halves, and both are needed:

  * **forbid** — the shape the overstatement takes, matched against one sentence at a time, with
    an `unless` that exonerates the sentence which is *refusing* the claim. Every document in this
    repository says "DPDP" at some point; only one way of saying it is a violation.
  * **require** — a term that may not appear naked. Naming C2ST at all obliges the same sentence
    to say what it is worth. This is the half that catches the honest-looking omission, which is
    the failure mode a banned-phrase list cannot see.

And one more, above both: a guardrail whose `statement` appears in no document is **unstated** and
fails. Listing the guardrails is not applying them, but applying them without writing them down
leaves the reader to trust that someone did.

**Fenced code blocks are specimens, not prose, and are skipped.** The write-up quotes the sentences
it refuses to write; quoting one inside a fence is how it does that without tripping the audit.
That is a documented hole and a deliberate one — an overstatement hidden in a fence would read to
any human as a code sample, which is a different failure from the one this catches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

RULES_PATH = Path("docs/guardrails.yaml")

FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
HTML_DROP = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
HTML_TAG = re.compile(r"<[^>]+>")
HTML_ENTITY = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ", "&mdash;": "—"}


# ── turning a document into the units a rule is matched against ─────────────────
def blocks(text: str) -> list[str]:
    """Paragraphs, with fenced code dropped. The unit a `paragraph`-scoped rule sees."""
    return [b.strip() for b in re.split(r"\n\s*\n", FENCE.sub("", text)) if b.strip()]


def sentences(block: str) -> list[str]:
    """One assertion at a time.

    Documents here wrap at 100 columns, so a sentence is not a line and a line-based checker would
    read every wrapped claim as two half-claims. Whitespace is flattened first, then the block is
    split on terminal punctuation followed by a capital — which leaves `0.836` and `e.g.` alone,
    because neither has whitespace after the dot.

    Table rows are their own units: a row carries a whole assertion, and rows have no full stops,
    so a flattened table would otherwise arrive as one enormous sentence in which any `unless`
    anywhere exonerates everything.
    """
    if block.lstrip().startswith("|"):
        return [line.strip() for line in block.splitlines() if line.strip()]
    flat = re.sub(r"\s+", " ", block).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'*_`\[])", flat)
    return [p for p in (part.strip() for part in parts) if p]


def readable(path: Path) -> str:
    """A document's prose. HTML is stripped to its text, because a pitch deck is prose too."""
    text = path.read_text()
    if path.suffix.lower() in {".html", ".htm"}:
        text = HTML_DROP.sub(" ", text)
        text = HTML_TAG.sub(" ", text)
        for entity, char in HTML_ENTITY.items():
            text = text.replace(entity, char)
        text = re.sub(r"[ \t]+", " ", text)
    return text


# ── the rules ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Forbid:
    """One shape an overstatement takes, and the sentence that is allowed to contain it anyway."""

    pattern: re.Pattern[str]
    why: str
    unless: re.Pattern[str] | None = None
    scope: str = "sentence"

    @classmethod
    def from_dict(cls, raw: dict) -> Forbid:
        return cls(
            pattern=re.compile(str(raw["pattern"]), re.IGNORECASE),
            why=str(raw["why"]),
            unless=re.compile(str(raw["unless"]), re.IGNORECASE) if raw.get("unless") else None,
            scope=str(raw.get("scope", "sentence")),
        )

    def hits(self, unit: str) -> bool:
        if not self.pattern.search(unit):
            return False
        return not (self.unless and self.unless.search(unit))


@dataclass(frozen=True)
class Require:
    """A term that may not be written without the qualifier that makes it honest."""

    when: re.Pattern[str]
    must: re.Pattern[str]
    why: str
    scope: str = "sentence"

    @classmethod
    def from_dict(cls, raw: dict) -> Require:
        return cls(
            when=re.compile(str(raw["when"]), re.IGNORECASE),
            must=re.compile(str(raw["must"]), re.IGNORECASE),
            why=str(raw["why"]),
            scope=str(raw.get("scope", "sentence")),
        )

    def hits(self, unit: str) -> bool:
        return bool(self.when.search(unit)) and not self.must.search(unit)


@dataclass(frozen=True)
class Guardrail:
    """One guardrail: what it says, where it has to be said, and how it is enforced."""

    id: str
    statement: str
    stated_in: tuple[str, ...]
    forbid: tuple[Forbid, ...]
    require: tuple[Require, ...]

    @classmethod
    def from_dict(cls, raw: dict) -> Guardrail:
        return cls(
            id=str(raw["id"]),
            statement=str(raw["statement"]),
            stated_in=tuple(raw.get("stated_in", ())),
            forbid=tuple(Forbid.from_dict(f) for f in raw.get("forbid", ())),
            require=tuple(Require.from_dict(r) for r in raw.get("require", ())),
        )

    @property
    def enforced(self) -> bool:
        """A guardrail with no rule is a guardrail that was listed, not applied."""
        return bool(self.forbid or self.require)


@dataclass(frozen=True)
class Violation:
    """One sentence that says more than the artefacts support, and which rule says so."""

    guardrail: str
    kind: str  # forbid | require | unstated
    file: str
    why: str
    excerpt: str

    def line(self) -> str:
        where = self.file or "-"
        return (
            f"  [FAIL] {self.guardrail:<28} {self.kind:<8} {where}\n"
            f"         {self.why}\n         {self.excerpt}"
        )


def excerpt(unit: str, limit: int = 150) -> str:
    flat = re.sub(r"\s+", " ", unit).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass
class Guardrails:
    """The seven guardrails, the documents they govern, and the audit over both."""

    documents: tuple[str, ...]
    guardrails: tuple[Guardrail, ...]
    path: Path = RULES_PATH

    @classmethod
    def load(cls, path: Path | str = RULES_PATH) -> Guardrails:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        return cls(
            documents=tuple(raw.get("documents", ())),
            guardrails=tuple(Guardrail.from_dict(g) for g in raw.get("guardrails") or ()),
            path=path,
        )

    def files(self, root: Path | str = Path(".")) -> list[Path]:
        """Every document in scope, in the order the registry names them."""
        root = Path(root)
        out: list[Path] = []
        for pattern in self.documents:
            out.extend(sorted(p for p in root.glob(pattern) if p.is_file()))
        return out

    def audit(self, root: Path | str = Path(".")) -> list[Violation]:
        """Every sentence in scope, against every rule. Empty means the wording holds."""
        root = Path(root)
        found: list[Violation] = []

        for guardrail in self.guardrails:
            for doc in guardrail.stated_in:
                path = root / doc
                text = readable(path) if path.exists() else ""
                if guardrail.statement.lower() not in re.sub(r"\s+", " ", text).lower():
                    found.append(
                        Violation(
                            guardrail.id,
                            "unstated",
                            doc,
                            "the guardrail is enforced but never written down where a reader looks",
                            guardrail.statement,
                        )
                    )

        for path in self.files(root):
            text = readable(path)
            rel = str(path.relative_to(root)) if path.is_absolute() else str(path)
            for block in blocks(text):
                units = sentences(block)
                for guardrail in self.guardrails:
                    for rule in guardrail.forbid:
                        for unit in [block] if rule.scope == "paragraph" else units:
                            if rule.hits(unit):
                                found.append(
                                    Violation(guardrail.id, "forbid", rel, rule.why, excerpt(unit))
                                )
                    for need in guardrail.require:
                        for unit in [block] if need.scope == "paragraph" else units:
                            if need.hits(unit):
                                found.append(
                                    Violation(guardrail.id, "require", rel, need.why, excerpt(unit))
                                )
        return found

    def unenforced(self) -> list[Guardrail]:
        return [g for g in self.guardrails if not g.enforced]


def report(rails: Guardrails, violations: list[Violation], scanned: int) -> str:
    """The human-readable verdict: every guardrail, its rules, and anything it caught."""
    by_id: dict[str, list[Violation]] = {}
    for violation in violations:
        by_id.setdefault(violation.guardrail, []).append(violation)

    lines = [f"guardrails: {len(rails.guardrails)} enforced over {scanned} document(s)"]
    for guardrail in rails.guardrails:
        hits = by_id.get(guardrail.id, [])
        mark = "ok " if not hits and guardrail.enforced else "FAIL"
        rules = f"{len(guardrail.forbid)} forbid, {len(guardrail.require)} require"
        lines.append(f"  [{mark}] {guardrail.id:<28} {rules:<22} {guardrail.statement}")
    if violations:
        lines.append("")
        lines.append("violations:")
        lines += [v.line() for v in violations]
    return "\n".join(lines)
