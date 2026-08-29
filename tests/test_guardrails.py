"""◆ A+B — ticket 22. The guardrails are rules, or they are decoration.

`test_reproduce.py` proves the numbers in the documents are still the artefacts'. This proves the
sentences around them are too, which is the half a numeric check cannot see: every figure in

    "our C2ST came back near chance, which proves the attacks are realistic"

can be correct and the sentence still claims something no run in this repository supports.

Four properties, in the order they matter:

  * **The repository passes.** The live documents, audited as they stand.
  * **The harness discriminates.** Planted overstatements, at least one per guardrail, whose
    answers are known — because a checker that has only ever seen agreeing inputs is not evidence.
  * **A refusal is not a claim.** Every document here says "DPDP" and "sub-50ms" somewhere; only
    one way of saying them is a violation, and the sentences that refuse the claim must survive.
  * **The guardrails are stated.** Enforcing one that appears in no document leaves the reader to
    take on faith that anybody applied it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from afl.repro.claims import Registry
from afl.repro.guardrails import RULES_PATH, Guardrails, blocks, sentences

REPO = Path(__file__).resolve().parents[1]
WRITE_UP = "docs/submission.md"

#: The seven from the ticket, by id, so a guardrail cannot be quietly dropped from the registry.
SEVEN = (
    "fidelity-is-a-diagnostic",
    "privacy-is-evidence-not-proof",
    "exposure-not-compliance",
    "latency-depends-on-the-decision-point",
    "ai-risk-depends-on-deployment",
    "projections-are-labelled",
    "vectors-are-demonstrated-capabilities",
)

#: Sentences this project has no evidence for, under the guardrail that has to catch each one.
#: Every entry is the shape the overstatement actually takes in a submission deck.
OVERSTATEMENTS = {
    "fidelity-is-a-diagnostic": [
        "A C2ST near chance proves the generated attacks are realistic.",
        "The rows are statistically indistinguishable from real traffic.",
    ],
    "privacy-is-evidence-not-proof": [
        "DCR and MIA guarantee privacy for every generated row.",
        "The output is privacy-preserving by construction.",
    ],
    "exposure-not-compliance": [
        "Training on synthetic data alone is DPDP compliant.",
    ],
    "latency-depends-on-the-decision-point": [
        "We score every transaction in sub-50ms.",
        "Detection is guaranteed within 40 ms.",
    ],
    "ai-risk-depends-on-deployment": [
        "The system is not a high-risk AI system.",
        "A harness like this is exempt from the AI Act.",
    ],
    "projections-are-labelled": [
        "The loop prevented Rs 4 crore of fraud last quarter.",
        "Losses prevented across the pilot came to $2.3M.",
    ],
    "vectors-are-demonstrated-capabilities": [
        "M3 first-party fraud is widespread across payments.",
        "Fraudsters are using bust-out attacks at scale.",
    ],
}
PLANTED = [(rail, prose) for rail, cases in OVERSTATEMENTS.items() for prose in cases]

#: The same claims, refused. These are the sentences this repository actually writes, and a rule
#: that cannot tell them from the ones above would make the audit unusable rather than strict.
REFUSALS = [
    "A C2ST near chance is a diagnostic and does not prove the attacks are realistic.",
    "DCR and MIA are evidence against memorisation, not a privacy guarantee.",
    "Synthetic-only lowers exposure and does not by itself mean DPDP compliance.",
    "Latency depends on the decision point, so there is no fixed sub-50ms claim here.",
    "Whether this is high-risk depends on deployment context, and we make no determination.",
    "Any rupee figure here would be a projection built on assumptions, not money saved.",
    "M1 and M3 are demonstrated capabilities, not necessarily mass-exploited patterns.",
]


@pytest.fixture(scope="module")
def rails() -> Guardrails:
    return Guardrails.load(REPO / RULES_PATH)


def _audit_prose(rails: Guardrails, tmp_path: Path, prose: str) -> list:
    """Audit one paragraph as if it were the README, ignoring where guardrails are stated.

    The `unstated` check is about the repository's own write-up and has its own test; here the
    question is only whether a sentence trips a rule.
    """
    (tmp_path / "README.md").write_text(f"# Fixture\n\n{prose}\n")
    return [v for v in rails.audit(tmp_path) if v.kind != "unstated"]


# ── the repository, as it stands ────────────────────────────────────────────────
def test_no_document_in_this_repository_overstates_the_runs(rails: Guardrails) -> None:
    violations = rails.audit(REPO)
    assert not violations, "sentences claiming more than the artefacts support: " + "; ".join(
        f"{v.file} [{v.guardrail}] {v.excerpt}" for v in violations
    )


def test_all_seven_guardrails_are_registered(rails: Guardrails) -> None:
    assert tuple(g.id for g in rails.guardrails) == SEVEN


def test_every_guardrail_has_a_rule_behind_it(rails: Guardrails) -> None:
    """Listing a guardrail is not applying it. A guardrail nobody can fail is decoration."""
    assert not rails.unenforced(), "listed with no rule: " + ", ".join(
        g.id for g in rails.unenforced()
    )


def test_every_guardrail_is_written_down_in_the_write_up(rails: Guardrails) -> None:
    for guardrail in rails.guardrails:
        assert WRITE_UP in guardrail.stated_in, f"{guardrail.id} is enforced but never stated"
    assert (REPO / WRITE_UP).exists(), "the write-up the guardrails are stated in has to exist"


def test_the_audit_reaches_the_documents_a_reviewer_reads(rails: Guardrails) -> None:
    """A rule pointed at nothing passes trivially, so the scope is asserted rather than assumed."""
    scanned = {str(p.relative_to(REPO)) for p in rails.files(REPO)}
    for required in ("README.md", WRITE_UP, "docs/claim.md", "docs/results.md", "SPEC.md"):
        assert required in scanned, f"{required} is not audited"
    assert "docs/architecture.html" in scanned, "the pitch deck is prose too"


# ── the harness discriminates: answers known in advance ─────────────────────────
@pytest.mark.parametrize("guardrail,prose", PLANTED, ids=[p[:40] for _, p in PLANTED])
def test_a_planted_overstatement_is_caught(
    rails: Guardrails, tmp_path: Path, guardrail: str, prose: str
) -> None:
    caught = _audit_prose(rails, tmp_path, prose)
    assert guardrail in {v.guardrail for v in caught}, f"{guardrail} missed {prose!r}"


@pytest.mark.parametrize("prose", REFUSALS, ids=[p[:40] for p in REFUSALS])
def test_refusing_a_claim_is_not_making_it(rails: Guardrails, tmp_path: Path, prose: str) -> None:
    caught = _audit_prose(rails, tmp_path, prose)
    assert not caught, "a refusal was read as a claim: " + "; ".join(v.why for v in caught)


def test_a_specimen_inside_a_fence_is_not_read_as_prose(rails: Guardrails, tmp_path: Path) -> None:
    """The write-up quotes the sentences it refuses to write; a fence is how it does that."""
    fenced = "```\nWe score every transaction in sub-50ms.\n```"
    assert not _audit_prose(rails, tmp_path, fenced)
    assert _audit_prose(rails, tmp_path, "We score every transaction in sub-50ms.")


# ── the unit a rule is matched against ──────────────────────────────────────────
def test_a_sentence_is_not_split_on_a_decimal_point() -> None:
    text = "Evasion falls to 0.054 over six rounds. The gate rejected 0 of 42 rounds."
    assert sentences(text) == [
        "Evasion falls to 0.054 over six rounds.",
        "The gate rejected 0 of 42 rounds.",
    ]


def test_a_wrapped_sentence_is_one_unit() -> None:
    """Documents wrap at 100 columns; a rule that reads lines reads every claim in halves."""
    assert sentences("no fixed\nsub-50ms claim is made here.") == [
        "no fixed sub-50ms claim is made here."
    ]


def test_a_table_row_is_its_own_unit() -> None:
    """Otherwise an `unless` anywhere in a table exonerates every row of it."""
    table = "| a | we score in sub-50ms |\n| b | no fixed latency claim |"
    assert len(sentences(table)) == 2


def test_a_fenced_block_is_not_a_paragraph() -> None:
    assert blocks("before\n\n```\nfenced\n```\n\nafter") == ["before", "after"]


# ── the two halves of the audit meet on the same document ──────────────────────
def test_the_write_up_is_also_covered_by_the_numeric_registry() -> None:
    """The guardrails police its sentences; `claims.yaml` has to police its numbers."""
    registry = Registry.load(REPO / "docs/claims.yaml")
    covered = {r.file for r in registry.regions}
    assert WRITE_UP in covered, "the write-up's result sections are not registry-covered"
    assert not [g for g in registry.coverage(REPO) if g.file == WRITE_UP]
