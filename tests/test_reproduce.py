"""▲ A — ticket 20. A number that cannot be traced to a run is a bug, not a rounding difference.

Three properties, each of which has already failed in this repository at least once:

  * **The documents agree with the artefacts.** Three documents quoted an evasion curve of
    `0.915 -> 0.201` after the artefact behind it had been regenerated to `0.836 -> 0.054`, and
    the A/B/C/D table quoted a population sd that no script in the repository produces.
  * **The harness would notice.** A checker that only ever sees agreeing inputs is not evidence,
    so it is run against a stale document and an unregistered number whose answers are known.
  * **Every artefact says what made it.** Commit, seed and library versions, and — deliberately —
    no clock, because a timestamp in an artefact destroys the cheapest reproducibility check
    there is: run it twice and diff the bytes.

The expensive half of ticket 20 — the loop actually running twice and agreeing — lives in
`scripts/reproduce.py`, because it costs about eighty seconds and the suite is a gate, not a
benchmark. What is tested here is everything that can be checked in under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from afl.repro.claims import NUMBER, Registry, section_text
from afl.utils.runcard import environment, stamp, with_provenance, write_run_card

REPO = Path(__file__).resolve().parents[1]
EXPECTED = REPO / "artifacts/reproduce/synthetic_headline.json"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(REPO / "docs/claims.yaml")


# ── the documents against the artefacts ─────────────────────────────────────────
def test_every_registered_claim_still_falls_out_of_its_artefact(registry: Registry) -> None:
    stale = [(c.claim.id, c.reason) for c in registry.check(REPO) if not c.ok]
    assert not stale, "documents disagree with the artefacts they cite: " + "; ".join(
        f"{cid} ({why})" for cid, why in stale
    )


def test_no_number_in_a_covered_section_traces_to_nothing(registry: Registry) -> None:
    gaps = [(g.file, g.token, g.context) for g in registry.coverage(REPO)]
    assert not gaps, "unregistered numbers in a covered section: " + "; ".join(
        f"{f}: {t!r}" for f, t, _ in gaps
    )


def test_the_registry_covers_the_document_that_carries_the_headline(registry: Registry) -> None:
    """The check is worth nothing if it is pointed at sections nobody quotes."""
    covered = {(r.file, r.section) for r in registry.regions}
    assert ("README.md", "## Main result") in covered
    assert ("docs/results.md", "## The A/B/C/D experiment") in covered


def test_every_allowed_constant_states_why_it_is_not_a_measurement(registry: Registry) -> None:
    assert registry.allow, "an empty allow list means the reasons went somewhere else"
    for token, reason in registry.allow.items():
        assert len(reason) > 12, f"{token!r} is allowed without a reason anyone can check"


# ── the harness discriminates: cases whose answers are known ────────────────────
def _fixture_repo(tmp_path: Path, quoted: str, prose: str) -> Path:
    """A one-artefact, one-document repository with a registry over it."""
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/run.json").write_text(json.dumps({"runs": [{"results": {"A": 0.5}}]}))
    (tmp_path / "doc.md").write_text(f"## Result\n\n{prose}\n")
    (tmp_path / "claims.yaml").write_text(
        yaml.safe_dump(
            {
                "artifacts": {"run": "artifacts/run.json"},
                "regions": [{"file": "doc.md", "section": "## Result"}],
                "allow": {},
                "claims": [
                    {
                        "id": "a.value",
                        "what": "the only number in the fixture",
                        "artifact": "run",
                        "expr": "at('runs.0.results.A')",
                        "fmt": "{:.2f}",
                        "quoted": quoted,
                        "docs": ["doc.md"],
                    }
                ],
            }
        )
    )
    return tmp_path


def test_a_document_that_has_gone_stale_is_caught(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path, quoted="0.60", prose="The system scored 0.60.")
    checks = Registry.load(root / "claims.yaml").check(root)
    assert [c.ok for c in checks] == [False]
    assert "artefact says 0.50" in checks[0].reason


def test_a_claim_the_document_stopped_quoting_is_caught(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path, quoted="0.50", prose="The system scored well.")
    checks = Registry.load(root / "claims.yaml").check(root)
    assert not checks[0].ok
    assert "not quoted" in checks[0].reason


def test_an_unregistered_number_in_a_covered_section_is_caught(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path, quoted="0.50", prose="It scored 0.50, up from 0.31.")
    registry = Registry.load(root / "claims.yaml")
    assert [c.ok for c in registry.check(root)] == [True]
    assert [g.token for g in registry.coverage(root)] == ["0.31"]


def test_a_number_is_not_matched_by_a_prefix_of_a_longer_one(tmp_path: Path) -> None:
    """`0.50` must not be found inside `0.503`, or a stale table passes by coincidence."""
    root = _fixture_repo(tmp_path, quoted="0.50", prose="It scored 0.503.")
    checks = Registry.load(root / "claims.yaml").check(root)
    assert not checks[0].ok


def test_a_phrase_wrapped_across_two_lines_is_still_found() -> None:
    """Documents are wrapped at 100 columns; a claim must survive the wrap."""
    from afl.repro.claims import occurrences

    assert occurrences("the gate rejected 0 of\n42 rounds", "0 of 42") == 1


def test_a_section_ends_at_the_next_heading_of_its_level() -> None:
    doc = "## One\n\nalpha 1\n\n### Deeper\n\nbeta 2\n\n## Two\n\ngamma 3\n"
    body = section_text(doc, "## One")
    assert "alpha" in body and "beta" in body and "gamma" not in body


def test_the_number_pattern_reads_a_document_the_way_a_reader_does() -> None:
    text = "round 0, 83.6% of 446,214 rows at 0.0806 ± 0.0765"
    assert NUMBER.findall(text) == ["0", "83.6%", "446,214", "0.0806", "0.0765"]


# ── every artefact says what made it ────────────────────────────────────────────
def test_the_stamp_records_the_commit_the_seed_and_the_arithmetic() -> None:
    block = stamp(7)
    assert set(block) >= {"git_commit", "git_dirty", "seed", "environment"}
    assert block["seed"] == 7
    assert "lightgbm" in block["environment"]["packages"]
    assert block["environment"]["python"].startswith("3.")


def test_the_stamp_carries_no_clock() -> None:
    """Two stamps must be equal, or `run it twice and diff the bytes` stops working."""
    assert stamp(7) == stamp(7)
    flat = json.dumps(stamp(7))
    for clock in ("started_at", "generated_at", "checked_at", "timestamp"):
        assert clock not in flat


def test_with_provenance_leaves_an_artefact_that_stamped_itself_alone() -> None:
    already = {"provenance": {"git_commit": "deadbeef"}}
    assert with_provenance(already) == already


def test_the_run_card_carries_config_seed_attack_params_and_metrics(tmp_path: Path) -> None:
    """Ticket 20's checklist item, as a test rather than as a promise."""
    path = write_run_card(
        tmp_path,
        seed=1337,
        config={"data": {"name": "synthetic"}},
        attack_params=[{"vector": "S1", "intensity": 0.5}],
        metrics=[{"system": "A", "pr_auc": 0.5}],
    )
    card = json.loads(path.read_text())
    assert card["seed"] == 1337
    assert card["config"]["data"]["name"] == "synthetic"
    assert card["attack_params"][0]["vector"] == "S1"
    assert card["metrics"][0]["pr_auc"] == 0.5
    assert card["run"]["started_at"] and card["run"]["command"]
    assert set(card) >= {"git_commit", "git_dirty", "environment"}


def test_every_artefact_save_goes_through_the_stamp() -> None:
    """A new report type must not be able to ship without provenance by simply not asking."""
    unstamped = []
    for path in sorted((REPO / "afl").rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if "json.dumps(self.to_dict()" in line and "with_provenance" not in line:
                unstamped.append(f"{path.relative_to(REPO)}:{i}")
    assert not unstamped, "artefacts written without a provenance stamp: " + ", ".join(unstamped)


# ── the committed expectation the one command checks against ────────────────────
def test_the_committed_expectation_exists_and_names_the_machine_that_recorded_it() -> None:
    assert EXPECTED.exists(), "make reproduce has nothing to compare a fresh run against"
    blob = json.loads(EXPECTED.read_text())
    assert blob["headline"]["systems"], "an expectation with no numbers in it"
    packages = blob["recorded_on"]["environment"]["packages"]
    assert packages["lightgbm"], "the library that decides the numbers is not recorded"
    assert blob["command"].startswith("python scripts/run_experiment.py")


def test_the_expectation_is_marked_as_a_pipeline_check_and_can_never_be_quoted() -> None:
    blob = json.loads(EXPECTED.read_text())
    assert blob["headline"]["pipeline_check"] is True
    assert blob["headline"]["data"] == "synthetic"
    assert "PIPELINE CHECK" in blob["what"]


def test_this_machine_is_described_the_same_way_the_expectation_describes_its_own() -> None:
    """The comparison is only meaningful if both sides record the same fields."""
    recorded = json.loads(EXPECTED.read_text())["recorded_on"]["environment"]
    assert set(recorded) == set(environment())
    assert set(recorded["packages"]) == set(environment()["packages"])
