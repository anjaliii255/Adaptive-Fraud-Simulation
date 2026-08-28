"""▲ A — an artefact has to say which code produced it.

The first A/B/C/D artefact recorded its split digest, its seeds and its operating point, and still
could not be regenerated: it was written from a working tree no commit captures, and nothing in the
file said so. A re-run days later disagreed with it and there was no way to tell why from the
artefact alone. These tests are the fix for that, and they fail if the field is ever dropped.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from afl.utils.provenance import git_provenance

REPO = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "abcd_experiment", REPO / "scripts" / "abcd_experiment.py"
)
abcd = importlib.util.module_from_spec(_SPEC)
sys.modules["abcd_experiment"] = abcd
_SPEC.loader.exec_module(abcd)


def _git_available() -> bool:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=REPO,
                capture_output=True,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _git_available(), reason="not a git checkout")
def test_the_commit_is_recorded_and_is_not_null_on_a_normal_run():
    """The whole point: on a normal checkout the field is present and answerable."""
    prov = git_provenance(REPO)
    assert set(prov) == {"git_commit", "git_dirty"}
    assert prov["git_commit"] is not None, "a run in a git checkout must record its commit"
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_commit"]), prov["git_commit"]
    assert isinstance(prov["git_dirty"], bool)


@pytest.mark.skipif(not _git_available(), reason="not a git checkout")
def test_the_recorded_commit_is_the_one_git_reports():
    """A provenance field that does not match `git rev-parse` is worse than none."""
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert git_provenance(REPO)["git_commit"] == expected


def test_no_git_is_reported_as_unprovable_rather_than_crashing(tmp_path, monkeypatch):
    """Git missing must degrade to null, never take a three-hour run down with it."""
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH, so `git` cannot be found
    prov = git_provenance(tmp_path)
    assert prov == {"git_commit": None, "git_dirty": None}


def test_a_dirty_tree_is_recorded_as_dirty_not_hidden(tmp_path):
    """`git_dirty` separates 're-run this commit' from 'you cannot regenerate this'."""

    def run(*a):
        return subprocess.run(a, cwd=tmp_path, capture_output=True, check=True)

    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("one")
    run("git", "add", "f.txt")
    run("git", "commit", "-qm", "first")
    assert git_provenance(tmp_path)["git_dirty"] is False

    (tmp_path / "f.txt").write_text("two")
    assert git_provenance(tmp_path)["git_dirty"] is True


@pytest.mark.skipif(not _git_available(), reason="not a git checkout")
def test_the_experiment_header_carries_the_commit_beside_the_split_digest():
    """The field has to reach the artefact, not just exist in a helper nobody calls."""

    class Args:
        data, typology = "amlworld", "GATHER-SCATTER"
        fixed_fpr, k = 0.01, 100
        allocation, leash, lambda_realism = "search", "binding", 0.5
        rounds, episodes = 6, 12

    import yaml

    cfg = yaml.safe_load((REPO / "config/data/amlworld.yaml").read_text())
    header = abcd.artifact_header(cfg, Args())

    assert header["git_commit"] is not None
    assert isinstance(header["git_dirty"], bool)
    # the two provenance facts a reader needs sit together: which data, which code
    assert "split_digest" in header and "git_commit" in header
