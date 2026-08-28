"""Which code produced this artefact.

A committed result that cannot be traced to the commit that made it is not reproducible, however
carefully every other input was recorded. This project learned that the expensive way: the first
A/B/C/D artefact carried its split digest, its seeds and its operating point, and still could not
be regenerated — it was written from a working tree that no commit captures, and the divergence
only surfaced days later when a re-run disagreed with it.

So every artefact records the commit it ran on, and whether that tree was dirty at the time. A
dirty tree is not an error and is not blocked; it is a fact the reader is entitled to, because
`git_dirty: true` is the difference between "re-run this commit" and "you cannot regenerate this".

Never raises. Git missing, no repository, a detached worktree: the field is `None`, which reads as
"cannot prove it" rather than being mistaken for clean.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Read-only git, or None when git is unavailable or the question has no answer."""
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


def git_provenance(cwd: Path | None = None) -> dict[str, Any]:
    """`{"git_commit": <sha or None>, "git_dirty": <bool or None>}` for the artefact header.

    `git_dirty` is None rather than False when the commit itself could not be read — an unknown
    tree state must not read as a clean one.
    """
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    if commit is None:
        return {"git_commit": None, "git_dirty": None}
    porcelain = _git("status", "--porcelain", cwd=cwd)
    return {"git_commit": commit, "git_dirty": None if porcelain is None else porcelain != ""}
