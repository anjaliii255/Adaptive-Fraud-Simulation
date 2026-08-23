"""Out-of-time splits. NEVER random.

A random split lets the model see tomorrow's fraud ring while scoring today's, which inflates
every number in the table. Fraud is non-stationary; the only honest split is chronological, with
an embargo gap so a slow-moving ring does not straddle the boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from afl.contract.schema import Transaction


def out_of_time_split(
    txns: list[Transaction],
    train_frac: float = 0.7,
    embargo_days: float = 1.0,
    val_frac: float = 0.0,
) -> (
    tuple[list[Transaction], list[Transaction]]
    | tuple[list[Transaction], list[Transaction], list[Transaction]]
):
    """Chronological split with an embargo gap; rows inside the gap are dropped, not assigned.

    Returns (train, test), or (train, val, test) when `val_frac > 0`.
    """
    if not txns:
        return ([], []) if val_frac <= 0 else ([], [], [])
    rows = sorted(txns, key=lambda t: t.ts)
    t0, t1 = rows[0].ts, rows[-1].ts
    span = (t1 - t0) or timedelta(seconds=1)
    embargo = timedelta(days=embargo_days)

    cut = t0 + span * train_frac
    if val_frac <= 0:
        train = [t for t in rows if t.ts <= cut]
        test = [t for t in rows if t.ts > cut + embargo]
        return train, test

    val_cut = t0 + span * (train_frac + val_frac)
    train = [t for t in rows if t.ts <= cut]
    val = [t for t in rows if cut + embargo < t.ts <= val_cut]
    test = [t for t in rows if t.ts > val_cut + embargo]
    return train, val, test


def split_at(
    txns: list[Transaction], cutoff: datetime, embargo_days: float = 1.0
) -> tuple[list[Transaction], list[Transaction]]:
    """Split at an explicit timestamp — use when the cutoff is a business fact, not a fraction."""
    embargo = timedelta(days=embargo_days)
    return (
        [t for t in txns if t.ts <= cutoff],
        [t for t in txns if t.ts > cutoff + embargo],
    )


def holdout_vector(
    txns: list[Transaction], vector_id: str
) -> tuple[list[Transaction], list[Transaction]]:
    """Leave-one-attack-out: pull one synthetic family out entirely.

    Legit rows stay on both sides — the held-out set still needs a haystack for FPR to mean
    anything.
    """
    seen = [t for t in txns if t.vector_id != vector_id]
    held = [t for t in txns if t.vector_id == vector_id or not t.is_fraud]
    return seen, held


def assert_no_leakage(train: list[Transaction], test: list[Transaction]) -> None:
    """Cheap guard worth running before every reported number."""
    if not train or not test:
        return
    latest_train = max(t.ts for t in train)
    earliest_test = min(t.ts for t in test)
    if earliest_test <= latest_train:
        raise AssertionError(
            f"temporal leakage: test starts {earliest_test} but train runs to {latest_train}"
        )
    overlap = {t.txn_id for t in train} & {t.txn_id for t in test}
    if overlap:
        raise AssertionError(
            f"{len(overlap)} txn_id(s) appear in both splits, e.g. {list(overlap)[:3]}"
        )


# ── the committed boundary ──────────────────────────────────────────────────────
#: Bump when the artefact's fields change shape, so an old file fails loudly instead of being
#: read with the wrong meaning.
SPLIT_ARTEFACT_VERSION = 1

DEFAULT_SPLIT_DIR = Path(os.getenv("AFL_SPLIT_DIR", "artifacts/splits"))


@dataclass(frozen=True)
class CommittedSplit:
    """An out-of-time boundary, computed once and read forever after.

    The point of committing it is comparability: `out_of_time_split` above takes a *fraction*,
    so the boundary it picks moves the moment the row set changes — a different sample fraction,
    a new download, one extra vector in the pool, and last week's number is measured against a
    different partition than this week's. A committed boundary is two timestamps. Applying it is
    a filter, and a filter gives the same answer every time.

    `train_end` is inclusive, `test_start` is inclusive, and everything strictly between them is
    the embargo: dropped, never assigned. The gap is what stops a slow-moving ring being
    half-trained-on and half-tested-on, and what stops a test row's velocity window being
    computed over training rows.
    """

    dataset: str
    train_end: datetime  # inclusive
    test_start: datetime  # inclusive
    embargo_rationale: str
    train_frac: float = 0.7
    time_unit: str = "hours"
    epoch: datetime | None = None
    train_end_step: int | None = None
    test_start_step: int | None = None
    stats: dict = field(default_factory=dict)
    version: int = SPLIT_ARTEFACT_VERSION

    def __post_init__(self) -> None:
        if self.test_start <= self.train_end:
            raise ValueError(
                f"{self.dataset}: embargo must be non-zero — test_start {self.test_start} is not "
                f"after train_end {self.train_end}"
            )
        if not str(self.embargo_rationale).strip():
            raise ValueError(
                f"{self.dataset}: an embargo with no recorded rationale is a magic number"
            )

    @property
    def embargo(self) -> timedelta:
        return self.test_start - self.train_end

    @property
    def digest(self) -> str:
        """Fingerprint of the boundary alone. Two runs agree iff this string agrees."""
        key = f"{self.dataset}|{self.train_end.isoformat()}|{self.test_start.isoformat()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    # ── use ─────────────────────────────────────────────────────────────────────
    def apply(self, txns: list[Transaction]) -> tuple[list[Transaction], list[Transaction]]:
        """(train, test) at the committed boundary. Deterministic, and independent of row count."""
        train = [t for t in txns if t.ts <= self.train_end]
        test = [t for t in txns if t.ts >= self.test_start]
        return train, test

    # ── round trip ──────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "dataset": self.dataset,
            "digest": self.digest,
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "embargo_seconds": int(self.embargo.total_seconds()),
            "embargo_rationale": " ".join(str(self.embargo_rationale).split()),
            "train_frac": self.train_frac,
            "time_unit": self.time_unit,
            "epoch": self.epoch.isoformat() if self.epoch else None,
            "train_end_step": self.train_end_step,
            "test_start_step": self.test_start_step,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> CommittedSplit:
        if int(raw.get("version", 0)) != SPLIT_ARTEFACT_VERSION:
            raise ValueError(
                f"split artefact version {raw.get('version')} != {SPLIT_ARTEFACT_VERSION}; "
                "rebuild it with scripts/build_splits.py rather than reading it as-is"
            )
        split = cls(
            dataset=raw["dataset"],
            train_end=datetime.fromisoformat(raw["train_end"]),
            test_start=datetime.fromisoformat(raw["test_start"]),
            embargo_rationale=raw["embargo_rationale"],
            train_frac=float(raw.get("train_frac", 0.7)),
            time_unit=str(raw.get("time_unit", "hours")),
            epoch=datetime.fromisoformat(raw["epoch"]) if raw.get("epoch") else None,
            train_end_step=raw.get("train_end_step"),
            test_start_step=raw.get("test_start_step"),
            stats=raw.get("stats", {}),
        )
        if raw.get("digest") and raw["digest"] != split.digest:
            raise ValueError(
                f"{split.dataset}: split artefact digest {raw['digest']} does not match the "
                f"boundary it contains ({split.digest}) — the file was edited by hand"
            )
        return split

    def save(self, directory: str | Path = DEFAULT_SPLIT_DIR) -> Path:
        path = Path(directory) / f"{self.dataset}_oot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, dataset: str, directory: str | Path = DEFAULT_SPLIT_DIR) -> CommittedSplit:
        path = Path(directory) / f"{dataset}_oot.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no committed split for {dataset!r} at {path} — run "
                "`python scripts/build_splits.py` once and commit the result"
            )
        return cls.from_dict(json.loads(path.read_text()))


def committed_split_for(cfg) -> CommittedSplit | None:
    """The committed boundary named by a data config, or `None` for the synthetic default.

    Read, never recomputed: a run that derives its own boundary is a run whose numbers are not
    comparable to yesterday's.
    """
    name = cfg.get("name")
    if not name or not cfg.get("loader"):
        return None
    directory = Path((cfg.get("split") or {}).get("commit_to", DEFAULT_SPLIT_DIR)).parent
    return CommittedSplit.load(str(name), directory)
