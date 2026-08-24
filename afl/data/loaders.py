"""dataset → contract schema.

Every dataset enters the system through here and leaves as `list[Transaction]`. Nothing
downstream may know which dataset it came from — that is what makes the same detector, the same
features and the same evaluation run over PaySim, AMLSim and synthetic batches unchanged.

Two rules this module exists to enforce:

1. **Real rows carry no provenance.** `vector_id` and `attack_run_id` are for synthetic rows
   only. A real row that ever gains one has leaked a label path, so every real loader ends by
   calling `assert_no_provenance`.
2. **Only mapped columns are read.** The leak columns a dataset ships with (PaySim's
   `isFlaggedFraud` and its four balance columns) cannot reach a model because the loader never
   reads them into the contract at all. That is enforcement, not documentation.

Adding a single-file tabular dataset is a YAML edit: name it in `config/data/`, give it a
`schema_map`, and `load_from_config` does the rest. AMLSim is the documented exception — its
typology labels live in a second file and need a join.
"""

from __future__ import annotations

import logging
import os
import zlib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from afl.contract.schema import Rail, Transaction

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("AFL_DATA_DIR", "data/raw"))

#: Fixed anchor so an integer step maps to the same timestamp on every machine, every run.
DEFAULT_EPOCH = datetime(2023, 1, 1)

_UNIT_SECONDS = {"seconds": 1, "minutes": 60, "hours": 3_600, "days": 86_400}

#: Entity-hash sampling resolution. 1000 buckets means a fraction is honoured to 0.1%.
SAMPLE_BUCKETS = 1_000

#: Amounts at or below zero are clamped to this rather than dropped — dropping a row silently
#: changes the base rate, and the count of clamped rows is logged so the quirk stays visible.
MIN_AMOUNT = 0.01

#: Fields that must be `None` on every real row.
PROVENANCE_FIELDS = ("vector_id", "attack_run_id")


# ── errors ──────────────────────────────────────────────────────────────────────
class DatasetNotDownloaded(FileNotFoundError):
    """The config names a real anchor, the file is not on disk, and it is not ours to fetch."""


def _require(path: Path, dataset: str, source: Mapping | None = None) -> Path:
    if path.exists():
        return path
    hint = ""
    if source:
        url, place_in = source.get("url"), source.get("place_in")
        hint = f"\n  download: {url}\n  place in: {place_in}" if url else ""
    raise DatasetNotDownloaded(
        f"{dataset}: {path} not found. Real anchors are not committed (data/** is gitignored)."
        f"{hint}\n  or run on the zero-download default: data=synthetic"
    )


# ── the pieces every tabular loader shares ──────────────────────────────────────
def steps_to_timestamps(steps: np.ndarray, unit: str, epoch: datetime) -> np.ndarray:
    """Integer simulation steps → real timestamps, vectorised.

    PaySim and AMLSim both ship a step index rather than a clock. Anchoring on a fixed epoch is
    what makes `step → timestamp` deterministic, and therefore what makes a committed split
    boundary mean the same thing next week.
    """
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"unknown time unit {unit!r}; known: {sorted(_UNIT_SECONDS)}")
    base = np.datetime64(epoch, "s")
    return base + (steps.astype("int64") * _UNIT_SECONDS[unit]).astype("timedelta64[s]")


def entity_bucket(ids: pd.Series, buckets: int = SAMPLE_BUCKETS) -> pd.Series:
    """Stable hash bucket per entity id — crc32, not `hash()`, which is salted per process."""
    return ids.map(lambda s: zlib.crc32(str(s).encode()) % buckets)


def sample_by_entity(
    df: pd.DataFrame, column: str, fraction: float, buckets: int = SAMPLE_BUCKETS
) -> pd.DataFrame:
    """Keep every row of a deterministic hash-sample of entities.

    Row-wise sampling would shred the only thing worth having: an entity's history. Velocity,
    RFM and in-degree features are all computed over one entity's past, so a sample that keeps
    half of each account's rows produces features no production scorer would ever see.

    Sampling whole entities instead preserves each kept entity's complete history and the full
    time span, and — because the label is a per-row property spread across entities — holds the
    fraud base rate to within a couple of percent relative. Both are reported by
    `scripts/build_splits.py` rather than assumed.
    """
    if fraction >= 1.0:
        return df
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"sample fraction must be in (0, 1]; got {fraction}")
    cutoff = fraction * buckets
    keep = entity_bucket(df[column], buckets) < cutoff
    log.info(
        "entity sample: %.1f%% of %s → %d/%d rows", fraction * 100, column, int(keep.sum()), len(df)
    )
    return df[keep]


def _amounts(raw: pd.Series, dataset: str) -> np.ndarray:
    """Positive amounts, with the clamp counted out loud rather than applied in silence."""
    amounts = pd.to_numeric(raw, errors="coerce").to_numpy(dtype="float64")
    bad = ~np.isfinite(amounts) | (amounts <= 0)
    if bad.any():
        log.warning(
            "%s: %d row(s) with a non-positive or missing amount clamped to %.2f",
            dataset,
            int(bad.sum()),
            MIN_AMOUNT,
        )
        amounts = np.where(bad, MIN_AMOUNT, amounts)
    return amounts


def _to_transactions(
    *,
    dataset: str,
    txn_ids: list[str],
    timestamps: np.ndarray,
    src: list[str],
    dst: list[str],
    amounts: np.ndarray,
    rail: Rail,
    is_fraud: np.ndarray,
    device_ids: list[str | None] | None = None,
) -> list[Transaction]:
    """Build the contract rows.

    `model_construct` skips pydantic validation, which is what makes six-figure row counts
    tractable — so the guarantees the validators give are re-established here: amounts are made
    positive by `_amounts` above, and a canary row is pushed through the full constructor so a
    genuine schema break still fails loudly instead of being constructed around.
    """
    ts = pd.DatetimeIndex(timestamps).to_pydatetime()
    devices = device_ids or [None] * len(txn_ids)

    if txn_ids:  # the canary: full validation on one row, so a schema break cannot slip past
        Transaction(
            txn_id=txn_ids[0],
            ts=ts[0],
            src=src[0],
            dst=dst[0],
            amount=float(amounts[0]),
            rail=rail,
            device_id=devices[0],
            is_fraud=bool(is_fraud[0]),
        )

    rows = [
        Transaction.model_construct(
            txn_id=txn_ids[i],
            ts=ts[i],
            src=src[i],
            dst=dst[i],
            amount=float(amounts[i]),
            rail=rail,
            device_id=devices[i],
            is_fraud=bool(is_fraud[i]),
            # real rows are never given provenance — see assert_no_provenance below
            vector_id=None,
            attack_run_id=None,
        )
        for i in range(len(txn_ids))
    ]
    assert_no_provenance(rows, dataset)
    return rows


def assert_no_provenance(txns: list[Transaction], dataset: str = "real") -> None:
    """Real rows must carry `vector_id=None` and `attack_run_id=None`.

    Provenance is the red side's label path. On a real row it is an answer key: the evaluation
    carves families out by `vector_id`, so a real row wearing one lands in a holdout it has no
    business being in, and a feature builder that ever learned to read it would score perfectly
    on synthetic traffic and nothing at all in production.
    """
    for t in txns:
        for field in PROVENANCE_FIELDS:
            if getattr(t, field) is not None:
                raise AssertionError(
                    f"{dataset}: real row {t.txn_id!r} carries {field}={getattr(t, field)!r} — "
                    "provenance is for synthetic rows only and is a label path on a real one"
                )


def base_rate(txns: list[Transaction]) -> float:
    """Fraud share. Every operating point in the project is a function of this number."""
    return (sum(1 for t in txns if t.is_fraud) / len(txns)) if txns else 0.0


# ── PaySim ──────────────────────────────────────────────────────────────────────
#: The columns the contract needs. `isFlaggedFraud` (a naive >200k rule, not ground truth) and
#: the four balance columns (a generation artifact: fraud empties the account, so the label is
#: recoverable by arithmetic) are absent by construction, not by later filtering.
PAYSIM_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]


def load_paysim(
    path: str | Path | None = None,
    *,
    limit: int | None = None,
    sample_fraction: float = 1.0,
    sample_by: str = "nameDest",
    epoch: datetime = DEFAULT_EPOCH,
    time_unit: str = "hours",
    source: Mapping | None = None,
) -> list[Transaction]:
    """PaySim mobile-money A2A traffic.

    `step` is an hour index over ~31 days, so timestamps are synthesised off a fixed epoch.

    The default sample is taken over `nameDest` because that is the only entity in PaySim with a
    history: `nameOrig` is very nearly unique per row (6,353,307 distinct origins over 6,362,620
    rows), so there is no sender past to sample. See the data card.
    """
    path = Path(path or DATA_DIR / "PS_20174392719_1491204439457_log.csv")
    _require(path, "paysim", source)

    df = pd.read_csv(path, usecols=PAYSIM_COLUMNS)
    df = sample_by_entity(df, sample_by, sample_fraction)
    if limit:
        df = df.head(limit)

    return _to_transactions(
        dataset="paysim",
        # PaySim has no transaction id; the row's position in the file is the stable one
        txn_ids=[f"paysim-{i}" for i in df.index.to_numpy()],
        timestamps=steps_to_timestamps(df["step"].to_numpy(), time_unit, epoch),
        src=df["nameOrig"].tolist(),
        dst=df["nameDest"].tolist(),
        amounts=_amounts(df["amount"], "paysim"),
        rail=Rail.A2A,  # mobile money is account-to-account; PaySim has no card or UPI rail
        is_fraud=df["isFraud"].to_numpy().astype(bool),
    )


# ── BankSim ─────────────────────────────────────────────────────────────────────
BANKSIM_COLUMNS = ["step", "customer", "merchant", "category", "amount", "fraud"]


def load_banksim(
    path: str | Path | None = None,
    *,
    limit: int | None = None,
    sample_fraction: float = 1.0,
    sample_by: str = "customer",
    epoch: datetime = DEFAULT_EPOCH,
    time_unit: str = "days",
    source: Mapping | None = None,
) -> list[Transaction]:
    """BankSim retail card traffic: repeated customers paying a small set of merchants.

    Every value in the file is wrapped in literal single quotes ('C1093826151'), which have to
    come off or every entity id carries punctuation into the graph features.

    `category` is not a contract field, but it does not need to be: each of the 50 merchants has
    exactly one of 15 categories, so the beneficiary id already carries it and the merchant-side
    features can reach the signal.
    """
    path = Path(path or DATA_DIR / "banksim" / "bs140513_032310.csv")
    _require(path, "banksim", source)

    df = pd.read_csv(path, usecols=BANKSIM_COLUMNS)
    for column in ("customer", "merchant", "category"):
        df[column] = df[column].astype(str).str.strip("'")
    df = sample_by_entity(df, sample_by, sample_fraction)
    if limit:
        df = df.head(limit)

    return _to_transactions(
        dataset="banksim",
        txn_ids=[f"banksim-{i}" for i in df.index.to_numpy()],
        timestamps=steps_to_timestamps(df["step"].to_numpy(), time_unit, epoch),
        src=df["customer"].tolist(),
        dst=df["merchant"].tolist(),
        amounts=_amounts(df["amount"], "banksim"),
        rail=Rail.CARD,  # BankSim is card data, so CARD is the anchor's rail, not a leak
        is_fraud=df["fraud"].to_numpy().astype(bool),
    )


# ── AMLSim (the IBM example dump) ───────────────────────────────────────────────
AMLSIM_COLUMNS = ["TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TIMESTAMP"]


def _amlsim_frame(directory: Path, source: Mapping | None = None) -> pd.DataFrame:
    path = _require(directory / "transactions.csv", "amlsim", source)
    df = pd.read_csv(path, usecols=[*AMLSIM_COLUMNS, "IS_FRAUD"])
    # IS_FRAUD is a real bool in transactions.csv and the lowercase strings "true"/"false" in
    # accounts.csv. Normalising through str() costs nothing and survives either.
    df["IS_FRAUD"] = df["IS_FRAUD"].astype(str).str.lower().eq("true")
    return df


def load_amlsim(
    directory: str | Path | None = None,
    *,
    limit: int | None = None,
    sample_fraction: float = 1.0,
    sample_by: str = "SENDER_ACCOUNT_ID",
    epoch: datetime = DEFAULT_EPOCH,
    time_unit: str = "days",
    source: Mapping | None = None,
) -> list[Transaction]:
    """AMLSim mule-graph traffic. `TIMESTAMP` is a daily step index over 200 days.

    Unlike PaySim, senders here have deep histories (~132 transactions each), so the sample is
    taken over the sender — the entity the layering topology hangs off.
    """
    directory = Path(directory or DATA_DIR / "IBMAml")
    df = _amlsim_frame(directory, source)
    df = sample_by_entity(df, sample_by, sample_fraction)
    if limit:
        df = df.head(limit)

    return _to_transactions(
        dataset="amlsim",
        txn_ids=[f"amlsim-{i}" for i in df["TX_ID"].to_numpy()],
        timestamps=steps_to_timestamps(df["TIMESTAMP"].to_numpy(), time_unit, epoch),
        src=[f"acct-{s}" for s in df["SENDER_ACCOUNT_ID"].tolist()],
        dst=[f"acct-{d}" for d in df["RECEIVER_ACCOUNT_ID"].tolist()],
        amounts=_amounts(df["TX_AMOUNT"], "amlsim"),
        rail=Rail.A2A,  # every row is TX_TYPE=TRANSFER; the column carries no information
        is_fraud=df["IS_FRAUD"].to_numpy(),
    )


def amlsim_typologies(directory: str | Path | None = None) -> dict[str, str]:
    """`txn_id → 'fan_in' | 'cycle'`, deliberately kept off `Transaction`.

    The typology is what makes a leave-one-*variant*-out test on the mule family possible, but
    it is a label-side annotation, not a wire field. Writing it into `vector_id` would put
    provenance on a real row and make the carve-out treat AMLSim rows as a synthetic family.
    Ticket 11 reads this map; the seam never sees it.

    Joined on `TX_ID`, which is unique in both files. `ALERT_ID` is not: the dump's 1,719 alert
    rows share only 391 alert ids, so joining on it needs a de-duplication step that joining on
    the transaction id does not.
    """
    directory = Path(directory or DATA_DIR / "IBMAml")
    path = _require(directory / "alerts.csv", "amlsim")
    alerts = pd.read_csv(path, usecols=["TX_ID", "ALERT_TYPE"])
    return {f"amlsim-{int(r.TX_ID)}": str(r.ALERT_TYPE) for r in alerts.itertuples()}


# ── the config-driven entry point ───────────────────────────────────────────────
LOADERS = {"paysim": load_paysim, "amlsim": load_amlsim, "banksim": load_banksim}

#: Config keys `load_from_config` forwards to a loader. Anything else in a data config is for
#: the split, the data card or the feature side, and is deliberately not the loader's business.
_LOADER_KEYS = ("path", "directory", "limit", "sample_fraction", "sample_by")


def load(name: str, **kwargs) -> list[Transaction]:
    """Load a named dataset into contract types."""
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)


def load_from_config(cfg: Mapping) -> list[Transaction]:
    """One `config/data/*.yaml` → `list[Transaction]`.

    Returns `[]` for a config with no `loader` — the synthetic default, where the simulator's own
    background traffic stands in for real rows and there is nothing to download.
    """
    loader = cfg.get("loader")
    if not loader:
        return []

    kwargs = {k: v for k, v in (cfg.get("sample") or {}).items() if k in _LOADER_KEYS}
    kwargs.update({k: cfg[k] for k in _LOADER_KEYS if cfg.get(k) is not None})

    source = cfg.get("source") or {}
    if source:
        kwargs["source"] = source
        place_in = source.get("place_in")
        if place_in and "path" not in kwargs and "directory" not in kwargs:
            root = Path(place_in)
            file = source.get("transactions_file") or source.get("file")
            kwargs["directory" if loader == "amlsim" else "path"] = (
                root if loader == "amlsim" else root / file
            )

    time_cfg = cfg.get("time") or {}
    if time_cfg.get("epoch"):
        kwargs["epoch"] = datetime.fromisoformat(str(time_cfg["epoch"]))
    if time_cfg.get("unit"):
        kwargs["time_unit"] = str(time_cfg["unit"])

    txns = load(str(loader), **kwargs)
    log.info(
        "loaded %d rows from %s: %d fraud (%.4f%%), %s → %s",
        len(txns),
        cfg.get("name", loader),
        sum(1 for t in txns if t.is_fraud),
        base_rate(txns) * 100,
        min((t.ts for t in txns), default="-"),
        max((t.ts for t in txns), default="-"),
    )
    return txns


# ── the one place the two representations meet ──────────────────────────────────
def to_frame(txns: list[Transaction]) -> pd.DataFrame:
    """Contract → dataframe."""
    return pd.DataFrame([t.model_dump() for t in txns])


def from_frame(df: pd.DataFrame) -> list[Transaction]:
    """Dataframe back to contract types."""
    return [Transaction(**r) for r in df.to_dict(orient="records")]
