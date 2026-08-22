"""dataset → contract schema.

Every dataset enters the system through here and leaves as `list[Transaction]`. Nothing
downstream may know which dataset it came from — that is what makes the same detector,
features, and evaluation run over PaySim, IEEE-CIS, and synthetic batches unchanged.

Real rows carry `vector_id=None`: provenance fields are for synthetic rows only, and a real row
that ever gains one has leaked a label path.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from afl.contract.schema import Rail, Transaction

DATA_DIR = Path(os.getenv("AFL_DATA_DIR", "data/raw"))
PAYSIM_EPOCH = datetime(2024, 1, 1)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_paysim(path: str | Path | None = None, limit: int | None = None) -> list[Transaction]:
    """PaySim: `step` is an hour index, so timestamps are synthesised off a fixed epoch."""
    path = Path(path or DATA_DIR / "paysim.csv")
    df = _read_any(path)
    if limit:
        df = df.head(limit)
    out: list[Transaction] = []
    for i, row in df.iterrows():
        out.append(
            Transaction(
                txn_id=f"ps-{i}",
                ts=PAYSIM_EPOCH + timedelta(hours=int(row["step"])),
                src=str(row["nameOrig"]),
                dst=str(row["nameDest"]),
                amount=max(0.01, float(row["amount"])),
                rail=Rail.A2A,
                device_id=None,
                is_fraud=bool(int(row["isFraud"])),
            )
        )
    return out


def load_ieee_cis(
    txn_path: str | Path | None = None,
    identity_path: str | Path | None = None,
    limit: int | None = None,
) -> list[Transaction]:
    """IEEE-CIS: card rail, `TransactionDT` is seconds from an unstated reference point."""
    txn_path = Path(txn_path or DATA_DIR / "train_transaction.csv")
    df = _read_any(txn_path)
    if identity_path or (DATA_DIR / "train_identity.csv").exists():
        ident = _read_any(Path(identity_path or DATA_DIR / "train_identity.csv"))
        df = df.merge(ident, on="TransactionID", how="left")
    if limit:
        df = df.head(limit)

    ref = datetime(2017, 12, 1)
    device_col = "DeviceInfo" if "DeviceInfo" in df.columns else None
    out: list[Transaction] = []
    for _, row in df.iterrows():
        card = str(row.get("card1", "unknown"))
        out.append(
            Transaction(
                txn_id=f"ic-{int(row['TransactionID'])}",
                ts=ref + timedelta(seconds=int(row["TransactionDT"])),
                src=f"card-{card}",
                dst=f"merch-{row.get('P_emaildomain', 'unknown')}",
                amount=max(0.01, float(row["TransactionAmt"])),
                rail=Rail.CARD,
                device_id=(
                    str(row[device_col]) if device_col and pd.notna(row.get(device_col)) else None
                ),
                is_fraud=bool(int(row["isFraud"])),
            )
        )
    return out


LOADERS = {"paysim": load_paysim, "ieee_cis": load_ieee_cis}


def load(name: str, **kwargs) -> list[Transaction]:
    """Load a named dataset into contract types."""
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)


def to_frame(txns: list[Transaction]) -> pd.DataFrame:
    """Contract → dataframe. The only place the two representations are allowed to meet."""
    return pd.DataFrame([t.model_dump() for t in txns])


def from_frame(df: pd.DataFrame) -> list[Transaction]:
    """Dataframe back to contract types."""
    return [Transaction(**r) for r in df.to_dict(orient="records")]
