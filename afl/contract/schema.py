"""Wire types shared by the red and blue sides.

Rules of the seam:
  * every field name here is frozen as of day one;
  * synthetic rows carry provenance (`vector_id`, `attack_run_id`), real rows do not;
  * every batch carries the seed that produced it, so any row is reproducible.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

import pydantic


class Rail(str, Enum):
    """Payment rail the transaction settled on."""

    CARD = "card"
    UPI = "upi"
    A2A = "a2a"


class EntityRole(str, Enum):
    """What an entity is in the simulation. Never a feature; the defence must earn it."""

    NORMAL = "normal"
    FRAUDSTER = "fraudster"
    MULE = "mule"
    MERCHANT = "merchant"


class Entity(pydantic.BaseModel):
    """An account / card / merchant — a node in the transaction graph."""

    entity_id: str
    role: EntityRole = EntityRole.NORMAL
    opened_at: datetime | None = None
    country: str | None = None
    # free-form actor knobs (risk appetite, device pool size, ...) — never read by the
    # defence side, which must earn its features from transactions alone.
    attributes: dict = pydantic.Field(default_factory=dict)


class Transaction(pydantic.BaseModel):
    """One movement of money. The atom both sides trade in."""

    txn_id: str
    ts: datetime
    src: str  # source entity id
    dst: str  # beneficiary entity id
    amount: float
    rail: Rail
    device_id: str | None = None
    # label + provenance (synthetic rows only)
    is_fraud: bool = False
    vector_id: str | None = None  # e.g. "S1" for mule; None for legit/real
    attack_run_id: str | None = None

    @pydantic.field_validator("amount")
    @classmethod
    def _amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be > 0")
        return v


class AttackParams(pydantic.BaseModel):
    """The knobs one attack run was generated from."""

    vector_id: str  # S1..M3
    engine: str  # "graph" | "velocity" | "drift"
    params: dict = pydantic.Field(default_factory=dict)  # engine knobs, searched by optimiser


class AttackBatch(pydantic.BaseModel):
    """One attack run: its params, its rows, and the seed that reproduces both."""

    run_id: str
    params: AttackParams
    transactions: list[Transaction] = pydantic.Field(default_factory=list)
    seed: int  # every batch is reproducible
    entities: list[Entity] = pydantic.Field(default_factory=list)

    @property
    def fraud_transactions(self) -> list[Transaction]:
        return [t for t in self.transactions if t.is_fraud]

    def __len__(self) -> int:
        return len(self.transactions)
