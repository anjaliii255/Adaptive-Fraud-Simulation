"""THE SEAM.

Field names here are frozen. Both owners code against them; changing one after
day one costs a day of merge pain on both sides.
"""

from afl.contract.metrics import Action, DetectorScore, MetricResult
from afl.contract.schema import AttackBatch, AttackParams, Entity, EntityRole, Rail, Transaction

__all__ = [
    "Action",
    "AttackBatch",
    "AttackParams",
    "DetectorScore",
    "Entity",
    "EntityRole",
    "MetricResult",
    "Rail",
    "Transaction",
]
