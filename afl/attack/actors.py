"""Actor parameter bundles.

An actor is *behaviour*, not a label: the engines read these knobs to decide how an entity
moves money. The defence side never sees them — it must earn every feature from transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from afl.contract.schema import Entity, EntityRole, Rail


@dataclass
class ActorParams:
    """Behavioural knobs for one actor type. Engines read these; the defence never sees them."""

    role: EntityRole
    #: lognormal(mu, sigma) over transaction amount
    amount_mu: float = 3.0
    amount_sigma: float = 0.8
    #: mean seconds between this actor's transactions
    interarrival_mean_s: float = 3_600.0
    #: how many distinct counterparties the actor touches per day
    fanout_mean: float = 2.0
    #: device reuse — 1.0 means one device forever, 0.0 a fresh device per txn
    device_stickiness: float = 0.95
    rails: tuple[Rail, ...] = (Rail.A2A,)
    #: probability the actor keeps an amount just under a round reporting threshold
    threshold_awareness: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_entity(self, entity_id: str) -> Entity:
        return Entity(entity_id=entity_id, role=self.role, attributes={"actor": self.role.value})


NORMAL = ActorParams(
    role=EntityRole.NORMAL,
    amount_mu=3.2,
    amount_sigma=0.9,
    interarrival_mean_s=6 * 3_600.0,
    fanout_mean=1.5,
    device_stickiness=0.97,
    rails=(Rail.CARD, Rail.UPI, Rail.A2A),
)

FRAUDSTER = ActorParams(
    role=EntityRole.FRAUDSTER,
    amount_mu=4.0,
    amount_sigma=1.1,
    interarrival_mean_s=600.0,
    fanout_mean=8.0,
    device_stickiness=0.35,
    rails=(Rail.UPI, Rail.A2A),
    threshold_awareness=0.6,
)

MULE = ActorParams(
    role=EntityRole.MULE,
    amount_mu=3.6,
    amount_sigma=0.6,
    interarrival_mean_s=1_800.0,
    fanout_mean=4.0,
    device_stickiness=0.8,
    rails=(Rail.A2A,),
    threshold_awareness=0.8,
)

MERCHANT = ActorParams(
    role=EntityRole.MERCHANT,
    amount_mu=2.8,
    amount_sigma=1.3,
    interarrival_mean_s=120.0,
    fanout_mean=50.0,
    device_stickiness=1.0,
    rails=(Rail.CARD,),
)

REGISTRY: dict[EntityRole, ActorParams] = {
    EntityRole.NORMAL: NORMAL,
    EntityRole.FRAUDSTER: FRAUDSTER,
    EntityRole.MULE: MULE,
    EntityRole.MERCHANT: MERCHANT,
}


def get_actor(role: EntityRole | str, **overrides) -> ActorParams:
    """Fetch an actor bundle, optionally overriding knobs the optimiser is searching."""
    role = EntityRole(role)
    base = REGISTRY[role]
    if not overrides:
        return base
    merged = {**base.__dict__, **overrides}
    return ActorParams(**merged)
