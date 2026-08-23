"""actors + engines → a labelled, schema-valid AttackBatch.

The simulator owns population and background traffic; the engines own attack shape. Fraud rows
are always injected into a *background* of legit traffic, because an attack batch with no
haystack is not a detection problem.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from afl.attack import actors as actor_lib
from afl.attack.engines import drift as drift_engine
from afl.attack.engines import graph as graph_engine
from afl.attack.engines import velocity as velocity_engine
from afl.attack.templates import registry
from afl.contract.schema import AttackBatch, AttackParams, Entity, EntityRole, Transaction
from afl.utils.seed import child_seed
from afl.utils.seed import rng as make_rng

DEFAULT_START = datetime(2024, 1, 1)


class Simulator:
    """▲ A's half of the seam. `generate(params) -> AttackBatch`, and nothing else."""

    def __init__(
        self,
        seed: int = 1337,
        n_entities: int = 400,
        n_background: int = 2_000,
        start_ts: datetime = DEFAULT_START,
        window_days: int = 30,
        n_episodes: int = 5,
    ) -> None:
        self.seed = seed
        self.n_entities = n_entities
        self.n_background = n_background
        self.start_ts = start_ts
        self.window_days = window_days
        self.n_episodes = n_episodes
        self._round = 0
        self.entities = self._build_population(make_rng(child_seed(seed, "population")))

    # ── population ──────────────────────────────────────────────────────────────
    def _build_population(self, r) -> list[Entity]:
        ents: list[Entity] = []
        n_merchant = max(1, int(self.n_entities * 0.10))
        n_mule = max(1, int(self.n_entities * 0.05))
        n_fraudster = max(1, int(self.n_entities * 0.02))
        for i in range(self.n_entities):
            if i < n_merchant:
                role = EntityRole.MERCHANT
            elif i < n_merchant + n_mule:
                role = EntityRole.MULE
            elif i < n_merchant + n_mule + n_fraudster:
                role = EntityRole.FRAUDSTER
            else:
                role = EntityRole.NORMAL
            ents.append(
                Entity(
                    entity_id=f"e{i:05d}",
                    role=role,
                    opened_at=self.start_ts - timedelta(days=int(r.integers(30, 2_000))),
                    country="IN",
                )
            )
        return ents

    def _pool(self, role: EntityRole) -> list[str]:
        return [e.entity_id for e in self.entities if e.role == role]

    # ── background ──────────────────────────────────────────────────────────────
    def _background(self, r, run_id: str, end_ts: datetime | None = None) -> list[Transaction]:
        """Legit traffic across the whole window the attack actually occupies.

        `end_ts` is passed by `generate` so a long-dormancy vector (M2 can sleep for months once
        ticket 06 lands it) does not end up with its payout window sitting outside the legit
        traffic entirely — fraud with no contemporaneous haystack is trivially separable by
        timestamp alone.
        """
        normals = self._pool(EntityRole.NORMAL)
        merchants = self._pool(EntityRole.MERCHANT)
        actor = actor_lib.NORMAL
        default_end = self.start_ts + timedelta(days=self.window_days)
        span_s = int((max(end_ts or default_end, default_end) - self.start_ts).total_seconds())
        out: list[Transaction] = []
        for i in range(self.n_background):
            src = str(r.choice(normals))
            dst = str(r.choice(merchants if r.random() < 0.7 else normals))
            if dst == src:
                continue
            out.append(
                Transaction(
                    txn_id=f"{run_id}-b{i:06d}",
                    ts=self.start_ts + timedelta(seconds=int(r.integers(0, span_s))),
                    src=src,
                    dst=dst,
                    amount=round(float(r.lognormal(actor.amount_mu, actor.amount_sigma)), 2),
                    rail=actor.rails[int(r.integers(0, len(actor.rails)))],
                    device_id=f"dev-{src}",
                    is_fraud=False,
                )
            )
        return out

    def _fit_into_window(self, rows: list[Transaction]) -> list[Transaction]:
        """Slide an episode back so it finishes inside the simulation window.

        A seasoned family sleeps for weeks before it pays out. Left alone, its whole payout tail
        lands after every other family's traffic has ended — and an out-of-time split then sorts
        families by vector id instead of by time, which is not a temporal split at all.
        """
        if not rows:
            return rows
        end = self.start_ts + timedelta(days=self.window_days)
        overflow = (max(t.ts for t in rows) - end).total_seconds()
        if overflow <= 0:
            return rows
        room = (min(t.ts for t in rows) - self.start_ts).total_seconds()
        shift = timedelta(seconds=min(overflow, max(room, 0.0)))
        for t in rows:
            t.ts -= shift
        return rows

    # ── the seam method ─────────────────────────────────────────────────────────
    def generate(self, params: AttackParams) -> AttackBatch:
        # a declared-but-unbuilt vector refuses here rather than returning an empty episode:
        # an attack family that silently generates nothing reads exactly like one we caught
        spec = registry.require_generatable(params.vector_id)
        knobs = registry.clamp(params.vector_id, {**spec.params, **params.params})
        seed = child_seed(self.seed, params.vector_id, self._round, str(sorted(knobs.items())))
        r = make_rng(seed)
        run_id = f"{params.vector_id}-{self._round:03d}-{uuid.UUID(int=seed).hex[:8]}"

        # per-vector actor retune: card testing settles on the card rail, not on the shared default
        actor = actor_lib.get_actor(spec.actor, **spec.actor_overrides)
        victims = self._pool(EntityRole.NORMAL)
        mules = self._pool(EntityRole.MULE)
        cashout = self._pool(EntityRole.MERCHANT) + mules

        fraud: list[Transaction] = []
        for ep in range(self.n_episodes):
            ep_rng = make_rng(child_seed(seed, "episode", ep))
            start = self.start_ts + timedelta(
                seconds=int(ep_rng.integers(0, self.window_days * 86_400))
            )
            ep_run = f"{run_id}e{ep}"
            episode: list[Transaction] = []
            if spec.engine == "graph":
                episode += graph_engine.generate(
                    rng=ep_rng,
                    run_id=ep_run,
                    vector_id=spec.vector_id,
                    actor=actor,
                    start_ts=start,
                    params=knobs,
                    victim_pool=victims,
                    mule_pool=mules,
                    cashout_pool=cashout,
                )
            elif spec.engine == "velocity":
                episode += velocity_engine.generate(
                    rng=ep_rng,
                    run_id=ep_run,
                    vector_id=spec.vector_id,
                    actor=actor,
                    start_ts=start,
                    params=knobs,
                    src=str(ep_rng.choice(mules)),
                    dst_pool=cashout,
                )
            elif spec.engine == "drift":
                episode += drift_engine.generate(
                    rng=ep_rng,
                    run_id=ep_run,
                    vector_id=spec.vector_id,
                    actor=actor,
                    start_ts=start,
                    params=knobs,
                    src=str(ep_rng.choice(victims)),
                    benign_dst_pool=self._pool(EntityRole.MERCHANT),
                    cashout_pool=cashout,
                )
            else:  # pragma: no cover - registry.load_vectors already rejects this
                raise ValueError(f"unknown engine {spec.engine!r}")

            fraud += self._fit_into_window(episode)

        latest_fraud = max((t.ts for t in fraud), default=None)
        txns = self._background(r, run_id, end_ts=latest_fraud) + fraud
        txns.sort(key=lambda t: t.ts)
        # every fraud row must carry the run id, so evasions stay traceable to their params
        for t in txns:
            if t.is_fraud:
                t.attack_run_id = run_id

        self._round += 1
        return AttackBatch(
            run_id=run_id,
            params=AttackParams(vector_id=spec.vector_id, engine=spec.engine, params=knobs),
            transactions=txns,
            seed=seed,
            entities=self.entities,
        )
