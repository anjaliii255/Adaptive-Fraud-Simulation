"""actors + engines → a labelled, schema-valid AttackBatch.

The simulator owns population and background traffic; the engines own attack shape. Fraud rows
are always injected into a *background* of legit traffic, because an attack batch with no
haystack is not a detection problem.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from afl.attack import actors as actor_lib
from afl.attack.engines import drift as drift_engine
from afl.attack.engines import graph as graph_engine
from afl.attack.engines import velocity as velocity_engine
from afl.attack.envelope import AnchorEnvelope
from afl.attack.templates import registry
from afl.contract.schema import AttackBatch, AttackParams, Entity, EntityRole, Transaction
from afl.utils.seed import child_seed
from afl.utils.seed import rng as make_rng

log = logging.getLogger(__name__)

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
        envelope: AnchorEnvelope | None = None,
    ) -> None:
        self.seed = seed
        self.n_entities = n_entities
        self.n_background = n_background
        self.n_episodes = n_episodes
        # an anchored run lives inside the real traffic's own window and amount scale, so an
        # attack cannot be picked out by its timestamp or its order of magnitude
        self.envelope = envelope
        self.start_ts = envelope.start if envelope else start_ts
        self.window_days = envelope.window_days if envelope else window_days
        self._round = 0
        self.entities = self._build_population(make_rng(child_seed(seed, "population")))
        self._merchant_pool: list[str] = list(envelope.payee_pool) if envelope else []
        self._relay_pool: list[str] = list(envelope.relays) if envelope else []

    # ── population ──────────────────────────────────────────────────────────────
    def _build_population(self, r) -> list[Entity]:
        ents: list[Entity] = []
        n_merchant = max(1, int(self.n_entities * 0.10))
        n_mule = max(1, int(self.n_entities * 0.05))
        n_fraudster = max(1, int(self.n_entities * 0.02))
        # anchored runs stage attacks on the anchor's own busy entities, so a synthetic sender
        # carries the same history a real one does instead of appearing from nowhere. Payees come
        # from their own pool: the two sides are different namespaces, and paying a customer
        # instead of a merchant is a row no real payment looks like.
        payees = list(self.envelope.active_payees) if self.envelope else []
        relays = sorted(set(self.envelope.relays)) if self.envelope else []
        senders = list(self.envelope.active_senders) if self.envelope else []
        minted = 0
        taken = 0
        for i in range(self.n_entities):
            if i < n_merchant:
                role, pool, index = EntityRole.MERCHANT, payees, i
            elif i < n_merchant + n_mule:
                # a mule receives money and forwards it, so it has to be an account the anchor
                # is also seen paying — drawing one from sender-space alone pays a stranger
                role, pool, index = EntityRole.MULE, relays, i - n_merchant
            else:
                role = (
                    EntityRole.FRAUDSTER
                    if i < n_merchant + n_mule + n_fraudster
                    else EntityRole.NORMAL
                )
                pool, index = senders, taken
                taken += 1
            # An anchored run draws every account from the anchor's own namespace, and wraps the
            # pool rather than inventing an id when the population is larger than the pool. An id
            # the anchor has never seen is a perfect label: `sender_in_anchor` alone separated
            # the held-out family from PaySim at PR-AUC 1.000 while this minted 332 of them.
            # Only the un-anchored synthetic default has no namespace to draw from.
            if pool:
                entity_id = pool[index % len(pool)]
            else:
                entity_id = f"e{i:05d}"
                minted += self.envelope is not None
            ents.append(
                Entity(
                    entity_id=entity_id,
                    role=role,
                    opened_at=self.start_ts - timedelta(days=int(r.integers(30, 2_000))),
                    country="IN",
                )
            )
        if minted:
            log.warning(
                "%d of %d simulated accounts had no anchor entity to stand on and were minted — "
                "they are separable from %s by account id alone",
                minted,
                self.n_entities,
                self.envelope.dataset,
            )
        # Wrapping a short pool means two population slots can name the same account. They are
        # the same account, so the entity list keeps one of it; drawing is unaffected, since the
        # pools are re-derived from `entities` by role.
        deduped = list({e.entity_id: e for e in ents}.values())
        if self.envelope and len(deduped) < len(ents):
            log.info(
                "%s supplies %d distinct accounts for a population of %d — the rest of the "
                "population would have been the same accounts twice",
                self.envelope.dataset,
                len(deduped),
                len(ents),
            )
        return deduped

    def _pool(self, role: EntityRole) -> list[str]:
        # merchants are drawn in the anchor's own proportions, so the category mix the
        # beneficiary id carries matches the traffic the attack is hiding in
        if role is EntityRole.MERCHANT and self._merchant_pool:
            return self._merchant_pool
        if role is EntityRole.MULE and self._relay_pool:
            return self._relay_pool
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
        actor = self.envelope.rescale(actor_lib.NORMAL) if self.envelope else actor_lib.NORMAL
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
        if self.envelope:
            actor = self.envelope.rescale(actor)
        victims = self._pool(EntityRole.NORMAL)
        mules = self._pool(EntityRole.MULE)
        cashout = self._pool(EntityRole.MERCHANT) + mules

        fraud: list[Transaction] = []
        minted: list[Entity] = []  # accounts this run had to invent, e.g. a fabricated identity
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
                # who pays decides the endpoints: an APP-scam victim pays out of their own account
                # to a hostile payee, an attacker probing a rail pays out of one they control
                if spec.actor == "normal":
                    v_src, v_dsts = str(ep_rng.choice(victims)), mules
                    v_device = f"dev-{v_src}"
                else:
                    v_src, v_dsts, v_device = str(ep_rng.choice(mules)), cashout, None
                episode += velocity_engine.generate(
                    rng=ep_rng,
                    run_id=ep_run,
                    vector_id=spec.vector_id,
                    actor=actor,
                    start_ts=start,
                    params=knobs,
                    src=v_src,
                    dst_pool=v_dsts,
                    device=v_device,
                )
            elif spec.engine == "drift":
                # a fabricated identity has no history to drift away from, so it gets an account
                # the population has never transacted with; every other vector drifts on a real one
                if knobs.get("new_account"):
                    d_src = f"{ep_run}-acct"
                    minted.append(
                        Entity(
                            entity_id=d_src, role=EntityRole.NORMAL, opened_at=start, country="IN"
                        )
                    )
                else:
                    d_src = str(ep_rng.choice(victims))
                episode += drift_engine.generate(
                    rng=ep_rng,
                    run_id=ep_run,
                    vector_id=spec.vector_id,
                    actor=actor,
                    start_ts=start,
                    params=knobs,
                    src=d_src,
                    benign_dst_pool=self._pool(EntityRole.MERCHANT),
                    cashout_pool=cashout,
                )
            else:  # pragma: no cover - registry.load_vectors already rejects this
                raise ValueError(f"unknown engine {spec.engine!r}")

            fraud += self._fit_into_window(episode)

        # the anchor has no device column, so the simulator does not invent one: a device id on
        # every synthetic row and none on every real row separates the two perfectly
        if self.envelope and not self.envelope.carries_devices:
            for t in fraud:
                t.device_id = None

        # and land on the anchor's own clock: AMLSim's rows are whole days, so traffic spread
        # across the hours is separable on hour-of-day alone
        if self.envelope and self.envelope.time_granularity_s > 1:
            daily = self.envelope.time_granularity_s >= 86_400
            for t in fraud:
                t.ts = t.ts.replace(
                    minute=0, second=0, microsecond=0, **({"hour": 0} if daily else {})
                )

        latest_fraud = max((t.ts for t in fraud), default=None)
        txns = self._background(r, run_id, end_ts=latest_fraud) + fraud
        txns.sort(key=lambda t: t.ts)
        # every row an episode emitted carries the run id, so evasions and the synthesised
        # history around them stay traceable to the params that produced them
        for t in txns:
            if t.attack_run_id:  # set by an engine, so this row came from an episode
                t.attack_run_id = run_id

        self._round += 1
        return AttackBatch(
            run_id=run_id,
            params=AttackParams(vector_id=spec.vector_id, engine=spec.engine, params=knobs),
            transactions=txns,
            seed=seed,
            entities=self.entities + minted,
        )
