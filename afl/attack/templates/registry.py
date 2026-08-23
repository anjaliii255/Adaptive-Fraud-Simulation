"""S1..M3 → (engine, actor, param ranges).

The registry is the only place that knows which vectors exist. Adding a vector is a YAML edit;
if it ever requires an engine edit, the engine is under-parameterised.

It also knows which vectors this repo can actually *generate* today. A declared vector whose
engine cannot yet express it is `planned`, and asking for one raises rather than quietly
returning an empty episode — an attack family that silently generates nothing looks exactly
like an attack family the detector caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from afl.contract.schema import AttackParams, Rail

VECTORS_PATH = Path(__file__).with_name("vectors.yaml")
ENGINES = ("graph", "velocity", "drift")
MATURITIES = ("mature", "emerging", "research")
LEVELS = ("mechanism", "enabler", "model-attack")
TIERS = ("strong", "common", "mid")

#: What the code can do today, as opposed to what the taxonomy declares.
BUILT, TEMPLATE, PLANNED = "built", "template", "planned"
STATUSES = (BUILT, TEMPLATE, PLANNED)


@dataclass(frozen=True)
class VectorSpec:
    """One attack vector: which engine runs it, and the envelope its params must stay inside."""

    vector_id: str
    name: str
    engine: str
    actor: str
    level: str  # mechanism | enabler | model-attack — the taxonomy level, never flattened
    tier: str  # strong | common | mid — its role in the build
    maturity: str  # how well-evidenced the family is, so how much weight its numbers carry
    status: str  # built | template | planned — what this repo can generate today
    why: str
    params: dict[str, Any]
    search_space: dict[str, dict[str, Any]]
    #: what this vector does not yet do, and the ticket that fixes it. Empty when `built`.
    gap: str = ""
    #: actor-bundle knobs retuned for this vector alone (a card-testing vector must settle on
    #: the card rail). Merged over the shared bundle at generation time.
    actor_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def generatable(self) -> bool:
        """Whether `Simulator.generate` can produce this vector at all."""
        return self.status != PLANNED

    @property
    def reportable(self) -> bool:
        """Whether numbers measured on this family may be reported as a result for it.

        A `template` vector produces valid traffic of roughly the right shape, which is enough
        to train against and to fill a haystack — but not enough to claim a recall figure *for
        that family*, because its defining tell is still missing.
        """
        return self.status == BUILT

    def to_attack_params(self, overrides: dict[str, Any] | None = None) -> AttackParams:
        return AttackParams(
            vector_id=self.vector_id,
            engine=self.engine,
            params={**self.params, **(overrides or {})},
        )


def _parse_actor_overrides(vid: str, raw: dict[str, Any]) -> dict[str, Any]:
    """YAML gives rails as strings; the actor bundle wants a tuple of `Rail`."""
    out = dict(raw)
    if "rails" in out:
        try:
            out["rails"] = tuple(Rail(r) for r in out["rails"])
        except ValueError as exc:
            raise ValueError(f"{vid}: unknown rail in actor_overrides — {exc}") from exc
    return out


@lru_cache(maxsize=1)
def load_vectors(path: str | Path = VECTORS_PATH) -> dict[str, VectorSpec]:
    """Parse and validate vectors.yaml. Cached: the file is read once per process."""
    raw = yaml.safe_load(Path(path).read_text())["vectors"]
    specs: dict[str, VectorSpec] = {}
    for vid, body in raw.items():
        spec = VectorSpec(
            vector_id=vid,
            name=body["name"],
            engine=body["engine"],
            actor=body["actor"],
            level=body["level"],
            tier=body["tier"],
            maturity=body["maturity"],
            status=body["status"],
            why=body["why"],
            params=dict(body.get("params") or {}),
            search_space=dict(body.get("search_space") or {}),
            gap=str(body.get("gap") or "").strip(),
            actor_overrides=_parse_actor_overrides(vid, dict(body.get("actor_overrides") or {})),
        )
        for value, allowed, label in (
            (spec.engine, ENGINES, "engine"),
            (spec.maturity, MATURITIES, "maturity"),
            (spec.level, LEVELS, "level"),
            (spec.tier, TIERS, "tier"),
            (spec.status, STATUSES, "status"),
        ):
            if value not in allowed:
                raise ValueError(f"{vid}: unknown {label} {value!r}; expected one of {allowed}")
        if spec.status != BUILT and not spec.gap:
            raise ValueError(
                f"{vid}: status is {spec.status!r} but no `gap` says what is missing — "
                "an unfinished vector with no stated gap is indistinguishable from a finished one"
            )
        specs[vid] = spec
    return specs


def get(vector_id: str) -> VectorSpec:
    """One vector by id, or a KeyError naming the ones that exist."""
    vectors = load_vectors()
    if vector_id not in vectors:
        raise KeyError(f"unknown vector {vector_id!r}; known: {sorted(vectors)}")
    return vectors[vector_id]


def require_generatable(vector_id: str) -> VectorSpec:
    """The spec, or a loud refusal naming the ticket that would make it generatable."""
    spec = get(vector_id)
    if not spec.generatable:
        raise NotImplementedError(
            f"{vector_id} ({spec.name}) is declared but not implemented — {spec.gap}"
        )
    return spec


def list_vectors(
    maturity: str | None = None,
    engine: str | None = None,
    tier: str | None = None,
    level: str | None = None,
    status: str | None = None,
    generatable: bool | None = None,
) -> list[VectorSpec]:
    """Every vector, optionally filtered. `generatable=True` is what a run should iterate."""
    out = list(load_vectors().values())
    for value, attr in (
        (maturity, "maturity"),
        (engine, "engine"),
        (tier, "tier"),
        (level, "level"),
        (status, "status"),
    ):
        if value:
            out = [v for v in out if getattr(v, attr) == value]
    if generatable is not None:
        out = [v for v in out if v.generatable is generatable]
    return sorted(out, key=lambda v: v.vector_id)


def clamp(vector_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Clip searched params back into the vector's declared realism envelope."""
    space = get(vector_id).search_space
    out = dict(params)
    for k, bounds in space.items():
        if k not in out or out[k] is None:
            continue
        lo, hi = bounds["low"], bounds["high"]
        v = min(max(out[k], lo), hi)
        out[k] = int(round(v)) if bounds.get("type") == "int" else float(v)
    return out
