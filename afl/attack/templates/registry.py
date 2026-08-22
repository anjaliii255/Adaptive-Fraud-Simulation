"""S1..M3 → (engine, actor, param ranges).

The registry is the only place that knows which vectors exist. Adding a vector is a YAML edit;
if it ever requires an engine edit, the engine is under-parameterised.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from afl.contract.schema import AttackParams

VECTORS_PATH = Path(__file__).with_name("vectors.yaml")
ENGINES = ("graph", "velocity", "drift")
MATURITIES = ("mature", "emerging", "research")


@dataclass(frozen=True)
class VectorSpec:
    """One attack vector: which engine runs it, and the envelope its params must stay inside."""

    vector_id: str
    name: str
    engine: str
    actor: str
    maturity: str
    why: str
    params: dict[str, Any]
    search_space: dict[str, dict[str, Any]]

    def to_attack_params(self, overrides: dict[str, Any] | None = None) -> AttackParams:
        return AttackParams(
            vector_id=self.vector_id,
            engine=self.engine,
            params={**self.params, **(overrides or {})},
        )


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
            maturity=body["maturity"],
            why=body["why"],
            params=dict(body.get("params") or {}),
            search_space=dict(body.get("search_space") or {}),
        )
        if spec.engine not in ENGINES:
            raise ValueError(f"{vid}: unknown engine {spec.engine!r}")
        if spec.maturity not in MATURITIES:
            raise ValueError(f"{vid}: unknown maturity {spec.maturity!r}")
        specs[vid] = spec
    return specs


def get(vector_id: str) -> VectorSpec:
    """One vector by id, or a KeyError naming the ones that exist."""
    vectors = load_vectors()
    if vector_id not in vectors:
        raise KeyError(f"unknown vector {vector_id!r}; known: {sorted(vectors)}")
    return vectors[vector_id]


def list_vectors(maturity: str | None = None, engine: str | None = None) -> list[VectorSpec]:
    """Every vector, optionally filtered by maturity or engine."""
    out = list(load_vectors().values())
    if maturity:
        out = [v for v in out if v.maturity == maturity]
    if engine:
        out = [v for v in out if v.engine == engine]
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
