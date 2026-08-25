"""FastAPI surface over the lab: /simulate /score /loop/step /metrics.

The demo is not a separate implementation. Every endpoint drives the same objects the
experiment scripts drive, so anything shown here is reproducible by `make loop` — a demo that
runs its own private code path is a sales tool, not evidence.

State is a single in-process lab, deliberately small so a round returns while someone watches.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import pydantic
import yaml
from fastapi import FastAPI, HTTPException

from afl.attack.optimiser import AttackOptimiser
from afl.attack.realism import check as realism_check
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.metrics import DetectorScore
from afl.contract.schema import AttackBatch, AttackParams, Transaction
from afl.data.splits import out_of_time_split
from afl.defend.decision import cost_model_for, policy_from_config
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import protocol
from afl.evaluation.leave_one_attack_out import DEFAULT_HOLDOUT, LeaveOneAttackOut
from afl.loop.closed_loop import find_evasions
from afl.tracking import InMemoryTracker
from afl.utils.seed import set_all_seeds

SEED = int(os.getenv("AFL_SEED", "1337"))
HELD_OUT = os.getenv("AFL_HELD_OUT_VECTOR", DEFAULT_HOLDOUT)
MAX_ROWS_RETURNED = 500
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def shipped_policy(rows: list[Transaction]):
    """The same decision policy `run_experiment.py` builds, from the same two config files.

    Read from config rather than defaulted, because a demo that shows different actions from
    the ones `make loop` takes is a sales tool. The cost model is denominated in the median
    payment of the traffic the demo actually generates — the same rule an anchored run uses,
    and without it the flat costs would be quoted against a scale this lab does not have.
    """
    costs_cfg = yaml.safe_load((CONFIG_DIR / "costs" / "default.yaml").read_text())
    lgbm_cfg = yaml.safe_load((CONFIG_DIR / "defend" / "lgbm.yaml").read_text())
    return policy_from_config(lgbm_cfg["decision"], cost_model_for(costs_cfg, rows))


@asynccontextmanager
async def lifespan(_: FastAPI):
    set_all_seeds(SEED)
    LAB.warm()
    yield


app = FastAPI(title="Adaptive Fraud Simulation Lab", version="0.1.0", lifespan=lifespan)


@dataclass
class Lab:
    """One warm lab per process. Cheap enough to rebuild, small enough to stay interactive."""

    simulator: Simulator = field(
        default_factory=lambda: Simulator(seed=SEED, n_background=600, n_episodes=2)
    )
    optimiser: AttackOptimiser = field(
        default_factory=lambda: AttackOptimiser(vector_id="S1", seed=SEED)
    )
    detector: LGBMDetector = field(
        default_factory=lambda: LGBMDetector(seed=SEED, explain="always")
    )
    tracker: InMemoryTracker = field(default_factory=lambda: InMemoryTracker("serve"))
    evaluator: LeaveOneAttackOut | None = None
    round: int = 0
    lock: Lock = field(default_factory=Lock)

    def warm(self) -> None:
        """Fit once on a pool that excludes the held-out family, so /score means something."""
        if self.evaluator is not None:
            return
        pool: list[Transaction] = []
        for vid in ("S1", "S2", "S3", HELD_OUT):
            pool.extend(self.simulator.generate(registry.get(vid).to_attack_params()).transactions)
        self.evaluator, train = LeaveOneAttackOut.from_pool(pool, held_out_vector=HELD_OUT)
        self.detector.policy = shipped_policy(train)
        # Same three steps as `scripts/run_experiment.py:calibrate` — fit on the head, learn the
        # score → probability map on the tail, refit on all of it. Skipped when the tail is too
        # thin, which at this deliberately small demo scale it often is; the calibrator then
        # stays the identity and says so.
        fit_rows, val_rows = out_of_time_split(train, train_frac=0.8, embargo_days=1.0)
        if val_rows and any(t.is_fraud for t in fit_rows):
            self.detector.fit(fit_rows)
            self.detector.policy.reset_calibration()
            y, s = protocol.align(
                val_rows, protocol.score_transactions(self.detector, val_rows, "calibration")
            )
            self.detector.policy.fit_calibrator(s, y)
        self.detector.fit(train)

    def reset(self) -> None:
        lock = self.lock  # keep the lock the caller is currently holding
        self.__init__()  # noqa: PLC2801 - a fresh lab is exactly what reset means
        self.lock = lock


LAB = Lab()


# ── request bodies ──────────────────────────────────────────────────────────────
class SimulateRequest(pydantic.BaseModel):
    """Generate one batch of a named vector."""

    vector_id: str = "S1"
    params: dict[str, Any] = pydantic.Field(default_factory=dict)
    include_transactions: bool = False


class ScoreRequest(pydantic.BaseModel):
    """Score arbitrary transactions against the warm detector."""

    transactions: list[Transaction]


class LoopRequest(pydantic.BaseModel):
    """Advance the closed loop by one or more rounds."""

    rounds: int = 1
    vector_id: str | None = None


# ── endpoints ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus the two facts that decide whether a number is trustworthy."""
    return {
        "status": "ok",
        "seed": SEED,
        "held_out_vector": HELD_OUT,
        "rounds_run": LAB.round,
        # str(), not the dataclass: this is JSON, and the demo shows it in a metric tile.
        # The full backend record — version and why it was chosen — is on the model card.
        "detector_backend": str(LAB.detector.backend),
        "detector_backend_reason": LAB.detector.backend.reason,
    }


@app.get("/vectors")
def vectors() -> list[dict[str, Any]]:
    """Every vector, flagged with which one is held out."""
    return [
        {
            "vector_id": v.vector_id,
            "name": v.name,
            "engine": v.engine,
            "actor": v.actor,
            "maturity": v.maturity,
            "why": v.why,
            "searchable": sorted(v.search_space),
            "held_out": v.vector_id == HELD_OUT,
        }
        for v in registry.list_vectors()
    ]


@app.post("/simulate")
def simulate(req: SimulateRequest) -> dict[str, Any]:
    """Generate one batch and report its realism verdict."""
    try:
        spec = registry.get(req.vector_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    with LAB.lock:
        batch = LAB.simulator.generate(spec.to_attack_params(req.params))
    report = realism_check(batch)
    out: dict[str, Any] = {
        "run_id": batch.run_id,
        "seed": batch.seed,
        "params": batch.params.model_dump(),
        "n_transactions": len(batch.transactions),
        "n_fraud": len(batch.fraud_transactions),
        "realism": {"penalty": report.penalty, "violations": report.violations, **report.detail},
    }
    if req.include_transactions:
        out["transactions"] = [
            t.model_dump(mode="json") for t in batch.transactions[:MAX_ROWS_RETURNED]
        ]
    return out


@app.post("/score")
def score(req: ScoreRequest) -> list[DetectorScore]:
    """Score arbitrary transactions with the warm detector."""
    if not req.transactions:
        raise HTTPException(status_code=400, detail="no transactions given")
    batch = AttackBatch(
        run_id="api-score",
        params=AttackParams(vector_id="api", engine="none"),
        transactions=req.transactions,
        seed=SEED,
    )
    with LAB.lock:
        return LAB.detector.score(batch)


@app.post("/loop/step")
def loop_step(req: LoopRequest) -> dict[str, Any]:
    """One full turn of the loop, so the convergence curve grows while you watch."""
    if req.vector_id == HELD_OUT:
        raise HTTPException(
            status_code=400,
            detail=f"{HELD_OUT} is the held-out family — generating it would train on the answer",
        )
    with LAB.lock:
        if req.vector_id and req.vector_id != LAB.optimiser.vector_id:
            LAB.optimiser = AttackOptimiser(vector_id=req.vector_id, seed=SEED)

        for _ in range(max(1, req.rounds)):
            params = LAB.optimiser.propose()
            batch = LAB.simulator.generate(params)
            LAB.optimiser.observe_batch(batch)
            scores = LAB.detector.score(batch)
            evasions = find_evasions(batch, scores)

            LAB.optimiser.update(evasions)
            LAB.detector.retrain(batch, evasions)
            metrics = LAB.evaluator.leave_one_attack_out(LAB.detector)

            n_fraud = len(batch.fraud_transactions)
            LAB.tracker.log(
                round=LAB.round,
                vector_id=batch.params.vector_id,
                n_transactions=len(batch.transactions),
                n_fraud=n_fraud,
                n_evasions=len(evasions),
                evasion_rate=(len(evasions) / n_fraud) if n_fraud else 0.0,
                **metrics.model_dump(),
            )
            LAB.round += 1

    latest = LAB.tracker.history[-1]
    return {
        "round": latest["round"],
        "latest": latest,
        "best_attack": LAB.optimiser.best.__dict__ if LAB.optimiser.best else None,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """The convergence history and every attack the optimiser has tried."""
    return {
        "held_out_vector": HELD_OUT,
        "rounds_run": LAB.round,
        "history": LAB.tracker.history,
        "attack_trials": LAB.optimiser.history(),
    }


@app.post("/loop/reset")
def loop_reset() -> dict[str, str]:
    """Throw the lab away and warm a fresh one."""
    with LAB.lock:
        LAB.reset()
        LAB.warm()
    return {"status": "reset"}
