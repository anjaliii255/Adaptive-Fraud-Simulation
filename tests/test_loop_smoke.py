"""⚑ The day-one gate: the loop runs end to end.

A hollow loop that runs beats two polished halves that never connect. This file must stay green
from the first commit — first against stubs, then against the real simulator and detector, with
neither side importing the other.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import afl.attack as attack_pkg
import afl.defend as defend_pkg
from afl.attack.optimiser import AttackOptimiser
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.metrics import Action, DetectorScore
from afl.contract.schema import AttackParams
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation.leave_one_attack_out import LeaveOneAttackOut
from afl.loop.closed_loop import Detector, Evaluator, Optimiser, find_evasions, run_closed_loop
from afl.loop.closed_loop import Simulator as SimulatorProtocol
from afl.loop.stubs import StubDetector, StubEvaluator, StubOptimiser, StubSimulator
from afl.tracking import InMemoryTracker

# ── the seam holds ──────────────────────────────────────────────────────────────
#: What either side is allowed to reach for. Anything else couples red to blue.
SHARED_PACKAGES = ("afl.contract", "afl.utils")


def afl_imports(source: Path) -> set[str]:
    """Every `afl.*` module this file imports, read from the parse tree rather than the text.

    A substring scan over the source would also match prose in a docstring, which is how a
    boundary check quietly becomes a spell-checker.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return {module for module in found if module == "afl" or module.startswith("afl.")}


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [(attack_pkg, "afl.defend"), (defend_pkg, "afl.attack")],
)
def test_neither_side_imports_the_other(package, forbidden):
    """The decoupling that lets two people work in parallel. Break it and the layout is a lie."""
    own_side = package.__name__
    for source in sorted(Path(package.__file__).parent.rglob("*.py")):
        for module in afl_imports(source):
            assert not module.startswith(forbidden), f"{source} imports {module}"
            assert module.startswith((own_side, *SHARED_PACKAGES)), (
                f"{source} imports {module}, which is neither its own side nor shared "
                f"({', '.join(SHARED_PACKAGES)})"
            )


def test_stubs_satisfy_the_loop_protocols():
    assert isinstance(StubSimulator(), SimulatorProtocol)
    assert isinstance(StubOptimiser(), Optimiser)
    assert isinstance(StubDetector(), Detector)
    assert isinstance(StubEvaluator(), Evaluator)


def test_real_components_satisfy_the_loop_protocols():
    assert isinstance(Simulator(n_entities=20, n_background=10), SimulatorProtocol)
    assert isinstance(AttackOptimiser(backend="random"), Optimiser)
    assert isinstance(LGBMDetector(), Detector)
    assert isinstance(LeaveOneAttackOut(holdout=[]), Evaluator)


# ── the evasion seam ────────────────────────────────────────────────────────────
def test_find_evasions_matches_by_id_not_position():
    sim = StubSimulator(seed=1, n_per_batch=20, fraud_rate=0.25)
    batch = sim.generate(AttackParams(vector_id="S1", engine="graph"))
    scores = [
        DetectorScore(txn_id=t.txn_id, score=0.0, action=Action.ALLOW) for t in batch.transactions
    ]
    assert len(find_evasions(batch, scores)) == len(batch.fraud_transactions)

    # shuffled order must not change the answer
    assert find_evasions(batch, list(reversed(scores))) == find_evasions(batch, scores)


def test_missing_score_is_a_loud_error():
    sim = StubSimulator(seed=1, n_per_batch=10)
    batch = sim.generate(AttackParams(vector_id="S1", engine="graph"))
    scores = [
        DetectorScore(txn_id=t.txn_id, score=0.0, action=Action.ALLOW)
        for t in batch.transactions[:-1]
    ]
    with pytest.raises(ValueError, match="one DetectorScore per transaction"):
        find_evasions(batch, scores)


def test_friction_is_not_evasion():
    sim = StubSimulator(seed=1, n_per_batch=10, fraud_rate=0.5)
    batch = sim.generate(AttackParams(vector_id="S1", engine="graph"))
    for action in (Action.STEP_UP, Action.HOLD, Action.REVIEW, Action.DECLINE):
        scores = [
            DetectorScore(txn_id=t.txn_id, score=0.5, action=action) for t in batch.transactions
        ]
        assert find_evasions(batch, scores) == []


# ── the hollow loop ─────────────────────────────────────────────────────────────
def test_loop_runs_on_dummy_data():
    tracker = InMemoryTracker("smoke")
    history = run_closed_loop(
        simulator=StubSimulator(seed=1),
        optimiser=StubOptimiser(),
        detector=StubDetector(),
        evaluator=StubEvaluator(),
        rounds=5,
        tracker=tracker,
    )
    assert len(history) == 5
    assert [h["round"] for h in history] == [0, 1, 2, 3, 4]
    for h in history:
        assert 0.0 <= h["evasion_rate"] <= 1.0
        assert {"pr_auc", "recall_at_fixed_fpr", "fixed_fpr", "precision_at_k"} <= set(h)


def test_loop_feeds_both_sides():
    optimiser, detector = StubOptimiser(), StubDetector()
    run_closed_loop(
        StubSimulator(seed=2), optimiser, detector, StubEvaluator(), 4, InMemoryTracker()
    )
    assert len(optimiser.history) == 4, "the optimiser was never told what got through"
    assert detector.retrain_calls == 4, "the detector never learnt from a round"


def test_evasion_rate_is_over_fraud_rows_not_all_rows():
    tracker = InMemoryTracker()
    sim = StubSimulator(seed=3, n_per_batch=100, fraud_rate=0.1)

    class AllowEverything:
        def score(self, batch):
            return [
                DetectorScore(txn_id=t.txn_id, score=0.0, action=Action.ALLOW)
                for t in batch.transactions
            ]

        def retrain(self, batch, evasions):
            pass

    run_closed_loop(sim, StubOptimiser(), AllowEverything(), StubEvaluator(), 1, tracker)
    assert tracker.history[0]["evasion_rate"] == 1.0  # not 0.1


# ── the real thing, small ───────────────────────────────────────────────────────
def test_loop_runs_with_real_components():
    """Same loop, stubs swapped for the real halves. This is step 4 of the build order."""
    simulator = Simulator(seed=13, n_entities=120, n_background=400, n_episodes=2)
    pool = []
    for vid in ("S1", "S2", "M3"):
        pool.extend(simulator.generate(registry.get(vid).to_attack_params()).transactions)

    evaluator, train = LeaveOneAttackOut.from_pool(pool, held_out_vector="M3")
    assert train, "nothing left to train on"
    assert all(t.vector_id != "M3" for t in train), "the held-out family leaked into training"

    detector = LGBMDetector(seed=13, params={"n_estimators": 40})
    detector.fit(train)

    optimiser = AttackOptimiser(vector_id="S1", seed=13, backend="random")
    tracker = InMemoryTracker("real-smoke")
    history = run_closed_loop(
        simulator=optimiser.bind(simulator),
        optimiser=optimiser,
        detector=detector,
        evaluator=evaluator,
        rounds=2,
        tracker=tracker,
    )
    assert len(history) == 2
    assert all(h["n_fraud"] > 0 for h in history)
    assert len(optimiser.trials) == 2
    assert detector.model is not None
