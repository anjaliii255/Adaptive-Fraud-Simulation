"""The one entry point. Every reported number comes out of here.

    python scripts/run_experiment.py experiment=adaptive
    python scripts/run_experiment.py -m experiment=baseline,smote,adaptive
    python scripts/run_experiment.py data=paysim eval.held_out_vector=S2 seed=7

Nothing in this file decides anything: it assembles components from config, runs them, and
writes artefacts. A number that cannot be reproduced from (config, seed) is a bug.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf

from afl.attack.optimiser import AttackOptimiser
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import out_of_time_split
from afl.defend.decision import DecisionPolicy
from afl.defend.features import FeatureBuilder
from afl.defend.models.anomaly import AnomalyDetector, EnsembleDetector
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import leave_one_attack_out as loao
from afl.evaluation import three_system
from afl.fidelity import scorecard
from afl.tracking import get_tracker
from afl.utils.seed import set_all_seeds

log = logging.getLogger(__name__)


# ── assembly ────────────────────────────────────────────────────────────────────
def build_simulator(cfg: DictConfig) -> Simulator:
    e = cfg.attack.engines
    from datetime import datetime

    return Simulator(
        seed=cfg.seed,
        n_entities=int(e.n_entities),
        n_background=int(e.n_background),
        start_ts=datetime.fromisoformat(str(e.start_ts)),
        window_days=int(e.window_days),
        n_episodes=int(e.n_episodes),
    )


def build_detector_factory(cfg: DictConfig):
    sup, uns = cfg.defend.supervised, cfg.defend.unsupervised
    dec = sup.decision

    def factory():
        policy = DecisionPolicy(
            mode=str(dec.mode),
            step_up_at=float(dec.step_up_at),
            hold_at=float(dec.hold_at),
            review_at=float(dec.review_at),
            decline_at=float(dec.decline_at),
        )
        model = LGBMDetector(
            policy=policy,
            features=FeatureBuilder(
                stateful=bool(sup.features.stateful),
                windows_s=tuple(int(w) for w in sup.features.windows_s),
            ),
            params=OmegaConf.to_container(sup.params),
            seed=int(cfg.seed),
            replay_weight=float(sup.replay_weight),
            explain=bool(sup.explain),
        )
        if bool(uns.ensemble.enabled):
            return EnsembleDetector(
                supervised=model,
                unsupervised=AnomalyDetector(
                    kind=str(uns.kind), contamination=float(uns.contamination), seed=int(cfg.seed)
                ),
                weight=float(uns.ensemble.weight),
                policy=policy,
            )
        return model

    return factory


def build_pool(cfg: DictConfig, simulator: Simulator) -> list[Transaction]:
    """Real traffic plus one attack batch per allowed vector — the input to every split."""
    held_out = str(cfg.eval.held_out_vector)
    allowed = [v for v in cfg.attack.engines.vectors]
    if held_out in allowed:
        raise ValueError(
            f"{held_out!r} is both the eval holdout and a generated vector — "
            "the red side would be handing the blue side the answer"
        )

    real: list[Transaction] = []
    if cfg.data.loader:
        kwargs = {
            k: v
            for k, v in OmegaConf.to_container(cfg.data).items()
            if k in ("path", "txn_path", "identity_path", "limit") and v is not None
        }
        real = loaders.load(str(cfg.data.loader), **kwargs)
        log.info("loaded %d real rows from %s", len(real), cfg.data.name)

    pool = list(real)
    for vid in allowed:
        batch = simulator.generate(registry.get(vid).to_attack_params())
        pool.extend(
            batch.transactions if not real else [t for t in batch.transactions if t.is_fraud]
        )

    # The holdout family is generated too — it is what the detector is measured on, and
    # make_splits keeps it out of training. More episodes of it, so the out-of-time test window
    # is reliably populated rather than left empty by chance.
    episodes = simulator.n_episodes
    simulator.n_episodes = max(episodes, int(cfg.eval.holdout_episodes))
    batch = simulator.generate(registry.get(held_out).to_attack_params())
    simulator.n_episodes = episodes
    pool.extend(batch.transactions if not real else [t for t in batch.transactions if t.is_fraud])
    return pool


def calibrate(detector, train: list[Transaction], target_fpr: float, embargo_days: float) -> None:
    """Set the action bands on a validation tail of the training data — never on the holdout."""
    from afl.evaluation import protocol

    fit_rows, val_rows = out_of_time_split(train, train_frac=0.8, embargo_days=embargo_days)
    if len(val_rows) < 50 or not any(t.is_fraud for t in fit_rows):
        log.warning("not enough validation rows to calibrate — keeping configured bands")
        return
    detector.fit(fit_rows)
    y, s = protocol.align(val_rows, protocol.score_transactions(detector, val_rows, "calibration"))
    detector.policy.calibrate_to_fpr(s, y, target_fpr=target_fpr)
    detector.fit(train)


def is_pipeline_check(cfg: DictConfig) -> bool:
    """True when the run has no real anchor dataset, so its numbers are not reportable."""
    return bool(cfg.data.get("is_pipeline_check", not cfg.data.get("loader")))


def banner(data_name: str) -> str:
    """The line that stops a synthetic run being quoted as a result."""
    return (
        f"{'=' * 78}\n"
        f"PIPELINE CHECK - NOT A RESULT\n"
        f"data={data_name}: no real anchor dataset, so these numbers verify that the pipeline\n"
        f"runs, nothing more. Reportable numbers need data=paysim or data=amlsim.\n"
        f"{'=' * 78}"
    )


# ── the run ─────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_all_seeds(cfg.seed)
    log.info("\n%s", OmegaConf.to_yaml(cfg))

    pipeline_check = is_pipeline_check(cfg)
    if pipeline_check:
        log.warning("\n%s", banner(str(cfg.data.name)))

    artifact_dir = Path(cfg.artifact_dir) / str(cfg.run_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tracker = get_tracker(str(cfg.tracker), run_name=str(cfg.run_name))
    tracker.log_params(
        {
            "seed": cfg.seed,
            "system": cfg.experiment.system,
            "data": cfg.data.name,
            "held_out_vector": cfg.eval.held_out_vector,
            "rounds": cfg.experiment.rounds,
            "pipeline_check": pipeline_check,
        }
    )

    simulator = build_simulator(cfg)
    detector_factory = build_detector_factory(cfg)
    pool = build_pool(cfg, simulator)
    log.info("pool: %d rows, %d fraud", len(pool), sum(1 for t in pool if t.is_fraud))

    optimiser = AttackOptimiser(
        vector_id=str(cfg.attack.optimiser.vector_id),
        seed=int(cfg.seed),
        lambda_realism=float(cfg.attack.optimiser.lambda_realism),
        backend=str(cfg.attack.optimiser.backend),
    )

    if cfg.experiment.system == "adaptive" or cfg.experiment.compare:
        results = three_system.run_three_systems(
            pool=pool,
            detector_factory=detector_factory,
            simulator=optimiser.bind(simulator),
            optimiser=optimiser,
            held_out_vector=str(cfg.eval.held_out_vector),
            rounds=int(cfg.experiment.rounds),
            smote_ratio=float(cfg.experiment.get("smote_ratio", 1.0)),
            train_frac=float(cfg.eval.train_frac),
            embargo_days=float(cfg.eval.embargo_days),
            seed=int(cfg.seed),
            tracker=tracker,
            real_vectors=tuple(cfg.data.get("known_fraud_vectors") or ()),
        )
    else:
        results = _single_system(cfg, pool, detector_factory)

    for r in results:
        tracker.log(system=r.name, **r.metrics.model_dump(), **r.operational)

    # the banner travels with the artefacts, not just the terminal it scrolled past in
    caveat = (
        f"> **PIPELINE CHECK - NOT A RESULT** (data={cfg.data.name})\n\n" if pipeline_check else ""
    )
    (artifact_dir / "three_system.md").write_text(caveat + three_system.to_markdown(results))
    (artifact_dir / "metrics.json").write_text(
        json.dumps(
            {
                "pipeline_check": pipeline_check,
                "data": str(cfg.data.name),
                "systems": [{**r.row(), **r.operational} for r in results],
            },
            indent=2,
        )
    )
    (artifact_dir / "history.json").write_text(json.dumps(tracker.history, indent=2, default=str))
    (artifact_dir / "attack_trials.json").write_text(json.dumps(optimiser.history(), indent=2))
    (artifact_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    if cfg.eval.sweep_all_vectors:
        matrix = loao.sweep(
            pool,
            detector_factory,
            train_frac=float(cfg.eval.train_frac),
            embargo_days=float(cfg.eval.embargo_days),
            fixed_fpr=float(cfg.eval.fixed_fpr),
            k=int(cfg.eval.k),
        )
        (artifact_dir / "loao_matrix.json").write_text(
            json.dumps({k: v.model_dump() for k, v in matrix.items()}, indent=2)
        )

    if cfg.fidelity.enabled:
        _fidelity(cfg, pool, detector_factory, artifact_dir)

    print("\n" + three_system.to_markdown(results))
    print("\nlift over controls:", three_system.lift(results))
    print(f"\nartefacts -> {artifact_dir}")
    if pipeline_check:
        print("\n" + banner(str(cfg.data.name)))


def _single_system(cfg: DictConfig, pool, detector_factory):
    """Systems A and B on their own, when `experiment.compare` is off."""
    from afl.evaluation.three_system import SystemResult

    evaluator, train = loao.LeaveOneAttackOut.from_pool(
        pool,
        held_out_vector=str(cfg.eval.held_out_vector),
        train_frac=float(cfg.eval.train_frac),
        embargo_days=float(cfg.eval.embargo_days),
        fixed_fpr=float(cfg.eval.fixed_fpr),
        k=int(cfg.eval.k),
    )
    known = set(cfg.data.get("known_fraud_vectors") or ())
    rows = [t for t in train if t.vector_id is None or t.vector_id in known]
    if cfg.experiment.augment == "smote":
        rows = rows + three_system.smote_transactions(
            rows, ratio=float(cfg.experiment.smote_ratio), seed=int(cfg.seed)
        )

    detector = detector_factory()
    detector.fit(rows)
    calibrate(detector, rows, float(cfg.eval.fixed_fpr), float(cfg.eval.embargo_days))
    return [
        SystemResult(
            name=f"{cfg.experiment.system}",
            metrics=evaluator.leave_one_attack_out(detector),
            operational=evaluator.operational(detector),
            n_train=len(rows),
            n_train_fraud=sum(1 for t in rows if t.is_fraud),
        )
    ]


def _fidelity(cfg: DictConfig, pool, detector_factory, artifact_dir: Path) -> None:
    real = [t for t in pool if t.vector_id is None]
    synth = [t for t in pool if t.vector_id is not None]
    real_train, real_test = out_of_time_split(
        real, train_frac=float(cfg.eval.train_frac), embargo_days=float(cfg.eval.embargo_days)
    )
    card = scorecard.build(
        real=real,
        synth=synth,
        real_train=real_train,
        real_test=real_test,
        detector_factory=detector_factory,
        thresholds=scorecard.Thresholds(
            level1_min=float(cfg.fidelity.level1_min),
            level2_min=float(cfg.fidelity.level2_min),
            max_tstr_gap=float(cfg.fidelity.max_tstr_gap),
            min_recall_lift=float(cfg.fidelity.min_recall_lift),
            min_dcr_ratio=float(cfg.fidelity.min_dcr_ratio),
            max_mia_advantage=float(cfg.fidelity.max_mia_advantage),
        ),
        seed=int(cfg.seed),
        meta={"run_name": str(cfg.run_name), "data": str(cfg.data.name)},
    )
    card.save(artifact_dir)
    print(f"\nfidelity verdict: {card.verdict} ({card.score})")
    for reason in card.reasons:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
