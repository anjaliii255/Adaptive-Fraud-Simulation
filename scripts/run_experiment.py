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
from afl.data.splits import committed_split_for, out_of_time_split
from afl.defend import baseline
from afl.defend.decision import DecisionPolicy, assert_one_operating_point
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
def build_simulator(cfg: DictConfig, anchor: list[Transaction] | None = None) -> Simulator:
    """The attack simulator, with its window aligned to the real anchor when there is one.

    Alignment is not cosmetic. `config/attack/engines.yaml` starts the simulation on 2024-01-01;
    PaySim's fixed epoch puts its 743 hourly steps in January 2023. Left alone, every synthetic
    fraud row lands a year after every real row, so the out-of-time split degenerates into
    "real = train, synthetic = test" and the held-out family is separable by timestamp alone.
    An attack has to happen inside the traffic it is hiding in.
    """
    from datetime import datetime

    e = cfg.attack.engines
    start = datetime.fromisoformat(str(e.start_ts))
    window = int(e.window_days)
    if anchor:
        start = min(t.ts for t in anchor)
        window = max(1, int((max(t.ts for t in anchor) - start).total_seconds() // 86_400))
        log.info("simulator window aligned to the anchor: %s + %d days", start, window)

    return Simulator(
        seed=cfg.seed,
        n_entities=int(e.n_entities),
        n_background=int(e.n_background),
        start_ts=start,
        window_days=window,
        n_episodes=int(e.n_episodes),
    )


def detector_params(cfg: DictConfig) -> tuple[dict, str]:
    """The supervised params for this run, and where they came from.

    `config/defend/lgbm.yaml` holds the *inputs* — the starting point and the search envelope.
    `artifacts/detector/<anchor>.json` holds the *decision*, the params `make baseline` landed
    on for that anchor, committed next to the numbers they produced. Same division as the
    committed split: a config that carried the tuned params would drift from the run that
    justified them the first time either was edited alone.
    """
    base = OmegaConf.to_container(cfg.defend.supervised.params) or {}
    if not bool(cfg.defend.supervised.get("use_committed_params", True)):
        return base, "config (use_committed_params=false)"
    tuned, source = baseline.tuned_params(str(cfg.data.name))
    if not tuned:
        log.info("no committed baseline for %s — using the configured params", cfg.data.name)
        return base, source
    log.info("using the committed tuned params for %s from %s", cfg.data.name, source)
    return {**base, **tuned}, source


def build_detector_factory(cfg: DictConfig):
    sup, uns = cfg.defend.supervised, cfg.defend.unsupervised
    dec = sup.decision
    params, params_source = detector_params(cfg)

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
            params=params,
            seed=int(cfg.seed),
            replay_weight=float(sup.replay_weight),
            explain=bool(sup.explain),
            params_source=params_source,
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


def load_anchor(cfg: DictConfig) -> list[Transaction]:
    """The real rows, straight from the data config. `[]` on the synthetic default."""
    return loaders.load_from_config(OmegaConf.to_container(cfg.data))


def build_pool(
    cfg: DictConfig, simulator: Simulator, real: list[Transaction] | None = None
) -> list[Transaction]:
    """Real traffic plus one attack batch per allowed vector — the input to every split."""
    held_out = str(cfg.eval.held_out_vector)
    allowed = [v for v in cfg.attack.engines.vectors]
    if held_out in allowed:
        raise ValueError(
            f"{held_out!r} is both the eval holdout and a generated vector — "
            "the red side would be handing the blue side the answer"
        )

    real = list(real or [])
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
        detector.fit(train)
        return
    detector.fit(fit_rows)
    y, s = protocol.align(val_rows, protocol.score_transactions(detector, val_rows, "calibration"))
    detector.policy.calibrate_to_fpr(s, y, target_fpr=target_fpr)
    detector.fit(train)


def build_fit(cfg: DictConfig):
    """How every system in this run is trained — one function, applied to all of them.

    The operating point is not only `fixed_fpr` and `k` in the metric; it is also where the
    action bands sit, because `evasion_rate` in the same table is a function of them. Before
    this was a hook, `experiment=baseline` calibrated and `make compare` did not, so System A
    appeared in two tables at two operating points under one name.
    """
    target_fpr = float(cfg.eval.fixed_fpr)
    embargo_days = float(cfg.eval.embargo_days)
    configured = cfg.defend.supervised.decision.get("calibrate_to_fpr")
    assert_one_operating_point(configured, target_fpr)
    enabled = configured is not None

    def fit(detector, rows: list[Transaction]) -> None:
        if enabled:
            calibrate(detector, rows, target_fpr, embargo_days)
        else:
            detector.fit(rows)

    return fit


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

    real = load_anchor(cfg)
    simulator = build_simulator(cfg, anchor=real)
    detector_factory = build_detector_factory(cfg)
    pool = build_pool(cfg, simulator, real)
    log.info("pool: %d rows, %d fraud", len(pool), sum(1 for t in pool if t.is_fraud))

    # The boundary is read, never re-derived. Absent (synthetic default) the fraction is used
    # and the run is a pipeline check anyway.
    split = committed_split_for(OmegaConf.to_container(cfg.data))
    if split:
        log.info(
            "committed split %s: train <= %s, test >= %s (embargo %s, digest %s)",
            split.dataset,
            split.train_end,
            split.test_start,
            split.embargo,
            split.digest,
        )
        tracker.log_params({"split_digest": split.digest, "base_rate": loaders.base_rate(real)})

    optimiser = AttackOptimiser(
        vector_id=str(cfg.attack.optimiser.vector_id),
        seed=int(cfg.seed),
        lambda_realism=float(cfg.attack.optimiser.lambda_realism),
        backend=str(cfg.attack.optimiser.backend),
    )

    fit_detector = build_fit(cfg)
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
            split=split,
            fixed_fpr=float(cfg.eval.fixed_fpr),
            k=int(cfg.eval.k),
            fit_detector=fit_detector,
        )
    else:
        results = _single_system(cfg, pool, detector_factory, split, fit_detector)

    backends = sorted({r.backend for r in results if r.backend})
    log.info("detector backend: %s", ", ".join(backends) or "unknown")
    tracker.log_params({"detector_backend": ", ".join(backends)})
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
                "n_real_rows": len(real),
                "real_base_rate": loaders.base_rate(real),
                "split": split.to_dict() if split else None,
                "operating_point": {"fixed_fpr": float(cfg.eval.fixed_fpr), "k": int(cfg.eval.k)},
                # which library produced these numbers, per system. On macOS the LightGBM wheel
                # imports and then fails to load libomp, so "LightGBM" is a claim about the run
                # that only the run can settle.
                "backend": backends,
                "systems": [
                    {**r.row(), **r.operational, "model_card": r.model_card} for r in results
                ],
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
            split=split,
        )
        (artifact_dir / "loao_matrix.json").write_text(
            json.dumps({k: v.model_dump() for k, v in matrix.items()}, indent=2)
        )

    if cfg.fidelity.enabled:
        _fidelity(cfg, pool, detector_factory, artifact_dir, split)

    print("\n" + three_system.to_markdown(results))
    print("\nlift over controls:", three_system.lift(results))
    print(f"\nartefacts -> {artifact_dir}")
    if pipeline_check:
        print("\n" + banner(str(cfg.data.name)))


def _single_system(cfg: DictConfig, pool, detector_factory, split=None, fit_detector=None):
    """Systems A and B on their own, when `experiment.compare` is off."""
    evaluator, train = loao.LeaveOneAttackOut.from_pool(
        pool,
        held_out_vector=str(cfg.eval.held_out_vector),
        train_frac=float(cfg.eval.train_frac),
        embargo_days=float(cfg.eval.embargo_days),
        split=split,
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
    (fit_detector or build_fit(cfg))(detector, rows)
    return [three_system.measure(str(cfg.experiment.system), detector, evaluator, rows)]


def _fidelity(cfg: DictConfig, pool, detector_factory, artifact_dir: Path, split=None) -> None:
    real = [t for t in pool if t.vector_id is None]
    synth = [t for t in pool if t.vector_id is not None]
    if split is not None:
        real_train, real_test = split.apply(real)
    else:
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
        fixed_fpr=float(cfg.eval.fixed_fpr),
        k=int(cfg.eval.k),
    )
    card.save(artifact_dir)
    print(f"\nfidelity verdict: {card.verdict} ({card.score})")
    for reason in card.reasons:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
