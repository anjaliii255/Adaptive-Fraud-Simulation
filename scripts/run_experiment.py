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
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf

from afl.attack.envelope import AnchorEnvelope
from afl.attack.envelope import audit as envelope_audit
from afl.attack.optimiser import AttackOptimiser
from afl.attack.simulator import Simulator
from afl.attack.templates import registry
from afl.contract.schema import Transaction
from afl.data import loaders
from afl.data.splits import committed_split_for, out_of_time_split
from afl.defend import baseline
from afl.defend.decision import (
    assert_one_operating_point,
    cost_model_for,
    policy_from_config,
)
from afl.defend.features import FeatureBuilder
from afl.defend.models.anomaly import AnomalyDetector, EnsembleDetector
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import leave_one_attack_out as loao
from afl.evaluation import three_system
from afl.fidelity import provenance, scorecard
from afl.tracking import get_tracker
from afl.utils.runcard import stamp, write_run_card
from afl.utils.seed import set_all_seeds

log = logging.getLogger(__name__)

THRESHOLDS_CONFIG = Path("config/fidelity/thresholds.yaml")


# ── assembly ────────────────────────────────────────────────────────────────────
def build_simulator(
    cfg: DictConfig,
    anchor: list[Transaction] | None = None,
    envelope: AnchorEnvelope | None = None,
) -> Simulator:
    """The attack simulator, with its window aligned to the real anchor when there is one.

    Alignment is not cosmetic. `config/attack/engines.yaml` starts the simulation on 2024-01-01;
    PaySim's fixed epoch puts its 743 hourly steps in January 2023. Left alone, every synthetic
    fraud row lands a year after every real row, so the out-of-time split degenerates into
    "real = train, synthetic = test" and the held-out family is separable by timestamp alone.
    An attack has to happen inside the traffic it is hiding in.

    `envelope` short-circuits the measurement for a caller that builds many simulators over the
    same anchor — the leave-one-attack-out matrix rebuilds one per fold, and re-measuring 600k
    rows nine times says the same thing nine times.

    The same argument applies to size. PaySim's median payment is ~67,000 and the actor bundles
    were authored around ~25, so uncalibrated attacks land three orders of magnitude below every
    real row: 99.7% of them below the anchor's 1st percentile. Sorting on amount alone then scores
    PR-AUC 0.80 against that holdout, beating every trained model in the run. So the anchor's
    measured envelope sets both the window and the amount scale.
    """
    from datetime import datetime

    e = cfg.attack.engines
    start = datetime.fromisoformat(str(e.start_ts))
    window = int(e.window_days)
    if anchor and envelope is None:
        envelope = AnchorEnvelope.measure(anchor, str(cfg.data.name))
    if envelope is not None:
        log.info(
            "simulator anchored to %s: %s + %d days, amount median %.0f",
            envelope.dataset,
            envelope.start,
            envelope.window_days,
            math.exp(envelope.amount_log_mu),
        )

    return Simulator(
        seed=cfg.seed,
        n_entities=int(e.n_entities),
        n_background=int(e.n_background),
        start_ts=start,
        window_days=window,
        n_episodes=int(e.n_episodes),
        envelope=envelope,
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


def build_detector_factory(cfg: DictConfig, rows: list[Transaction] | None = None):
    """A fresh detector per system, all of them on one decision policy built from one cost model.

    `rows` is the traffic the policy will decide on, and it is what the cost model's flat costs
    are denominated against — an analyst's time is priced as a share of a typical payment, so it
    has to know what a typical payment is here. See `config/costs/default.yaml`; without it the
    same config declines everything on PaySim and nothing on AMLSim.

    A *fresh* `DecisionPolicy` per detector, not a shared one, because each carries its own
    fitted calibrator now: sharing it would fit System C's score → probability map on System A's
    scores and quietly put the three rows of the hero table on three operating points.
    """
    sup, uns = cfg.defend.supervised, cfg.defend.unsupervised
    dec = OmegaConf.to_container(sup.decision)
    params, params_source = detector_params(cfg)
    costs = cost_model_for(OmegaConf.to_container(cfg.costs), rows)

    def factory():
        policy = policy_from_config(dec, costs)
        model = LGBMDetector(
            policy=policy,
            features=FeatureBuilder(
                stateful=bool(sup.features.stateful),
                windows_s=tuple(int(w) for w in sup.features.windows_s),
            ),
            params=params,
            seed=int(cfg.seed),
            replay_weight=float(sup.replay_weight),
            explain=sup.explain,  # bool accepted for a pre-ticket-09 config; see `_explain_mode`
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


def calibrate(
    detector,
    train: list[Transaction],
    target_fpr: float,
    embargo_days: float,
    fpr_bands: bool = False,
) -> None:
    """Fit the score → probability map on a validation tail of training — never on the holdout.

    The map is fitted from a model trained on the head of the training window and then applied
    to a model refitted on all of it. That is the same compromise the FPR bands always made, and
    it is the price of not spending a slice of the holdout on calibration; the reliability
    numbers in the artefact are what say whether it cost anything.

    `fpr_bands` additionally pins `decline_at` to a target FPR. Threshold mode only — in cost
    mode the bands come from the cost model, and `assert_one_operating_point` refuses the config
    that asks for both.
    """
    from afl.evaluation import protocol

    fit_rows, val_rows = out_of_time_split(train, train_frac=0.8, embargo_days=embargo_days)
    if len(val_rows) < 50 or not any(t.is_fraud for t in fit_rows):
        log.warning("not enough validation rows to calibrate — keeping the derived bands")
        detector.fit(train)
        return
    detector.fit(fit_rows)
    # identity first, or the scores below are already calibrated and the fit stacks two maps
    detector.policy.reset_calibration()
    y, s = protocol.align(val_rows, protocol.score_transactions(detector, val_rows, "calibration"))
    detector.policy.fit_calibrator(s, y)
    if fpr_bands:
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
    dec = cfg.defend.supervised.decision
    configured = dec.get("calibrate_to_fpr")
    mode = str(dec.get("mode", "cost"))
    assert_one_operating_point(configured, target_fpr, mode=mode)
    # The score → probability map is fitted whenever there is a validation tail to fit it on;
    # `calibrate_to_fpr` is the separate, threshold-mode-only question of where to pin the bands.
    fpr_bands = configured is not None
    calibrating = fpr_bands or str(dec.get("calibration", "sigmoid")) != "none"

    def fit(detector, rows: list[Transaction]) -> None:
        if calibrating:
            calibrate(detector, rows, target_fpr, embargo_days, fpr_bands=fpr_bands)
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


def commensurability_audit(cfg: DictConfig, real, pool) -> dict | None:
    """Check the injected attacks are not separable from the anchor by one field alone.

    A held-out family that a single contract field picks out is measuring which generator wrote
    the row, not whether the detector generalises. Three versions of that shipped before this
    check existed, so it runs on every anchored run and says so in the log.
    """
    if not real:
        return None
    held_out = str(cfg.eval.held_out_vector)
    synth = [t for t in pool if t.vector_id == held_out]
    envelope = AnchorEnvelope.measure(real, str(cfg.data.name))
    report = envelope_audit(real, synth)
    report["anchor"] = {
        "supports_behavioural_vectors": envelope.supports_behavioural_vectors,
        "sender_reuse_rate": round(envelope.sender_reuse_rate, 6),
        "time_granularity_s": envelope.time_granularity_s,
        "carries_devices": envelope.carries_devices,
    }

    if not envelope.supports_behavioural_vectors:
        log.warning(
            "%s has a sender reuse rate of %.4f — almost no account transacts twice, so velocity "
            "and drift families have no history to be anomalous against and every src_* feature "
            "is structurally zero. Behavioural numbers on this anchor are not measurable.",
            cfg.data.name,
            envelope.sender_reuse_rate,
        )
    if report["trivially_separable"]:
        log.warning(
            "COMMENSURABILITY: %r separates held-out %s from %s traffic at PR-AUC %.4f against a "
            "base rate of %.4f. That number measures provenance, not generalisation.",
            report["worst"],
            held_out,
            cfg.data.name,
            report["score"],
            report["base_rate"],
        )
    else:
        log.info(
            "commensurability ok: worst single field %r at %.4f", report["worst"], report["score"]
        )
    return report


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
    pool = build_pool(cfg, simulator, real)
    # after the pool, not before: the cost model's flat costs are denominated in the median
    # payment of the traffic being decided on. Real rows when there are any — the injected
    # attacks are a fraction of a percent of an anchored pool and should not move the scale.
    detector_factory = build_detector_factory(cfg, real or pool)
    log.info("pool: %d rows, %d fraud", len(pool), sum(1 for t in pool if t.is_fraud))

    audit = commensurability_audit(cfg, real, pool)
    if audit:
        (artifact_dir / "commensurability.json").write_text(json.dumps(audit, indent=2))

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
                # which code, which seed, which libraries — the artefact answers it itself, so a
                # number that will not reproduce can be diffed against the run that made it
                # rather than argued about. No clock here: see afl/utils/runcard.py.
                "provenance": stamp(int(cfg.seed)),
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
    # config, seed, attack params and metrics in one file, plus the wall-clock facts that are
    # kept out of the artefacts so that two runs of one seed still diff to nothing.
    write_run_card(
        artifact_dir,
        seed=int(cfg.seed),
        config=OmegaConf.to_container(cfg, resolve=True),
        attack_params=optimiser.history(),
        metrics=[r.row() for r in results],
        detector_backend=backends,
        pipeline_check=pipeline_check,
        split_digest=split.digest if split else None,
    )

    if cfg.eval.sweep_all_vectors:
        # Every requested fold gets a row, including the ones that could not be run — a fold
        # that vanishes from the file reads as "not applicable" when it means "we did not look".
        matrix = loao.sweep(
            pool,
            detector_factory,
            train_frac=float(cfg.eval.train_frac),
            embargo_days=float(cfg.eval.embargo_days),
            fixed_fpr=float(cfg.eval.fixed_fpr),
            k=int(cfg.eval.k),
            split=split,
            min_positives=int(cfg.eval.min_meaningful_positives),
            fit=build_fit(cfg),
        )
        (artifact_dir / "loao_matrix.json").write_text(
            json.dumps([f.to_dict() for f in matrix], indent=2, default=str)
        )
        for fold in matrix:
            log.info("loao %s", fold.summary())

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
    """The loop's own scorecard, judged against the same bars `make fidelity` uses.

    The thresholds are read from the file rather than from six floats in the composed config, so
    the run also carries the record of when they were set. `hydra` has already moved the working
    directory to the run dir by the time this is called, hence the original cwd — without it the
    provenance check reads "no git history" and says so, which is honest but useless.
    """
    real = [t for t in pool if t.vector_id is None]
    synth = [t for t in pool if t.vector_id is not None]
    if split is not None:
        real_train, real_test = split.apply(real)
    else:
        real_train, real_test = out_of_time_split(
            real, train_frac=float(cfg.eval.train_frac), embargo_days=float(cfg.eval.embargo_days)
        )
    try:
        root = Path(hydra.utils.get_original_cwd())
    except ValueError:  # not under a hydra run
        root = Path.cwd()
    values, _why, prov = provenance.load(root / THRESHOLDS_CONFIG, repo=root)
    card = scorecard.build(
        real=real,
        synth=synth,
        real_train=real_train,
        real_test=real_test,
        detector_factory=detector_factory,
        thresholds=scorecard.Thresholds.from_values(values),
        seed=int(cfg.seed),
        meta={"run_name": str(cfg.run_name), "data": str(cfg.data.name)},
        fixed_fpr=float(cfg.eval.fixed_fpr),
        k=int(cfg.eval.k),
        min_positives=int(cfg.eval.min_meaningful_positives),
        provenance=prov.to_dict(),
    )
    card.save(artifact_dir)
    print(f"\nfidelity verdict: {card.verdict} ({card.score})")
    for reason in card.reasons:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
