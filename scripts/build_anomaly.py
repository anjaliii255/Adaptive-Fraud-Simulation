"""Score the anomaly layer against the supervised model on the family neither of them trained on.

    python scripts/build_anomaly.py                  # every real anchor
    python scripts/build_anomaly.py paysim           # just one
    python scripts/build_anomaly.py --sample 0.02    # a quick pass
    python scripts/build_anomaly.py --doc-only       # rewrite the doc from committed artefacts

Ticket 10's deliverable, and it is a comparison rather than a model. The supervised detector can
only catch what it has labels for, and the leave-one-attack-out fold is defined by holding one
family's labels back — so on that fold the headline detector is being asked a question it was
never given the answer to. An outlier score fitted on legit traffic alone is not: it has no
notion of "the fraud I have seen", only of "what normal looked like". The bet was that it
therefore degrades more gracefully and sits underneath the ensemble as a floor. On both anchors
it does not — see `docs/anomaly.md`, which this file generates, and which reports whichever way
the comparison falls.

Five systems, one fold, one operating point:

**supervised** — the committed tuned detector from `make baseline`, refitted with M3 carved out.
**anomaly** — an isolation forest over the same causal features, fitted on the *legit rows of the
training window only*. Not one fraud row, and `AnomalyDetector.fit` has no argument that would
let one in.
**anomaly (contaminated)** — the same detector fitted on the fraud rows too. A control, not a
configuration: "fit on legit only" is a design claim, and a design claim with nothing measured
against it is a preference.
**ensemble** — the blend the loop actually runs. The weight is swept end to end on the same
scores, so the shipped 0.7 is answerable to a curve rather than to a comment.
**amount_only** — rank by amount, no model, no training, direction chosen on the training window.
Two earlier results were walked back for want of this column: the transfer test and the BankSim
spike both produced a detector that lost to it.

The fold's own honesty checks travel with the numbers: whether one contract field alone separates
the injected family from the anchor (if it does, the table measures provenance, not detection),
how many positives the fold actually carries, and whether the same transaction scores the same
value in two different batches — which it did not before this ticket, because the outlier score
was min-maxed over whatever it happened to be scored with.

Everything lands in `artifacts/anomaly/<anchor>.json` and in `docs/anomaly.md`, which is
generated from those files and never hand-typed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import yaml
from omegaconf import OmegaConf

# One pool definition, not two. The comparison in this file is only meaningful if the fold is the
# same fold the loop runs on, so the simulator and the pool come from `run_experiment` itself
# rather than from a re-implementation here that would drift the first time either was edited.
from run_experiment import build_pool, build_simulator, detector_params  # noqa: E402

from afl.attack import envelope as envelope_lib
from afl.contract.metrics import DetectorScore
from afl.contract.schema import AttackBatch, AttackParams, Transaction
from afl.data import loaders
from afl.data.splits import (
    CommittedSplit,
    assert_no_leakage,
    committed_split_for,
    out_of_time_split,
)
from afl.defend import baseline, calibration, explain
from afl.defend.decision import (
    CostModel,
    action_mix,
    assert_one_operating_point,
    cost_model_for,
    policy_from_config,
)
from afl.defend.features import FeatureBuilder
from afl.defend.models.anomaly import AnomalyDetector, EnsembleDetector, contaminated_control
from afl.defend.models.lgbm import LGBMDetector
from afl.evaluation import leave_one_attack_out as loao
from afl.evaluation import protocol
from afl.utils.seed import set_all_seeds

log = logging.getLogger("build_anomaly")

DATA_DIR = Path("config/data")
LGBM_CONFIG = Path("config/defend/lgbm.yaml")
ANOMALY_CONFIG = Path("config/defend/anomaly.yaml")
EVAL_CONFIG = Path("config/eval/leave_one_attack_out.yaml")
ENGINES_CONFIG = Path("config/attack/engines.yaml")
COSTS_CONFIG = Path("config/costs/default.yaml")
DETECTOR_DIR = Path("artifacts/detector")
ARTIFACT_DIR = Path("artifacts/anomaly")
DOC_PATH = Path("docs/anomaly.md")

ANOMALY_ARTEFACT_VERSION = 1

#: The blend is swept end to end, including both endpoints, so the two halves appear in the same
#: curve as the blends of them. 0.0 is the anomaly layer alone and 1.0 is the supervised model
#: alone, and a curve whose best point is an endpoint is an ensemble that has not earned its row.
BLEND_WEIGHTS = tuple(round(0.1 * i, 1) for i in range(11))

#: Below this many positives the fold is reported as too thin to carry a claim rather than as a
#: low score. Read from the harness rather than restated here: ticket 11 owns the rule, and two
#: files holding the same threshold is two files that can disagree about what a thin fold is.
MIN_MEANINGFUL_POSITIVES = loao.MIN_MEANINGFUL_POSITIVES

#: A PR-AUC gap smaller than this is not called a change in the generated write-up. Two numbers
#: that both round to 1.000 produced the sentence "degrades from 1.000 to 1.000" before it
#: existed, which is what a threshold-free comparison does at the top of a metric's range.
MATERIAL_GAP = 0.01

#: How many holdout rows the batch-independence check re-scores in shuffled chunks. The property
#: is per-row, so a sample settles it; the whole holdout would just cost more to say the same.
BATCH_CHECK_ROWS = 4_000
BATCH_CHECK_CHUNKS = 8


def anchors(selected: list[str]) -> list[dict]:
    """Every data config that names a loader — the real anchors, not the synthetic default."""
    out = []
    for path in sorted(DATA_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("loader") and (not selected or cfg["name"] in selected):
            out.append(cfg)
    return out


def stats(rows: list[Transaction]) -> dict:
    fraud = sum(1 for t in rows if t.is_fraud)
    return {
        "rows": len(rows),
        "fraud": fraud,
        "base_rate": round(fraud / len(rows), 8) if rows else 0.0,
        "first_ts": min((t.ts for t in rows), default=None),
        "last_ts": max((t.ts for t in rows), default=None),
    }


def as_batch(rows: list[Transaction], run_id: str = "anomaly") -> AttackBatch:
    return AttackBatch(
        run_id=run_id,
        params=AttackParams(vector_id="eval", engine="none"),
        transactions=list(rows),
        seed=0,
    )


def compose(cfg: dict, sup: dict, uns: dict, eval_cfg: dict, engines: dict, costs: dict, seed: int):
    """The same config tree `config/config.yaml` composes, built here from the same files.

    `run_experiment.build_pool` and `build_simulator` take the composed config, and this script
    calls them rather than re-deriving the fold, so there is one definition of "the M3 fold".
    """
    return OmegaConf.create(
        {
            "seed": seed,
            "data": cfg,
            "attack": {"engines": engines},
            "defend": {"supervised": sup, "unsupervised": uns},
            "eval": eval_cfg,
            "costs": costs,
        }
    )


# ── the systems ─────────────────────────────────────────────────────────────────
def measured(
    rows: list[Transaction],
    probabilities: np.ndarray,
    scores: list[DetectorScore],
    fixed_fpr: float,
    k: int,
    held_out: str,
) -> dict:
    """The three ranking metrics and the four operational rates, from one set of scores."""
    y = np.array([int(t.is_fraud) for t in rows], dtype=int)
    result = protocol.evaluate(y, probabilities, fixed_fpr=fixed_fpr, k=k, held_out_vector=held_out)
    return {
        **result.model_dump(),
        **{key: round(v, 6) for key, v in protocol.operational_rates(rows, scores).items()},
        **{f"action_{a}": v for a, v in action_mix(scores).items()},
    }


def calibrate_on(policy, val_rows: list[Transaction], val_scores: np.ndarray) -> dict:
    """Fit the score → probability map on the training window's validation tail, and say how well.

    The anomaly layer's *model* sees no labels — that is the whole point of it. Its score → cost
    map does, and has to: a cost model compares `p x amount` against a flat analyst cost, so an
    uncalibrated score makes the arithmetic right and the inputs meaningless (ticket 09 measured
    that at 99.3% friction). The labels used here are the training window's own, M3 carved out,
    on the same tail every other detector calibrates on. The holdout is never touched.
    """
    y = np.array([int(t.is_fraud) for t in val_rows], dtype=int)
    policy.reset_calibration()
    before = {
        "brier": round(calibration.brier(val_scores, y), 8),
        "expected_calibration_error": round(
            calibration.expected_calibration_error(val_scores, y), 8
        ),
    }
    policy.fit_calibrator(val_scores, y)
    after_p = np.array([policy.probability(float(s)) for s in val_scores], dtype=float)
    return {
        "n_val_rows": len(val_rows),
        "n_val_positives": int(y.sum()),
        "before": before,
        "after": {
            "brier": round(calibration.brier(after_p, y), 8),
            "expected_calibration_error": round(
                calibration.expected_calibration_error(after_p, y), 8
            ),
        },
        "calibrator": policy.calibrator.to_dict(),
    }


def amount_only(train: list[Transaction], holdout: list[Transaction], fixed_fpr: float, k: int):
    """The floor: rank by amount alone. Direction chosen on the training window, never on test.

    Two tickets were walked back for want of this column — the transfer test and the BankSim
    spike both produced a detector that lost to it. An anomaly layer that beats the supervised
    model on M3 and loses to sorting by amount has not found anything.
    """
    y_tr = np.array([int(t.is_fraud) for t in train], dtype=int)
    a_tr = np.array([t.amount for t in train], dtype=float)
    high_first = protocol.pr_auc(y_tr, a_tr) >= protocol.pr_auc(y_tr, -a_tr)
    a_te = np.array([t.amount for t in holdout], dtype=float)
    y_te = np.array([int(t.is_fraud) for t in holdout], dtype=int)
    signal = a_te if high_first else -a_te
    result = protocol.evaluate(y_te, signal, fixed_fpr=fixed_fpr, k=k)
    return {
        **result.model_dump(exclude={"held_out_vector"}),
        "direction": "largest amount first" if high_first else "smallest amount first",
        "direction_chosen_on": "train",
    }


def batch_independence(detector: AnomalyDetector, rows: list[Transaction], seed: int) -> dict:
    """Does a row score the same value in two different batches? The map used not to.

    Two separable questions, measured separately, because only one of them is a bug.

    **The score map.** `predict_proba` min-maxed the raw scores of whatever batch it was handed,
    so `DetectorScore.score` was a statement about a batch rather than about a transaction, and
    the ensemble blended that batch-relative number against a probability. Measured here with the
    design matrix computed **once** and only the mapping repeated, so the number below is the
    map's own contribution and nothing else. The ranking metrics never noticed a within-batch
    min-max — it is monotone and PR-AUC only reads order — which is exactly why this is measured
    rather than reviewed.

    **The feature builder.** `FeatureBuilder.transform(update=False)` lets a row see the rows
    before it in the same call, deliberately: by the time the second payment of a burst is
    scored in production, the first one has happened. So the full path is *not* batch-invariant,
    and it is not supposed to be. It is reported here because it is the residual left after the
    map is fixed, and because it applies identically to the supervised detector — it is a
    property of the feature contract, not of this layer.
    """
    rng = np.random.default_rng(seed)
    sample = list(rows[: min(len(rows), BATCH_CHECK_ROWS)])
    if len(sample) < BATCH_CHECK_CHUNKS or detector.model is None:
        return {"checked": False, "reason": "not enough holdout rows to split into batches"}
    order = rng.permutation(len(sample))
    pieces = np.array_split(order, BATCH_CHECK_CHUNKS)

    def min_max(raw: np.ndarray) -> np.ndarray:
        lo, hi = float(raw.min()), float(raw.max())
        return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

    # the map alone: one design matrix, one set of raw scores, two ways of mapping them
    scaled = detector.scaled(sample)
    raw = detector.raw_scores(scaled=scaled)
    mapped_whole, legacy_whole = detector.score_map.apply(raw), min_max(raw)
    mapped_chunked = np.zeros(len(sample), dtype=float)
    legacy_chunked = np.zeros(len(sample), dtype=float)
    for piece in pieces:
        mapped_chunked[piece] = detector.score_map.apply(raw[piece])
        legacy_chunked[piece] = min_max(raw[piece])

    # the whole path, features included, which is batch-dependent on purpose
    path_chunked = np.zeros(len(sample), dtype=float)
    for piece in pieces:
        path_chunked[piece] = detector.predict_proba([sample[i] for i in piece])

    return {
        "checked": True,
        "n_rows": len(sample),
        "n_batches": BATCH_CHECK_CHUNKS,
        "score_map": {
            "max_drift": round(float(np.max(np.abs(mapped_whole - mapped_chunked))), 12),
            "max_drift_under_the_old_min_max": round(
                float(np.max(np.abs(legacy_whole - legacy_chunked))), 6
            ),
            "mean_drift_under_the_old_min_max": round(
                float(np.mean(np.abs(legacy_whole - legacy_chunked))), 6
            ),
        },
        "whole_path": {
            "max_drift": round(float(np.max(np.abs(mapped_whole - path_chunked))), 6),
            "mean_drift": round(float(np.mean(np.abs(mapped_whole - path_chunked))), 6),
            "why": (
                "FeatureBuilder.transform(update=False) lets a row see earlier rows in the same "
                "call, on purpose and identically for the supervised detector"
            ),
        },
        "raw_score_range_on_the_holdout": [round(float(raw.min()), 6), round(float(raw.max()), 6)],
    }


def blend_sweep(
    holdout: list[Transaction], p_sup: np.ndarray, p_uns: np.ndarray, fixed_fpr: float, k: int
) -> list[dict]:
    """Every blend weight scored on the same two probability vectors.

    One scoring pass, eleven rows: the blend is arithmetic over the halves, so measuring the
    curve costs nothing beyond having scored each half once. There is no excuse for shipping a
    weight nobody plotted.
    """
    y = np.array([int(t.is_fraud) for t in holdout], dtype=int)
    out = []
    for w in BLEND_WEIGHTS:
        blended = w * p_sup + (1 - w) * p_uns
        result = protocol.evaluate(y, blended, fixed_fpr=fixed_fpr, k=k)
        out.append(
            {
                "weight": w,
                "pr_auc": result.pr_auc,
                "recall_at_fixed_fpr": result.recall_at_fixed_fpr,
                "precision_at_k": result.precision_at_k,
            }
        )
    return out


def run_anchor(cfg: dict, args, sup: dict, uns: dict, eval_cfg: dict, engines: dict, costs_cfg):
    """One anchor's M3 fold, five systems, one operating point."""
    name = cfg["name"]
    seed = int(args.seed)
    set_all_seeds(seed)
    held_out = str(eval_cfg["held_out_vector"])
    fixed_fpr, k = float(eval_cfg["fixed_fpr"]), int(eval_cfg["k"])
    assert_one_operating_point(
        sup["decision"].get("calibrate_to_fpr"), fixed_fpr, mode=str(sup["decision"]["mode"])
    )

    if args.sample is not None:
        cfg = {**cfg, "sample": {**(cfg.get("sample") or {}), "sample_fraction": args.sample}}
    real = loaders.load_from_config(cfg)
    composed = compose(cfg, sup, uns, eval_cfg, engines, costs_cfg, seed)
    costs: CostModel = cost_model_for(costs_cfg, real)

    split: CommittedSplit | None = committed_split_for(cfg)
    if split is None:
        raise SystemExit(f"{name}: no committed split — run `make splits` first")

    simulator = build_simulator(composed, anchor=real)
    pool = build_pool(composed, simulator, real)
    train, holdout = loao.make_splits(pool, held_out, split=split)
    assert_no_leakage(train, holdout)
    leaked = [t for t in train if t.vector_id == held_out]
    if leaked:  # pragma: no cover - make_splits carves it out; this is the guard, not the path
        raise SystemExit(f"{name}: {len(leaked)} {held_out} rows reached training")

    n_positives = sum(1 for t in holdout if t.is_fraud)
    log.info(
        "%s: pool %d rows -> train %d (%d fraud, no %s) / holdout %d (%d %s positives)",
        name,
        len(pool),
        len(train),
        sum(1 for t in train if t.is_fraud),
        held_out,
        len(holdout),
        n_positives,
        held_out,
    )
    if not n_positives:
        raise SystemExit(
            f"{name}: the holdout carries no {held_out} rows — every metric would read 0.0 "
            "without having measured anything. Widen the window or raise eval.holdout_episodes."
        )

    fit_rows, val_rows = out_of_time_split(
        train,
        train_frac=float(sup["tuning"]["fit_frac"]),
        embargo_days=float(eval_cfg["embargo_days"]),
    )
    params, params_source = detector_params(composed)

    def features() -> FeatureBuilder:
        return FeatureBuilder(
            stateful=bool(sup["features"]["stateful"]),
            windows_s=tuple(int(w) for w in sup["features"]["windows_s"]),
        )

    systems: dict[str, dict] = {}
    cards: dict[str, dict] = {}
    calibrations: dict[str, dict] = {}
    timings: dict[str, dict] = {}

    # ── supervised ──────────────────────────────────────────────────────────────
    started = time.perf_counter()
    set_all_seeds(seed)
    supervised = LGBMDetector(
        policy=policy_from_config(sup["decision"], costs),
        features=features(),
        params=params,
        seed=seed,
        replay_weight=float(sup["replay_weight"]),
        params_source=params_source,
    )
    supervised.fit(fit_rows)
    _, s_val = protocol.align(val_rows, protocol.score_transactions(supervised, val_rows, "cal"))
    calibrations["supervised"] = calibrate_on(supervised.policy, val_rows, s_val)
    supervised.fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    sup_scores = supervised.score(as_batch(holdout, "supervised"))
    _, p_sup = protocol.align(holdout, sup_scores)
    systems["supervised"] = measured(holdout, p_sup, sup_scores, fixed_fpr, k, held_out)
    timings["supervised"] = {
        "fit_seconds": round(fit_seconds, 2),
        "score_seconds": round(time.perf_counter() - started, 2),
    }
    cards["supervised"] = supervised.model_card()

    # ── anomaly, on legit rows only ─────────────────────────────────────────────
    started = time.perf_counter()
    set_all_seeds(seed)
    anomaly = AnomalyDetector(
        kind=str(uns["kind"]),
        contamination=float(uns["contamination"]),
        policy=policy_from_config(sup["decision"], costs),
        features=features(),
        seed=seed,
    )
    anomaly.fit(fit_rows)
    _, a_val = protocol.align(val_rows, protocol.score_transactions(anomaly, val_rows, "cal"))
    calibrations["anomaly"] = calibrate_on(anomaly.policy, val_rows, a_val)
    anomaly.fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    anom_scores = anomaly.score(as_batch(holdout, "anomaly"))
    _, p_uns = protocol.align(holdout, anom_scores)
    systems["anomaly"] = measured(holdout, p_uns, anom_scores, fixed_fpr, k, held_out)
    timings["anomaly"] = {
        "fit_seconds": round(fit_seconds, 2),
        "score_seconds": round(time.perf_counter() - started, 2),
    }
    cards["anomaly"] = anomaly.model_card()
    explain.assert_flagged_rows_are_explained(anom_scores)

    # ── the control: the same layer, fitted on the fraud rows too ───────────────
    set_all_seeds(seed)
    contaminated = contaminated_control(
        train,
        AnomalyDetector(
            kind=str(uns["kind"]),
            contamination=float(uns["contamination"]),
            policy=policy_from_config(sup["decision"], costs),
            features=features(),
            seed=seed,
        ),
    )
    # calibrated on its own scores, like every other system here: sharing the legit-only
    # detector's map would make the control differ in two things at once
    _, c_val = protocol.align(val_rows, protocol.score_transactions(contaminated, val_rows, "cal"))
    calibrations["anomaly_contaminated"] = calibrate_on(contaminated.policy, val_rows, c_val)
    cont_scores = contaminated.score(as_batch(holdout, "contaminated"))
    _, p_cont = protocol.align(holdout, cont_scores)
    systems["anomaly_contaminated"] = measured(holdout, p_cont, cont_scores, fixed_fpr, k, held_out)
    cards["anomaly_contaminated"] = contaminated.model_card()

    # ── the ensemble the loop runs ──────────────────────────────────────────────
    weight = float(uns["ensemble"]["weight"])
    ensemble = EnsembleDetector(
        supervised=supervised,
        unsupervised=anomaly,
        weight=weight,
        policy=policy_from_config(sup["decision"], costs),
    )
    calibrations["ensemble"] = calibrate_on(
        ensemble.policy, val_rows, weight * s_val + (1 - weight) * a_val
    )
    ens_scores = ensemble.score(as_batch(holdout, "ensemble"))
    _, p_ens = protocol.align(holdout, ens_scores)
    # The blend is arithmetic over the two halves, and this is the assertion that says so: the
    # sweep below reuses `p_sup` and `p_uns` rather than re-scoring, and that reuse is only valid
    # if the seam agrees with the arithmetic.
    drift = float(np.max(np.abs(p_ens - (weight * p_sup + (1 - weight) * p_uns))))
    if drift > 1e-9:  # pragma: no cover - a guard on the reuse below, not a path
        raise SystemExit(f"{name}: EnsembleDetector.score disagrees with the blend by {drift:.3g}")
    systems[f"ensemble@{weight}"] = measured(holdout, p_ens, ens_scores, fixed_fpr, k, held_out)
    cards["ensemble"] = ensemble.model_card()
    explain.assert_flagged_rows_are_explained(ens_scores)

    systems["amount_only"] = amount_only(train, holdout, fixed_fpr, k)

    for label, block in systems.items():
        log.info("%s [%s]: %s", name, label, json.dumps(block, default=str))

    reference = None
    reference_path = DETECTOR_DIR / f"{name}.json"
    if reference_path.exists():
        card = json.loads(reference_path.read_text())
        tuned = card["metrics"]["tuned"]
        reference = {
            "source": str(reference_path),
            "measured_on": "this anchor's OWN labelled fraud, out of time at the same boundary",
            "pr_auc": tuned["pr_auc"],
            "recall_at_fixed_fpr": tuned["recall_at_fixed_fpr"],
            "precision_at_k": tuned["precision_at_k"],
            "n_positives": tuned["n_positives"],
        }

    return {
        "version": ANOMALY_ARTEFACT_VERSION,
        "dataset": name,
        "seed": seed,
        "held_out_vector": held_out,
        "operating_point": {
            "fixed_fpr": fixed_fpr,
            "k": k,
            "source": str(EVAL_CONFIG),
            "band_units": "calibrated probability",
        },
        "fold": {
            "config": f"config/data/{name}.yaml",
            "sample_fraction": (cfg.get("sample") or {}).get("sample_fraction"),
            "split": split.to_dict(),
            "real": stats(real),
            "pool": stats(pool),
            "train": stats(train),
            "holdout": stats(holdout),
            "calibration_fit": stats(fit_rows),
            "calibration_validation": stats(val_rows),
            "n_holdout_positives": n_positives,
            "enough_positives_to_report": n_positives >= MIN_MEANINGFUL_POSITIVES,
            "min_meaningful_positives": MIN_MEANINGFUL_POSITIVES,
        },
        "commensurability": envelope_lib.audit(real, [t for t in pool if t.vector_id == held_out]),
        "systems": systems,
        "blend_sweep": blend_sweep(holdout, p_sup, p_uns, fixed_fpr, k),
        "shipped_blend_weight": weight,
        "batch_independence": batch_independence(anomaly, holdout, seed),
        "calibration": calibrations,
        "timings": timings,
        "model_cards": cards,
        "supervised_reference": reference,
        "meta": {
            "generated_by": "scripts/build_anomaly.py",
            "note": (
                "Every positive in this fold is an injected synthetic M3 row and every negative "
                "is a real one, so recall here reports the distance between two distributions as "
                "well as detection. Read `commensurability` before reading `systems`."
            ),
        },
    }


# ── the document ────────────────────────────────────────────────────────────────
def _pct(x: float) -> str:
    """One decimal, except near the ends, where rounding would erase the point being made."""
    if x <= 0.0:
        return "0%"
    if 0 < x < 0.0005:
        return f"{x * 100:.4f}%"
    if x >= 0.9995:
        return f"{x * 100:.4f}%"
    return f"{x * 100:.1f}%"


def _findings(card: dict) -> list[str]:
    """The sentences this anchor's fold actually supports, ranked before they are written.

    Generated rather than typed, so the write-up cannot drift from the artefact it describes —
    and so a re-run that reverses the result reverses the prose with it.
    """
    sys_ = card["systems"]
    sup, anom, floor = sys_["supervised"], sys_["anomaly"], sys_["amount_only"]
    control = sys_["anomaly_contaminated"]
    weight = card["shipped_blend_weight"]
    ens = sys_[f"ensemble@{weight}"]
    best = max(card["blend_sweep"], key=lambda r: r["pr_auc"])
    ref = card.get("supervised_reference")

    if not card["fold"]["enough_positives_to_report"]:
        return [
            f"**The fold is too thin to carry a claim.** {card['fold']['n_holdout_positives']} "
            f"positives against a floor of {card['fold']['min_meaningful_positives']}: every "
            "number below is reported as missing, not as a low score."
        ]
    if card["commensurability"]["trivially_separable"]:
        return [
            f"**This fold measures provenance, not detection.** "
            f"`{card['commensurability']['worst']}` alone separates the injected "
            f"{card['held_out_vector']} rows from the anchor at PR-AUC "
            f"{card['commensurability']['score']:.3f}. Every row below inherits that, and no "
            "comparison between them means anything until the generator is fixed."
        ]

    out: list[str] = []
    # 1. did the supervised model collapse on the family it has no labels for?
    if ref and abs(sup["pr_auc"] - ref["pr_auc"]) < MATERIAL_GAP:
        out.append(
            f"**The supervised model neither collapses nor improves on "
            f"{card['held_out_vector']}.** PR-AUC {sup['pr_auc']:.3f} on the held-out family "
            f"against {ref['pr_auc']:.3f} on this anchor's own labelled fraud in the same test "
            f"window (`{ref['source']}`) — inside the {MATERIAL_GAP} either way that this file "
            "treats as a difference."
        )
    elif ref and sup["pr_auc"] > ref["pr_auc"]:
        out.append(
            f"**The supervised model does not collapse on {card['held_out_vector']} — it does "
            f"*better* there than on this anchor's own fraud.** PR-AUC {sup['pr_auc']:.3f} and "
            f"recall@FPR {sup['recall_at_fixed_fpr']:.3f} on the held-out family, against "
            f"{ref['pr_auc']:.3f} / {ref['recall_at_fixed_fpr']:.3f} on the real labelled fraud "
            f"in the same test window (`{ref['source']}`). That is a finding about the injected "
            "rows rather than about generalisation: a family the detector has never seen should "
            "not be easier than one it trains on, and this fold flatters whatever is measured "
            "on it."
        )
    elif ref:
        out.append(
            f"**The supervised model degrades on the family it has no labels for**, from PR-AUC "
            f"{ref['pr_auc']:.3f} on this anchor's own fraud to {sup['pr_auc']:.3f} on the "
            f"held-out family at the same boundary and the same operating point."
        )

    if ref and ref["pr_auc"] >= 0.99:
        out.append(
            f"Read that reference number with the caveat `docs/detector.md` attaches to it: a "
            f"supervised PR-AUC of {ref['pr_auc']:.3f} on {card['dataset']} is a property of the "
            "generator that wrote the anchor, not evidence about production fraud. Everything "
            "compared against it here inherits the caveat."
        )

    # 2. the layer this ticket is about
    if anom["pr_auc"] > sup["pr_auc"]:
        out.append(
            f"**The anomaly layer beats the supervised model on the held-out family** — PR-AUC "
            f"{anom['pr_auc']:.3f} against {sup['pr_auc']:.3f}, recall@FPR "
            f"{anom['recall_at_fixed_fpr']:.3f} against {sup['recall_at_fixed_fpr']:.3f}. The "
            "promotion this ticket held open is earned here."
        )
    else:
        margin = f"{sup['pr_auc'] / anom['pr_auc']:.0f}x" if anom["pr_auc"] else "outright"
        out.append(
            f"**The anomaly layer is not the floor it was designed to be.** PR-AUC "
            f"{anom['pr_auc']:.3f} against the supervised model's {sup['pr_auc']:.3f} — beaten "
            f"{margin} on the one fold that exists to favour it. The promotion this ticket held "
            "open is not earned on this anchor."
        )
    if anom["pr_auc"] <= floor["pr_auc"]:
        out.append(
            f"**And it loses to the amount floor** — {floor['pr_auc']:.3f} for sorting by amount "
            f"({floor['direction']}, chosen on the training window) against the layer's "
            f"{anom['pr_auc']:.3f}. A detector that a sort beats has not found anything."
        )
    else:
        out.append(
            f"It does clear the amount floor ({anom['pr_auc']:.3f} against {floor['pr_auc']:.3f}, "
            f"{floor['direction']}), which is the least a trained layer owes."
        )

    # 3. the blend, which is the one thing that is measured rather than assumed
    endpoint = best["weight"] in (0.0, 1.0)
    if endpoint and best["weight"] == 1.0:
        out.append(
            f"**The blend does not earn its row on this anchor.** The curve rises monotonically "
            f"to w=1.0 — the supervised model alone, PR-AUC {best['pr_auc']:.3f} — and the "
            f"shipped w={weight} costs {best['pr_auc'] - ens['pr_auc']:.3f} of it "
            f"({ens['pr_auc']:.3f})."
        )
    elif endpoint:
        out.append(
            f"**The blend does not earn its row on this anchor.** The best point on the curve is "
            f"w={best['weight']} — the anomaly layer alone, PR-AUC {best['pr_auc']:.3f}."
        )
    else:
        out.append(
            f"**The blend does earn its row here**, and the curve has an interior optimum to "
            f"prove it: w={best['weight']} reaches PR-AUC {best['pr_auc']:.3f} against "
            f"{sys_['supervised']['pr_auc']:.3f} for the supervised model alone and "
            f"{anom['pr_auc']:.3f} for the anomaly layer alone. The shipped w={weight} takes "
            f"{ens['pr_auc'] - sup['pr_auc']:.3f} of the {best['pr_auc'] - sup['pr_auc']:.3f} "
            "available — reported, not adopted, because a weight chosen on the fold it is quoted "
            "from is the tuning-on-test the baseline forbids."
        )

    # 4. the control on the ticket's own design constraint
    direction = "higher" if control["pr_auc"] > anom["pr_auc"] else "lower"
    out.append(
        f"**Fitting on the fraud rows too scores {direction}, not lower** — PR-AUC "
        f"{control['pr_auc']:.3f} against the legit-only {anom['pr_auc']:.3f}, with "
        f"{_pct(control['evasion_rate'])} of the held-out family allowed through against "
        f"{_pct(anom['evasion_rate'])}. The legit-only rule is not paying for itself on this "
        "fold, and it is kept anyway: the training fraud here is other *known* families, which "
        "is not the case the rule exists for."
    )
    return out


def _headline(cards: dict[str, dict]) -> str:
    """One paragraph across every anchor, so the result is not buried in the per-anchor tables."""
    measurable = [c for c in cards.values() if c["fold"]["enough_positives_to_report"]]
    if not measurable:
        return "No anchor carried enough held-out positives to report.\n"
    wins = [
        c["dataset"]
        for c in measurable
        if c["systems"]["anomaly"]["pr_auc"] > c["systems"]["supervised"]["pr_auc"]
    ]
    blend_wins = []
    for c in measurable:
        best = max(c["blend_sweep"], key=lambda r: r["pr_auc"])
        if best["weight"] not in (0.0, 1.0):
            blend_wins.append(c["dataset"])
    names = ", ".join(sorted(c["dataset"] for c in measurable))
    if wins:
        verdict = (
            f"**The anomaly layer is promoted on {', '.join(sorted(wins))}**, where it beats the "
            "supervised model on the held-out family — the outcome this ticket held open."
        )
    else:
        flattered = sorted(
            c["dataset"]
            for c in measurable
            if c.get("supervised_reference")
            and c["systems"]["supervised"]["pr_auc"]
            > c["supervised_reference"]["pr_auc"] + MATERIAL_GAP
        )
        aside = (
            f" On {', '.join(flattered)} it scores *higher* on the held-out family than on the "
            "anchor's own labelled fraud, which is a finding about the injected rows rather than "
            "about the detector."
            if flattered
            else ""
        )
        verdict = (
            f"**The premise did not hold on any anchor measured.** This layer was built on two "
            f"claims: that the supervised model collapses on a family it has no labels for, and "
            f"that an outlier score fitted on legit traffic degrades more gracefully. Measured "
            f"on {names}, neither survives. The supervised model does not collapse.{aside} The "
            f"anomaly layer sits far below it rather than under it."
        )
    blend = (
        f" The blend is where the layer earns its keep instead: on {', '.join(sorted(blend_wins))} "
        f"the weight curve has an interior optimum, so the two halves together beat either alone."
        if blend_wins
        else " And the blend curve peaks at an endpoint on every anchor, so it earns nothing."
    )
    return verdict + blend + "\n"


def anomaly_doc(cards: dict[str, dict], missing: list[str]) -> str:
    """`docs/anomaly.md`, generated from the committed artefacts and never hand-typed."""
    out: list[str] = ["# The anomaly layer on the held-out family\n"]
    out.append(
        "_Generated by `scripts/build_anomaly.py` from the anchors on disk. Every number below "
        "traces to `artifacts/anomaly/<anchor>.json`, which carries the split digest, the seed, "
        "the model cards and the commensurability audit that produced it. Do not edit this file "
        "— re-run `make anomaly`._\n"
    )
    out.append(
        "Ticket 10. The supervised detector is trained without one attack family and then asked "
        "about it; the anomaly layer is fitted on legit traffic alone and asked the same "
        "question. The comparison is the deliverable, whichever way it falls.\n"
    )
    out.append("## The result\n")
    out.append(_headline(cards))
    out.append(
        "**Read the two audit lines before each table.** Every positive in these folds is an "
        "injected synthetic row and every negative is a real one, so a fold where one contract "
        "field separates the two is measuring which generator wrote the row rather than whether "
        "anything detects fraud. And a fold with a handful of positives moves several points of "
        "recall per row.\n"
    )

    for name in sorted(cards):
        card = cards[name]
        fold, sys_ = card["fold"], card["systems"]
        weight = card["shipped_blend_weight"]
        out.append(f"## {name}\n")
        for finding in _findings(card):
            out.append(f"{finding}\n")

        audit = card["commensurability"]
        out.append(
            f"- **Commensurability** — worst single contract field `{audit['worst']}` at PR-AUC "
            f"{audit['score']:.4f} against a base rate of {audit['base_rate']:.4f}: "
            f"{'TRIVIALLY SEPARABLE' if audit['trivially_separable'] else 'ok'}.\n"
            f"- **Fold size** — {fold['holdout']['rows']:,} holdout rows carrying "
            f"{fold['n_holdout_positives']} {card['held_out_vector']} positives, against "
            f"{fold['train']['rows']:,} training rows with {fold['train']['fraud']:,} fraud and "
            f"not one {card['held_out_vector']} row. Split digest `{fold['split']['digest']}`.\n"
        )

        header = (
            f"| system | PR-AUC | rec@{card['operating_point']['fixed_fpr']:.0%}FPR | "
            f"p@{card['operating_point']['k']} | evasion | friction | trained on |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        rows = []
        trained_on = {
            "supervised": f"{fold['train']['fraud']:,} labelled fraud rows, no "
            f"{card['held_out_vector']}",
            "anomaly": f"{card['model_cards']['anomaly']['training']['n_rows']:,} legit rows, "
            f"{card['model_cards']['anomaly']['training']['n_fraud_excluded']:,} fraud excluded",
            "anomaly_contaminated": "the same rows **including** the fraud — a control",
            f"ensemble@{weight}": f"both, blended {weight:.2f} supervised",
            "amount_only": "nothing",
        }
        for label in (
            "supervised",
            "anomaly",
            "anomaly_contaminated",
            f"ensemble@{weight}",
            "amount_only",
        ):
            block = sys_[label]
            evasion, friction = block.get("evasion_rate"), block.get("friction_rate")
            rows.append(
                f"| {label} | {block['pr_auc']:.3f} | {block['recall_at_fixed_fpr']:.3f} | "
                f"{block['precision_at_k']:.2f} | "
                f"{_pct(evasion) if evasion is not None else '—'} | "
                f"{_pct(friction) if friction is not None else '—'} | "
                f"{trained_on[label]} |\n"
            )
        out.append(header + "".join(rows))
        out.append(
            "`evasion` is the share of the held-out family the policy let through untouched and "
            "`friction` the share of legit traffic that carried any action; the floor row has "
            "neither because it has no policy. Both are decision-layer numbers, so they move "
            "when the cost model does and the three ranking columns do not — see "
            "`docs/decisions.md`.\n"
        )

        out.append("### The blend, measured\n")
        out.append(
            "`w` is the supervised share. w=1.0 is the supervised model alone and w=0.0 the "
            "anomaly layer alone, so both halves sit in the same curve as the blends of them. "
            "One scoring pass produces every row: the blend is arithmetic over the two halves, "
            "and a test holds the seam to that arithmetic exactly.\n"
        )
        sweep = ["| w | PR-AUC | rec@FPR | p@k |\n| ---: | ---: | ---: | ---: |\n"]
        for row in card["blend_sweep"]:
            marker = " ←shipped" if abs(row["weight"] - weight) < 1e-9 else ""
            sweep.append(
                f"| {row['weight']:.1f}{marker} | {row['pr_auc']:.3f} | "
                f"{row['recall_at_fixed_fpr']:.3f} | {row['precision_at_k']:.2f} |\n"
            )
        out.append("".join(sweep))

        cal = card["calibration"]["anomaly"]
        out.append("### The score → probability map\n")
        out.append(
            f"The anomaly layer's *model* sees no labels. Its cost map does, and has to — a cost "
            f"model compares `p x amount` against a flat analyst cost, so an uncalibrated score "
            f"makes the arithmetic right and the inputs meaningless. Fitted on the training "
            f"window's validation tail ({cal['n_val_rows']:,} rows, {cal['n_val_positives']} "
            f"positives), the same tail every other detector uses, never the holdout. Expected "
            f"calibration error {cal['before']['expected_calibration_error']:.5f} → "
            f"{cal['after']['expected_calibration_error']:.5f}. It is not cosmetic: an isolation "
            f"forest's raw score sits in "
            f"{card['batch_independence'].get('raw_score_range_on_the_holdout', ['?', '?'])}, "
            "so a cost band placed at a genuine probability catches every row until the map has "
            "been fitted.\n"
        )

        bi = card["batch_independence"]
        if bi.get("checked"):
            m, w = bi["score_map"], bi["whole_path"]
            out.append("### One row, one score\n")
            out.append(
                f"The same {bi['n_rows']:,} holdout rows scored whole and again in "
                f"{bi['n_batches']} shuffled batches.\n"
            )
            out.append(
                f"- **The score map**, with the features held fixed so only the mapping repeats: "
                f"maximum drift **{m['max_drift']:.1e}**. Under the min-max normalisation this "
                f"ticket removed, the same rows move by up to "
                f"**{m['max_drift_under_the_old_min_max']:.3f}** (mean "
                f"{m['mean_drift_under_the_old_min_max']:.3f}) — a score that was a statement "
                f"about a batch rather than about a transaction.\n"
                f"- **The whole path**, features included: maximum drift **{w['max_drift']:.3f}** "
                f"(mean {w['mean_drift']:.3f}). That residual is the feature contract, not this "
                f"layer — `transform(update=False)` lets a row see earlier rows in the same call "
                f"on purpose, and does so identically for the supervised detector.\n"
            )

    if missing:
        out.append("## Not measured\n")
        out.append(
            f"{', '.join(sorted(missing))} — no raw file on disk, so the fold was skipped rather "
            "than reported as a zero. See the data card for how to fetch it.\n"
        )
    return "\n".join(out)


# ── the CLI ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("anchors", nargs="*", help="anchor names; default every real anchor")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--sample", type=float, default=None, help="override sample_fraction for a quick pass"
    )
    parser.add_argument(
        "--doc-only", action="store_true", help="rewrite docs/anomaly.md from committed artefacts"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    sup = yaml.safe_load(LGBM_CONFIG.read_text())
    uns = yaml.safe_load(ANOMALY_CONFIG.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    engines = yaml.safe_load(ENGINES_CONFIG.read_text())
    costs_cfg = yaml.safe_load(COSTS_CONFIG.read_text())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cards: dict[str, dict] = {}
    missing: list[str] = []

    if args.doc_only:
        for path in sorted(ARTIFACT_DIR.glob("*.json")):
            cards[path.stem] = json.loads(path.read_text())
    else:
        for cfg in anchors(args.anchors):
            name = cfg["name"]
            try:
                card = run_anchor(cfg, args, sup, uns, eval_cfg, engines, costs_cfg)
            except FileNotFoundError as e:
                # the same rule `make baseline` and `make decisions` follow: an anchor that is
                # not downloaded is named as skipped, never reported as a zero
                log.warning("%s: skipped — %s", name, e)
                missing.append(name)
                continue
            baseline.assert_no_forbidden_metrics(card["systems"])
            (ARTIFACT_DIR / f"{name}.json").write_text(json.dumps(card, indent=2, default=str))
            cards[name] = card
        # anchors this run did not touch, but whose artefact is committed, still belong in the doc
        for path in sorted(ARTIFACT_DIR.glob("*.json")):
            cards.setdefault(path.stem, json.loads(path.read_text()))

    if not cards:
        log.error("no anchors measured and no committed artefacts — nothing to write")
        return 1
    DOC_PATH.write_text(anomaly_doc(cards, missing))
    print(f"\nartefacts -> {ARTIFACT_DIR}/  doc -> {DOC_PATH}")
    print(f"\n{_headline(cards)}")
    for name in sorted(cards):
        for finding in _findings(cards[name]):
            print(f"  {name}: {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
