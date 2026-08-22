# Adaptive Fraud Simulation Lab

A closed-loop red-team / blue-team system for payment fraud. An attack simulator generates
adaptive fraud, a detector scores it, and the attacks that slip through get fed back, making the
attacker harder and the detector smarter each round. The whole system is then measured on an
attack family it was never trained on.

Built for the Mastercard Innovation Challenge (GFF 2026). The single claim the system exists to
test:

> Adaptive adversarial simulation improves detection recall on a held-out attack family, compared
> with real-only training and ordinary synthetic augmentation, reproducibly, with the mechanism
> shown.

The full design, the nine attack vectors, and the reasoning behind each choice live in the
architecture doc (`docs/architecture.html`). Read that for intent; this README is for running the
thing.

## Where this is right now

This is the skeleton. The pipeline runs end to end, the test suite passes, and every make target
works, but out of the box it runs on a synthetic placeholder config so a fresh clone works with
nothing to download.

The numbers you get from that default are a smoke test, not a result. Real numbers come once we
anchor on a real dataset (PaySim / IEEE-CIS). Anything a synthetic run prints is labelled as a
pipeline check.

## Setup

Needs Python 3.11. On macOS, LightGBM wants libomp. Without it, it quietly falls back to a slower
sklearn path, so install it or your detector isn't the real one.

```bash
brew install libomp            # macOS only
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'        # or: uv sync --extra dev
```

Sanity check before you do anything else:

```bash
make smoke                     # runs the whole loop on dummy data; has to pass
```

## Running it

```bash
make fidelity    # build the fidelity scorecard, before trusting any generator
make loop        # run the adaptive loop (synthetic default, no download)
make compare     # real-only vs SMOTE vs adaptive: the three-system table
make figures     # convergence curve + table, regenerated from run logs
make demo        # FastAPI + Streamlit, via Docker
```

Everything is config-driven with Hydra, so you compose runs with overrides:

```bash
python scripts/run_experiment.py experiment=adaptive data=paysim
```

## How it's laid out

The one rule that makes two teams possible: the red side and the blue side never import each
other. They both import `afl/contract`, and that's the only thing they share.

```
afl/contract     schema + metrics both sides code against; break this carefully
afl/attack       simulator, engines (graph / velocity / drift), vectors, optimiser   [red]
afl/defend       features, models, graded decision, SHAP explanations                [blue]
afl/fidelity     3-level scorecard (statistical / structural / utility) + privacy
afl/loop         where attack meets defend; the closed loop lives here
afl/evaluation   out-of-time split, leave-one-attack-out, three-system table         [blue]
serve            FastAPI + Streamlit demo
config           Hydra configs; experiment/{baseline,smote,adaptive}
scripts          run_experiment, build_fidelity, make_figures
```

## How it's scored

The split is out-of-time, never random, because random splits leak the future.

We use leave-one-attack-out evaluation: train without an attack family, then measure recall on it.
That's the number that matters; everything else is supporting evidence.

We report PR-AUC, recall at a fixed false-positive rate, and precision@k. We do not rely on
accuracy or ROC-AUC alone because, at a sub-2% fraud rate, they can flatter a model that catches
almost nothing.

## Current numbers (honest)

On the synthetic placeholder config, the adaptive system currently lands below the baselines:

```
baseline   PR-AUC 0.97   recall@1%FPR 0.95
SMOTE      PR-AUC 0.96   recall@1%FPR 0.95
adaptive   PR-AUC 0.77   recall@1%FPR 0.49
```

That's the untuned output. Nobody massaged it. It's the weak-side reading the design itself
predicts when the loop searches a single vector against a detector that already generalises to the
holdout.

Fixing it is build work, not skeleton work. The optimiser needs to search across the stronger
vectors on a real dataset. The first pass also caught and fixed a set of leakage bugs, including
velocity windows peeking forward, retraining not accumulating, and the loop training on the
holdout window. Most of the earlier apparent signal was leakage, so these are the real starting
numbers.

## Vectors

There are nine vectors across three engines: graph, velocity, and drift.

The IDs in `config/attack/*` and `vectors.yaml` are still being aligned with the architecture doc,
so treat the doc as the source of truth for what each vector is meant to be.

Aligning them, and adding the vectors that aren't built yet, including bust-out, UPI
collect-request / APP, instant-A2A pass-through, and first-party fraud as the anomaly holdout, is
the first build task.

## What we're not claiming

Fidelity metrics are diagnostics, not proofs. A C2ST score near chance does not mean "realistic,"
and DCR/MIA do not mean "private." They mean we tested for memorisation and membership leakage.

Synthetic-only data lowers exposure but does not automatically mean DPDP compliance.

Frontier vectors are demonstrated capabilities, not necessarily mass-exploited attack patterns,
and we label them that way.

Saying this plainly is deliberate.
