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

On the synthetic placeholder config, held out on M3, the adaptive system lands below both
baselines:

```
baseline   PR-AUC 0.736   recall@1%FPR 0.546
SMOTE      PR-AUC 0.748   recall@1%FPR 0.600
adaptive   PR-AUC 0.239   recall@1%FPR 0.167
```

Produced on the sklearn HistGradientBoosting fallback, not LightGBM, because libomp was missing on
the machine that ran it. Install libomp before quoting any of it.

These are lower than the pre-freeze numbers, and nothing regressed to cause that. Freezing the
taxonomy changed which families get generated and changed what the holdout *is* — M3 went from
gradual behavioural drift to first-party fraud, which is a harder family on purpose. The old figures
are not comparable and were replaced rather than kept alongside.

That's the untuned output. Nobody massaged it. It's the weak-side reading the design itself
predicts when the loop searches a single vector against a detector that already generalises to the
holdout.

Fixing it is build work, not skeleton work. The optimiser needs to search across the stronger
vectors on a real dataset. The first pass also caught and fixed a set of leakage bugs, including
velocity windows peeking forward, retraining not accumulating, and the loop training on the
holdout window. Most of the earlier apparent signal was leakage, so these are the real starting
numbers.

## Vectors

Nine vectors, three engines. The ids match the architecture doc and are frozen; see
`docs/adr/0001-vector-taxonomy-and-holdout.md` for why each one is where it is, and
`afl/attack/templates/vectors.yaml` for the definitions. Adding a vector is a YAML edit.

| id | vector | engine | level | tier | status |
| --- | --- | --- | --- | --- | --- |
| S1 | Mule network & layering | graph | mechanism | strong | built |
| S2 | Card testing / BIN enumeration | velocity | mechanism | strong | built |
| S3 | Account takeover via drift | drift | mechanism | strong | built |
| C1 | Bust-out | drift | mechanism | common | template |
| C2 | UPI collect-request / APP scam | velocity | enabler | common | planned |
| C3 | Instant-A2A pass-through | graph | mechanism | common | template |
| M1 | Boundary probing / paced evasion | velocity | model-attack | mid | template |
| M2 | Synthetic-identity lifecycle | drift | enabler | mid | planned |
| M3 | First-party / friendly fraud | drift | mechanism | mid | template · **holdout** |

`level` is the taxonomy level and never gets flattened: mechanisms are the fraud, enablers are what
make it possible, and M1 is an attack against our own model. `tier` is the role in the build: the
adaptive loop wraps the strong three, the common three are must-catch load, the mid three are
novelty and the holdout.

`status` is the honest part — what the code can generate today, as opposed to what the taxonomy
declares:

- **built** — the engines express the vector's defining behaviour.
- **template** — valid traffic of roughly the right shape, but the defining tell is missing. Fine as
  training load and haystack; not reportable as a recall figure for that family. Each carries a
  `gap` naming what is missing and the ticket that fixes it.
- **planned** — cannot be generated. `Simulator.generate` raises and names the ticket, because a
  family that silently emits nothing looks exactly like a family the detector caught.

So three of the nine are done and six still owe work, and the file says which and why. **M3 is
the leave-one-attack-out holdout** because `user == fraudster` breaks the legit-vs-attacker assumption every supervised
feature rests on — and it is `template` today, so the headline number is currently measured on a
proxy for first-party fraud rather than the real thing.

## What we're not claiming

Fidelity metrics are diagnostics, not proofs. A C2ST score near chance does not mean "realistic,"
and DCR/MIA do not mean "private." They mean we tested for memorisation and membership leakage.

Synthetic-only data lowers exposure but does not automatically mean DPDP compliance.

Frontier vectors are demonstrated capabilities, not necessarily mass-exploited attack patterns,
and we label them that way.

Saying this plainly is deliberate.
