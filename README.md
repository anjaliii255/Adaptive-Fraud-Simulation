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

The pipeline runs end to end on **real data** — PaySim and the IBM AMLSim dump both load through
the contract and drive the whole run on a config override alone. Out of the box it still runs on
a synthetic placeholder so a fresh clone works with nothing to download, and anything a synthetic
run prints is stamped as a pipeline check.

The detector itself is still the skeleton's. Real-anchor numbers exist but they are a *first
reading*, not a result: tickets 07 and 08 are what turn them into one.

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

## Data

Two real anchors, neither committed (`data/**` is gitignored) and neither fetched for you:

| | PaySim | AMLSim (IBM example dump) |
| --- | --- | --- |
| what it anchors | behaviour: velocity, drift, the account-drain arc | the mule graph: layering topology and typology labels |
| rows | 6,362,620 | 1,323,234 |
| fraud | 8,213 (**0.129%**) | 1,719 (**0.130%**) |
| span | 743 hourly steps (~31 days) | 200 daily steps |
| put it in | `data/raw/` | `data/raw/IBMAml/` |

Both sit **~37x below** the synthetic default's measured 4.74% fraud rate. That is more than an
order of magnitude, so no operating point carries between the two regimes and their numbers never
share a table. `scripts/build_splits.py` measures the gap on every run rather than quoting it.

```bash
make splits      # compute + commit the out-of-time boundary and the data cards
```

The split boundary is **computed once and committed**, not re-derived per run: a fraction splits
at 70% of whatever rows it was handed, so the partition moves whenever the pool composition does
and two runs stop being comparable with nothing in the diff to show it. `artifacts/splits/*.json`
holds two timestamps and a digest; every run reads them.

The data cards in `docs/data-cards/` carry the rest — licence, base rate, the embargo and its
rationale, the measured integrity checks, the quirks and the limits. Two quirks are worth knowing
before you write a feature:

- **PaySim has no sender history.** `nameOrig` is effectively unique per row (6,353,307 distinct
  origins over 6,362,620 rows). Every `src`-side velocity and RFM feature is structurally empty on
  that anchor; `nameDest` is the only entity with a past, which is also why the run-time sample is
  taken over beneficiaries.
- **The out-of-time cut lands on two different base rates.** PaySim fraud is 3.5x denser in the
  test half than the train half, because the label is spread evenly across steps and the legit
  volume is not.

Full PaySim is ~7.7 GB once it is contract rows, so the default reads a deterministic 10%
hash-sample of beneficiaries (636,409 rows, base rate within 2.2% relative of the full file).
`data.sample.sample_fraction=1.0` reads the lot.

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
scripts          run_experiment, build_splits, build_fidelity, make_figures
```

## How it's scored

The split is out-of-time, never random, because random splits leak the future.

We use leave-one-attack-out evaluation: train without an attack family, then measure recall on it.
That's the number that matters; everything else is supporting evidence.

We report PR-AUC, recall at a fixed false-positive rate, and precision@k. We do not rely on
accuracy or ROC-AUC alone because, at a sub-2% fraud rate, they can flatter a model that catches
almost nothing.

## Current numbers (honest)

Two regimes, and they do not go in the same table. The real anchors run at ~0.13% fraud and the
synthetic default at 4.74%; `recall@1%FPR` and `precision@100` are not the same measurement at
those two rates.

### On the real anchors — first reading, not a result

Held out on M3, System A only, at the committed split:

```
paysim   PR-AUC 0.025   recall@1%FPR 0.243   precision@100 0.00
amlsim   PR-AUC 0.007   recall@1%FPR 0.067   precision@100 0.08
```

```bash
python scripts/run_experiment.py data=paysim experiment=baseline run_name=paysim_baseline
```

These are weak and they are supposed to be read as weak. Three reasons, none of them mysterious:

- **The features have not been built for this data yet.** Ticket 07 is unstarted, and on PaySim
  every sender-side feature is structurally empty (see Data, above). The model is working from
  roughly half the feature table it thinks it has.
- **It is not LightGBM.** libomp was missing, so both runs fell back to sklearn
  HistGradientBoosting. The run artefact records which backend produced each number.
- **`precision@100 = 0.00` on PaySim is the honest headline, not `caught_rate = 1.0`.** The
  calibrated bands apply friction to 14% of holdout traffic, which technically leaves nothing
  ALLOWed — but the top 100 ranked rows contain no M3 fraud at all. Blanket friction is not
  detection, and the ranking is what ticket 08 has to fix.

### On the synthetic placeholder — pipeline check, not comparable

```
A_baseline   PR-AUC 0.508   recall@1%FPR 0.233
B_smote      PR-AUC 0.499   recall@1%FPR 0.233
C_adaptive   PR-AUC 0.126   recall@1%FPR 0.051
```

The adaptive system lands below both controls. That is the weak-side reading the design predicts
when the loop searches a single vector against a detector that already generalises to the holdout,
and widening the search is ticket 12's job.

M3 is no longer a proxy — it is genuine first-party fraud, where no device changes, no new operator
appears and no new beneficiary is ever paid, so none of the signals a supervised model leans on
fire at all. A harder holdout is the point of the holdout.

Nobody massaged any of this. The first pass also caught and fixed a set of leakage bugs — velocity
windows peeking forward, retraining not accumulating, the loop training on the holdout window — so
most of the earlier apparent signal was leakage and these are the real starting numbers. Each
regime supersedes the last rather than sitting beside it; see `docs/adr/0002-dataset-anchors.md`.

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
| M3 | First-party / friendly fraud | drift | mechanism | mid | built · **holdout** |

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

So four of the nine are done and five still owe work, and the file says which and why. **M3 is
the leave-one-attack-out holdout** because `user == fraudster` breaks the legit-vs-attacker
assumption every supervised feature rests on, and it is `built`: the abuse runs on the owner's own
device, to beneficiaries the account already pays, elevated only against that account's own history.
A one-line amount rule at 1% FPR catches 3% of it, which is the point.

## What we're not claiming

Fidelity metrics are diagnostics, not proofs. A C2ST score near chance does not mean "realistic,"
and DCR/MIA do not mean "private." They mean we tested for memorisation and membership leakage.

Synthetic-only data lowers exposure but does not automatically mean DPDP compliance.

Frontier vectors are demonstrated capabilities, not necessarily mass-exploited attack patterns,
and we label them that way.

Saying this plainly is deliberate.
