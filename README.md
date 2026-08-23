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

The **feature table is now built for that data** rather than for the synthetic placeholder —
56 causal features, each with a rationale, measured per anchor in `docs/features.md`. The
detector on top of it is still the skeleton's, so real-anchor numbers are a *first reading*
rather than a result: ticket 08 is what turns them into one.

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
make features    # build the feature table over every anchor; record cost and coverage
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

## Features

56 columns, all computed from events strictly *before* the row they belong to, none of them
derived from the label. `docs/features.md` is the dictionary: every column, one line on why it
exists, and how much of it each anchor actually fills in — generated from the code and the files
on disk by `make features`, so it cannot drift from the table it describes.

```bash
make features    # rebuild the dictionary and the per-anchor cost + coverage artefacts
```

Two things about it are worth knowing before you read a number that came out of it.

**Direction is the design.** Every entity has two histories — what it sent and what it received —
and the blocks that matter are the ones crossing them. Fan-out (`src_out_uniq_dst_*`) is card
testing and mule spraying; fan-in (`dst_in_uniq_src_*`) is the collector; money arriving and
leaving inside the hour (`src_seconds_since_last_in`, `src_passthrough_ratio_3600s`) is
pass-through, and it is invisible if the two directions are added together. The previous version
of this module kept one history per entity and did add them together.

**A third of the table is structurally empty on PaySim, and that is a property of the anchor,
not a bug.** `nameOrig` is effectively unique per row, so there is no sender to have a history:
17 of 56 columns never take a second value there, against 8 on AMLSim and 1 on synthetic traffic.
The feature dictionary marks each one **dead** per anchor rather than letting it read as a
feature the model has. The beneficiary block is the one that carries signal on both real anchors,
and `tests/test_features.py` asserts on the real files that it does.

Causality is proved, not asserted: the tests check the property directly (appending later traffic
never changes an earlier row's features) and cross-check all 56 columns against a brute-force
reference that shares no code with the implementation.

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
scripts          run_experiment, build_splits, build_features, build_fidelity, make_figures
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
paysim   PR-AUC 0.006   recall@1%FPR 0.040   precision@100 0.00      (was 0.025 / 0.243 / 0.00)
amlsim   PR-AUC 0.040   recall@1%FPR 0.160   precision@100 0.24      (was 0.007 / 0.067 / 0.08)
```

```bash
python scripts/run_experiment.py data=paysim experiment=baseline run_name=paysim_baseline
python scripts/run_experiment.py data=amlsim experiment=baseline run_name=amlsim_baseline
```

The bracketed figures are the same command before ticket 07's feature table landed. AMLSim went
up and PaySim went down, and the second one is the more informative of the two:

- **This fold cannot carry a claim on a real anchor, and finding that out is the useful part.**
  Every positive in the M3 holdout is an injected synthetic row and every negative is a real one,
  so the number partly measures how far the injected family sits from the real distribution
  rather than how well the detector finds first-party fraud. The committed fidelity scorecards
  say how far that is: on PaySim, KS 0.86 on log-amount and 0.89 on the inter-transaction gap,
  and a TSTR ratio of 0.03. A classifier told to sort real rows from injected M3 rows on either
  feature table does it at AUC 1.00. Neither 0.025 nor 0.006 is evidence about detection.
  **Ticket 11** is where the fold has to say this itself; **ticket 15** is where the generator
  closes the distance.
- **On each anchor's own labelled fraud — same haystack, same labels — the features do move the
  number.** With the model, the seed and the committed boundary held fixed, the table went from
  35 columns to 56: AMLSim PR-AUC 0.83 → 0.95, recall@1%FPR 0.93 → 0.97, precision@100 0.98 →
  1.00; PaySim PR-AUC 0.14 → 0.13 with precision@100 0.38 → 0.47. That split is exactly what the
  anchors are: AMLSim has real sender *and* beneficiary histories for the new directional and
  graph features to read, and PaySim has almost none — 17 of the 56 columns never take a second
  value there. AMLSim is itself a simulator with a deliberately distinctive fan-in / cycle
  topology, so read its near-perfect column as "graph features find graph fraud", not as a
  production number. See `docs/features.md` and the ticket 07 carry-out.
- **It is not LightGBM.** libomp was missing, so both runs fell back to sklearn
  HistGradientBoosting. The run artefact records which backend produced each number.
- **`precision@100 = 0.00` on PaySim is the honest headline.** The calibrated bands apply
  friction to 16% of holdout traffic, and the top 100 ranked rows still contain no M3 fraud at
  all. Blanket friction is not detection, and the ranking is what ticket 08 has to fix.

### On the synthetic placeholder — pipeline check, not comparable

```
A_baseline   PR-AUC 0.574   recall@1%FPR 0.255   precision@100 0.63
B_smote      PR-AUC 0.574   recall@1%FPR 0.255   precision@100 0.63
C_adaptive   PR-AUC 0.187   recall@1%FPR 0.000   precision@100 0.04
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
