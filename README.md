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
56 causal features, each with a rationale, measured per anchor in `docs/features.md`.

The **detector on top of it is now tuned, and it is genuinely LightGBM** — which it had not been
before, because libomp was missing and the wheel was silently falling back to sklearn. Its
reference numbers are committed per anchor in `artifacts/detector/` and written up in
`docs/detector.md`. That is System A of the hero table: the bar everything else has to clear.

## Setup

Needs Python 3.11. **On macOS, install libomp first.** The LightGBM wheel imports cleanly without
it and then fails to load its own shared library at fit time, so the code falls back to sklearn
HistGradientBoosting and keeps running — a different model under a table headed "LightGBM". That
is not hypothetical: every number in this README before ticket 08 came out of the fallback.
Every run now records which backend produced it, so you can check rather than assume.

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
make baseline    # tune the detector on every anchor; commit the reference numbers
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

## The detector

Gradient-boosted trees over those 56 features, tuned per anchor and committed. `docs/detector.md`
is the write-up; `artifacts/detector/<anchor>.json` is the evidence, carrying the params, the
backend and version, the split digest and the seed that produced each number.

```bash
make baseline    # retune from scratch and rewrite both
```

The same division as the split boundary: **config holds the inputs** — the starting params and
the search envelope, in `config/defend/lgbm.yaml` — and **the artefact holds the decision**, the
params 40 Optuna trials landed on. A run on `data=paysim` picks the committed params up on its
own, so a config that carried them could never drift from the run that justified them.

The search only ever sees a validation tail *inside* the training window, and
`afl/defend/tuning.py` raises rather than warns if that tail is not strictly after the rows it
fitted on. The action bands are calibrated on the same tail. An operating point chosen on the
window it is reported from is not an operating point, it is a result.

Two things worth knowing before quoting a number from it.

**A baseline is only "strong" relative to how hard the anchor is,** so every artefact carries an
`amount_only` floor: rank the rows by amount, no model, no features, no training, direction
chosen on train. On PaySim the floor reaches PR-AUC 0.057 against the detector's 0.152. On AMLSim
it reaches **0.456, with precision@100 of 1.00** — because every alerted row in that file is a
sub-20 amount against legit traffic reaching 21.5M, so 78% of the negatives are excluded before
anything is fitted. **AMLSim's near-perfect column is the generator being legible, not the
detector being good.** PaySim is the anchor to read.

**Tuning was not a formality.** On PaySim, same features, same seed, same boundary: PR-AUC
0.060 → 0.152, recall@1%FPR 0.371 → 0.478, precision@100 0.14 → 0.48. Both sides are committed,
so the claim that the search earned its keep is checkable rather than asserted.

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
scripts          run_experiment, build_splits, build_features, build_baseline, build_fidelity,
                 make_figures
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
those two rates. All of these came out of LightGBM 4.5.0, and every artefact says so.

### The detector's own reference — on each anchor's real, labelled fraud

Out of time at the committed boundary, tuned on a validation tail inside the training window.
This is the bar; `docs/detector.md` and `artifacts/detector/` are the full record.

```
                 PR-AUC   recall@1%FPR   precision@100
paysim   tuned    0.152      0.478           0.48
         stock    0.060      0.371           0.14     <- same detector, untuned params
         floor    0.057      0.212           0.23     <- amount alone, no model at all

amlsim   tuned    1.000      1.000           1.00
         stock    0.305      0.697           0.65
         floor    0.456      0.474           1.00     <- read this row before the one above it
```

```bash
make baseline
```

- **Read PaySim, not AMLSim.** Every alerted row in the AMLSim dump is a sub-20 amount against
  legit traffic reaching 21.5M, so sorting on amount alone — no model, no features, no training —
  already fills the entire top-100 queue. AMLSim's perfect column is the generator being legible.
  PaySim's fraud spans the whole amount range, so its detector has to earn every point.
- **Tuning was not a formality.** PaySim PR-AUC 0.060 → 0.152 and precision@100 0.14 → 0.48, with
  the features, the seed and the boundary held fixed. Anything ever compared against the untuned
  detector was compared against a straw man.
- **The validation tail is thin** — 46 fraud rows on PaySim, 196 on AMLSim. `n_val_positives` sits
  next to the score in the artefact because a search maximised against 46 positives has variance
  nobody should read past.

### On the M3 leave-one-attack-out fold — still not a claim

Held out on M3, System A only, at the committed split:

```
paysim   PR-AUC 0.007   recall@1%FPR 0.043   precision@100 0.00     (was 0.006 / 0.040 / 0.00)
amlsim   PR-AUC 0.167   recall@1%FPR 0.173   precision@100 0.25     (was 0.040 / 0.160 / 0.24)
```

```bash
python scripts/run_experiment.py data=paysim experiment=baseline run_name=paysim_baseline
python scripts/run_experiment.py data=amlsim experiment=baseline run_name=amlsim_baseline
```

The bracketed figures are the same command before this ticket: untuned params, and the
sklearn fallback rather than LightGBM, because libomp was missing.

- **A 2.5x better detector moved PaySim's fold by nothing, and that is the finding.** Every
  positive in the M3 holdout is an injected synthetic row and every negative is a real one, so
  the number measures how far the injected family sits from the real distribution as much as it
  measures detection. The committed fidelity scorecards say how far: on PaySim, KS 0.86 on
  log-amount, 0.89 on the inter-transaction gap, TSTR ratio 0.03, and a classifier told to sort
  real rows from injected M3 rows does it at AUC 1.00. A better detector cannot fix a fold that
  is not measuring detection. **Ticket 11** is where the fold has to say this itself; **ticket
  15** is where the generator closes the distance.
- **These rows are the ensemble, not the reference above.** `defend.unsupervised.ensemble` is on
  by default, so every system here is the supervised detector blended with an isolation forest at
  weight 0.7. The model card in `metrics.json` names both halves.

### On the synthetic placeholder — pipeline check, not comparable

```
A_baseline   PR-AUC 0.574   recall@1%FPR 0.255   precision@100 0.63   friction 57%
B_smote      PR-AUC 0.574   recall@1%FPR 0.255   precision@100 0.63   friction 62%
C_adaptive   PR-AUC 0.183   recall@1%FPR 0.000   precision@100 0.17   friction 91%
```

The adaptive system lands below both controls. That is the weak-side reading the design predicts
when the loop searches a single vector against a detector that already generalises to the holdout,
and widening the search is ticket 12's job.

**Ignore the `evasion_rate` column for now.** It reads 0.00 for all three systems, which sounds
like every attack was stopped; the friction column says what actually happened. Calibration places
`decline_at` at the target FPR and then puts the three softer bands at fixed ratios beneath it,
so almost everything picks up *some* friction and nothing is technically "allowed". Blanket
friction on 91% of traffic is not detection. **Ticket 09** replaces those ratios with bands chosen
by expected cost, and that is the column to re-read afterwards.

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
