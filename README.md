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
make decisions   # price the graded action bands and reason codes; commit them
make anomaly     # score the zero-day layer against the supervised model on the held-out family
make loao        # the leave-one-attack-out matrix: every family held out in turn, with the guards
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

Both sit **~31x below** the synthetic default's measured 4.03% fraud rate. That is more than an
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
fitted on. The score → probability calibration is fitted on the same tail. An operating point
chosen on the window it is reported from is not an operating point, it is a result.

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

## Decisions

A score is not an answer. `docs/decisions.md` is what happens when a rank becomes an action
somebody has to work; `artifacts/decisions/<anchor>.json` is the evidence behind it.

```bash
make decisions   # price the bands on every anchor, and rewrite the write-up from the artefacts
```

Five graded actions — allow, step-up, hold, review, decline — and the one a transaction gets is
whichever **minimises expected cost** at its own probability and its own amount. There are no
threshold numbers in `config/defend/lgbm.yaml` any more; there is nowhere left to type one. The
eight business numbers that place them live in `config/costs/default.yaml`, each with a stated
`why`, and `CostModel.from_config` refuses to load a parameter whose `why` is blank.

Five things this got wrong before, all of them measured rather than reasoned about.

**The bands used to sit inside the score distribution's noise floor.** `calibrate_to_fpr` pinned
`decline_at` to the target FPR and put the other three at 0.8, 0.6 and 0.3 of it — ratios
calibrated to nothing. On the M3 fold the detector's highest probability is `1.8e-05`, so all four
bands landed inside that range and 45.6% of holdout traffic picked up friction while precision@100
was 0.00. That is not a strict policy, it is a threshold placed in numerical noise. A cost model
declines to act at a fraud probability of 0.0018%, which is the correct answer.

**A cost model needs a probability, and a boosted tree emits a ranking score.** `p × amount`
against a flat analyst cost is arithmetic on a probability. Running the same synthetic loop with
`decision.calibration=none` puts friction on **99.3%** of legit traffic against 9.3% with Platt
scaling fitted on the validation tail.

**So the two scales are kept apart.** The calibrated probability chooses the action and appears in
the reason code; `DetectorScore.score` stays the detector's own score, which is what every metric
reads. That division means the decision layer cannot move PR-AUC, recall@1%FPR or precision@k *by
construction* — not by a monotonicity argument. The argument was tried first and it failed:
`1/(1+exp(-z))` rounds to exactly 1.0 in float64 past z ≈ 37, and on PaySim's committed test window
the fitted map collapsed 129 distinct scores in the top 200 rows into a single value across 480 of
them, moving precision@100 on the stock-params control from 0.14 to 0.06. A detection metric had
moved because of a decision knob.

**A flat cost in absolute currency cannot serve two anchors.** PaySim's median payment is 74,872
and AMLSim's is 157. Flat costs are therefore quoted against `unit_amount` — the anchor's own
median payment — and resolved to currency at load, so the same eight numbers place the same ladder
on both files instead of declining everything on one and nothing on the other.

**Cost-derived does not mean less friction, and the artefact is where that gets settled.** A
policy that minimises expected cost will buy *more* friction when the fraud it stops is worth more
than the friction costs. On PaySim's committed test window it frictions 3.80% of legit traffic
against the ratio bands' 2.19%, declines almost nothing where the old policy declined 0.75%, and
lets 36.3% of fraud through against 44.2%. Under the cost model that is a 86.7% improvement on
allowing everything, where the policy it replaced managed 1.5%.

Whether that trade pays is an empirical question about a particular anchor, so `make decisions`
measures it rather than assuming it — against the ratio bands, against allowing everything and
against declining everything, all four scored from the same probabilities so the only difference
is where the bands sit. **On AMLSim every one of them loses to doing nothing**, and the artefact
says why in a sentence: the entire fold's fraud is worth 5,206, an analyst review is priced at
7.84, so the whole window is worth 664 reviews and no threshold anywhere can pay for itself. That
is a fact about the anchor, and one more reason not to quote AMLSim.

The tests assert only the floor that always holds — a policy has to beat doing nothing and beat
blocking everything — because "lower cost than the ratio bands" is not guaranteed on a finite
sample and, on the small synthetic window, is not even true.

Every flagged transaction carries at least three reason codes in analyst language, and that is an
invariant rather than a target: `explain` chooses whether *allowed* rows are explained too, never
whether flagged ones are. When SHAP is unavailable the fallback to global importance is labelled
inside the reason string, so an explanation that is not about this transaction says so wherever it
is shown.

## The anomaly layer, and the result it did not produce

The supervised model can only catch what it has labels for, and leave-one-attack-out is defined by
holding one family's labels back. An outlier score fitted on legit traffic alone has no notion of
"the fraud I have seen", so the design bet was that it degrades more gracefully against the unseen
family and sits underneath the ensemble as a floor. `docs/anomaly.md` is where that bet was
settled; `artifacts/anomaly/<anchor>.json` is the evidence.

```bash
make anomaly     # five systems, one fold, one operating point — and rewrite the write-up
```

**The bet lost, on both anchors.** The supervised model does not collapse on the held-out family:
on PaySim it reaches PR-AUC 0.524 and recall@1%FPR 1.000 there, against 0.152 / 0.478 on the
anchor's own real labelled fraud in the same test window. A family it has never seen is *easier*
than one it trains on, which is a finding about the injected rows rather than about
generalisation. And the anomaly layer is nowhere near it — 0.033 on PaySim, 0.003 on AMLSim, where
sorting by amount alone reaches 0.034. It is not the floor under the ensemble; it is below the
floor.

**The blend is the part that earns its place, and only on one anchor.** The weight is swept end to
end on the same pair of score vectors, so both halves appear in the same curve as the blends of
them. On PaySim the curve has an interior optimum — w=0.5 reaches 0.551 against 0.524 for the
supervised model alone — so the two together genuinely beat either. On AMLSim it rises
monotonically to w=1.0 and the shipped 0.7 costs 0.013 PR-AUC. The shipped weight stays where it
is: a weight chosen on the fold it is reported from is the tuning-on-test the baseline forbids.

**Two things were wrong underneath, and both are the kind that leave the metrics looking fine.**
The outlier score was min-maxed over whatever batch it was handed, so a transaction's score was a
statement about its company — 0.25 of drift on the same PaySim rows, invisible to PR-AUC because a
within-batch min-max is monotone, and blended 0.3-to-0.7 against a probability. And on PaySim the
simulator had barely any anchor accounts to stage attacks on, because the envelope's "seasoned
account" filter wanted senders that transact twice and PaySim's senders are unique per row: it
found 86 for 340 population slots and minted the other 254, so `sender_in_anchor` separated the
held-out family from the anchor at **PR-AUC 0.800** — 1.000 on a smaller sample, where the pool
empties entirely. Every anchored PaySim number produced before this ticket inherits that. Both
are fixed, and both are measured in the artefact rather than asserted in a comment.

## Leave-one-attack-out, and what a fold is allowed to claim

`make loao` holds out every family in turn and writes the matrix to `artifacts/loao/<anchor>.json`
and `docs/loao.md`. The three guards above make the carve-out airtight. They are not what decides
most of the table.

**A fold that runs is not a fold that means something.** The carve-out drops the anchor's own
fraud from the holdout, so in every fold *every positive is an injected synthetic row and every
negative is a real one* — "caught the fraud" and "spotted the synthetic row" are the same label.
Ticket 07 noticed this and measured it by hand at AUC 1.00; it is a check in the harness now, and
it decides rows. A classifier gets the fold's own features and is asked to sort injected from
real. Where it succeeds, the fold's recall is a statement about the generator and the numbers are
withheld — they move out of `metrics` into `withheld_metrics`, so a reader who quotes the obvious
field gets `None` rather than a number they should not have.

So every fold lands on one of three outcomes, and only the first carries a claim:

- **measured** — the numbers stand.
- **withheld** — the fold ran and the numbers exist, but nothing follows from them: too few
  positives to move a metric by less than a rounding error per row, separable from the anchor by
  one contract field, separable by a whole model, or a `template` vector whose defining tell is
  not modelled yet.
- **skipped** — the fold never ran, and the reason sits where the number would be. Every
  requested fold gets a row either way; a fold that vanishes reads as "not applicable" when it
  means "we did not look".

The amount floor rides along on every fold — rank by amount, no model, direction chosen on the
training window. Two earlier results in this repo were walked back for want of that column.

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
config           Hydra configs; costs/ is the operating point, experiment/{baseline,smote,adaptive}
scripts          run_experiment, build_splits, build_features, build_baseline, build_decisions,
                 build_anomaly, build_fidelity, make_figures
```

## How it's scored

The split is out-of-time, never random, because random splits leak the future.

We use leave-one-attack-out evaluation: train without an attack family, then measure recall on it.
That's the number that matters; everything else is supporting evidence. Any vector can be the
holdout — `config/eval/leave_one_attack_out.yaml` names which fold is the headline, and `make
loao` runs the whole matrix regardless.

We report PR-AUC, recall at a fixed false-positive rate, and precision@k. We do not rely on
accuracy or ROC-AUC alone because, at a sub-2% fraud rate, they can flatter a model that catches
almost nothing. `afl/defend/baseline.py` refuses to save an artefact containing either.

**Three guards make the carve-out mean something,** and all three are assertions with a test that
deliberately tries to leak a row past them:

- Not one row of the held-out family reaches training — **the detector's replay buffer included**.
  The audit runs against the fitted detector's `training_rows`, not the list handed to `fit`,
  because the replay buffer is where a carved-out family walks back into training four rounds
  later without the split changing. A detector that cannot say what it trained on fails the guard.
- The split is still out-of-time with the committed embargo intact **after** the carve-out.
- Every legit row of the test window stays in the holdout. An FPR with no negatives is not an FPR.

**And a fourth check that is a verdict rather than a guard** — the provenance probe, in
*Leave-one-attack-out, and what a fold is allowed to claim* above. It is the one that
decides most of the matrix.

## Current numbers (honest)

On the synthetic placeholder config, held out on M3, the adaptive system lands below both
baselines. Regenerated by `make compare` after ticket 09, on LightGBM 4.5.0:

```
system       PR-AUC   recall@1%FPR   precision@100   evasion   friction
A_baseline   0.567    0.289          0.66            0.113     0.093
B_smote      0.567    0.289          0.66            0.320     0.076
C_adaptive   0.145    0.062          0.21            0.354     0.226
```

**This is a pipeline check, not a result.** `data=synthetic` has no real anchor, and the run says
so in a banner and in its own artefact. Reportable numbers need `data=paysim`; the detector's
reference is `docs/detector.md` and the decision layer's is `docs/decisions.md`.

Two things about the table have changed since it was last written down, and neither is a
regression. It runs on **LightGBM** now rather than the sklearn fallback — libomp was missing on
the machine that produced the older numbers, so none of them were ever LightGBM's, which ticket 08
found and fixed. And `evasion` and `friction` come from a cost model now rather than from four
thresholds nobody chose; the three ranking columns cannot move for that reason, and did not.

**A and B are identical, and that is the honest reading.** SMOTE interpolates between existing
fraud rows; on this holdout it cannot invent the one thing that would help, so it reproduces the
baseline to six decimals while doubling the training fraud. That is precisely why System B is in
the table — it makes System C falsifiable, and here it falsifies it.

M3 is a hard holdout on purpose: genuine first-party fraud, where no device changes, no new
operator appears and no new beneficiary is ever paid, so none of the signals a supervised model
leans on fire at all. C searching a single vector against a detector that already generalises to
that holdout is the weak-side reading the design itself predicts. Ticket 12 widened the search and
`artifacts/abcd/` records what happened: adaptive did not beat non-adaptive, 4 of 7 seeds,
p = 0.500. Reported as a negative rather than re-run until it wasn't.

Each numeric regime supersedes the last rather than sitting beside it — the vectors, the holdout,
the backend and now the decision layer have each moved the table, and a run from before any of
them is not comparable. **System C in particular does not survive a decision-layer change**, since
the loop retrains on whatever the policy allowed through.

## Vectors

Nine vectors, three engines. The ids match the architecture doc and are frozen; see
`docs/adr/0001-vector-taxonomy-and-holdout.md` for why each one is where it is, and
`afl/attack/templates/vectors.yaml` for the definitions. Adding a vector is a YAML edit.

| id | vector | engine | level | tier | status |
| --- | --- | --- | --- | --- | --- |
| S1 | Mule network & layering | graph | mechanism | strong | built |
| S2 | Card testing / BIN enumeration | velocity | mechanism | strong | built |
| S3 | Account takeover via drift | drift | mechanism | strong | built |
| C1 | Bust-out | drift | mechanism | common | built |
| C2 | UPI collect-request / APP scam | velocity | enabler | common | built |
| C3 | Instant-A2A pass-through | graph | mechanism | common | built |
| M1 | Boundary probing / paced evasion | velocity | model-attack | mid | template |
| M2 | Synthetic-identity lifecycle | drift | enabler | mid | built |
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

So eight of the nine are done. Only M1 still owes work, and it arrives free as the optimiser's
own boundary walk. **M3 is the leave-one-attack-out holdout** because `user == fraudster` breaks
the legit-vs-attacker assumption every supervised feature rests on: the abuse runs on the owner's
own device, to beneficiaries the account already pays, elevated only against that account's own
history. A one-line amount rule at 1% FPR catches 3% of it, which is the point.

The commodity families are the mirror image and are meant to be caught. C1 spikes 7-10x against
its own tenure, C2 pays a first-time payee in plain sight, C3 relays money onward in under two
minutes. They are training load and fixed benchmarks, never the holdout. See
`docs/adr/0003-template-vectors.md`.

## What we're not claiming

Fidelity metrics are diagnostics, not proofs. A C2ST score near chance does not mean "realistic,"
and DCR/MIA do not mean "private." They mean we tested for memorisation and membership leakage.

Synthetic-only data lowers exposure but does not automatically mean DPDP compliance.

Frontier vectors are demonstrated capabilities, not necessarily mass-exploited attack patterns,
and we label them that way.

Saying this plainly is deliberate.
