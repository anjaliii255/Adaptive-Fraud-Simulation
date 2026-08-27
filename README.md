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

## How it fits together

```mermaid
flowchart LR
    subgraph DATA["data"]
        direction TB
        ANCHOR[Real anchor<br/>PaySim · AMLSim<br/>BankSim · AMLworld]
        LOADER[Config-driven loader]
        SCHEMA[Contract schema]
        SPLIT[Committed split<br/>digest · embargo]
        ANCHOR --> LOADER --> SCHEMA --> SPLIT
    end

    subgraph RED["RED · attack"]
        direction TB
        ENV[Anchor envelope<br/>scale · rails · pools]
        SIM[Simulator<br/>3 engines · 9 vectors]
        OPT[Adaptive optimiser<br/>search over knobs]
        ENV --> SIM --> OPT
    end

    GATE{{Audit gate<br/>separable? reject}}

    subgraph BLUE["BLUE · defend"]
        direction TB
        FEAT[Feature engineering<br/>causal · graph]
        DET[LightGBM detector]
        ACT[Graded action · SHAP]
        FEAT --> DET --> ACT
    end

    subgraph EVAL["evaluation"]
        direction TB
        LOAO[Held-out family]
        ABCD[A/B/C/D]
        CURVE[Convergence curve]
        LOAO --> ABCD --> CURVE
    end

    subgraph INST["INSTRUMENTS · guards"]
        direction LR
        COMM[Commensurability audit]
        FID[Fidelity gate]
        TRANS[Transfer test]
        PROV[Provenance probe]
    end

    SCHEMA --> ENV
    OPT --> GATE
    GATE -->|admitted · retrain| FEAT
    SPLIT --> FEAT
    DET --> LOAO
    DET ==>|loop · evasions| OPT

    GATE -.-> COMM
    SIM -.-> FID
    ABCD -.-> TRANS
    LOAO -.-> PROV

    classDef red stroke:#c0392b,stroke-width:2px,fill:none
    classDef blue stroke:#2471a3,stroke-width:2px,fill:none
    classDef inst stroke:#7f8c8d,stroke-width:1px,stroke-dasharray:4 3,fill:none
    classDef plain stroke:#566573,stroke-width:1px,fill:none
    class ENV,SIM,OPT red
    class FEAT,DET,ACT blue
    class COMM,TRANS,FID,PROV inst
    class ANCHOR,LOADER,SCHEMA,SPLIT,LOAO,ABCD,CURVE,GATE plain
    style DATA fill:none,stroke:#95a5a6
    style RED fill:none,stroke:#c0392b
    style BLUE fill:none,stroke:#2471a3
    style EVAL fill:none,stroke:#95a5a6
    style INST fill:none,stroke:#7f8c8d
```

The contract schema is the seam: both sides code against one internal type, so nothing downstream
knows or cares which anchor the rows came from, and `afl/attack/**` and `afl/defend/**` never import
each other. The bold edge is the loop — the detector's evasions steer the optimiser's next search,
the detector retrains on what the gate admitted, and the round repeats. The dashed instruments are what
make the whole thing answerable: they gate a batch before it is ever trained on, and verify a fold
before a number from it is allowed to be quoted, which is why several numbers in this README are
reported as withheld rather than reported as wins.

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

The **leave-one-attack-out harness is built, and most of what it produced is a refusal to
report.** Nine families held out in turn on two anchors; three guards on every carve-out; and a
fourth check that asks whether the fold is measuring detection at all. **Four of the eighteen
folds carry a quotable number, and the headline fold is not one of them** — on PaySim a
classifier sorts the injected M3 rows from real traffic at PR-AUC 0.970 where the detector
reaches 0.893, so the fold cannot tell the two apart. The matrix is in `artifacts/loao/` and
written up in `docs/loao.md`, generated from it.

The **three-system table is built**, on both anchors, three seeds each — and its one apparent win
does not survive its own audit. Real-only, SMOTE and the adaptive loop, on one carve-out at one
operating point, reported in two columns: the held-out family nobody trained on, and the anchor's
own labelled fraud that everybody did. System C appears to beat SMOTE by **+0.76 recall** on the
held-out family on AMLSim — and a model given System C's training rows and told *only which rows
the generator wrote* scores 0.995 on the same column against System C's 0.998. The number is the
generator's fingerprint, so it is withheld. On the column that is measurable, the three systems
are within the seed-to-seed spread of each other. `artifacts/three_system/`, written up in
`docs/three_system.md`.

The **sequence model was built, measured against LightGBM, and did not earn its seat.** A GRU over
per-entity history on the drift arc — S3 takeover and C1 bust-out, each generated at both ends of
the `ramp` axis so sudden and gradual drift are reported apart. On the anchor whose accounts have
real pasts it loses on merit: PR-AUC 0.391 against LightGBM's 0.997 on gradual S3, 0.300 against
0.998 on gradual C1, at a *lower* compute cost than the baseline. On the anchor with no per-entity
history it wins — and the win is what disqualifies it, because window length alone sorts the
injected episodes from real PaySim rows at PR-AUC 0.933 and 0.985. `config/defend/sequence.yaml`
stays `enabled: false` and a test refuses to let it be turned on against the committed evidence.
`artifacts/sequence/`, written up in `docs/sequence.md`.

The **temporal GNN was built, measured against those hand-rolled graph features, and did not earn
its seat** — so the fallback ticket 18 named up front is what ships. Graph attention over the
account-beneficiary graph inside an explicit seven-day window, on the two mule families, at three
seeds with the lift paired across them. On AMLSim it loses by almost the whole scale (PR-AUC 0.023
against LightGBM's 0.998 on S1) and that fold is *still* withheld, because AMLSim's clock is whole
days: an injected ring is instantaneous on it, and only 8.2% of its rows can see any earlier edge
of their own episode, so a causal temporal graph was asked about a shape it cannot see. On PaySim,
where the clock has the resolution, it is level with what ships and ahead of the graph-blocks-only
baseline — and those folds are withheld too, because a quarter of the injected rows sit in a
neighbourhood made only of other injected rows. `artifacts/gnn/`, written up in `docs/gnn.md`.

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

`make sequence` and `make gnn` are the only targets that need anything more — torch and
torch-geometric, via `make setup-deep` (or `uv sync --extra dev --extra deep`). Nothing else
imports them, `SequenceDetector` and `TemporalGNNDetector` raise rather than degrading to a
stand-in when they are absent, and the default suite stays green without either.

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
make sequence    # GRU vs LightGBM on the drift arc, sudden and gradual apart (needs setup-deep)
make gnn         # temporal GNN vs hand-rolled graph features on the mule families (setup-deep)
make fidelity    # the 3-level scorecard on every real anchor; level 3 is the gate
make loop        # run the adaptive loop (synthetic default, no download)
make table       # the three-system table: real-only vs SMOTE vs adaptive, both columns, 3 seeds
make compare     # the same three through the hydra loop on the default config — a pipeline check
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
and `docs/loao.md`.

**Three guards make the carve-out airtight,** and all three are assertions with a test that
deliberately tries to leak a row past them:

- Not one row of the held-out family reaches training — **the detector's replay buffer included**.
  The audit runs against the fitted detector's `training_rows`, not the list handed to `fit`,
  because the replay buffer is where a carved-out family walks back into training four rounds
  later without the split changing. A detector that cannot say what it trained on fails the guard.
- The split is still out-of-time with the committed embargo intact **after** the carve-out.
- Every legit row of the test window stays in the holdout. An FPR with no negatives is not an FPR.

They are not what decides most of the table.

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

## The sequence model, and the seat it did not earn

`make sequence` puts a GRU over per-entity history against the tuned LightGBM baseline on the
drift arc, writes `artifacts/sequence/<anchor>.json` and generates `docs/sequence.md` from it.
It needs the `deep` extra (`make setup-deep`); nothing else in the repo does, and the default
suite stays green without it.

**The axis is the experiment.** `ramp` is the drift engine's shape knob: 0 is a hard switch at the
takeover event, 1 is escalation spread across the whole tail. Sudden takeover is an *event* — the
amount jumps, the beneficiary is new, the device changes — and a per-row feature table sees all
three on the row itself. Gradual drift has no event to anchor on, and that is where per-row
features are supposed to run out. So each family is generated twice, at both ends of its own
declared search space with nothing else changed, and the two ends are reported separately against
the same haystack at the same threshold. The gate is decided on the gradual end only: a win on
sudden drift is a win at the easy end and does not promote anything.

**It lost, and the two anchors lost it differently.** On AMLSim, whose accounts carry 28 steps of
real history apiece, the sequence model reaches PR-AUC 0.391 on gradual S3 against LightGBM's
0.997, and 0.300 against 0.998 on gradual C1 — while fitting in 21s against LightGBM's 32s and
scoring at twice the rate. It is not an expensive model that bought a small lift; it is a cheap
model that lost. On PaySim it wins C1 by 0.987 to 0.773 — and PaySim's `nameOrig` is effectively
unique per row, so a real window there is **one step long** while the injected episodes carry
eight or nine. Window length alone sorts the two at PR-AUC 0.985. The win is the fold's shape, not
the model's skill, and the audit built for exactly this model is what catches it.

**None of those four numbers is quotable, and the refusal is the point.** All four folds are
withheld — two on the provenance probe, two on the history audit this ticket added — so the
comparison is printed in brackets in `docs/sequence.md` and quoted nowhere. What stands is the
decision, not the margin: a layer nothing could measure honestly is a layer that does not enter
the reported table, which is the same answer a measured loss would have given.

**Enabling it is not a preference.** `assert_config_matches_promotion` refuses
`defend.sequence.enabled: true` while a committed artefact says the gate said no, and a test runs
it against what is on disk. The claim "it only ships if it wins" is enforced rather than
remembered.

One more thing fell out of it, and it revises a row we already published. These folds carry 550
injected AMLSim C1 rows where the leave-one-attack-out matrix carries 80, so the provenance probe
is far better powered here — and at that size it separates injected from real at PR-AUC 0.688,
over the bar, where the matrix measured 0.236 and reported the fold. Nothing about the generator
changed; the episode count did. `docs/sequence.md` names it, and `docs/loao.md`'s AMLSim C1 row
should be read as underpowered until `make loao` is re-run at this episode count.

## The temporal GNN, and which one shipped

`make gnn` puts graph attention over the account-beneficiary graph against the hand-rolled
graph features + LightGBM baseline on the mule families — S1 fan-in and layering, C3 instant
relay — writes `artifacts/gnn/<anchor>.json` and generates `docs/gnn.md` from it. Like `make
sequence` it needs the `deep` extra (`make setup-deep`), and `TemporalGNNDetector` raises without
it rather than degrading to a stand-in.

**The window is the design.** Time is cut into daily strides, and a payment is scored against the
graph of the previous seven days *up to the start of its own stride* — nothing at or after it,
and nothing older than the window. That is what makes the graph temporal rather than a graph with
timestamps on it, and it is also the constraint that decides the whole ticket: a model that may
only read what happened earlier cannot see a ring that has not formed yet.

**Three seeds, and the lift is paired across them.** Each seed regenerates its own pool and
refits every system, and the margin is reported as a per-seed difference with its spread and a
sign test — the same `Spread` and `Comparison` the three-system table uses, so the two are read
at one bar. A margin smaller than its own seed-to-seed spread does not promote anything, and
neither does one seed.

**It did not earn its seat, and the two anchors refused it for opposite reasons.** On AMLSim it
loses by almost the whole scale: PR-AUC 0.023 ± 0.038 on S1 against LightGBM's 0.998 and 0.002 ±
0.002 on C3 against 0.984, 0/3 seeds in its favour — but that fold is withheld rather than
counted, because AMLSim's rows are whole **days**. An injected mule ring is instantaneous on that
clock, and only **8.2%** of its rows (0.6% for C3) can see any earlier edge of their own episode,
against a floor of 20%. The model was asked about a shape it structurally cannot see. On PaySim,
whose clock is hourly and where 86% of injected rows can watch their own ring form, it is level
with what ships — −0.023 ± 0.229 on S1 (1/3 seeds) and +0.024 ± 0.343 on C3 (2/3), both inside
their own spread — and ahead of the graph-blocks-only baseline by +0.079 and +0.139. It costs
more either way: 47s to fit against LightGBM's 30s on AMLSim, 38s against 15s on PaySim.

**All four folds are withheld, and on PaySim the audit that catches them was built for this
model.** A row's *neighbourhood provenance* — what share of its endpoints' in-window neighbours
are injected rows — sorts injected S1 and C3 from real PaySim traffic at PR-AUC 0.53 and 0.54,
because a quarter to a third of the injected rows sit in a neighbourhood made only of other
injected rows. PaySim accounts appear roughly once, so a staged ring there is its own synthetic
island, and message passing over an island returns "synthetic" before it returns anything about
topology. The ordinary provenance probe agrees on S1 at 0.865.

**So the hand-rolled graph features are what ship**, which is the fallback ticket 18 named before
the experiment ran rather than after it. `config/defend/gnn.yaml` stays `enabled: false`,
`assert_config_matches_promotion` refuses to let it be turned on while the committed artefacts say
no, and a test checks both that config and this sentence against `artifacts/gnn/`.

One more thing fell out of it, and it revises a row we already published — the same way `make
sequence` did. These folds carry 173 injected PaySim S1 rows where the leave-one-attack-out matrix
carries 50, and at that size the provenance probe separates injected from real at 0.865 where the
matrix measured 0.298 and reported the fold. Nothing about the generator changed; the episode
count did. `docs/gnn.md` names it, and `docs/loao.md`'s PaySim S1 row should be read as
underpowered until `make loao` is re-run at this episode count.

## Fidelity, and the level that decides it

`make fidelity` scores the generator against each real anchor on three levels plus a privacy
panel, writes `artifacts/fidelity/<anchor>.json` and rewrites `docs/fidelity.md` from it. The
levels are not equal. **Level 3 is the gate** — does training on this data teach a detector
anything about real fraud — and levels 1 and 2 are diagnostics that explain where level 3 landed.
A generator that resembles real traffic and teaches a model nothing has failed, however pretty
its histograms.

**Both anchors fail, and the gate is what fails them.** On PaySim, the anchor to read:

```
system         trained on                          PR-AUC   recall@1%FPR   beats the floor
trtr           real rows, real labels              0.158    0.444          yes
tstr           real legit + generated fraud        0.005    0.024          no
augmented      real rows + generated fraud         0.047    0.215          no
amount floor   nothing                             0.057    0.212          --
standalone     the generator's whole output        0.003    0.000          no
```

A detector trained on the generated fraud reaches PR-AUC 0.005 against real PaySim fraud, an
order of magnitude *below* sorting the test window by amount. Adding those rows to a real
training set does not help it either: recall at 1% FPR falls from 0.444 to 0.215, a 22.9-point
loss. Level 1 passes on the same card (0.749, above its 0.70 bar) and rescues nothing — that
division is the design, and it is enforced in the arithmetic as well as the prose, because the
headline score is capped at the level-3 score.

**The bars predate the numbers, and that is checked rather than claimed.** They live in
`config/fidelity/thresholds.yaml`, one bar per stated reason, refused at load if the reason is
blank. Each names the commit it was first committed in, and every run reads that commit back out
of git and compares the value committed there against the value being applied now. Six of the
seven trace unchanged to the day-one skeleton; the seventh — a TSTR score must beat the amount
floor — was committed before the first anchored run existed. Edit one and the artefact says so,
names the direction, and spells LOOSENED in capitals. A failing card exits non-zero *after*
writing itself, so it is committed rather than quietly re-run at a friendlier setting.

**Two measurement bugs were found by running it, and both flattered the generator.** The privacy
embedding standardised by `std + 1e-9`, and three of its seven columns are exactly constant on
PaySim — the anchor has no sender history, so no gaps, no out-degree, no unique-payee count. A
synthetic row with a real sender history was divided by a billionth, and the first PaySim card
reported a distance-to-closest-record ratio of **1.0e11** and passed the memorisation check
because of it. Constant columns are now dropped and named; the ratio is 2.43 over four real
dimensions. And membership inference on an out-of-time split measures the calendar as well as
membership, so the same attack now runs between two halves of the holdout, where nothing was ever
in training: on AMLSim it scores 0.343 there against 0.351 observed, so 0.008 of that advantage
is about membership and the rest is drift.

**What the privacy panel does not say.** DCR and MIA are evidence against memorisation, not a
guarantee. Neither can see the disclosure path this generator actually has: it stages attacks on
the anchor's own accounts by design, and 100% of generated rows name an account that exists in
the anchor. That is measured and reported rather than flagged, because it is the envelope working
as intended — but nobody should read "synthetic" as "contains no real identifiers".
**What the matrix said.** Four of eighteen folds carry a number: AMLSim C1 (0.975), M3 (0.996)
and S2 (1.000), and PaySim S1 (0.275). The AMLSim three are on an anchor where the same detector
already scores 1.000 on the anchor's *own* labelled fraud and sorting by amount alone reaches
0.456 — a near-perfect fold there says the simulator is legible, not that anything generalised.
PaySim S1 sits above PaySim's own labelled fraud (0.152), which is ticket 10's reading unchanged.
PaySim M3 — the headline fold — is withheld: probe 0.970 against the detector's 0.893. The
sharpest row is PaySim S2, where a classifier separates the injected card-testing rows at 1.000
and the detector scores 0.005 on them: perfectly identifiable by provenance, and invisible to the
model.

Seven more folds are withheld only for being thin at the committed `eval.holdout_episodes: 12`.
PaySim S3 is the one that costs something — detector 0.916, provenance probe 0.042, withheld for
six rows under the floor of 30. Raising the episode count would move ticket 10's committed fold
too, so it is a decision rather than an oversight, and it was not taken here.

## The three-system table

`make table` is the hero run: real-only, real + SMOTE, and real + the adaptive loop, on one
carve-out at one operating point, repeated over three seeds, on every real anchor. It writes
`artifacts/three_system/<anchor>.json` and regenerates `docs/three_system.md` from it —
`--doc-only` rebuilds the document from the committed artefacts alone, so nothing in it can
disagree with a run.

**System B is there to make System C falsifiable.** Row-level oversampling can move an amount and
a timestamp; it cannot invent a new fan-in shape, a new pacing strategy or a beneficiary that
never existed, which is precisely the gap the adaptive loop claims to fill. If C does not beat B
on the held-out column, this project is an expensive way of duplicating rows — and the table is
built to say so, with the reason attached.

**Two columns, because one is not a result.** *unseen* is the held-out family nobody trained on:
the claim. *known* is the anchor's own labelled fraud, scored on the same window against the same
legit haystack: the price of the claim. A system that buys the first by giving up the second has
traded rather than improved, and a one-column table cannot see the trade. The two columns share
their negatives, asserted rather than assumed — recall at a fixed FPR is a quantile of the
negatives, so two haystacks would be two operating points wearing one table.

**Every cell carries its spread, and every comparison is paired by seed.** The seed turns the
whole pipeline — the attack episodes in the pool, the SMOTE draw, the optimiser's search, the
model's own randomness — so the spread is the spread of the system rather than of a refit. A gap
smaller than its own spread is reported as inside the noise, whichever way it points, and the
sign test says plainly that three seeds cannot reach p < 0.05.

**System C gets one check the other two do not need.** It is the only row trained on generated
rows, so it is the only row whose held-out score can be the generator's fingerprint rather than
detection — and the fold's own provenance probe cannot settle that, because it learns "injected"
from the holdout's hundred-odd positives while System C learns it from thousands. So the
counterfactual is fitted directly: same training rows as System C, labelled only by *who wrote the
row*, never shown a row of the held-out family, then asked the question System C is scored on. If
it reaches System C's number, System C's number is provenance, and the cell is withheld with that
as the reason. On both anchors, it does.

**The held-out column inherits the leave-one-attack-out verdicts.** It is the same carve-out
`make loao` builds, with the same three guards, the same commensurability audit and the same
provenance probe — so where that harness withholds, this one withholds too: the numbers move to
`withheld_metrics` and print in brackets next to the reason. These are not the matrix's numbers
and are not comparable to them row by row: the fold is the same, but System A trains on the
anchor's real rows alone where the matrix's detector trains on the whole training side.

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
                 build_anomaly, build_sequence, build_fidelity, build_loao, build_three_system,
                 make_figures
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

The guards that make a carve-out mean something — and the check that decides whether a fold may
be quoted at all — are in *Leave-one-attack-out, and what a fold is allowed to claim* above.

## Current numbers (honest)

`make table` on both real anchors, three seeds each, held out on M3, at recall@1% FPR and
precision@100. Every cell is mean ± sd over the seeds; **numbers in brackets are withheld** —
they exist, and nothing may be concluded from them. Regenerated from
`artifacts/three_system/`, written up in `docs/three_system.md`, on LightGBM 4.5.0.

**PaySim** — 446,214 training rows, 369 of them fraud; 385k-row test window.

```
system         known PR-AUC   known rec@1%FPR   unseen PR-AUC     unseen rec@1%FPR
A_baseline     0.162 ± 0.016  0.437 ± 0.006     [0.064 ± 0.012]   [0.316 ± 0.043]
B_smote        0.166 ± 0.015  0.417 ± 0.007     [0.277 ± 0.014]   [0.429 ± 0.035]
C_adaptive     0.163 ± 0.036  0.443 ± 0.005     [0.679 ± 0.055]   [0.997 ± 0.006]
amount floor   0.057          0.212             0.006             0.040
```

**AMLSim** — 930,465 training rows, 1,170 of them fraud; 385k-row test window.

```
system         known PR-AUC   known rec@1%FPR   unseen PR-AUC     unseen rec@1%FPR
A_baseline     1.000 ± 0.000  1.000 ± 0.000     0.105 ± 0.007     0.214 ± 0.035
B_smote        0.996 ± 0.002  0.996 ± 0.002     0.121 ± 0.032     0.243 ± 0.044
C_adaptive     0.994 ± 0.004  0.994 ± 0.003     [0.998 ± 0.003]   [1.000 ± 0.000]
amount floor   0.456          0.474             0.030             0.034
```

**System C's held-out column is withheld on both anchors, and the second reason is the one that
matters.** On PaySim the whole column goes, for the reason ticket 11 already found: a classifier
sorts the injected M3 rows from real traffic at PR-AUC 0.970 where the detector reaches 0.285, so
nothing measured there can tell detection from provenance. On AMLSim that probe scores 0.24–0.36
and the fold passes it — A and B carry quotable numbers there — but System C's does not, because
a model given **System C's own training rows and told only which rows the generator wrote** —
never which are fraud, never a row of the held-out family — scores **0.995** on that column
against System C's **0.998**. Provenance alone reproduces the number. The +0.76 recall that
System C appears to win over SMOTE is the generator's fingerprint transferring between families,
not a detector generalising to an unseen attack.

**On the column that is measurable, the three systems are indistinguishable.** PaySim's known
column — real labelled fraud, 410 positives, out of time — reads 0.162 / 0.166 / 0.163 PR-AUC for
A / B / C, and every pairwise difference is inside the seed-to-seed spread. Adding 4,972 generated
fraud rows to a training set with 369 real ones moved nothing that could be measured. On AMLSim
the same column is at the ceiling for all three (1.000 / 0.996 / 0.994) against an amount floor of
0.456, which says that anchor's own fraud is trivially separable rather than that anything
generalised.

**The controls behave exactly as the design predicted, which is the one clean result here.**
SMOTE beats real-only on the held-out column on both anchors (+0.113 recall on PaySim, +0.029 on
AMLSim, 3/3 seeds each) and gives a little back on the known one (−0.020 recall on PaySim, −0.004
PR-AUC on AMLSim) — an oversampler doing precisely what an oversampler can do. The gap between B
and C is where the argument lives, and the audit takes it away.

**What the loop itself did.** Over twelve rounds evasion falls from 0.86–0.88 to 0.002–0.060 on
AMLSim and from 0.35–0.62 to 0.000–0.014 on PaySim: the attacker finds the detector's weak surface
early and the detector closes it. The commensurability gate rejected **none** of the 72 rounds
across the two anchors under the `envelope` rule, and would have rejected **71 of them** under the
optimiser's shipped `lift` rule — see `docs/three_system.md` for why the table runs on the former
and records both verdicts on every round.

**Three seeds cannot reach significance and the table says so.** Every comparison is paired by
seed and carries a sign test; 3/3 in one direction is p = 0.125 at best. A difference smaller
than its own spread is reported as inside the noise, whichever way it points.

`make compare` still runs the three systems through the hydra loop on the synthetic default. That
path is a **pipeline check, not a result** — `data=synthetic` has no real anchor, and the run says
so in a banner and in its own artefact.

Each numeric regime supersedes the last rather than sitting beside it — the vectors, the holdout,
the backend and the decision layer have each moved the table, and a run from before any of them is
not comparable. Ticket 12's A/B/C/D experiment on AMLworld (`artifacts/abcd/`) reported adaptive
failing to beat non-adaptive at 4 of 7 seeds, p = 0.500; this table is a different fold on
different anchors and does not overturn it.

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

The sequence model's negative result is about *this* GRU, on *these* two anchors, at one seed and
one set of hyperparameters — not about sequence models. What would change the answer is an anchor
with real per-entity histories on both sides of the label, which is the precondition
`docs/sequence.md`'s history audit measures and the one PaySim does not meet.

Saying this plainly is deliberate.
