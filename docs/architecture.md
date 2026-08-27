# Architecture — the seam, the features, the detector, the decision layer

_Design and rationale. `docs/architecture.html` is the original design doc; this is the prose that
used to live in the README. The repo tree is in the README._

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

