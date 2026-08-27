# Evaluation — what a number has to survive before it is quoted

_How the project decides whether a measurement means anything: the split, the carve-out, the
commensurability and provenance checks, the fidelity gate, and what several seeds can and cannot
settle._

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


## The transfer test

The held-out score cannot tell a detector that learned fraud from one that learned "the simulator
wrote this row". So `scripts/transfer_test.py` trains a detector that has never seen a real fraud
label — real legit traffic plus synthetic attacks only — and scores it against the anchor's own
real fraud. Fraud behaviour transfers; provenance does not.

```bash
python scripts/transfer_test.py --data amlworld
```

Four systems, all fitted on the training window and scored on the same test window: `real` (the
ceiling), `synthetic` (the transfer test), `both`, and `amount` (no model at all). On AMLworld the
synthetic-trained detector reaches PR-AUC 0.000532 on real GATHER-SCATTER against an amount floor of
0.001329 — training on the generated attacks is worse than sorting the test window by amount.
`artifacts/transfer/<anchor>.json`.

## The commensurability audit

Every fold in this project puts injected synthetic positives against real negatives, so "caught the
fraud" and "spotted the synthetic row" are the same label unless something rules the second out. The
audit scores each contract field on whether it alone separates synthetic from real: `log_amount`,
`hour_of_day`, `rail`, `sender_in_anchor`, `payee_in_anchor`, `payee_popularity`, `has_device`.

Five classes of leak have been caught by it — amount scale, rail, device column, time granularity
and entity namespace — one of which the audit itself initially missed (senders and payees were
unioned, so a customer appearing as a payee passed) and which is now a regression test.

The same question is asked at two points in the pipeline, and the two rules disagree about where the
bar sits. `docs/realism-leash.md` reconciles them.

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


## What several seeds can settle

The seed turns the whole pipeline — the attack episodes in the pool, the SMOTE draw, the optimiser's
search, the model's own randomness — so a spread measured over refits alone would answer a narrower
question than the one a reader has.

Every comparison is paired by seed and carries an exact sign test on the per-seed direction, because
seed variance in this project exceeds every between-system gap. Three seeds cannot reach p < 0.05 by
construction: 3/3 in one direction is p = 0.125 at best, and the tables say so rather than working
around it. Seven seeds can reach p = 0.008, which is why the A/B/C/D experiment was run at seven.

A gap smaller than its own spread is reported as inside the noise, whichever way it points.
