# The realism leash, the two audit rules, and why they are the same finding

_Ticket 14. Written against the committed A/B/C/D run (`artifacts/abcd/amlworld_gather-scatter.json`),
B's ticket 16 table (`artifacts/three_system/*.json`) and the shipped code as of this commit. No run
was repeated to produce it._

The loop has two guards on the attacker. This document is about the discovery that **one of them was
not holding anything, the other was refusing everything, and both facts are the same fact seen from
opposite ends of the pipeline.**

## 1. The leash was not binding

Fitness is `evasion − λ · realism_penalty`. That is only a leash if the penalty moves between
candidates: subtracting an equal constant from every trial leaves the argmax alone, so **a constant
penalty is a λ that does nothing, however large λ is.**

Across the committed run — 7 seeds × 6 rounds — the penalty was:

```
seed 7     0.0652  0.0633  0.0654  0.0653  0.0660  0.0657
seed 11    0.0651  0.0648  0.0637  0.0646  0.0647  0.0655
seed 23    0.0654  0.0661  0.0658  0.0655  0.0640  0.0649
seed 42    0.0645  0.0649  0.0650  0.0663  0.0651  0.0653
seed 101   0.0651  0.0649  0.0651  0.0629  0.0648  0.0641
seed 1337  0.0638  0.0667  0.0657  0.0649  0.0648  0.0642
seed 2024  0.0656  0.0646  0.0661  0.0633  0.0644  0.0649
```

**All 42 rounds between 0.0629 and 0.0667** — a mean of 0.065 and a total spread of 0.0037. One
value, not a distribution, and the hard violation cliff never fires at all: no round in the
committed run is vetoed by the leash.

_This table was regenerated under ticket 20. It previously carried a `1.0000` in seed 7's last
round and read "41 of 42 rounds", because it had been copied from the **retired** v1.0 artefact —
the one `artifacts/abcd/README.md` says no document may quote. The canonical run has 0.0657 there.
The argument below is unaffected and slightly stronger: not even the cliff fired._

### Why: three bounds, none of which can bind

`realism.check` sums three soft terms and divides by three. Measured against the real anchors:

| term | shipped bound | what the anchors actually measure | can it fire? |
|---|---|---|---|
| `degree` — max share of fraud edges on one beneficiary | ceiling 0.6 | amlworld 0.0084, amlsim 0.0035, paysim 0.0025 | no: the ceiling is **70×** the busiest anchor |
| `round` — share of round-hundred amounts | ceiling 0.5 | amlworld 0.0001, amlsim 0.0002, paysim 0.0007 | no: the ceiling is **700×** anything real |
| `precision` — share of amounts with sub-unit precision | target 0.6 | amlworld 0.9875, amlsim 0.9826, paysim 0.9893 | it never stops firing |

The first two ceilings sit so far above anything real data does that no plausible generator reaches
them. The third is worse than loose — it is **pointed the wrong way.** Every real anchor sits at
~0.99, and the target was guessed at 0.6, so a generator that correctly matches reality is charged
for it, at a fixed rate:

```
|0.99 − 0.6| × 0.5 / 3 = 0.065
```

That is the constant, derived. The leash was not measuring realism; it was measuring the distance
between one guess and the truth, and reporting it every round as though it were news.

**λ was therefore a no-op on the search for the entire committed experiment.** Two of the three
terms were structurally dead, the third was a constant, and a constant cannot change an argmax.
`tests/test_realism.py` pins all of this, including λ at two values against a constant penalty
(no change) and against a moving one (the winner changes) — the demonstration the ticket asks for,
run against a controlled stand-in rather than a repeat of the anchor.

### What was changed, and what deliberately was not

`RealismBounds.from_anchor` measures the three quantities off the real anchor, which is what the
ticket asked for instead of three guesses. `RealismReport` now carries `terms` — each term's own
contribution — and `binding`, which answers "is this number responding to anything?" directly.

**The shipped defaults are unchanged.** `check()` behaves exactly as it did, so no committed number
moves and the A/B/C/D artefact still traces to the code that produced it. Enabling measured bounds
changes the optimiser's search and would invalidate that artefact; that is a re-run, and a re-run is
a decision rather than a patch. The finding ships; the fix is staged behind it.

## 2. The audit gate was refusing everything

Meanwhile, at the other end of the pipeline, B hit the mirror image. `afl/attack/multi.py` now
carries two rules for the same question:

- **`lift`** — reject when the commensurability score exceeds `3 × base_rate`. What ticket 12
  shipped and what the A/B/C/D run used. It has no floor, so it *tightens as the anchor grows*: a
  hundred injected rows in a 600k-row anchor put the bar near 5e-4 PR-AUC, which log-amount alone
  clears. **On amlsim and paysim it rejects the batch the loop kept in 71 of the 72 rounds run.**
- **`envelope`** — reject on `envelope.audit`'s own `trivially_separable` verdict, floor included.
  The rule the rest of the repo already applies to this question.

B ran ticket 16 on `envelope`, because a gate that refuses all but one round makes System C a copy of
System A and the table vacuous. Both verdicts are now recorded on every trial, so no run has to be
repeated to learn what the other rule would have said.

## 3. The reconciliation: both rules reach the same verdict by different routes

This reads at first like a disagreement between A's gate and B's gate. It is not. Follow what
actually happened on amlsim and paysim:

```
lift rule       →  refuses the batch AT GENERATION           →  no candidate survives
envelope rule   →  admits the batch at generation
                →  probe finds it separable AT EVALUATION    →  PR-AUC 0.998 / 0.970
                →  ticket 11 withholds the column            →  no quotable number
```

**Two independent guards, opposite ends of the pipeline, same verdict: separable, nothing
quotable.** Loosening the gate did not rescue the table — it only moved where the refusal happened,
from generation to evaluation. `lift` was not being over-strict on those anchors; it was right, and
its rejection rate was the first signal of a fact B's provenance probe later confirmed with a
number.

That is the reconciliation, and it is worth more than either half alone: a guard that agrees with an
independent guard built by someone else, measuring a different thing at a different stage, is
evidence the guards are measuring something real rather than reflecting the assumptions of whoever
wrote them.

**Which rule should ship** is now answerable on evidence rather than preference. `lift` has the
right instinct and the wrong scaling — its bar is a function of anchor size, which is why it goes
from admitting everything on amlworld (0 of 42 rounds rejected) to refusing everything on a smaller
anchor. `envelope` has the right shape — an absolute floor — and is what the rest of the repo
already uses. The recommendation is `envelope` as the default with `lift`'s lift-ratio kept as a
recorded diagnostic, which is already what the code does on every trial. Changing the default is a
re-run, and is left as a decision rather than taken here.

## 4. Why this is the keystone

Five instruments now report on the same claim, and none of them validates it:

| instrument | verdict | what refused |
|---|---|---|
| transfer test | synthetic-trained loses to the amount floor | behaviour did not transfer |
| A/B/C/D, 7 seeds | D vs C is 4/7, p = 0.500 | nothing separated the systems |
| fidelity scorecard | both anchors FAIL on level-3 utility | the generator did not earn its seat |
| sequence gate | did not earn its seat | history belonged to the generator, not the account |
| GNN gate | fell back to hand-rolled features | the motif was not visible to a causal graph |

What ties them together is not that they all say no. It is that **every one of them says no for a
reason that traces to the same underlying fact**: on these anchors, injected synthetic rows are
distinguishable from real ones, so anything measured on a fold of them is measuring provenance
rather than fraud. The audit gate says it at generation. The provenance probe says it at evaluation.
The fidelity gate says it as a utility score. The sequence and GNN gates say it as "the model is
reading which generator wrote the row."

And the leash finding is what makes that argument honest rather than convenient. A project that
reports a null has to answer one question before anyone believes it: *did your attacker cheat, and
would you have noticed?* The answer here is uncomfortable and specific — **the mechanism intended to
notice was not working, we found out by plotting it, and here is the derivation of the constant it
was reporting instead.** That is a stronger position than a leash that was never examined, because a
leash nobody checked is not evidence of anything.

## What ticket 15 does and this does not

The expensive verdict is ticket 15's. This is the cheap per-round check, and its job is to make
cheating visible *while the loop runs* rather than at write-up time. It failed at that job for the
committed experiment. The evidence that the attacker did not in fact cheat comes from elsewhere —
the audit gate rejected 0 of 42 rounds, and the fidelity scorecard was run independently — not from
the leash, which had no opinion.
