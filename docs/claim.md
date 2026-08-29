# What this project claims, and what it does not

_Every number here traces to a committed artefact. Where a number is withheld, the reason is
stated rather than the number quietly dropped._

## The claim

**We build a closed-loop adaptive fraud simulator that searches for detector blind spots while
auditing whether its own synthetic attacks are realistic enough to support valid training and
evaluation — and it gates its own results, withholding any apparent gain that the provenance probe
explains.**

The second half is the part that took the work. Generating adaptive attacks is not hard. Knowing
whether the numbers they produce mean anything is, and most of this repository is the instrument
that answers that question, including in the cases where the answer disqualifies our own result.

## What changed, and why

An earlier version of this README claimed:

> Adaptive adversarial simulation improves detection recall on a held-out attack family, compared
> with real-only training and ordinary synthetic augmentation, reproducibly, with the mechanism
> shown.

**We do not claim that, because our own instruments contradict it.** A 7-seed A/B/C/D on AMLworld
put adaptive against non-adaptive at 1/7 seeds, p = 0.992 — and static template augmentation beat
the adaptive loop on 6 of 7 seeds. On the two anchors
where an apparent +0.76 recall gain did show up, the provenance probe sorted the injected rows from
real traffic at PR-AUC 0.998, so the gain is explained by which generator wrote the row rather than
by anything about fraud. The claim above is what survives.

## The optimiser's objective, stated exactly

The attacker's search is a constrained optimisation, and the constraint structure matters as much
as the objective. Hard gates **veto**; soft objectives are **graded**.

```
maximise    evasion_rate  -  λ · soft_fidelity_penalty          (λ = 0.5, config)

subject to  [hard, veto]  no schema or provenance leak in the batch
                          — self-transfer, duplicate id, unlabelled fraud row,
                            provenance on a legit row, non-positive amount, empty attack
            [hard, veto]  not separable from the real anchor, under BOTH audit rules
                          — `lift`: commensurability score < 3 × base rate
                          — `envelope`: the audit's own `trivially_separable` verdict

where       soft_fidelity_penalty = mean of three graded terms, each measured against the
            anchor's own statistics rather than a guessed constant:
                statistical   amount-precision distance from the anchor's share
                structural    beneficiary-degree concentration above the anchor's
                structural    round-amount share above the anchor's
```

A vetoed candidate scores −1.0, is never trained on, and can never become `best`. **Only
separability and leaks are vetoes.** Making every fidelity metric a hard gate would very likely
leave no feasible region at all on public synthetic anchors — the `lift` rule alone already rejects
100% of candidates on AMLSim and PaySim — so statistical and structural fidelity enter as a
gradient that steers the search rather than a cliff that empties it.

**Which configuration produced which number.** The committed v1.0 result below was produced under
an earlier version of this objective in which the soft penalty was **not binding**: its bounds were
guessed rather than measured, so it reported a near-constant 0.065 and λ had no effect on the search
(`docs/realism-leash.md`). Separability was audited and reported, but not vetoed. The objective as
stated above is the corrected one; any result produced under it is reported separately and is not
the v1.0 artefact.

## The four questions

The project is a success or a failure on four questions, asked in order. Each one gates the next:
question 4 is only meaningful if question 3 passes, which is exactly where most of the evidence
stops.

### 1. Can the attacker find evasions?

**Demonstrated.** On AMLworld, the optimiser's first round put **83.6%** of its generated fraud
past the detector, averaged over 7 seeds. The detector then closes on it round by round, and the
attacker keeps finding fresh gaps at each retrain rather than being solved outright.

"Demonstrated" rather than "yes": this is a mechanism shown to operate, not a measured advantage
over any alternative. Nothing here says the evasions it finds are the ones a real attacker would.

`artifacts/abcd/amlworld_gather-scatter.json`, plotted in `..._convergence.png`.

### 2. Can the detector close known gaps?

**Partially.** The same curve read the other way: evasion falls **0.836 → 0.054** over six rounds,
a 15× reduction, and it falls on **all 7 seeds** — the spread is small next to the drop. The loop
closes, and this is the one place where the system does what the architecture diagram says it does.

Three qualifications keep it at "partially" rather than "yes". The rate **plateaus near 0.20 rather
than reaching zero**, so a fifth of the attacker's traffic still gets through after six retrains.
The gap-closing is measured in aggregate across the vectors in the loop, not per vector, so "which
gaps" is not answered. And every layer built specifically to close a gap the baseline could not —
the anomaly layer, the sequence model, the temporal GNN — **lost to what already ships** and was
benched (`docs/negative-results.md`). What closes gaps here is retraining the existing detector on
admitted attacks, and nothing more exotic than that.

### 3. Do the generated attacks pass the fidelity and provenance gates?

**Mostly no, and that is the finding.** Four instruments ask this from different angles:

| instrument | anchor | verdict |
|---|---|---|
| commensurability audit | amlworld | **pass** — worst field `log_amount` at 0.0002 against a 0.0002 base rate; audit gate rejected 0 of 42 rounds |
| commensurability audit | amlsim, paysim | **fail** — the `lift` rule rejects 100% of candidate batches |
| provenance probe | amlsim, paysim | **fail** — injected rows sort from real traffic at PR-AUC 0.998 and 0.970 |
| fidelity scorecard | amlsim, paysim | **fail** — level-3 utility 0.25 and 0.00 |
| transfer test | amlworld | **fail** — synthetic-trained scores 0.000532 against a 0.001329 amount floor |

Downstream, this is why so little is quotable: **4 of 18 leave-one-attack-out folds** carry a number,
and **2 of 6 columns in the three-system table** (ticket 16, a separate experiment from the
A/B/C/D one below — `docs/results.md` sets the two side by side, and warns that their `C` labels
mean different things). The rest are withheld, in brackets, with the reason attached.

On one anchor of four the attacks are commensurable with the traffic they are injected into. On
none of them does training on synthetic alone beat sorting the test window by amount.

### 4. When they do pass, does adaptive beat non-adaptive?

**No — and unconstrained.** AMLworld is the one anchor where question 3's gate passed, so it
is the one place question 4 can honestly be asked. This is the **A/B/C/D experiment** (ticket 12), whose
`C_template` arm is the static-synthetic control that makes "adaptive" falsifiable. Held out
GATHER-SCATTER, 7 seeds:

| system | PR-AUC (mean ± sd) | recall@1%FPR |
|---|---|---|
| A_real | 0.0806 ± 0.0765 | 0.440 ± 0.185 |
| B_smote | 0.0274 ± 0.0154 | 0.520 ± 0.148 |
| C_template | 0.0557 ± 0.0560 | 0.544 ± 0.255 |
| D_adaptive | 0.0168 ± 0.0131 | 0.378 ± 0.292 |
| amount_floor | 0.0013 | 0.012 |

Per-seed direction, D against each alternative, exact one-sided sign test:

| comparison | PR-AUC | recall@1%FPR |
|---|---|---|
| D > C_template | 1/7, p = 0.992 | 2/7, p = 0.938 |
| D > B_smote | 4/7, p = 0.500 | 2/7, p = 0.938 |
| D > A_real | 1/7, p = 0.992 | 2/7, p = 0.938 |
| D > amount_floor | 7/7, p = 0.008 | 6/7, p = 0.062 |
| **C > D** (the same test, read the other way) | **6/7, p = 0.062** | 5/7, p = 0.227 |

**Adaptive does not beat non-adaptive; it loses to it.** D wins on 1 of 7 seeds by PR-AUC and 2 of 7
by recall. Read the other way, static template augmentation beats the adaptive loop on **6 of 7
seeds, p = 0.062**.

That is **directional, not significant.** 6/7 does not clear the 0.05 line, and every standard
deviation in the table above is comparable to or larger than its own mean, so this fold is
underpowered for an effect of this size — the same limitation that applied when the result read as
a null. What we can say is that the direction is consistent and it is not the direction the project
hoped for. **The one comparison that clears significance is that every system beats the no-model
amount floor on PR-AUC, 7/7, p = 0.008** — which establishes that the fold is not measuring
amount-legibility, and nothing about adaptive.

An earlier 2-seed reading showed adaptive ahead 2/2 and was reported as established. It evaporated
at 7 seeds. That retraction is recorded in `docs/adr/0005-amlworld-anchor-and-the-abcd-null.md`, and
it is the reason the sign test exists in this project at all.

**Unconstrained, in a second sense that survives the result.** The per-round realism leash that was
supposed to stop the optimiser buying evasion with unrealistic traffic **is inert**. It vetoed 0 of
42 rounds. Correcting its inverted bounds — measured off the anchor rather than guessed — made the
penalty vary properly (41 distinct values across 42 rounds instead of a pinned 0.065) and changed
the outcome **not at all**: a control run with the old leash and a run with the corrected one are
bit-identical on the same code. So the leash was never the binding constraint, and this number is a
measurement of an *unconstrained* attacker. The attacker was therefore never actually penalised for unrealism while these numbers
were produced. Nothing suggests it exploited that (the audit gate rejected 0 of 42 rounds, and the
fidelity scorecard was run independently), but the constraint was not enforced, so **D is not a
measurement of "the best attack subject to staying realistic" — it is a measurement of an
unconstrained search.** Making the leash bind is the stated next step, and it would change the
optimiser's search and invalidate this artefact, which is why it is a re-run rather than a patch.
Derivation in `docs/realism-leash.md`.

So question 4 answers **no, directionally, and unconstrained**. Two caveats bound it in both
directions: the fold is underpowered, so 6/7 at p = 0.062 is a direction rather than a proof; and
the attacker was never actually constrained, so this is not evidence about what a realism-bounded
adaptive attacker could do.

**Provenance.** Every number in this section traces to `artifacts/abcd/amlworld_gather-scatter.json`,
regenerated on commit `4050fc46` with a clean tree (`git_dirty: false`). The previous artefact
reported here could not be reproduced from any commit; it is retired in `artifacts/abcd/retired/`
with the evidence, and the stamp that now catches this is described in `afl/utils/provenance.py`.

## What we claim

- A closed adaptive loop that **runs, converges, and is reproducible from committed artefacts** —
  evasion 0.836 → 0.054 on all 7 seeds, regenerable by one command, byte-identical.
- A **commensurability audit** that catches five classes of leak between synthetic and real rows —
  amount scale, rail, device column, time granularity, entity namespace — one of which the audit
  itself initially missed and which is now a regression test.
- A **provenance probe and fidelity gate** that withhold results rather than report them, applied to
  our own headline numbers first.
- A **transfer test** that asks the question the held-out score cannot: does a detector trained on
  our synthetic attacks catch *real* fraud? It does not, and we say so.
- A **four-anchor negative result**, reported with the instruments that produced it, each of which
  was itself validated — the fidelity harness against copy/shuffle/noise generators whose answers
  were known in advance.

## What we do not claim

- **That adaptive augmentation improves recall on a held-out family.** On the one anchor where
  the question was fair to ask, our own 7-seed test puts adaptive ahead of the static-template
  control on 2 of 7 seeds by recall (p = 0.938) and 1 of 7 by PR-AUC (p = 0.992).
- **That the synthetic attacks are realistic enough to train on.** The transfer test says they are
  not, on every anchor tried.
- **That the +0.76 recall gain on amlsim/paysim is real.** The provenance probe explains it, and the
  column is withheld in `docs/three_system.md` rather than quoted.
- **That the loop was verifiably not cheating while it ran.** The per-round realism leash sat
  between 0.0629 and 0.0667 in all 42 rounds and could not have detected cheating; the evidence
  that the attacker stayed honest comes from the audit gate and the fidelity scorecard instead,
  not from the leash. Derivation in `docs/realism-leash.md`.
- **That any of this generalises to production payment traffic.** All four anchors are generated
  datasets. A better generator is not observed fraud.

## Why the negative result is the deliverable

A pipeline that produces a confident number is easy. A pipeline that produces a number **and can
tell you when not to believe it** is the harder artefact, and it is the one a fraud team actually
needs — because the failure mode this repository documents, a model learning which generator wrote
a row rather than what fraud looks like, is the failure mode of every synthetic-augmentation
programme, and it is invisible without exactly these instruments.

Five independent instruments reached the same verdict here for the same underlying reason: on these
anchors, injected rows are distinguishable from real ones, so anything measured on a fold of them
measures provenance rather than fraud. That reasoning is set out in `docs/realism-leash.md`.
