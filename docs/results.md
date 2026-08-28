# Results — what was measured, and what was withheld

_The two comparison experiments, and the reason most of the apparent gains are in brackets.
`docs/claim.md` states what all of this adds up to._

## Two experiments, and the labels do not mean the same thing

This repository contains **two separate comparison experiments**. They are not versions of each
other, neither supersedes the other, and their system labels collide. Read this before either table.

| | **A/B/C/D** (ticket 12) | **three-system table** (ticket 16) |
| --- | --- | --- |
| artefact | `artifacts/abcd/` | `artifacts/three_system/` |
| anchor | AMLworld | PaySim and AMLSim |
| held out | GATHER-SCATTER, a **real** laundering typology | M3, an **injected synthetic** family |
| seeds | 7 | 3 |
| arms | A_real, B_smote, **C_template**, **D_adaptive**, amount_floor | A_baseline, B_smote, **C_adaptive** |

**The collision to watch: `C` is template-static in the first and adaptive in the second.** So
"C beats B by +0.76 recall" (three-system) and "C beats D on 6/7 seeds" (A/B/C/D) are statements
about different arms, and neither contradicts the other. The A/B/C/D experiment has a
template-static control that the three-system table does not; that control is the whole point of
ticket 12, because it separates *any synthetic augmentation* from *adaptive* synthetic augmentation.

**Which one carries the claim.** The A/B/C/D experiment is the one `docs/claim.md` answers question
4 with: it is on a real held-out attack shape, at 7 seeds, with the template-static control that
makes "adaptive" falsifiable. The three-system table is on injected synthetic holdouts at 3 seeds,
and its headline column is withheld on both anchors.

## The A/B/C/D experiment

`scripts/abcd_experiment.py` runs four systems plus a no-model floor on AMLworld, holding out the
GATHER-SCATTER laundering typology entirely — a real attack shape no system trained on, rather than
a synthetic family injected for the purpose. Split digest `f5e33a878d68b792`, 173 positives in the
fold, base rate 0.053%, 6 rounds, 7 seeds.

| system | trained on | PR-AUC (mean ± sd) | recall@1%FPR |
| --- | --- | --- | --- |
| A_real | the anchor's real rows and labels | 0.0806 ± 0.0709 | 0.440 ± 0.172 |
| B_smote | the same plus row-level oversampling | 0.0274 ± 0.0143 | 0.520 ± 0.137 |
| C_template | the same plus **static** template attacks | 0.0557 ± 0.0518 | 0.544 ± 0.236 |
| D_adaptive | the same plus the **adaptive** loop's attacks | 0.0168 ± 0.0121 | 0.378 ± 0.270 |
| amount_floor | nothing — rank by amount | 0.0013 | 0.012 |

**C_template is what makes D falsifiable.** C and D share an episode budget, so the only difference
between them is whether the attacks were searched adaptively or generated statically. If D does not
beat C, the adaptive search bought nothing that plain synthetic augmentation would not have.

Per-seed direction, exact one-sided sign test:

| comparison | PR-AUC | recall@1%FPR |
|---|---|---|
| D > C_template | 1/7, p = 0.992 | 2/7, p = 0.938 |
| D > B_smote | 4/7, p = 0.500 | 2/7, p = 0.938 |
| D > A_real | 1/7, p = 0.992 | 2/7, p = 0.938 |
| D > amount_floor | 7/7, p = 0.008 | 6/7, p = 0.062 |
| **C > D**, the same test read the other way | **6/7, p = 0.062** | 5/7, p = 0.227 |

**Adaptive loses to static template augmentation** — D wins 1 of 7 seeds on PR-AUC, and C wins 6 of
7 (p = 0.062). Directional, not significant: 6/7 does not clear 0.05, and every standard deviation
above is comparable to or larger than its own mean, so the fold is underpowered for an effect this
size. The one comparison that clears significance is that every system beats the amount floor on
PR-AUC, which says the fold is not measuring amount-legibility and nothing about adaptive.

The loop itself converged: evasion falls 0.836 → 0.054 over six rounds, and the audit gate rejected
0 of 42 rounds, so the result is not an artefact of leaky synthesis. The comparison is also
**unconstrained** — the realism leash is inert, vetoing 0 of 42 rounds, and correcting its bounds
changes the outcome not at all (`docs/realism-leash.md`).

**Provenance.** These numbers come from `artifacts/abcd/amlworld_gather-scatter.json`, regenerated
on commit `4050fc46` with a clean tree — the header carries `git_commit` and `git_dirty: false`
beside the split digest. An earlier artefact reported here could not be reproduced from any commit
in the repository; it is retired in `artifacts/abcd/retired/` and the defect is written up in
`artifacts/abcd/README.md`.

Full reasoning, including the retraction of an earlier 2-seed reading, is in
`docs/adr/0005-amlworld-anchor-and-the-abcd-null.md`.

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


## Current numbers

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

