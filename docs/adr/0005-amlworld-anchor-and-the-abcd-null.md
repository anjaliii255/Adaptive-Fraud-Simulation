---
status: accepted
---

# AMLworld as the loop anchor, and the 7-seed A/B/C/D null

Relates to ADR 0002 (dataset anchors) and ADR 0004 (BankSim spike). 0004 said no further dataset
hunting; this records the one anchor we had already agreed was worth finishing, and the result it
produced.

**NO-GO on the adaptive claim.** Adaptive augmentation does not beat static template augmentation
on this fold; it loses to it on 6 of 7 seeds. Directional rather than significant, on an
underpowered fold, and measured on an attacker that was never actually constrained.

**Superseded numbers.** The figures in this ADR were regenerated on 2026-08-28 from commit
`4050fc46` with a clean tree. The artefact originally reported here could not be reproduced from
any commit in this repository and is retired in `artifacts/abcd/retired/`; see
`artifacts/abcd/README.md` for how it was caught and what fixed it. `A_real`, `B_smote` and the
amount floor are unchanged — they never touch the simulator, which is how the defect was localised.

## Context

Three anchors had already failed to validate the loop's held-out claim:

- **PaySim** — no repeated entities. Behaviour cannot be anomalous against a history that does not
  exist.
- **AMLSim** — its own fraud is amount-trivial. Floor 0.455644 against a ceiling of 0.593653,
  ratio 0.7675: no room left to measure anything a model contributes.
- **BankSim** — transfer fails. Synthetic-trained 0.2377 against a 0.7023 amount floor, and that is
  the *second* number, measured after fixing a payee-namespace leak the audit itself initially
  missed.

We ran one more anchor: **AMLworld** (Altman et al., IBM), HI-Small — the one structure not yet
tried. Millions of transactions, repeated entities, and fraud defined by laundering typologies
(fan-in/out, cycles, gather-scatter, stacking) rather than by amount or category.

## The gate results

Both gates below are recomputed by `scripts/spike_gates.py` and committed to
`artifacts/spike/amlworld.json`, against split digest `f5e33a878d68b792`.

- **Gate 1 (hosts behaviour): SPLIT.** `src_*` history coverage **92.93%** (> 50% ✅) but median 2.0
  txns/account (threshold ≥ 10 ❌). The low median is AMLworld's genuine hub-and-spoke shape: 98.22%
  of rows sit on a repeat account, mean 14.15, max 168,672 across 118,053 accounts. Behaviour is
  clearly hosted; the median fails the stated number because of the long tail, which is realistic
  AML structure rather than a defect.

  Coverage here is defined as the share of rows whose sender has at least one strictly earlier row
  in the anchor — the precondition for any `src_*` feature to read anything. **An earlier draft of
  this ADR recorded 54.6%. That number does not reproduce and appears nowhere in the repo**; it was
  read off a terminal whose code was not kept. The committed definition and its 92.93% supersede it.
  The verdict is SPLIT under either figure, and the corrected one makes the point more strongly
  than the number it replaces.
- **Gate 2 (own fraud non-trivial): PASS, decisively.** Amount-only floor 0.004169 against a trained
  ceiling of 0.178768, ratio **0.0233** — against a threshold of ≤ 0.6, where AMLSim scored 0.7675 and
  BankSim 0.7359 (both derived, see Provenance). Measured on the anchor's own fraud in the test window, 623 positives. Finding:
  **the IBM generator family is not uniformly amount-legible** — the small sibling was, the
  typology-rich large set is not.

  On where the lift comes from: the model's six most-used features are `amount`, `hour_of_day`,
  `dst_account_age_s`, `src_account_age_s`, `src_amount_z`, `src_amount_ratio_to_mean`. **Amount is
  the most-split feature, and amount alone scores 0.004** — so the gain is not a richer amount
  distribution but amount read against relational context, which is a different and stronger claim
  than "the lift is relational". An earlier draft listed `src_seconds_since_last_out` among the top
  features; it is not in the top six, and that list is corrected here.
- **Gate 3 (transfer / does adaptive help): NO.** See the 7-seed result below. The standalone
  transfer test agrees: trained on synthetic alone, PR-AUC 0.000532 on real GATHER-SCATTER against
  an amount floor of 0.001329 — synthetic-only loses to sorting by amount
  (`artifacts/transfer/amlworld.json`).

**Decision on the gate rule.** Gates 1 and 3 fail their literal thresholds, so the mechanical rule
returns NO-GO. We record that, and also that the thresholds tested a stricter question ("synthetic
replaces real labels") than the claim the project was pursuing *at the time this spike ran*
("adaptive augmentation improves a detector"). That claim has since been withdrawn — the 7-seed
result below is what withdrew it, and `docs/claim.md` states what replaced it. AMLworld was taken forward as the loop anchor because Gate 2 — the gate that killed the
whole IBM family's small sibling and every prior anchor — passed cleanly, and for the right
relational reason.

## The experiment

Multi-vector adaptive optimiser (search over S1/S2/S3 knobs plus budget allocation) run as an
A/B/C/D comparison. GATHER-SCATTER held out entirely, out-of-time split digest `f5e33a878d68b792`,
173 positives, base rate 0.0532%, 6 rounds, 7 seeds (7, 11, 23, 42, 101, 1337, 2024). Every number below traces to `artifacts/abcd/amlworld_gather-scatter.json`, whose header carries
`git_commit: 4050fc46…` and `git_dirty: false` beside the split digest.

| system | PR-AUC (mean ± sd) | recall@1%FPR (mean ± sd) |
|---|---|---|
| A_real | 0.0806 ± 0.0709 | 0.440 ± 0.172 |
| B_smote | 0.0274 ± 0.0143 | 0.520 ± 0.137 |
| C_template | 0.0557 ± 0.0518 | 0.544 ± 0.236 |
| D_adaptive | 0.0168 ± 0.0121 | 0.378 ± 0.270 |
| amount_floor | 0.0013 | 0.012 |

Sign tests on per-seed direction:

- **D > C: PR-AUC 1/7 (p = 0.992), recall 2/7 (p = 0.938)** — the comparison the ticket exists to
  answer, and adaptive loses it. Read the other way, **C > D on 6/7 (p = 0.062)** by PR-AUC.
- D > A: PR-AUC 1/7 (p = 0.992), recall 2/7 (p = 0.938).
- C > A: PR-AUC 3/7 (p = 0.773), recall 4/7 (p = 0.500).
- Every system beats the amount floor on PR-AUC: 7/7 (p = 0.008), all four. On recall it is 7/7 for
  A and B, and **6/7 (p = 0.062) for C and D** — stated separately because rounding it up to a
  blanket 7/7 would be the same species of error this ADR retracts below.

## Decision

**NO-GO on the adaptive claim**, recorded as a definitive negative on an underpowered fold rather
than a disproof.

Every standard deviation is comparable to or larger than its own mean; seed variance exceeds every
between-system gap. At 173 positives and a 0.053% base rate this fold cannot resolve differences of
this size, and **more seeds will not fix a power problem.**

An earlier 2-seed reading showed D > A on 2/2 and C > A on one seed, and was reported as
established. Both evaporated at 7 seeds. **That number was seed noise and is retracted here.** It is
recorded rather than deleted because it is the clearest evidence in the project for why the sign
test was added at all.

The only comparison that clears significance is that every system beats the no-model amount floor,
which establishes that the fold is not measuring amount-legibility.

**And the attacker was unconstrained.** The per-round realism leash vetoed 0 of 42 rounds. Its
bounds were inverted — guessed at 0.6 where every anchor measures ~0.99 — and correcting them made
the penalty vary properly while changing the outcome not at all: a control run under the old leash
and a run under the corrected one are bit-identical on the same code. Nothing here is evidence about
a realism-bounded adaptive attacker. `docs/realism-leash.md`.

## Consequences, and what holds

- **The adaptive loop runs and converges.** Evasion falls 0.915 → 0.201 across six rounds on all 7
  seeds — the one curve in this project that moves the way the architecture doc predicted. Held-out
  recall stays flat and noisy beside it, which is this null in a picture rather than a table
  (`artifacts/abcd/amlworld_gather-scatter_convergence.png`, ticket 19).
- **The commensurability audit rejected 0 of 42 rounds.** The search stayed on-anchor unforced, so
  the null is not an artefact of broken or leaky synthesis. This is the load-bearing claim: a null
  from a generator that was quietly producing separable rows would mean nothing.
- **Guardrails held.** No HI/LI-Medium variants — out of compute scope, and the honest answer to
  "underpowered" is more seeds, not a bigger dataset. No pooled typologies — pooling changes the
  question from "unseen shape" to "unseen instance", which is a weaker claim dressed as a stronger
  one.
- **This is the fourth public anchor to fail the claim**, now joined by the fidelity gate (both
  anchors fail on level-3 utility), the sequence gate (did not earn its seat) and the GNN gate (fell
  back to hand-rolled graph features). Five independent instruments, one verdict. The submission
  ships that convergent null together with the instruments that produced it — each of which was
  itself audited: the payee-namespace bug found here, and the fidelity self-test run against
  copy/shuffle/noise generators whose answers were known in advance.

## Provenance

Every number in this ADR now traces to a committed artefact:

| section | artefact | regenerated by |
|---|---|---|
| Gates 1 and 2 | `artifacts/spike/amlworld.json` | `python scripts/spike_gates.py --data amlworld` |
| Gate 3, commensurability | `artifacts/transfer/amlworld.json` | `python scripts/transfer_test.py --data amlworld` |
| the A/B/C/D table and sign tests | `artifacts/abcd/amlworld_gather-scatter.json` | `python scripts/abcd_experiment.py` |
| the convergence figure | the same file | `make figures` |

**What re-running found, and why it is recorded here rather than quietly fixed.** Gate 2 reproduced
to four decimal places, which is the number this anchor was chosen on. Gate 1 did not: its median,
mean, max and repeat-share all matched, and its headline coverage figure did not — 92.93% measured
against 54.6% written down. The discrepancy was invisible for as long as the number lived only in a
terminal, and that is the argument for the artefact, not an argument against the gate.

The comparison ratios quoted for AMLSim (0.7675) and BankSim (0.7359) were first written in ADR 0002
and ADR 0004, before `spike_gates.py` existed. They have since been checked and **do trace to
committed artefacts**: both are `amount_floor.real_fraud.pr_auc ÷ real.real_fraud.pr_auc` in
`artifacts/transfer/<anchor>.json` — AMLSim 0.455644 / 0.593653, BankSim 0.702308 / 0.954362. They
are cited as context for AMLworld's 0.0233 rather than as measurements under the same gate
definition, and should not be compared to it at the fourth decimal; `spike_gates.py` has not been
run on those two anchors.
