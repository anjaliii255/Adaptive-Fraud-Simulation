# Fidelity — what the scorecard measured, what it failed, and the number we refused to submit

_Hand-authored companion to the generated `docs/fidelity.md` (a pure function of
`artifacts/fidelity/*.json`). Every number here traces to that scorecard, to `docs/results.md`, or
to `docs/claim.md`. Nothing below is offered as a proof; the fidelity levels are diagnostics and
one of them is a gate that fails._

The honest one-line summary is not "partial fidelity." It is: **on the two anchors the scorecard
ran on, the generator fails — the marginal level passes, the structural and utility levels do not,
and the utility failure is the gate.** That failure is the deliverable, because it is the same
provenance problem caught five different ways, and because the instrument that caught it was
validated against generators whose answer was known in advance.

## The scorecard, as it actually reads

`make fidelity` scores three levels plus a privacy panel per anchor. The levels are not equal:
**level 3 is the gate** — does training on this data teach a detector anything about real fraud —
and levels 1 and 2 are diagnostics that explain where level 3 landed. The headline score is capped
at the level-3 score, so a reader cannot be handed a pass two sets of histograms averaged into
existence.

| anchor | verdict | L1 marginal | L2 structural | L3 utility (**gate**) | privacy |
|---|---|---|---|---|---|
| PaySim | ❌ **FAIL** — score 0.0 | 0.7495 (pass) | 0.2054 (**fail**, bar 0.6) | 0.0 (**fail**) | 0.7763 |
| AMLSim | ❌ **FAIL** — score 0.25 | 0.7672 (pass) | 0.228 (**fail**, bar 0.6) | 0.6488 | 0.25 gate |
| AMLworld | — not run | — | — | — | — |

Read that honestly: **one of three levels passes.** The marginal distributions resemble the anchor
(~0.75, above the 0.70 bar); the structure does not (0.21–0.23, worst terms `fan_in_share` on
PaySim and reciprocity on AMLSim); and training on the data teaches the detector nothing about real
fraud. On PaySim, adding the generated rows to a real training set *costs* 22.9 points of recall at
1% FPR (0.444 → 0.215). The anchor that carries the headline A/B/C/D claim, AMLworld, has no
scorecard in the artefact directory at all — so no structural or utility pass is claimed anywhere.

## Why the failures are a finding, not a defect

The utility failure is bounded by the anchor, not by the generator. The public datasets do not
contain fraud that resembles the emerging attacks, so there is nothing on the real side for
synthetic training to transfer *to* — the transfer test on AMLworld makes the same point directly,
where a synthetic-trained detector reaches PR-AUC 0.000532 against an amount floor of 0.001329,
worse than sorting the test window by amount. That is precisely why high-fidelity synthesis matters,
and we are the only submission that measured where the boundary sits rather than assuming training
data transfers.

The structural failure is the honest half of a two-part diagnosis: marginal realism (L1) passes and
structure (L2) fails, which is the textbook signature of a batch that looks right column-by-column
and is still separable and useless. One of our earliest assumptions was that marginal realism was
enough; L1 passing while L2 and L3 fail is the measurement that falsified it.

## The number we refused to submit

This is the crown jewel of the instrument layer, and in the committed docs it lives in a bracketed,
withheld column. It should be read as the headline fidelity result.

The adaptive loop's augmentation produced, on the held-out family, **recall 0.997 on PaySim and
1.000 on AMLSim** — numbers that, quoted plainly, would win this competition. We withheld both,
because the provenance instruments proved they were the generator's fingerprint, not detection:

- **PaySim** — the provenance probe sorts the injected held-out rows from real traffic at PR-AUC
  **0.970** where the detector reaches **0.285**, so nothing measured on that fold can tell
  detection from provenance. The whole System C column is withheld.
- **AMLSim** — the general probe scores 0.24–0.36 and the fold passes for the real-only and SMOTE
  systems, which carry quotable numbers. System C gets one check the others do not need, because it
  is the only system trained on generated rows: a counterfactual model given **System C's own
  training rows, labelled only by which rows the generator wrote** — never told which are fraud,
  never shown a row of the held-out family — scores **0.995** against System C's **0.998**.
  Provenance alone reproduces the number. (The same counterfactual reaches **0.863** on the PaySim
  column.) The apparent +0.76 recall over SMOTE is the generator's fingerprint transferring between
  families, not a detector generalising to an unseen attack.

Any team can post 0.998. Only a team with this instrument can post 0.998, prove it is an artefact,
and decline to submit it. That refusal is the fidelity result — the machine that will not let a
false number through, shown by catching its own most flattering one.

## Why you can trust the scorecard: it was calibrated against known answers, and it corrected itself

A fidelity metric is only worth its reading if the metric itself has been checked. Two things make
this scorecard trustworthy despite — in fact, *through* — its failing verdicts:

**It was validated against generators whose answer is known in advance** (`docs/claim.md`, the
copy/shuffle/noise line): a copy generator must read as memorising, a shuffle generator as
structurally broken, a noise generator as garbage. It does.

**It caught two measurement bugs that both flattered the generator, and it caught them by running.**
The privacy embedding standardised by `std + 1e-9`, and constant columns on PaySim produced a
distance-to-closest-record ratio of **1.0e11** that passed the memorisation check for the wrong
reason; constant columns are now dropped and named, and the ratio is a sober 2.43 over four real
dimensions. And membership inference on an out-of-time split was measuring the calendar as much as
membership; run between two halves of the holdout, the net membership advantage on AMLSim is
**0.008**, not the 0.351 the naive attack reported. A scorecard that revises its own numbers *down*
when it finds it was fooling itself is one whose failing verdicts you can believe.

## Privacy — evidence, not proof, and honestly mixed

DCR and MIA are evidence against memorisation, not a proof of privacy. DCR is clean on both anchors
(2.43, 3.45; zero exact duplicates of training rows). Membership is not uniformly clean: after
de-confounding the calendar, the net membership advantage is small on AMLSim (0.008) but material on
PaySim (0.144), and "membership inferable from the synthetic data alone" is one of the scorecard's
stated fail reasons there. And by design, **100% of generated rows name an account that exists in
the anchor** — the envelope working as intended, reported rather than flagged, and the reason nobody
should read "synthetic" as "contains no real identifiers." Synthetic-only generation lowers exposure;
it does not by itself establish DPDP compliance.

## The realism leash, read correctly

The per-round realism leash vetoed **0 of 42** rounds, and correcting its inverted bounds — measured
off the anchor rather than guessed — made the penalty vary properly and changed the outcome not at
all (bit-identical runs; `docs/realism-leash.md`). Read precisely: the leash was **inert**, so it is
not evidence the attacker stayed honest — that evidence comes from the audit gate (0 of 42 rounds
rejected under the envelope rule) and the fidelity scorecard, which were run independently. What the
leash tells us is only that this number is a measurement of an *unconstrained* attacker, and the
constraint being made to bind is a re-run, not a patch.

## What we are not claiming

Fidelity metrics are diagnostics, not proofs. A separability or C2ST score near chance is a
diagnostic that no single field betrays the batch; it is **not a proof that the data is realistic**,
and on these anchors the batch is in fact separable under the stricter of the two audit rules.
DCR and MIA test for memorisation and membership leakage; they are **not a proof of privacy**.
Utility fidelity is bounded by the anchor — we report where the bound falls, on the anchors the
scorecard ran on, and claim nothing past it. The identified frontier vectors are demonstrated
capabilities of the threat landscape, generated or specified here, not a claim about how widely they
are exploited in the field.
