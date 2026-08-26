# Fidelity scorecard — amlsim

**Verdict: ❌ FAIL** — score 0.25

| level | what it asks | score |
| --- | --- | --- |
| level1 | do the marginals and joints match? | 0.7672 |
| level2 | is the structure and pacing right? | 0.228 |
| level3 | does it teach a model anything? (**the bar**) | 0.25 |
| privacy | is it copying rows? (evidence, not proof) | 0.6488 |

The gate is **level 3**: TSTR gap 0.404137 exceeds the bar 0.15; membership is inferable from the synthetic data alone

## Why

- TSTR gap 0.404137 exceeds the bar 0.15
- membership is inferable from the synthetic data alone
- level 2 score 0.228 below 0.6 (worst: reciprocity)
- the generator's standalone output trains a detector to 0.000897, below the amount floor 0.455644 — reported, not gating

## The numbers that matter

One real test window (385909 rows, 544 of them fraud), one operating point, four systems.

| system | trained on | rows | PR-AUC | recall@FPR | p@k | beats the floor |
| --- | --- | --- | --- | --- | --- | --- |
| trtr | real rows, real labels | 930465 | 1.0 | 1.0 | 1.0 | yes |
| tstr | real legit + generated fraud, no real fraud label | 929930 | 0.595863 | 0.630515 | 0.96 | yes |
| augmented | real rows + generated fraud | 931100 | 0.991541 | 1.0 | 0.99 | yes |
| amount_floor | nothing | 0 | 0.455644 | 0.474265 | 1.0 | — |
| standalone | the generator's whole output, background included | 19496 | 0.000897 | 0.0 | 0.0 | **no** |

- TSTR gap 0.404137 against a bar of ≤ 0.15 (ratio 0.595863 of what the real labels reach)
- held-out recall lift from augmentation: 0.0 (bar ≥ 0.0)
- the floor ranks by amount alone, smallest amount first, direction chosen on train

## Privacy — evidence, not proof

- DCR ratio 3.453889 (bar ≥ 0.8): synthetic rows sit 0.519459 from their closest training row, where training rows sit 0.150398 from each other
- exact duplicates of training rows: 0.0 of synthetic rows
- membership-inference AUC 0.324405, advantage 0.35119 (bar ≤ 0.2), against 0.343365 for the same attack run between two halves of the holdout, where nothing was ever in training — the difference is the part that is about membership
- generated rows naming an account that exists in the anchor: 1.0 (src 0.937008, dst 1.0) — by design, and reported rather than flagged

Neither number is a privacy guarantee. They are two ways of catching a generator that learned the distribution by copying it; passing them means the memorisation we tested for is not there, and nothing more. A formal claim needs differential privacy, and this project does not make one.

## Did the bars predate the numbers?

✅ the bars were committed in 6989a9ef9848 on 2026-08-22 and every value still matches that commit; this run started 2026-08-26T11:43:22+00:00

- thresholds: `config/fidelity/thresholds.yaml` (sha256 de751b7998cee773)
- origin commit `6989a9ef9848` (2026-08-23T01:26:05+05:30), read back out of git and compared value by value
- inherited unchanged from it: level1_min, level2_min, max_mia_advantage, max_tstr_gap, min_dcr_ratio, min_recall_lift
- introduced later: require_tstr_beats_amount_floor
- working copy clean: True

Every commit that has ever changed a bar:

- `c55dc089e6e5` 2026-08-26 — Ticket 15: the fidelity bars, committed before the numbers they judge (config/fidelity/thresholds.yaml)
- `6989a9ef9848` 2026-08-23 — Skeleton: closed-loop fraud sim + detector, fidelity harness, eval, demo (config/config.yaml)