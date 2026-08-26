# Fidelity scorecard — paysim

**Verdict: ❌ FAIL** — score 0.0

| level | what it asks | score |
| --- | --- | --- |
| level1 | do the marginals and joints match? | 0.7495 |
| level2 | is the structure and pacing right? | 0.2054 |
| level3 | does it teach a model anything? (**the bar**) | 0.0 |
| privacy | is it copying rows? (evidence, not proof) | 0.7763 |

The gate is **level 3**: TSTR gap 0.153009 exceeds the bar 0.15; held-out recall lift -0.229268 below 0.0 — the data does not help; TSTR PR-AUC 0.004856 loses to the amount floor 0.057261 — training on this data is worse than sorting the test window by amount; membership is inferable from the synthetic data alone

## Why

- TSTR gap 0.153009 exceeds the bar 0.15
- held-out recall lift -0.229268 below 0.0 — the data does not help
- TSTR PR-AUC 0.004856 loses to the amount floor 0.057261 — training on this data is worse than sorting the test window by amount
- membership is inferable from the synthetic data alone
- level 2 score 0.2054 below 0.6 (worst: fan_in_share)
- the generator's standalone output trains a detector to 0.003195, below the amount floor 0.057261 — reported, not gating

## The numbers that matter

One real test window (150766 rows, 410 of them fraud), one operating point, four systems.

| system | trained on | rows | PR-AUC | recall@FPR | p@k | beats the floor |
| --- | --- | --- | --- | --- | --- | --- |
| trtr | real rows, real labels | 446214 | 0.157865 | 0.443902 | 0.53 | yes |
| tstr | real legit + generated fraud, no real fraud label | 446480 | 0.004856 | 0.02439 | 0.0 | **no** |
| augmented | real rows + generated fraud | 446849 | 0.047476 | 0.214634 | 0.14 | **no** |
| amount_floor | nothing | 0 | 0.057261 | 0.212195 | 0.23 | — |
| standalone | the generator's whole output, background included | 19496 | 0.003195 | 0.0 | 0.0 | **no** |

- TSTR gap 0.153009 against a bar of ≤ 0.15 (ratio 0.03076 of what the real labels reach)
- held-out recall lift from augmentation: -0.229268 (bar ≥ 0.0)
- the floor ranks by amount alone, largest amount first, direction chosen on train

## Privacy — evidence, not proof

- DCR ratio 2.431659 (bar ≥ 0.8): synthetic rows sit 0.337379 from their closest training row, where training rows sit 0.138744 from each other
- exact duplicates of training rows: 0.0 of synthetic rows
- membership-inference AUC 0.388137, advantage 0.223726 (bar ≤ 0.2), against 0.079803 for the same attack run between two halves of the holdout, where nothing was ever in training — the difference is the part that is about membership
- generated rows naming an account that exists in the anchor: 1.0 (src 0.87874, dst 0.984252) — by design, and reported rather than flagged

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