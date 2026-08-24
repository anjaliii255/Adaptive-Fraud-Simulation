---
status: accepted
---

# Two real anchors, one parked, and a base rate that breaks comparability

`docs/adr/0001` froze what we generate. This one fixes what we generate *against*. Four datasets
were on the table; two are wired, one is parked, one is gone.

## The two anchors

**PaySim** is the behavioural anchor: 6,362,620 rows over 743 hourly steps, 8,213 fraud
(**0.129%**), all of it in `TRANSFER` or `CASH_OUT`. It carries the account-drain arc the velocity
and drift engines imitate.

**AMLSim** (the IBM example dump) is the mule-graph anchor: 1,323,234 rows over 200 daily steps,
1,719 SAR rows (**0.130%**). Its value is not the rows but the topology and the typology labels,
which is what makes a leave-one-*variant*-out test on S1 possible at all.

## What the AMLSim dump actually contains

The config's original column names were guesses and all of them were wrong. Verified against the
header row: `TX_ID`, `SENDER_ACCOUNT_ID`, `RECEIVER_ACCOUNT_ID`, `TX_AMOUNT`, `TIMESTAMP`,
`IS_FRAUD`, `ALERT_ID`, `TX_TYPE`.

Two findings changed the design rather than just the spelling:

**The typology lives in `alerts.csv`, not `transactions.csv`.** Transactions carry `ALERT_ID`;
`ALERT_TYPE` is one join away. The original config named `accounts.csv` as the join file, which
would have produced no typologies at all.

**Only two typologies exist in this dump: `fan_in` (783) and `cycle` (936).** The intended
variant split trained on three and held out two; three of those five are not in the data. So the
split collapses to **train `fan_in`, hold out `cycle`** — one against one. That is still a real
leave-one-variant-out on the strong vector, and it is easier to state than a 3-vs-2. We fit the
data rather than fighting it to preserve the original list.

## BankSim: not on disk, parked

BankSim is **not** downloaded — `data/raw/` holds PaySim and the IBM AMLSim dump and nothing else.
An earlier revision of this ADR said it was; that was wrong and is corrected here.

It stays the named "if time allows" slot. It is well shaped for S2 card testing — the merchant and
category fields are exactly what PaySim lacks — but a third anchor mid-sprint is scope creep.
Named as a cut, not a gap.

## IEEE-CIS: removed

`config/data/ieee_cis.yaml` pointed at files nobody downloaded and was referenced by nothing.
PaySim + AMLSim is the two-dataset story, and BankSim now fills the "if time allows" slot it used
to hold. `load_ieee_cis` was removed from `afl/data/loaders.py` in ticket 02, as planned.

## The base rate is the trap

Both real anchors sit at **~0.13% fraud** (PaySim 0.12908%, AMLSim 0.12991%). The synthetic
default runs at **4.74%** — roughly **37x**. (An earlier revision of this ADR said ~17% and 130x;
both were wrong, and `scripts/build_splits.py` now measures the figure on every run rather than
quoting it. See the amendment below.)

`recall@1%FPR` and `precision@100` are not the same measurement at those two rates. A 1% FPR on
6.3M rows is 63,000 false positives against 8,213 real fraud rows, so precision@k collapses in a
way it never does on synthetic traffic. This is a comparability trap, not a config nit:

- Operating points must be defined at the **real** base rate. That is ticket 11's constraint, and
  B needs it before the eval harness hardens.
- The moment real numbers exist, the synthetic figures in the README are pulled or quarantined
  under an explicit "pipeline check, not comparable" heading. Two number regimes never share a
  table.
- Pre-freeze, post-freeze and post-anchor numbers are three separate regimes. Each supersedes the
  last rather than sitting beside it.

---

# Amendment (ticket 02): what the files actually said

Wiring the loaders meant reading the files rather than their headers, and four things in the
text above turned out to be wrong. Each is corrected in place in `config/data/*.yaml`; they are
listed here because the reasoning changed, not just the spelling.

**The synthetic base rate is not ~17%.** Measured now, from the code as it stands: the pool
`scripts/run_experiment.py` actually builds on `data=synthetic` runs at **4.74%**, and
`config/data/synthetic.yaml` on its own terms at **1.41%**. Against PaySim's 0.129% that is
**37x**, not 130x. `scripts/build_splits.py` measures it on every run instead of quoting it, so
this number cannot go stale again. The conclusion is unchanged and is the only part that
mattered: more than an order of magnitude, so no operating point carries across.

**A step-fraction split is not a row-fraction split.** PaySim's traffic is violently
front-loaded — 341 of its 743 steps carry under 100 rows while step 19 alone carries 51,352 — so
the original `train_end_step: 500` put **95.3%** of the rows in train and left a 4% test tail.
The boundary is now derived from the *row* quantile: step 323 for PaySim (70.2% / 23.7% after the
embargo), step 140 for AMLSim (70.3% / 29.2%). A split described as 70/30 has to be 70/30 in the
thing being counted.

**The typology join key is `TX_ID`, not `ALERT_ID`.** `alerts.csv` has 1,719 rows sharing only
**391 distinct `ALERT_ID` values**, so joining on it needs a de-duplication step; `TX_ID` is
unique in both files and covers all 1,719 fraud rows and no legit row. The typology is exposed as
a `txn_id → type` map by `loaders.amlsim_typologies()` and is deliberately **not** written onto
`Transaction`: a real row carrying a `vector_id` has gained a label path, and the family carve-out
would hold it out as though it were synthetic.

**PaySim has no sender history.** `nameOrig` is effectively unique per row — 6,353,307 distinct
origins over 6,362,620 rows, mean 1.001, max 3. Every `src`-side velocity, RFM and recency feature
is structurally empty on this anchor. The only entity with a past is `nameDest` (mean 2.34, max
113). This is the single most consequential finding for ticket 07: the feature set has to be built
on the beneficiary side, or it is built on nothing. It is also why the run-time sample is taken
over `nameDest` — sampling whole beneficiaries preserves the only history there is.

## Two more the split surfaced

**The out-of-time cut lands on two different base rates.** On PaySim, fraud is **3.5x denser** in
the test half (0.289%) than in the train half (0.082%), because the label is spread near-uniformly
across steps while the legit volume is not. Any chronological boundary on this anchor does that.
A threshold calibrated on a held-out slice of train does not transfer unchanged to test, and a
recall figure has to name which side it was measured on. Ticket 08's operating point inherits this.

**Real traffic breaks a realism rule we enforce.** AMLSim contains **181 self-transfers**
(sender == receiver) and 19 zero-amount rows; PaySim contains 16 zero-amount rows and no
self-transfers. `afl/attack/realism.py` penalises the generator for emitting a self-transfer. On
this anchor that rule is a modelling choice, not a fact about payments — ticket 14 derives its
empirical bounds from these files and has to decide which.
