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

## BankSim: on disk, unwired, parked

BankSim is downloaded (594,644 rows; merchant, category, age, gender) and deliberately not wired.
It is well shaped for S2 card testing — the merchant and category fields are exactly what PaySim
lacks — but a third anchor mid-sprint is scope creep. It becomes the S2 transfer check if the loop
lands early. Named as a cut, not a gap.

## IEEE-CIS: removed

`config/data/ieee_cis.yaml` pointed at files nobody downloaded and was referenced by nothing.
PaySim + AMLSim is the two-dataset story, and BankSim now fills the "if time allows" slot it used
to hold. `afl/data/loaders.py` still carries a `load_ieee_cis` function; that is ticket 02's file
and gets removed there, not here.

## The base rate is the trap

Both real anchors sit at **~0.13% fraud**. The synthetic default runs at **~17%** — roughly
**130x**, two orders of magnitude.

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
