# Data card — amlworld

_Measured from the file on disk, not copied from the source's documentation. The split was
committed by hand rather than by `scripts/build_splits.py`: AMLworld ships real timestamps
instead of the integer step index that script assumes._

## What it is

The fourth and last anchor. Interbank transfers with laundering labelled by **typology**, which
is why it is here: a held-out fold can be a real unseen laundering shape rather than a synthetic
stand-in.

| | |
| --- | --- |
| source | https://www.kaggle.com/datasets/ealaxi/ibm-transactions-for-anti-money-laundering-aml |
| paper | Altman et al., Realistic Synthetic Financial Transactions for AML Models (IBM) |
| licence | CC BY-NC-SA 4.0 |
| files | `data/raw/IBM AML Graphs/` |
| download | manual — not committed (`data/**` is gitignored) |
| rows (raw) | 5,078,345 |
| rows (loaded) | 1,670,437 after the currency and self-transfer filters |
| laundering (raw) | 5,177 |
| laundering (loaded) | 1,912 |
| **base rate** | **0.11446%** |
| time span | 2022-09-01 to 2022-09-18 (18 days, real timestamps) |
| first / last | 2022-09-01 00:00:00 → 2022-09-18 16:18:00 |

## The committed split

| | |
| --- | --- |
| train ends | 2022-09-07T22:43:00 |
| test starts | 2022-09-08T22:43:00 |
| embargo | 86400s |
| digest | `f5e33a878d68b792` |
| train | 1,169,395 rows, 1,108 laundering |
| test | 325,350 rows, 623 laundering |

Velocity features look back up to seven days and the span is only eighteen, so the embargo is kept to one day: enough to stop a test row's window being computed over training rows without spending a week of an eighteen-day file on the gap.

## Base rate against the synthetic default

The synthetic default runs at 4.0313%; this anchor is 0.0284 of it.
That is more than an order of magnitude apart, so it has to be said out loud. `recall@1%FPR` and `precision@100` are not the
same measurement at 0.11% as at 4%, and no operating point carries across the gap.

## Two filters, both about making `amount` mean one thing

Amounts are quoted in the payment currency, so a mixed frame puts millions of US Dollars beside
thousands of Euros on one axis and every amount comparison after that is meaningless. The loader
keeps **US Dollar**, the largest share at 37% of rows.

Separately, 11.6% of raw rows are an account paying itself — almost all `Reinvestment`, and only
0.21% of laundering rows. They are self-loops rather than payments between entities, and they
would fill the graph features with edges to nowhere, so they are dropped. The two filters together
take 5,078,345 raw rows to 1,670,437 loaded.

## Typologies

370 labelled laundering attempts over 8 typologies, 3,209 rows, every one joining verbatim to a
transaction row. GATHER-SCATTER, SCATTER-GATHER, STACK, FAN-OUT, FAN-IN, CYCLE, BIPARTITE, RANDOM.

After the currency and self-transfer filters, 1,178 typology-labelled rows survive. The
best-powered held-out fold is **GATHER-SCATTER** with 173 positives in the test window.

## Quirks

- Two columns named `Account`; the receiver is the second one.
- Amounts are multi-currency; the loader keeps US Dollar so amount is one scale.
- 11.6% of rows are an account paying itself, almost all Reinvestment.
- Only 3209 of 5177 laundering rows carry a typology; the rest are unlabelled laundering.

## What it cannot tell us

- Nothing about card or UPI behaviour; it is interbank transfer traffic.
- It is still generated data, not observed traffic — a better generator, not real fraud.
