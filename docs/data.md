# Data — the anchors, and the limitation they share

Two real anchors, neither committed (`data/**` is gitignored) and neither fetched for you:

| | PaySim | AMLSim (IBM example dump) |
| --- | --- | --- |
| what it anchors | behaviour: velocity, drift, the account-drain arc | the mule graph: layering topology and typology labels |
| rows | 6,362,620 | 1,323,234 |
| fraud | 8,213 (**0.129%**) | 1,719 (**0.130%**) |
| span | 743 hourly steps (~31 days) | 200 daily steps |
| put it in | `data/raw/` | `data/raw/IBMAml/` |

Both sit **~31x below** the synthetic default's measured 4.03% fraud rate. That is more than an
order of magnitude, so no operating point carries between the two regimes and their numbers never
share a table. `scripts/build_splits.py` measures the gap on every run rather than quoting it.

```bash
make splits      # compute + commit the out-of-time boundary and the data cards
```

The split boundary is **computed once and committed**, not re-derived per run: a fraction splits
at 70% of whatever rows it was handed, so the partition moves whenever the pool composition does
and two runs stop being comparable with nothing in the diff to show it. `artifacts/splits/*.json`
holds two timestamps and a digest; every run reads them.

The data cards in `docs/data-cards/` carry the rest — licence, base rate, the embargo and its
rationale, the measured integrity checks, the quirks and the limits. Two quirks are worth knowing
before you write a feature:

- **PaySim has no sender history.** `nameOrig` is effectively unique per row (6,353,307 distinct
  origins over 6,362,620 rows). Every `src`-side velocity and RFM feature is structurally empty on
  that anchor; `nameDest` is the only entity with a past, which is also why the run-time sample is
  taken over beneficiaries.
- **The out-of-time cut lands on two different base rates.** PaySim fraud is 3.5x denser in the
  test half than the train half, because the label is spread evenly across steps and the legit
  volume is not.

Full PaySim is ~7.7 GB once it is contract rows, so the default reads a deterministic 10%
hash-sample of beneficiaries (636,409 rows, base rate within 2.2% relative of the full file).
`data.sample.sample_fraction=1.0` reads the lot.


## The four-anchor limitation

Four public anchors have been tried: PaySim, AMLSim, BankSim and AMLworld. **All four are generated
datasets.** A better generator is not observed fraud, and nothing measured here transfers to
production payment traffic on the strength of these files alone.

Each failed the claim for its own reason, and the reasons are recorded in the ADRs rather than
summarised away:

- **PaySim** — no repeated senders, so behaviour cannot be anomalous against a history that does not
  exist.
- **AMLSim** — its own fraud is amount-trivial: floor 0.456 against a ceiling of 0.594.
- **BankSim** — transfer fails, 0.238 against a 0.702 amount floor. `docs/adr/0004-*.md`.
- **AMLworld** — passes the non-triviality gate decisively (ratio 0.0233) and still produces a null
  on the adaptive claim. `docs/adr/0005-*.md`, with the gate numbers in `artifacts/spike/`.
