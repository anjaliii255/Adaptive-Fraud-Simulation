# Tickets — Adaptive Fraud Simulation Lab

Tracer-bullet tickets derived from `SPEC.md` and `docs/architecture.html`. Each is a vertical
slice: a narrow but complete path through generation, detection, evaluation and tests, sized to
fit one fresh context window, demoable or verifiable on its own.

**Two owners, one seam.** `▲ A` is the red side and product. `■ B` is the blue side and
evaluation. `◆ A+B` is done together. The two lanes never import each other — they meet only at
`afl/contract`, and at the loop that is written against it.

Work the **frontier**: any ticket whose blockers are all done. Numbers are in dependency order,
not priority order.

**The gate that never moves:** `make smoke` stays green through every ticket. A red smoke test
blocks the frontier for both lanes.

---

## Board

| # | Ticket | Owner | Blocked by |
|---|--------|-------|-----------|
| 01 | ~~Freeze the seam and the taxonomy~~ **done** | ◆ A+B | — |
| 02 | ~~Anchor on a real dataset with a committed split~~ **done** | ■ B | — |
| 03 | ~~M3 — first-party fraud, the holdout family~~ **done** | ▲ A | 01 |
| 04 | ~~C2 — UPI collect-request / APP scam~~ **done** | ▲ A | 01 |
| 05 | ~~C3 — instant-A2A pass-through~~ **done** | ▲ A | 01 |
| 06 | ~~C1 + M2 — bust-out and the synthetic-identity lifecycle~~ **done** | ▲ A | 01 |
| 07 | ~~Causal features on real anchor data~~ **done** | ■ B | 02 |
| 08 | ~~LightGBM baseline at a fixed operating point~~ **done** | ■ B | 07 |
| 09 | ~~Graded decisions and SHAP reason codes~~ **done** | ■ B | 08 |
| 10 | ~~Anomaly layer scored on the holdout family~~ **done** | ■ B | 08, 03 |
| 11 | ~~Leave-one-attack-out harness with leakage guards~~ **done** | ■ B | 08, 03 |
| 12 | ~~Multi-vector adaptive optimiser~~ **done** | ▲ A | 01, 02, 08 |
| 13 | M1 — the optimiser's boundary walk as a vector | ▲ A | 12 |
| 14 | Realism leash, reported every round | ▲ A | 12 |
| 15 | ~~Fidelity scorecard on real anchor data~~ **done** | ■ B | 06, 08 |
| 16 | ~~The three-system table~~ **done** | ■ B | 11, 12 |
| 17 | ~~Sequence model — earn it or report it~~ **done — reported, not promoted** | ■ B | 11 |
| 18 | ~~Temporal GNN — earn it or fall back~~ **done — fell back, and said so** | ■ B | 11 |
| 19 | ~~The convergence artefact~~ **done — the loop closes, the transfer does not** | ▲ A | 11, 12 |
| 20 | One command reproduces a headline number | ▲ A | 16, 19 |
| 21 | The live demo | ▲ A | 19 |
| 22 | Submission pack and claims audit | ◆ A+B | 15, 16, 20, 21 |

**Person A owns 10 tickets** (03, 04, 05, 06, 12, 13, 14, 19, 20, 21).
**Person B owns 10 tickets** (02, 07, 08, 09, 10, 11, 15, 16, 17, 18).
**Two are joint** (01, 22) — the first day and the last.

### Where the lanes touch

Only three edges cross the lanes, and they are the only coordination points:

- **01** is done together, in one sitting, before either lane goes deep.
- **12** (A's adaptive loop) needs **08** (B's real detector). A develops against `StubDetector`
  and the ticket closes when it runs against the real one — the stub keeps A's lane moving.
- **11** and **15** (B's evaluation and fidelity) need A's vectors, **03** above all. That is
  why 03 is A's first ticket and not their fifth.

---

# 01: Freeze the seam and the taxonomy

**Owner:** ◆ A+B, together, day one, before either lane goes deep.

**What to build:** The nine attack vectors carry the IDs the architecture doc uses, both people
agree what each one means, and the leave-one-attack-out holdout family is chosen and written
down. After this ticket, "S2" means the same thing in a config file, a test, a metric and a
conversation.

Today `vectors.yaml` uses an S/V/M split that disagrees with the doc, and the doc is the source
of truth. The target taxonomy: S1 mule network & layering (graph), S2 card testing / BIN
enumeration (velocity), S3 account takeover via drift (drift), C1 bust-out (drift), C2 UPI
collect-request / APP scam (velocity + payee), C3 instant-A2A pass-through (graph + velocity),
M1 boundary probing (the optimiser itself), M2 synthetic-identity lifecycle (drift → bust-out),
M3 first-party / friendly fraud (the holdout).

The rename touches config, the evaluation default holdout, and tests. Do it as one change, with
both people present, rather than discovering the disagreement in week two.

Also agree — out loud, and in the contract's docstrings — the two definitions everything else
rests on: an attack **evades** only if it was let through untouched, and the evasion rate is
computed over fraud rows, never over all rows.

**Blocked by:** None (can start immediately).

**Status:** done — decision of record in `docs/adr/0001-vector-taxonomy-and-holdout.md`

- [x] `vectors.yaml` uses the doc's nine IDs, each with `name`, `engine`, `actor`, `maturity`,
      `why`, `params` and a bounded `search_space` — plus `level`, `tier` and `status`
- [x] Vectors not yet implemented are present with their spec and marked so an unimplemented
      vector fails loudly rather than silently generating nothing — three-state `status`
      (built / template / planned); `planned` raises from `Simulator.generate` naming its ticket,
      and a non-built vector with no stated `gap` fails to load at all
- [x] Every config, default and test referencing an old ID is updated in the same change; a
      grep for the old IDs returns nothing
- [x] The leave-one-attack-out default holdout is M3 (first-party fraud), and the reason is
      recorded where the default lives
- [x] `afl/contract` is confirmed unchanged — zero diff, the seam held
- [ ] Both owners can state, without looking, what the three taxonomy levels are and which
      vectors sit at each — *for A and B to do together; the code cannot check it*
- [x] `make test` green (85 passed), plus `make smoke`, `make loop`, `make compare`,
      `make fidelity` and `make figures` all run

**Carried out of this ticket, and worth knowing before you start yours:**

- **Six of the nine vectors still owe work**, and `vectors.yaml` says which and why. Built: S1, S2,
  S3. Template (right shape, defining tell missing): C1, C3, M1, M3. Planned (raises): C2, M2.
- **The holdout is a proxy today.** M3 is `template`, so ticket 11's headline number is measured on
  an amount-shift drift standing in for first-party fraud until ticket 03 lands. Label it, don't
  quote it.
- **The numbers moved and nothing regressed.** Changing which families are generated and what the
  holdout *is* changed every downstream number. README refreshed; pre-freeze figures are not
  comparable and were replaced, not kept alongside.
- **`motif` is a fixed knob, not a searched one.** S1 spans fan-in to deep layering via `n_hops`,
  which the optimiser does search. Searching `motif` needs categorical support — ticket 12's call.
- **New in the registry for ticket 12:** `list_vectors(tier="strong")` is the three the loop should
  wrap, and `list_vectors(generatable=True)` is what any run should iterate.
- **Fixed in passing:** `scorecard.build` read `thresholds` off its argument instead of off the
  card, so any caller omitting it crashed. That was every `make fidelity` run.

---

# 02: Anchor on a real dataset with a committed split

**Owner:** ■ B

**What to build:** The lab runs on real transaction data, not just the synthetic placeholder,
and it always runs on the *same* split. Someone re-running the pipeline next week gets numbers
comparable to today's.

A real dataset (PaySim, AMLSim, or both) enters through the loaders and leaves as
`list[Transaction]`. Nothing downstream may know which dataset it came from — that is what makes
one detector, one feature set and one evaluation run over all three sources unchanged. Real rows
carry no provenance: a real row that gains a `vector_id` has leaked a label path.

The out-of-time split with its embargo gap is computed once and committed as an artefact, not
regenerated per run. A short data card records what the dataset is, its base rate, its time span,
its known quirks, and what it cannot tell us.

**Blocked by:** None (can start immediately, in parallel with 01).

**Status:** done — split artefacts in `artifacts/splits/`, data cards in `docs/data-cards/`,
decisions of record amended into `docs/adr/0002-dataset-anchors.md`

- [x] **Both** real datasets load end to end into contract types and the whole pipeline runs on
      them via a config override, no code change — `data=paysim` (636k rows, 74s) and
      `data=amlsim` (1.32M rows, 3m19s)
- [x] Real rows have `vector_id=None` and `attack_run_id=None`; `loaders.assert_no_provenance`
      runs inside every loader and three tests assert it, one per failure mode
- [x] The out-of-time split boundary is committed as an artefact and re-used — `CommittedSplit`
      stores two timestamps plus a digest, and a test proves the partition does not move when the
      pool grows, where the fraction-based split does
- [x] Re-running produces an identical partition — asserted on the real files, txn_id by txn_id,
      across both the entity sample and the boundary
- [x] The embargo gap is non-zero and its rationale is recorded — enforced in `__post_init__`:
      a zero gap and a blank rationale both raise
- [x] A data card exists per anchor: source, licence, row count, base rate, span, the committed
      split, sampling, measured integrity checks, quirks, limits — generated from the files by
      `make splits`, never hand-typed
- [x] The synthetic placeholder config still works with nothing to download — `make smoke`,
      `make loop`, `make compare`, `make fidelity`, `make figures` all green, and every synthetic
      number is byte-identical to pre-change `HEAD`
- [x] Fraud base rate reported and the gap called out — measured, not quoted: both anchors at
      ~0.13% against the synthetic default's 4.74%, ~37x, printed as a warning by `make splits`
      and written into every data card
- [x] `make test` green (128 passed, up from 85), `ruff` clean

**Carried out of this ticket, and worth knowing before you start yours:**

- **PaySim has no sender history, and this is the biggest finding.** `nameOrig` is effectively
  unique per row — 6,353,307 distinct origins over 6,362,620 rows, mean 1.001. Every `src`-side
  velocity, RFM and recency feature is *structurally empty* on that anchor. **Ticket 07 has to
  build on the beneficiary side or it is building on nothing.** `nameDest` is the only entity
  with a past (mean 2.34, max 113).
- **Four claims in ADR 0002 did not survive contact with the files** and are corrected in the
  amendment: the synthetic base rate is 4.74% not ~17% (so 37x, not 130x); the typology join key
  is `TX_ID` not `ALERT_ID` (1,719 alert rows share only 391 alert ids); a step-fraction split is
  not a row-fraction split; and BankSim is not on disk at all.
- **The committed boundary moved, and the old one was wrong.** PaySim's `train_end_step: 500` put
  95.3% of rows in train and left a 4% test tail, because 341 of its 743 steps carry under 100
  rows. Boundaries are now derived from the row quantile: PaySim step 323 (70.2%/23.7%), AMLSim
  step 140 (70.3%/29.2%). `train_end_step` is gone from the configs — the config holds the
  *inputs*, the artefact holds the *decision*.
- **The out-of-time cut lands on two different base rates.** PaySim fraud is 3.5x denser in test
  (0.289%) than train (0.082%). **Ticket 08 inherits this:** a threshold calibrated on a tail of
  train does not transfer unchanged to test, and every recall figure has to name its side.
- **Real traffic breaks a realism rule we enforce.** AMLSim has 181 self-transfers and 19
  zero-amount rows; PaySim has 16 zero-amount rows. `afl/attack/realism.py` penalises the
  generator for emitting a self-transfer. **Ticket 14** derives its empirical bounds from these
  files and has to decide whether that rule is a fact or a modelling choice.
- **Full PaySim is ~7.7 GB as contract rows**, so the default reads a deterministic 10%
  hash-sample of beneficiaries. Whole entities are kept or dropped, never individual rows — half
  an account's history is a velocity profile no production scorer would ever see. Base rate holds
  to within 2.2% relative. `data.sample.sample_fraction=1.0` reads everything.
- **The simulator's window is now aligned to the anchor.** `engines.yaml` starts on 2024-01-01
  and PaySim's epoch puts it in January 2023; left alone, every synthetic fraud row landed a year
  after every real row and the out-of-time split degenerated into "real = train, synthetic =
  test". An attack has to happen inside the traffic it hides in.
- **The AMLSim typology is a side-channel, not a wire field.** `loaders.amlsim_typologies()`
  returns `txn_id → fan_in|cycle`. It is deliberately not on `Transaction`: **ticket 11** reads
  the map, and writing it into `vector_id` would put provenance on a real row and make the family
  carve-out treat AMLSim rows as a synthetic family.
- **The README's adaptive figure was already stale** (0.312; it reproduces at 0.126). Verified
  against pre-change `HEAD` — nothing in this ticket moved it. Refreshed, and the two number
  regimes are now quarantined into separate sections as ADR 0002 requires.
- **First real-anchor reading, and it is weak on purpose:** PaySim PR-AUC 0.025, recall@1%FPR
  0.243, **precision@100 0.00**; AMLSim 0.007 / 0.067 / 0.08. `caught_rate = 1.0` on PaySim is
  blanket friction on 14% of traffic, not detection. Features (07) and the tuned detector (08)
  are what make these mean anything.
- **Fixed in passing:** `load_ieee_cis` removed as ADR 0002 planned; `make splits` added.

---

# 03: M3 — first-party fraud, the holdout family

**Owner:** ▲ A · *A's first ticket. B's entire evaluation lane waits on this one.*

**What to build:** The simulator generates first-party / friendly fraud — the legitimate account
owner abusing the product through chargeback abuse, intentional default, or refund abuse — as a
vector B can hold out of training and measure recall on.

This is the crown jewel of the evaluation because `user == fraudster` breaks the legit-vs-attacker
assumption every supervised feature quietly rests on. There is no compromised device, no new
operator, no ring. The account behaves like itself, right up until it doesn't. That is exactly why
it is the honest stress test, and why the anomaly layer exists at all.

Built on the anomaly/drift arc: a long genuine history from a real owner, then a pattern of abuse
that is anomalous against *that account's own* baseline rather than against the population.

**Blocked by:** 01.

**Status:** done — `M3` is `built` in `vectors.yaml`; six tests in `tests/test_engines.py`

- [x] M3 generates through `Simulator.generate` — but it needed two *generic* drift knobs,
      `beneficiary_reuse` and `pace_factor`, because the engine hardcoded post-event payments to
      the cash-out pool and no YAML setting could reach it. No per-vector branch; both default to
      the previous behaviour and S3/C1/M2 regenerate byte-identical. Flagged rather than glossed
- [x] The pre-abuse history is labelled legit; only the abuse rows are fraud
- [x] No device change, no new operator, no ring — zero new payees across every account, and a
      one-line amount rule at 1% FPR catches 3% of it
- [x] Declared `search_space` with bounds that keep the family plausible
- [x] Batches pass the realism check with no violations
- [x] `tests/test_engines.py` asserts labelling, provenance, schema validity and the first-party
      shape on the emitted batch
- [x] B told: tickets 10 and 11 unblocked
---

# 04: C2 — UPI collect-request / APP scam

**Owner:** ▲ A

**What to build:** The simulator generates the dominant India account-to-account scam class: a
socially engineered push payment to a first-time payee, an atypical amount, a rapid drain, often
from an older account.

The victim authorises the payment, so there is no compromised credential to detect. The signal is
entirely in the payee relationship and the pacing — first-ever payment to this beneficiary, amount
far outside this account's normal, balance drained in minutes. Those features are cheap and
high-signal, and this vector is what keeps the submission on-rail for a Mastercard / GFF audience.

Velocity engine plus payee behaviour. No new engine.

**Blocked by:** 01.

**Status:** done — `C2` is `built`; four tests in `tests/test_engines.py`

- [x] C2 generates through `Simulator.generate` on the UPI rail. Needed three generic velocity
      knobs (`n_payees`, `amount_shift`, `device`) plus an actor-keyed endpoint rule, so a victim
      pays out of their own account. No per-vector branch; S2 and M1 byte-identical
- [x] First-time payee, atypical amount, rapid drain — all verifiable from the batch: one payee
      per victim, 100% never paid before, drain in 1–9 minutes
- [x] The victim's prior history is present and legit, so the atypicality is measured not asserted
- [x] Declared `search_space` with bounds
- [x] Realism check passes with no violations
- [x] Test in `tests/test_engines.py` asserting on the emitted batch
---

# 05: C3 — instant-A2A pass-through

**Owner:** ▲ A

**What to build:** The simulator generates instant account-to-account pass-through: funds arrive
at a new payee and drain onward immediately, across institutions, on rails with no chargeback.

This vector exists to exploit the exact property that makes real-time payments hard to defend —
irrevocability. Money that has left is gone; the only defence is catching it in the seconds it is
in transit, which makes dwell time the knob that matters. It reuses the graph and velocity
primitives already built for S1 and S2, so the marginal build is small.

**Blocked by:** 01.

**Status:** done except cross-institution, which the contract cannot express — `C3` is `built`

- [x] C3 generates through `Simulator.generate`, via one generic graph knob `fresh_beneficiary`
- [x] Inbound then immediate outbound to a beneficiary with no prior inbound — pass-through ratio
      0.990 and worst dwell 132s across 48 episodes on 12 seeds
- [x] Dwell time is searchable with bounds (`hold_time_s`, 5–900s)
- [ ] Cross-institution structure is represented — **not done, and not doable here**: the
      contract has no institution field and adding one is a joint decision. Named as a limit in
      `docs/adr/0003-template-vectors.md` rather than faked
- [x] Realism check passes with no violations
- [x] Test in `tests/test_engines.py` asserting on the emitted batch
---

# 06: C1 + M2 — bust-out and the synthetic-identity lifecycle

**Owner:** ▲ A

**What to build:** Two vectors that share one arc, built together because the second is nearly
free once the first exists.

**C1 bust-out:** a long clean tenure, then a sudden utilisation spike across correlated accounts
before cash-out. A dominant real-loss category and a pure sequence problem — the ticket that gives
B's sequence model something real to be measured on.

**M2 synthetic-identity lifecycle:** identity fabrication, new-account onboarding, seasoning, then
payment abuse. It covers the onboarding surface and models fraud that begins *before* the first
transaction. It is built as a template that hands off into the bust-out arc rather than needing an
engine of its own.

**Blocked by:** 01.

**Status:** done except the C1 ring, deliberately scoped out — both are `built`

- [x] C1 generates a long legit tenure then a visible spike, only the bust-out labelled fraud —
      64–71 tenure rows then a 10-row spike from ~35 to ~240–370 on the card rail
- [ ] Correlation across accounts — **not built, deliberately**: a ring busting out together is
      S1's territory, and keeping it out is what keeps the two families distinguishable. Recorded
      in `docs/adr/0003-template-vectors.md`
- [x] M2 reuses C1's bust-out tail behind a seasoning phase
- [x] The M2 account has no prior history by construction — the simulator mints it (`new_account`),
      so it carries exactly its 25 seasoning rows and nothing else
- [x] Both declare bounded `search_space`s; both pass the realism check
- [x] Tests in `tests/test_engines.py` for both, asserting on emitted batches
- [x] Both needed generic knobs (`pace_factor`, `new_account`), which per `SPEC.md` is the signal
      the engine was under-parameterised — fixed generically, never per vector
---

# 07: Causal features on real anchor data

**Owner:** ■ B

**What to build:** The feature table is built from the real anchor dataset, every feature is
computed strictly from prior rows, and that property is proven by a test rather than believed.

Velocity windows, RFM, time-since-last-event, graph-derived features (in-degree, unique
counterparties, first-ever-payment-to-this-beneficiary), device features. All of it stateful
across batches, because a mule's history does not reset because a new round began.

This repo has already been bitten once: the first pass found velocity windows peeking forward,
retraining not accumulating, and the loop training on the holdout window. Most of the apparent
signal was leakage. So causality gets a property test, not a code review.

**Blocked by:** 02.

**Status:** done — feature dictionary in `docs/features.md`, per-anchor cost and coverage in
`artifacts/features/`, both regenerated by `make features`

- [x] Features build over the real anchor dataset at a workable speed, and the timing is recorded
      — PaySim's 636,409 rows in about 6s and AMLSim's 1,323,234 in about 10s, both for the full
      56 columns, at over 110k rows/s each. The exact figures live in `docs/features.md`, written
      there by the script that measured them rather than typed in, and `tests/test_features.py`
      holds a throughput floor that catches a regression to quadratic
- [x] Property test: a feature computed for a row does not change when later rows are appended —
      twice over, on one hand-built probe and on random traffic, plus a brute-force reference
      that recomputes all 56 columns the obvious way and must agree exactly
- [x] Test: no forbidden column (`is_fraud`, `vector_id`, `attack_run_id`, `txn_id`) reaches X —
      and `assert_no_forbidden_columns` runs inside `transform`, with a test that deliberately
      leaks one to prove the guard fires
- [x] Stateful mode carries entity history across batches and a test proves accumulation — over
      four consecutive batches, against a stateless builder that must *not* accumulate
- [x] Graph features are computed as-of the row's timestamp, never over the full graph — proved
      by adding twenty later edges and re-scoring the same row
- [x] Feature names are stable and human-readable — the emitted columns are asserted equal to
      the registry, in order, and `explain.FEATURE_PHRASES` is asserted to name no column that
      does not exist
- [x] The feature list, with a one-line rationale each, is written down where the next person
      will find it — `docs/features.md`, generated from `feature_specs()`, with the measured
      per-anchor coverage beside each rationale
- [x] `make test` green (161 passed, up from 128), `ruff check` and `ruff format` clean, and
      `make smoke`, `make splits`, `make features`, `make fidelity`, `make loop`, `make compare`
      and `make figures` all run. `make splits` regenerates the committed boundaries and the data
      cards byte-identically, so ticket 02's decisions of record did not move

**Carried out of this ticket, and worth knowing before you start yours:**

- **The M3 fold on a real anchor cannot carry a claim, and this is the biggest finding.** After
  the family carve-out, *every positive in the holdout is an injected synthetic row and every
  negative is a real one*. A classifier told to sort the two apart does it at **AUC 1.00** on
  PaySim and 0.994 on AMLSim, on either the old or the new feature table — so the fold's recall
  is partly reporting how far the injected family sits from the real distribution, not how well
  anything detects first-party fraud. The committed fidelity scorecards agree from the other
  side: PaySim KS 0.86 on log-amount, 0.89 on the inter-transaction gap, TSTR ratio 0.03.
  **Ticket 11** needs the fold to say this itself — "too few positives to be meaningful" is not
  the only way a fold can be meaningless. **Ticket 15's** Level 3 is the gate that closes it.
- **Direction was the bug, and it was not a small one.** The old builder kept *one* history per
  entity and appended every transaction to both the sender's and the beneficiary's, so "payments
  sent in the last hour" also counted payments received, and fan-in and fan-out were literally
  the same number. Every entity now has two streams, `out` and `in`. The features that matter
  most are the ones crossing them: `src_seconds_since_last_in` is the dwell time C3 trades
  against, and `src_passthrough_ratio_3600s` is near 1.0 for a mule and well under it for a real
  account. Neither is expressible if the directions are added together.
- **The beneficiary block is the one that works on both anchors, exactly as ticket 02 predicted.**
  Nine windowed fan-in features plus seven lifetime ones, all on `dst`. On PaySim they run 11-57%
  informative while the whole `src` block is dead or under 1%.
- **17 of 56 columns are structurally dead on PaySim, 8 on AMLSim, 1 on synthetic** — measured per
  anchor and published in `docs/features.md` rather than inferred. PaySim kills the entire
  `src_out_*` 1h block, `src_amount_z`, the pass-through block and — a new one — the whole
  `pair_*` block, because a sender that appears once has never paid anyone twice. **Ticket 08**
  should read that table before it tunes: a fifth of the columns are constant on the anchor it
  will report from.
- **The features moved AMLSim and not PaySim, and that is the anchors talking, not the code.**
  Model, seed and committed boundary held fixed, measured on each anchor's *own* labelled fraud:
  AMLSim PR-AUC 0.83 → 0.95, recall@1%FPR 0.93 → 0.97, precision@100 0.98 → 1.00; PaySim
  0.14 → 0.13 with precision@100 0.38 → 0.47. AMLSim has real histories on both sides for the
  directional and graph features to read; PaySim has almost none. Do not quote AMLSim's column
  as a production number — it is a simulator whose SAR rows carry a deliberately distinctive
  fan-in / cycle topology, so what it shows is that graph features find graph fraud. (One-off
  before/after, model held fixed at sklearn HistGradientBoosting; the old table is in git at
  `f27b335`.)
- **Every feature name changed shape, on purpose, and `explain.py` moved with it.** `src_cnt_*`
  became `src_out_cnt_*`, `dst_cnt_*` became `dst_in_cnt_*`, `dst_is_new_counterparty` became
  `pair_is_first_payment`. A name now states the entity, the direction and the window. **Ticket
  09** inherits the vocabulary: 34 of the 56 have an analyst phrase, the rest fall back to the
  column name with the underscores knocked out, and a test fails if a phrase outlives its column.
- **`dst_in_degree` changed meaning and its old one was wrong.** It counted inbound *events*
  while its own reason code said "how many accounts have ever paid this beneficiary". It now
  counts distinct payers as of the row's timestamp; `dst_in_txn_count` is the event count.
- **Scoring a batch that predates the committed history is the slow path, and the loop's batches
  are that shape.** In-order scoring — a detector fitted on train scoring an out-of-time test
  window — runs at 84-135k rows/s. A batch generated *inside* the traffic it hides in is inserted
  into the middle of each entity's history instead of appended, which is roughly an order of
  magnitude slower per row on a deep-history anchor. It is bounded by one entity's history, not by
  total rows, so it stays workable at the batch sizes **ticket 12** generates — but do not score
  50,000 old rows against a 1.3M-row history and be surprised.
- **The amount z-score needed a numerically stable variance, and the naive one was already
  wrong.** `E[x^2] - E[x]^2` over an account that pays nearly the same amount every time — a
  subscription, an EMI, a salary — has both terms agreeing to fifteen digits, so what survives is
  rounding error, and `src_amount_z` divides by it. Measured before the fix: at amounts around
  1e7 with a spread of 1 the standard deviation was **10% wrong**, and at 1e9 it collapsed to
  zero and took the z-score with it. The prefix sums are now kept shifted by the stream's first
  amount, which costs one float per stream, is exact when the amounts are constant, and has two
  regression tests. Worth knowing because `src_amount_z` is the drift tell behind S3, C1 and M3 —
  a spurious variance there is a spurious "this account is behaving unusually".
- **History retention is unbounded on purpose.** `src_out_txn_count`, `dst_in_degree` and the two
  account ages are exact as-of-the-row *because* nothing is ever dropped; the old builder trimmed
  to seven days and called the result a lifetime count. The cost is linear in events, two per
  transaction, and `FeatureBuilder.state_size()` reports it — 2.6M events for AMLSim, 1.3M for
  the PaySim sample.
- **`amount_is_round` is constant zero on synthetic traffic and non-zero on both real anchors.**
  The generator rounds amounts to the paisa and therefore never emits a round-hundred figure,
  while 0.07% of PaySim and 0.02% of AMLSim rows are. Structuring — the behaviour M1's boundary
  walk models — is characterised by round numbers, so **ticket 14** has an empirical bound to set
  here, and it is the kind of tell a fidelity check would otherwise miss.
- **Two tests moved.** `test_features_do_not_see_the_future` and
  `test_scoring_does_not_mutate_feature_state` left `tests/test_eval.py` for the new
  `tests/test_features.py`, which is where the rest of the causality work now lives.
- **`artifacts/features/` is committed**, like `artifacts/splits/`, so every number in
  `docs/features.md` traces to the file that produced it — the discipline **ticket 20** audits.

---

# 08: LightGBM baseline at a fixed operating point

**Owner:** ■ B · *This is System A of the hero table, and it is A's unblock for ticket 12.*

**What to build:** A genuinely strong supervised detector on the real anchor, scored at one fixed
operating point, reporting PR-AUC, recall @ fixed FPR and precision@k on the out-of-time split.

This is the hard baseline everything else must beat, so it must be tuned honestly rather than
left weak to flatter what comes later. A soft baseline makes every subsequent result meaningless.

Two environment facts to get right: LightGBM needs libomp on macOS, and without it the code
falls back to a slower sklearn path — so the fallback must be logged loudly, and the reported
baseline must state which backend produced it. Retraining must accumulate; evasions are weighted
up because they are the expensive examples.

**Blocked by:** 07.

**Status:** done — reference in `artifacts/detector/`, written up in `docs/detector.md`, both
regenerated by `make baseline`

- [x] Trains and scores on the real anchor through the `score` / `retrain` seam — `make baseline`
      fits on the training side of the committed boundary and scores the test side through
      `detector.score(batch)`; `retrain` is exercised and asserted in `tests/test_detector.py`
- [x] PR-AUC, recall @ fixed FPR and precision@k reported on the out-of-time split — PaySim
      **0.152 / 0.478 / 0.48**, AMLSim **1.000 / 1.000 / 1.00**, each on its own labelled fraud
      at the committed boundary. Read the AMLSim row with the carry-out below, not on its own
- [x] Accuracy and ROC-AUC are not reported as headline numbers anywhere — enforced rather than
      promised: `afl/defend/baseline.py` refuses to save an artefact whose metrics contain
      either, at any depth, and a test tries three spellings including `balanced_accuracy`
- [x] The operating point is fixed in config and every comparison uses it — `eval.fixed_fpr` and
      `eval.k` now reach `run_three_systems`, the fidelity scorecard's Level 3 and the baseline
      script; `decision.assert_one_operating_point` refuses a config that names two
- [x] Which backend was used (LightGBM vs sklearn fallback) is recorded in the run artefact —
      a `Backend` record (name, version, why, what the fallback had to drop) on every model card,
      in `metrics.json` per system, under the three-system table, and in `/health`
- [x] `retrain` accumulates rather than replacing, and a test proves it — two tests: one over two
      generated batches, one over three rounds, because a bug that keeps only the last round
      survives a one-round test
- [x] Evasions are weighted above ordinary training rows, and the weight is config, not a literal
      — `sample_weights()` is asserted at two different configured weights, and the count that
      was weighted up is on the model card
- [x] Baseline numbers are committed as the reference every later ticket compares against —
      `artifacts/detector/<anchor>.json` carries the metrics, the params, the backend, the split
      digest and the seed; four tests check a committed reference against the boundary and the
      operating point in force, and that it still beats both of its controls
- [x] A is told it is ready, because ticket 12 unblocks on it — **▲ A: the real detector is
      ready.** `build_detector_factory` loads the committed tuned params automatically on
      `data=paysim` / `data=amlsim`, so ticket 12 needs no code change to swap off `StubDetector`

**Carried out of this ticket, and worth knowing before you start yours:**

- **libomp was missing, so no number in the README was ever LightGBM's.** It is installed now and
  every number above came out of LightGBM 4.5.0. The wheel imports cleanly and *then*
  fails to `dlopen` its own shared library, which is why this went unnoticed: the code fell back
  to sklearn HistGradientBoosting and kept running. The backend is now a record, not a string —
  name, version, why it was chosen, and which searched params the fallback has no equivalent for.
- **The old baseline was soft, and the margin is the whole point of this ticket.** On PaySim, the
  same features, the same seed, the same committed boundary: PR-AUC **0.060 → 0.152**,
  recall@1%FPR **0.371 → 0.478**, precision@100 **0.14 → 0.48**. The only difference is 40 Optuna
  trials against a validation tail *inside* the training window. Every comparison anyone makes
  against a pre-08 detector was a comparison against a straw man, and that includes the honest
  failure the README reports for the adaptive loop.
- **AMLSim is separable before any model runs, and this is the biggest finding.** Every alerted
  row in the file is a sub-20 amount, against legit traffic reaching 21.5M — a band containing
  only 21.9% of the negatives. Sorting on **amount alone**, direction chosen on train, reaches
  PR-AUC 0.456 and **precision@100 of 1.00** on the test window. The graph features finish the
  job and the tuned detector hits 1.000 across the board. **No AMLSim number is evidence about
  detection**; it is evidence that the generator is legible. Tickets 11, 15, 17 and 18 all plan
  to report from it — do not, or label it as a generator artefact every time.
- **So every artefact now carries an `amount_only` floor.** A baseline is only "strong" relative
  to how hard the anchor was, and this is the cheapest honest way to say how hard it was: no
  model, no features, no training, direction chosen on train. PaySim's floor is PR-AUC 0.057
  against the detector's 0.152, and its fraud spans the entire amount range, so **PaySim is the
  anchor to read.** A test fails if a committed reference stops beating its own floor.
- **`subsample` has been an inert knob since the skeleton.** LightGBM ignores `subsample` unless
  `subsample_freq > 0`, and no config ever set it. Every run's "0.9 row sampling" did nothing.
  Fixed in `DEFAULT_PARAMS`, and it is a searched knob now that it means something.
- **`A_baseline` meant two different things in two artefacts.** `experiment=baseline` calibrated
  the action bands to the target FPR; the three-system table did not, so the same system name
  carried two sets of bands and therefore two `evasion_rate` columns. All three systems now go
  through one `fit_detector` hook. **The synthetic hero-table numbers moved because of this**, not
  because the loop changed.
- **The operating point was never actually config-driven.** `run_three_systems` never received
  `eval.fixed_fpr` or `eval.k` and silently used the protocol defaults, which happened to match —
  so the config knob had no effect and nobody could have noticed. Same for the fidelity
  scorecard's Level 3, which compared detectors at a threshold the rest of the run did not use.
- **The M3 fold did not move, and that is ticket 11's problem statement.** A detector 2.5x better
  on PaySim's own fraud scores **0.0066 / 0.043 / 0.00** on the M3 leave-one-attack-out fold —
  within noise of the untuned one. Ticket 07 said that fold measures the distance between an
  injected family and the real distribution; this is the confirmation, from the other side. A
  better detector cannot fix a fold that is not measuring detection.
- **Only the decline band is calibrated, and ticket 09 should start here.** `calibrate_to_fpr`
  places `decline_at` at the target FPR and then puts the other three bands at fixed ratios below
  it — 0.8, 0.6 and 0.3 of it — which are calibrated to nothing. On the PaySim M3 fold that lands
  friction on **45.6% of holdout traffic** while precision@100 is 0.00: blanket friction, not
  detection. The three metrics are unaffected (they are ranking metrics), but `evasion_rate` and
  `friction_rate` in the same table are a function of those ratios. **Ticket 09** replaces them
  with bands chosen by expected cost, which is the fix.
- **The hero table's System A is not the reference in `docs/detector.md`.** `defend.unsupervised`
  has `ensemble.enabled: true`, so every system in `run_experiment` is a blend of the supervised
  detector and an isolation forest at weight 0.7 — the model card says `EnsembleDetector` and
  names both halves. `make baseline` measures the supervised detector alone. Two different
  systems, two artefacts; do not quote one for the other.
- **PaySim's test window is 3.3x denser in fraud than its training window** (0.083% → 0.272%), as
  ticket 02 warned. The bands are calibrated on a validation tail of *train* and land at
  `decline_at = 0.0287`; every metric names the side it was measured on, in the artefact.
- **The tuning validation tail is thin, and the artefact says so.** 46 fraud rows on PaySim, 196
  on AMLSim. `n_val_positives` is recorded next to the score for exactly this reason — a search
  maximised against 46 positives has a real variance nobody should read past.
- **`artifacts/detector/` is committed, like `artifacts/splits/`.** Config holds the inputs (the
  starting params and the search envelope); the artefact holds the decision (the params the
  search landed on). `make baseline` regenerates both, and re-running it reproduced every metric
  identically. `make baseline --doc-only` rewrites the document from the committed artefacts
  alone, since it is a pure function of them.
- **Fixed in passing:** `/health` returned a dataclass where the Streamlit demo expected a string;
  `calibrate()` left the detector unfitted when there were too few validation rows to calibrate
  with; `three_system.measure` is public now, because `run_experiment` was reaching for `_measure`.

---

# 09: Graded decisions and SHAP reason codes

**Owner:** ■ B

**What to build:** A score becomes an action an analyst can act on. Every `DetectorScore` carries
a graded action — allow / step-up / hold / review / decline — chosen by expected cost, and reason
codes in language a human reads.

A binary block/allow decision throws away the part of the distribution where the money is: the
uncertain middle, where friction is cheap and a decline is expensive. Thresholds come from the
cost model, so "what FPR do we run at" becomes a business question with an auditable answer
instead of a number someone picked.

Reason codes turn 0.91 into "beneficiary saw 14 inbound payments in an hour, from 14 accounts
that had never paid it before" — which is also what makes a false positive arguable instead of
mysterious. When SHAP is unavailable the fallback to global importance is labelled in the reason
string, so nobody mistakes a global explanation for a local one.

**Blocked by:** 08.

**Status:** done — cost model in `config/costs/default.yaml`, evidence in `artifacts/decisions/`,
written up in `docs/decisions.md`, both regenerated by `make decisions`

- [x] Actions are chosen by expected cost under the cost model, not by hand-set cut-offs — all
      five actions are priced at the transaction's own probability and its own amount and the
      cheapest wins. `config/defend/lgbm.yaml` has no band numbers left in it; the four `*_at`
      values still exist for the model card and the demo gauge, derived from
      `CostModel.bands()`, which is the lower envelope of the five cost lines. A test holds the
      derived ladder and the per-transaction argmin to the same answer at four amounts
- [x] The cost model's parameters live in config with a stated rationale per number — eight
      `{value, why}` pairs, plus `unit_amount`, and `CostModel.from_config` **refuses to load** one whose `why` is
      blank. A comment can be deleted and nothing notices; a required field cannot
- [x] Every flagged transaction carries at least three reason codes in analyst language —
      an invariant, not a target: `explain` chooses whether *allowed* rows are explained too,
      never whether flagged ones are. `explain.assert_flagged_rows_are_explained` is the guard,
      it runs in `make decisions`, and a test proves it fires
- [x] The SHAP-unavailable fallback is labelled in the reason string itself — `GLOBAL_PREFIX`
      travels with the text, so it survives anything that drops a sibling field on the way to a
      UI. Tested by making `import shap` raise
- [x] Changing a cost parameter visibly moves the action mix, demonstrated in a test — a 7x
      cheaper false decline declines more, a 100x dearer analyst reviews less. The artefact
      carries the magnitude on real rows next to the test's direction
- [x] The evasion definition still holds: only `allow` counts as evaded, and a test asserts it —
      `EVASION_ACTIONS` is unchanged, and all five actions are checked, not just the two ends

**Carried out of this ticket, and worth knowing before you start yours:**

- **Cost bands on an uncalibrated score put friction on 99.3% of legit traffic, and this is the
  biggest finding.** The same synthetic run at `decision.calibration=none` against `sigmoid`:
  System A's friction rate is **0.9933 → 0.0933**. A cost model compares `p x amount` against a
  flat analyst cost, so if `p` is a ranking score rather than a probability the arithmetic is
  right and the inputs are meaningless. `afl/defend/calibration.py` fits Platt scaling on the
  same validation tail the tuner uses, `calibration: sigmoid` ships, and a one-shot warning fires
  if anything decides in cost mode without it.
- **"Calibration is monotone so it cannot move a metric" is true in exact arithmetic and false
  in float64, and it cost a committed number before it was caught.** The first cut reported the
  calibrated probability as `DetectorScore.score`, on that argument. But `1/(1+exp(-z))` rounds
  to exactly 1.0 past z ~ 37, and a fitted Platt slope reaches that easily: on PaySim's test
  window the map collapsed the 129 distinct scores in the top 200 rows to **one value across 480
  rows**, and precision@100 on the stock-params control moved **0.14 → 0.06**. Two changes
  followed. `Z_LIMIT` clamps the logistic at the last value float64 can still tell apart, and —
  the structural one — **the calibrated probability no longer reaches the reported score**. It
  chooses the action and appears in the reason code; `DetectorScore.score` stays the detector's
  own. The decision layer now cannot move a detection metric *by construction*, which is a
  guarantee rather than an argument, and `tests/test_decision.py` asserts it end to end across
  two wildly different cost models. **The band numbers in an artefact are therefore in calibrated
  probability units and `score` is not** — `band_units` says so on every card.
- **System C's numbers do not survive a decision-layer change, and A's and B's do.** With the
  calibrator on and off, A and B report byte-identical PR-AUC, recall@1%FPR and precision@100
  (0.567018 / 0.288660 / 0.66). **System C does not** (0.14517 vs 0.141968), and that is not a
  bug: the loop retrains on evasions, "evaded" means "allowed", and which rows are allowed is a
  function of the calibrated probability. **Tickets 16 and 19 inherit this** — a System C column
  from before this ticket is not comparable to one after it, and the reason is the feedback loop
  rather than the metric.
- **The old bands sat inside the score distribution's noise floor.** On the M3 fold the
  detector's single highest probability is **1.8e-05**. `calibrate_to_fpr` put `decline_at` at the
  99th percentile of validation negatives — the same 1e-5 neighbourhood — and the other three at
  0.8, 0.6 and 0.3 of it. That is what ticket 08's "friction on 45.6% of holdout traffic while
  precision@100 is 0.00" was: four thresholds placed inside numerical noise. A cost model
  declines to act at a fraud probability of 0.0018%, which is the right answer and is why the
  M3 fold now flags nothing at all.
- **"Cost-derived means less friction" is false, and the test that asserted it was wrong.** What
  the ratio bands did depended entirely on where the score distribution sat relative to them: on
  the M3 fold they frictioned 45.6%, on PaySim's own fraud at the committed boundary they
  friction 0.67% while letting 58% of the fraud through. A cost-minimising policy buys **more**
  friction when the fraud it stops is worth more than the friction costs — 3.7% against 0.67%
  there, for 17 points less evasion. "Lower realised cost than the ratio bands" is not
  guaranteed either, and only just: on the small synthetic window it loses by 0.2%
  (5,186 against 5,174), because minimising *expected* cost under an imperfect probability says
  nothing about realised cost on one sample. The test asserts the floor that *is* guaranteed —
  beating allow-everything and decline-everything — and the ratio-band comparison is **measured**
  per anchor in `artifacts/decisions/`, where it is a result rather than an assumption.
- **A flat cost in absolute currency cannot serve two anchors.** PaySim's median payment is
  74,872 and AMLSim's is 157 — 477x. A review priced at an absolute "4.0" opens the review band
  at p=0.00005 on one and never opens it on the other, looking equally principled in both
  artefacts. Flat costs are quoted against `unit_amount`, the anchor's own median payment, and
  resolved at load; the same eight numbers then place the *same* ladder
  (0.005 / 0.16 / 0.38 / 0.75 at the median payment) on PaySim, AMLSim and synthetic alike.
- **The house cost model had a dominated rung and the ladder was silently four actions, not
  five.** A hold cost 3x a step-up and stopped less fraud (0.5 residual loss against 0.2), so it
  could never minimise expected cost at any probability or any amount.
  `CostModel.__post_init__` now raises `DominatedAction` on a non-monotone ladder rather than
  dropping a rung in silence. Efficacies are named knobs now (`hold_efficacy`,
  `review_efficacy`) instead of the literals 0.5 and 0.1 buried in `expected_cost`.
- **A short ladder on a cheap payment is the policy working, not a band going missing.** On a
  5-unit payment a hold and a review are both dominated — an analyst costs more than the whole
  amount at risk — so `bands()` omits them and `to_dict()` reports them as `unreachable_bands`.
  On PaySim the *decline* band never fires either, because the detector's top probability sits
  below it; `docs/decisions.md` says so in a sentence rather than leaving a column of zeroes.
- **The ensemble is the detector every anchored run actually uses, and its entire explanation was
  the string `"ensemble"`.** `defend.unsupervised.ensemble.enabled` is true, so `run_experiment`
  never scored through the reason-code path at all — the SHAP work sat behind a supervised
  detector nothing called. `EnsembleDetector.score` now carries the supervised SHAP codes, the
  anomaly half's largest standardised deviations from legit traffic, and the blend split.
- **Reason codes are priced per flagged row, not per scored row, which is what made them
  unconditional.** Under 1% of a batch carries an action, so SHAP runs on a hundred rows instead
  of a hundred thousand. `explain` is `flagged | always`; the old `false` maps to `flagged` and a
  test holds that, because "explain nothing" is no longer on offer.
- **Two columns, one fact.** `amount` and `log_amount` both read "transaction amount", so a tree
  splitting on both spent two of an analyst's three reason codes saying it twice. Reason codes
  deduplicate on the *phrase* rather than the column, and values are formatted for a person —
  `7,576,325` not `7.57632e+06`, and a dwell time as `4.6m` not `276480`.
- **The three-system table gained a `friction_rate` column.** Under a cost-derived policy the
  false-positive rate is an *output*, not a target, so the column that reconciles `recall@1%FPR`
  with what the policy actually did belongs in the table rather than in a sibling artefact.
  `assert_one_operating_point` takes the mode now and refuses cost mode plus a `calibrate_to_fpr`
  outright, since that would be two operating points wearing one config.
- **The measured result, on PaySim's committed test window.** Friction 3.80% of legit traffic
  (against the ratio bands' 2.19%), declines under 0.01% (against 0.75%), fraud allowed through
  36.3% (against 44.2%). Realised cost **77,991,948 against 578,261,907** — 86.7% better than
  allowing everything, where the policy it replaced managed 1.5%. Calibration moved the Brier
  score 0.00060 → 0.00058 and the expected calibration error **0.00121 → 0.00020**, fitted on 46
  positives. The trade is more friction for less loss, which is what a cost model is *for*.
- **On AMLSim every policy loses to doing nothing, and that is the anchor talking.** The shipped
  policy realises 61,419 and the ratio bands 150,607,219 — but allowing everything costs 5,206,
  because that *is* the total value of the fraud in the window. Against 7.84 for one analyst
  review, the whole fold is worth 664 reviews and no threshold anywhere pays for itself. One more
  reason ticket 08's warning stands: **AMLSim is a generator artefact, not evidence.** The doc
  says this itself rather than leaving two numbers in a table to be read against each other.
- **`artifacts/decisions/` is committed, like `artifacts/splits/` and `artifacts/detector/`.**
  It carries the resolved cost model with its rationale, the ladder at four amount quantiles, the
  calibration reliability, the action mix, and the realised cost against three controls — allow
  everything, decline everything, and the ratio bands this ticket replaced, all four scored from
  the same probabilities so that only the band placement differs. The ratio-band control is
  reproduced by `decision.ratio_band_policy` rather than by the current `calibrate_to_fpr`, which
  no longer places bands the way the old one did: a control built from the new code would be
  comparing against a policy that never ran. It reproduces the pre-ticket artefact's operational
  numbers exactly, which is how we know it is faithful.
- **`make splits` had been crashing, so PaySim's committed artefact was never regenerated.**
  `build_splits.py` raised `KeyError` on `amlworld` — whose boundary is committed by hand, as its
  own data card says — and anchors after it alphabetically, PaySim included, were never reached.
  It skips what it has no raw reader for now, the way `build_baseline` and `build_decisions`
  already skip an anchor that is not downloaded. **No committed boundary moved and no digest
  changed**; what moved is the stale synthetic base rate both cards quote for comparison, 4.74%
  → 4.03%, so the headline is "~31x below" rather than "~37x". Same root cause as the feature
  artefact below: the synthetic pool grew when tickets 04-06 added four vectors.
- **`artifacts/features/synthetic.json` had been stale for two tickets, and running the gate is
  what found it.** It was last written at `52be737` (23 Aug); `config/attack/engines.yaml` gained
  C1, C2, C3 and M2 at `e1d516a` (24 Aug), so the committed artefact described a 15,913-row pool
  from the old vector list while the config now generates 20,093. Regenerated. Nothing structural
  moved — 8 dead columns on AMLSim, 17 on PaySim, 1 on synthetic, exactly as ticket 07 recorded —
  only the synthetic coverage percentages, which is what a bigger pool does. **`make features`
  after a change to the vector list, not just after a change to the features.**
- **The README's current-numbers section was stale and one of its sentences was false.** It still
  said the table was "produced on the sklearn HistGradientBoosting fallback, not LightGBM, because
  libomp was missing" — which ticket 08 fixed. Refreshed from a fresh `make compare`, with the two
  operational columns added. Worth reading once: **A and B are now identical to six decimals**
  (PR-AUC 0.567018, recall@1%FPR 0.288660, precision@100 0.66) while B trains on twice the fraud.
  SMOTE interpolates between existing fraud rows and on this holdout cannot invent the one thing
  that would help, which is exactly the falsification System B is in the table to provide.
- **Fixed in passing:** the Platt fit is written out rather than routed through
  `LogisticRegression`, which overflowed inside its own line search on this data; the demo's
  detector reads its policy from the shipped configs instead of defaulting its own, so it is no
  longer a private code path; `docs/detector.md` no longer claims the action bands are calibrated
  on the tuning tail, because they are not — the score → probability map is.

---

# 10: Anomaly layer scored on the holdout family

**Owner:** ■ B

**What to build:** An unsupervised detector fit on legit traffic only, and an honest measurement
of how it does on the held-out first-party family that the supervised model has no labels for.

The supervised model can only catch what it has labels for — which by construction excludes the
family the loop is about to hold out. An outlier score trained on legit traffic degrades more
gracefully against an unseen vector, so it is the floor under the ensemble.

The interesting result is the comparison, whichever way it goes: if supervised recall collapses on
M3 and the anomaly layer holds up, the anomaly layer is promoted to headline and that is a finding
worth reporting on its own.

**Blocked by:** 08, 03.

**Status:** done — evidence in `artifacts/anomaly/`, written up in `docs/anomaly.md`, both
regenerated by `make anomaly`. **The layer did not earn its promotion, and the ticket's premise
did not survive the measurement.**

- [x] Fits on legit rows only; a test asserts no fraud row entered training — the assertion is on
      what the fit *recorded* (`AnomalyTraining.n_fraud_seen`, `n_fraud_excluded`), not on the
      filter, and it is on every model card. `fit` has no `legit_only` argument any more: a
      guarantee with a default is a setting, and a second test proves the keyword is gone. The
      contaminated variant still exists, as `contaminated_control`, because the design claim is
      worth measuring — see below
- [x] Scores through the same `score` seam and returns graded actions like every other detector —
      one `DecisionPolicy`, the same five actions, `explain.assert_flagged_rows_are_explained`
      run over the emitted scores in `make anomaly` rather than trusted. Its reason codes are the
      columns furthest from legit traffic's normal, in σ, which is the honest thing an outlier
      score can say about itself; there is no SHAP path because an isolation forest has no
      `TreeExplainer` contract
- [x] Recall on the held-out M3 family reported side by side with the supervised model's — plus
      the amount floor, plus the contaminated control, plus the supervised model's number on the
      anchor's *own* fraud, all five in one table per anchor at one operating point
- [x] Ensemble behaviour with the supervised model is defined and measured, not assumed — defined
      as `w * p_sup + (1-w) * p_uns` and a test holds `EnsembleDetector.score` to that arithmetic
      exactly at four weights; measured as an eleven-point sweep from one scoring pass, both
      endpoints included, so each half sits in the same curve as the blends of them
- [x] If supervised recall collapses on M3 and the anomaly layer does not, that is written up
      as a result rather than buried — it did not collapse, and that is written up instead. The
      write-up is *generated* from the artefacts, so a re-run that reverses the result reverses
      the prose with it

**Carried out of this ticket, and worth knowing before you start yours:**

- **On PaySim the held-out family was separable from the anchor by account id alone, at PR-AUC
  0.800, and had been since anchoring landed.** This is the biggest finding and it invalidates
  numbers, not just prose. `afl/attack/envelope.py:_busiest` kept only entities that transact more
  than once — a "seasoned account" filter that is right on AMLSim and catastrophic on PaySim,
  whose senders are effectively unique per row. At the committed 10% sample it returned **86**
  senders for 340 sender-side population slots; `Simulator._build_population` minted
  `e00042`-style ids for the other **254**, and those attacks ran in a namespace the anchor has
  never seen. It gets worse as the sample shrinks — at `--sample 0.01` the filter returns *one*
  sender, everything else is minted, and the separation is a clean **1.000**. The audit that
  exists to catch exactly this (`envelope.audit`) runs on every anchored run and does warn: it was
  the warning nobody had read, which is why `make anomaly` writes the audit into the artefact and
  the doc leads with it rather than logging it. The filter now relaxes to "the busiest, seasoned
  or not" when it cannot fill the pool, the population wraps a short pool instead of inventing
  ids, and a test builds a unique-sender anchor and asserts the attack lands on real accounts.
  After the fix the worst single field on PaySim is `hour_of_day` at 0.0031 against a base rate of
  0.0005. **`artifacts/paysim_baseline/` predates the fix** and its PR-AUC of 0.0066 is a number
  from a fold with a near-perfect label in it.
- **An anchored population could hold the same account twice under two roles, and now cannot.**
  The merchant, relay and sender pools are separate namespaces but not disjoint sets — an account
  the anchor shows both sending and receiving appears in two of them — so `_build_population`
  could emit one `entity_id` as a MERCHANT *and* as a NORMAL. It deduplicates by account id now
  (first role wins), which on AMLSim makes the population 380 distinct accounts rather than 400
  slots with 20 of them doubled, and is what makes wrapping a short pool safe. Merchants and
  mules are drawn from the envelope's own pools rather than from `entities`, so what this
  actually shrinks is the victim pool. **Anchored artefacts produced before this ticket —
  `artifacts/amlsim_baseline/`, `artifacts/transfer/` — used the duplicated population.**
- **The premise of this ticket is false on both anchors, and the sharp version is worse than
  "the anomaly layer lost".** The supervised model does not collapse on the family it has no
  labels for: on PaySim it scores PR-AUC **0.524** and recall@1%FPR **1.000** on held-out M3
  against **0.152** / **0.478** on the anchor's own real fraud in the same test window. An unseen
  family that is *easier* than the seen one is not a generalisation result, it is a statement
  about the injected rows — which is the same thing ticket 07's carry-out and the transfer test
  said, arriving from a third direction. **Ticket 11 inherits this**: a leave-one-attack-out
  headline built on M3 is measuring the distance between two distributions at least as much as
  detection, and the fidelity scorecard is the only thing that can say how much.
- **The anomaly layer is below the floor, not under the ensemble.** PR-AUC **0.033** on PaySim and
  **0.003** on AMLSim, where sorting by amount alone — no model, no features, no training,
  direction chosen on train — reaches 0.003 and **0.034**. It clears the floor on one anchor and
  loses to it on the other. It is kept because the blend measurably needs it on PaySim, not
  because it holds up alone.
- **The blend does earn its place on PaySim and costs on AMLSim, which is why it is swept rather
  than argued about.** PaySim's weight curve has an interior optimum: w=0.5 reaches **0.551**
  against 0.524 for the supervised model alone and 0.033 for the anomaly layer alone, and the
  shipped w=0.7 takes 0.017 of the 0.027 available. AMLSim's rises monotonically to w=1.0, so the
  shipped weight costs 0.013 there. The weight is **not** auto-tuned per anchor: chosen on the
  fold it is reported from, it would be the tuning-on-test ticket 08 exists to forbid. The sweep
  is free — the blend is arithmetic over two probability vectors, so eleven rows cost one scoring
  pass, and a test holds `EnsembleDetector.score` to that arithmetic so the reuse stays valid.
- **The outlier score was a statement about the batch, not about the transaction.**
  `predict_proba` min-maxed the raw scores of whatever it was handed. The same PaySim rows scored
  whole and in eight shuffled batches moved by up to **0.255** (mean 0.041) — and no metric
  noticed, because a within-batch min-max is monotone and PR-AUC only reads order. What did notice
  was the ensemble, which was blending a batch-relative rank statistic 0.3-to-0.7 against a
  probability. The fix is that there is nothing to rescale: sklearn's `-score_samples` is
  `2 ** (-E[h(x)] / c(n))`, already in (0, 1) by construction, so the map is the identity and the
  drift is now **exactly 0.0**. The autoencoder's reconstruction error is unbounded and gets a
  fit-time reference instead, with the float64 saturation point named rather than assumed absent —
  ticket 09's `Z_LIMIT` lesson, applied before it cost anything.
- **The whole scoring path is still batch-dependent, on purpose, and that is a feature-contract
  property rather than this layer's.** `FeatureBuilder.transform(update=False)` lets a row see the
  rows before it in the same call — by the time the second payment of a burst is scored in
  production, the first one has happened. Residual drift on the same experiment is 0.137 on PaySim
  and 0.172 on AMLSim, identically for the supervised detector. It is measured and reported next
  to the map's 0.0 so the two are never confused again.
- **"Fit on legit only" did not pay for itself on either fold, and it is kept anyway.** The
  contaminated control scores *higher*: PR-AUC 0.043 against 0.033 on PaySim and 0.005 against
  0.003 on AMLSim — while allowing **more** of the held-out family through under the policy (74.3%
  against 66.3%). Both readings are in the artefact. The rule stays because the training fraud in
  this fold is other *known* families, which is not the case the rule exists for; what the numbers
  say is that this fold cannot test it, not that it is wrong.
- **Calibrating the anomaly layer is not cosmetic, and the failure mode is ticket 09's, exactly.**
  An isolation forest's raw score sits in a narrow band — [0.358, 0.685] on AMLSim — so a cost band
  placed at a genuine probability catches every row. Run `python scripts/build_anomaly.py --sample 0.01` and the
  validation tail drops below `calibration.MIN_POSITIVES`, the map is correctly refused, and the
  layer puts friction on **100%** of legit traffic. At the committed sample it fits on 50 (PaySim)
  and 281 (AMLSim) positives and expected calibration error goes **0.420 → 0.00034**. The layer's
  *model* still sees no labels; its cost map sees the same validation tail every other detector's
  does, never the holdout, and the distinction is stated in the doc rather than left to be
  inferred.
- **An ensemble's model card was its supervised half's card wearing the ensemble's name.**
  `three_system._card` and `lgbm.model_card_of` both unwrapped `.supervised` before asking for a
  card, so the layer producing 30% of every blended score did not appear in a single run artefact.
  Both prefer a detector's own card now; `EnsembleDetector.model_card` carries the weight, the
  supervised card and the anomaly card with its `legit_only` and `n_fraud_excluded` counts.
- **`config/defend/anomaly.yaml` has no `legit_only` key left in it**, the same way
  `config/defend/lgbm.yaml` has no band numbers left in it. The knob was `true` by default and
  read by nothing, which is the state a guarantee reaches just before somebody flips it.

---

# 11: Leave-one-attack-out harness with leakage guards

**Owner:** ■ B · *The headline evaluation. Everything after this is measured through it.*

**What to build:** Train without one attack family, then measure recall on that family alone —
for any chosen family, with assertions that make silent leakage impossible.

Reporting recall on a family the model trained on measures memorisation. The claim is
generalisation to an unseen attack, so the carve-out has to be airtight. Two guards, both easy to
lose and expensive to lose quietly: not one row of the held-out family reaches training — replay
buffer included — and the split stays out-of-time, so "unseen family" does not smuggle in "seen
future".

The holdout keeps all legit rows. Without a haystack, FPR and precision@k mean nothing.

Start with the M3 fold end to end. The remaining folds fill in as A's vectors land — the harness
should run whatever exists and say plainly which folds it skipped and why.

**Blocked by:** 08, 03. *(Additional folds unblock as 04, 05 and 06 land; the harness must not
wait on them.)*

**Status:** done — harness in `afl/evaluation/leave_one_attack_out.py`, matrix built by
`make loao`, evidence in `artifacts/loao/`, written up in `docs/loao.md`. **The guards all hold
and almost nothing survives them: on both anchors the matrix is mostly withheld, and the reason
is the generator rather than the detector.**

- [x] Any vector can be named as the holdout via config — `eval.held_out_vector` picks the
      headline row, `eval.folds` picks which rows exist (`auto` is every vector the registry
      knows), and `--held-out` / `--folds` override either for one run. A test carves S1, S2 and
      M3 out of the same pool and asserts each one leaves training clean
- [x] Assertion: zero rows of the held-out family in training, replay buffer included, and the
      assertion fires in a test that deliberately tries to leak one — **two** tests, because the
      two leaks are different bugs. `assert_family_held_out` audits the training list, and with a
      detector it audits `detector.training_rows`, a new property on `LGBMDetector`,
      `AnomalyDetector` and `EnsembleDetector` that returns the corpus *and the replay buffer*.
      One test leaks a row into the split; the other leaves the split spotless and pushes an M3
      row into `_replay`, which is the leak a split-side assertion cannot see. A detector with no
      `training_rows` **fails** the guard rather than passing it: unauditable is not clean
- [x] Assertion: the split is still out-of-time with the embargo intact after the carve-out —
      `assert_embargo_intact` compares the realised gap against the committed boundary's own
      embargo, not against a config number, and the test closes the gap to an hour and watches it
      raise. The arithmetic says a carve-out can only widen the gap; an argument is what an
      assertion replaces
- [x] All legit rows are retained in the holdout — `assert_haystack_intact` diffs the test
      window's legit ids against the holdout's and names the dropped ones. Recorded in every
      artefact as `guards.haystack`, so a committed number says the haystack was whole
- [x] A fold with too few positives to be meaningful is reported as such, never as a low score —
      and the numbers move out of `metrics` into `withheld_metrics` when it happens, so a reader
      who quotes the obvious field gets `None`. The floor is `MIN_MEANINGFUL_POSITIVES = 30`,
      which `scripts/build_anomaly.py` now imports rather than restating
- [x] Skipped folds are named, with the reason, in the output — every requested fold gets a row
      in the artefact, the doc and the console, whether it ran or not. A fold that vanishes reads
      as "not applicable" when it means "we did not look"
- [x] Results write to an artefact with the config and seed that produced them —
      `artifacts/loao/<anchor>.json` carries the eval config as read, the seed, the committed
      split digest, the sample fraction, per-fold row counts, all three guard reports, the amount
      floor, the provenance probe and the model card. Versioned, and an old file raises rather
      than being read with the wrong meaning
- [x] **Not on the list, and the finding of the ticket:** ticket 07's carry-out asked this ticket
      to make the fold say for itself when it is meaningless. It does — see below
- [x] `make test` green (305 passed, up from 288 — 17 new tests in `tests/test_eval.py`),
      `ruff check` and `ruff format` clean, and `make loao` runs both anchors end to end

**Carried out of this ticket, and worth knowing before you start yours:**

- **Four of eighteen folds carry a number, and the headline fold is not one of them.** Nine
  families across two anchors. `measured`: AMLSim C1 **0.975**, M3 **0.996**, S2 **1.000**, and
  PaySim S1 **0.275**. The other fourteen are withheld — seven for being too thin, three because
  a classifier can sort the injected rows from the anchor's own, two because one contract field
  can, two because the vector is still a template. **PaySim M3, the fold this project's headline
  rests on, is withheld.** Every number is in `artifacts/loao/`, none of them is gone, and each
  one sits next to the reason it may not be quoted.
- **The check that decides the table was not on the ticket's list — ticket 07 put it there.** Its
  carry-out asked ticket 11 to make the fold say for itself when it is meaningless, and this is
  it: the carve-out drops the anchor's own fraud from the holdout, so in every fold *every
  positive is an injected row and every negative is a real one*, and "caught the fraud" and
  "spotted the synthetic row" are the same label. `scripts/build_loao.py:provenance_probe`
  cross-validates a classifier on the fold's own feature space to make exactly that call. It
  takes three rows outright — PaySim C1 (**0.906** against the detector's 0.604), M3 (**0.970**
  against 0.893) and S2 (**1.000** against 0.005) — and would have taken PaySim M1 (0.945) if the
  template gate had not fired first. Ticket 07 measured this by hand at AUC 1.00; it is a gate
  now, so it cannot be quoted around.
- **PaySim S2 is the sharpest row in the table, and it cuts both ways.** A classifier separates
  the injected card-testing rows from PaySim traffic at PR-AUC **1.000**. The detector scores
  **0.005** on the same rows. Perfectly identifiable by provenance and invisible to the model —
  which is the cleanest available proof that the fold's two questions are not the same question,
  in the direction nobody expects.
- **M2 cannot be evaluated by injection into a real anchor at all, and that is structural rather
  than a bug.** `sender_in_anchor` separates the injected M2 rows at PR-AUC **1.000** on *both*
  anchors. A synthetic-identity vector invents accounts by definition, so the family's defining
  behaviour is the thing the commensurability audit flags. Nothing in the generator can fix that
  while the fold is "inject into real traffic". **Ticket 15** or a fully synthetic anchor is the
  instrument for M2; leave-one-attack-out is not.
- **AMLSim's three measured rows are not a result either, and the document says so without being
  asked.** The same detector scores **1.000** on AMLSim's own labelled fraud in the same test
  window, where sorting by amount alone already reaches **0.456**. A near-perfect fold on an
  anchor like that says the simulator is legible — ticket 08's carry-out, arriving from a third
  direction. The generated write-up puts every measured row next to its anchor's own-fraud
  reference for exactly this reason.
- **PaySim's one measured row is easier than the fraud the detector trains on.** S1 at 0.275
  against **0.152** on PaySim's own labelled fraud in the same window. That is ticket 10's
  reading, unchanged: an unseen family the detector finds easier than the seen ones is a
  statement about the injected rows.
- **The provenance probe is underpowered exactly where the fold is thin, and the asymmetry runs
  the wrong way.** It learns "injected" from the fold's own positives — at best a hundred of them
  across three cross-validation folds — while the detector it checks learned from a 930k-row
  window. Every AMLSim probe scores under 0.36 and every thin fold's probe scores under 0.01. So
  a **high** probe score is strong evidence a fold is provenance-bound; a **low** one on a thin
  fold is weak evidence of anything. `MIN_MEANINGFUL_POSITIVES` is applied first so the weakest
  probes belong to folds that were already withheld, and the positive count travels with every
  probe score. **Do not read a low probe on a thin fold as a clean bill of health.**
- **Seven folds are too thin at the committed `eval.holdout_episodes: 12`** — AMLSim C2 (18), C3
  (12), S1 (20), S3 (12) and PaySim C2 (21), C3 (20), S3 (24) positives. The knob that fixes it
  is one line of config, and raising it moves ticket 10's committed fold too, so it is a decision
  rather than an oversight and it was not taken here. **PaySim S3 is the one that costs
  something**: detector 0.916, provenance probe 0.042, withheld for six rows. It is the most
  likely candidate in the matrix for a fold that would carry a real claim.
- **`training_rows` is a new seam on every detector**, and the replay-buffer guard is the reason
  for it. `LGBMDetector`, `AnomalyDetector` and `EnsembleDetector` each expose what they have
  fitted on — corpus *and* replay buffer — and the guard audits that rather than the list handed
  to `fit`. A detector without the property **fails** the guard instead of passing it, because
  the failure this whole ticket exists to prevent is the silent one.
- **The withheld numbers live under `withheld_metrics`, never under `metrics`.** A consumer that
  reads the obvious field on a fold that cannot carry a claim gets `None`, not a number. The
  document still prints them, in brackets, beside the reason — hiding evidence is its own kind
  of dishonesty, but a bracketed number next to "withheld" is not one anybody quotes by accident.
- **What this hands to the tickets that depend on it.** **16** (the three-system table) inherits a
  headline fold that is withheld on PaySim and vacuous on AMLSim — the table can still be built,
  but its held-out column needs the same treatment this matrix gives every row. **17** and **18**
  inherit the same: a sequence model or a GNN measured on PaySim M3 is being scored on a fold
  whose positives a classifier finds at 0.970. **19**'s convergence artefact is measured through
  this harness and inherits its verdicts.

---

# 12: Multi-vector adaptive optimiser

**Owner:** ▲ A · *The novel part. Also where the current numbers are expected to turn around.*

**What to build:** The attacker searches across the strong three vectors (S1 mule, S2 card
testing, S3 ATO) within one loop run, finding the detector's weakest surface instead of grinding
one vector against a detector that already generalises.

This is the diagnosis behind today's honest failure: the adaptive system currently scores below
both baselines because the loop searches a single vector against a detector that already handles
the holdout. Widening the search is the fix the design predicts.

Fitness stays `evasion_rate − λ · realism_penalty`, with evasion rate over fraud rows only —
diluting by legit volume would make the convergence curve a function of batch composition rather
than attack success. Optuna where available, random/hill-climb fallback otherwise, so the loop
never hard-depends on the backend.

Develop against `StubDetector` so this lane is not blocked; the ticket closes when it runs against
B's real detector from ticket 08.

**Blocked by:** 01, 02, 08. *(Development can start on the stub as soon as 01 is done.)*

**Status:** done — `afl/attack/multi.py`, nine tests, result in
`artifacts/abcd/amlworld_gather-scatter.json`. The result is a **negative**: adaptive did not beat
non-adaptive, 4/7 seeds, p = 0.500

- [x] One run searches S1, S2 and S3 together, and budget allocation is a stated configurable
      choice — `uniform`, `search`, `fitness`
- [x] Evasion rate is over fraud rows only, asserted by a test
- [x] Every trial logs params, allocation, evasion, realism penalty, audit score and fitness
- [x] Searched params are clamped to each vector's envelope; a test tries to escape it
- [x] The Optuna-absent fallback is exercised — every optimiser test runs `backend="random"`
- [x] Runs against the real detector on AMLworld and the evasion trajectory is recorded per round
      (~0.9 → ~0.1 every seed); the audit gate rejected 0 of 42 rounds
- [ ] Comparison against the *single-vector* loop — **not run on this fold**. The A/B/C/D brief
      replaced it with adaptive-vs-template (D vs C), which is reported in full including its
      failure to reach significance. The original comparison was superseded, not skipped quietly
---

# 13: M1 — the optimiser's boundary walk as a vector

**Owner:** ▲ A

**What to build:** Boundary probing surfaced as a first-class attack vector: the optimiser's own
search trace, showing an attacker mapping the model's approve/decline boundary and walking along
the safe side of it.

This costs almost nothing to build because it *is* the optimiser's search strategy — the vector
is the mechanism, made visible. It is also the cleanest "attack against your own AI" in the
taxonomy, which is what upgrades the taxonomy from a list of frauds to a model of the domain.

The artefact is the point: activity clustered just under a learned threshold, with the trace
showing how the attacker found where the threshold was.

**Blocked by:** 12.

**Status:** ready-for-agent

- [ ] M1 appears in the vector registry as its own vector, with its mechanism documented
- [ ] The boundary walk is logged as a trajectory: probe, response, next probe
- [ ] An artefact shows attack params converging toward the detector's decision boundary
- [ ] It is presented at the third taxonomy level — attacks against our own AI — never flattened
      in with the fraud mechanisms
- [ ] The trace regenerates from logs; nothing is hand-drawn

---

# 14: Realism leash, reported every round

**Owner:** ▲ A

**What to build:** Every loop round reports how realistic the attacker's traffic still is,
alongside how well it evaded — so that an optimiser winning by cheating is visible immediately
rather than at write-up time.

Without this, the optimiser wins by producing traffic no bank would ever see: negative dwell
times, amounts to the paisa, chains no money launderer would run. Recall lift bought against
absurd traffic means nothing, and this is the answer to the obvious question — *how do you stop
it generating unrealistic attacks?*

Cheap checks run every round: schema validity, cross-row rules, empirical distribution bounds,
and clamping to the declared envelope. The expensive verdict is ticket 15's job.

**Blocked by:** 12.

**Status:** ready-for-agent

- [ ] Each round logs a realism penalty and any violations by name, next to the evasion rate
- [ ] Cross-row rules enforced: no self-transfers, no duplicate ids, no unlabelled fraud rows,
      no provenance on legit rows
- [ ] Empirical bounds derived from the real anchor, not hard-coded guesses
- [ ] A deliberately absurd param set is caught by the leash in a test
- [ ] λ is config, and its effect on the search is demonstrated with a run at two values
- [ ] Realism penalty over rounds is plottable next to evasion rate — the "is it cheating?" chart

---

# 15: Fidelity scorecard on real anchor data

**Owner:** ■ B

**What to build:** One scorecard covering all three fidelity levels plus privacy evidence,
comparing generated traffic against the real anchor, emitted as a committed artefact every run.

The levels are not equal and the scorecard must not pretend they are. **Level 3 (utility) is the
gate:** does training on it improve detection on a held-out family and on future time periods?
A generator that resembles real traffic but teaches a model nothing has failed, however pretty
its histograms. Levels 1 (KS, Wasserstein, correlation delta) and 2 (graph motifs, velocity
match, α-precision / β-recall) are diagnostics that explain *why* Level 3 failed.

Thresholds are set before any result exists. Moving one afterwards is how a bar stops being one.

**Blocked by:** 06, 08.

**Status:** done — bars in `config/fidelity/thresholds.yaml`, harness in `afl/fidelity/`,
scorecards built by `make fidelity`, evidence in `artifacts/fidelity/`, written up in
`docs/fidelity.md`. **Both anchors FAIL, and the gate is what fails them.** On PaySim a detector
trained on the generated fraud reaches PR-AUC 0.005 against real fraud, an order of magnitude
below the 0.057 amount floor, and adding those rows to a real training set drops recall at 1% FPR
from 0.444 to 0.215. Level 1 passes on the same card at 0.749 and rescues nothing, which is the
ticket's thesis arriving as a measurement rather than as a design note. Running it also found two
measurement bugs that had been flattering the generator — see the last box below.

- [x] All three levels computed against the real anchor and written to a committed artefact —
      `make fidelity` runs every data config that names a loader, at that anchor's committed
      split and the operating point in `config/eval/leave_one_attack_out.yaml`, and writes
      `artifacts/fidelity/<anchor>.json` and `.md` plus the generated `docs/fidelity.md`.
      Levels 1 and 2 run **twice**: the headline compares generated fraud against the anchor's
      own labelled fraud, because that is the only part of the batch an anchored run injects,
      and the whole batch against the whole anchor is reported underneath as the reading the
      phrase usually has
- [x] Thresholds are recorded before results are generated, and the record shows they predate
      them — the bars moved out of six bare floats in `config/config.yaml` into
      `config/fidelity/thresholds.yaml`, one bar per stated reason, refused at load when the
      reason is blank (`ThresholdError`, the rule `CostModel.from_config` already applies). Each
      names the commit it was first committed in, and `afl/fidelity/provenance.py` reads that
      commit **back out of git** on every run, compares the value committed there against the
      value being applied now, and writes the comparison into the artefact next to the verdict.
      Six of the seven bars trace to `6989a9e`, the day-one skeleton, unchanged; the seventh was
      added by this ticket and committed in `c55dc08` **before the first anchored number
      existed**. An uncommitted edit to the file makes the record say `UNPROVEN` rather than
      claim an age it cannot show
- [x] Level 3 gates the verdict; a Level 1/2 pass cannot rescue a Level 3 fail — enforced twice.
      `_judge` sorts findings into hard (level 3, privacy) and soft (levels 1 and 2), and only
      hard findings can fail a card. And the headline `score` is now **capped at the level-3
      score**: weighting it double still let two diagnostic levels at 1.0 average a level-3 0.1
      up to 0.62, which reads like half a pass. A test asserts both on the same card
- [x] TSTR gap and augmentation lift both measured on real held-out data at the standard
      operating point — four systems, one real test window, one operating point: `trtr` (real
      rows, real labels), `tstr` (real legit + generated fraud, no real fraud label), `augmented`
      (real + generated fraud) and the **amount floor**, which is new here and is a hard bar.
      What "train on synthetic" means is written into the thresholds config *before* the run, so
      the gate could not be swapped afterwards for whichever of the two readings scored better;
      the literature's standalone reading is measured beside it and never gates
- [x] DCR and MIA reported as evidence against memorisation, phrased as evidence, not proof —
      and the MIA gained the control it needed. On an out-of-time split, members and non-members
      differ by *when* as well as by membership, so the same attack is run between two halves of
      the holdout, where nothing was ever in training; an advantage at or below that control is
      reported as drift rather than flagged as a leak. Identifier reuse is counted separately,
      because the generator stages attacks on real accounts by design and DCR cannot see that
      path
- [x] The scorecard regenerates by one command — `make fidelity`. The day-one discrimination
      check keeps its own, `make fidelity-selftest`, and still passes all four of its checks
- [x] A failing scorecard is reported, never quietly re-run with looser thresholds — the
      artefacts and the doc are written *before* the non-zero exit, so a FAIL is a committed
      result. If a bar is ever moved, the provenance block names it, states its direction with
      LOOSENED in capitals, and lists every commit that has ever changed one
- [x] `make test` green (316 passed, up from 305 — 11 new tests in `tests/test_eval.py`),
      `ruff check` and `ruff format` clean, and `make fidelity` runs both anchors end to end in
      about eight minutes
- [x] **Not on the list, and the reason the harness had to be run rather than reviewed:** three
      bugs, two of them flattering the generator. The privacy and support metrics standardised
      by `std + 1e-9`, and three of the seven embedding columns are *exactly constant* on PaySim
      — the anchor has no sender history — so a synthetic row with a real one was divided by a
      billionth and the first PaySim card reported a DCR ratio of **1.0e11**, passing the
      memorisation check on the strength of it. Constant columns are dropped and named now, and
      the ratio is 2.43 over four real dimensions. Membership inference on an out-of-time split
      measures the calendar as well as membership, so it gained the control described above. And
      `_judge` scored a level that never ran as 0.0 and then indexed it for its worst column,
      inventing a finding about a measurement nobody took

---

# 16: The three-system table

**Owner:** ■ B · *The slide that beats a bigger team.*

**What to build:** Real-only vs SMOTE vs the adaptive loop — three systems, one holdout, one
operating point, with the held-out column being the one that decides everything.

System B (SMOTE) exists to make System C falsifiable. Row-level oversampling can move an amount
and a timestamp; it cannot invent a new fan-in shape, a new pacing strategy, or a beneficiary
that never existed — which is precisely the gap the adaptive system claims to fill. If C does not
beat B on the held-out family, the project reduces to an expensive way of duplicating rows, and
this table is where that shows up.

The table must be able to say the adaptive loop lost. Today, on the placeholder config, it does.

**Blocked by:** 11, 12.

**Status:** done — table in `afl/evaluation/three_system.py`, built by `make table`
(`scripts/build_three_system.py`), evidence in `artifacts/three_system/`, written up in
`docs/three_system.md`. **The adaptive loop appears to beat SMOTE by +0.76 recall on the held-out
family, and the number does not survive its own audit.** A model handed System C's training rows
and told *only which rows the generator wrote* — never which are fraud, never a row of the
held-out family — scores **0.995** on that column against System C's **0.998**. Provenance alone
reproduces the result, so System C's held-out cell is withheld on both anchors. On the one column
that is measurable, PaySim's real labelled fraud, the three systems sit inside each other's
seed-to-seed spread: 0.162 / 0.166 / 0.163 PR-AUC.

- [x] All three systems trained and scored on the same holdout at the same fixed operating point —
      one `Fold.carve` per seed against the committed boundary, one `detector_factory` and one
      `fit` for all three rows *and* for the loop's own detector, and the operating point read
      from `config/eval/leave_one_attack_out.yaml` rather than passed per system. "The same
      holdout" is an assertion rather than a claim: `assert_same_haystack` refuses two columns
      that do not share their negatives, because recall at a fixed FPR is a quantile of them
- [x] Both columns reported: known attacks and unseen attacks — `unseen` is the held-out family
      nobody trained on, `known` is the fraud everybody did: the anchor's own labelled rows,
      scored on the same window against the same haystack. Defined by what reached training
      rather than by what the pool happens to carry, so the families in the pool that nobody
      trained on land in **neither** column — they are not the claim and not the control, and
      counting them as negatives would label real fraud as legit traffic
- [x] The table regenerates from run logs by one command, with no hand-entered numbers —
      `make table` runs it; `--doc-only` rebuilds `docs/three_system.md` from the committed
      artefacts alone, so the document is a pure function of them. Every number in the doc and in
      the README's current-numbers section traces to `artifacts/three_system/<anchor>.json`,
      which carries the eval config as read, the committed split digest, every seed, per-column
      row counts, four guards, both probes, the loop's per-round history and each system's model
      card. Versioned, and an old file raises rather than being read with the wrong meaning
- [x] Run-to-run variance is reported, so a small difference is not read as a result — three
      seeds, mean ± sd in every cell, and every comparison **paired by seed** rather than pooled.
      A gap smaller than its own spread prints as *inside the noise* whichever way it points, and
      the sign test says out loud that 3/3 seeds is p = 0.125 at best. The seed turns the whole
      pipeline — pool, SMOTE draw, optimiser search, model — so the spread is the system's, not a
      refit's
- [x] If adaptive does not beat SMOTE on the held-out column, the result is reported as-is and
      the likely reason is stated — `compare()` reports the direction whichever way it falls
      (a test builds a losing table and asserts it reads as a loss), and `diagnose()` derives the
      likely reasons from the run's own logs rather than from a sentence typed after seeing the
      number: a column that cannot carry a claim, a control that reproduced the baseline, a table
      under the amount floor, a loop whose output is a rounding error, an evasion rate that
      collapsed, rounds the audit rejected, a known column at the ceiling, and a held-out score
      that a provenance-only model reproduces
- [x] The README's current-numbers section is refreshed from this run — both anchors, both
      columns, brackets on every withheld cell, and the synthetic `make compare` table it
      replaced is named as the pipeline check it always was
- [x] `make test` green (333 passed, up from 316 — 15 new tests in `tests/test_eval.py`, 2 in
      `tests/test_multi_optimiser.py`), `ruff check` and `ruff format` clean, and `make table`
      runs both anchors end to end in about fifty minutes
- [x] **Not on the list, and the reason the table exists at all:** the check that took the result
      away is not one this ticket was asked for. See the first box below

**Carried out of this ticket, and worth knowing before you start yours:**

- **The one big number in this project is provenance, and it is now measured rather than
  suspected.** Ticket 11's probe learns "injected" from the fold's own hundred-odd positives, and
  its carry-out warned that a low score there is weak evidence of anything. System C learns the
  same thing from ~5,000 generated rows. So this ticket fits the counterfactual with System C's
  advantage — same training rows, labelled only by *who wrote the row*, never shown the held-out
  family — and asks it System C's question. It scores **0.998 / 0.993 / 0.994** on AMLSim against
  System C's 0.998, and **0.999 / 0.964 / 0.626** on PaySim against 0.679. The fingerprint
  transfers between families; the recall does not have to. `scripts/build_three_system.py:
  loop_provenance_probe`, and it decides the cell rather than annotating it.
- **On AMLSim the fold probe passes and the loop probe does not, which is the whole argument for
  having both.** The fold probe scores 0.24–0.36 there — under its 0.5 floor — so Systems A and B
  carry quotable held-out numbers on that anchor (0.105 and 0.121 PR-AUC). Only System C's cell is
  withheld. A harness with one probe would have published 0.998.
- **The audit gate has two rules and they disagree by an order of magnitude on a real anchor.**
  Ticket 12's `AUDIT_LIFT_LIMIT * base_rate` has no floor, so the bar falls as the anchor grows:
  a hundred injected rows in a 600k-row anchor put it at ~5e-4 PR-AUC, which log-amount alone
  clears. It would have rejected **71 of the 72 rounds** in this run; `envelope.audit`'s own
  `trivially_separable` verdict — the rule the rest of the repo applies to the same question —
  rejected **none**. The table declares `audit_rule=envelope` in its artefact and records both
  verdicts on every round. **Ticket 14 owns the leash and should settle which rule is the rule**;
  `MultiVectorOptimiser`'s default is unchanged, so nothing else moved.
- **System A here is not the leave-one-attack-out matrix's detector, and the numbers are not
  comparable row by row.** Same fold, same guards, same boundary — different training set: System
  A sees the anchor's real rows alone, where the matrix's detector trains on the whole training
  side, injected families included. That difference is the table's subject, which is why it is not
  a bug and why `docs/three_system.md` says so before it prints a number.
- **The attacker only sees the training window, and that changed the result.** System C's
  simulator is anchored to an envelope measured on the training rows rather than on the whole
  anchor. Generating from the whole-anchor envelope hands the attacker accounts that only exist
  after the split boundary — knowledge of the future wearing a realism setting — and auditing
  those rows against the training window then rejects them for `sender_in_anchor`. On PaySim,
  where no account transacts twice, that rejected *every* batch and made System C a byte-copy of
  System A for a reason that was an artefact of the harness.
- **The loop converges, and convergence is not the same as teaching.** Evasion falls from
  0.86–0.88 to 0.002–0.060 on AMLSim and from 0.35–0.62 to 0.000–0.014 on PaySim over twelve
  rounds, with zero rounds rejected. `diagnose()` reports that as a finding rather than a success:
  a loop whose attacker stops evading is a loop whose kept rows are rows the detector already
  catches.
- **PaySim's known column is the honest headline of this table.** Real labelled fraud, 410
  positives, out of time: A 0.162, B 0.166, C 0.163 PR-AUC, every pairwise difference inside the
  spread. Adding 4,972 generated fraud rows to a training set carrying 369 real ones moved nothing
  that could be measured. AMLSim's known column is at the ceiling for all three (1.000 / 0.996 /
  0.994) against a 0.456 amount floor, which is a statement about that anchor rather than about
  any system on it.
- **What this hands to the tickets that depend on it.** **20** (one command reproduces a headline
  number) should point at `make table` and quote the *known* column or the withheld verdict, never
  the 0.998. **21**'s demo has the same constraint. **22**'s claims audit inherits a hero table
  whose adaptive row is withheld on both anchors, and the honest version of the pitch is the
  refusal, not the number. **13** and **14** each own a piece of what would make the held-out
  column mean something: a vector whose tell is behavioural rather than generated, and a leash
  whose gate is settled.

---

# 17: Sequence model — earn it or report it

**Owner:** ■ B

**What to build:** A GRU or small transformer over per-entity history, applied to the drift arc
(S3 ATO and C1 bust-out), with an honest comparison against the LightGBM baseline on the same
out-of-time split — reported either way.

The sudden-vs-gradual axis is what makes a sequence model earn its seat instead of decorating the
deck. Sudden takeover is easy; gradual drift with no event to anchor on is where per-row features
run out. If the sequence model does not beat LightGBM there, it does not enter the reported table,
and "it didn't help" is a published result, not a deleted branch.

Requires the `deep` extra. Without torch it must raise rather than silently degrade — a silently
degraded model in an ensemble is a number nobody can explain later.

**Blocked by:** 11.

**Status:** done — model in `afl/defend/models/sequence.py`, axis and gate in
`afl/evaluation/drift_arc.py`, built by `make sequence` (`scripts/build_sequence.py`), evidence in
`artifacts/sequence/`, written up in `docs/sequence.md`. **It did not earn its seat, and the two
anchors refused it for opposite reasons.** On AMLSim, whose accounts carry ~28 steps of real
history apiece, it loses on merit: PR-AUC 0.391 on gradual S3 against LightGBM's 0.997, and 0.300
against 0.998 on gradual C1 — at a *lower* compute cost than the baseline, so it is not an
expensive model that bought a small lift but a cheap model that lost. On PaySim it wins C1 by
0.987 to 0.773, and that win is what disqualifies it: `nameOrig` is effectively unique per row, so
a real window there is **one step long** while the injected episodes carry eight or nine, and
window length alone sorts injected from real at PR-AUC 0.933 (S3) and 0.985 (C1). All four folds
are withheld — two on the provenance probe, two on the history audit this ticket added — so the
comparison is published in brackets and quoted nowhere. `config/defend/sequence.yaml` stays
`enabled: false`, and `assert_config_matches_promotion` refuses to let it be turned on while the
committed artefacts say no.

**Carry-out for ticket 11.** These folds carry 550 injected AMLSim C1 rows against the matrix's
80, so the provenance probe is far better powered here — and at that size it separates injected
from real at PR-AUC 0.688, over the bar, where `artifacts/loao/amlsim.json` measured 0.236 and
reported the fold. Nothing about the generator changed; the episode count did. AMLSim C1 is one of
only four measured rows in the whole matrix, and it should be read as underpowered until `make
loao` is re-run at this episode count.

- [x] Trains on per-entity histories and scores through the standard `score` seam — one window per
      *transaction*, ending at that transaction and labelled with that transaction's own label.
      The version this replaced took one window per *entity* labelled
      `any(t.is_fraud for t in window)` and broadcast the score back onto the account's clean
      baseline rows, which is a lookup rather than a detector; `sequence_tensor` is deleted rather
      than deprecated, because leaving it importable leaves the bug importable. Scoring-time
      history crosses the fit/score boundary the way the stateful `FeatureBuilder`'s does, so a
      holdout row whose baseline sits in the training window still has a baseline to drift from
- [x] Compared against LightGBM on the same split at the same operating point — the same
      `Fold.carve` against the committed boundary, the same calibration on the same validation
      tail, the same cost model and bands, and `amount_only` under both of them. The champion is
      `LGBMDetector` rather than the ensemble the loop ships: the ticket's bar is the supervised
      baseline, and a blend would move two things at once
- [x] The sudden-drift vs gradual-drift breakdown is reported separately — each family generated
      twice with `ramp` at the low and high end of *its own* declared search space and nothing
      else changed, so the arcs stay inside ticket 14's realism envelope and C1's gradual end is
      0.6 rather than S3's 1.0. Both arcs are ranked against every legit row of the fold, so only
      the needles change between the two rows; `recall_at_shared_threshold` is computed from the
      whole fold and asserted to match `recall_at_fixed_fpr`, which is what says the axis was read
      at one operating point rather than at two
- [x] Enters the headline table only if it beats the baseline; the comparison is published either
      way — `drift_arc.decide_promotion` gates on the **gradual** end only, because sudden
      takeover is an event a per-row table already sees. It refuses a blocked fold first, then a
      model the amount floor beats, then anything inside `material_gap`. It refused all four, and
      `assert_config_matches_promotion` makes `enabled` answerable to that rather than to memory
- [x] Raises clearly when torch is missing; the default test suite stays green without the extra —
      the constructor calls `require_torch` up front, so a config that enables the layer without
      the extra fails before it spends an hour generating a pool it cannot score. `config/defend/
      sequence.yaml` no longer promises a pooled-sequence fallback; there is none. Only the tests
      that actually build a network are skipped without torch — the window arithmetic, the arc
      breakdown, the gate and the artefact are pure numpy and run on the default install
- [x] Compute cost is reported next to the lift, so the trade is visible — fit and score seconds,
      rows/second and parameter count on the model card and in the same table as the metrics. The
      answer is not the one the ticket expected: the GRU is *cheaper* than LightGBM on both anchors
      (21s vs 32s to fit on AMLSim, ~2.3x the scoring throughput) and still loses where the anchor
      has histories to read

---

# 18: Temporal GNN — earn it or fall back

**Owner:** ■ B

**What to build:** Graph attention over the account-device-beneficiary graph, applied to the mule
family, measured against the hand-rolled graph-features + LightGBM baseline — with the fallback
stated up front.

Mule networks are a graph problem and message passing should beat degree features. But a temporal
GNN is also the easiest place in this repo to produce a number that does not replicate. So it is
never in the headline table unless it beats LightGBM on the same out-of-time split, and if it does
not, hand-rolled graph features are what ship, and the README says so.

**Blocked by:** 11.

**Status:** done — model in `afl/defend/models/gnn.py`, families/audits/gate in
`afl/evaluation/mule_graph.py`, built by `make gnn` (`scripts/build_gnn.py`), evidence in
`artifacts/gnn/`, written up in `docs/gnn.md`. **It did not earn its seat, so the stated fallback
is what ships — and the two anchors refused it for opposite reasons.** On AMLSim it loses by
almost the whole scale (PR-AUC 0.023 ± 0.038 on S1 against LightGBM's 0.998, 0.002 against 0.984
on C3, 0/3 seeds) at 1.6x the fit cost — and that fold is withheld anyway, on a precondition
nobody expected to be the deciding one: **AMLSim's rows are whole days**, so `Simulator.generate`
quantises every injected fraud row to midnight to stay commensurable, an entire mule ring lands on
one timestamp, and a model that may only read strictly-earlier edges was asked about a shape it
structurally cannot see. Only 8.2% of injected S1 rows (0.6% for C3) can see any earlier edge of
their own episode, against a floor of 20%. On PaySim, whose clock is hourly and where 86% can, the
GNN is **level with what ships** (−0.023 ± 0.229 on S1 at 1/3 seeds, +0.024 ± 0.343 on C3 at 2/3,
both inside their own spread) and ahead of the graph-blocks-only baseline by +0.079 and +0.139 —
and those folds are withheld too, on the audit this ticket built for this model: a quarter to a
third of the injected rows sit in a neighbourhood made only of other injected rows, and that share
alone sorts injected from real at PR-AUC 0.53–0.54. `config/defend/gnn.yaml` stays `enabled:
false`, and `assert_config_matches_promotion` refuses to let it be turned on while the committed
artefacts say no.

**Carry-out for ticket 11.** Same shape as ticket 17's. These folds carry 173 injected PaySim S1
rows against the matrix's 50, and at that size the provenance probe separates injected from real
at PR-AUC 0.865 — over the bar — where `artifacts/loao/paysim.json` measured 0.298 and *reported*
the fold. Nothing about the generator changed; the episode count did. PaySim S1 is one of only
four measured rows in the whole matrix, and it should be read as underpowered until `make loao` is
re-run at this episode count. That is now two families flagged this way from two different
tickets, so it is a property of `eval.holdout_episodes` rather than of one vector.

- [x] Builds a temporal graph with an explicit window; edges older than the window are dropped —
      time is cut into `stride_hours` steps and a payment is scored against the edges in
      `[stride_start - window, stride_start)`. Nothing in a snapshot is at or after the stride it
      scores, so no row informs its own score and no later row can move an earlier one; a test
      appends future traffic and asserts the earlier scores do not budge. Self-payments are
      dropped and counted, because `GATConv` strips them before adding its own and the shift
      would silently misalign the attention the reason codes are read through
- [x] Scores through the standard `score` seam — one score per *transaction*, labelled with that
      transaction's own label, from its two endpoints' embeddings and its own row. The version
      this replaced labelled the *node*: every beneficiary of a fraud row was marked positive and
      the node's score was broadcast back onto that account's legitimate inbound payments, which
      is a lookup rather than a detector. Scoring-time history crosses the fit/score boundary the
      way the stateful `FeatureBuilder`'s does, and the attention weights become the local
      explanation `explain.assert_flagged_rows_are_explained` demands — including an honest
      refusal to narrate a ring over a beneficiary that is an isolated node
- [x] Compared against graph-features + LightGBM on the same split at the same operating point —
      the same `Fold.carve` against the committed boundary, the same calibration on the same
      validation tail, the same cost model and bands, and `amount_only` under all of them. **Two**
      baselines are reported: `lgbm` over the whole hand-rolled table, and `graph_lgbm` over the
      graph blocks alone (`features.graph_feature_names`). The gate is decided on the *former* on
      purpose — a challenger promoted over a deliberately narrowed champion is a number that does
      not survive contact with the deployed system, and this is the easiest place in the repo to
      manufacture a lift that way. The narrower column is published because "it loses to graph
      features" and "it loses to the velocity block next to them" are different findings
- [x] Lift is reported with variance across seeds — three seeds per fold, each regenerating its
      own pool *and* refitting every system, so the spread covers the attacker's draw and not
      only the network's initialisation. The margin is a paired per-seed difference with its
      standard deviation and a sign test, reusing ticket 16's `Spread` and `Comparison` rather
      than a second copy of them, so the hero table and this experiment are read at one bar. The
      gate refuses one seed outright and refuses any margin smaller than its own spread — which
      is what both PaySim folds fell to
- [x] The documented fallback is what ships if the lift is not there, and the README says which
      one shipped — `TemporalGNNDetector.fallback()` returns the hand-rolled detector, every
      refusal branch names it in `Promotion.shipped`, `GNNReport.shipped` is in the artefact, and
      a test reads `README.md` and asserts the sentence there agrees with the committed evidence.
      The claim about what is deployed is enforced rather than remembered
- [x] Raises clearly when the deep extra is missing; default suite stays green — the constructor
      calls `require_deep` up front, so a config that enables the layer without torch and
      torch-geometric fails before it spends an hour generating a pool it cannot score. Only the
      tests that actually build a network are skipped without the extra: the window arithmetic,
      the two audits, the seed aggregation, the gate and the artefact are pure numpy and run on
      the default install

---

# 19: The convergence artefact

**Owner:** ▲ A · *The hero artefact.*

**What to build:** The chart from the architecture doc's header, regenerated from run logs:
attacker evasion rate spiking as it finds a way through, detector recall recovering after each
retrain, both trending as the loop closes — with the retrain points marked.

This is the single image the whole project exists to produce. It has to come from logs, by
command, with no hand-drawing and no smoothing that hides a bad round. The numbers need not be
spectacular; a measurable adaptive loop *existing* is the win.

**Blocked by:** 11, 12.

**Status:** done — `make figures` draws from `artifacts/abcd/`, the committed 7-seed A/B/C/D run
on amlworld holding out GATHER-SCATTER. No re-run: the figure is a read of a logged result.

- [x] `make figures` regenerates the curve from logged runs alone — reads the committed artefact,
      runs no model, and is the only way these images are produced.
- [x] Both series plotted: evasion rate and held-out recall @ 1% FPR, all 7 seeds drawn faintly
      under the bold mean so a bad round stays visible instead of being averaged away.
- [x] Retrain points marked — the detector refits at every round boundary, and the marks say so.
- [x] The per-round numbers are written beside the figure as `..._convergence.md`, mean ± sd plus
      every seed's raw evasion trace, so a curve can be checked against its values.
- [x] Axes and operating point labelled, and the fold's own facts are baked into the image
      caption — 173 positives, base rate 0.0532%, split `f5e33a878d68b792` — because a figure
      gets screenshotted away from the README that carries its caveats.
- [x] Regenerating twice produces an identical figure, verified by sha256 and held by a test.
      Matplotlib stamps its own version into the PNG, which would have broken this silently.
- [x] The curves do not converge, and the figure ships. Evasion falls 0.915 → 0.201 over six
      rounds on all 7 seeds; held-out recall stays flat and noisy. **The loop closes on the
      attacker; what does not materialise is the transfer to a held-out family** — the same
      thing the A/B/C/D null says, in the picture rather than the table.
- [x] **Not on the list, and the reason ticket 14 now has a second job:** the same pass drew the
      realism leash, and it is pinned at 0.065 ± 0.001 in 41 of 42 rounds. Two of `realism.py`'s
      three soft terms never fire and the third sits at its ceiling, so the penalty is a constant
      — and a constant subtracted from every trial leaves the argmax alone. **λ is currently a
      no-op on the search.** Reported here rather than fixed, because fixing it changes the
      committed A/B/C/D result and that is a decision, not a patch.

---

# 20: One command reproduces a headline number

**Owner:** ▲ A · *The stated differentiator.*

**What to build:** A fresh clone, one command, one headline number that matches what is written
down. Plus the artefact discipline that makes every other number traceable to the run that made it.

Reproducibility is what separates this from a demo, and it is the cheapest credibility available.
Every attack run saves its parameter set and resulting metrics. Every run writes its config and
seed. A number that cannot be reproduced from (config, seed) is a bug, not a rounding difference.

**Blocked by:** 16, 19.

**Status:** ready-for-agent

- [ ] `docker compose up` or a single `run_experiment.py` invocation reproduces a stated headline
      number from a clean clone
- [ ] Global seed set; the same seed twice gives the same number, and this is verified on a
      machine that was not used to develop it
- [ ] Every run writes config, seed, attack params and metrics into the artefact directory
- [ ] Dependencies pinned; the README's commands are exact and were run verbatim to check
- [ ] Every number in the README traces to a committed artefact
- [ ] Where determinism is not achievable, the residual variance is stated rather than glossed
- [ ] The whole thing still runs on the synthetic default with nothing to download

---

# 21: The live demo

**Owner:** ▲ A

**What to build:** Pick an attack, run the loop, watch the evasion and recall curves move — over
FastAPI and Streamlit, on one command, with a recording as backup.

The demo is not a separate implementation. Every endpoint drives the same objects the experiment
scripts drive, so anything shown live is reproducible by `make loop`. A demo that runs its own
private code path is a sales tool, not evidence.

Rounds must return while someone is watching, so the demo's lab instance is deliberately small —
and the fact that it is smaller than the reported runs is stated on screen, not hidden.

**Blocked by:** 19.

**Status:** ready-for-agent

- [ ] `docker compose up` brings up API and UI together, first try, on a machine that has never
      run it
- [ ] The UI runs a loop round and updates the curves live
- [ ] Reason codes are shown for flagged transactions, so the explainability story is visible
- [ ] The demo drives the same objects as the experiment scripts — no private code path
- [ ] A round completes fast enough to hold attention, and the reduced scale is labelled on screen
- [ ] A recording of a successful run is committed as the backup
- [ ] With ScamShield removed, the experiment still fully runs and reproduces

---

# 22: Submission pack and claims audit

**Owner:** ◆ A+B, together.

**What to build:** The write-up, with every claim checked against what the artefacts actually
support, and every deliberate cut named as a decision.

Go through the results line by line and strike anything the runs do not support. The guardrails
are not decoration — they are what moves the work from a seven to a nine, because a reviewer who
finds one overstatement stops trusting the rest:

- C2ST near chance is a diagnostic under our protocol, not proof of realism.
- DCR / MIA are evidence against memorisation, not proof of privacy.
- Synthetic-only reduces exposure; it does not automatically mean DPDP compliance.
- Latency depends on the decision point — no fixed sub-50ms claim.
- AI-risk classification depends on deployment context — no blanket carve-out.
- Projected losses are labelled projections, never realised figures.
- Frontier vectors are demonstrated capabilities, not necessarily mass-exploited patterns.

And the story: not "we built an AI fraud platform", but "we tested whether a closed
attacker–defender loop makes a detector more robust to attacks it has never seen, and here is the
reproducible curve".

**Blocked by:** 15, 16, 20, 21.

**Status:** ready-for-agent

- [ ] Every claim in the write-up traces to a committed artefact, checked one by one
- [ ] All seven guardrails applied to the actual wording, not just listed
- [ ] The deliberate cuts are stated as scope decisions: multimodal/deepfake, LLM
      social-engineering generator, multiple tabular generators, mandatory GNN, full RL,
      agent-swarm simulation
- [ ] The taxonomy is presented at three levels, never flattened
- [ ] Any result contradicting the hypothesis is reported plainly, in the same voice as the
      supporting ones
- [ ] Both owners have re-run the headline command on their own machine and got the stated number
- [ ] A reviewer given only the repo and the README can reproduce the headline number unaided
