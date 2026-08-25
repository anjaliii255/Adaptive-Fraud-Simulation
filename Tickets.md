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
| 11 | Leave-one-attack-out harness with leakage guards | ■ B | 08, 03 |
| 12 | ~~Multi-vector adaptive optimiser~~ **done** | ▲ A | 01, 02, 08 |
| 13 | M1 — the optimiser's boundary walk as a vector | ▲ A | 12 |
| 14 | Realism leash, reported every round | ▲ A | 12 |
| 15 | Fidelity scorecard on real anchor data | ■ B | 06, 08 |
| 16 | The three-system table | ■ B | 11, 12 |
| 17 | Sequence model — earn it or report it | ■ B | 11 |
| 18 | Temporal GNN — earn it or fall back | ■ B | 11 |
| 19 | The convergence artefact | ▲ A | 11, 12 |
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

**Status:** ready-for-agent

- [ ] Any vector can be named as the holdout via config
- [ ] Assertion: zero rows of the held-out family in training, replay buffer included, and the
      assertion fires in a test that deliberately tries to leak one
- [ ] Assertion: the split is still out-of-time with the embargo intact after the carve-out
- [ ] All legit rows are retained in the holdout
- [ ] A fold with too few positives to be meaningful is reported as such, never as a low score
- [ ] Skipped folds are named, with the reason, in the output
- [ ] Results write to an artefact with the config and seed that produced them

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

**Status:** ready-for-agent

- [ ] All three levels computed against the real anchor and written to a committed artefact
- [ ] Thresholds are recorded before results are generated, and the record shows they predate them
- [ ] Level 3 gates the verdict; a Level 1/2 pass cannot rescue a Level 3 fail
- [ ] TSTR gap and augmentation lift both measured on real held-out data at the standard
      operating point
- [ ] DCR and MIA reported as evidence against memorisation, phrased as evidence, not proof
- [ ] The scorecard regenerates by one command
- [ ] A failing scorecard is reported, never quietly re-run with looser thresholds

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

**Status:** ready-for-agent

- [ ] All three systems trained and scored on the same holdout at the same fixed operating point
- [ ] Both columns reported: known attacks and unseen attacks
- [ ] The table regenerates from run logs by one command, with no hand-entered numbers
- [ ] Run-to-run variance is reported, so a small difference is not read as a result
- [ ] If adaptive does not beat SMOTE on the held-out column, the result is reported as-is and
      the likely reason is stated
- [ ] The README's current-numbers section is refreshed from this run

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

**Status:** ready-for-agent

- [ ] Trains on per-entity histories and scores through the standard `score` seam
- [ ] Compared against LightGBM on the same split at the same operating point
- [ ] The sudden-drift vs gradual-drift breakdown is reported separately — that is the whole point
- [ ] Enters the headline table only if it beats the baseline; the comparison is published either way
- [ ] Raises clearly when torch is missing; the default test suite stays green without the extra
- [ ] Compute cost is reported next to the lift, so the trade is visible

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

**Status:** ready-for-agent

- [ ] Builds a temporal graph with an explicit window; edges older than the window are dropped
- [ ] Scores through the standard `score` seam
- [ ] Compared against graph-features + LightGBM on the same split at the same operating point
- [ ] Lift is reported with variance across seeds, because a single-seed GNN result is not a result
- [ ] The documented fallback is what ships if the lift is not there, and the README says which
      one shipped
- [ ] Raises clearly when the deep extra is missing; default suite stays green

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

**Status:** ready-for-agent

- [ ] `make figures` regenerates the curve from logged runs alone
- [ ] Both series plotted: evasion rate over fraud rows, and held-out recall at the fixed FPR
- [ ] Retrain points marked on the curve
- [ ] The underlying per-round numbers are written next to the figure so it can be checked
- [ ] Axes and operating point labelled; no unlabelled or rescaled axis
- [ ] Regenerating twice from the same logs produces an identical figure
- [ ] If the curves do not cross or converge, the figure still ships and the write-up says so

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
