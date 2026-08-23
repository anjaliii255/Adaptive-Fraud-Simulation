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
| 03 | M3 — first-party fraud, the holdout family | ▲ A | 01 |
| 04 | C2 — UPI collect-request / APP scam | ▲ A | 01 |
| 05 | C3 — instant-A2A pass-through | ▲ A | 01 |
| 06 | C1 + M2 — bust-out and the synthetic-identity lifecycle | ▲ A | 01 |
| 07 | ~~Causal features on real anchor data~~ **done** | ■ B | 02 |
| 08 | LightGBM baseline at a fixed operating point | ■ B | 07 |
| 09 | Graded decisions and SHAP reason codes | ■ B | 08 |
| 10 | Anomaly layer scored on the holdout family | ■ B | 08, 03 |
| 11 | Leave-one-attack-out harness with leakage guards | ■ B | 08, 03 |
| 12 | Multi-vector adaptive optimiser | ▲ A | 01, 02, 08 |
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

**Status:** ready-for-agent

- [ ] M3 generates through `Simulator.generate` with no engine edit — YAML plus existing engine
      parameters only
- [ ] The pre-abuse history is labelled legit; only the abuse rows are labelled fraud, because
      that is what an investigator would call it
- [ ] No device change, no new operator signal, no ring structure — if a hand-rolled rule catches
      it easily, the vector is not doing its job and that is noted
- [ ] Declared `search_space` with bounds that keep the family plausible
- [ ] Batches pass the realism check with no violations
- [ ] A test in `tests/test_engines.py` asserts on the emitted batch: labelling, provenance,
      schema validity, and the shape that makes it first-party
- [ ] B is told it is ready, because ticket 10 and 11 unblock on it

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

**Status:** ready-for-agent

- [ ] C2 generates through `Simulator.generate`, on the UPI rail, with no engine edit
- [ ] Emitted rows exhibit a first-time payee, an amount atypical for the source account, and a
      rapid drain — verifiable from the batch alone
- [ ] The victim account's prior history is present and labelled legit, so the atypicality is
      measurable rather than asserted
- [ ] Declared `search_space` with bounds
- [ ] Realism check passes with no violations
- [ ] Test in `tests/test_engines.py` asserting on the emitted batch

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

**Status:** ready-for-agent

- [ ] C3 generates through `Simulator.generate` with no engine edit
- [ ] Emitted batches show inbound-then-immediate-outbound at a beneficiary with no prior
      inbound history
- [ ] Dwell time between inbound and outbound is a searchable parameter with bounds — it is the
      knob that trades detectability for realism, and instant pass-through is loud
- [ ] Cross-institution structure is represented in the emitted graph
- [ ] Realism check passes with no violations
- [ ] Test in `tests/test_engines.py` asserting on the emitted batch

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

**Status:** ready-for-agent

- [ ] C1 generates a long legit tenure followed by a correlated spike, with only the bust-out
      window labelled fraud
- [ ] The correlation across accounts is present in the emitted batch, not just implied
- [ ] M2 generates the seasoning phase and hands off into the bust-out arc, reusing C1 rather
      than duplicating it
- [ ] The M2 account has no prior real history by construction — that is the tell, and the
      emitted batch must make it observable
- [ ] Both declare bounded `search_space`s; both pass the realism check
- [ ] Tests in `tests/test_engines.py` for both, asserting on emitted batches
- [ ] Neither required an engine edit; if one did, the engine was under-parameterised and that
      is fixed here

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

**Status:** ready-for-agent

- [ ] Trains and scores on the real anchor through the `score` / `retrain` seam
- [ ] PR-AUC, recall @ fixed FPR and precision@k reported on the out-of-time split
- [ ] Accuracy and ROC-AUC are not reported as headline numbers anywhere
- [ ] The operating point is fixed in config and every comparison uses it
- [ ] Which backend was used (LightGBM vs sklearn fallback) is recorded in the run artefact
- [ ] `retrain` accumulates rather than replacing, and a test proves it
- [ ] Evasions are weighted above ordinary training rows, and the weight is config, not a literal
- [ ] Baseline numbers are committed as the reference every later ticket compares against
- [ ] A is told it is ready, because ticket 12 unblocks on it

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

**Status:** ready-for-agent

- [ ] Actions are chosen by expected cost under the cost model, not by hand-set cut-offs
- [ ] The cost model's parameters live in config with a stated rationale per number
- [ ] Every flagged transaction carries at least three reason codes in analyst language
- [ ] The SHAP-unavailable fallback is labelled in the reason string itself
- [ ] Changing a cost parameter visibly moves the action mix, demonstrated in a test
- [ ] The evasion definition still holds: only `allow` counts as evaded, and a test asserts it

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

**Status:** ready-for-agent

- [ ] Fits on legit rows only; a test asserts no fraud row entered training
- [ ] Scores through the same `score` seam and returns graded actions like every other detector
- [ ] Recall on the held-out M3 family reported side by side with the supervised model's
- [ ] Ensemble behaviour with the supervised model is defined and measured, not assumed
- [ ] If supervised recall collapses on M3 and the anomaly layer does not, that is written up
      as a result rather than buried

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

**Status:** ready-for-agent

- [ ] One loop run searches across S1, S2 and S3, and how budget is allocated between them is a
      stated, configurable decision
- [ ] Evasion rate is computed over fraud rows only, and a test asserts it
- [ ] Every trial's params, evasion rate, realism penalty and fitness are logged
- [ ] Searched params are clamped to each vector's declared envelope; a test tries to escape it
- [ ] The Optuna-absent fallback path is exercised in CI
- [ ] Runs against the real detector, not just the stub, and the resulting evasion trajectory is
      recorded
- [ ] The comparison against the single-vector loop is reported — including if it does not help

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
