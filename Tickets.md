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
| 02 | Anchor on a real dataset with a committed split | ■ B | — |
| 03 | M3 — first-party fraud, the holdout family | ▲ A | 01 |
| 04 | C2 — UPI collect-request / APP scam | ▲ A | 01 |
| 05 | C3 — instant-A2A pass-through | ▲ A | 01 |
| 06 | C1 + M2 — bust-out and the synthetic-identity lifecycle | ▲ A | 01 |
| 07 | Causal features on real anchor data | ■ B | 02 |
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

A real dataset (PaySim, IEEE-CIS, or both) enters through the loaders and leaves as
`list[Transaction]`. Nothing downstream may know which dataset it came from — that is what makes
one detector, one feature set and one evaluation run over all three sources unchanged. Real rows
carry no provenance: a real row that gains a `vector_id` has leaked a label path.

The out-of-time split with its embargo gap is computed once and committed as an artefact, not
regenerated per run. A short data card records what the dataset is, its base rate, its time span,
its known quirks, and what it cannot tell us.

**Blocked by:** None (can start immediately, in parallel with 01).

**Status:** ready-for-agent

- [ ] At least one real dataset loads end to end into contract types and the whole pipeline runs
      on it via a config override, no code change
- [ ] Real rows have `vector_id=None` and `attack_run_id=None`; a test asserts this
- [ ] The out-of-time split boundary is committed as an artefact and re-used, and re-running
      produces an identical partition
- [ ] The embargo gap is non-zero and its rationale is recorded
- [ ] A data card exists: source, licence, row count, fraud base rate, time span, quirks, limits
- [ ] The synthetic placeholder config still works with nothing to download
- [ ] Fraud base rate on the real anchor is reported; if it differs from the synthetic default by
      an order of magnitude, that is called out, because every operating point depends on it

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

**Status:** ready-for-agent

- [ ] Features build over the real anchor dataset at a workable speed, and the timing is recorded
- [ ] Property test: a feature computed for a row does not change when later rows are appended
- [ ] Test: no forbidden column (`is_fraud`, `vector_id`, `attack_run_id`, `txn_id`) reaches X
- [ ] Stateful mode carries entity history across batches and a test proves accumulation
- [ ] Graph features are computed as-of the row's timestamp, never over the full graph
- [ ] Feature names are stable and human-readable — ticket 09's reason codes depend on them
- [ ] The feature list, with a one-line rationale each, is written down where the next person
      will find it

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
