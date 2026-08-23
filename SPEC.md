# Spec: Adaptive Fraud Simulation Lab — from skeleton to defended claim

**Source:** `docs/architecture.html` (Mastercard Innovation Challenge · GFF 2026)
**Status:** ready-for-agent
**Tickets:** `Tickets.md`

---

## Problem Statement

A fraud team can only train on the fraud it has already seen. When a new attack family
appears — a new mule topology, a new pacing strategy, a scam class that arrives with a new
payment rail — the detector meets it with nothing. The team finds out it was blind at the
same moment the money leaves.

The usual answers do not close that gap. Oversampling the fraud you have (SMOTE and friends)
duplicates the past; it cannot invent a fan-in shape or a payee that never existed. A tabular
generator can produce rows whose histograms match beautifully and whose training value is nil.
And nobody can tell the difference, because the field's habit is to report accuracy or ROC-AUC
on a random split — three separate ways of flattering a model at a sub-2% base rate.

So the question a fraud team actually has — *will this detector catch something it has never
seen?* — normally goes unanswered, and unanswerable.

Concretely, right now: this repo's skeleton runs, but its vector IDs disagree with the
architecture doc, only three of the nine vectors exist, the optimiser searches a single vector,
nothing is anchored on a real dataset, and the adaptive system scores **below** both baselines
on the placeholder config. The claim is unproven and, on today's evidence, unsupported.

## Solution

One closed loop that makes the attacker and the defender co-adapt, and one evaluation honest
enough that the loop can fail in public.

The attacker generates payment fraud from parameterised engines. The detector scores it. Fraud
that was let through untouched is logged as an **evasion**, and that log does double duty: it
becomes the attacker's next brief (search harder in the direction that worked) and the
detector's next training batch (weighted up, because evasions are the expensive examples). Each
round the attacker gets quieter and the detector recovers. That co-adaptation, plotted, is the
hero artefact.

The attacker is not given unconstrained freedom. Its fitness is `evasion_rate − λ · realism_penalty`,
and its parameters are clamped to a declared realism envelope per vector. An optimiser rewarded
on evasion alone discovers traffic no bank would ever see, and recall lift bought against
absurd traffic means nothing.

Honesty is enforced structurally, not by good intentions:

- The split is **out-of-time with an embargo**, never random.
- The headline number is **leave-one-attack-out**: pull one whole attack family out of
  training, then measure recall on it alone. That is the only number that answers the question.
- Three systems, one holdout, one operating point: real-only, real+SMOTE, real+adaptive loop.
  System B exists so System C can fail — if the loop does not beat naive oversampling on the
  held-out family, the project reduces to an expensive way of duplicating rows, and the table
  says so.
- Fidelity is a three-level scorecard where **Level 3 (utility) is the gate**. Levels 1 and 2
  explain why Level 3 failed; they never override it.

Two people build it, working separately, meeting only at `afl/contract`.

## User Stories

**The fraud team (the user of the resulting detector)**

1. As a fraud analyst, I want the detector to return a graded action — allow / step-up / hold /
   review / decline — so that I am not forced to choose between blocking a good customer and
   letting fraud through.
2. As a fraud analyst, I want every flagged transaction to carry reason codes in plain language,
   so that I can action an alert instead of interpreting a probability.
3. As a fraud analyst, I want the alert volume expressed as precision@k against my actual review
   budget, so that I know what my queue looks like tomorrow morning.
4. As a fraud operations lead, I want the decision thresholds derived from a cost model, so that
   "what false-positive rate do we run at" is a business decision with an auditable answer
   rather than a number someone picked.
5. As a fraud operations lead, I want to know the detector's recall on an attack family it was
   never trained on, so that I can size the risk of the next novel scam instead of assuming
   yesterday's recall carries over.
6. As a risk owner, I want to see mule-network behaviour caught from structure, so that a ring
   whose every individual transaction looks ordinary is still stopped.
7. As a risk owner, I want first-party fraud treated as its own family, so that the case where
   the customer *is* the fraudster does not silently break the model's assumptions.

**The red-side engineer (Person A)**

8. As the red-side engineer, I want to add an attack vector by editing YAML alone, so that the
   taxonomy can grow without touching an engine — and so that needing an engine edit is a signal
   the engine is under-parameterised.
9. As the red-side engineer, I want every attack vector to declare a search space with bounds,
   so that the optimiser has a realism envelope it cannot leave.
10. As the red-side engineer, I want the optimiser to search across several vectors within one
    run, so that the loop finds the detector's weakest surface instead of over-fitting one.
11. As the red-side engineer, I want a per-round realism report alongside the evasion rate, so
    that I can see immediately when the optimiser is winning by cheating.
12. As the red-side engineer, I want every generated batch to carry its params and its seed, so
    that any row in any result can be regenerated exactly.
13. As the red-side engineer, I want the optimiser's boundary walk logged as a first-class
    attack vector, so that "attack against our own model" is demonstrated rather than asserted.
14. As the red-side engineer, I want to develop against a stub detector, so that my lane is not
    blocked while the blue side is still building.

**The blue-side engineer (Person B)**

15. As the blue-side engineer, I want every feature computed strictly from prior rows, so that
    the offline table cannot flatter a model that would be useless in deployment.
16. As the blue-side engineer, I want provenance fields (`is_fraud`, `vector_id`,
    `attack_run_id`) structurally barred from the feature matrix, so that no model can score
    1.0 by reading the answer key.
17. As the blue-side engineer, I want the out-of-time split committed rather than regenerated,
    so that two runs a week apart are comparable.
18. As the blue-side engineer, I want leave-one-attack-out to assert that not one row of the
    held-out family reached training — replay buffer included — so that the headline number
    cannot quietly become a memorisation score.
19. As the blue-side engineer, I want the LightGBM baseline to be genuinely strong, so that the
    exotic layers have something real to beat.
20. As the blue-side engineer, I want the sequence model and the GNN to report their comparison
    against LightGBM either way, so that "it didn't help" is a published result rather than a
    deleted branch.
21. As the blue-side engineer, I want an unsupervised layer trained on legit traffic only, so
    that there is an honest floor under the ensemble when the held-out family has no labels.
22. As the blue-side engineer, I want every system in the table measured at the same fixed
    operating point, so that no comparison is a threshold artefact.

**The judge / reviewer**

23. As a judge, I want one command to reproduce a headline number from a fresh clone, so that I
    can distinguish a result from a slide.
24. As a judge, I want the convergence curve regenerated from run logs rather than drawn, so
    that the hero artefact is evidence.
25. As a judge, I want to see the three-system table on the held-out column, so that I can tell
    whether the adaptive loop earned its complexity.
26. As a judge, I want fidelity claims stated as diagnostics, so that a C2ST near chance is not
    sold to me as proof of realism.
27. As a judge, I want the scope cuts named deliberately, so that I can tell a decision from a
    gap.
28. As a domain judge, I want the taxonomy presented at three levels — fraud mechanisms, fraud
    enablers, attacks against the AI itself — so that it reads as a model of the domain rather
    than a flat list.
29. As a judge, I want a result that contradicts the hypothesis reported plainly, so that I can
    trust the results that support it.

**The reproducer (anyone re-running this later)**

30. As a reproducer, I want a fresh clone to run end to end with nothing to download, so that
    setup failure and result failure are distinguishable.
31. As a reproducer, I want every run to write its config, seed, params and metrics into an
    artefact directory, so that a number in the README can be traced to the run that produced it.
32. As a reproducer, I want the demo to drive the same objects the experiments drive, so that
    what I see live is the thing that was measured.

## Implementation Decisions

### The seam — the one thing both sides share

`afl/contract` is the entire shared surface. The red side (`afl/attack`) and the blue side
(`afl/defend`, `afl/evaluation`) never import each other; they both import the contract, and the
loop (`afl/loop`) is written against Protocols so either half can be swapped stub-for-real
without touching it.

The seam is already frozen and stays frozen:

- `Transaction`, `Entity`, `AttackParams`, `AttackBatch` — the request half.
- `DetectorScore`, `Action`, `MetricResult`, `EVASION_ACTIONS` — the return half.
- `find_evasions(batch, scores)` matches by `txn_id`, never by position, and raises when the
  detector returns fewer scores than transactions. A silent zip-misalignment here would corrupt
  every number downstream.
- An attack **evades** only if it was let through untouched (`Action.ALLOW`). Anything
  friction-bearing counts as caught. Both sides must agree on this, so the definition lives in
  the contract, not in either half.

Any change to these types is a joint decision, made in ticket 01 and not after.

### Seams under test

Testing happens at the **highest** seam available, and prefers existing seams:

- **`Simulator.generate(params) -> AttackBatch`** — one seam covers every attack vector. A
  vector test asserts on the emitted batch (shape, labelling, provenance, schema validity), not
  on engine internals.
- **`Detector.score(batch) -> list[DetectorScore]`** and `retrain(batch, evasions)` — one seam
  covers every model. Swapping LightGBM for the GNN changes nothing about the test.
- **`Evaluator.leave_one_attack_out(detector) -> MetricResult`** — the evaluation seam. Leakage
  guards are asserted here, where they are cheap to check and expensive to lose.
- **`run_closed_loop(...)`** — the integration seam. `tests/test_loop_smoke.py` already runs the
  whole loop on stubs; it stays the day-one gate and must stay green through every ticket.

No new seams are proposed. If a ticket seems to need one, that is a signal the work belongs
behind an existing seam.

### Taxonomy

The vector IDs in `afl/attack/templates/vectors.yaml` currently disagree with the architecture
doc, and the doc is the source of truth. The nine vectors become:

| ID | Vector | Engine | Tier |
|----|--------|--------|------|
| S1 | Mule network & layering | graph | strong — full adaptive loop |
| S2 | Card testing / BIN enumeration | velocity | strong — full adaptive loop |
| S3 | Account takeover via drift | drift | strong — full adaptive loop |
| C1 | Bust-out | drift | common — template + fixed benchmark |
| C2 | UPI collect-request / APP scam | velocity + payee | common — template + fixed benchmark |
| C3 | Instant-A2A pass-through | graph + velocity | common — template + fixed benchmark |
| M1 | Boundary probing / paced evasion | *is* the optimiser | mid — free by construction |
| M2 | Synthetic-identity lifecycle | drift → bust-out | mid — template |
| M3 | First-party / friendly fraud | anomaly / drift | mid — **the holdout** |

Three levels, never flattened: mechanisms (S1, S3, C1, C3), enablers (C2, M2), attacks against
our own AI (M1). Nine vectors, three engines — adding a vector is a YAML edit.

**M3 (first-party fraud) is the leave-one-attack-out holdout.** It is the family where
`user == fraudster`, which breaks the legit-vs-attacker assumption every supervised feature
rests on. That makes it the honest stress test, and it is why the anomaly layer exists.

### Attack generation

Two tiers. Tier 1 is deterministic/probabilistic templates for all nine vectors — cheap,
reproducible, forming the baseline training set and the fixed benchmarks. Tier 2 wraps the
adaptive optimiser around **S1–S3 only**; that is the novel part and it is deliberately narrow.

Search, not reinforcement learning: Optuna TPE where available, with a random/hill-climb
fallback so the loop never hard-depends on the backend. Easier to build, debug and explain than
RL, and better suited to a short build.

Fitness is `evasion_rate − λ · realism_penalty`, with the evasion rate computed over **fraud
rows only** — diluting by legit volume would make the convergence curve a function of batch
composition rather than attack success.

Realism is enforced at two costs. Cheap, every round: schema validity, cross-row rules
(no self-transfers, no duplicate IDs, no unlabelled fraud, no provenance on legit rows),
empirical distribution bounds, and clamping to each vector's declared `search_space`. Expensive,
per experiment: the `afl/fidelity` scorecard.

An LLM, if used at all, proposes scenarios inside the schema. It never emits transactions.

### Detection

Layers earn their seat, in this order:

1. **LightGBM** over causal velocity windows, RFM features, time-since-last-event and
   graph-derived features. The workhorse and the hard baseline. Falls back to sklearn's
   `HistGradientBoosting` when LightGBM is unavailable, and the fallback is logged, never silent.
2. **Anomaly layer** (isolation forest / autoencoder), fit on legit rows only. The only thing
   with a chance on an unlabelled held-out family. Promoted to headline if supervised recall
   collapses there.
3. **Sequence model** (GRU or small transformer), for the drift arc only — S3 and C1. Enters the
   reported table only if it beats LightGBM on the same out-of-time split. If it does not, that
   is a reported result.
4. **Temporal GNN** over the account-device-beneficiary graph. Same rule, plus a stated
   fallback: hand-rolled graph features + LightGBM is what ships if the GNN does not replicate.
5. **Decision & explainability** — graded action from a cost model, SHAP reason codes from day
   one, with a labelled fallback to global importance when SHAP is unavailable.

Feature construction has two non-negotiable rules: every feature for a row is computed from rows
strictly before it, and `is_fraud` / `vector_id` / `attack_run_id` / `txn_id` never enter X.

### Evaluation

- Report PR-AUC, recall @ fixed FPR, precision@k. Never accuracy or ROC-AUC alone.
- Split out-of-time with an embargo gap, never random. Committed, not regenerated.
- Headline: leave-one-attack-out. The holdout keeps all legit rows — without a haystack, FPR and
  precision@k mean nothing.
- Three systems (real-only / SMOTE / adaptive), one holdout, one operating point.
- Every reported number comes out of `scripts/run_experiment.py`. Nothing in that file decides
  anything: it assembles components from Hydra config, runs them, and writes artefacts. A number
  that cannot be reproduced from (config, seed) is a bug.

### Anchoring on real data

The synthetic placeholder config exists so a fresh clone runs with nothing to download; its
numbers are a pipeline check and are labelled as such. Real numbers require a real anchor —
PaySim and/or IEEE-CIS through `afl/data/loaders.py`. Real rows carry `vector_id=None`;
a real row that ever gains provenance has leaked a label path.

### Ownership

Two owners, one seam. **Person A** takes the red side and product: attack schema, actor params,
the three engines, all nine template vectors, the adaptive optimiser, and the wiring —
FastAPI/Streamlit, Docker, experiment tracking, the closed-loop handshake, the convergence demo,
and the reproducibility gate. **Person B** takes the blue side and evaluation: feature
engineering, the LightGBM baseline, graph features, the anomaly baseline, the sequence and GNN
experiments, and — most important — the leave-one-attack-out folds, the out-of-time split and
the three-system comparison.

The loop only closes where they meet, so ticket 01 is done together on day one and the hollow
loop stays green from then on. A hollow loop that runs beats two polished halves that never
connect.

## Testing Decisions

**What makes a good test here.** Test the observable behaviour at a seam, not the way the code
reaches it. `test_engines.py` should assert that a graph vector emits a fan-in with the declared
width and correct labelling — not that it called `rng.dirichlet`. A test that pins an
implementation detail will be deleted the first time the implementation improves, which means it
was never protecting anything.

Three properties are worth testing harder than anything else, because losing them is silent and
expensive:

1. **Causality.** A feature computed for a row must not change when rows after it are appended.
   That is a property test, and it is the strongest guard against the leakage class this repo
   has already been bitten by once.
2. **Leakage.** After a leave-one-attack-out split, assert no row of the held-out family appears
   in training — including the replay buffer, which is the easy place to lose it.
3. **Provenance hygiene.** No forbidden column reaches X; no legit row carries a `vector_id`;
   no fraud row lacks one.

**Modules under test**

- `tests/test_contract.py` — schema round-trips, validators, evasion semantics. Any contract
  change lands here first.
- `tests/test_engines.py` — one test per vector, asserting on the emitted `AttackBatch` through
  `Simulator.generate`. Each new vector adds a case; no vector is done without one.
- `tests/test_eval.py` — split correctness, embargo behaviour, metric definitions at fixed
  operating points, and the leakage assertions above.
- `tests/test_loop_smoke.py` — the day-one gate. The whole loop, end to end, on stubs. It must
  stay green through every single ticket; a red smoke test blocks the frontier.

**Prior art.** All four files already exist and establish the idioms: fixed seeds, contract-typed
fixtures, assertions on returned objects rather than internals. Follow them rather than
inventing a new style — and prefer extending an existing file to adding a new one.

Deep-learning tickets (sequence, GNN) must skip cleanly when the `deep` extra is absent, so the
suite stays green on a default install.

## Out of Scope

Named deliberately, and stated as future extensions rather than omissions:

- Multimodal / deepfake detection.
- An LLM social-engineering content generator.
- Multiple tabular generators — one baseline generator is kept, no bake-off.
- A mandatory GNN. It is an experiment with a stated fallback, not a deliverable.
- Full reinforcement learning for the attacker. Search is the deliberate choice.
- Agent-swarm simulation with autonomous actors. Actors are parameter bundles.
- Production serving concerns: real-time latency SLOs, model registry, drift monitoring, A/B
  infrastructure.
- Any claim of DPDP compliance or formal privacy. DCR/MIA are evidence against memorisation,
  not proof of privacy.

ScamShield is a bolt-on, not a spine: reuse its FastAPI/Qdrant scaffold to save setup time and
its LLM-explanation pattern to narrate "why flagged" for C2. The hard rule is that the
experiment must fully run and reproduce with ScamShield removed.

## Further Notes

**The current numbers are the honest starting point.** On the synthetic placeholder config the
adaptive system lands *below* both baselines (PR-AUC 0.77 vs 0.97; recall@1%FPR 0.49 vs 0.95).
Nobody massaged that. It is the weak-side reading the design predicts when the loop searches a
single vector against a detector that already generalises to the holdout. Fixing it is build
work — the optimiser must search across the strong vectors, on a real dataset. The first pass
also caught and fixed a set of leakage bugs (velocity windows peeking forward, retraining not
accumulating, the loop training on the holdout window); most of the earlier apparent signal was
leakage. These are the real numbers to improve from.

**The hypothesis is allowed to fail.** If the adaptive loop does not beat SMOTE on the held-out
column, that result gets reported. A measurable adaptive loop *existing*, with a curve that
regenerates from logs, is a defensible outcome; a spectacular number nobody can reproduce is not.

**Honesty guardrails** carried through to the write-up:

- C2ST near chance is a diagnostic under our protocol, not proof of realism.
- DCR / MIA are evidence against memorisation, not proof of privacy.
- Synthetic-only reduces exposure; it does not automatically mean DPDP compliance.
- Latency budget depends on the decision point — no fixed sub-50ms assertion.
- AI-risk classification depends on deployment context — no blanket carve-out claim.
- Projected losses are labelled projections, never realised figures.
- Frontier vectors are demonstrated capabilities, not necessarily mass-exploited patterns.
