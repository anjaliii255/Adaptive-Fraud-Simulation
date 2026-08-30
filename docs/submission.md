# Submission — the Adaptive Fraud Simulation Lab

_Mastercard Innovation Challenge, GFF 2026. This is the document for a reviewer with an hour: the
question we asked, the answer we got, and the commands that reproduce both. Every number in the
two result sections below is recomputed from a committed artefact by `make claims`, and every
sentence in this repository is checked against the honesty guardrails by `make guardrails`. Both
run in under a second and both are wired into `make reproduce`._

| | |
|---|---|
| **Live prototype** | https://epoch-fraud-lab.streamlit.app/ |
| **Code** | https://github.com/anjaliii255/epoch |

The prototype is a five-act walkthrough. Attack generation and the commensurability audit run live
in-process; results replay from committed artefacts and are badged, so replayed data is never
shown as live.

## The story

Not: we built an AI fraud platform.

**We tested whether a closed attacker–defender loop makes a detector more robust to attacks it has
never seen — and here is the reproducible curve.**

The loop is real and it closes. The robustness did not follow. Adaptive augmentation lost to
static template augmentation on the one anchor where the question could be asked fairly, and the
instruments we built to catch ourselves overstating are the reason we can say that with a number
instead of a shrug.

Generating adaptive attacks turned out not to be the hard part. Knowing whether the numbers they
produce mean anything is, and most of this repository is the instrument that answers that — up to
and including the cases where the answer disqualifies our own result.

## Four questions, four answers

Each question gates the next. Question 4 is only meaningful if question 3 passes, and question 3
is where most of the evidence stops.

| | question | answer |
|---|---|---|
| 1 | Can the attacker find evasions? | **Demonstrated** — 83.6% of generated fraud past the detector in round 0, before the detector adapts |
| 2 | Can the detector close known gaps? | **Partially** — evasion falls 0.836 → 0.054, on all 7 seeds, but in aggregate rather than per vector, and every extra layer built to close a gap was benched |
| 3 | Do the attacks pass the fidelity and provenance gates? | **On one anchor of four** — 4 of 18 leave-one-attack-out folds carry a quotable number, and 2 of 6 held-out columns in the other experiment |
| 4 | When they do, does adaptive beat non-adaptive? | **No — it loses.** D beats C on 1/7 seeds (p = 0.992); C beats D on 6/7 (p = 0.062, directional). And unconstrained: the leash vetoed 0/42 |

Two of those four answers are worse than the project hoped for, and they are the ones stated
first, in the same voice as the other two. That is the point of the document.

## The result, in one table

Four systems and a no-model floor on AMLworld, holding out the GATHER-SCATTER laundering typology
entirely — a real attack shape nothing trained on, not a synthetic family injected for the
purpose. 7 seeds, 6 rounds each, 173 positives in the fold, base rate 0.053%, split digest
`f5e33a878d68b792`. From `artifacts/abcd/amlworld_gather-scatter.json`, regenerated on commit
`4050fc46` with `git_dirty: false`.

```
system         trained on                          PR-AUC (mean ± sd)   recall@1%FPR
A_real         the anchor's real rows and labels    0.0806 ± 0.0765      0.440 ± 0.185
B_smote        the same plus row-level oversampling 0.0274 ± 0.0154      0.520 ± 0.148
C_template     the same plus STATIC template attacks 0.0557 ± 0.0560     0.544 ± 0.255
D_adaptive     the same plus the ADAPTIVE loop      0.0168 ± 0.0131      0.378 ± 0.292
amount_floor   nothing — rank by amount             0.0013               0.012
```

`C_template` is what makes `D_adaptive` falsifiable: C and D share an episode budget, so the only
difference between them is whether the attacks were searched adaptively or generated statically.
If D does not beat C, the adaptive search bought nothing that plain synthetic augmentation would
not have bought.

Per-seed direction, exact one-sided sign test:

| comparison | PR-AUC | recall@1%FPR |
|---|---|---|
| D > C_template | 1/7, p = 0.992 | 2/7, p = 0.938 |
| D > B_smote | 4/7, p = 0.500 | 2/7, p = 0.938 |
| D > A_real | 1/7, p = 0.992 | 2/7, p = 0.938 |
| D > amount_floor | 7/7, p = 0.008 | 6/7, p = 0.062 |
| **C > D**, the same test read the other way | **6/7, p = 0.062** | 5/7, p = 0.227 |

**Adaptive does not beat non-adaptive; it loses to it.** D wins on 1 of 7 seeds by PR-AUC and 2 of
7 by recall. Read the other way, static template augmentation beats the adaptive loop on 6 of 7
seeds.

**That is directional, not significant, and the write-up says so wherever it is quoted.** 6/7 does
not clear 0.05, and every standard deviation in the table is comparable to or larger than its own
mean, so this fold is underpowered for an effect of this size — the same limitation that applied
when an earlier reading of the same experiment came back as a null. The one comparison that clears
significance is that every system beats the no-model amount floor on PR-AUC, 7/7, p = 0.008, which
establishes that the fold is not measuring amount-legibility and says nothing about adaptive.

**What did work, and it is the one place the architecture does what the diagram says.** Evasion
falls 0.836 → 0.054 over six rounds, on every one of the 7 seeds, and the audit gate rejected 0 of
42 rounds under the stricter of the two separability rules — the same rule that rejects the kept
batch in 35 of 36 rounds on AMLSim and 36 of 36 on PaySim. The loop closes, and the closing is not an
artefact of a gate that was not looking.

**And it is unconstrained.** The per-round realism leash that was supposed to stop the optimiser
buying evasion with unrealistic traffic vetoed 0 of 42 rounds. Correcting its bounds — measured off
the anchor rather than guessed — made the penalty vary properly and changed the outcome not at all;
a control run with the old leash and a run with the corrected one are bit-identical on the same
code. So D is not a measurement of "the best attack subject to staying realistic". It is a
measurement of an unconstrained search, and nothing here is evidence about what a realism-bounded
adaptive attacker could do.

## The taxonomy, at three levels

26 vectors identified, 8 fully simulated. The nine below are the ones the engines produce; the other seventeen are specified with the surface each would need, in `docs/coverage.md`. The level is the organising fact and it is never flattened
into one list, because the three levels are answerable by different defences and a flat list hides
that. Definitions in `afl/attack/templates/vectors.yaml`, reasoning in
`docs/adr/0001-vector-taxonomy-and-holdout.md`, status in `docs/threat-model.md`.

**Level 1 — fraud mechanisms.** The fraud itself: what moves the money.

| id | vector | engine | tier | status |
|---|---|---|---|---|
| S1 | Mule network & layering | graph | strong | built |
| S2 | Card testing / BIN enumeration | velocity | strong | built |
| S3 | Account takeover via drift | drift | strong | built |
| C1 | Bust-out | drift | common | built |
| C3 | Instant-A2A pass-through | graph | common | built |
| M3 | First-party / friendly fraud | drift | mid | built · **the holdout** |

**Level 2 — fraud enablers.** Not fraud on their own: what makes the mechanism possible.

| id | vector | engine | tier | status |
|---|---|---|---|---|
| C2 | UPI collect-request / APP scam | velocity | common | built |
| M2 | Synthetic-identity lifecycle | drift | mid | built |

**Level 3 — attacks against the model itself.** Not fraud against a bank: fraud against the
detector.

| id | vector | engine | tier | status |
|---|---|---|---|---|
| M1 | Boundary probing / paced evasion | velocity | mid | template |

M1 is the only vector not fully simulated: it emits valid traffic of roughly the right shape while
the defining tell is missing, so it is training load and haystack and is never quoted as a recall
figure for that family. It arrives for real as the optimiser's own boundary walk, which is the
level-3 attack running rather than being described.

**M3 is the leave-one-attack-out holdout** because `user == fraudster` breaks the legit-versus-
attacker assumption every supervised feature rests on: the abuse runs on the owner's own device, to
beneficiaries the account already pays, elevated only against that account's own history.

## The instruments, and what they refused

The instruments are most of the work, and they are what turns a negative result into a deliverable.
Five of them, each of which withheld one of our own numbers:

| instrument | what it asks | what it did to us |
|---|---|---|
| commensurability audit | is this batch separable from the anchor by one contract field? | passed AMLworld on every round; rejects the kept batch in 71 of 72 rounds on AMLSim and PaySim under the strict rule |
| provenance probe | can a model score this fold knowing only *who wrote the row*? | withheld System C's held-out column on both anchors of the three-system table, and A's and B's as well on PaySim |
| transfer test | does a detector trained on our synthetic attacks catch *real* fraud? | it does not, on all three anchors it was run on — below the amount floor on every one |
| fidelity scorecard | does training on the generated data teach anything about real fraud? | level-3 utility failed on both anchors it was run on |
| leave-one-attack-out guards | did one row of the held-out family reach training, replay buffer included? | held on all 18 folds; the 14 that were withheld went for separability, too few positives or a template vector — never for a leak |

The scorecard was itself validated before it was trusted, against copy, shuffle and noise
generators whose answers were known in advance (`make fidelity-selftest`).

**Why the negative result is the deliverable.** A pipeline that produces a confident number is
easy. A pipeline that produces a number *and can tell you when not to believe it* is the harder
artefact, and it is the one a fraud team actually needs — because the failure mode this repository
documents, a model learning which generator wrote a row rather than what fraud looks like, is the
failure mode of every synthetic-augmentation programme and is invisible without exactly these
instruments.

## What this audit found in our own documents

`make claims` asks whether a number is still the artefact's. Reading the documents line by line
asks the harder version: whether the *sentence around it* is still the artefact's — because a
figure can be exactly right while the clause it sits in describes a run that no longer exists.
Four sentences did not survive that reading, all four now corrected in place:

1. **"The rate plateaus near 0.20 rather than reaching zero"** (`docs/claim.md`, question 2) — a
   sentence left behind when the evasion curve was regenerated. The curve plateaus near 0.05, not
   0.20; the mean is 0.106 by round 2 and 0.054 by round 5, and one seed of seven ends at 0.204.
   The qualification the sentence was making is still true and is now made with the right numbers.
2. **"The `lift` rule rejects 100% of candidate batches"** (`docs/claim.md` twice,
   `docs/realism-leash.md` once, and a comment in `afl/attack/multi.py`) — the committed artefacts
   record the rule rejecting the batch the loop kept in **71 of 72 rounds**, not all of them. The
   argument is unaffected and the number is now the one the runs support.
3. **The provenance-probe row of the instrument table** (`docs/claim.md`, question 3) attributed a
   0.998 to "injected rows sort from real traffic" on AMLSim. Two different probes were being
   quoted as one: on AMLSim the injected-versus-real probe scores 0.355 and the fold **passes** it;
   what fails there is the System C counterfactual, a model told only which rows the generator
   wrote, at 0.995 against System C's 0.998. On PaySim the injected-versus-real probe is the one
   that fails, at 0.970. The row now names which probe failed on which anchor.
4. **"The loop converges on all four anchors"** (`README.md`) — the loop was run to convergence on
   three. BankSim was a time-boxed three-gate spike that returned NO-GO at gate 2 and gate 3
   (`docs/adr/0004-banksim-spike-and-the-null-result.md`) and never had the loop pointed at it. The
   sentence now says three, and says what happened to the fourth.

One finding was recorded rather than fixed, because fixing it means regenerating a headline
artefact to change a sentence inside it: the `reason` strings that
`artifacts/three_system/*.json` stores for System C's withheld column describe the counterfactual
probe using the ordinary probe's wording — "a classifier sorts the injected M3 rows from this
anchor's own traffic" — where the `provenance.question` field in the same block correctly says "the
generator wrote this row". `docs/three_system.md` is generated from those strings and inherits the
imprecision. The number is right, the label on it is not, and it belongs to the ticket that owns
that table rather than to this one.

## The seven guardrails, applied to the wording

These are not a list at the end of a deck. Each one is a rule in `docs/guardrails.yaml` that
`make guardrails` runs over every document in the repository, one sentence at a time. Each rule has
been tested against planted overstatements whose answers were known in advance, and against the
refusals of the same claims, so the audit is checked rather than trusted (`tests/test_guardrails.py`).

| guardrail | where it bites here | what we write instead |
|---|---|---|
| C2ST near chance is a diagnostic under our protocol, not proof of realism. | we never ran a test under that name; the instrument asking the same question is the provenance probe | the probe did not land near chance — it sorted injected rows from real traffic — and the result it touches is withheld rather than quoted |
| DCR / MIA are evidence against memorisation, not proof of privacy. | the fidelity scorecard's privacy panel | the panel bounds memorisation and membership leakage, and on AMLSim it reports membership as inferable from the synthetic data alone, which is a failure reason rather than a reassurance |
| Synthetic-only reduces exposure; it does not automatically mean DPDP compliance. | every anchor here is a public generated dataset | no personal data was processed by this work at all, which is a property of these anchors and not a compliance position on any deployment |
| Latency depends on the decision point — no fixed sub-50ms claim. | the prototype and the FastAPI service | the only timings measured are the prototype's warm-up on one laptop, about a second to build the attack world and about 1.3 seconds to fit the detector; nothing was measured under load at a decision point |
| AI-risk classification depends on deployment context — no blanket carve-out. | what this system would be if deployed | it is a research harness with no deployment, and where a detector sits decides its risk class, so we make no determination |
| Projected losses are labelled projections, never realised figures. | the business case | this write-up quotes no monetary figure as a result — the only currency anywhere in it is inside the block of sentences below that we refuse to write |
| Frontier vectors are demonstrated capabilities, not necessarily mass-exploited patterns. | M1, M2 and M3 | the simulator demonstrates the vector can be generated; how common it is in the field is a separate question and nothing here measures the field |

The audit is two-sided on purpose. A banned-phrase list only catches the sentence someone wrote; it
cannot catch the qualifier someone left out. So naming an instrument obliges the same sentence to
say what it is worth — write `C2ST` without `diagnostic` or `not proof` nearby and the check goes
red, the same way it does for the sentences below.

These are the sentences the audit exists to stop, quoted as specimens so they can be tested rather
than trusted. Not one of them is supported by anything in this repository:

```
Our C2ST came back near chance, which proves the generated attacks are realistic.
DCR and MIA guarantee privacy for every generated row.
Synthetic-only training is DPDP compliant.
We score every transaction in sub-50ms.
A research harness like this is exempt from the AI Act.
The loop prevented Rs 4 crore of fraud last quarter, an ROI of 340%.
M3 first-party fraud is widespread across Indian payments today.
```

## Cut on purpose

Named as decisions taken before the work, not as gaps found after it. Each carries the reason and
what would change it. The full list, with two more that are about deployment rather than research,
is in `SPEC.md` under Out of Scope.

| cut | why it was cut | what would change the answer |
|---|---|---|
| **Multimodal / deepfake detection** | a different sensing problem with different data, sharing no seam with the transaction contract; adding it would have bought breadth at the cost of the closed loop being genuinely closed | a modality that reaches the detector as a contract field rather than as a second pipeline |
| **LLM social-engineering content generator** | the loop's objective is measured on transactions; generated scam *text* is not scored by anything downstream here, so it would be a demo rather than a variable | a decision layer that consumes message content, at which point the text becomes a feature and can be optimised against |
| **Multiple tabular generators, in a bake-off** | one baseline generator kept deliberately: comparing generators answers "which synthesiser wins", and the question here is whether *adaptive* search beats *static* synthesis, which needs a control rather than a field | the adaptive-versus-static comparison coming out positive, at which point which generator is worth asking |
| **A mandatory GNN** | declared an experiment with a stated fallback rather than a deliverable. It was built, measured against the hand-rolled graph features, and lost — so the fallback is what ships, and a test refuses to let the config switch it on while the committed evidence says no | an anchor whose graph carries structure the hand-rolled features miss; `docs/gnn.md` records what was measured |
| **Full reinforcement learning for the attacker** | search is the deliberate choice: the attacker's parameter space is small and bounded, an RL policy would add an outer training loop and a second reproducibility problem on top of one we already had to instrument heavily | a parameter space too large to search, or a sequential attack whose reward genuinely arrives at the end |
| **Agent-swarm simulation with autonomous actors** | actors here are parameter bundles, which is enough to express every vector in the taxonomy. Autonomous agents would add emergent behaviour nobody can hold fixed across seeds, and everything in this repository is built to be re-run bit-identically | a research question about coordination itself rather than about detection |

Two of these are stronger than cuts: **the GNN and the sequence model were built, measured and
benched**, with the evidence published in `docs/gnn.md`, `docs/sequence.md` and
`docs/negative-results.md` rather than deleted with the branch. A cut that was tested is worth more
than a cut that was assumed, and this write-up distinguishes the two.

## Reproduce it, in this order

Needs Python 3.11, and on macOS `brew install libomp` before anything else — the LightGBM wheel
imports cleanly without it and then fails to load its own shared library at fit time, so the code
silently falls back to sklearn. Every artefact records which backend produced it.

```bash
pip install -e '.[dev]'        # or: uv sync --extra dev

make reproduce   # the one command: ~84s, nothing to download
make claims      # every quoted number, recomputed from the artefact it names
make guardrails  # every sentence, against the seven guardrails above
```

To see it rather than read it, `streamlit run prototype/app.py` walks five acts over the committed
artefacts, with the generate-and-audit step and the detector running live in-process. Every screen
renders from a committed artefact even with the live path dead, and replayed data is badged rather
than shown as live (`prototype/README.md`).

`make reproduce` runs the other two and then the expensive half: the whole loop end to end on the
zero-download synthetic default, compared against a committed expectation, then run again and
compared with itself. It exits 0 if everything checked reproduced, 1 if a number that should have
matched did not, and 2 — UNCONFIRMED — if the numbers differ *and* so does the environment, which
is not automatically a defect and is not a pass either.

**Without the anchor**, which is every fresh clone, the headline is re-derived rather than re-run:
the A/B/C/D result is recomputed from `artifacts/abcd/` and the documents are checked to still
quote it. **With the anchor** in `data/raw/` (see `docs/data.md`, none of it is committed and none
of it is fetched for you):

```bash
make splits
python scripts/abcd_experiment.py --data amlworld \
    --typology GATHER-SCATTER --seeds 7 11 23 42 101 1337 2024
python scripts/reproduce.py --anchor amlworld --anchor-seed 7   # re-run one seed and diff it
```

The last line is the strong form: it re-runs a seed of the committed experiment and reports every
system whose number moved. Without the anchor on disk it names the missing file and skips, rather
than passing quietly.

**Checked from a clean tree, not only from the working copy.** `make reproduce` was run again in a
copy holding only the repository's tracked files — no `data/`, no `.git`, no untracked artefacts —
and all five stages passed in 82 seconds. That is the reviewer's path, walked.

## Independent reproduction

The claim `make reproduce` supports is "this machine reproduces its own numbers", and that is a
weaker claim than "this repository reproduces". The gap is one machine, and it is recorded here
rather than assumed away.

| owner | machine | date | command | verdict |
|---|---|---|---|---|
| A | macOS 26.6 · arm64 · 14 cores · Python 3.11.16 · LightGBM 4.5.0 | 2026-08-29 | `make reproduce` | **PASS**, all stages |
| B | — | — | `make reproduce` | **outstanding** |

Owner B's row is empty and is meant to stay visible until it is not. To fill it: run
`make reproduce`, and paste the verdict line plus the environment line the run prints. If it comes
back UNCONFIRMED, `artifacts/reproduce/report.json` names every environment field that differs and
every number that moved — that is the interesting outcome, not a failure, and
`docs/reproducibility.md` says what is known to be stable across machines and what is not.

What is already known about the cross-machine question: LightGBM's histogram construction is a
function of its build and its thread count, nothing here pins thread count, and on macOS a missing
libomp silently changes the model rather than the last digits. Direct dependencies are pinned
exactly in `requirements.txt` and transitively in `uv.lock`.

## What is machine-checked in this document, and what is cited

Two sections of this document — **Four questions, four answers** and **The result, in one table** —
are covered regions in `docs/claims.yaml`. That means both directions are enforced: every number in
them is recomputed from the artefact it names and has to format to exactly the string written here,
*and* any number appearing in them that is not a registered claim or an explicitly excused constant
fails the check. Adding a number to either section means adding a row to the registry. That cost is
deliberate.

Every other number in this document is a count quoted from a named document — `docs/threat-model.md`
for the taxonomy, `docs/loao.md` for the fold counts, `docs/three_system.md` for the round tables,
`docs/adr/0004-...` for the BankSim gates — each of which is itself generated from artefacts or
carries its own citation. Those are checked by reading, which is the same standard as the rest of
the prose, and is why `make guardrails` exists to check the prose too.

`docs/claim.md` is the long form of what is and is not claimed. `docs/results.md` sets the two
comparison experiments side by side and warns that their system labels collide — `C` is
template-static in one and adaptive in the other. `docs/negative-results.md` is the three layers
that were built and did not earn their place. Nothing in this document supersedes those; it is the
front door to them.
