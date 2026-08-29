# Reproducibility — what reproduces, what does not, and how to check in two minutes

_Ticket 20. Every claim in this document was produced by running the command it describes, on the
machine named in the environment block below. Where something could not be checked here, it says
so rather than being written as though it had been._

Reproducibility is the cheapest credibility available, and it is the thing that separates this
from a demo. It is also the thing this repository has already got wrong twice, in both directions:

- an A/B/C/D artefact that carried its split digest, its seeds and its operating point and still
  **could not be regenerated**, because it was written from a working tree no commit captures
  (`artifacts/abcd/README.md`);
- an evasion curve that three documents kept quoting as `0.915 → 0.201` after the artefact behind
  it had been regenerated to `0.836 → 0.054`.

Neither was caught by a person reading carefully. Both are now caught by a command.

## The one command

```bash
make reproduce            # or: python scripts/reproduce.py
```

Needs nothing downloaded. On the machine this was written on — macOS 26.6, arm64, 14 cores,
Python 3.11.16, LightGBM 4.5.0 — it takes **84 seconds** and prints:

```
[PASS] environment                          python 3.11.16 · arm64 · lightgbm 4.5.0 · 14 cpus
[PASS] claims (documents)             0.1s  67/67 claims verified
[PASS] guardrails (wording)           0.1s  7 guardrails over 33 documents
[PASS] synthetic headline            41.3s  every headline number matches the committed expectation
[PASS] determinism (same seed twice)  42.6s  two runs of seed 1337 agree on every number
```

Five stages, four of which can fail:

| stage | what it proves | cost |
| --- | --- | --- |
| environment | the context every other line is read in | instant |
| claims | every number quoted in the documents still falls out of the artefact it names | ~0.1 s |
| guardrails | no sentence around those numbers claims more than the run supports | ~0.1 s |
| synthetic headline | the whole loop runs end to end from a clean clone and lands on the committed numbers | ~40 s |
| determinism | the same seed twice gives the same number | ~40 s |
| anchor (opt-in) | one seed of a **real** experiment re-run and diffed against the committed artefact | 7 min (PaySim) |

Exit codes: `0` everything checked reproduced, `1` something that should have matched did not,
`2` UNCONFIRMED — the numbers differ **and** so does the environment, which is not automatically a
defect and is not a pass either.

**Checked in a clean tree, not only in the working copy.** The command was run again in a copy
containing only the repository's tracked files — no `data/`, no `.git`, no untracked artefacts —
and all five stages passed in 82 seconds, the guardrail audit included. What that copy did *not*
re-do is the install: the dependencies were the ones already on this machine, so
`pip install -r requirements.txt` on a machine that has never installed them remains part of the
outstanding cross-machine check below.

## What a fresh clone can check, and what it cannot

**The anchors are not in the repository.** All four are public but licensed for manual download
(`docs/data.md`), so `data/**` is gitignored and a clean clone has no real data. That splits the
headline into two questions, answered differently:

- **Does the A/B/C/D result still follow from its artefact?** Checked on every clone, in 0.1 s, by
  recomputing all 67 registered numbers from the committed artefacts — the A/B/C/D headline from
  `artifacts/abcd/amlworld_gather-scatter.json` — and confirming the documents still quote them.
  This is re-derivation, not re-running.
- **Does the artefact still follow from the code?** Needs AMLworld in `data/raw/`. With it:
  `python scripts/reproduce.py --anchor amlworld --anchor-seed 7` re-runs that seed and diffs
  every system against the committed run. Without it, the stage says which file is missing and
  where to get it, and skips.

### The real anchor does reproduce

The strongest evidence in this ticket is not the synthetic stage. On the machine described above,
with PaySim in `data/raw/`:

```bash
python scripts/reproduce.py --anchor paysim --anchor-seed 1337
```

re-ran seed 1337 of the three-system table (ticket 16) end to end — 12 loop rounds, four systems,
a 637k-row pool — and found **zero differences** against `artifacts/three_system/paysim.json`,
across every cell of both columns, the withheld ones included. It took **412 seconds**, and the
committed artefact it matched was produced days earlier at a different commit.

So the loop, the optimiser's search, the SMOTE draw, the feature build and the detector's own
randomness are all reproducible from `(config, seed)` on real data, not just on the toy default.
`--anchor amlworld` runs the same check against the A/B/C/D headline; that anchor is not on this
machine, so that path has only been exercised through its skip branch.

**The synthetic stage is a pipeline check, not a result.** `data=synthetic` has no real anchor:
the numbers it reproduces say the pipeline runs and is deterministic, and nothing about fraud. The
expectation file says so in its own `what` field, the run stamps a banner into its artefacts, and
`make figures` prints the banner across any figure built from one.

## Determinism, and the residual variance

**Measured on this machine: none.** Two consecutive runs of `python scripts/run_experiment.py
experiment=adaptive` on seed 1337 produced **byte-identical** `metrics.json`, `history.json` and
`attack_trials.json`. The only difference anywhere in the artefact directory was one field —
`threshold_provenance.checked_at` in the fidelity scorecard, which is a clock by design.

That is why nothing this ticket added carries a timestamp. The provenance block stamped onto every
artefact (`afl/utils/runcard.py`) records the commit, the dirty flag, the seed and the library
versions, and deliberately **no clock**: a timestamp in an artefact destroys the cheapest
reproducibility check there is, which is to run it twice and diff the bytes. The wall-clock facts
live in a separate `run_card.json` beside the artefacts, which is excluded from that diff.

**Across machines: not measured, and not claimed.** No second machine was available when this was
written, and the Docker daemon was not running, so the "same number on a machine that never
developed it" check is **outstanding**. What is known:

- LightGBM's histogram construction is a function of its build and its thread count. Two machines
  with different core counts can legitimately differ in the last digits, and nothing in this
  repository pins thread count.
- On macOS the LightGBM wheel imports cleanly and then fails to load libomp if it is missing, and
  the code silently falls back to sklearn. That is not a rounding difference, it is a different
  model — which is why every artefact records `backend` and every run prints it.
- The direct dependencies are pinned exactly in `requirements.txt`, and transitively in `uv.lock`.
  Before this ticket `requirements.txt` carried ranges, which allowed LightGBM 4.3 and 4.5 to
  satisfy the same file.

`scripts/reproduce.py` handles this honestly rather than optimistically: if the numbers differ and
the environment differs, it reports **UNCONFIRMED** and exits 2, listing exactly which fields of
the environment moved. If the environment matches and the numbers do not, that is a defect and it
exits 1.

## Every sentence, as well as every number

`make claims` proves the numbers in a document are still the artefacts'. It cannot see the other
half: a figure can be exactly right and the sentence around it can still claim something no run
supports — a fidelity diagnostic sold as proof of realism, a projection written as a realised
figure, a latency budget asserted without naming the decision point it applies at.

`docs/guardrails.yaml` is the registry for that half, and `make guardrails` (also stage 3 of
`make reproduce`) runs it over every document in the repository, one sentence at a time. Each of
the seven guardrails carries three things: the statement itself, which has to appear in the
write-up; `forbid` patterns for the shape the overstatement takes, each with an `unless` that
exonerates the sentence *refusing* the claim rather than making it; and `require` patterns for the
qualifier that may not be left out — name `C2ST` without `diagnostic` or `not proof` nearby and
the check goes red. A guardrail with no rule behind it fails on its own, because a guardrail
nobody can fail is decoration.

Fenced code blocks are skipped, which is how `docs/submission.md` quotes the sentences it refuses
to write. `tests/test_guardrails.py` runs the rules against thirteen planted overstatements and
seven refusals of the same claims, so the audit is checked against answers known in advance rather
than trusted because it is green.

## Every number traces to an artefact

`docs/claims.yaml` is the registry: one row per quoted number, each naming the artefact, the
expression that recomputes it, the string the documents must contain, and the documents that must
contain it. `make claims` (also stage 2 of `make reproduce`) enforces both directions:

- **forward** — the expression is recomputed over the committed artefact and has to format to
  exactly the quoted string;
- **backward** — inside a covered *region*, every numeric literal must be a registered claim or an
  allowed constant with a stated reason, so the registry cannot fall behind the prose.

Covered regions today:

| document | section |
| --- | --- |
| `README.md` | the four-question table under the title |
| `README.md` | `## Main result` |
| `docs/results.md` | `## The A/B/C/D experiment` |
| `docs/claim.md` | `### 4. When they do pass, does adaptive beat non-adaptive?` |
| `docs/submission.md` | `## Four questions, four answers` |
| `docs/submission.md` | `## The result, in one table` |

An allowance may be scoped to the documents whose reason it holds in (`in: [...]` beside its
`why`), so a count quoted out of another document — nine vectors, eighteen folds — earns its place
in the write-up without also being waved through in the headline table it has nothing to do with.

**What is not covered, and why.** `docs/results.md`'s `## Current numbers` (the three-system
table) has its numbers registered as claims but is not a covered region — the generated
`docs/three_system.md` is the authority there and is rebuilt from the artefacts by
`build_three_system.py --doc-only`. The generated documents (`docs/features.md`, `docs/detector.md`,
`docs/decisions.md`, `docs/anomaly.md`, `docs/loao.md`, `docs/fidelity.md`, `docs/sequence.md`,
`docs/gnn.md`, `docs/three_system.md`) are pure functions of their artefacts and are not policed by
hand. Everything else — architecture, threat model, ADRs — is prose about design rather than
measurement.

Adding a number to a covered section means adding a row to `docs/claims.yaml`. That cost is the
point.

## What every run now writes

Every artefact carries a `provenance` block, and every run of the loop writes a `run_card.json`:

| file | what is in it |
| --- | --- |
| `metrics.json` | the numbers, plus `provenance`: commit, dirty flag, seed, python, platform, cpu count, library versions |
| `config.yaml` | the fully composed Hydra config, every override resolved |
| `attack_trials.json` | the optimiser's parameters and fitness, trial by trial |
| `history.json` | the per-round trace |
| `run_card.json` | config, seed, attack params and metrics in one file, plus the command line and the clock |
| `commensurability.json` | the audit that licenses reading the fold at all (anchored runs) |

The same stamp is written by `make splits`, `make features`, `make baseline`, `make decisions`,
`make anomaly`, `make loao`, `make fidelity`, `make table`, `make sequence`, `make gnn`, the
transfer test and the anchor gates. `tests/test_reproduce.py` fails if a new artefact type is
added that writes itself without one.

## Checked, and outstanding

Checked when this document was written, on the machine described above:

- [x] one command reproduces a stated headline number from a clean clone, with nothing
      downloaded — checked in a tracked-files-only copy with no `data/` and no `.git`
- [x] the same seed twice gives the same number — byte-identical, not merely close
- [x] every run writes config, seed, attack params and metrics into the artefact directory
- [x] every number in the README traces to a committed artefact, checked by a command rather than
      by reading
- [x] direct dependencies pinned exactly; `uv.lock` pins the rest
- [x] the whole thing still runs on the synthetic default with nothing to download

- [x] **a real anchor reproduces exactly** — PaySim seed 1337, 412 s, zero differences against the
      committed artefact

Outstanding, and stated rather than glossed:

- [ ] **verified on a machine that was not used to develop it.** Not done: no second machine, and
      the Docker daemon was not running. The command is built to make this cheap for whoever does
      it first — run `make reproduce`, and if it says UNCONFIRMED, `artifacts/reproduce/report.json`
      names every environment field that differs and every number that moved.
- [ ] **the real A/B/C/D re-run.** The AMLworld anchor is not on this machine (`data/raw/` holds
      PaySim and AMLSim only), so `--anchor amlworld` has been exercised only through its skip
      path — the equivalent check on PaySim passed, and the committed A/B/C/D artefact was itself
      regenerated on a clean tree at `4050fc46`, which is the evidence that stands until someone
      with the anchor re-runs it.
- [ ] **the expectation was recorded on a dirty tree**, mid-ticket. Re-record it with
      `python scripts/reproduce.py --record` on the commit that ships it, so it names a clean
      one. The command says so on every run until that happens.
