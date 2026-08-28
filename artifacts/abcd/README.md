# artifacts/abcd — which file is the result

**Canonical:** `amlworld_gather-scatter.json`. Regenerated on commit `4050fc46` with a clean tree,
and it says so: `git_commit` and `git_dirty: false` sit beside `split_digest` in its header. This
is the only file any document may quote.

**Retired:** `retired/v1.0-irreproducible_amlworld_gather-scatter.json`. The original 7-seed run,
generated 2026-08-24 and committed 2026-08-26 in `d880459`. **It cannot be reproduced from any
commit in this repository** — it was written from a working tree that no commit captures. Kept as
the evidence for the defect, never as a number.

How it was caught: a re-run on the same seeds and the same fold disagreed with it on exactly the
two systems that depend on the simulator (`C_template`, `D_adaptive`), while the two that do not
(`A_real`, `B_smote`) reproduced to ten decimal places. Runs at `d880459` and at HEAD agree with
each other and disagree with the artefact, so no code change explains it. The fix is the provenance
stamp now in every artefact — had it existed, this file would have carried `git_dirty: true` and
the defect would have been visible in the file rather than surfacing days later.

**Diagnostics:** `diagnostics/`. Working files from the ticket-14 / Phase-4 investigation, kept for
audit and **superseded by the canonical artefact**:

- `amlworld_gather-scatter_binding.json` — 7 seeds under the binding leash (measured bounds,
  separability vetoed under both audit rules). Produced on a dirty tree at `52da03c`. Its numbers
  match the canonical run exactly, which is what established that the leash is inert.
- `_control_default_seed7.json` — seed 7 with the default leash on the same code, bit-identical to
  the binding run. This is the comparison that proved the leash changed nothing.
