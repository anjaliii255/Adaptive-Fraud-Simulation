---
status: accepted
---

# The commodity families, and the knobs they cost

Tickets 04–06 turned C1, C2, C3 and M2 from declared-but-unbuildable into generating vectors.
Eight of the nine are now built; M1 is left, and it arrives free as the optimiser's boundary walk.

## These are allowed to be caught

M3 had to survive naive rules, because it is the held-out stress test. C1/C2/C3/M2 are the
opposite posture: they are must-catch commodity load, they feed training, and they act as fixed
benchmarks. A detector that misses a bust-out is not credible, so C1 spikes 7-10x on purpose and
C2 pays a first-time payee in plain sight. Making them subtle would defeat what they are for.
None of them is routed through the holdout or the eval path.

## What each family needed, and why the engine had to grow

`SPEC.md` says needing an engine edit means the engine was under-parameterised. Four vectors
needed five knobs. Every one is generic, defaults to the previous behaviour, and belongs to an
engine rather than to a vector:

| knob | engine | why it did not exist |
| --- | --- | --- |
| `n_payees` | velocity | the engine sprayed across a pool; an APP scam pays one payee, repeatedly |
| `amount_shift` | velocity | a scam is atypical *for the payer*; the engine only drew from the actor's own distribution |
| `device` | velocity | the victim uses their own phone. A minted device id would have been a false takeover tell |
| `fresh_beneficiary` | graph | the relay exit came from the cash-out pool, which already has inbound history |
| `new_account` | simulator | every entity had a real opening date and real traffic, so "no history" could not be said |

The endpoint choice for velocity is now keyed on the vector's **actor** rather than hardcoded: a
victim actor pays out of their own account to a hostile payee, an attacker actor pays out of one
they control. One rule, no per-vector branch, and S2 and M1 are untouched by it.

Payee novelty is structural rather than lucky: background traffic only ever pays merchants and
ordinary accounts, so a beneficiary drawn from the mule pool provably has no prior inbound.

Every already-built vector was regenerated against the pre-change engines and is byte-identical,
except for the provenance change below. Knobs short-circuit before drawing, so an unused one
cannot shift the random stream.

## Synthesised legit history keeps its run id

Legit rows carry `vector_id=None` as before, but a row an attack episode synthesised now also
carries `attack_run_id`. Background traffic still carries none. So the two questions stay
separate: `vector_id` says *this is fraud of this family*, `attack_run_id` says *this row came out
of this run*. Without it M2's seasoning is indistinguishable from ordinary background, and the
trail from a result back to the params that produced it breaks.

This changed S3, C1 and M3 output in exactly one field and nothing else, which was verified
field-by-field rather than asserted.

## M2 is labelled at the bust-out, not through the seasoning

Only M2's bust-out rows are fraud. The seasoning is legit background.

The alternative — labelling the seasoning fraud, so the task becomes "catch the account while it
is being groomed" — was rejected because it teaches the model that ordinary small payments are
fraud whenever they later precede abuse, which is hindsight, not a signal. The synthetic-identity
tell lives in the *thinness and speed* of the file: an account with no history at all, seasoned in
days, then drained. Features can see that; a label on the seasoning would only hide it.

## Named limits

**C1 is one account.** A ring of accounts busting out in concert is a real variant, and it is
S1's territory. Keeping it out of C1 keeps the two families distinguishable.

**C3 has no cross-institution marker.** The contract has no institution field and adding one is a
joint decision, so the relay is modelled without it. Everything else in the signature — one hop,
in ≈ out, near-zero dwell, a payee with no prior inbound — is present.
