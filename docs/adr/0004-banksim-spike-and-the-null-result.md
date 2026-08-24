---
status: accepted
---

# Three anchors, three failures, and what the project is actually contributing

The transfer test showed neither PaySim nor AMLSim could validate the held-out claim. BankSim was
the cheapest remaining fix and the last one we agreed to try: a time-boxed spike with three gates
and a binary answer. This records the answer.

**NO-GO.** Gate 1 passes, Gates 2 and 3 fail. No further dataset hunting.

## The gates, in the order they fail cheapest-first

**Gate 1 — can it host behaviour? PASS.** Median **165** transactions per customer against a
threshold of 10, 100% of rows on a repeat customer, and 63.3% mean non-zero coverage across the 22
`src_*` history features against a threshold of 50%. This is the gate PaySim fails at ~1
transaction per sender and ~0% coverage. BankSim is emphatically not another PaySim.

**Gate 2 — is its own fraud non-trivial? FAIL.** On BankSim's own fraud, out of time at the
committed boundary: an amount-only rule with no model reaches PR-AUC **0.7023**, a trained
LightGBM reaches **0.9544**. Ratio **0.736** against a threshold of ≤ 0.6. For reference AMLSim
was 0.768. In absolute terms BankSim has more headroom than AMLSim (0.25 PR-AUC of model value
against 0.14), so this is the softer of the two failures — but it is still a failure by the
threshold agreed before the numbers existed, and it is not the one the decision rests on.

**Gate 3 — the transfer test. FAIL, decisively.** A detector trained without a single real fraud
label scores **0.2377** on real BankSim fraud, against **0.7023** for sorting on amount alone. It
does not clear the floor; it reaches a third of it.

```
                tested on REAL fraud          tested on synthetic M3
trained on      PR-AUC  recall@fpr  P@100    PR-AUC  recall@fpr  P@100
real             0.954     0.984     1.00     0.013     0.070     0.12
synthetic        0.238     0.296     0.68     0.934     0.970     1.00
both             0.955     0.980     1.00     0.687     0.827     1.00
amount_floor     0.702     0.753     1.00     0.006     0.070     0.02
```

The decision is robust to how Gate 2 is scored: relax it entirely and Gate 3 still fails by a
factor of three.

## The audit bug this spike found, and why the number above is the second one

The first Gate 3 run reported 0.2804, and it was measured through a hole in our own instrument.

`_busiest_senders` collected only `t.src`, so the anchored entity pool held nothing but customers
and the simulator handed the MERCHANT role to customer ids. Every real BankSim payment goes
customer → merchant; every synthetic one went customer → **customer**. Zero overlap in payee
space, and since category is a function of merchant, synthetic rows carried no category at all —
the exact leak the spike brief had warned to watch for.

The commensurability audit did not catch it, because `payee_in_anchor` tested membership against
a single set unioning senders and payees. A customer payee is in that set. The check passed a
leak it existed to find.

Fixed in three parts: the envelope now measures sender and payee pools separately, the simulator
draws merchant-role entities from the payee pool, and the audit tests each side against its own
namespace. Merchants are drawn in the anchor's own proportions too, because BankSim sends 84.9%
of its payments to one merchant and an even draw produces a category mix the anchor never emits;
synthetic now sits at 83.7% against that 84.9%.

Two things are worth recording about the correction:

**It moved the number the wrong way for the hypothesis.** Transfer went from 0.2804 to **0.2377**.
Closing a leak that was inflating a result is the normal direction, and it means the original
NO-GO was not an artefact of the bug.

**The saturated column came down.** Synthetic-trained on the synthetic holdout fell from a perfect
**1.000** to **0.934**. A perfect score was the leak announcing itself, and we read it as strength.

The lesson generalises past this dataset: an audit that pools two namespaces cannot see a
namespace error. The regression test now asserts that a customer-as-payee fails the payee check.

## What the three anchors have in common

The same shape appears every time, and it is not a property of any one dataset:

| | PaySim | AMLSim | BankSim |
| --- | --- | --- | --- |
| repeated entities | ✗ ~1 txn/sender | ✓ 132 | ✓ 165 |
| own fraud non-trivial | not measurable | ✗ 0.77 | ✗ 0.74 |
| synthetic → real transfer | not measurable | ✗ 0.420 vs 0.456 | ✗ 0.238 vs 0.702 |
| real → synthetic transfer | — | ✗ 0.083 | ✗ 0.013 |

The reverse direction is the most telling. A detector trained on *real* fraud scores 0.013 on our
held-out family while scoring 0.954 on real fraud. Our M3 does not look like the fraud these
datasets contain, and their fraud does not look like our M3. Both directions of transfer fail, so
this is not a tuning problem or a leakage problem — the generated families and the public
datasets' fraud are different phenomena.

One number cuts the other way and is worth keeping: on BankSim `both` (0.9547) edges out `real`
(0.9544) on real fraud, so augmentation is at worst harmless here, unlike AMLSim where it hurt.
It is a rounding-error improvement, not a result.

## What the submission claims now

The original hypothesis — adaptive simulation lifts held-out recall on real data — is **not
supported on any anchor we can obtain**, and we can say precisely why rather than vaguely.

What the work does contribute, and what should be presented:

1. **A closed adaptive loop that runs, reproducibly**, with a convergence curve regenerated from
   logs rather than drawn. That was always the demonstrable part.
2. **The commensurability audit** — the part worth keeping. Five independent ways a synthetic
   attack was separable from its anchor by one field alone (amount scale, rail, device column,
   sender namespace, payee namespace), each of which made a held-out number look like skill. Every
   one was invisible until measured, including one the audit itself missed until the check was
   split per namespace. Every published synthetic-augmentation result we are aware of would fail
   the same check.
3. **A negative result with the instrument that produced it.** Three public anchors, three
   failures, each with a committed number and a split digest. A judge can rerun any of it.

That is a smaller claim than the one we set out to make and a more defensible one. It is also the
outcome `SPEC.md` explicitly reserved the right to report: *"If the adaptive loop does not beat
SMOTE on the held-out column, that result gets reported."*

## What is deliberately not next

No fourth dataset. The failure is consistent across three anchors with different structures, which
makes a fourth a lottery ticket rather than a plan. If the claim is to be rescued it needs either
real labelled fraud with genuine behavioural variety — which is exactly the data nobody publishes,
and the reason this problem is open — or generated families built to imitate a specific anchor's
fraud rather than a taxonomy written independently of it. Both are larger than the time remaining.
