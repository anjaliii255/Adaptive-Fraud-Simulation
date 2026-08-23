---
status: accepted
---

# The nine vectors, the three levels, and first-party fraud as the holdout

The architecture doc and `vectors.yaml` disagreed about what the nine attack vectors were: the code
carried an S/V/M split by *engine* (three graph, three velocity, three drift), the doc a strong /
common / mid split by *role in the argument*. The doc wins, because the taxonomy is part of the
claim rather than an implementation detail — a judge reads it as our model of the domain. The nine
ids are now S1–S3 (strong: the adaptive loop runs here), C1–C3 (common: must-catch load) and M1–M3
(mid: emerging, novelty, and the holdout), and they mean the same thing in a config file, a test, a
metric and a conversation.

## The three levels

A flat list of nine frauds reads as a list. Three levels read as a model, so `level` is a first-class
field and nothing may collapse it:

- **mechanism** — the fraud itself: S1 mule networks, S2 card testing, S3 account takeover, C1
  bust-out, C3 instant-A2A pass-through, M3 first-party fraud.
- **enabler** — what makes the fraud possible, where the loss lands elsewhere: C2 the APP scam
  (social engineering), M2 the synthetic-identity lifecycle (a fabricated identity).
- **model-attack** — an attack against our own detector: M1 boundary probing.

## Why first-party fraud is the leave-one-attack-out holdout

Every other family shares one assumption: there is an attacker, and they are not the account owner.
Supervised features quietly encode it — new device, new operator, unusual counterparty, ring
structure. First-party fraud (M3) is the family where `user == fraudster`, so none of those signals
fire. A holdout that merely happens to be *unseen* still lives inside the assumption; this one does
not, which is what makes it a generalisation test rather than a harder fold. It is also the single
biggest omission a domain judge would probe, and the reason the anomaly layer exists at all.

The trade: M3 is the hardest family we have, so the headline number will be low. That is the
intended shape of an honest result, not a problem to tune away.

## `status`: what the taxonomy declares vs what the code can generate

Declaring nine vectors while building three would have been a quiet lie, so every vector carries a
`status` alongside its taxonomy fields:

- **built** — the engines express the vector's defining behaviour (S1, S2, S3).
- **template** — schema-valid traffic of roughly the right shape, but the defining tell is not
  modelled yet (C1, C3, M1, M3). Usable as training load and as haystack; **not** reportable as a
  recall figure *for that family*. Every one carries a `gap` naming what is missing and the ticket
  that closes it, and `load_vectors` refuses a non-built vector with no `gap`.
- **planned** — cannot be generated at all (C2, M2). `Simulator.generate` raises `NotImplementedError`
  naming the ticket, because an attack family that silently emits nothing is indistinguishable from
  one the detector caught.

M3 being `template` means today's headline number is measured on a proxy for first-party fraud.
Ticket 03 makes it faithful; until then the number is labelled, not quoted.

## What moved, so nothing looks lost

The old ids were not deleted so much as absorbed, and every behaviour they carried survives as a
parameter rather than as a vector:

| old | old meaning | now |
| --- | --- | --- |
| S1, S2, S3 | fan-in, multi-hop layering, cycle | one vector, **S1**, with `motif` and `n_hops` |
| V1 | card-testing burst | **S2** |
| V2 | threshold-aware structuring | **M1** — pacing under a known limit *is* boundary probing |
| V3 | slow-drip siphon | S2's search space: low-and-slow is a pacing regime, not a family |
| M1 | account takeover | **S3** |
| M2 | dormant mule activation | C1's `dormancy_s` and S1's `hold_time_s` |
| M3 | gradual behavioural drift | S3's `ramp`: sudden vs gradual is the axis, not two vectors |

## Consequences

Two that will surprise someone otherwise:

**`motif` is a fixed template knob, not a searched one.** S1 spans fan-in through deep layering via
`n_hops`, which the optimiser does search. Searching `motif` itself needs categorical support in the
optimiser; that is ticket 12's call to make, not a decision to smuggle in here.

**The headline numbers moved, and not because anything improved.** Changing which families are
generated and what the holdout *is* changes every number downstream. The pre-freeze figures in the
README are not comparable to the post-freeze ones and were replaced rather than kept side by side.
