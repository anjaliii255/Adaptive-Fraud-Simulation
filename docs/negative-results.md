# Negative results — three layers that did not earn deployment

_Each of these was built, measured against the thing it was supposed to beat, and benched. They are
published because a negative result that lives on one laptop is not a result, and because the next
person to reach for one of these on this problem should be able to read what happened when we did._

In all three cases the gate is enforced rather than remembered: `assert_config_matches_promotion`
refuses to let a benched model be switched on while the committed artefact says the gate said no,
and a test runs it against what is on disk.

## The anomaly layer, and the result it did not produce

The supervised model can only catch what it has labels for, and leave-one-attack-out is defined by
holding one family's labels back. An outlier score fitted on legit traffic alone has no notion of
"the fraud I have seen", so the design bet was that it degrades more gracefully against the unseen
family and sits underneath the ensemble as a floor. `docs/anomaly.md` is where that bet was
settled; `artifacts/anomaly/<anchor>.json` is the evidence.

```bash
make anomaly     # five systems, one fold, one operating point — and rewrite the write-up
```

**The bet lost, on both anchors.** The supervised model does not collapse on the held-out family:
on PaySim it reaches PR-AUC 0.524 and recall@1%FPR 1.000 there, against 0.152 / 0.478 on the
anchor's own real labelled fraud in the same test window. A family it has never seen is *easier*
than one it trains on, which is a finding about the injected rows rather than about
generalisation. And the anomaly layer is nowhere near it — 0.033 on PaySim, 0.003 on AMLSim, where
sorting by amount alone reaches 0.034. It is not the floor under the ensemble; it is below the
floor.

**The blend is the part that earns its place, and only on one anchor.** The weight is swept end to
end on the same pair of score vectors, so both halves appear in the same curve as the blends of
them. On PaySim the curve has an interior optimum — w=0.5 reaches 0.551 against 0.524 for the
supervised model alone — so the two together genuinely beat either. On AMLSim it rises
monotonically to w=1.0 and the shipped 0.7 costs 0.013 PR-AUC. The shipped weight stays where it
is: a weight chosen on the fold it is reported from is the tuning-on-test the baseline forbids.

**Two things were wrong underneath, and both are the kind that leave the metrics looking fine.**
The outlier score was min-maxed over whatever batch it was handed, so a transaction's score was a
statement about its company — 0.25 of drift on the same PaySim rows, invisible to PR-AUC because a
within-batch min-max is monotone, and blended 0.3-to-0.7 against a probability. And on PaySim the
simulator had barely any anchor accounts to stage attacks on, because the envelope's "seasoned
account" filter wanted senders that transact twice and PaySim's senders are unique per row: it
found 86 for 340 population slots and minted the other 254, so `sender_in_anchor` separated the
held-out family from the anchor at **PR-AUC 0.800** — 1.000 on a smaller sample, where the pool
empties entirely. Every anchored PaySim number produced before this ticket inherits that. Both
are fixed, and both are measured in the artefact rather than asserted in a comment.


## The sequence model, and the seat it did not earn

`make sequence` puts a GRU over per-entity history against the tuned LightGBM baseline on the
drift arc, writes `artifacts/sequence/<anchor>.json` and generates `docs/sequence.md` from it.
It needs the `deep` extra (`make setup-deep`); nothing else in the repo does, and the default
suite stays green without it.

**The axis is the experiment.** `ramp` is the drift engine's shape knob: 0 is a hard switch at the
takeover event, 1 is escalation spread across the whole tail. Sudden takeover is an *event* — the
amount jumps, the beneficiary is new, the device changes — and a per-row feature table sees all
three on the row itself. Gradual drift has no event to anchor on, and that is where per-row
features are supposed to run out. So each family is generated twice, at both ends of its own
declared search space with nothing else changed, and the two ends are reported separately against
the same haystack at the same threshold. The gate is decided on the gradual end only: a win on
sudden drift is a win at the easy end and does not promote anything.

**It lost, and the two anchors lost it differently.** On AMLSim, whose accounts carry 28 steps of
real history apiece, the sequence model reaches PR-AUC 0.391 on gradual S3 against LightGBM's
0.997, and 0.300 against 0.998 on gradual C1 — while fitting in 21s against LightGBM's 32s and
scoring at twice the rate. It is not an expensive model that bought a small lift; it is a cheap
model that lost. On PaySim it wins C1 by 0.987 to 0.773 — and PaySim's `nameOrig` is effectively
unique per row, so a real window there is **one step long** while the injected episodes carry
eight or nine. Window length alone sorts the two at PR-AUC 0.985. The win is the fold's shape, not
the model's skill, and the audit built for exactly this model is what catches it.

**None of those four numbers is quotable, and the refusal is the point.** All four folds are
withheld — two on the provenance probe, two on the history audit this ticket added — so the
comparison is printed in brackets in `docs/sequence.md` and quoted nowhere. What stands is the
decision, not the margin: a layer nothing could measure honestly is a layer that does not enter
the reported table, which is the same answer a measured loss would have given.

**Enabling it is not a preference.** `assert_config_matches_promotion` refuses
`defend.sequence.enabled: true` while a committed artefact says the gate said no, and a test runs
it against what is on disk. The claim "it only ships if it wins" is enforced rather than
remembered.

One more thing fell out of it, and it revises a row we already published. These folds carry 550
injected AMLSim C1 rows where the leave-one-attack-out matrix carries 80, so the provenance probe
is far better powered here — and at that size it separates injected from real at PR-AUC 0.688,
over the bar, where the matrix measured 0.236 and reported the fold. Nothing about the generator
changed; the episode count did. `docs/sequence.md` names it, and `docs/loao.md`'s AMLSim C1 row
should be read as underpowered until `make loao` is re-run at this episode count.


## The temporal GNN, and which one shipped

`make gnn` puts graph attention over the account-beneficiary graph against the hand-rolled
graph features + LightGBM baseline on the mule families — S1 fan-in and layering, C3 instant
relay — writes `artifacts/gnn/<anchor>.json` and generates `docs/gnn.md` from it. Like `make
sequence` it needs the `deep` extra (`make setup-deep`), and `TemporalGNNDetector` raises without
it rather than degrading to a stand-in.

**The window is the design.** Time is cut into daily strides, and a payment is scored against the
graph of the previous seven days *up to the start of its own stride* — nothing at or after it,
and nothing older than the window. That is what makes the graph temporal rather than a graph with
timestamps on it, and it is also the constraint that decides the whole ticket: a model that may
only read what happened earlier cannot see a ring that has not formed yet.

**Three seeds, and the lift is paired across them.** Each seed regenerates its own pool and
refits every system, and the margin is reported as a per-seed difference with its spread and a
sign test — the same `Spread` and `Comparison` the three-system table uses, so the two are read
at one bar. A margin smaller than its own seed-to-seed spread does not promote anything, and
neither does one seed.

**It did not earn its seat, and the two anchors refused it for opposite reasons.** On AMLSim it
loses by almost the whole scale: PR-AUC 0.023 ± 0.038 on S1 against LightGBM's 0.998 and 0.002 ±
0.002 on C3 against 0.984, 0/3 seeds in its favour — but that fold is withheld rather than
counted, because AMLSim's rows are whole **days**. An injected mule ring is instantaneous on that
clock, and only **8.2%** of its rows (0.6% for C3) can see any earlier edge of their own episode,
against a floor of 20%. The model was asked about a shape it structurally cannot see. On PaySim,
whose clock is hourly and where 86% of injected rows can watch their own ring form, it is level
with what ships — −0.023 ± 0.229 on S1 (1/3 seeds) and +0.024 ± 0.343 on C3 (2/3), both inside
their own spread — and ahead of the graph-blocks-only baseline by +0.079 and +0.139. It costs
more either way: 47s to fit against LightGBM's 30s on AMLSim, 38s against 15s on PaySim.

**All four folds are withheld, and on PaySim the audit that catches them was built for this
model.** A row's *neighbourhood provenance* — what share of its endpoints' in-window neighbours
are injected rows — sorts injected S1 and C3 from real PaySim traffic at PR-AUC 0.53 and 0.54,
because a quarter to a third of the injected rows sit in a neighbourhood made only of other
injected rows. PaySim accounts appear roughly once, so a staged ring there is its own synthetic
island, and message passing over an island returns "synthetic" before it returns anything about
topology. The ordinary provenance probe agrees on S1 at 0.865.

**So the hand-rolled graph features are what ship**, which is the fallback ticket 18 named before
the experiment ran rather than after it. `config/defend/gnn.yaml` stays `enabled: false`,
`assert_config_matches_promotion` refuses to let it be turned on while the committed artefacts say
no, and a test checks both that config and this sentence against `artifacts/gnn/`.

One more thing fell out of it, and it revises a row we already published — the same way `make
sequence` did. These folds carry 173 injected PaySim S1 rows where the leave-one-attack-out matrix
carries 50, and at that size the provenance probe separates injected from real at 0.865 where the
matrix measured 0.298 and reported the fold. Nothing about the generator changed; the episode
count did. `docs/gnn.md` names it, and `docs/loao.md`'s PaySim S1 row should be read as
underpowered until `make loao` is re-run at this episode count.

