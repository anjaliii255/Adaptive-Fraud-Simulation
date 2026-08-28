# Threat model — the vector taxonomy

_9 vectors identified, 8 fully simulated; M1 is an adversarial boundary-probing extension in template mode._

9 vectors identified, 8 fully simulated; M1 is an adversarial boundary-probing extension in template mode. Three engines. The ids match the architecture doc and are frozen; see
`docs/adr/0001-vector-taxonomy-and-holdout.md` for why each one is where it is, and
`afl/attack/templates/vectors.yaml` for the definitions. Adding a vector is a YAML edit.

| id | vector | engine | level | tier | status |
| --- | --- | --- | --- | --- | --- |
| S1 | Mule network & layering | graph | mechanism | strong | built |
| S2 | Card testing / BIN enumeration | velocity | mechanism | strong | built |
| S3 | Account takeover via drift | drift | mechanism | strong | built |
| C1 | Bust-out | drift | mechanism | common | built |
| C2 | UPI collect-request / APP scam | velocity | enabler | common | built |
| C3 | Instant-A2A pass-through | graph | mechanism | common | built |
| M1 | Boundary probing / paced evasion | velocity | model-attack | mid | template |
| M2 | Synthetic-identity lifecycle | drift | enabler | mid | built |
| M3 | First-party / friendly fraud | drift | mechanism | mid | built · **holdout** |

`level` is the taxonomy level and never gets flattened: mechanisms are the fraud, enablers are what
make it possible, and M1 is an attack against our own model. `tier` is the role in the build: the
adaptive loop wraps the strong three, the common three are must-catch load, the mid three are
novelty and the holdout.

`status` is the honest part — what the code can generate today, as opposed to what the taxonomy
declares:

- **built** — the engines express the vector's defining behaviour.
- **template** — valid traffic of roughly the right shape, but the defining tell is missing. Fine as
  training load and haystack; not reportable as a recall figure for that family. Each carries a
  `gap` naming what is missing and the ticket that fixes it.
- **planned** — cannot be generated. `Simulator.generate` raises and names the ticket, because a
  family that silently emits nothing looks exactly like a family the detector caught.

So **8 of the 9 are fully simulated.** Only M1 still owes work — it is an adversarial
boundary-probing extension, in template mode, and it arrives as the optimiser's own boundary walk. **M3 is the leave-one-attack-out holdout** because `user == fraudster` breaks
the legit-vs-attacker assumption every supervised feature rests on: the abuse runs on the owner's
own device, to beneficiaries the account already pays, elevated only against that account's own
history. A one-line amount rule at 1% FPR catches 3% of it, which is the point.

The commodity families are the mirror image and are meant to be caught. C1 spikes 7-10x against
its own tenure, C2 pays a first-time payee in plain sight, C3 relays money onward in under two
minutes. They are training load and fixed benchmarks, never the holdout. See
`docs/adr/0003-template-vectors.md`.
