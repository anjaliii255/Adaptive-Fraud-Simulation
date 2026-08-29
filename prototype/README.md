# The working prototype

Five acts that walk a judge through the investigation: the mission and the threat model, the red
team generating an attack and the audit gate ruling on it, the blue team scoring a transaction, the
loop closing, the verdict, and the forensic verification of every number on screen.

```bash
streamlit run prototype/app.py
```

## The rule this app is built on

**Static-first.** Every act renders in full from artefacts committed to this repository, with no
live call and no dataset download. Live operations are enhancements; when one fails the act still
shows real numbers, badged so replayed data is never mistaken for live.

That is not defensive habit. `data/**` is gitignored . the real anchors are gigabytes and licensed
. so a judge opening the public URL cold has no AMLworld to run against. What they get instead is
every headline number read from the artefact that produced it, stamped with its commit.

Two badges, used without exception:

- **● LIVE SIMULATION** . this ran just now, in this process
- **◐ REPRODUCED · captured from canonical artifact** . this is replayed

## The three pillars, and where each one is

| pillar | act | live or replayed |
| --- | --- | --- |
| **Identify** | 01 MISSION . 9 vectors, 8 fully simulated, grouped by taxonomy level | static, from `vectors.yaml` |
| **Generate** | 02 RED TEAM . the real `Simulator` and `envelope.audit`, in-process | **live**, ~1s |
| **Defend** | 02 BLUE TEAM RESPONDS . the real `LGBMDetector` scores a transaction | **live**, ~1.3s to fit |
| the loop | 03 THE CLOSED LOOP . evasion 0.836 → 0.054 over 6 rounds, 7 seeds | replayed |
| the result | 04 THE VERDICT . recomputed from the artefact on every render | replayed |
| provenance | 05 VERIFY . re-derives the sign test and hashes in front of you | live-safe |

## Deploying to Streamlit Community Cloud

Entrypoint `prototype/app.py`, branch `main`. Dependencies come from `prototype/requirements.txt`,
which is measured rather than guessed:

- `scikit-learn` is there because `envelope.audit` imports `average_precision_score` **lazily** .
  the separability bars are PR-AUC per contract field. Found by installing this file into a clean
  environment and running the path, not by reading imports.
- `lightgbm`, `pandas` and `shap` are there for act 02's blue panel: the real detector, its graded
  action, and its SHAP reason codes.
- **`torch` and `torch-geometric` stay out.** The GNN is benched, not demonstrated, and they would
  not fit the free tier.

Measured on a clean environment built from this file alone: **580 MB installed, 301 MB resident
with the detector fitted**, against a ~1 GB tier.

The two live worlds are cached separately on purpose. `attack_world()` builds the anchor and
envelope in about a second and holds no model; `defence_world()` adds the fitted detector and is
paid for only if the blue panel is used. A judge who only runs the audit never waits for LightGBM.

## What this app must never do

Touch a committed number, or write to any artefact. Act 04 recomputes its means and sign tests from
the JSON on every render precisely so a figure cannot drift from its artefact . if the file changes
the screen changes with it, and if they ever disagree the app is wrong rather than the record.
