"""Aggregate the three levels plus privacy into one verdict, and write it down.

The levels are not equal and the scorecard must not pretend they are. Level 3 is the gate:
a generator that resembles real traffic but teaches a model nothing has failed, however pretty
its histograms are. Levels 1 and 2 are diagnostics — they explain *why* level 3 failed.

That ordering is enforced in two places rather than stated in one. `_judge` sorts every finding
into **hard** (level 3 and privacy) and **soft** (levels 1 and 2), and only hard findings can
fail a card. And the headline `score` is *capped* at the level-3 score whenever level 3 ran, so
a reader who quotes the single number rather than the verdict cannot be handed a pass that two
sets of histograms averaged into existence.

The scorecard is emitted every run and committed alongside the numbers it justifies, because a
fidelity claim with no artefact behind it is a sentence in a slide deck.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from afl.contract.schema import Transaction
from afl.evaluation import protocol
from afl.fidelity import level1_statistical, level2_structural, level3_utility, privacy

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass(frozen=True)
class Thresholds:
    """Set once, before any result exists. Moving one after is how a bar stops being one.

    The defaults are the values committed in the skeleton, before there was a generator to
    judge. `config/fidelity/thresholds.yaml` is where they actually live now, each with a stated
    reason, and `afl/fidelity/provenance.py` is what checks they have not moved since.
    """

    level1_min: float = 0.70
    level2_min: float = 0.60
    max_tstr_gap: float = 0.15
    min_recall_lift: float = 0.0
    min_dcr_ratio: float = 0.80
    max_mia_advantage: float = 0.20
    require_tstr_beats_amount_floor: bool = True

    @classmethod
    def from_values(cls, values: dict[str, Any]) -> Thresholds:
        """Build from the parsed `{value, why}` config, ignoring keys that are not bars."""
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in fields})


@dataclass
class Scorecard:
    """The three levels plus privacy, aggregated into one verdict."""

    levels: dict[str, Any] = field(default_factory=dict)
    verdict: str = WARN
    reasons: list[str] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)
    meta: dict[str, Any] = field(default_factory=dict)
    #: the evidence that `thresholds` predate these numbers; `afl.fidelity.provenance.load`
    provenance: dict[str, Any] | None = None
    #: which findings failed the card, and which only annotated it
    gate: dict[str, Any] = field(default_factory=dict)

    @property
    def level3_score(self) -> float | None:
        body = self.levels.get("level3")
        return body.get("score") if isinstance(body, dict) else None

    @property
    def score(self) -> float:
        """Level 3 weighted double and used as a ceiling — utility is the claim.

        Weighting it double is not enough on its own: two diagnostic levels at 0.9 still drag a
        level-3 score of 0.1 up to 0.5, and 0.5 reads like half a pass rather than like a
        generator that teaches a model nothing. So the weighted mean is capped at the level-3
        score. Resemblance can lower this number; it can never raise it past what the data is
        worth to a detector.
        """
        weights = {"level1": 1.0, "level2": 1.0, "level3": 2.0, "privacy": 1.0}
        got = {
            k: v["score"]
            for k, v in self.levels.items()
            if isinstance(v, dict) and isinstance(v.get("score"), int | float)
        }
        total = sum(weights.get(k, 1.0) for k in got)
        blended = sum(got[k] * weights.get(k, 1.0) for k in got) / total if total else 0.0
        ceiling = self.level3_score
        return round(blended if ceiling is None else min(blended, ceiling), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "gate": self.gate,
            "reasons": self.reasons,
            "thresholds": self.thresholds.__dict__,
            "threshold_provenance": self.provenance,
            "meta": self.meta,
            "levels": self.levels,
        }

    # ── the write-up ────────────────────────────────────────────────────────────
    def to_markdown(self) -> str:
        marks = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}
        title = self.meta.get("title") or "Fidelity scorecard"
        lines = [
            f"# {title}",
            "",
            f"**Verdict: {marks[self.verdict]} {self.verdict.upper()}** — score {self.score}",
            "",
            "| level | what it asks | score |",
            "| --- | --- | --- |",
        ]
        asks = {
            "level1": "do the marginals and joints match?",
            "level2": "is the structure and pacing right?",
            "level3": "does it teach a model anything? (**the bar**)",
            "privacy": "is it copying rows? (evidence, not proof)",
        }
        for key, body in self.levels.items():
            if isinstance(body, dict):
                score = body.get("score")
                shown = "withheld" if score is None else score
                lines.append(f"| {key} | {asks.get(key, '')} | {shown} |")

        if self.gate:
            lines += [
                "",
                f"The gate is **level 3**: {self.gate.get('summary', '')}",
            ]
        if self.reasons:
            lines += ["", "## Why", ""] + [f"- {r}" for r in self.reasons]

        lines += self._level3_section() + self._privacy_section() + self._provenance_section()
        return "\n".join(lines)

    def _level3_section(self) -> list[str]:
        l3 = self.levels.get("level3")
        if not isinstance(l3, dict):
            return []
        if l3.get("outcome") == "withheld":
            return ["", "## The numbers that matter", "", f"Withheld — {l3.get('why')}."]
        u, a, f = l3.get("tstr", {}), l3.get("augmentation", {}), l3.get("amount_floor", {})
        beats = l3.get("beats_amount_floor", {})
        out = [
            "",
            "## The numbers that matter",
            "",
            f"One real test window ({l3.get('n_real_test')} rows, "
            f"{l3.get('n_real_test_positives')} of them fraud), one operating point, four systems.",
            "",
            "| system | trained on | rows | PR-AUC | recall@FPR | p@k | beats the floor |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, body in (l3.get("systems") or {}).items():
            clears = beats.get(name)
            out.append(
                f"| {name} | {body.get('trained_on')} | {body.get('n_train')} | "
                f"{body.get('pr_auc')} | {body.get('recall_at_fixed_fpr')} | "
                f"{body.get('precision_at_k')} | "
                f"{'—' if clears is None else ('yes' if clears else '**no**')} |"
            )
        out += [
            "",
            f"- TSTR gap {u.get('tstr_gap')} against a bar of ≤ {self.thresholds.max_tstr_gap} "
            f"(ratio {u.get('tstr_ratio')} of what the real labels reach)",
            f"- held-out recall lift from augmentation: {a.get('recall_lift')} "
            f"(bar ≥ {self.thresholds.min_recall_lift})",
            f"- the floor ranks by amount alone, {f.get('direction')}, "
            f"direction chosen on {f.get('direction_chosen_on')}",
        ]
        return out

    def _privacy_section(self) -> list[str]:
        p = self.levels.get("privacy")
        if not isinstance(p, dict):
            return []
        d, m = p.get("dcr", {}), p.get("mia", {})
        c, ids = p.get("mia_time_control", {}), p.get("identifier_reuse", {})
        return [
            "",
            "## Privacy — evidence, not proof",
            "",
            f"- DCR ratio {d.get('dcr_ratio')} (bar ≥ {self.thresholds.min_dcr_ratio}): synthetic "
            f"rows sit {d.get('dcr_synth_median')} from their closest training row, where "
            f"training rows sit {d.get('dcr_train_median')} from each other",
            f"- exact duplicates of training rows: {d.get('identical_share')} of synthetic rows",
            f"- membership-inference AUC {m.get('mia_auc')}, advantage {m.get('advantage')} "
            f"(bar ≤ {self.thresholds.max_mia_advantage}), against {c.get('advantage')} for the "
            "same attack run between two halves of the holdout, where nothing was ever in "
            "training — the difference is the part that is about membership",
            f"- generated rows naming an account that exists in the anchor: "
            f"{ids.get('either_in_anchor')} (src {ids.get('src_in_anchor')}, "
            f"dst {ids.get('dst_in_anchor')}) — by design, and reported rather than flagged",
            "",
            "Neither number is a privacy guarantee. They are two ways of catching a generator "
            "that learned the distribution by copying it; passing them means the memorisation "
            "we tested for is not there, and nothing more. A formal claim needs differential "
            "privacy, and this project does not make one.",
        ]

    def _provenance_section(self) -> list[str]:
        p = self.provenance
        if not p:
            return []
        origin, wc = p.get("origin", {}), p.get("working_copy", {})
        mark = "✅" if p.get("predates_results") else "⚠️"
        out = [
            "",
            "## Did the bars predate the numbers?",
            "",
            f"{mark} {p.get('verdict')}",
            "",
            f"- thresholds: `{p.get('source')}` (sha256 {p.get('sha256')})",
            f"- origin commit `{(origin.get('commit') or '?')[:12]}` "
            f"({origin.get('committed_at')}), read back out of git and compared value by value",
            "- inherited unchanged from it: "
            + (", ".join(origin.get("bars_inherited") or []) or "—"),
            f"- introduced later: {', '.join(origin.get('bars_introduced_later') or []) or '—'}",
            f"- working copy clean: {wc.get('clean')}",
        ]
        moved = origin.get("moved_since_origin") or {}
        if moved:
            out += ["", "**A bar has moved since it was committed:**", ""]
            out += [
                f"- `{k}` {v['then']} → {v['now']} ({v['direction']})"
                for k, v in sorted(moved.items())
            ]
        history = p.get("history") or []
        if history:
            out += ["", "Every commit that has ever changed a bar:", ""]
            out += [
                f"- `{h['commit']}` {h['committed_at'][:10]} — {h['subject']} ({h['file']})"
                for h in history
            ]
        return out

    def save(self, directory: str | Path = "artifacts", stem: str = "fidelity_scorecard") -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        js = directory / f"{stem}.json"
        md = directory / f"{stem}.md"
        js.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        md.write_text(self.to_markdown())
        return {"json": js, "markdown": md}


def build(
    real: list[Transaction],
    synth: list[Transaction],
    real_train: list[Transaction] | None = None,
    real_test: list[Transaction] | None = None,
    detector_factory: Callable[[], Any] | None = None,
    thresholds: Thresholds | None = None,
    seed: int = 1337,
    meta: dict[str, Any] | None = None,
    fixed_fpr: float = protocol.DEFAULT_FPR,
    k: int = protocol.DEFAULT_K,
    synth_fraud: list[Transaction] | None = None,
    standalone: list[Transaction] | None = None,
    min_positives: int = 0,
    provenance: dict[str, Any] | None = None,
) -> Scorecard:
    """Run every level that has the inputs it needs, then judge.

    Levels 3 and privacy need a real train/test split; without one they are skipped and said to
    be skipped, never quietly scored as passing.

    `fixed_fpr` and `k` are the run's operating point. Level 3 compares detectors, so it has to
    compare them where everything else in the project is compared — a TSTR gap measured at a
    different threshold from the hero table is not evidence about the hero table.
    """
    card = Scorecard(thresholds=thresholds or Thresholds(), meta=meta or {}, provenance=provenance)
    card.meta.setdefault("operating_point", {"fixed_fpr": fixed_fpr, "k": k})
    card.levels["level1"] = level1_statistical.report(real, synth)
    card.levels["level2"] = level2_structural.report(real, synth)

    if real_train and real_test and detector_factory is not None:
        # Level 3 trains on the generated *fraud*; levels 1, 2 and privacy compare the generated
        # *traffic*. On an anchored pool those are the same list; on a standalone batch they are
        # not, and handing the whole batch to level 3 would swap the anchor's own background for
        # the simulator's and then measure that instead.
        fraud = synth_fraud if synth_fraud is not None else [t for t in synth if t.is_fraud]
        card.levels["level3"] = level3_utility.report(
            real_train,
            real_test,
            fraud,
            detector_factory,
            fixed_fpr=fixed_fpr,
            k=k,
            max_gap=card.thresholds.max_tstr_gap,
            standalone=standalone,
            min_positives=min_positives,
        )
        card.levels["privacy"] = privacy.report(
            real_train,
            real_test,
            synth,
            seed=seed,
            min_dcr_ratio=card.thresholds.min_dcr_ratio,
            max_mia_advantage=card.thresholds.max_mia_advantage,
        )
    else:
        card.reasons.append("level 3 + privacy skipped: no real train/test split or detector given")

    return _judge(card)


def _judge(card: Scorecard) -> Scorecard:
    """Hard findings fail the card; soft findings only annotate it.

    The split is the whole design. Levels 1 and 2 cannot fail a generator and cannot rescue one:
    a level-1 score of 0.95 next to a level-3 failure is a diagnostic sentence about *why* the
    generator failed, not a mitigating one.
    """
    t = card.thresholds
    hard, soft = [], []

    # A level that did not run is not a level that scored zero. Judging an absent one would
    # invent a finding about a measurement nobody took.
    for key, bar, worst in (
        ("level1", t.level1_min, "worst_column"),
        ("level2", t.level2_min, "worst_motif"),
    ):
        body = card.levels.get(key)
        if not isinstance(body, dict) or not isinstance(body.get("score"), int | float):
            continue
        if body["score"] < bar:
            soft.append(
                f"{key[:5]} {key[5:]} score {body['score']} below {bar} "
                f"(worst: {body.get(worst)})"
            )

    l3 = card.levels.get("level3")
    measured = isinstance(l3, dict) and l3.get("outcome") != "withheld"
    if measured:
        gap = l3["tstr"]["tstr_gap"]
        lift = l3["augmentation"]["recall_lift"]
        floor = l3["amount_floor"]["pr_auc"]
        if gap > t.max_tstr_gap:
            hard.append(f"TSTR gap {gap} exceeds the bar {t.max_tstr_gap}")
        if lift < t.min_recall_lift:
            hard.append(
                f"held-out recall lift {lift} below {t.min_recall_lift} — the data does not help"
            )
        if t.require_tstr_beats_amount_floor and not l3["beats_amount_floor"].get("tstr"):
            hard.append(
                f"TSTR PR-AUC {l3['tstr']['tstr_pr_auc']} loses to the amount floor {floor} — "
                "training on this data is worse than sorting the test window by amount"
            )
        if not l3["beats_amount_floor"].get("trtr"):
            soft.append(
                f"the real labels do not beat the amount floor either "
                f"({l3['tstr']['trtr_pr_auc']} vs {floor}) — this anchor's ceiling is the "
                "constraint, not the generator"
            )
        standalone = (l3.get("systems") or {}).get("standalone")
        if standalone and not l3["beats_amount_floor"].get("standalone"):
            soft.append(
                f"the generator's standalone output trains a detector to {standalone['pr_auc']}, "
                f"below the amount floor {floor} — reported, not gating"
            )
    elif isinstance(l3, dict):
        soft.append(f"level 3 withheld: {l3.get('why')}")

    p = card.levels.get("privacy")
    if p:
        hard.extend(p.get("flags", []))

    card.reasons.extend(hard + soft)
    card.verdict = FAIL if hard else (WARN if soft or not measured else PASS)
    card.gate = {
        "gate": "level3",
        "level3_measured": measured,
        "hard_findings": hard,
        "soft_findings": soft,
        "summary": (
            "level 3 was not measured, so no verdict above `warn` is available"
            if not measured
            else ("it holds" if not hard else "; ".join(hard))
        ),
        "note": (
            "levels 1 and 2 are diagnostics: they can annotate this card and lower its score, "
            "and they can never fail it or rescue it"
        ),
    }
    return card
