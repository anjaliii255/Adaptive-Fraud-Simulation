"""Aggregate the three levels plus privacy into one verdict, and write it down.

The levels are not equal and the scorecard must not pretend they are. Level 3 is the gate:
a generator that resembles real traffic but teaches a model nothing has failed, however pretty
its histograms are. Levels 1 and 2 are diagnostics — they explain *why* level 3 failed.

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
from afl.fidelity import level1_statistical, level2_structural, level3_utility, privacy

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass(frozen=True)
class Thresholds:
    """Set once, before any result exists. Moving one after is how a bar stops being one."""

    level1_min: float = 0.70
    level2_min: float = 0.60
    max_tstr_gap: float = 0.15
    min_recall_lift: float = 0.0
    min_dcr_ratio: float = 0.80
    max_mia_advantage: float = 0.20


@dataclass
class Scorecard:
    """The three levels plus privacy, aggregated into one verdict."""

    levels: dict[str, Any] = field(default_factory=dict)
    verdict: str = WARN
    reasons: list[str] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Level 3 weighted double — utility is the claim, resemblance is the evidence for it."""
        weights = {"level1": 1.0, "level2": 1.0, "level3": 2.0, "privacy": 1.0}
        got = {k: v.get("score", 0.0) for k, v in self.levels.items() if isinstance(v, dict)}
        total = sum(weights.get(k, 1.0) for k in got)
        return round(sum(got[k] * weights.get(k, 1.0) for k in got) / total, 4) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "reasons": self.reasons,
            "thresholds": self.thresholds.__dict__,
            "meta": self.meta,
            "levels": self.levels,
        }

    def to_markdown(self) -> str:
        marks = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}
        lines = [
            "# Fidelity scorecard",
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
                lines.append(f"| {key} | {asks.get(key, '')} | {body.get('score', 'n/a')} |")

        if self.reasons:
            lines += ["", "## Why", ""] + [f"- {r}" for r in self.reasons]

        u = self.levels.get("level3", {}).get("tstr", {})
        a = self.levels.get("level3", {}).get("augmentation", {})
        if u:
            lines += [
                "",
                "## The numbers that matter",
                "",
                f"- TSTR PR-AUC {u.get('tstr_pr_auc')} vs TRTR {u.get('trtr_pr_auc')} "
                f"(gap {u.get('tstr_gap')}, bar ≤ {self.thresholds.max_tstr_gap})",
                f"- held-out recall lift from augmentation: {a.get('recall_lift')}",
            ]
        return "\n".join(lines)

    def save(self, directory: str | Path = "artifacts") -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        js = directory / "fidelity_scorecard.json"
        md = directory / "fidelity_scorecard.md"
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
) -> Scorecard:
    """Run every level that has the inputs it needs, then judge.

    Levels 3 and privacy need a real train/test split; without one they are skipped and said to
    be skipped, never quietly scored as passing.
    """
    card = Scorecard(thresholds=thresholds or Thresholds(), meta=meta or {})
    card.levels["level1"] = level1_statistical.report(real, synth)
    card.levels["level2"] = level2_structural.report(real, synth)

    if real_train and real_test and detector_factory is not None:
        card.levels["level3"] = level3_utility.report(
            real_train, real_test, synth, detector_factory, max_gap=thresholds.max_tstr_gap
        )
        card.levels["privacy"] = privacy.report(
            real_train,
            real_test,
            synth,
            seed=seed,
            min_dcr_ratio=thresholds.min_dcr_ratio,
            max_mia_advantage=thresholds.max_mia_advantage,
        )
    else:
        card.reasons.append("level 3 + privacy skipped: no real train/test split or detector given")

    return _judge(card)


def _judge(card: Scorecard) -> Scorecard:
    t = card.thresholds
    hard, soft = [], []

    l1 = card.levels.get("level1", {}).get("score", 0.0)
    l2 = card.levels.get("level2", {}).get("score", 0.0)
    if l1 < t.level1_min:
        soft.append(
            f"level 1 score {l1} below {t.level1_min} "
            f"(worst: {card.levels['level1'].get('worst_column')})"
        )
    if l2 < t.level2_min:
        soft.append(
            f"level 2 score {l2} below {t.level2_min} "
            f"(worst: {card.levels['level2'].get('worst_motif')})"
        )

    l3 = card.levels.get("level3")
    if l3:
        gap = l3["tstr"]["tstr_gap"]
        lift = l3["augmentation"]["recall_lift"]
        if gap > t.max_tstr_gap:
            hard.append(f"TSTR gap {gap} exceeds the bar {t.max_tstr_gap}")
        if lift < t.min_recall_lift:
            hard.append(
                f"held-out recall lift {lift} below {t.min_recall_lift} — the data does not help"
            )

    p = card.levels.get("privacy")
    if p:
        hard.extend(p.get("flags", []))

    card.reasons.extend(hard + soft)
    card.verdict = FAIL if hard else (WARN if soft or not l3 else PASS)
    return card
