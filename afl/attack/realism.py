"""Realism constraints — the leash on the optimiser.

Without this, the optimiser wins by producing traffic no bank would ever see: negative dwell
times, 500-hop chains, amounts to the paisa. Each check returns a penalty in [0, 1]; the
optimiser's fitness is `evasion − λ·realism_penalty`.

This is a *cheap* gate that runs every round. The expensive verdict lives in `afl/fidelity/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from afl.contract.schema import AttackBatch, Transaction


@dataclass(frozen=True)
class RealismBounds:
    """What the leash compares a batch against.

    `DEFAULT` is what ticket 12 shipped and what the committed A/B/C/D run was scored under. Its
    three numbers are guesses, and ticket 14 measured what they cost: see `docs/realism-leash.md`.
    `from_anchor` replaces them with the anchor's own statistics, which is what the ticket asked
    for — bounds derived from real data rather than chosen at a keyboard.
    """

    max_degree_concentration: float = 0.6
    max_round_share: float = 0.5
    target_precision_share: float = 0.6
    source: str = "default"

    @classmethod
    def from_anchor(
        cls, txns: list[Transaction], name: str = "anchor", *, headroom: float = 0.1
    ) -> RealismBounds:
        """Measure the anchor, so 'unrealistic' means 'unlike this data' rather than 'unlike 0.6'.

        Concentration and round-share are ceilings, so they take the anchor's value plus headroom:
        exceeding what the real data does is the tell. Precision is a target to sit near, not a
        ceiling, so it is taken as measured.
        """
        if not txns:
            return cls()
        fraud = [t for t in txns if t.is_fraud] or txns
        return cls(
            max_degree_concentration=min(1.0, _degree_concentration(fraud) + headroom),
            max_round_share=min(1.0, _round_number_share(txns) + headroom),
            target_precision_share=_amount_precision_share(txns),
            source=name,
        )


DEFAULT_BOUNDS = RealismBounds()


@dataclass
class RealismReport:
    """Cheap per-batch verdict on whether generated traffic still looks like traffic."""

    penalty: float  # 0 = indistinguishable-shaped, 1 = obviously fake
    violations: list[str] = field(default_factory=list)
    detail: dict[str, float] = field(default_factory=dict)
    #: Each soft term's own contribution, so a penalty that never moves can be seen to be one
    #: term saturating rather than three terms agreeing. This is how ticket 14 found the leash
    #: was not binding: `precision` sat at its ceiling while the other two never fired at all.
    terms: dict[str, float] = field(default_factory=dict)
    #: The soft score alone, *not* overridden by the hard-violation cliff. `penalty` pins to 1.0
    #: when a violation is present, which is right for a gate and wrong for a gradient — an
    #: optimiser needs the graded signal separately from the veto.
    soft_penalty: float = 0.0
    bounds: str = "default"

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def binding(self) -> bool:
        """Is any term actually responding to this batch, or is the number a constant?

        A leash whose penalty is the same for every candidate cannot change which candidate wins:
        fitness is `evasion - λ·penalty`, and subtracting an equal constant from every trial leaves
        the argmax alone. That is a λ that does nothing, and it is invisible unless asked directly.
        """
        return any(v > 0 for k, v in self.terms.items() if k != "precision")


def _schema_violations(txns: list[Transaction]) -> list[str]:
    """Things pydantic cannot catch because they are cross-row, not per-row."""
    out: list[str] = []
    if any(t.src == t.dst for t in txns):
        out.append("self_transfer")
    if len({t.txn_id for t in txns}) != len(txns):
        out.append("duplicate_txn_id")
    if any(t.is_fraud and t.vector_id is None for t in txns):
        out.append("unlabelled_fraud_row")
    if any((not t.is_fraud) and t.vector_id is not None for t in txns):
        out.append("provenance_leak_on_legit_row")
    return out


def _round_number_share(txns: list[Transaction]) -> float:
    """Real traffic has some round amounts, but not 90% of them."""
    if not txns:
        return 0.0
    return sum(1 for t in txns if abs(t.amount - round(t.amount, -2)) < 1e-9) / len(txns)


def _amount_precision_share(txns: list[Transaction]) -> float:
    """Share of amounts with sub-rupee precision — a generator tell when it hits 100%."""
    if not txns:
        return 0.0
    return sum(1 for t in txns if abs(t.amount - int(t.amount)) > 1e-9) / len(txns)


def _degree_concentration(txns: list[Transaction]) -> float:
    """Max share of fraud edges landing on a single beneficiary. 1.0 = one node does everything."""
    if not txns:
        return 0.0
    counts: dict[str, int] = {}
    for t in txns:
        counts[t.dst] = counts.get(t.dst, 0) + 1
    return max(counts.values()) / len(txns)


def check(
    batch: AttackBatch,
    *,
    max_degree_concentration: float | None = None,
    max_round_share: float | None = None,
    min_fraud_rows: int = 1,
    bounds: RealismBounds | None = None,
) -> RealismReport:
    """Cheap per-batch verdict. Hard violations pin the penalty at 1.0.

    Defaults are unchanged from what the committed A/B/C/D run was scored under; pass `bounds` to
    compare against a measured anchor instead. The explicit float arguments still win over
    `bounds`, so existing callers keep their behaviour exactly.
    """
    bounds = bounds or DEFAULT_BOUNDS
    max_degree = (
        max_degree_concentration
        if max_degree_concentration is not None
        else bounds.max_degree_concentration
    )
    max_round = max_round_share if max_round_share is not None else bounds.max_round_share

    fraud = batch.fraud_transactions
    violations = _schema_violations(batch.transactions)
    if len(fraud) < min_fraud_rows:
        violations.append("empty_attack")
    if any(t.amount <= 0 for t in batch.transactions):
        violations.append("non_positive_amount")

    # time must move forward within an attack run
    for run in {t.attack_run_id for t in fraud if t.attack_run_id}:
        rows = sorted((t for t in fraud if t.attack_run_id == run), key=lambda t: t.ts)
        if any(
            a.ts > b.ts for a, b in zip(rows, rows[1:], strict=False)
        ):  # pragma: no cover - sorted by construction
            violations.append("non_monotonic_time")
            break

    degree = _degree_concentration(fraud)
    round_share = _round_number_share(fraud)
    precision_share = _amount_precision_share(fraud)

    terms = {
        "degree": max(0.0, degree - max_degree) / max(1e-9, 1 - max_degree),
        "round": max(0.0, round_share - max_round) / max(1e-9, 1 - max_round),
        # a target to sit near, not a ceiling: real rails give a mix, not all-or-nothing
        "precision": abs(precision_share - bounds.target_precision_share) * 0.5,
    }
    soft = min(1.0, sum(terms.values()) / 3.0)
    penalty = 1.0 if violations else soft

    return RealismReport(
        penalty=round(penalty, 6),
        soft_penalty=round(soft, 6),
        violations=violations,
        detail={
            "degree_concentration": round(degree, 4),
            "round_number_share": round(round_share, 4),
            "amount_precision_share": round(precision_share, 4),
            "n_fraud": float(len(fraud)),
        },
        terms={k: round(v, 6) for k, v in terms.items()},
        bounds=bounds.source,
    )
