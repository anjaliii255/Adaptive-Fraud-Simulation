"""Realism constraints — the leash on the optimiser.

Without this, the optimiser wins by producing traffic no bank would ever see: negative dwell
times, 500-hop chains, amounts to the paisa. Each check returns a penalty in [0, 1]; the
optimiser's fitness is `evasion − λ·realism_penalty`.

This is a *cheap* gate that runs every round. The expensive verdict lives in `afl/fidelity/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from afl.contract.schema import AttackBatch, Transaction


@dataclass
class RealismReport:
    """Cheap per-batch verdict on whether generated traffic still looks like traffic."""

    penalty: float  # 0 = indistinguishable-shaped, 1 = obviously fake
    violations: list[str] = field(default_factory=list)
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


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
    max_degree_concentration: float = 0.6,
    max_round_share: float = 0.5,
    min_fraud_rows: int = 1,
) -> RealismReport:
    """Cheap per-batch verdict. Hard violations pin the penalty at 1.0."""
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

    soft = 0.0
    soft += max(0.0, degree - max_degree_concentration) / max(1e-9, 1 - max_degree_concentration)
    soft += max(0.0, round_share - max_round_share) / max(1e-9, 1 - max_round_share)
    soft += abs(precision_share - 0.6) * 0.5  # real rails give a mix, not all-or-nothing
    penalty = 1.0 if violations else min(1.0, soft / 3.0)

    return RealismReport(
        penalty=round(penalty, 6),
        violations=violations,
        detail={
            "degree_concentration": round(degree, 4),
            "round_number_share": round(round_share, 4),
            "amount_precision_share": round(precision_share, 4),
            "n_fraud": float(len(fraud)),
        },
    )
