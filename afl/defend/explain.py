"""SHAP reason codes.

A score of 0.91 is not an answer an analyst can act on. Reason codes turn it into "beneficiary
saw 14 inbound payments in an hour, from 14 accounts that had never paid it before" — which is
also what makes a false positive arguable instead of mysterious.

Three rules this module keeps, all of them enforced by `tests/test_decision.py` rather than
promised here:

**A flagged transaction is never unexplained.** `MIN_REASONS` codes, or the row does not get
flagged. Not a target: `reason_codes` pads from global importance when SHAP has fewer than that
many non-zero drivers, because "we stepped this customer up and cannot say why" is the failure
mode reason codes exist to prevent.

**A global explanation is labelled as one, in the string itself.** Not in a log line nobody
reads, not in a sibling field somebody drops on the way to the UI. `GLOBAL_PREFIX` travels with
the text, so an explanation that is not about *this* transaction says so wherever it is shown.

**Explaining is not a mode you can switch off for flagged rows.** It is priced per flagged row,
not per scored row, which is what makes that affordable: on the loop's batches under 1% of rows
carry an action, so the SHAP call is on a hundred rows and not on a hundred thousand.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from afl.contract.metrics import Action, DetectorScore
from afl.contract.schema import Transaction

log = logging.getLogger(__name__)

#: The floor a flagged transaction is never allowed to fall below.
MIN_REASONS = 3

#: Carried in the reason text itself, never alongside it. See the module docstring.
GLOBAL_PREFIX = "(global, not this transaction)"

#: Human phrasing for the features an analyst will actually see. Keyed by the names in
#: `afl/defend/features.py`; anything missing falls back to the column name with the underscores
#: knocked out, which is readable because the columns are named to be.
FEATURE_PHRASES = {
    "amount": "transaction amount",
    "log_amount": "transaction amount",
    "is_night": "overnight activity",
    "amount_is_round": "suspiciously round amount",
    "amount_under_10k": "amount just below the 10k reporting threshold",
    "amount_under_1k": "amount just below the 1k threshold",
    # what the paying account has been doing
    "src_out_cnt_3600s": "payments sent in the last hour",
    "src_out_cnt_86400s": "payments sent in the last day",
    "src_out_sum_86400s": "amount sent in the last day",
    "src_out_uniq_dst_3600s": "distinct beneficiaries paid in the last hour",
    "src_out_uniq_dst_86400s": "distinct beneficiaries paid in the last day",
    "src_seconds_since_last_out": "time since the account's last payment",
    "src_out_txn_count": "payments this account has ever sent",
    "src_amount_z": "amount vs this account's normal",
    "src_amount_ratio_to_mean": "amount vs this account's average",
    "src_out_uniq_beneficiaries": "beneficiaries this account has ever paid",
    "src_account_age_s": "how long this account has been active",
    # what has been happening to the beneficiary
    "dst_in_cnt_3600s": "payments received by the beneficiary in the last hour",
    "dst_in_sum_3600s": "amount received by the beneficiary in the last hour",
    "dst_in_uniq_src_3600s": "distinct accounts paying this beneficiary in the last hour",
    "dst_in_uniq_src_86400s": "distinct accounts paying this beneficiary in the last day",
    "dst_in_degree": "how many accounts have ever paid this beneficiary",
    "dst_is_first_ever_inbound": "nobody has ever paid this beneficiary before",
    "dst_amount_z": "amount vs what this beneficiary usually receives",
    "dst_account_age_s": "how long the beneficiary has been active",
    # money passing straight through
    "src_seconds_since_last_in": "time between the money arriving and leaving again",
    "src_passthrough_ratio_3600s": "share of what arrived that is being forwarded on",
    "dst_out_cnt_3600s": "payments the beneficiary itself sent in the last hour",
    # the payer-payee relationship
    "pair_is_first_payment": "first-ever payment to this beneficiary",
    "pair_txn_count": "past payments to this beneficiary",
    "pair_seconds_since_last": "time since this beneficiary was last paid",
    # device
    "src_uniq_devices": "devices seen on this account",
    "device_is_new": "unrecognised device",
    "device_seconds_since_first": "how long this device has been on the account",
}


def readable(feature: str) -> str:
    """The analyst phrase for a column, or the column name made readable."""
    return FEATURE_PHRASES.get(feature, feature.replace("_", " "))


def _duration(seconds: float) -> str:
    """Seconds as something a human reads at a glance. `276480` is not a reason code."""
    seconds = abs(float(seconds))
    if not math.isfinite(seconds):
        return str(seconds)
    for size, unit in ((86_400.0, "d"), (3_600.0, "h"), (60.0, "m")):
        if seconds >= size:
            return f"{seconds / size:,.1f}{unit}"
    return f"{seconds:,.0f}s"


def _value(feature: str, value: float) -> str:
    """A feature's value, formatted for the person reading it rather than for a debugger.

    `7.57632e+06` is a number a model produced; `7,576,325` is a number an analyst can argue
    with, and a dwell time is worth reading as `4.6m` rather than as 276 thousand of anything.

    Non-finite values pass through as themselves. A reason code is the last thing that should
    raise: a scorer that crashes formatting an explanation has turned an explainability
    feature into an outage.
    """
    if not math.isfinite(value):
        return str(value)
    if feature.endswith("_s") or "_seconds_" in feature:
        return _duration(value)
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    if abs(value) >= 1_000:
        return f"{value:,.0f}"
    if abs(value) >= 0.01:
        return f"{value:,.2f}"
    return f"{value:.3g}"


def phrase(feature: str, value: float, direction: float) -> str:
    """One feature rendered as an analyst-readable reason code."""
    arrow = "↑" if direction > 0 else "↓"
    return f"{arrow} {readable(feature)} ({_value(feature, value)})"


def global_phrases(detector, top_k: int, spoken: set[str] | None = None) -> list[str]:
    """Top features model-wide, each labelled as a global explanation in its own text.

    `spoken` is the set of analyst phrases already used on this row, so padding a short local
    explanation never repeats something the row has already said.
    """
    spoken = set(spoken or ())
    out = []
    for name in detector.feature_importance():
        text = readable(name)
        if text in spoken:
            continue
        spoken.add(text)
        out.append(f"{GLOBAL_PREFIX} {text}")
        if len(out) >= top_k:
            break
    return out


def reason_codes(
    detector,
    txns: list[Transaction],
    top_k: int = MIN_REASONS,
    values: np.ndarray | None = None,
) -> list[list[str]]:
    """Top-`top_k` local drivers per transaction, as one list of strings per input row.

    `values` is the design matrix for `txns` when the caller already built it — which the
    detector always has, having just scored these rows. Rebuilding it here costs a second pass
    over every entity's history, and on an anchor with deep histories that is the expensive half
    of scoring, not a rounding error.
    """
    if not txns:
        return []
    if getattr(detector, "model", None) is None:
        # No model means no scores above zero, so nothing is flagged and nothing needs a reason.
        # Said explicitly rather than returned empty, in case a caller flags on something else.
        return [["(no model fitted — this score has no explanation)"] for _ in txns]

    if values is None:
        values = (
            detector.features.transform(txns, update=False)
            .reindex(columns=detector.feature_names, fill_value=0.0)
            .to_numpy()
        )
    values = np.asarray(values, dtype=float)

    try:
        import shap

        explainer = getattr(detector, "_explainer", None)
        if explainer is None:
            explainer = shap.TreeExplainer(detector.model)
            detector._explainer = explainer
        sv = explainer.shap_values(values)
        if isinstance(sv, list):  # older shap returns one array per class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, features, classes)
            sv = sv[:, :, -1]
    except Exception as e:  # shap missing, or model unsupported by TreeExplainer
        log.warning("SHAP unavailable (%s) — falling back to global importance", e)
        fallback = global_phrases(detector, top_k)
        return [list(fallback) for _ in txns]

    out: list[list[str]] = []
    for i in range(len(txns)):
        row: list[str] = []
        used: set[str] = set()  # analyst phrases, not column names — see below
        for j in np.argsort(-np.abs(sv[i])):
            if sv[i, j] == 0.0 or len(row) >= top_k:
                break
            name = detector.feature_names[j]
            # Deduplicate on the *phrase*, not the column. `amount` and `log_amount` are two
            # columns and one fact, and a tree splitting on both spends two of an analyst's
            # three reason codes saying "transaction amount" twice. The weaker of the pair is
            # dropped and the next real driver takes the slot.
            spoken = readable(name)
            if spoken in used:
                continue
            used.add(spoken)
            row.append(phrase(name, float(values[i, j]), float(sv[i, j])))
        if len(row) < top_k:
            # A tree can genuinely attribute a score to fewer than `top_k` distinct facts — a
            # shallow model, or a row that took a short path. Pad rather than return two reasons.
            row += global_phrases(detector, top_k - len(row), spoken=used)
        out.append(row)
    return out


def unexplained(scores: list[DetectorScore], min_reasons: int = MIN_REASONS) -> list[str]:
    """Ids of flagged transactions carrying fewer than `min_reasons` reason codes.

    The invariant as a function, so the test, the artefact builder and the demo all check the
    same thing rather than three approximations of it.
    """
    return [
        s.txn_id for s in scores if s.action is not Action.ALLOW and len(s.reasons) < min_reasons
    ]


def assert_flagged_rows_are_explained(
    scores: list[DetectorScore], min_reasons: int = MIN_REASONS
) -> None:
    """Raise if a flagged transaction is short of reasons — a decision nobody can read."""
    short = unexplained(scores, min_reasons)
    if short:
        raise AssertionError(
            f"{len(short)} flagged transaction(s) carry fewer than {min_reasons} reason codes, "
            f"e.g. {short[:3]} — a decision an analyst cannot argue with is not a decision"
        )


def global_importance(detector, top_k: int = 15) -> dict[str, float]:
    """Model-level view — for the model card, not for an individual decision."""
    return dict(list(detector.feature_importance().items())[:top_k])


__all__ = [
    "FEATURE_PHRASES",
    "GLOBAL_PREFIX",
    "MIN_REASONS",
    "assert_flagged_rows_are_explained",
    "global_importance",
    "global_phrases",
    "phrase",
    "readable",
    "reason_codes",
    "unexplained",
]
