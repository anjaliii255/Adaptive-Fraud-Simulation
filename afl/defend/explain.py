"""SHAP reason codes.

A score of 0.91 is not an answer an analyst can act on. Reason codes turn it into "beneficiary
saw 14 inbound payments in an hour, from 14 accounts that had never paid it before" — which is
also what makes a false positive arguable instead of mysterious.

Falls back to global feature importance when SHAP is unavailable; the fallback is labelled in the
reason string so nobody mistakes a global explanation for a local one.
"""

from __future__ import annotations

import logging

import numpy as np

from afl.contract.schema import Transaction

log = logging.getLogger(__name__)

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


def phrase(feature: str, value: float, direction: float) -> str:
    """One feature rendered as an analyst-readable reason code."""
    base = FEATURE_PHRASES.get(feature, feature.replace("_", " "))
    arrow = "↑" if direction > 0 else "↓"
    return f"{arrow} {base} ({value:g})"


def reason_codes(detector, txns: list[Transaction], top_k: int = 3) -> list[list[str]]:
    """Top-k local drivers per transaction. Returns one list of strings per input row."""
    if detector.model is None:
        return [[] for _ in txns]

    X = detector.features.transform(txns, update=False).reindex(
        columns=detector.feature_names, fill_value=0.0
    )
    values = X.to_numpy()

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
        importance = detector.feature_importance()
        top = [k for k, _ in list(importance.items())[:top_k]]
        return [[f"(global) {FEATURE_PHRASES.get(k, k)}" for k in top] for _ in txns]

    out: list[list[str]] = []
    for i in range(len(txns)):
        order = np.argsort(-np.abs(sv[i]))[:top_k]
        out.append(
            [phrase(detector.feature_names[j], float(values[i, j]), float(sv[i, j])) for j in order]
        )
    return out


def global_importance(detector, top_k: int = 15) -> dict[str, float]:
    """Model-level view — for the model card, not for an individual decision."""
    return dict(list(detector.feature_importance().items())[:top_k])
