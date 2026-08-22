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

#: Human phrasing for the features an analyst will actually see.
FEATURE_PHRASES = {
    "amount": "transaction amount",
    "log_amount": "transaction amount",
    "src_cnt_3600s": "payments sent in the last hour",
    "src_cnt_86400s": "payments sent in the last day",
    "src_sum_86400s": "amount sent in the last day",
    "src_uniq_dst_3600s": "distinct beneficiaries in the last hour",
    "dst_cnt_3600s": "payments received by beneficiary in the last hour",
    "dst_sum_3600s": "amount received by beneficiary in the last hour",
    "src_seconds_since_last": "time since the account's last payment",
    "src_amount_z": "amount vs this account's normal",
    "src_amount_ratio_to_mean": "amount vs this account's average",
    "dst_in_degree": "how many accounts have ever paid this beneficiary",
    "dst_is_new_counterparty": "first-ever payment to this beneficiary",
    "src_uniq_devices": "devices seen on this account",
    "device_is_new": "unrecognised device",
    "amount_under_10k": "amount just below the 10k reporting threshold",
    "amount_under_1k": "amount just below the 1k threshold",
    "is_night": "overnight activity",
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
