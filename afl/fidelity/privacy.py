"""Privacy — evidence, not proof.

Nothing here is a guarantee. These are the two checks that catch the failure mode that matters
in practice: a generator that "learned the distribution" by quietly copying rows.

  DCR  distance to closest record. If synthetic rows sit closer to training rows than training
       rows sit to each other, the generator is memorising.
  MIA  membership inference. If you can tell a training row from a held-out row by looking at
       the synthetic data alone, the synthetic data is leaking who was in the training set.

Say "evidence" in the write-up. A formal claim needs DP, and we are not making one.
"""

from __future__ import annotations

import numpy as np

from afl.contract.schema import Transaction
from afl.fidelity.level2_structural import embedding

MAX_ROWS = 4_000  # pairwise distances are O(n²); subsample rather than melt the machine


def _standardised(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(0), train.std(0) + 1e-9
    return (train - mu) / sd, (other - mu) / sd


def _subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if x.shape[0] <= n:
        return x
    idx = np.random.default_rng(seed).choice(x.shape[0], size=n, replace=False)
    return x[idx]


def _nearest(a: np.ndarray, b: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """For each row of `a`, the distance to its closest row in `b`."""
    if a.size == 0 or b.size == 0:
        return np.zeros(a.shape[0])
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def dcr(train: list[Transaction], synth: list[Transaction], seed: int = 1337) -> dict[str, float]:
    """Distance-to-closest-record, with the train-to-train distance as the honest baseline.

    A ratio near or above 1.0 is healthy. Well below 1.0 means synthetic rows hug real ones.
    """
    tr = _subsample(embedding(train), MAX_ROWS, seed)
    sy = _subsample(embedding(synth), MAX_ROWS, seed + 1)
    if tr.size == 0 or sy.size == 0:
        return {"dcr_ratio": 0.0, "identical_share": 0.0}
    tr_s, sy_s = _standardised(tr, sy)

    d_synth = _nearest(sy_s, tr_s)
    d_train = _nearest(tr_s, tr_s, exclude_self=True)
    med_train = float(np.median(d_train)) or 1e-9
    return {
        "dcr_synth_median": round(float(np.median(d_synth)), 6),
        "dcr_train_median": round(med_train, 6),
        "dcr_ratio": round(float(np.median(d_synth)) / med_train, 6),
        "dcr_5th_percentile": round(float(np.percentile(d_synth, 5)), 6),
        "identical_share": round(float((d_synth < 1e-6).mean()), 6),
    }


def mia_auc(
    members: list[Transaction],
    non_members: list[Transaction],
    synth: list[Transaction],
    seed: int = 1337,
) -> dict[str, float]:
    """Nearest-neighbour membership inference. 0.5 = no signal; the further above, the worse.

    `members` were in the generator's training set, `non_members` were not. The attacker's only
    tool is "how close is this record to the synthetic data".
    """
    m = _subsample(embedding(members), MAX_ROWS, seed)
    n = _subsample(embedding(non_members), MAX_ROWS, seed + 1)
    sy = _subsample(embedding(synth), MAX_ROWS, seed + 2)
    if min(m.shape[0], n.shape[0], sy.shape[0]) == 0:
        return {"mia_auc": 0.5, "advantage": 0.0}

    mu, sd = sy.mean(0), sy.std(0) + 1e-9
    scores = np.concatenate(
        [-_nearest((m - mu) / sd, (sy - mu) / sd), -_nearest((n - mu) / sd, (sy - mu) / sd)]
    )
    labels = np.concatenate([np.ones(m.shape[0]), np.zeros(n.shape[0])])

    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(labels, scores))
    return {"mia_auc": round(auc, 6), "advantage": round(abs(auc - 0.5) * 2, 6)}


def report(
    train: list[Transaction],
    holdout: list[Transaction],
    synth: list[Transaction],
    seed: int = 1337,
    min_dcr_ratio: float = 0.8,
    max_mia_advantage: float = 0.2,
) -> dict[str, object]:
    """DCR plus membership inference, with the flags that matter called out."""
    d = dcr(train, synth, seed)
    m = mia_auc(train, holdout, synth, seed)
    flags = []
    if d.get("dcr_ratio", 0.0) < min_dcr_ratio:
        flags.append(
            "synthetic rows sit closer to training rows than training rows do to each other"
        )
    if d.get("identical_share", 0.0) > 0.0:
        flags.append("exact duplicates of training rows present")
    if m["advantage"] > max_mia_advantage:
        flags.append("membership is inferable from the synthetic data alone")

    return {
        "level": "privacy",
        "dcr": d,
        "mia": m,
        "flags": flags,
        "score": round(
            min(1.0, d.get("dcr_ratio", 0.0) / min_dcr_ratio) * (1.0 - m["advantage"]), 4
        ),
        "caveat": "evidence, not proof — no formal privacy guarantee is claimed",
    }
