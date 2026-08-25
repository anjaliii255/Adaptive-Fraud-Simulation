"""Sequence model — GRU / small transformer over per-entity transaction histories.

**Earn it.** This layer is additive: it only enters the reported table if it beats the LGBM
baseline on the same out-of-time split, and the comparison is reported either way. Requires the
`deep` extra (`uv sync --extra deep`); without torch it raises rather than silently degrading,
because a silently-degraded model in an ensemble is a number nobody can explain later.
"""

from __future__ import annotations

import numpy as np

from afl.contract.metrics import DetectorScore
from afl.contract.schema import AttackBatch, Transaction
from afl.defend.decision import DecisionPolicy
from afl.defend.features import sequence_tensor


class SequenceDetector:
    """GRU or small transformer over per-entity history. Earns its seat only by beating LightGBM."""

    def __init__(
        self,
        arch: str = "gru",  # "gru" | "transformer"
        hidden: int = 32,
        max_len: int = 32,
        epochs: int = 10,
        lr: float = 1e-3,
        policy: DecisionPolicy | None = None,
        seed: int = 1337,
    ) -> None:
        self.arch = arch
        self.hidden = hidden
        self.max_len = max_len
        self.epochs = epochs
        self.lr = lr
        self.policy = policy or DecisionPolicy()
        self.seed = seed
        self.model = None
        self._entity_scores: dict[str, float] = {}

    def _require_torch(self):
        try:
            import torch

            return torch
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "SequenceDetector needs the `deep` extra: uv sync --extra deep"
            ) from e

    def _build(self, n_feats: int):
        torch = self._require_torch()
        import torch.nn as nn

        torch.manual_seed(self.seed)
        if self.arch == "gru":

            class Net(nn.Module):
                def __init__(self, n_in, hidden):
                    super().__init__()
                    self.rnn = nn.GRU(n_in, hidden, batch_first=True)
                    self.head = nn.Linear(hidden, 1)

                def forward(self, x):
                    out, _ = self.rnn(x)
                    return self.head(out[:, -1, :]).squeeze(-1)
        elif self.arch == "transformer":

            class Net(nn.Module):
                def __init__(self, n_in, hidden):
                    super().__init__()
                    self.proj = nn.Linear(n_in, hidden)
                    layer = nn.TransformerEncoderLayer(
                        hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True
                    )
                    self.enc = nn.TransformerEncoder(layer, num_layers=2)
                    self.head = nn.Linear(hidden, 1)

                def forward(self, x):
                    return self.head(self.enc(self.proj(x))[:, -1, :]).squeeze(-1)
        else:
            raise ValueError(f"unknown arch {self.arch!r}")
        return Net(n_feats, self.hidden)

    def fit(self, txns: list[Transaction]) -> SequenceDetector:
        torch = self._require_torch()
        X, y, _ = sequence_tensor(txns, self.max_len)
        if len(set(y.tolist())) < 2:
            return self
        self.model = self._build(X.shape[-1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(max(1.0, (y == 0).sum() / max(1, (y == 1).sum())))
        )
        xb, yb = torch.tensor(X), torch.tensor(y, dtype=torch.float32)
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(xb), yb)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        """Entity-level score broadcast to that entity's transactions."""
        if self.model is None:
            return np.zeros(len(txns), dtype=float)
        torch = self._require_torch()
        X, _, ids = sequence_tensor(txns, self.max_len)
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.tensor(X))).numpy()
        self._entity_scores = dict(zip(ids, (float(p) for p in probs), strict=False))
        return np.array([self._entity_scores.get(t.src, 0.0) for t in txns])

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        """One reason code per row, which is below the floor ticket 09 set.

        Left as-is deliberately: this model is `enabled: false` and never reaches a reported
        table, so padding it here would be inventing an explanation for a detector nobody is
        scoring through. **ticket 17** owns making it earn its place, and
        `explain.assert_flagged_rows_are_explained` is the bar it has to clear if it does —
        an attention weight per step is the local explanation a sequence model owes.
        """
        probs = self.predict_proba(batch.transactions)
        return [
            self.policy.decide(
                t.txn_id, float(p), amount=t.amount, reasons=[f"sequence:{self.arch}"]
            )
            for t, p in zip(batch.transactions, probs, strict=False)
        ]

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        self.fit(batch.transactions)
