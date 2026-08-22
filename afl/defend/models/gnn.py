"""Temporal GNN over the transaction graph — experiment, with a stated fallback.

Honest position: mule networks are a graph problem, and message passing should beat hand-rolled
degree features. But a temporal GNN is also the easiest place in this repo to produce a number
that does not replicate. So:

  * it is **never** in the headline table unless it beats LGBM on the same out-of-time split;
  * `fallback()` returns the hand-rolled graph-feature detector, which is what ships if it does
    not — and the README says so.

Requires the `deep` extra (torch + torch-geometric).
"""

from __future__ import annotations

import numpy as np

from afl.contract.metrics import DetectorScore
from afl.contract.schema import AttackBatch, Transaction
from afl.defend.decision import DecisionPolicy


class TemporalGNNDetector:
    """Temporal graph attention over accounts and beneficiaries.

    An experiment with a stated fallback, not a mandate.
    """

    def __init__(
        self,
        hidden: int = 32,
        n_layers: int = 2,
        epochs: int = 30,
        lr: float = 1e-2,
        window_days: float = 7.0,
        policy: DecisionPolicy | None = None,
        seed: int = 1337,
    ) -> None:
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.window_days = window_days  # edges older than this are dropped: "temporal"
        self.policy = policy or DecisionPolicy()
        self.seed = seed
        self.model = None
        self._node_index: dict[str, int] = {}

    # ── availability ────────────────────────────────────────────────────────────
    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def fallback(**kwargs):
        """The stated fallback: LGBM over hand-rolled graph features."""
        from afl.defend.models.lgbm import LGBMDetector

        return LGBMDetector(**kwargs)

    # ── graph construction ──────────────────────────────────────────────────────
    def _to_graph(self, txns: list[Transaction]):
        import torch
        from torch_geometric.data import Data

        nodes = sorted({t.src for t in txns} | {t.dst for t in txns})
        self._node_index = {n: i for i, n in enumerate(nodes)}
        latest = max(t.ts for t in txns)
        keep = [t for t in txns if (latest - t.ts).total_seconds() <= self.window_days * 86_400]

        edge_index = torch.tensor(
            [[self._node_index[t.src] for t in keep], [self._node_index[t.dst] for t in keep]],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [[np.log1p(t.amount), (latest - t.ts).total_seconds() / 86_400.0] for t in keep],
            dtype=torch.float,
        )
        # node features: degree in/out and volume, so the GNN starts from parity with the
        # hand-rolled baseline rather than from nothing
        feats = np.zeros((len(nodes), 4), dtype="float32")
        for t in keep:
            i, j = self._node_index[t.src], self._node_index[t.dst]
            feats[i, 0] += 1
            feats[i, 1] += np.log1p(t.amount)
            feats[j, 2] += 1
            feats[j, 3] += np.log1p(t.amount)
        y = np.zeros(len(nodes), dtype="int64")
        for t in keep:
            if t.is_fraud:
                y[self._node_index[t.dst]] = 1  # beneficiary is the node under suspicion
        return Data(
            x=torch.tensor(feats),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y),
        )

    def fit(self, txns: list[Transaction]) -> TemporalGNNDetector:
        if not self.available():  # pragma: no cover - environment dependent
            raise ImportError("TemporalGNNDetector needs the `deep` extra: uv sync --extra deep")
        import torch
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv

        torch.manual_seed(self.seed)
        data = self._to_graph(txns)

        class Net(nn.Module):
            def __init__(self, n_in, hidden, n_layers):
                super().__init__()
                dims = [n_in] + [hidden] * n_layers
                self.convs = nn.ModuleList(
                    [SAGEConv(dims[i], dims[i + 1]) for i in range(n_layers)]
                )
                self.head = nn.Linear(hidden, 1)

            def forward(self, x, edge_index):
                for conv in self.convs:
                    x = torch.relu(conv(x, edge_index))
                return self.head(x).squeeze(-1)

        self.model = Net(data.x.shape[1], self.hidden, self.n_layers)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        pos = max(1, int(data.y.sum()))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(data.y) - pos) / pos))
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(data.x, data.edge_index), data.y.float())
            loss.backward()
            opt.step()
        self._data = data
        return self

    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(txns), dtype=float)
        import torch

        data = self._to_graph(txns)
        self.model.eval()
        with torch.no_grad():
            node_p = torch.sigmoid(self.model(data.x, data.edge_index)).numpy()
        return np.array([node_p[self._node_index.get(t.dst, 0)] for t in txns])

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        probs = self.predict_proba(batch.transactions)
        return [
            self.policy.decide(t.txn_id, float(p), amount=t.amount, reasons=["gnn:beneficiary"])
            for t, p in zip(batch.transactions, probs, strict=False)
        ]

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        self.fit(batch.transactions)
