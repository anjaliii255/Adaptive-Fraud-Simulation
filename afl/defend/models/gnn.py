"""Temporal GNN over the payment graph — graph attention, with a stated fallback.

**Earn it, or fall back.** Mule networks are a graph problem: fan-in to a collector, then
layering hops before cash-out, and every individual payment in that shape looks ordinary. Message
passing should beat hand-rolled degree features there. But a temporal GNN is also the easiest
place in this repo to produce a number that does not replicate, so ticket 18 wrote the exit
first: this layer is **never** in the headline table unless it beats LightGBM over the hand-rolled
graph features on the same out-of-time split at the same operating point, across seeds — and if
it does not, `fallback()` is what ships and the README says so.

`afl/evaluation/mule_graph.py` holds the gate, `scripts/build_gnn.py` runs it, `docs/gnn.md` is
the write-up.

**The temporal window is the whole design, and it is what keeps this causal.** Time is cut into
strides. A payment landing in stride *b* is scored against the graph of edges in
``[b_start - window, b_start)`` — strictly before the stride it lands in, and no older than
`window_hours`. Edges older than that are dropped, which is the ticket's first criterion and also
the only thing that makes a "temporal" graph different from a graph with timestamps on it: a mule
account that fanned in six months ago is not fanning in now, and a model that keeps the edge says
it is. Nothing in a snapshot is at or after the stride it scores, so no row informs its own score
and no later row can move an earlier one. `tests/test_gnn.py` asserts exactly that.

**The version this replaced got two things wrong**, both of which produced a number that looked
fine — the same two the sequence model was rebuilt for, in graph clothing:

  * *The label was per node.* `_to_graph` marked every beneficiary of a fraud row as a positive
    node, trained node-wise, and then broadcast the node's score back onto every transaction into
    it. So the edge carrying the fraud was used to label the node that predicts that edge, and
    the account's legitimate inbound payments inherited the same score. That is a lookup, not a
    detector. Here the unit is the **transaction**, labelled with its own label, scored from its
    two endpoints' embeddings plus its own row.
  * *There was no graph at scoring time.* `predict_proba` rebuilt the graph from the batch alone,
    so a holdout row's beneficiary arrived with no history — on the one family where the history
    *is* the signal. This carries the last `window_hours` of fitted edges across the fit/score
    boundary, the way the stateful `FeatureBuilder` and the sequence model's tails do.

Requires the `deep` extra (`uv sync --extra deep`): torch and torch-geometric. Without it the
constructor raises. A detector that silently degrades to zeros scores in a metric exactly like a
detector that caught nothing, and in an ensemble it is a number nobody can explain later.
"""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from afl.contract.metrics import Action, DetectorScore
from afl.contract.schema import AttackBatch, Rail, Transaction
from afl.defend import explain
from afl.defend.decision import DecisionPolicy

log = logging.getLogger(__name__)

TORCH_HINT = "TemporalGNNDetector needs the `deep` extra: uv sync --extra deep"

#: Hours of graph a payment is judged against. Seven days: a layering chain hops over days, and a
#: window shorter than the chain cuts the motif in half.
DEFAULT_WINDOW_HOURS = 168.0

#: How often the graph is rebuilt. One day: every payment in a day is scored against the same
#: snapshot, which is what makes the whole thing affordable and is also how a batch scorer
#: actually runs. The cost is that a payment late in a stride is judged against a graph up to
#: `stride` older than one early in it — stated here rather than discovered later.
DEFAULT_STRIDE_HOURS = 24.0

#: A snapshot bigger than this keeps only its most recent edges. A guard against one pathological
#: stride, never a silent cap: the count is on the model card and the run logs it.
DEFAULT_MAX_EDGES = 400_000

# ── the raw per-row columns everything here is derived from ─────────────────────
RAW_TS, RAW_AMOUNT, RAW_HOUR, RAW_RAIL, RAW_SRC, RAW_DST = range(6)
N_RAW = 6

_RAIL_CODE = {Rail.CARD: 0.0, Rail.UPI: 1.0, Rail.A2A: 2.0}

#: One name per node feature, in the order `node_features` emits them. Every one is computed from
#: the snapshot's edges alone — which are all strictly before the stride the snapshot scores — so
#: a node's features cannot read the payment being judged.
#:
#: They are deliberately the hand-rolled degree block in the GNN's own units. The question this
#: model is here to answer is not "graph features versus none": it is whether *message passing*
#: adds anything to degree counters a table already has. Starting the GNN from parity is what
#: makes the answer about propagation rather than about which side was given the counters.
NODE_FEATURES = (
    "log_out_degree",
    "log_in_degree",
    "log_out_volume",
    "log_in_volume",
    "log_uniq_payees",
    "log_uniq_payers",
    "passthrough_ratio",
    "recency_in_window",
)
N_NODE_FEATURES = len(NODE_FEATURES)

#: The edge's own attributes, seen by the attention. `age_in_window` is what stops a seven-day-old
#: hop and a five-minute-old one being the same edge; `is_reverse` marks the backward copy.
EDGE_FEATURES = ("log_amount", "age_in_window", "is_reverse")
N_EDGE_FEATURES = len(EDGE_FEATURES)

#: The row being judged, as the head sees it. Everything else the head knows arrives through the
#: two endpoint embeddings, which is what makes this a graph model rather than a tabular one.
ROW_FEATURES = ("log_amount", "hour_of_day", "rail_is_card", "rail_is_upi")
N_ROW_FEATURES = len(ROW_FEATURES)


def available() -> bool:
    """Whether the `deep` extra is installed. Callable without it, unlike everything else here."""
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401

        return True
    except ImportError:  # pragma: no cover - environment dependent
        return False


def require_deep():
    """(torch, GATConv), or a refusal that names the command that fixes it."""
    try:
        import torch
        from torch_geometric.nn import GATConv
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(f"{TORCH_HINT} ({e})") from e
    return torch, GATConv


def _key(value: str) -> int:
    """A stable integer for an account id, comparable across processes.

    `hash()` is salted per interpreter, so a graph built in one run would not match one built in
    another — invisible in a metric and fatal to reproducing a number.
    """
    return int(zlib.crc32(value.encode()))


def raw_rows(txns: list[Transaction]) -> np.ndarray:
    """(n, N_RAW) float64 — the contract fields the graph and the head are built from.

    float64 rather than float32 because column 0 is epoch seconds: at ~1.7e9 a float32 mantissa
    resolves to about two minutes, which would collapse every edge age inside a stride onto the
    same number and delete the recency the window exists to express.
    """
    out = np.empty((len(txns), N_RAW), dtype=np.float64)
    for i, t in enumerate(txns):
        out[i] = (
            t.ts.timestamp(),
            t.amount,
            float(t.ts.hour),
            _RAIL_CODE.get(t.rail, 2.0),
            float(_key(t.src)),
            float(_key(t.dst)),
        )
    return out


def row_features(raw: np.ndarray) -> np.ndarray:
    """(n, N_ROW_FEATURES) float32 — the payment itself, with no history in it."""
    return np.stack(
        [
            np.log1p(np.maximum(raw[:, RAW_AMOUNT], 0.0)),
            raw[:, RAW_HOUR] / 23.0,
            (raw[:, RAW_RAIL] == _RAIL_CODE[Rail.CARD]).astype(np.float64),
            (raw[:, RAW_RAIL] == _RAIL_CODE[Rail.UPI]).astype(np.float64),
        ],
        axis=-1,
    ).astype(np.float32)


# ── the temporal window ─────────────────────────────────────────────────────────
def stride_of(ts, origin: float, stride_s: float) -> np.ndarray:
    """Which stride each row lands in. The unit a snapshot is built for."""
    return np.floor((np.asarray(ts, dtype=np.float64) - origin) / stride_s).astype(np.int64)


@dataclass(frozen=True)
class Snapshot:
    """The graph one stride of payments is judged against, and nothing else.

    Deliberately thin: it keeps the node table and *indices back into the raw array*, and the
    edge tensors are rebuilt from those on demand. A fit over a year of daily strides materialises
    a few hundred of these, and holding the edge tensors on every one of them costs more memory
    than rebuilding two concatenations costs time.
    """

    #: right-open: every edge is strictly before this, which is at or before every row it scores
    end_ts: float
    window_s: float
    nodes: np.ndarray  # (n_nodes,) int64 account codes, sorted
    x: np.ndarray  # (n_nodes, N_NODE_FEATURES) float32
    edge_rows: np.ndarray  # (n_edges,) int32 into the raw array the graph was built from
    local_src: np.ndarray  # (n_edges,) int32
    local_dst: np.ndarray  # (n_edges,) int32
    dropped_to_cap: int = 0
    dropped_self_loops: int = 0

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_rows.size)

    def local(self, codes) -> np.ndarray:
        """Global account codes -> local node indices. Every code is already a node."""
        return np.searchsorted(self.nodes, np.asarray(codes, dtype=np.int64))

    def edge_index(self) -> np.ndarray:
        """(2, 2*n_edges) — every payment forward, then the same payment backward.

        Money fans *in* to a collector and then hops *out* of it. A single-direction graph can
        only propagate one half of that motif, so the reverse copy is not symmetry for its own
        sake; it is the second half of the shape this model exists to read. `is_reverse` in the
        edge attributes is what keeps the two distinguishable.
        """
        return np.stack(
            [
                np.concatenate([self.local_src, self.local_dst]),
                np.concatenate([self.local_dst, self.local_src]),
            ]
        ).astype(np.int64)

    def edge_attr(self, raw: np.ndarray) -> np.ndarray:
        """(2*n_edges, N_EDGE_FEATURES) — amount, age inside the window, and the direction flag."""
        rows = self.edge_rows
        amount = np.log1p(np.maximum(raw[rows, RAW_AMOUNT], 0.0))
        age = np.clip((self.end_ts - raw[rows, RAW_TS]) / self.window_s, 0.0, 1.0)
        flag = np.zeros_like(age)
        return np.concatenate(
            [
                np.stack([amount, age, flag], axis=-1),
                np.stack([amount, age, flag + 1.0], axis=-1),
            ]
        ).astype(np.float32)

    def in_degree(self) -> np.ndarray:
        """Payments received per node inside the window — the forward direction only."""
        return np.bincount(self.local_dst, minlength=self.n_nodes)

    def degree(self) -> np.ndarray:
        """Payments touching each node inside the window, either direction."""
        return np.bincount(self.local_src, minlength=self.n_nodes) + self.in_degree()


def node_features(
    raw: np.ndarray,
    edge_rows: np.ndarray,
    local_src: np.ndarray,
    local_dst: np.ndarray,
    n_nodes: int,
    end_ts: float,
    window_s: float,
) -> np.ndarray:
    """The degree block, per node, over this snapshot's edges only.

    `bincount` rather than `np.add.at`: the same arithmetic, and far faster on the array sizes a
    seven-day window produces on a real anchor.
    """
    amount = np.log1p(np.maximum(raw[edge_rows, RAW_AMOUNT], 0.0))
    out_cnt = np.bincount(local_src, minlength=n_nodes).astype(np.float64)
    in_cnt = np.bincount(local_dst, minlength=n_nodes).astype(np.float64)
    out_sum = np.bincount(local_src, weights=amount, minlength=n_nodes)
    in_sum = np.bincount(local_dst, weights=amount, minlength=n_nodes)

    def uniq(side: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Distinct counterparties per node — one sort over the (node, other) pairs."""
        if side.size == 0:
            return np.zeros(n_nodes, dtype=np.float64)
        pairs = np.unique(np.stack([side, other], axis=1), axis=0)
        return np.bincount(pairs[:, 0], minlength=n_nodes).astype(np.float64)

    last = np.full(n_nodes, -np.inf)
    if edge_rows.size:
        ts = raw[edge_rows, RAW_TS]
        np.maximum.at(last, local_src, ts)
        np.maximum.at(last, local_dst, ts)
    recency = np.where(np.isfinite(last), np.clip((end_ts - last) / window_s, 0.0, 1.0), 1.0)

    return np.stack(
        [
            np.log1p(out_cnt),
            np.log1p(in_cnt),
            out_sum,
            in_sum,
            np.log1p(uniq(local_src, local_dst)),
            np.log1p(uniq(local_dst, local_src)),
            out_sum / (in_sum + 1.0),
            recency,
        ],
        axis=-1,
    ).astype(np.float32)


def build_snapshot(
    raw: np.ndarray,
    order: np.ndarray,
    ts_sorted: np.ndarray,
    end_ts: float,
    window_s: float,
    extra_nodes: np.ndarray,
    max_edges: int = DEFAULT_MAX_EDGES,
) -> Snapshot:
    """Edges in `[end_ts - window_s, end_ts)`, plus the accounts the stride is about to score.

    `extra_nodes` are the endpoints of the rows this snapshot will score. They join as isolated
    nodes when the window holds nothing for them — a real state that has to be representable
    rather than an error: a beneficiary nobody has paid in a week is exactly the cold account a
    first-hop mule looks like, and dropping those rows would score the model only on the accounts
    it happened to have something to say about.
    """
    lo = int(np.searchsorted(ts_sorted, end_ts - window_s, side="left"))
    hi = int(np.searchsorted(ts_sorted, end_ts, side="left"))
    edges = order[lo:hi]
    dropped_cap = 0
    if edges.size > max_edges:  # keep the most recent; loudly, never silently
        dropped_cap = int(edges.size - max_edges)
        edges = edges[-max_edges:]

    src_code = raw[edges, RAW_SRC].astype(np.int64)
    dst_code = raw[edges, RAW_DST].astype(np.int64)
    # A payment from an account to itself moves no money between two nodes, and `GATConv` strips
    # self-loops from the edge list before adding its own — which would silently break the
    # alignment the attention explanation is read through. Dropped here, and counted.
    real = src_code != dst_code
    dropped_loops = int((~real).sum())
    edges, src_code, dst_code = edges[real], src_code[real], dst_code[real]

    nodes = np.unique(
        np.concatenate([src_code, dst_code, np.asarray(extra_nodes, dtype=np.int64)])
    )
    local_src = np.searchsorted(nodes, src_code).astype(np.int32)
    local_dst = np.searchsorted(nodes, dst_code).astype(np.int32)
    return Snapshot(
        end_ts=float(end_ts),
        window_s=float(window_s),
        nodes=nodes,
        x=node_features(
            raw, edges, local_src, local_dst, int(nodes.size), float(end_ts), float(window_s)
        ),
        edge_rows=edges.astype(np.int32),
        local_src=local_src,
        local_dst=local_dst,
        dropped_to_cap=dropped_cap,
        dropped_self_loops=dropped_loops,
    )


# ── what one fit saw, and what it cost ──────────────────────────────────────────
@dataclass
class GNNTraining:
    """What one fit actually saw, and what it cost. Both belong next to the lift.

    A graph model that wins by 0.02 PR-AUC and costs sixty times the fit is a trade somebody has
    to be able to price, so the seconds are on the card rather than in a terminal somebody closed.
    """

    n_rows: int = 0
    n_fraud: int = 0
    n_scored: int = 0
    n_negatives_sampled: int = 0
    negative_ratio: float = 0.0
    n_snapshots: int = 0
    n_parameters: int = 0
    epochs: int = 0
    final_loss: float = 0.0
    fit_seconds: float = 0.0
    edges_dropped_to_cap: int = 0
    self_loops_dropped: int = 0
    torch_version: str = ""
    device: str = "cpu"
    fitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_fraud": self.n_fraud,
            "base_rate": round(self.n_fraud / self.n_rows, 8) if self.n_rows else 0.0,
            "n_scored_rows": self.n_scored,
            "n_negatives_sampled": self.n_negatives_sampled,
            "negative_ratio": self.negative_ratio,
            "n_snapshots": self.n_snapshots,
            "n_parameters": self.n_parameters,
            "epochs": self.epochs,
            "final_loss": round(self.final_loss, 6),
            "fit_seconds": round(self.fit_seconds, 2),
            "edges_dropped_to_cap": self.edges_dropped_to_cap,
            "self_loops_dropped": self.self_loops_dropped,
            "torch_version": self.torch_version,
            "device": self.device,
            "fitted": self.fitted,
        }


@dataclass
class GraphCoverage:
    """How much graph this anchor actually has inside the window — the model's own precondition.

    Message passing has nothing to pass on an anchor whose accounts appear once: both endpoints
    of every payment are isolated nodes, the two-hop neighbourhood is empty, and a graph attention
    layer degenerates into an MLP over the degree block it was handed. PaySim's sender side is
    close to exactly that. Measured on every fit and reported, so the precondition is a number in
    an artefact rather than a caveat somebody remembers.
    """

    n_snapshots: int = 0
    mean_nodes: float = 0.0
    mean_edges: float = 0.0
    mean_degree: float = 0.0
    share_rows_with_isolated_endpoints: float = 0.0
    share_rows_with_no_graph: float = 0.0
    max_in_degree: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_snapshots": self.n_snapshots,
            "mean_nodes_per_snapshot": self.mean_nodes,
            "mean_edges_per_snapshot": self.mean_edges,
            "mean_degree": self.mean_degree,
            "share_of_rows_whose_endpoints_are_isolated": self.share_rows_with_isolated_endpoints,
            "share_of_rows_with_no_graph_at_all": self.share_rows_with_no_graph,
            "max_in_degree_seen": self.max_in_degree,
        }

    @classmethod
    def measure(
        cls, snaps: dict[int, Snapshot], raw: np.ndarray, strides: np.ndarray
    ) -> GraphCoverage:
        isolated, no_graph, degrees, nodes, edges, in_deg = 0, 0, [], [], [], 0
        for b, snap in snaps.items():
            here = np.flatnonzero(strides == b)
            nodes.append(snap.n_nodes)
            edges.append(snap.n_edges)
            if not snap.n_edges:
                no_graph += int(here.size)
                isolated += int(here.size)
                continue
            deg = snap.degree()
            degrees.append(float(deg.mean()))
            in_deg = max(in_deg, int(snap.in_degree().max()))
            src = snap.local(raw[here, RAW_SRC])
            dst = snap.local(raw[here, RAW_DST])
            isolated += int(((deg[src] == 0) & (deg[dst] == 0)).sum())
        n = max(int(raw.shape[0]), 1)
        return cls(
            n_snapshots=len(snaps),
            mean_nodes=round(float(np.mean(nodes)), 2) if nodes else 0.0,
            mean_edges=round(float(np.mean(edges)), 2) if edges else 0.0,
            mean_degree=round(float(np.mean(degrees)), 4) if degrees else 0.0,
            share_rows_with_isolated_endpoints=round(isolated / n, 6),
            share_rows_with_no_graph=round(no_graph / n, 6),
            max_in_degree=in_deg,
        )


class TemporalGNNDetector:
    """Graph attention over the payment graph inside an explicit temporal window.

    Same seam as every other detector: `fit(rows)`, `score(batch)`, `retrain(batch, evasions)`,
    `training_rows` for the leave-one-attack-out guard to audit.
    """

    def __init__(
        self,
        hidden: int = 32,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.1,
        window_hours: float = DEFAULT_WINDOW_HOURS,
        stride_hours: float = DEFAULT_STRIDE_HOURS,
        epochs: int = 12,
        lr: float = 5e-3,
        negative_ratio: float = 20.0,
        max_edges: int = DEFAULT_MAX_EDGES,
        policy: DecisionPolicy | None = None,
        seed: int = 1337,
    ) -> None:
        if window_hours <= 0 or stride_hours <= 0:
            raise ValueError(
                f"window_hours={window_hours} stride_hours={stride_hours}: a temporal graph needs "
                "a positive window and a positive stride, or it is not temporal"
            )
        if stride_hours > window_hours:
            raise ValueError(
                f"stride_hours={stride_hours} > window_hours={window_hours} would leave payments "
                "scored against a graph that does not reach back to the previous stride — the "
                "window has to cover at least one full step of the clock it advances on"
            )
        if layers < 1:
            raise ValueError(
                f"layers={layers}: a graph model with no message-passing layer is an MLP over the "
                "degree block, which is the baseline this one is supposed to beat"
            )
        # Up front, not at the first forward pass: a config that enables this layer without the
        # extra should fail before it spends an hour generating the pool it cannot score.
        self.torch, self._gat_conv = require_deep()

        self.hidden = hidden
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.window_s = float(window_hours) * 3_600.0
        self.stride_s = float(stride_hours) * 3_600.0
        self.epochs = epochs
        self.lr = lr
        #: Negatives kept per positive among the rows the loss is computed on. The *graph* keeps
        #: every edge either way — this samples what is scored, never what is propagated over, so
        #: it changes the cost of the fit and not the structure any row is judged against.
        self.negative_ratio = negative_ratio
        self.max_edges = max_edges
        self.policy = policy or DecisionPolicy()
        self.seed = seed

        self.model = None
        self.training = GNNTraining()
        self.coverage = GraphCoverage()
        self._node_mu = np.zeros(N_NODE_FEATURES, dtype=np.float32)
        self._node_sd = np.ones(N_NODE_FEATURES, dtype=np.float32)
        self._row_mu = np.zeros(N_ROW_FEATURES, dtype=np.float32)
        self._row_sd = np.ones(N_ROW_FEATURES, dtype=np.float32)
        self._corpus: list[Transaction] = []
        self._replay: list[Transaction] = []
        #: The last `window_s` of raw rows this detector was fitted on. The graph a holdout row
        #: needs: without it the first stride after the split is scored against an empty graph,
        #: on the one family where the graph is the signal.
        self._tail: np.ndarray = np.zeros((0, N_RAW), dtype=np.float64)
        #: The clock strides are counted from. Frozen at fit time and reused at score time, so a
        #: holdout row lands in the stride it would have landed in during the fit.
        self._origin = 0.0
        self._score_seconds = 0.0
        self._scored_rows = 0

    # ── availability, and the exit ──────────────────────────────────────────────
    @staticmethod
    def available() -> bool:
        return available()

    @staticmethod
    def fallback(**kwargs):
        """The stated fallback: LightGBM over the hand-rolled feature table.

        Ticket 18 names it up front rather than after the result, because a fallback chosen once
        the experiment has come back is not a fallback, it is a rationalisation. It is also the
        champion the gate measures against, so "it did not beat the fallback" and "the fallback
        ships" are the same sentence rather than two decisions.
        """
        from afl.defend.models.lgbm import LGBMDetector

        return LGBMDetector(**kwargs)

    # ── layout ──────────────────────────────────────────────────────────────────
    def _prepare(self, txns: list[Transaction], with_history: bool):
        """(raw, graph_raw, offset, snapshots, strides) — everything a forward pass needs.

        `with_history` prepends the tail this detector was fitted on, so a holdout row is judged
        against the graph the training window left behind. At fit time there is no tail: the
        snapshots are built out of the training rows themselves.
        """
        raw = raw_rows(txns)
        if with_history and self._tail.size:
            graph_raw, offset = np.concatenate([self._tail, raw]), int(self._tail.shape[0])
        else:
            graph_raw, offset = raw, 0

        order = np.argsort(graph_raw[:, RAW_TS], kind="stable")
        ts_sorted = graph_raw[order, RAW_TS]
        strides = stride_of(raw[:, RAW_TS], self._origin, self.stride_s)

        snaps: dict[int, Snapshot] = {}
        for b in np.unique(strides):
            here = strides == b
            snaps[int(b)] = build_snapshot(
                graph_raw,
                order,
                ts_sorted,
                end_ts=self._origin + float(b) * self.stride_s,
                window_s=self.window_s,
                extra_nodes=np.concatenate(
                    [raw[here, RAW_SRC].astype(np.int64), raw[here, RAW_DST].astype(np.int64)]
                ),
                max_edges=self.max_edges,
            )
        return raw, graph_raw, offset, snaps, strides

    # ── the network ─────────────────────────────────────────────────────────────
    def _build(self):
        torch = self.torch
        gat_conv = self._gat_conv
        import torch.nn as nn

        hidden, heads, layers, dropout = self.hidden, self.heads, self.layers, self.dropout

        class Net(nn.Module):
            """Graph attention over the snapshot, then a head that scores one edge.

            The head sees both endpoints, their interaction, and the payment itself. Pooling the
            graph instead would answer a question about the *account* rather than about the
            transaction, which is the bug the node-labelled version of this module shipped.
            """

            def __init__(self) -> None:
                super().__init__()
                self.convs = nn.ModuleList()
                dim = N_NODE_FEATURES
                for _ in range(layers):
                    self.convs.append(
                        gat_conv(
                            dim,
                            hidden,
                            heads=heads,
                            concat=True,
                            edge_dim=N_EDGE_FEATURES,
                            dropout=dropout,
                            add_self_loops=True,
                            # an explicit constant rather than the "mean" default: a self-loop is
                            # not a payment, and a snapshot with no edges at all has no mean to
                            # take. Zero says "no payment here", which is what it is.
                            fill_value=0.0,
                        )
                    )
                    dim = hidden * heads
                self.head = nn.Sequential(
                    nn.Linear(dim * 3 + N_ROW_FEATURES, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            def embed(self, x, edge_index, edge_attr, attention: bool = False):
                """Node embeddings, and the last layer's attention when it is asked for."""
                alpha = None
                for i, conv in enumerate(self.convs):
                    if attention and i == len(self.convs) - 1:
                        x, (_, alpha) = conv(
                            x, edge_index, edge_attr, return_attention_weights=True
                        )
                    else:
                        x = conv(x, edge_index, edge_attr)
                    x = torch.relu(x)
                return x, alpha

            def forward(self, h, src, dst, rows):
                a, b = h[src], h[dst]
                return self.head(torch.cat([a, b, a * b, rows], dim=-1)).squeeze(-1)

        return Net()

    def _tensors(self, snap: Snapshot, graph_raw: np.ndarray):
        torch = self.torch
        x = ((snap.x - self._node_mu) / self._node_sd).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(snap.edge_index())),
            torch.from_numpy(np.ascontiguousarray(snap.edge_attr(graph_raw))),
        )

    def _standardise_rows(self, rows: np.ndarray) -> np.ndarray:
        return ((rows - self._row_mu) / self._row_sd).astype(np.float32)

    # ── training ────────────────────────────────────────────────────────────────
    def fit(
        self, txns: list[Transaction], sample_weight: np.ndarray | None = None
    ) -> TemporalGNNDetector:
        """Fit from scratch on `txns`, which become the corpus and the scoring-time graph.

        `sample_weight` is accepted and ignored so this drops into the same `fit(detector, rows)`
        hook the supervised detector uses; the replay buffer is applied by oversampling instead,
        which is what a minibatch loop can act on.
        """
        del sample_weight
        self._corpus = list(txns)
        return self._fit(txns)

    def _fit(self, txns: list[Transaction]) -> TemporalGNNDetector:
        torch = self.torch
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        y = np.array([int(t.is_fraud) for t in txns], dtype=np.int64)
        if len(txns) < 2 or len(set(y.tolist())) < 2:
            raise ValueError(
                f"a single-class training set of {len(txns)} row(s) is not something a graph "
                "model can be fitted on — it would score every row the same and the metric would "
                "read exactly like a detector that caught nothing"
            )

        self._origin = float(min(t.ts.timestamp() for t in txns))
        raw, graph_raw, _, snaps, strides = self._prepare(txns, with_history=False)
        self.coverage = GraphCoverage.measure(snaps, raw, strides)

        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        budget = int(min(len(negatives), max(len(positives), 1) * self.negative_ratio))
        keep = np.concatenate([positives, rng.choice(negatives, size=budget, replace=False)])
        # the rows that once evaded are the expensive examples; duplicated rather than weighted,
        # because a minibatch loop can act on a duplicate and not on a weight column
        heavy = {t.txn_id for t in self._replay}
        if heavy:
            extra = np.array([i for i in keep if txns[i].txn_id in heavy], dtype=np.int64)
            keep = np.concatenate([keep, extra, extra]) if extra.size else keep

        rows_x = row_features(raw)
        self._row_mu = rows_x[keep].mean(axis=0).astype(np.float32)
        self._row_sd = np.sqrt(np.maximum(rows_x[keep].var(axis=0), 1e-8)).astype(np.float32)
        node_x = np.concatenate([s.x for s in snaps.values()]) if snaps else np.zeros((1, 8), "f")
        self._node_mu = node_x.mean(axis=0).astype(np.float32)
        self._node_sd = np.sqrt(np.maximum(node_x.var(axis=0), 1e-8)).astype(np.float32)

        grouped: dict[int, list[int]] = {}
        for i in keep.tolist():
            grouped.setdefault(int(strides[i]), []).append(int(i))
        batches = {b: np.array(v, dtype=np.int64) for b, v in grouped.items()}

        self.model = self._build()
        n_params = sum(p.numel() for p in self.model.parameters())
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        pos = float(max(1, int(y[keep].sum())))
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(max(1.0, (len(keep) - pos) / pos))
        )

        self.model.train()
        loss_value = 0.0
        order = np.array(sorted(batches), dtype=np.int64)
        for _ in range(self.epochs):
            epoch_loss, seen = 0.0, 0
            for b in rng.permutation(order).tolist():
                idx = batches[int(b)]
                snap = snaps[int(b)]
                x, edge_index, edge_attr = self._tensors(snap, graph_raw)
                opt.zero_grad()
                h, _ = self.model.embed(x, edge_index, edge_attr)
                logits = self.model(
                    h,
                    torch.from_numpy(snap.local(raw[idx, RAW_SRC]).astype(np.int64)),
                    torch.from_numpy(snap.local(raw[idx, RAW_DST]).astype(np.int64)),
                    torch.from_numpy(self._standardise_rows(rows_x[idx])),
                )
                loss = loss_fn(logits, torch.from_numpy(y[idx].astype(np.float32)))
                loss.backward()
                opt.step()
                epoch_loss += float(loss) * len(idx)
                seen += len(idx)
            loss_value = epoch_loss / max(seen, 1)
        self.model.eval()

        self._remember(raw)
        self.training = GNNTraining(
            n_rows=len(txns),
            n_fraud=int(y.sum()),
            n_scored=int(keep.size),
            n_negatives_sampled=budget,
            negative_ratio=self.negative_ratio,
            n_snapshots=len(snaps),
            n_parameters=int(n_params),
            epochs=self.epochs,
            final_loss=loss_value,
            fit_seconds=time.perf_counter() - started,
            edges_dropped_to_cap=int(sum(s.dropped_to_cap for s in snaps.values())),
            self_loops_dropped=int(sum(s.dropped_self_loops for s in snaps.values())),
            torch_version=str(torch.__version__),
            device="cpu",
            fitted=True,
        )
        if self.training.edges_dropped_to_cap:
            log.warning(
                "%d edge(s) fell outside the %d-edge snapshot cap and were dropped oldest-first — "
                "raise defend.gnn.max_edges or shorten the window",
                self.training.edges_dropped_to_cap,
                self.max_edges,
            )
        log.info(
            "gnn fitted on %d scored rows (%d positives) over %d snapshots in %.1fs — mean %.0f "
            "edges per snapshot, %.1f%% of rows with isolated endpoints",
            keep.size,
            len(positives),
            len(snaps),
            self.training.fit_seconds,
            self.coverage.mean_edges,
            100 * self.coverage.share_rows_with_isolated_endpoints,
        )
        return self

    def _remember(self, raw: np.ndarray) -> None:
        """Keep the last `window_s` of fitted edges, so scoring has a graph to read."""
        if raw.size == 0:
            self._tail = np.zeros((0, N_RAW), dtype=np.float64)
            return
        latest = float(raw[:, RAW_TS].max())
        self._tail = raw[raw[:, RAW_TS] >= latest - self.window_s]

    @property
    def training_rows(self) -> list[Transaction]:
        """Every row this detector has fitted on, replay buffer included.

        What `leave_one_attack_out.assert_family_held_out` audits. The corpus accumulates across
        rounds and the replay buffer across evasions, so both are places a carved-out family can
        reappear in training long after the split that excluded it.
        """
        return [*self._corpus, *self._replay]

    def retrain(self, batch: AttackBatch, evasions: list[Transaction]) -> None:
        """Add the round to the corpus and refit. The round is added, never substituted."""
        self._replay.extend(evasions)
        known = {t.txn_id for t in self._corpus}
        self._corpus.extend(t for t in batch.transactions if t.txn_id not in known)
        self._fit(self._corpus)

    # ── scoring ─────────────────────────────────────────────────────────────────
    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "TemporalGNNDetector.score before fit() — an unfitted graph model has no score "
                "to give, and returning zeros would read in a metric exactly like a detector "
                "that caught nothing"
            )

    def _run(self, raw, graph_raw, snaps, strides) -> np.ndarray:
        """P(fraud) per row, one snapshot at a time. Every row is scored exactly once."""
        torch = self.torch
        probs = np.zeros(raw.shape[0], dtype=np.float64)
        rows_x = row_features(raw)
        with torch.no_grad():
            for b, snap in snaps.items():
                idx = np.flatnonzero(strides == b)
                x, edge_index, edge_attr = self._tensors(snap, graph_raw)
                h, _ = self.model.embed(x, edge_index, edge_attr)
                logits = self.model(
                    h,
                    torch.from_numpy(snap.local(raw[idx, RAW_SRC]).astype(np.int64)),
                    torch.from_numpy(snap.local(raw[idx, RAW_DST]).astype(np.int64)),
                    torch.from_numpy(self._standardise_rows(rows_x[idx])),
                )
                probs[idx] = torch.sigmoid(logits).numpy()
        return probs

    def _attend(self, raw, graph_raw, offset, snaps, strides, flagged) -> dict[int, dict]:
        """The neighbourhood evidence behind the rows the policy flagged.

        Run over the *same* snapshots the scores came from rather than over a batch rebuilt from
        the flagged rows alone — a graph rebuilt from 0.4% of the traffic is a different graph,
        and an explanation computed on one is not an explanation of a score computed on the other.
        Only strides that actually carry a flagged row get a forward pass, which is what keeps
        reason codes unconditional instead of a mode somebody switches off.
        """
        torch = self.torch
        wanted = set(int(i) for i in flagged)
        out: dict[int, dict] = {}
        with torch.no_grad():
            for b, snap in snaps.items():
                idx = np.array(
                    [i for i in np.flatnonzero(strides == b).tolist() if i in wanted], dtype=int
                )
                if not idx.size:
                    continue
                x, edge_index, edge_attr = self._tensors(snap, graph_raw)
                _, alpha = self.model.embed(x, edge_index, edge_attr, attention=True)
                out.update(self._evidence(snap, graph_raw, offset, idx, raw, alpha))
        return out

    def _evidence(self, snap, graph_raw, offset, idx, raw, alpha) -> dict[int, dict]:
        """What the beneficiary's neighbourhood held, and where the attention landed.

        Attention weights are the local explanation a message-passing model owes: an analyst told
        only "the graph says so" cannot argue with it, and ticket 09's floor is three reason codes
        on every flagged row. Only the *forward* half of the edge list is offered as evidence — a
        reverse edge is an implementation detail of propagation, not a payment somebody made.

        `GATConv` appends its self-loops after the edges it was handed, so the first `n_edges`
        rows of `alpha` line up with the forward edges of the snapshot. The snapshot drops
        self-payments for exactly this reason; without that, `GATConv`'s own `remove_self_loops`
        would shift the alignment and the explanation would name the wrong payment.
        """
        weights = alpha.mean(dim=1).numpy() if alpha is not None else np.zeros(0)
        n_fwd = snap.n_edges
        in_degree = snap.in_degree()
        dst_local = snap.local(raw[idx, RAW_DST])

        # the inbound edges of a node, found by two bisects rather than by a scan of the whole
        # edge list per row. A stride where the policy flags many rows would otherwise be
        # rows x edges, which on a real holdout is a number with ten digits in it
        by_target = np.argsort(snap.local_dst, kind="stable")
        targets = snap.local_dst[by_target]
        first = np.searchsorted(targets, dst_local, side="left")
        last = np.searchsorted(targets, dst_local, side="right")

        out: dict[int, dict] = {}
        for pos, row in enumerate(idx.tolist()):
            node = int(dst_local[pos])
            incoming = by_target[first[pos] : last[pos]]
            best = None
            if incoming.size and weights.size >= n_fwd:
                top = int(incoming[int(np.argmax(weights[incoming]))])
                edge_row = int(snap.edge_rows[top])
                best = {
                    "weight": float(weights[top]),
                    "amount": float(graph_raw[edge_row, RAW_AMOUNT]),
                    "age_s": float(snap.end_ts - graph_raw[edge_row, RAW_TS]),
                    "from_the_training_window": bool(edge_row < offset),
                }
            out[row] = {
                "in_degree": int(in_degree[node]),
                "n_payers": int(np.unique(snap.local_src[incoming]).size) if incoming.size else 0,
                "window_hours": snap.window_s / 3_600.0,
                "top_edge": best,
            }
        return out

    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        self._require_fitted()
        if not txns:
            return np.zeros(0, dtype=float)
        started = time.perf_counter()
        raw, graph_raw, _, snaps, strides = self._prepare(txns, with_history=True)
        probs = self._run(raw, graph_raw, snaps, strides)
        self._score_seconds = time.perf_counter() - started
        self._scored_rows = len(txns)
        return probs

    def reason_codes(self, txn: Transaction, evidence: dict | None) -> list[str]:
        """Why this payment's neighbourhood looked wrong, in an analyst's units.

        A per-row local explanation, so it carries no `GLOBAL_PREFIX`. Always at least
        `explain.MIN_REASONS` statements: what the model is, what the beneficiary's window
        actually held, and the neighbouring payment the attention landed on.

        A beneficiary with no prior edges in the window says so rather than inventing a
        neighbourhood to point at. That is not a corner case on every anchor — it is what the cold
        end of PaySim looks like from here, and a graph model that quietly narrated a ring over an
        isolated node would be the exact failure this module's docstring is about.
        """
        hours = evidence["window_hours"] if evidence else self.window_s / 3_600.0
        codes = [
            f"gnn:gat over the beneficiary's {hours:.0f}h payment graph "
            f"({self.layers} hops, {self.heads} attention heads)"
        ]
        if evidence and evidence["in_degree"]:
            codes.append(
                f"↑ {evidence['n_payers']} distinct account(s) paid this beneficiary in the "
                f"window, over {evidence['in_degree']} payment(s), before this {txn.amount:,.0f}"
            )
            top = evidence["top_edge"]
            codes.append(
                f"↑ {top['weight']:.2f} of the attention on the payment of "
                f"{top['amount']:,.0f} into it {_duration(top['age_s'])} earlier"
                if top
                else "· the attention spread evenly across the inbound edges"
            )
        else:
            codes.append(
                f"· nobody has paid this beneficiary in the last {hours:.0f}h, so it enters the "
                f"graph as an isolated node and this {txn.amount:,.0f} is its first edge"
            )
            codes.append(
                "· an isolated node carries no neighbourhood, so this score is the model's read "
                "of a single payment, which is what the supervised detector is for"
            )
        if len(codes) < explain.MIN_REASONS:  # pragma: no cover - both branches emit three
            raise AssertionError(
                f"a flagged row left this model with {len(codes)} reason code(s) against a floor "
                f"of {explain.MIN_REASONS} — see explain.assert_flagged_rows_are_explained"
            )
        return codes

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        """Score, decide, and explain whatever the decision flagged.

        Two passes over the actions, priced the way the supervised detector prices SHAP: the
        first finds which rows carry an action, the second re-runs only the strides those rows
        landed in, with the attention weights kept.
        """
        txns = batch.transactions
        if not txns:
            return []
        self._require_fitted()
        started = time.perf_counter()
        raw, graph_raw, offset, snaps, strides = self._prepare(txns, with_history=True)
        probs = self._run(raw, graph_raw, snaps, strides)
        actions = [
            self.policy.act(float(p), amount=t.amount) for t, p in zip(txns, probs, strict=False)
        ]
        flagged = [i for i, a in enumerate(actions) if a is not Action.ALLOW]

        evidence = (
            self._attend(raw, graph_raw, offset, snaps, strides, flagged) if flagged else {}
        )
        self._score_seconds = time.perf_counter() - started
        self._scored_rows = len(txns)

        return [
            self.policy.decide(
                t.txn_id,
                float(p),
                amount=t.amount,
                reasons=self.reason_codes(t, evidence.get(i))
                if i in evidence
                else ["gnn:gat over the beneficiary's payment graph"],
            )
            for i, (t, p) in enumerate(zip(txns, probs, strict=False))
        ]

    # ── introspection ───────────────────────────────────────────────────────────
    def model_card(self) -> dict[str, Any]:
        """Everything a reported number needs — including what it cost to produce."""
        return {
            "detector": type(self).__name__,
            "arch": "gat",
            "fallback": "LGBMDetector over the hand-rolled feature table",
            "hyperparameters": {
                "hidden": self.hidden,
                "heads": self.heads,
                "layers": self.layers,
                "dropout": self.dropout,
                "window_hours": self.window_s / 3_600.0,
                "stride_hours": self.stride_s / 3_600.0,
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "negative_ratio": self.negative_ratio,
                "max_edges_per_snapshot": self.max_edges,
            },
            "seed": self.seed,
            "training": self.training.to_dict(),
            "graph_coverage": self.coverage.to_dict(),
            "compute": self.compute_cost(),
            "node_features": list(NODE_FEATURES),
            "edge_features": list(EDGE_FEATURES),
            "row_features": list(ROW_FEATURES),
            "decision": self.policy.to_dict(),
        }

    def compute_cost(self) -> dict[str, Any]:
        """Fit and score seconds, and the rate — the denominator of every lift in the table."""
        return {
            "fit_seconds": round(self.training.fit_seconds, 2),
            "score_seconds": round(self._score_seconds, 2),
            "scored_rows": self._scored_rows,
            "rows_per_second": round(self._scored_rows / self._score_seconds, 1)
            if self._score_seconds > 0
            else None,
            "n_parameters": self.training.n_parameters,
            "device": self.training.device,
            "torch": self.training.torch_version,
        }


def _duration(seconds: float) -> str:
    """Seconds into something an analyst reads, matching `afl.defend.explain`'s phrasing."""
    seconds = max(float(seconds), 0.0)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5_400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172_800:
        return f"{seconds / 3_600:.0f}h"
    return f"{seconds / 86_400:.0f}d"


__all__ = [
    "DEFAULT_MAX_EDGES",
    "DEFAULT_STRIDE_HOURS",
    "DEFAULT_WINDOW_HOURS",
    "EDGE_FEATURES",
    "NODE_FEATURES",
    "ROW_FEATURES",
    "TORCH_HINT",
    "GNNTraining",
    "GraphCoverage",
    "Snapshot",
    "TemporalGNNDetector",
    "available",
    "build_snapshot",
    "node_features",
    "raw_rows",
    "require_deep",
    "row_features",
    "stride_of",
]
