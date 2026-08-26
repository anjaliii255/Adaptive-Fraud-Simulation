"""Sequence model — GRU / small transformer over per-entity transaction history.

**Earn it.** This layer is additive. It enters a reported table only if it beats the LightGBM
baseline on the same out-of-time split at the same operating point, and the comparison is
published whichever way it falls — `afl/evaluation/drift_arc.py` holds the gate,
`scripts/build_sequence.py` runs it, `docs/sequence.md` is the write-up.

The axis that decides it is `ramp`, the drift engine's sudden-vs-gradual knob. A takeover that
switches hard is an *event*: the amount jumps, the device changes, the beneficiary is new, and a
per-row feature table sees all three on the row itself. Gradual drift has no event to anchor on —
every individual row is unremarkable and only the trajectory is wrong. That is the case per-row
features run out on, so that is the case this model has to win, and `drift_arc` reports the two
ends of the axis separately rather than averaging them into one number that hides which one paid.

**Two things the first version of this module got wrong**, both of which produced a number that
looked fine:

  * *The label was per entity.* `sequence_tensor` took one window per account and labelled it
    ``any(t.is_fraud for t in window)`` — so the window containing the fraud row was used to
    predict that the window contains a fraud row, and the score was then broadcast back onto the
    account's legitimate baseline rows. That is not a detector; it is a lookup, and it puts a
    high score on the pre-takeover rows that an investigator would call clean.
  * *There was no history at scoring time.* A window built only from the rows in the batch means
    a holdout row whose baseline sits in the training window has no baseline to drift away from,
    which is precisely the arc this model exists for. `FeatureBuilder(stateful=True)` carries
    that history for the supervised detector; this carries the same thing, capped at `max_len`
    steps per entity, so the two models are asked the same question.

So: **one window per transaction, ending at that transaction, labelled with that transaction's
own label.** Every step in it is strictly at or before the row being scored. The model scores
per row through the same `score(batch)` seam as everything else, which is what lets it be read
against LightGBM at one operating point instead of at two.

Requires the `deep` extra (`uv sync --extra deep`). Without torch the constructor raises: a
detector that silently degrades to zeros scores exactly like one that caught nothing, and in an
ensemble it is a number nobody can explain later.
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

TORCH_HINT = "SequenceDetector needs the `deep` extra: uv sync --extra deep"

ARCHITECTURES = ("gru", "transformer")

#: Steps of per-entity history a row is judged against, including the row itself.
DEFAULT_MAX_LEN = 32

#: Rows scored per forward pass. Bounds the (rows x max_len x max_len) beneficiary-novelty
#: comparison, which is the widest array this module builds; the *windows* are always laid out
#: over the whole call, so chunking here cannot change what any row sees.
SCORE_CHUNK = 8_192

#: "this entity has no device on this anchor" — a 0 would read as "the device did not change".
NEVER = -1.0

# ── the raw per-row columns the step features are derived from ──────────────────
RAW_TS, RAW_AMOUNT, RAW_HOUR, RAW_RAIL, RAW_DEVICE, RAW_DST = range(6)
N_RAW = 6

_RAIL_CODE = {Rail.CARD: 0.0, Rail.UPI: 1.0, Rail.A2A: 2.0}

#: One name per step feature, in the order `step_tensor` emits them. Every one of them is a
#: function of the step's own row and of steps strictly before it in the same window.
STEP_FEATURES = (
    "log_amount",
    "log_gap_since_previous",
    "amount_vs_window_running_mean",
    "hour_of_day",
    "rail_is_card",
    "rail_is_upi",
    "device_changed",
    "beneficiary_new_in_window",
    "step_present",
)
N_STEP_FEATURES = len(STEP_FEATURES)


def available() -> bool:
    """Whether the `deep` extra is installed. Callable without it, unlike everything else here."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:  # pragma: no cover - environment dependent
        return False


def require_torch():
    """torch, or a refusal that names the command that fixes it."""
    try:
        import torch

        return torch
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(TORCH_HINT) from e


# ── windows ─────────────────────────────────────────────────────────────────────
def _key(value: str | None) -> float:
    """A stable integer for a beneficiary or device id, comparable across processes.

    `hash()` is salted per interpreter, so a window built in one run would not match one built in
    another — invisible in a metric and fatal to reproducing a number.
    """
    return NEVER if value is None else float(zlib.crc32(value.encode()))


def raw_rows(txns: list[Transaction]) -> np.ndarray:
    """(n, N_RAW) float64 — the contract fields the step features are computed from.

    float64 rather than float32 because column 0 is epoch seconds: at ~1.7e9 a float32 mantissa
    resolves to about two minutes, which would turn every inter-transaction gap under that into
    zero and delete the pacing signal this model is supposed to read.
    """
    out = np.empty((len(txns), N_RAW), dtype=np.float64)
    for i, t in enumerate(txns):
        out[i] = (
            t.ts.timestamp(),
            t.amount,
            float(t.ts.hour),
            _RAIL_CODE.get(t.rail, 2.0),
            _key(t.device_id),
            _key(t.dst),
        )
    return out


def window_index(entity: np.ndarray, ts: np.ndarray, max_len: int) -> np.ndarray:
    """(n, max_len) int32 of row indices — each row's own entity history, ending at that row.

    Right-aligned, so the transaction being judged is always the last column and padding is on
    the left; `-1` marks a padded step. Ties on `ts` are broken by position, so two payments in
    the same second keep the order they were handed over in rather than an arbitrary one.
    """
    n = int(entity.size)
    if n == 0:
        return np.zeros((0, max_len), dtype=np.int32)
    order = np.lexsort((np.arange(n), ts, entity))
    pos = np.arange(n)
    ent = entity[order]
    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ent[1:] != ent[:-1]
    # position of the first row of each entity, forward-filled: rank-within-entity in two lines
    rank = pos - np.maximum.accumulate(np.where(starts, pos, 0))

    offsets = np.arange(max_len - 1, -1, -1)
    taken = np.clip(pos[:, None] - offsets[None, :], 0, None)
    idx = np.where(offsets[None, :] <= rank[:, None], order[taken], -1).astype(np.int32)

    out = np.empty((n, max_len), dtype=np.int32)
    out[order] = idx  # back into the order the caller handed the rows over in
    return out


def step_tensor(raw: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(X[n, max_len, N_STEP_FEATURES] float32, mask[n, max_len] bool) for one block of windows.

    Every feature is computed from the step's own row and from steps *earlier in the same
    window*. Nothing reads forward, and nothing reads a row belonging to another entity, so a
    window is causal by construction rather than by review.
    """
    mask = idx >= 0
    m = mask.astype(np.float64)
    safe = np.where(mask, idx, 0)

    ts = raw[safe, RAW_TS] * m
    log_amount = np.log1p(np.maximum(raw[safe, RAW_AMOUNT], 0.0)) * m

    gap = np.zeros_like(ts)
    gap[:, 1:] = np.maximum(ts[:, 1:] - ts[:, :-1], 0.0)
    gap[:, 1:] *= mask[:, 1:] * mask[:, :-1]

    # the row against the account's own recent normal — the whole reason this is a sequence
    # rather than a row. `log_amount` is already a log, so the difference is a log ratio.
    prior_sum = np.cumsum(log_amount, axis=1) - log_amount
    prior_n = np.cumsum(m, axis=1) - m
    deviation = np.where(prior_n > 0, log_amount - prior_sum / np.maximum(prior_n, 1.0), 0.0)

    rail = raw[safe, RAW_RAIL]
    device = raw[safe, RAW_DEVICE]
    changed = np.zeros_like(ts)
    changed[:, 1:] = (device[:, 1:] != device[:, :-1]).astype(np.float64)
    changed[:, 1:] *= mask[:, 1:] * mask[:, :-1]
    # an anchor with no device column gets NEVER, not 0.0: "there is nothing to change" and
    # "it did not change" are different statements and only one of them is evidence
    changed = np.where(device < 0, NEVER, changed)

    dst = raw[safe, RAW_DST]
    pairwise = (dst[:, :, None] == dst[:, None, :]) & mask[:, None, :] & mask[:, :, None]
    earlier = np.tril(np.ones(idx.shape[1:2] * 2, dtype=bool), -1)
    fresh = (~(pairwise & earlier[None, :, :]).any(axis=2)) & mask

    X = np.stack(
        [
            log_amount,
            np.log1p(gap),
            np.clip(deviation, -8.0, 8.0),
            raw[safe, RAW_HOUR] / 23.0 * m,
            (rail == _RAIL_CODE[Rail.CARD]).astype(np.float64) * m,
            (rail == _RAIL_CODE[Rail.UPI]).astype(np.float64) * m,
            changed,
            fresh.astype(np.float64),
            m,
        ],
        axis=-1,
    )
    return X.astype(np.float32), mask


# ── what one fit saw, and what it cost ──────────────────────────────────────────
@dataclass
class SequenceTraining:
    """What one fit actually saw, and what it cost. Both belong next to the lift.

    A sequence model that wins by 0.02 PR-AUC and costs sixty times the fit is a trade somebody
    has to be able to price, so the seconds are on the card rather than in a terminal somebody
    closed.
    """

    n_rows: int = 0
    n_fraud: int = 0
    n_windows: int = 0
    n_negatives_sampled: int = 0
    negative_ratio: float = 0.0
    n_parameters: int = 0
    epochs: int = 0
    final_loss: float = 0.0
    fit_seconds: float = 0.0
    torch_version: str = ""
    device: str = "cpu"
    fitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_fraud": self.n_fraud,
            "base_rate": round(self.n_fraud / self.n_rows, 8) if self.n_rows else 0.0,
            "n_windows": self.n_windows,
            "n_negatives_sampled": self.n_negatives_sampled,
            "negative_ratio": self.negative_ratio,
            "n_parameters": self.n_parameters,
            "epochs": self.epochs,
            "final_loss": round(self.final_loss, 6),
            "fit_seconds": round(self.fit_seconds, 2),
            "torch_version": self.torch_version,
            "device": self.device,
            "fitted": self.fitted,
        }


@dataclass
class HistoryCoverage:
    """How much per-entity history this anchor actually has — the model's own precondition.

    A GRU over per-entity history has nothing to read on an anchor whose entities appear once.
    PaySim is exactly that: `nameOrig` is effectively unique per row, so every real window there
    is one step long while every injected drift episode carries a full arc. Window length then
    separates injected rows from real ones by itself, and the model's score inherits it. This is
    measured on every fit and reported, so the precondition is a number in an artefact rather
    than a caveat somebody remembers.
    """

    n_windows: int = 0
    n_entities: int = 0
    mean_length: float = 0.0
    median_length: float = 0.0
    max_length: int = 0
    share_length_one: float = 0.0
    share_at_cap: float = 0.0

    @classmethod
    def measure(cls, entity: np.ndarray, lengths: np.ndarray, max_len: int) -> HistoryCoverage:
        if lengths.size == 0:
            return cls()
        return cls(
            n_windows=int(lengths.size),
            n_entities=int(np.unique(entity).size),
            mean_length=round(float(lengths.mean()), 4),
            median_length=float(np.median(lengths)),
            max_length=int(lengths.max()),
            share_length_one=round(float((lengths == 1).mean()), 6),
            share_at_cap=round(float((lengths >= max_len).mean()), 6),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "n_entities": self.n_entities,
            "mean_window_length": self.mean_length,
            "median_window_length": self.median_length,
            "max_window_length": self.max_length,
            "share_of_rows_with_no_history": self.share_length_one,
            "share_of_rows_at_the_cap": self.share_at_cap,
        }


class SequenceDetector:
    """GRU or small transformer over per-entity history, scored one row at a time.

    Same seam as every other detector: `fit(rows)`, `score(batch)`, `retrain(batch, evasions)`,
    `training_rows` for the leave-one-attack-out guard to audit.
    """

    def __init__(
        self,
        arch: str = "gru",
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
        max_len: int = DEFAULT_MAX_LEN,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
        negative_ratio: float = 20.0,
        entity: str = "src",
        policy: DecisionPolicy | None = None,
        seed: int = 1337,
    ) -> None:
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")
        if entity not in ("src", "dst"):
            raise ValueError(f"unknown entity {entity!r}; expected 'src' or 'dst'")
        if max_len < 2:
            raise ValueError(f"max_len={max_len} leaves no history to read; a window needs >= 2")
        # Up front, not at the first forward pass: a config that enables this layer without the
        # extra should fail before it spends an hour generating the pool it cannot score.
        self.torch = require_torch()

        self.arch = arch
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        #: Negatives kept per positive when building the training set. At a 0.13% base rate a
        #: full pass is 99.87% padding through a recurrent net; the ranking metrics are computed
        #: on the untouched holdout, so this changes the cost of the fit and not the operating
        #: point. It is on the model card because a training-set composition nobody wrote down
        #: is a number nobody can reproduce.
        self.negative_ratio = negative_ratio
        self.entity = entity
        self.policy = policy or DecisionPolicy()
        self.seed = seed

        self.model = None
        self.training = SequenceTraining()
        self.coverage = HistoryCoverage()
        self._mu = np.zeros(N_STEP_FEATURES, dtype=np.float32)
        self._sd = np.ones(N_STEP_FEATURES, dtype=np.float32)
        self._corpus: list[Transaction] = []
        self._replay: list[Transaction] = []
        #: entity id -> the last `max_len` raw rows it was fitted on. The stateful feature
        #: builder's history, in this model's own units: without it a holdout row whose baseline
        #: sits in the training window has no baseline to drift away from.
        self._tails: dict[str, np.ndarray] = {}
        self._score_seconds = 0.0
        self._scored_rows = 0

    # ── availability ────────────────────────────────────────────────────────────
    @staticmethod
    def available() -> bool:
        return available()

    # ── windows ─────────────────────────────────────────────────────────────────
    def _entity_of(self, t: Transaction) -> str:
        return t.src if self.entity == "src" else t.dst

    def _layout(
        self, txns: list[Transaction], with_history: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(raw, idx, lengths) — the windows for these rows, over their own entities' history.

        `with_history=True` prepends the tail this detector was fitted on, so a holdout row is
        judged against the account's training-window baseline. At fit time there is no tail:
        the windows are built out of the training rows themselves.
        """
        rows_raw = raw_rows(txns)
        keys = [self._entity_of(t) for t in txns]

        if with_history and self._tails:
            wanted = sorted({k for k in keys if k in self._tails})
            if wanted:
                tail = np.concatenate([self._tails[k] for k in wanted])
                tail_keys = [k for k in wanted for _ in range(len(self._tails[k]))]
                raw = np.concatenate([tail, rows_raw])
                keys = tail_keys + keys
                offset = len(tail_keys)
            else:
                raw, offset = rows_raw, 0
        else:
            raw, offset = rows_raw, 0

        codes = np.array([zlib.crc32(k.encode()) for k in keys], dtype=np.int64)
        idx = window_index(codes, raw[:, RAW_TS], self.max_len)[offset:]
        return raw, idx, (idx >= 0).sum(axis=1)

    def _remember(self, txns: list[Transaction]) -> None:
        """Keep the last `max_len` rows per entity, so scoring has a baseline to read."""
        by_entity: dict[str, list[Transaction]] = {}
        for t in sorted(txns, key=lambda t: t.ts):
            by_entity.setdefault(self._entity_of(t), []).append(t)
        self._tails = {k: raw_rows(rows[-self.max_len :]) for k, rows in by_entity.items()}

    # ── the network ─────────────────────────────────────────────────────────────
    def _build(self):
        torch = self.torch
        import torch.nn as nn

        arch, hidden, layers, dropout = self.arch, self.hidden, self.layers, self.dropout

        class Net(nn.Module):
            """Encoder, attention pooling, and a head that always sees the row being judged.

            The attention weights are not decoration: they are the local explanation a sequence
            model owes, and `SequenceDetector.score` turns the two heaviest steps into reason
            codes. Pooling alone would let the model answer about the window rather than about
            the transaction, so the final step's state is concatenated onto the summary.
            """

            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(N_STEP_FEATURES, hidden)
                if arch == "gru":
                    self.enc = nn.GRU(
                        hidden,
                        hidden,
                        num_layers=layers,
                        batch_first=True,
                        dropout=dropout if layers > 1 else 0.0,
                    )
                else:
                    layer = nn.TransformerEncoderLayer(
                        hidden,
                        nhead=4,
                        dim_feedforward=hidden * 2,
                        dropout=dropout,
                        batch_first=True,
                    )
                    self.enc = nn.TransformerEncoder(layer, num_layers=layers)
                self.attn = nn.Linear(hidden, 1)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(hidden * 2, 1)

            def forward(self, x, mask):
                h = torch.tanh(self.proj(x))
                h = self.enc(h)[0] if arch == "gru" else self.enc(h, src_key_padding_mask=~mask)
                logits = self.attn(h).squeeze(-1).masked_fill(~mask, -1e9)
                weights = torch.softmax(logits, dim=1)
                pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
                out = self.head(self.drop(torch.cat([pooled, h[:, -1, :]], dim=-1)))
                return out.squeeze(-1), weights

        return Net()

    # ── training ────────────────────────────────────────────────────────────────
    def fit(
        self, txns: list[Transaction], sample_weight: np.ndarray | None = None
    ) -> SequenceDetector:
        """Fit from scratch on `txns`, which become the corpus and the scoring-time history.

        `sample_weight` is accepted and ignored so this drops into the same `fit(detector, rows)`
        hook the supervised detector uses; the replay buffer is applied by oversampling instead,
        which is what a minibatch loop can act on.
        """
        del sample_weight
        self._corpus = list(txns)
        return self._fit(txns)

    def _fit(self, txns: list[Transaction]) -> SequenceDetector:
        torch = self.torch
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        y = np.array([int(t.is_fraud) for t in txns], dtype=np.int64)
        if len(txns) < 2 or len(set(y.tolist())) < 2:
            raise ValueError(
                f"a single-class training set of {len(txns)} row(s) is not something a sequence "
                "model can be fitted on — it would score every row the same and the metric would "
                "read exactly like a detector that caught nothing"
            )

        raw, idx, lengths = self._layout(txns, with_history=False)
        entity = np.array([zlib.crc32(self._entity_of(t).encode()) for t in txns], dtype=np.int64)
        self.coverage = HistoryCoverage.measure(entity, lengths, self.max_len)

        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        budget = int(min(len(negatives), max(len(positives), 1) * self.negative_ratio))
        keep = np.concatenate([positives, rng.choice(negatives, size=budget, replace=False)])
        # the rows that once evaded are the expensive examples; they are duplicated rather than
        # weighted because a minibatch loop can act on a duplicate and not on a weight column
        heavy = {t.txn_id for t in self._replay}
        if heavy:
            extra = np.array([i for i in keep if txns[i].txn_id in heavy], dtype=np.int64)
            keep = np.concatenate([keep, extra, extra]) if extra.size else keep
        keep = rng.permutation(keep)

        X, mask = step_tensor(raw, idx[keep])
        present = mask[..., None]
        total = np.maximum(present.sum(axis=(0, 1)), 1.0)
        self._mu = (X * present).sum(axis=(0, 1)) / total
        var = ((X - self._mu) ** 2 * present).sum(axis=(0, 1)) / total
        self._sd = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

        xb = torch.from_numpy(self._standardise(X, mask))
        mb = torch.from_numpy(mask)
        yb = torch.from_numpy(y[keep].astype(np.float32))

        self.model = self._build()
        n_params = sum(p.numel() for p in self.model.parameters())
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        pos = float(max(1, int(yb.sum())))
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(max(1.0, (len(yb) - pos) / pos))
        )

        self.model.train()
        loss_value = 0.0
        for _ in range(self.epochs):
            order = torch.randperm(len(yb))
            epoch_loss, seen = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                take = order[start : start + self.batch_size]
                opt.zero_grad()
                logits, _ = self.model(xb[take], mb[take])
                loss = loss_fn(logits, yb[take])
                loss.backward()
                opt.step()
                epoch_loss += float(loss) * len(take)
                seen += len(take)
            loss_value = epoch_loss / max(seen, 1)
        self.model.eval()

        self._remember(txns)
        self.training = SequenceTraining(
            n_rows=len(txns),
            n_fraud=int(y.sum()),
            n_windows=int(keep.size),
            n_negatives_sampled=budget,
            negative_ratio=self.negative_ratio,
            n_parameters=int(n_params),
            epochs=self.epochs,
            final_loss=loss_value,
            fit_seconds=time.perf_counter() - started,
            torch_version=str(torch.__version__),
            device="cpu",
            fitted=True,
        )
        log.info(
            "sequence(%s) fitted on %d windows (%d positives) in %.1fs — mean window %.1f steps",
            self.arch,
            keep.size,
            len(positives),
            self.training.fit_seconds,
            self.coverage.mean_length,
        )
        return self

    def _standardise(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return ((X - self._mu) / self._sd * mask[..., None]).astype(np.float32)

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
                "SequenceDetector.score before fit() — an unfitted sequence model has no score "
                "to give, and returning zeros would read in a metric exactly like a detector "
                "that caught nothing"
            )

    def _forward(self, txns: list[Transaction], attention: bool = False):
        """(probabilities, attention | None, idx, raw) over `txns`.

        Windows are laid out over the whole call before any chunking, so how many rows a forward
        pass takes cannot change what a row is judged against — a test asserts that. `attention`
        is off by default because keeping the per-step weights for a 400k-row holdout is 50MB
        nobody asked for; `score` re-runs the flagged rows with it on, which is under 1% of them.
        """
        torch = self.torch
        raw, idx, _ = self._layout(txns, with_history=True)
        probs = np.zeros(len(txns), dtype=np.float64)
        weights = np.zeros((len(txns), self.max_len), dtype=np.float32) if attention else None

        with torch.no_grad():
            for start in range(0, len(txns), SCORE_CHUNK):
                stop = min(start + SCORE_CHUNK, len(txns))
                X, mask = step_tensor(raw, idx[start:stop])
                logits, attended = self.model(
                    torch.from_numpy(self._standardise(X, mask)), torch.from_numpy(mask)
                )
                probs[start:stop] = torch.sigmoid(logits).numpy()
                if weights is not None:
                    weights[start:stop] = attended.numpy()
        return probs, weights, idx, raw

    def predict_proba(self, txns: list[Transaction]) -> np.ndarray:
        self._require_fitted()
        if not txns:
            return np.zeros(0, dtype=float)
        started = time.perf_counter()
        probs, *_ = self._forward(txns)
        self._score_seconds = time.perf_counter() - started
        self._scored_rows = len(txns)
        return probs

    def reason_codes(
        self, rows: list[int], attention: np.ndarray, idx: np.ndarray, raw: np.ndarray
    ) -> dict[int, list[str]]:
        """Which steps of this account's history made this row look wrong.

        A per-row local explanation, so it carries no `GLOBAL_PREFIX`. Always at least
        `explain.MIN_REASONS` statements: what the model is, how this payment sits against the
        account's own recent normal, and the earlier step the attention actually landed on.

        A window of one step says so rather than inventing a history to point at. That is not a
        corner case on every anchor — it is what PaySim looks like from here, and a sequence model
        that quietly narrated a trajectory over a single row would be the exact failure this
        module's docstring is about.
        """
        out: dict[int, list[str]] = {}
        for i in rows:
            steps = idx[i]
            present = np.flatnonzero(steps >= 0)
            amounts = raw[steps[present], RAW_AMOUNT]
            here = float(amounts[-1])
            codes = [f"sequence:{self.arch} over this account's last {present.size} payment(s)"]

            if present.size > 1:
                normal = float(np.exp(np.mean(np.log1p(np.maximum(amounts[:-1], 0.0)))) - 1.0)
                ratio = here / normal if normal > 0 else float("inf")
                codes.append(
                    f"{'↑' if ratio >= 1 else '↓'} {here:,.0f} against a running mean of "
                    f"{normal:,.0f} over the {present.size - 1} before it ({ratio:.1f}x)"
                )
                w = attention[i][present]
                # the last step is the row being judged; the explanation is about the history
                back = int(np.argmax(w[:-1]))
                gap_s = float(raw[steps[present[-1]], RAW_TS] - raw[steps[present[back]], RAW_TS])
                codes.append(
                    f"↑ {float(w[back]):.2f} of the attention on the payment "
                    f"{present.size - 1 - back} back: {float(amounts[back]):,.0f}, "
                    f"{_duration(gap_s)} earlier"
                )
                fresh = int(np.unique(raw[steps[present], RAW_DST]).size == present.size)
                codes.append(
                    "↑ every beneficiary in the window is a different one"
                    if fresh
                    else f"· {np.unique(raw[steps[present], RAW_DST]).size} distinct "
                    f"beneficiaries across the window"
                )
            else:
                codes.append(f"· {here:,.0f}, and this account has no earlier payment to read")
                codes.append(
                    "· a one-step window carries no trajectory — this score is the model's read "
                    "of a single row, which is what the supervised detector is for"
                )
            if len(codes) < explain.MIN_REASONS:  # pragma: no cover - both branches emit three
                raise AssertionError(
                    f"a flagged row left this model with {len(codes)} reason code(s) against a "
                    f"floor of {explain.MIN_REASONS} — see "
                    "explain.assert_flagged_rows_are_explained"
                )
            out[i] = codes
        return out

    def score(self, batch: AttackBatch) -> list[DetectorScore]:
        """Score, decide, and explain whatever the decision flagged.

        Two passes over the actions, priced the same way the supervised detector prices SHAP:
        the first finds which rows carry an action, the second re-runs only those through the
        network to recover their attention weights. On a batch where well under 1% of rows are
        flagged that keeps reason codes unconditional instead of a mode somebody switches off.
        """
        txns = batch.transactions
        if not txns:
            return []
        self._require_fitted()
        started = time.perf_counter()
        probs, *_ = self._forward(txns)
        actions = [
            self.policy.act(float(p), amount=t.amount) for t, p in zip(txns, probs, strict=False)
        ]
        flagged = np.array([i for i, a in enumerate(actions) if a is not Action.ALLOW], dtype=int)

        reasons: dict[int, list[str]] = {}
        if flagged.size:
            rows = [txns[i] for i in flagged]
            _, attention, idx, raw = self._forward(rows, attention=True)
            local = self.reason_codes(list(range(len(rows))), attention, idx, raw)
            reasons = {int(g): local[n] for n, g in enumerate(flagged)}
        self._score_seconds = time.perf_counter() - started
        self._scored_rows = len(txns)

        return [
            self.policy.decide(
                t.txn_id,
                float(p),
                amount=t.amount,
                reasons=reasons.get(i, [f"sequence:{self.arch}"]),
            )
            for i, (t, p) in enumerate(zip(txns, probs, strict=False))
        ]

    # ── introspection ───────────────────────────────────────────────────────────
    def model_card(self) -> dict[str, Any]:
        """Everything a reported number needs — including what it cost to produce.

        `compute` is on the card because ticket 17 asks for the trade to be visible: a lift is
        only worth reading next to the seconds it was bought with.
        """
        return {
            "detector": type(self).__name__,
            "arch": self.arch,
            "entity": self.entity,
            "hyperparameters": {
                "hidden": self.hidden,
                "layers": self.layers,
                "dropout": self.dropout,
                "max_len": self.max_len,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "negative_ratio": self.negative_ratio,
            },
            "seed": self.seed,
            "training": self.training.to_dict(),
            "history_coverage": self.coverage.to_dict(),
            "compute": self.compute_cost(),
            "step_features": list(STEP_FEATURES),
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
    "ARCHITECTURES",
    "DEFAULT_MAX_LEN",
    "STEP_FEATURES",
    "TORCH_HINT",
    "HistoryCoverage",
    "SequenceDetector",
    "SequenceTraining",
    "available",
    "raw_rows",
    "require_torch",
    "step_tensor",
    "window_index",
]
