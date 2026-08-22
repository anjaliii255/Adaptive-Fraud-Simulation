"""Level 2 — is the *structure* right?

Level 1 can be passed by shuffled noise with the right histogram. Fraud lives in structure:
who pays whom, how often, in what shape, and how bursty it is. This level is where a generator
that only learned marginals gets caught.

The embedding here is deliberately the generator's own view of a transaction, not the
detector's feature table — scoring fidelity in the detector's feature space would flatter the
generator on exactly the dimensions the detector already ignores.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from afl.contract.schema import Transaction

EMBED_COLUMNS = (
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "log_gap_s",
    "src_out_degree",
    "dst_in_degree",
    "src_uniq_dst",
)


def embedding(txns: list[Transaction]) -> np.ndarray:
    """One row per transaction: amount, timing, pacing, and running graph position."""
    rows = sorted(txns, key=lambda t: t.ts)
    last_seen: dict[str, float] = {}
    out_deg: dict[str, int] = defaultdict(int)
    in_deg: dict[str, int] = defaultdict(int)
    uniq_dst: dict[str, set] = defaultdict(set)

    out = []
    for t in rows:
        ts = t.ts.timestamp()
        gap = ts - last_seen.get(t.src, ts)
        out.append(
            [
                np.log1p(t.amount),
                t.ts.hour + t.ts.minute / 60.0,
                float(t.ts.weekday()),
                np.log1p(max(gap, 0.0)),
                float(out_deg[t.src]),
                float(in_deg[t.dst]),
                float(len(uniq_dst[t.src])),
            ]
        )
        last_seen[t.src] = ts
        out_deg[t.src] += 1
        in_deg[t.dst] += 1
        uniq_dst[t.src].add(t.dst)
    return np.asarray(out, dtype=float) if out else np.zeros((0, len(EMBED_COLUMNS)))


# ── motifs ──────────────────────────────────────────────────────────────────────
def graph_stats(txns: list[Transaction], fan_threshold: int = 3) -> dict[str, float]:
    """Motif frequencies, as shares so two datasets of different size stay comparable."""
    if not txns:
        return {}
    in_deg: dict[str, int] = defaultdict(int)
    out_deg: dict[str, int] = defaultdict(int)
    edges: set[tuple[str, str]] = set()
    for t in txns:
        out_deg[t.src] += 1
        in_deg[t.dst] += 1
        edges.add((t.src, t.dst))

    nodes = set(in_deg) | set(out_deg)
    n_nodes = max(len(nodes), 1)
    reciprocal = sum(1 for (a, b) in edges if (b, a) in edges)

    stats = {
        "edge_density": len(edges) / (n_nodes * max(n_nodes - 1, 1)),
        "fan_in_share": sum(1 for n in nodes if in_deg[n] >= fan_threshold) / n_nodes,
        "fan_out_share": sum(1 for n in nodes if out_deg[n] >= fan_threshold) / n_nodes,
        "pass_through_share": sum(1 for n in nodes if in_deg[n] >= 1 and out_deg[n] >= 1) / n_nodes,
        "reciprocity": reciprocal / max(len(edges), 1),
        "max_in_degree_share": max(in_deg.values()) / len(txns),
        "repeat_edge_share": 1.0 - len(edges) / len(txns),
        "cycle_share": _cycle_share(edges, n_nodes),
    }
    return {k: float(v) for k, v in stats.items()}


def _cycle_share(edges: set[tuple[str, str]], n_nodes: int, max_len: int = 4) -> float:
    """Short directed cycles per node. Round-tripping money is a motif no marginal will show."""
    try:
        import networkx as nx
    except ImportError:
        return 0.0
    g = nx.DiGraph()
    g.add_edges_from(edges)
    try:
        cycles = 0
        for i, _ in enumerate(nx.simple_cycles(g, length_bound=max_len)):
            cycles = i + 1
            if (
                cycles >= 10_000
            ):  # a dense graph can enumerate forever; the share is saturated anyway
                break
        return cycles / max(n_nodes, 1)
    except TypeError:  # networkx < 3.1 has no length_bound
        return 0.0


def motif_delta(real: list[Transaction], synth: list[Transaction]) -> dict[str, float]:
    """Per-motif relative error, clipped to [0, 1]."""
    a, b = graph_stats(real), graph_stats(synth)
    return {
        k: float(min(1.0, abs(a[k] - b.get(k, 0.0)) / max(a[k], b.get(k, 0.0), 1e-6))) for k in a
    }


# ── pacing ──────────────────────────────────────────────────────────────────────
def burstiness(txns: list[Transaction]) -> dict[str, float]:
    """Per-entity inter-arrival shape. Real accounts are bursty; naive generators are Poisson."""
    by_src: dict[str, list[float]] = defaultdict(list)
    for t in sorted(txns, key=lambda t: t.ts):
        by_src[t.src].append(t.ts.timestamp())

    cvs, gaps = [], []
    for stamps in by_src.values():
        if len(stamps) < 3:
            continue
        d = np.diff(stamps)
        gaps.extend(d.tolist())
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    if not gaps:
        return {"cv": 0.0, "median_gap_s": 0.0, "burst_index": 0.0}
    cv = float(np.mean(cvs)) if cvs else 0.0
    return {
        "cv": cv,
        "median_gap_s": float(np.median(gaps)),
        # Goh-Barabási burstiness: -1 regular, 0 Poisson, +1 bursty
        "burst_index": (cv - 1.0) / (cv + 1.0),
    }


def velocity_match(real: list[Transaction], synth: list[Transaction]) -> dict[str, float]:
    """How closely synthetic pacing and burstiness track the real thing."""
    from scipy.stats import ks_2samp

    def gaps(txns):
        by_src: dict[str, list[float]] = defaultdict(list)
        for t in sorted(txns, key=lambda t: t.ts):
            by_src[t.src].append(t.ts.timestamp())
        return np.concatenate([np.diff(v) for v in by_src.values() if len(v) > 1] or [np.zeros(1)])

    gr, gs = np.log1p(gaps(real)), np.log1p(gaps(synth))
    ks = float(ks_2samp(gr, gs).statistic) if gr.size > 1 and gs.size > 1 else 1.0
    br, bs = burstiness(real), burstiness(synth)
    return {
        "gap_ks": ks,
        "burst_index_real": round(br["burst_index"], 4),
        "burst_index_synth": round(bs["burst_index"], 4),
        "burst_index_delta": round(min(1.0, abs(br["burst_index"] - bs["burst_index"])), 4),
    }


# ── support overlap ─────────────────────────────────────────────────────────────
def alpha_precision_beta_recall(
    real: np.ndarray, synth: np.ndarray, alpha: float = 0.95, beta: float = 0.95
) -> dict[str, float]:
    """Simplified (spherical) Alaa et al. precision/recall.

    alpha-precision: share of synthetic rows inside the real data's alpha-support — high means
    "not making things up". beta-recall: share of real rows inside the synthetic support — high
    means "not just memorising one mode". A generator needs both; either alone is easy to game.
    """
    if real.size == 0 or synth.size == 0:
        return {"alpha_precision": 0.0, "beta_recall": 0.0}
    mu, sd = real.mean(0), real.std(0) + 1e-9
    r, s = (real - mu) / sd, (synth - mu) / sd

    r_centre, s_centre = r.mean(0), s.mean(0)
    r_radius = float(np.quantile(np.linalg.norm(r - r_centre, axis=1), alpha))
    s_radius = float(np.quantile(np.linalg.norm(s - s_centre, axis=1), beta))
    return {
        "alpha_precision": float((np.linalg.norm(s - r_centre, axis=1) <= r_radius).mean()),
        "beta_recall": float((np.linalg.norm(r - s_centre, axis=1) <= s_radius).mean()),
    }


def report(real: list[Transaction], synth: list[Transaction]) -> dict[str, object]:
    """Level 2 verdict: motifs, pacing, and support overlap."""
    motifs = motif_delta(real, synth)
    velocity = velocity_match(real, synth)
    support = alpha_precision_beta_recall(embedding(real), embedding(synth))

    distances = list(motifs.values()) + [
        velocity["gap_ks"],
        velocity["burst_index_delta"],
        1.0 - support["alpha_precision"],
        1.0 - support["beta_recall"],
    ]
    return {
        "level": 2,
        "motif_delta": {k: round(v, 4) for k, v in motifs.items()},
        "real_motifs": {k: round(v, 4) for k, v in graph_stats(real).items()},
        "synth_motifs": {k: round(v, 4) for k, v in graph_stats(synth).items()},
        "velocity": velocity,
        "support": {k: round(v, 4) for k, v in support.items()},
        "worst_motif": max(motifs, key=motifs.get) if motifs else None,
        "score": round(1.0 - float(np.mean(distances)), 4),
    }
