"""Streamlit demo: pick an attack, run the loop, watch the curves cross.

The one thing worth showing live is the shape of the loop — evasion falling as held-out recall
rises. Everything else is a table someone can read afterwards.

Talks to the FastAPI service so the demo and the experiments share one implementation.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

API = os.getenv("AFL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Adaptive Fraud Simulation Lab", layout="wide")


def call(path: str, payload: dict | None = None, timeout: int = 120):
    """GET or POST against the API, failing loudly if it is not up."""
    url = f"{API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        st.error(f"cannot reach the API at {API} — is `make demo` up? ({e})")
        st.stop()


# ── header ──────────────────────────────────────────────────────────────────────
st.title("Adaptive Fraud Simulation Lab")
health = call("/health")
c1, c2, c3 = st.columns(3)
c1.metric("rounds run", health["rounds_run"])
c2.metric("held-out family", health["held_out_vector"])
c3.metric("detector", health["detector_backend"])
st.caption(
    "The held-out family is never generated during the loop. Every number below is measured "
    "on that family, out-of-time — the loop trains on attacks it has not been graded on."
)

vectors = call("/vectors")
by_id = {v["vector_id"]: v for v in vectors}
available = [v["vector_id"] for v in vectors if not v["held_out"]]

# ── controls ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Attack")
    vector_id = st.selectbox("vector", available, format_func=lambda v: f"{v} — {by_id[v]['name']}")
    spec = by_id[vector_id]
    st.write(
        f"**engine** `{spec['engine']}` · **actor** `{spec['actor']}` · "
        f"**maturity** `{spec['maturity']}`"
    )
    st.caption(spec["why"])
    st.write("**searched knobs:** " + (", ".join(f"`{k}`" for k in spec["searchable"]) or "none"))

    rounds = st.slider("rounds per click", 1, 10, 3)
    run = st.button("Run loop", type="primary", use_container_width=True)
    preview = st.button("Preview one batch", use_container_width=True)
    if st.button("Reset lab", use_container_width=True):
        call("/loop/reset", {})
        st.rerun()

# ── actions ─────────────────────────────────────────────────────────────────────
if preview:
    batch = call("/simulate", {"vector_id": vector_id, "include_transactions": True})
    st.subheader(f"Batch `{batch['run_id']}`")
    a, b, c = st.columns(3)
    a.metric("transactions", batch["n_transactions"])
    b.metric("fraud rows", batch["n_fraud"])
    c.metric("realism penalty", round(batch["realism"]["penalty"], 3))
    if batch["realism"]["violations"]:
        st.warning("realism violations: " + ", ".join(batch["realism"]["violations"]))
    st.dataframe(
        pd.DataFrame(batch["transactions"]).head(200), use_container_width=True, height=280
    )

if run:
    with st.spinner(f"running {rounds} round(s) of {vector_id}…"):
        call("/loop/step", {"rounds": rounds, "vector_id": vector_id})

# ── the curve ───────────────────────────────────────────────────────────────────
state = call("/metrics")
history = pd.DataFrame(state["history"])

st.subheader("Convergence")
if history.empty:
    st.info("No rounds yet — pick a vector and hit **Run loop**.")
else:
    curves = history.set_index("round")[["evasion_rate", "recall_at_fixed_fpr", "pr_auc"]]
    st.line_chart(curves)
    st.caption(
        "Attacker evasion should fall as held-out recall rises. If evasion falls while held-out "
        "recall is flat, the detector has learnt this batch, not the family."
    )

    last = history.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("evasion rate", f"{last['evasion_rate']:.2%}")
    m2.metric(f"recall @ {last['fixed_fpr']:.0%} FPR", f"{last['recall_at_fixed_fpr']:.2%}")
    m3.metric("PR-AUC", f"{last['pr_auc']:.3f}")
    m4.metric("fraud rows this round", int(last["n_fraud"]))

    with st.expander("Round log"):
        st.dataframe(history, use_container_width=True)

trials = pd.DataFrame(state["attack_trials"])
if not trials.empty:
    st.subheader("What the attacker tried")
    st.caption(
        "fitness = evasion rate − λ · realism penalty. The λ term is why the search "
        "cannot win by cheating."
    )
    st.dataframe(trials, use_container_width=True)
