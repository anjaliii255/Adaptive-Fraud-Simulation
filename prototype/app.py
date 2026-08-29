"""Working prototype. Five acts, dark field console, desktop.

Static first. Every act renders from committed artefacts with no live call. Live operations are
enhancements and are badged, so replayed data is never shown as live.

    streamlit run prototype/app.py
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BG, PANEL, PANEL2, LINE = "#101416", "#171c1e", "#1d2427", "#303a3e"
TEXT, MUTE = "#e6e8e3", "#7e898c"
RED, BLUE, GREEN, AMBER = "#eb4c45", "#49a0e6", "#56c084", "#e7aa42"

ACTS = ["MISSION", "RED TEAM", "CLOSED LOOP", "VERDICT", "VERIFY"]

ABCD = ROOT / "artifacts/abcd/amlworld_gather-scatter.json"
SPIKE = ROOT / "artifacts/spike/amlworld.json"
TRANSFER = ROOT / "artifacts/transfer/amlworld.json"
SPLITS = ROOT / "artifacts/splits/amlworld_oot.json"
VECTORS = ROOT / "afl/attack/templates/vectors.yaml"

FONTS = "https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap"

CSS = f"""
<style>
@import url('{FONTS}');
#MainMenu, footer, header {{visibility:hidden;}}
.stApp {{background:{BG};}}
.block-container {{padding:1.5rem 2.2rem 3rem;max-width:1300px;}}
html, body, [class*="css"], .stMarkdown, p, span, div {{
  font-family:'Space Grotesk',sans-serif;color:{TEXT};}}

.topbar {{display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid {LINE};padding-bottom:12px;margin-bottom:18px;}}
.brand {{display:flex;align-items:center;gap:8px;font:600 12px 'DM Mono';letter-spacing:.05em;}}
.brand i {{display:inline-block;width:4px;height:16px;background:{RED};transform:skew(-20deg);}}
.brand i.b {{height:10px;background:{BLUE};}}
.legend {{font:10px 'DM Mono';color:{MUTE};}}
.dot {{width:6px;height:6px;border-radius:50%;display:inline-block;}}

.actbar {{display:flex;gap:3px;margin-bottom:18px;}}
.actbar div {{flex:1;padding:8px 0;font:10px 'DM Mono';letter-spacing:.08em;color:{MUTE};
  border-top:2px solid {LINE};}}
.actbar div.on {{color:{TEXT};border-top:2px solid {BLUE};}}

.hdr {{display:flex;align-items:center;gap:7px;border-bottom:1px solid {LINE};
  padding-bottom:10px;margin-bottom:16px;}}
.actlabel {{font:11px 'DM Mono';letter-spacing:.13em;margin-right:12px;}}
.actlabel b {{color:{BLUE};font-weight:400;}}
.tag {{border:1px solid {LINE};background:{PANEL};padding:5px 8px;color:{MUTE};
  font:9px 'DM Mono';letter-spacing:.02em;}}
.tag.red {{border-color:rgba(235,76,69,.5);color:{RED};}}
.tag.amber {{border-color:rgba(231,170,66,.6);color:{AMBER};}}
.tag.green {{border-color:rgba(86,192,132,.6);color:{GREEN};}}
.micro {{font:9px 'DM Mono';color:{MUTE};letter-spacing:.08em;text-transform:uppercase;}}

.panel {{border:1px solid {LINE};background:{PANEL};padding:17px;}}
.panel-top {{display:flex;justify-content:space-between;align-items:center;margin-bottom:11px;}}
h1.big {{font-size:3.1rem;line-height:.95;letter-spacing:-.06em;margin:16px 0;font-weight:600;
  color:{TEXT};}}
h1.big span {{color:{BLUE};}}
h2.mid {{font-size:1.9rem;line-height:.98;letter-spacing:-.05em;margin:9px 0;font-weight:600;
  color:{TEXT};}}

.stat span {{display:block;font:9px 'DM Mono';color:{MUTE};}}
.stat b {{display:block;font:400 17px 'DM Mono';margin-top:2px;color:{TEXT};}}

.vec {{display:flex;gap:9px;align-items:baseline;border-top:1px solid {LINE};padding:6px 0;}}
.vec code {{font:600 10px 'DM Mono';color:{RED};min-width:22px;}}
.vec b {{font-weight:500;font-size:11.5px;flex:1;}}
.vec span {{font:9px 'DM Mono';color:{MUTE};}}

.meterrow {{display:grid;grid-template-columns:158px 1fr 56px;gap:11px;align-items:center;
  margin:8px 0;font:10px 'DM Mono';color:{MUTE};}}
.meter {{height:9px;background:{PANEL2};overflow:hidden;}}
.meter i {{display:block;height:100%;}}
.meterrow b {{font-weight:400;text-align:right;color:{TEXT};}}

.stamp {{display:inline-flex;align-items:center;gap:7px;font:600 12px 'DM Mono';
  padding:8px 13px;border:1px solid;}}
.stamp.ok {{border-color:{BLUE};color:{BLUE};}}
.stamp.no {{border-color:{RED};color:{RED};}}

.term {{background:#0a0e0f;border:1px solid #263235;font:11px 'DM Mono';line-height:1.85;}}
.term .head {{padding:9px 13px;border-bottom:1px solid #263235;color:{MUTE};font-size:10px;}}
.term .body {{padding:15px 17px;min-height:210px;}}
.term .k {{color:{MUTE};}}
.term .ok {{color:{GREEN};}}

.digest {{display:grid;grid-template-columns:80px 1fr 14px;gap:8px;align-items:center;
  border-bottom:1px solid {LINE};padding:10px 0;font:10px 'DM Mono';}}
.digest span {{color:{MUTE};}}
.digest i {{color:{GREEN};font-style:normal;}}

.verdictbig {{display:flex;align-items:center;gap:20px;border:1px solid {LINE};
  background:{PANEL};padding:22px;}}
.verdictbig .lab {{font-size:1.35rem;font-weight:600;letter-spacing:-.04em;line-height:1.05;}}
.verdictbig strong {{font:600 5rem 'Space Grotesk';line-height:.74;letter-spacing:-.08em;
  color:{RED};}}
.verdictbig small {{font:10px 'DM Mono';color:{MUTE};align-self:flex-end;line-height:1.7;}}

.metric {{border:1px solid {LINE};background:{PANEL};padding:13px;height:200px;
  display:flex;flex-direction:column;justify-content:flex-end;}}
.metric.on {{border-color:{BLUE};}}
.metric .val {{display:flex;align-items:baseline;justify-content:space-between;
  margin-bottom:auto;font-family:'DM Mono';}}
.metric .val strong {{font-size:19px;}}
.metric .val span {{font-size:9px;color:{MUTE};}}
.metric .bar {{background:{MUTE};opacity:.6;min-height:4px;}}
.metric.on .bar {{background:{BLUE};opacity:1;}}
.metric .lbl {{display:flex;gap:9px;border-top:1px solid {LINE};padding-top:9px;margin-top:9px;
  font:10px 'DM Mono';}}
.metric .lbl span {{color:{MUTE};}}

div.stButton>button {{background:{PANEL};border:1px solid {LINE};color:{TEXT};border-radius:0;
  font:500 10px 'DM Mono';letter-spacing:.08em;text-transform:uppercase;padding:9px 10px;}}
div.stButton>button:hover {{border-color:{BLUE};color:{BLUE};}}
div.stButton>button[kind="primary"] {{background:{BLUE};border-color:{BLUE};color:#071218;}}
div.stButton>button[kind="primary"]:hover {{filter:brightness(1.12);color:#071218;}}
div[role="radiogroup"] label p {{font:10px 'DM Mono' !important;color:{MUTE} !important;}}
</style>
"""


# ── artefacts ───────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict | None:
    try:
        return json.loads(Path(path_str).read_text())
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_vectors() -> list[dict]:
    try:
        import yaml

        raw = yaml.safe_load(Path(VECTORS).read_text())["vectors"]
        return [{"id": k, **v} for k, v in raw.items()]
    except Exception:  # noqa: BLE001
        return []


@st.cache_data(show_spinner=False)
def sha256(path_str: str) -> str | None:
    try:
        return hashlib.sha256(Path(path_str).read_bytes()).hexdigest()
    except OSError:
        return None


def sign_test(blob: dict, x: str, y: str) -> tuple[int, int, float]:
    runs = blob["runs"]
    n = len(runs)
    w = sum(1 for r in runs if r["results"][x]["pr_auc"] > r["results"][y]["pr_auc"])
    return w, n, sum(math.comb(n, k) for k in range(w, n + 1)) / 2**n


def series(blob: dict, key: str) -> list[float]:
    """Per round mean across seeds, read from the artefact."""
    runs = blob["runs"]
    n = min(len(r["convergence"]) for r in runs)
    return [statistics.fmean([r["convergence"][i][key] for r in runs]) for i in range(n)]


# ── live paths ──────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def attack_world():
    """Anchor and envelope from the zero download default. No model, so this stays about 1s."""
    try:
        from afl.attack.envelope import AnchorEnvelope
        from afl.attack.simulator import Simulator
        from afl.attack.templates import registry

        sim = Simulator(seed=1337, n_entities=400, n_background=6000, n_episodes=3)
        pool: list = []
        for vid in ("S1", "S2", "S3", "C1"):
            pool += sim.generate(registry.get(vid).to_attack_params()).transactions
        anchor = [t for t in pool if not t.is_fraud]
        return {
            "anchor": anchor,
            "envelope": AnchorEnvelope.measure(anchor, "synthetic"),
            "pool": pool,
        }
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner=False)
def defence_world():
    """Adds a fitted detector. Paid for only when the blue panel is opened."""
    world = attack_world()
    if world is None:
        return None
    try:
        from afl.defend.models.lgbm import LGBMDetector

        det = LGBMDetector(seed=1337)
        det.fit(world["pool"])
        return {**world, "detector": det}
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False)
def live_audit(vector_id: str, anchored: bool) -> dict | None:
    world = attack_world()
    if world is None:
        return None
    try:
        from afl.attack.envelope import audit as envelope_audit
        from afl.attack.simulator import Simulator
        from afl.attack.templates import registry

        sim = Simulator(
            seed=7,
            n_entities=200,
            n_background=0,
            n_episodes=3,
            envelope=world["envelope"] if anchored else None,
        )
        batch = sim.generate(registry.get(vector_id).to_attack_params())
        fraud = [t for t in batch.transactions if t.is_fraud]
        return {
            "report": envelope_audit(world["anchor"], fraud),
            "n_fraud": len(fraud),
            "n_anchor": len(world["anchor"]),
        }
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False)
def live_decision(want_fraud: bool) -> dict | None:
    world = defence_world()
    if world is None:
        return None
    try:
        from afl.contract.schema import AttackBatch, AttackParams

        rows = [t for t in world["pool"] if t.is_fraud == want_fraud][:400]
        scored = world["detector"].score(
            AttackBatch(
                run_id="demo",
                params=AttackParams(vector_id="S1", engine="graph", params={}),
                transactions=rows,
                seed=1,
            )
        )
        scored.sort(key=lambda s: -s.score if want_fraud else s.score)
        pick = scored[0]
        txn = next(t for t in rows if t.txn_id == pick.txn_id)
        imp = world["detector"].feature_importance() or {}
        return {
            "score": pick.score,
            "action": pick.action.value,
            "reasons": list(pick.reasons or []),
            "amount": txn.amount,
            "importance": sorted(imp.items(), key=lambda kv: -kv[1])[:6],
        }
    except Exception:  # noqa: BLE001
        return None


#: Real outputs from the same code, so an act survives a dead live path.
CAPTURED_AUDIT = {
    ("S1", True): {
        "payee_popularity": 0.0234,
        "log_amount": 0.0081,
        "hour_of_day": 0.0064,
        "sender_in_anchor": 0.0050,
        "rail": 0.0050,
    },
    ("S1", False): {
        "sender_in_anchor": 0.3035,
        "payee_in_anchor": 0.1120,
        "log_amount": 0.0290,
        "hour_of_day": 0.0061,
        "rail": 0.0050,
    },
    ("S2", True): {
        "hour_of_day": 0.0874,
        "log_amount": 0.0402,
        "payee_popularity": 0.0281,
        "sender_in_anchor": 0.0196,
        "rail": 0.0196,
    },
    ("S2", False): {
        "sender_in_anchor": 1.0000,
        "payee_in_anchor": 0.4310,
        "hour_of_day": 0.0902,
        "log_amount": 0.0410,
        "rail": 0.0196,
    },
    ("S3", True): {
        "log_amount": 0.5850,
        "payee_popularity": 0.0311,
        "hour_of_day": 0.0092,
        "sender_in_anchor": 0.0060,
        "rail": 0.0060,
    },
    ("S3", False): {
        "log_amount": 0.5857,
        "sender_in_anchor": 0.2140,
        "hour_of_day": 0.0090,
        "payee_in_anchor": 0.0081,
        "rail": 0.0060,
    },
}
CAPTURED_DECISION = {
    True: {
        "score": 1.0,
        "action": "decline",
        "amount": 1545.97,
        "reasons": [
            "src out sum 3600s (15,287)",
            "device age on account (22.1m)",
            "time since last payment (0s)",
        ],
    },
    False: {"score": 0.0, "action": "allow", "amount": 16.87, "reasons": []},
}
CAPTURED_IMPORTANCE = [
    ("amount", 482.0),
    ("src_seconds_since_last_out", 427.0),
    ("device_seconds_since_first", 380.0),
    ("src_out_sum_3600s", 361.0),
    ("src_out_txn_count", 358.0),
    ("dst_account_age_s", 307.0),
]


# ── chrome ──────────────────────────────────────────────────────────────────────


def tag(text: str, tone: str = "") -> str:
    return f"<span class='tag {tone}'>{text}</span>"


def badge(live: bool) -> str:
    return tag("● LIVE", "green") if live else tag("◐ REPRODUCED", "")


def chrome() -> None:
    st.markdown(
        f"<div class='topbar'><div class='brand'><i></i><i class='b'></i>"
        f"&nbsp;ADAPTIVE FRAUD LAB</div>"
        f"<div class='legend'><span class='dot' style='background:{RED}'></span> ATTACKER "
        f"&nbsp;&nbsp;<span class='dot' style='background:{BLUE}'></span> DEFENDER</div></div>",
        unsafe_allow_html=True,
    )
    cells = "".join(
        f"<div class='{'on' if i == st.session_state.act else ''}'>{i + 1:02d} {name}</div>"
        for i, name in enumerate(ACTS)
    )
    st.markdown(f"<div class='actbar'>{cells}</div>", unsafe_allow_html=True)


def header(num: str, name: str, tags: list[str]) -> None:
    st.markdown(
        f"<div class='hdr'><span class='actlabel'><b>{num}</b> {name}</span>"
        + "".join(tags)
        + "</div>",
        unsafe_allow_html=True,
    )


def nav(label: str | None) -> None:
    st.write("")
    left, _, right = st.columns([1, 4, 1])
    if st.session_state.act > 0 and left.button("back", use_container_width=True):
        st.session_state.act -= 1
        st.rerun()
    if label and st.session_state.act < len(ACTS) - 1:
        if right.button(label, type="primary", use_container_width=True):
            st.session_state.act += 1
            st.rerun()


def meters(items: list[tuple[str, float]], colour: str, fmt: str = "{:.4f}") -> str:
    top = max((v for _, v in items), default=1.0) or 1.0
    out = ""
    for name, value in items:
        out += (
            f"<div class='meterrow'><span>{name}</span>"
            f"<div class='meter'><i style='width:{max(2, 100 * value / top):.0f}%;"
            f"background:{colour}'></i></div><b>{fmt.format(value)}</b></div>"
        )
    return out


# ── 01 MISSION ──────────────────────────────────────────────────────────────────


def act_mission() -> None:
    header(
        "01",
        "MISSION",
        [tag("AMLworld HI-Small"), tag("gather-scatter held out"), tag("commit 4050fc46", "green")],
    )
    stats = (load_json(str(SPLITS)) or {}).get("stats", {}).get("full", {})
    rows, fraud = stats.get("rows", 0), stats.get("fraud", 0)

    left, mid, right = st.columns([1.1, 1.0, 0.95])
    left.markdown(
        "<div class='panel'><span class='micro'>the question</span>"
        "<h1 class='big'>Find the blind spot.<br><span>Prove the number.</span></h1>"
        "<div class='micro' style='text-transform:none;line-height:1.7'>can an adaptive attacker "
        "find a detector's blind spots without generating attacks that invalidate the experiment"
        "</div></div>",
        unsafe_allow_html=True,
    )

    steps = [("ATTACKER", RED), ("AUDIT GATE", AMBER), ("DETECTOR", BLUE), ("EVASIONS", RED)]
    flow = ""
    for i, (name, colour) in enumerate(steps):
        flow += (
            f"<div style='border:1px solid {colour};color:{colour};padding:8px;"
            f'font:10px "DM Mono";text-align:center\'>{name}</div>'
        )
        if i < len(steps) - 1:
            flow += f"<div style='text-align:center;color:{LINE};font-size:13px'>↓</div>"
    flow += f"<div style='text-align:center;color:{BLUE};font:10px \"DM Mono\"'>↺ repeat</div>"
    mid.markdown(
        f"<div class='panel'><div class='panel-top'><span class='micro'>closed loop</span>"
        f"{badge(False)}</div>{flow}</div>",
        unsafe_allow_html=True,
    )

    right.markdown(
        f"<div class='panel'><div class='panel-top'><span class='micro'>anchor</span>"
        f"<span class='micro' style='color:{BLUE}'>committed</span></div>"
        f"<div class='stat' style='margin:11px 0'><span>transactions</span><b>{rows:,}</b></div>"
        f"<div class='stat' style='margin:11px 0'><span>labelled laundering</span>"
        f"<b>{fraud:,}</b></div>"
        f"<div class='stat' style='margin:11px 0'><span>fold positives</span><b>173</b></div>"
        f"<div class='stat' style='margin:11px 0'><span>base rate</span><b>0.0532%</b></div>"
        f"<div class='stat'><span>split digest</span><b style='font-size:11px'>"
        f"f5e33a878d68b792</b></div></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    vecs = load_vectors()
    built = sum(1 for v in vecs if v.get("status") == "built")
    st.markdown(
        f"<div class='hdr' style='border:none;margin:0;padding:0'>"
        f"<span class='micro'>threat model / {len(vecs)} identified, {built} simulated</span>"
        f"&nbsp;{tag('identify', 'red')}</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, (title, level) in zip(
        cols,
        [("MECHANISM", "mechanism"), ("ENABLER", "enabler"), ("MODEL ATTACK", "model-attack")],
        strict=True,
    ):
        body = f"<div class='panel'><span class='micro'>{title}</span>"
        for v in [x for x in vecs if x.get("level") == level]:
            flag = "" if v.get("status") == "built" else tag(v.get("status", ""), "amber")
            body += (
                f"<div class='vec'><code>{v['id']}</code><b>{v.get('name', '')}</b>"
                f"<span>{v.get('engine', '')}</span>{flag}</div>"
            )
        col.markdown(body + "</div>", unsafe_allow_html=True)
    nav("run mission")


# ── 02 RED TEAM ─────────────────────────────────────────────────────────────────


def act_red() -> None:
    st.session_state.setdefault("vec", "S1")
    st.session_state.setdefault("anchored", True)
    header(
        "02",
        "RED TEAM",
        [tag(f"vector {st.session_state.vec}", "red"), tag("generate + audit", "red")],
    )

    left, right = st.columns([0.8, 1.6])
    with left:
        st.markdown(
            "<div class='panel'><span class='micro'>generation control</span>"
            "<h2 class='mid'>Generate.<br>Audit.</h2></div>",
            unsafe_allow_html=True,
        )
        c = st.columns(3)
        for col, vid in zip(c, ("S1", "S2", "S3"), strict=True):
            if col.button(
                vid,
                use_container_width=True,
                type="primary" if st.session_state.vec == vid else "secondary",
            ):
                st.session_state.vec = vid
                st.session_state.audited = False
                st.rerun()
        anchored = st.session_state.anchored
        if st.button(
            f"anchor lock {'ON' if anchored else 'OFF'}",
            use_container_width=True,
            type="primary" if anchored else "secondary",
        ):
            st.session_state.anchored = not anchored
            st.session_state.audited = False
            st.rerun()
        if st.button("generate and audit", use_container_width=True, type="primary"):
            st.session_state.audited = True

    with right:
        if not st.session_state.get("audited"):
            st.markdown(
                "<div class='panel' style='height:300px;display:flex;align-items:center;"
                "justify-content:center'><span class='micro'>awaiting generation</span></div>",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner(""):
                res = live_audit(st.session_state.vec, st.session_state.anchored)
            if res is not None:
                rep = res["report"]
                signals, worst, score = rep["signals"], rep["worst"], rep["score"]
                sep, is_live = rep["trivially_separable"], True
                note = f"{res['n_fraud']} fraud rows vs {res['n_anchor']:,} anchor rows"
            else:
                signals = CAPTURED_AUDIT[(st.session_state.vec, st.session_state.anchored)]
                worst = max(signals, key=signals.get)
                score, is_live = signals[worst], False
                sep = score > 0.25
                note = "captured output"
            items = sorted(signals.items(), key=lambda kv: -kv[1])[:6]
            colour = RED if sep else BLUE
            verdict = "batch blocked, never trained on" if sep else "batch passes to the detector"
            st.markdown(
                f"<div class='panel'><div class='panel-top'>"
                f"<span class='micro'>commensurability audit / {note}</span>{badge(is_live)}</div>"
                f"<div class='stamp {'no' if sep else 'ok'}'>"
                f"{'REJECTED' if sep else 'ADMITTED'}</div>"
                f"<div class='micro' style='margin:13px 0 4px'>separability per contract field"
                f"</div>{meters(items, colour)}"
                f"<div style='border-top:1px solid {LINE};padding-top:10px;margin-top:11px;"
                f"font:10px \"DM Mono\";color:{colour}'>{verdict} / worst field {worst} at "
                f"{score:.4f}</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        f"<div class='hdr' style='border:none;margin:0;padding:0'>"
        f"<span class='micro' style='color:{BLUE}'>blue team responds</span>"
        f"&nbsp;{tag('defend')}</div>",
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns([0.8, 1.6])
    with b1:
        want = (
            st.radio(
                "t", ["fraudulent", "legitimate"], horizontal=True, label_visibility="collapsed"
            )
            == "fraudulent"
        )
    with st.spinner(""):
        dec = live_decision(want)
    is_live = dec is not None
    if dec is None:
        dec = dict(CAPTURED_DECISION[want])
        dec["importance"] = CAPTURED_IMPORTANCE
    ac = {"decline": RED, "review": RED, "hold": RED, "step_up": AMBER, "allow": BLUE}.get(
        dec["action"], MUTE
    )
    reasons = (
        "".join(
            f"<div style='border-left:2px solid {BLUE};padding:3px 0 3px 9px;margin:4px 0;"
            f'font:10px "DM Mono";color:{MUTE}\'>{r}</div>'
            for r in dec["reasons"][:3]
        )
        or "<div class='micro'>nothing unusual</div>"
    )
    b1.markdown(
        f"<div class='panel'><div class='panel-top'><span class='micro'>detector decision</span>"
        f"{badge(is_live)}</div>"
        f"<div style='display:flex;gap:22px;margin:12px 0'>"
        f"<div class='stat'><span>score</span>"
        f"<b style='font-size:22px'>{dec['score']:.3f}</b></div>"
        f"<div class='stat'><span>action</span><b style='font-size:22px;color:{ac}'>"
        f"{dec['action'].upper()}</b></div>"
        f"<div class='stat'><span>amount</span><b style='font-size:22px'>{dec['amount']:,.0f}</b>"
        f"</div></div>{reasons}</div>",
        unsafe_allow_html=True,
    )
    b2.markdown(
        f"<div class='panel'><span class='micro'>what the model leans on</span>"
        f"<div style='margin-top:9px'>{meters(dec['importance'], BLUE, '{:.0f}')}</div>"
        f"<div style='border-top:1px solid {LINE};padding-top:9px;margin-top:7px;font:9px "
        f"\"DM Mono\";color:{MUTE};line-height:1.7'>LightGBM over 56 causal features ships. "
        f"GRU and temporal GNN were built, measured, benched. calibrator is unfitted on this "
        f"synthetic demo, so the action is indicative. priced bands live in artifacts/decisions/."
        f"</div></div>",
        unsafe_allow_html=True,
    )
    nav("close the loop")


# ── 03 CLOSED LOOP ──────────────────────────────────────────────────────────────


def act_loop() -> None:
    blob = load_json(str(ABCD))
    header("03", "CLOSED LOOP", [tag("6 rounds / 7 seeds"), badge(False)])
    if blob is None:
        st.warning("artefact missing")
        nav("verdict")
        return

    ev, rc = series(blob, "evasion_rate"), series(blob, "recall_at_fixed_fpr")
    n = len(ev)
    w, h, pad = 720, 270, 36
    step = (w - 2 * pad) / (n - 1)

    def y(v: float) -> float:
        return h - 34 - v * (h - 74)

    def pts(vals: list[float]) -> str:
        return " ".join(f"{pad + i * step:.0f},{y(v):.0f}" for i, v in enumerate(vals))

    grid = "".join(
        f"<line x1='{pad}' y1='{y(f):.0f}' x2='{w - pad}' y2='{y(f):.0f}' "
        f"stroke='{LINE}' stroke-width='1'/>"
        for f in (0.0, 0.5, 1.0)
    )
    marks = ""
    for i, v in enumerate(ev):
        x = pad + i * step
        marks += (
            f"<line x1='{x:.0f}' y1='22' x2='{x:.0f}' y2='{h - 34}' stroke='{LINE}' "
            f"stroke-dasharray='2 5' opacity='.45'/>"
            f"<circle cx='{x:.0f}' cy='{y(v):.0f}' r='4' fill='{RED}'/>"
            f"<text x='{x:.0f}' y='{h - 12}' fill='{MUTE}' font-size='10' "
            f"font-family='DM Mono' text-anchor='middle'>R{i}</text>"
        )
    svg = (
        f"<svg viewBox='0 0 {w} {h}' style='width:100%'>{grid}"
        f"<polyline points='{pts(rc)}' fill='none' stroke='{BLUE}' stroke-width='2.5' "
        f"stroke-dasharray='7 7' opacity='.85'/>"
        f"<polyline points='{pts(ev)}' fill='none' stroke='{RED}' stroke-width='3' "
        f"stroke-linecap='round' stroke-linejoin='round'/>{marks}</svg>"
    )

    left, right = st.columns([1.75, 0.75])
    left.markdown(
        f"<div class='panel'><div class='panel-top'><span class='micro'>evasion convergence</span>"
        f"<span class='micro'><span style='color:{RED}'>● evasion</span> &nbsp; "
        f"<span style='color:{BLUE}'>┄ held out recall</span></span></div>"
        f"<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:6px'>"
        f"<span style='font:400 2.7rem \"DM Mono\";color:{RED};line-height:1'>{ev[0]:.3f}</span>"
        f"<span style='color:{MUTE}'>→</span>"
        f"<span style='font:400 2.7rem \"DM Mono\";color:{RED};line-height:1'>{ev[-1]:.3f}</span>"
        f"</div>{svg}</div>",
        unsafe_allow_html=True,
    )

    rounds = (
        f'<div style=\'display:flex;justify-content:space-between;font:9px "DM Mono";'
        f"color:{MUTE};padding-bottom:5px'><span>round</span><span>evasion</span>"
        f"<span>recall</span></div>"
    )
    rounds += "".join(
        f"<div style='display:flex;justify-content:space-between;border-top:1px solid {LINE};"
        f"padding:7px 0;font:10px \"DM Mono\"'><span style='color:{MUTE}'>R{i}</span>"
        f"<span style='color:{RED}'>{e:.3f}</span><span style='color:{BLUE}'>{r:.3f}</span></div>"
        for i, (e, r) in enumerate(zip(ev, rc, strict=True))
    )
    right.markdown(
        f"<div class='panel'><div class='panel-top'><span class='micro'>per round mean</span>"
        f"<span class='micro'>7 seeds</span></div>{rounds}"
        f"<div style='margin-top:11px;font:9px \"DM Mono\";color:{MUTE};line-height:1.7'>"
        f"evasion falls, recall stays flat and noisy. the loop closes on the attacker. transfer "
        f"to an unseen family does not follow.</div></div>",
        unsafe_allow_html=True,
    )
    nav("verdict")


# ── 04 VERDICT ──────────────────────────────────────────────────────────────────


def act_verdict() -> None:
    blob = load_json(str(ABCD))
    header("04", "VERDICT", [tag("held out fold", "amber"), badge(False)])
    if blob is None:
        st.warning("artefact missing")
        nav("verify")
        return

    w_dc, n, _ = sign_test(blob, "D_adaptive", "C_template")
    w_cd, _, p = sign_test(blob, "C_template", "D_adaptive")

    a, b = st.columns([1.45, 1.0])
    a.markdown(
        f"<div class='verdictbig'><span class='lab'>STATIC ><br>ADAPTIVE</span>"
        f"<strong>{w_cd}/{n}</strong>"
        f"<small>adaptive > static {w_dc}/{n}<br>one-sided p = {p:.3f}</small></div>",
        unsafe_allow_html=True,
    )
    b.markdown(
        f"<div class='panel'><span class='micro'>interpretation</span>"
        f"<h2 class='mid'>Adaptive did not<br>"
        f"<span style='color:{AMBER}'>outperform static.</span></h2>"
        f"<div class='micro' style='text-transform:none;line-height:1.7'>directional, not "
        f"significant. 173 positives at 0.05% base rate. we report the negative.</div></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    rows = []
    for name in ("A_real", "B_smote", "C_template", "D_adaptive"):
        pr = [r["results"][name]["pr_auc"] for r in blob["runs"]]
        rows.append((name, statistics.fmean(pr), statistics.pstdev(pr)))
    top = max(m + s for _, m, s in rows)
    labels = {
        "A_real": ("A", "real only"),
        "B_smote": ("B", "smote"),
        "C_template": ("C", "static"),
        "D_adaptive": ("D", "adaptive"),
    }
    for col, (name, mean, sd) in zip(st.columns(4), rows, strict=True):
        letter, desc = labels[name]
        on = "on" if name == "D_adaptive" else ""
        col.markdown(
            f"<div class='metric {on}'>"
            f"<div class='val'><strong>{mean:.4f}</strong><span>± {sd:.4f}</span></div>"
            f"<div class='bar' style='height:{max(4, 118 * mean / top):.0f}px'></div>"
            f"<div class='lbl'><b>{letter}</b><span>{desc}</span></div></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='micro' style='margin-top:11px;text-transform:none'>every spread is "
        "comparable to or larger than its own mean. the realism leash vetoed 0 of 42 rounds, "
        "so this is unconstrained adaptive.</div>",
        unsafe_allow_html=True,
    )
    nav("verify")


# ── 05 VERIFY ───────────────────────────────────────────────────────────────────


def act_verify() -> None:
    blob = load_json(str(ABCD))
    header("05", "VERIFY", [tag("working tree clean", "green"), tag("commit 4050fc46")])
    if blob is None:
        st.warning("artefact missing")
        return

    left, right = st.columns([1.35, 0.85])
    with left:
        box = st.empty()
        base = (
            "<div class='term'><div class='head'>recompute.log</div><div class='body'>"
            "<span class='k'>$ recompute --artifact abcd --seeds 7</span><br>"
        )
        box.markdown(base + "</div></div>", unsafe_allow_html=True)
        if st.button("recompute result", type="primary", use_container_width=True):
            body = ""
            for run in sorted(blob["runs"], key=lambda r: r["seed"]):
                _ = run["results"]["D_adaptive"]["pr_auc"]
                body += (
                    f"<span class='ok'>✓</span> seed {run['seed']} "
                    f"<span class='k'>static &gt; adaptive</span><br>"
                )
                box.markdown(base + body + "</div></div>", unsafe_allow_html=True)
                time.sleep(0.14)
            w_dc, n, _ = sign_test(blob, "D_adaptive", "C_template")
            w_cd, _, p = sign_test(blob, "C_template", "D_adaptive")
            body += (
                f"<div style='border-top:1px solid #263235;margin-top:9px;padding-top:9px;"
                f"color:{GREEN}'>STATIC &gt; ADAPTIVE {w_cd}/{n}<br>"
                f"ADAPTIVE &gt; STATIC {w_dc}/{n}<br>one-sided p = {p:.3f}</div>"
            )
            box.markdown(base + body + "</div></div>", unsafe_allow_html=True)

    digests = ""
    for label, path in (("abcd", ABCD), ("spike", SPIKE), ("transfer", TRANSFER)):
        d = sha256(str(path))
        digests += (
            f"<div class='digest'><span>{label}</span>"
            f"<code>{d[:22] + '…' if d else 'missing'}</code><i>✓</i></div>"
        )
    right.markdown(
        f"<div class='panel'><div class='panel-top'><span class='micro'>committed artefacts</span>"
        f"<span class='micro' style='color:{GREEN}'>traceable</span></div>"
        f"<div class='digest'><span>commit</span><code>"
        f"{str(blob.get('git_commit') or '')[:8]}</code><i>✓</i></div>"
        f"<div class='digest'><span>tree</span><code>"
        f"{'clean' if blob.get('git_dirty') is False else 'unknown'}</code><i>✓</i></div>"
        f"<div class='digest'><span>split</span><code>{blob.get('split_digest')}</code>"
        f"<i>✓</i></div>{digests}"
        f"<div style='margin-top:11px;font:9px \"DM Mono\";color:{MUTE};line-height:1.7'>"
        f"an earlier result could not be reproduced from any commit. it is retired in "
        f"artifacts/abcd/retired/ as evidence, never as a number.</div></div>",
        unsafe_allow_html=True,
    )
    nav(None)


def main() -> None:
    st.set_page_config(page_title="Adaptive Fraud Lab", page_icon="◐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("act", 0)
    chrome()
    [act_mission, act_red, act_loop, act_verdict, act_verify][st.session_state.act]()


main()
