"""
MemPulse: a single-screen visual debugger for AI agent / RAG memory.

    uv run streamlit run app.py

HOW TO READ THE SCREEN
----------------------
  Top     pipeline strip. Where we are in time (ingest, then each turn).
  Left    memory stack. How full each memory tier is at this exact step.
  Centre  stage lens. A visual specific to the active node (chunk ribbon,
          vector space, prompt assembly...).
  Right   state diff. What the last node changed in AgentState.
  Bottom  "Why memory state matters". The same question under other memory
          configurations, with a verdict on where each one breaks.

HOW THE APP WORKS
-----------------
Streamlit reruns this whole script on every click. So we keep the expensive,
stateful things in st.session_state (the list of snapshots and the cursor)
and in st.cache_* (the vector index and diagnostics). Navigation just moves
the cursor, which is time-travel through recorded memory states.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

from src.data.sample import CLINIC_GUIDE, INITIAL_LTM, SCRIPTED_TURNS, needle_span
from src.graph.pipeline import NODE_EXPLAIN, NODE_LABELS, NODES, initial_state, run_turn
from src.graph.trace import INGEST_EXPLAIN, INGEST_LABELS, INGEST_NODES, Step, build_ingest_steps, diff_states
from src.llm import claude
from src.rag.context import TIER_COLORS
from src.rag.diagnostics import VERDICT_SHORT, VERDICT_STYLE, evaluate, sensitivity_grid, what_if
from src.rag.embeddings import EMBEDDER_LABELS, pca_2d
from src.rag.index import VectorIndex
from src.ui import charts
from src.ui import components as ui
from src.ui.styles import get_carbon_css

st.set_page_config(page_title="MemPulse", page_icon="🧠", layout="wide")
st.markdown(get_carbon_css(), unsafe_allow_html=True)

# Widget defaults. Keys let the What-if "Apply" buttons change the sliders.
DEFAULTS = dict(chunk_size=400, chunk_overlap=60, strategy="recursive", embedder="bge-small", top_k=3, budget=500)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# =============================================================== SIDEBAR

with st.sidebar:
    st.markdown("## Document")
    src = st.radio("Source", ["Sample clinic guide", "Paste text", "Upload .md/.txt"], label_visibility="collapsed")
    doc = CLINIC_GUIDE
    if src == "Paste text":
        doc = st.text_area("Paste a document", height=160, value="") or CLINIC_GUIDE
    elif src == "Upload .md/.txt":
        f = st.file_uploader("Upload", type=["md", "txt"])
        if f is not None:
            doc = f.read().decode("utf-8", errors="replace")
    if doc != CLINIC_GUIDE:
        st.caption("Custom document: answer-fact tracking only works for questions whose fact is in the sample guide.")

    st.markdown("## Chunking")
    st.radio("Strategy", ["recursive", "heading"], key="strategy", horizontal=True,
             help="recursive = fixed size, breaks at paragraph/line/word; heading = one chunk per markdown section")
    st.slider("chunk_size (chars)", 50, 2000, step=10, key="chunk_size")
    st.slider("chunk_overlap (chars)", 0, 400, step=5, key="chunk_overlap",
              help="Clamped to half the chunk size.")

    st.markdown("## Embedding & retrieval")
    st.selectbox("Embedder", list(EMBEDDER_LABELS), format_func=EMBEDDER_LABELS.get, key="embedder")
    st.slider("top-k", 1, 8, key="top_k")
    st.slider("Context budget (tokens)", 200, 2000, step=50, key="budget",
              help="Deliberately small so you can watch eviction happen. Real models have far more, but real prompts are bigger too.")

    st.markdown("## Claude (optional)")
    api_key = st.text_input("ANTHROPIC_API_KEY", type="password", help="Leave empty to use ANTHROPIC_API_KEY from .env or the environment.")
    model = st.selectbox("Model", claude.MODELS)

cfg = {k: st.session_state[k] for k in DEFAULTS}
client = claude.get_client(api_key)


# ======================================================= CACHED BUILDERS

@st.cache_resource(show_spinner="Chunking & embedding (first run downloads the bge-small model, ~70 MB)…")
def get_index(doc: str, size: int, overlap: int, strategy: str, embedder: str) -> VectorIndex:
    return VectorIndex.build(doc, size, overlap, strategy, embedder)


@st.cache_data(show_spinner="Running What-if configurations…")
def cached_what_if(doc, query, span, cfg, ltm, history):
    return what_if(doc, query, span, cfg, ltm, history)


@st.cache_data(show_spinner="Sweeping chunk_size × overlap…")
def cached_grid(doc, query, span, cfg, ltm, history):
    return sensitivity_grid(doc, query, span, cfg, ltm, history)


index = get_index(doc, cfg["chunk_size"], cfg["chunk_overlap"], cfg["strategy"], cfg["embedder"])


# ================================================= TRACE (session state)
# The trace is a flat list of Steps. If the memory config changes, we rebuild
# ingestion AND replay the turns already asked, so you can compare the exact
# same conversation under new chunking/embedding settings.

cfg_key = hashlib.md5((doc + json.dumps(cfg, sort_keys=True) + model + str(bool(client))).encode()).hexdigest()
ss = st.session_state
ss.setdefault("prompts", [])       # user prompts asked so far (replayed on config change)
ss.setdefault("cursor", 0)


def rebuild(keep_cursor: bool = True) -> None:
    steps = build_ingest_steps(doc, index)
    state = initial_state(INITIAL_LTM)
    for t, p in enumerate(ss.prompts, start=1):
        snaps = run_turn(index, cfg, state, p, client, model)
        steps += [Step("turn", s["active_node"], t, s) for s in snaps]
        state = snaps[-1]
    ss.steps, ss.final_state, ss.cfg_key = steps, state, cfg_key
    ss.cursor = min(ss.cursor, len(steps) - 1) if keep_cursor else 0


def ask(prompt: str) -> None:
    """Run a new turn and jump the cursor to its first node."""
    snaps = run_turn(index, cfg, ss.final_state, prompt, client, model)
    t = len(ss.prompts) + 1
    ss.prompts.append(prompt)
    first = len(ss.steps)
    ss.steps += [Step("turn", s["active_node"], t, s) for s in snaps]
    ss.final_state = snaps[-1]
    ss.cursor = first


if ss.get("cfg_key") != cfg_key:
    with st.spinner("Memory config changed: rebuilding index and replaying the conversation…"):
        rebuild()

steps: list[Step] = ss.steps
next_scripted = SCRIPTED_TURNS[len(ss.prompts)] if len(ss.prompts) < len(SCRIPTED_TURNS) else None


# ================================================================ HEADER

h1, h2 = st.columns([3, 2])
with h1:
    st.markdown("# MemPulse")
    st.markdown('<div class="subtitle">Visual Agent Memory &amp; State Debugger: see what the agent remembers, at every step.</div>',
                unsafe_allow_html=True)
with h2:
    llm_badge = f'<span class="badge on">Claude · {model}</span>' if client else '<span class="badge off">Claude off · offline answers</span>'
    st.markdown(
        '<div style="text-align:right;margin-top:14px">'
        f'<span class="badge">{cfg["strategy"]} {cfg["chunk_size"]}/{cfg["chunk_overlap"]}</span>'
        f'<span class="badge">{cfg["embedder"]} {index.vectors.shape[1]}D</span>'
        f'<span class="badge">{len(index.chunks)} chunks</span>'
        f'<span class="badge">top-k {cfg["top_k"]}</span>'
        f'<span class="badge">budget {cfg["budget"]} tok</span>{llm_badge}</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------ pipeline + controls
cur: Step = steps[ss.cursor]
key_of = lambda s: f"{s.turn}:{s.node}"  # noqa: E731
groups = [("Ingest", [(f"0:{n}", INGEST_LABELS[n]) for n in INGEST_NODES])]
for t in range(1, len(ss.prompts) + 2):
    if t > len(ss.prompts) and next_scripted is None:
        break
    groups.append((f"Turn {t}", [(f"{t}:{n}", NODE_LABELS[n]) for n in NODES]))
done = {key_of(s) for s in steps[: ss.cursor]}
ui.pipeline_strip(groups, key_of(cur), done)

c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1.4, 1, 3, 0.8])
if c1.button("◀ Prev", width="stretch", disabled=ss.cursor == 0):
    ss.cursor -= 1
    st.rerun()
at_end = ss.cursor == len(steps) - 1
if c2.button("Next ▶", width="stretch", type="primary", disabled=at_end and next_scripted is None):
    if at_end:
        ask(next_scripted["prompt"])  # next step = first node of the next scripted turn
    else:
        ss.cursor += 1
    st.rerun()
if c3.button("⏭ Run next turn", width="stretch", disabled=next_scripted is None):
    ask(next_scripted["prompt"])
    ss.cursor = len(ss.steps) - 1
    st.rerun()
if c4.button("↺ Reset", width="stretch"):
    ss.prompts, ss.cursor = [], 0
    rebuild(keep_cursor=False)
    st.rerun()
with c5:
    custom = st.text_input("Ask your own question", placeholder="Ask your own question (runs a new turn)…",
                           label_visibility="collapsed")
if c6.button("Ask", width="stretch", disabled=not custom):
    ask(custom)
    st.rerun()

# One-line teaching caption for the active step (+ the scripted turn's lesson).
explain = INGEST_EXPLAIN.get(cur.node) if cur.phase == "ingest" else NODE_EXPLAIN.get(cur.node)
lesson = ""
if cur.phase == "turn" and cur.turn <= len(SCRIPTED_TURNS) and ss.prompts[cur.turn - 1] == SCRIPTED_TURNS[cur.turn - 1]["prompt"]:
    lesson = f" <b>Turn {cur.turn} lesson:</b> {SCRIPTED_TURNS[cur.turn - 1]['lesson']}"
where = "Ingest" if cur.phase == "ingest" else f'Turn {cur.turn} · “{ui.esc(cur.state.get("user_prompt", ""))}”'
st.markdown(f'<div class="lesson"><b>Step {ss.cursor + 1}/{len(steps)} · {where}.</b> {ui.esc(explain or "")}{lesson}</div>',
            unsafe_allow_html=True)


# ====================================================== SHARED ANALYSIS
# Which question are we diagnosing? The current turn's, or (during ingest)
# the first scripted question, so the ribbon can already show its answer fact.

def needle_for(prompt: str | None):
    for t in SCRIPTED_TURNS:
        if t["prompt"] == prompt:
            return needle_span(doc, t["needle"])
    return None


query = cur.state.get("user_prompt") if cur.phase == "turn" else SCRIPTED_TURNS[0]["prompt"]
span = needle_for(query)
ltm_now = cur.state.get("long_term_memory", INITIAL_LTM)
# Diagnose with the memory as it was when this turn STARTED (LTM + history
# before the turn), so the numbers match what the Assemble node actually packed.
turn_start = next((s.state for s in steps if s.phase == "turn" and s.turn == cur.turn), None)
hist_at_start = turn_start["conversation_history"] if turn_start else []
ltm_at_start = turn_start["long_term_memory"] if turn_start else ltm_now
diag = evaluate(index, query, span, cfg["top_k"], cfg["budget"], ltm_at_start, hist_at_start)
ids = [c.chunk_id for c in index.chunks]
fact_intact = diag["verdict"] != "split"


# ================================================================ BODY

left, center, right = st.columns([1.05, 2.1, 1.15], gap="medium")

# ------------------------------------------------------------ MEMORY STACK
with left:
    ui.panel_title("Memory stack · at this step")
    ingest_idx = INGEST_NODES.index(cur.node) if cur.phase == "ingest" else 99
    n = len(index.chunks)

    # 1. Knowledge memory (vector store): fills up during ingestion.
    if ingest_idx == 0:
        kn_label, kn_fill, kn_items = "raw text", 0.1, f"{len(doc):,} chars · not searchable yet"
    elif ingest_idx == 1:
        kn_label, kn_fill, kn_items = f"{n} chunks", 0.5, "chunks exist as text · no vectors yet"
    else:
        kn_label, kn_fill = f"{n} × {index.vectors.shape[1]}", 1.0
        kn_items = "vectors ready" if ingest_idx >= 3 else "embedding…"
        if cur.phase == "turn" and cur.state.get("retrieved_chunks"):
            kn_items += " · pulled: " + ", ".join(c["chunk_id"] for c in cur.state["retrieved_chunks"])
    ui.tier_card("Knowledge · vector store", TIER_COLORS["chunk"], kn_label, kn_fill, ui.esc(kn_items))

    # 2. Long-term memory: persistent profile; new writes pulse green.
    ltm = ltm_now
    writes = {w["fact"] for w in cur.state.get("memory_writes", [])} if cur.node == "write_memory" else set()
    facts = [("medical_history", f) for f in ltm.get("medical_history", [])] + [("preferences", f) for f in ltm.get("preferences", [])]
    items = "<br>".join(
        f'<span class="{"new" if f in writes else ""}">{"＋ " if f in writes else "• "}{ui.esc(f)}</span>' for _, f in facts)
    ui.tier_card("Long-term memory · profile", TIER_COLORS["ltm"], f"{len(facts)} facts", len(facts) / 8, items,
                 pulse=bool(writes))

    # 3. Short-term memory: the running conversation.
    hist = cur.state.get("conversation_history", [])
    h_items = "<br>".join(f'<span class="dim">{m["role"]}:</span> {ui.esc(m["content"][:70])}{"…" if len(m["content"]) > 70 else ""}'
                          for m in hist[-4:]) or '<span class="dim">empty: no turns yet</span>'
    ui.tier_card("Short-term memory · history", TIER_COLORS["history"], f"{len(hist)} msgs", len(hist) / 10, h_items)

    # 4. Working memory: the AgentState keys that are populated right now.
    if cur.phase == "turn":
        filled = [k for k, v in cur.state.items() if v not in ("", [], {}, None)]
        w_items = ui.esc(", ".join(filled))
        ui.tier_card("Working memory · AgentState", TIER_COLORS["prompt"], f"{len(filled)}/{len(cur.state)} keys",
                     len(filled) / len(cur.state), w_items)
    else:
        ui.tier_card("Working memory · AgentState", TIER_COLORS["prompt"], "idle", 0.0,
                     '<span class="dim">no turn running: agent is idle during ingest</span>')

    # 5. Context window gauge (only exists once a prompt has been assembled).
    if cur.state.get("context_window"):
        ui.context_gauge(cur.state["context_window"], cfg["budget"])

# -------------------------------------------------------------- STAGE LENS
with center:
    ui.panel_title(f"Stage lens · {INGEST_LABELS.get(cur.node) or NODE_LABELS.get(cur.node)}")
    node = cur.node

    if node == "load":
        ui.doc_ribbon(doc, [], span)
    elif node == "chunk":
        lens = [len(c.text) for c in index.chunks]
        ov = sum(c.overlap_prev for c in index.chunks)
        ui.tiles([
            ("Chunks", str(n), f"avg {np.mean(lens):.0f} chars", TIER_COLORS["chunk"]),
            ("Overlap stored", f"{ov} ch", f"{ov / max(1, len(doc)):.0%} duplication", "#f1c21b"),
            ("Answer fact", "intact" if fact_intact else "SPLIT", "turn-1 fact, underlined", "#42be65" if fact_intact else "#fa4d56"),
        ])
        ui.doc_ribbon(doc, index.chunks, span, fact_intact=fact_intact)
    elif node in ("embed", "index"):
        st.plotly_chart(charts.embedding_heatmap(index.vectors, ids), width="stretch")
        pts, _ = pca_2d(index.vectors)
        st.plotly_chart(charts.pca_map(pts, ids, None, None, [], None), width="stretch")
    elif node == "recall_ltm":
        st.markdown("**Loaded long-term profile** (this is all the agent knows about *you* before searching):")
        st.json(cur.state["long_term_memory"])
        st.caption("Without this tier, every conversation starts from zero, and turn 3 would forget the allergy from turn 2.")
    elif node == "retrieve":
        sims = index.scores(query)
        pts, qpt = pca_2d(index.vectors, index.query_vector(query)[None, :])
        top_ids = [c["chunk_id"] for c in cur.state["retrieved_chunks"]]
        a, b = st.columns(2)
        a.plotly_chart(charts.pca_map(pts, ids, sims, qpt[0], top_ids, diag["answer_chunk"]), width="stretch")
        b.plotly_chart(charts.similarity_bars(ids, sims, cfg["top_k"], diag["answer_chunk"]), width="stretch")
        ui.doc_ribbon(doc, index.chunks, span, hit_ids=set(top_ids), fact_intact=fact_intact)
    elif node == "assemble_context":
        ui.prompt_blocks(cur.state["context_window"])
    elif node == "reason":
        meta = cur.state.get("llm_meta", {})
        # Escape the model's text for safety, then allow **bold** through.
        ans = re.sub(r"\*\*(.+?)\*\*", lambda m: f"<b>{m.group(1)}</b>", ui.esc(cur.state["generated_response"]))
        st.markdown(f'<div class="answer">{ans}</div>', unsafe_allow_html=True)
        used = [s["chunk_id"] for s in cur.state["context_window"] if s["tier"] == "chunk" and s["included"]]
        cap = f"Model: {meta.get('model', '?')} · chunks in context: {', '.join(used) or 'none'}"
        if "input_tokens" in meta:
            cap += f" · real usage: {meta['input_tokens']} in / {meta['output_tokens']} out tokens"
        st.caption(cap)
        if meta.get("error"):
            st.warning(f"Claude call failed, fell back to offline: {meta['error']}")
        with st.expander("Exact prompt sent to the model", expanded=False):
            st.code(cur.state["prompt_text"], language="xml")
    elif node == "write_memory":
        w = cur.state.get("memory_writes", [])
        if w:
            for x in w:
                st.success(f"Committed to long-term memory → **{x['category']}**: {x['fact']}  ·  via {x['method']}")
        else:
            st.info("Nothing durable in this message. Long-term memory unchanged. Only short-term history grew.")
        st.markdown(f"**Short-term history now holds {len(cur.state['conversation_history'])} messages.** "
                    "These are replayed into the next turn's context (if the budget allows).")

# -------------------------------------------------------------- STATE DIFF
with right:
    ui.panel_title("State diff · vs previous step")
    prev_state = steps[ss.cursor - 1].state if ss.cursor > 0 else {}
    ui.state_diff(prev_state, cur.state, diff_states(prev_state, cur.state))
    st.markdown('<div class="gauge-legend" style="margin-top:4px"><span style="color:#42be65">+ added</span>'
                '<span style="color:#f1c21b">~ changed</span><span>unchanged</span></div>', unsafe_allow_html=True)
    with st.expander("Full snapshot JSON"):
        st.json(cur.state, expanded=1)


# ================================================== WHY MEMORY MATTERS
st.markdown("<hr>", unsafe_allow_html=True)
ui.panel_title(f"Why memory state matters · diagnosing: “{query}”")

if span is None:
    st.info("No ground-truth answer fact for this question (it's a memory-writing turn or a custom question). "
            "Step to a retrieval turn (1 or 3) to see the health check and What-if Lab.")
else:
    icon, text = VERDICT_STYLE[diag["verdict"]]
    vcol = {"ok": "#42be65", "split": "#fa4d56", "missed": "#fa4d56", "evicted": "#fa4d56", "partial": "#f1c21b"}.get(diag["verdict"], "#8d8d8d")
    ui.tiles([
        ("Verdict", f"{icon} {diag['verdict'].upper()}", text, vcol),
        ("Answer fact intact", f"{diag['coverage']:.0%}", "largest piece of the fact inside one in-context chunk", vcol),
        ("Answer chunk rank", f"#{diag['answer_rank']} of {diag['n_chunks']}", f"{diag['answer_chunk']} · needs ≤ {cfg['top_k']}",
         "#42be65" if diag["answer_rank"] <= cfg["top_k"] else "#fa4d56"),
        ("Top-1 sim · margin", f"{diag['top1']:.2f} · {diag['margin']:+.2f}", "small margin = fragile ranking", TIER_COLORS["chunk"]),
        ("Signal / noise", f"{diag['snr']:.0%}", "share of chunk text that IS the answer", "#be95ff"),
        ("Context used", f"{diag['tokens']}/{cfg['budget']}", "estimated tokens", "#f1c21b"),
    ])

    wl, wr = st.columns([1.35, 1], gap="medium")
    with wl:
        st.markdown("**What-if Lab**: the same question under different memory configurations")
        rows = cached_what_if(doc, query, span, cfg, ltm_at_start, hist_at_start)
        df = pd.DataFrame([{
            "Config": r["name"],
            "Size/overlap": f'{r["cfg"]["chunk_size"]}/{r["cfg"]["chunk_overlap"]}',
            "Embedder": r["cfg"]["embedder"],
            "Chunks": r["n_chunks"],
            "Verdict": VERDICT_SHORT[r["verdict"]],
            "Fact intact": f'{r["coverage"]:.0%}',
            "Rank": r["answer_rank"],
            "Top-1 sim": round(r["top1"], 3),
            "Tokens": r["tokens"],
        } for r in rows])
        st.dataframe(df, hide_index=True, width="stretch")

        def apply(c: dict) -> None:
            """Callback: runs BEFORE the rerun, so the sliders pick up the values."""
            for k in DEFAULTS:
                st.session_state[k] = c[k]

        bcols = st.columns(len(rows) - 1)
        for bc, r in zip(bcols, rows[1:]):
            bc.button(f"Apply ▸ {r['name']}", on_click=apply, args=(r["cfg"],), width="stretch")
        st.caption("Apply a config and the whole conversation is replayed under it. Step through and watch where memory breaks. "
                   "Offline answers are extractive, so a missing chunk produces a visibly wrong or irrelevant answer. "
                   "In turn 3 that can mean no safe antibiotic guidance at all for a penicillin-allergic patient.")
    with wr:
        # The sweep re-chunks and re-embeds ~24 configs, which is slow on CPU the first
        # time for each question. So it's opt-in; once enabled it stays on (and cached).
        if not ss.get("grid_on"):
            st.markdown("**Sensitivity sweep**: try every chunk_size × overlap and see where the answer survives.")
            if st.button("▦ Run sensitivity sweep (≈20–40 s first time per question)", width="stretch"):
                ss.grid_on = True
                st.rerun()
        else:
            grid = cached_grid(doc, query, span, cfg, ltm_at_start, hist_at_start)
            st.plotly_chart(charts.sensitivity_map(grid, cfg["chunk_size"], cfg["chunk_overlap"]), width="stretch")
        st.caption("Green = the full answer fact reaches the LLM. Too small → facts get cut. "
                   "Too big → chunks blow the context budget and get evicted. □ = your current settings (nearest cell).")
