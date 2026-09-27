"""
Reusable HTML widgets for MemPulse.

WHAT THIS TEACHES (UI-side)
---------------------------
Each widget is one way of *looking at memory*:
  pipeline_strip  - where we are in the flow (time)
  memory_stack    - how full each memory tier is right now (space)
  context_gauge   - how the finite prompt budget is being spent
  doc_ribbon      - how the chunker carved up the source (boundaries/overlap)
  prompt_blocks   - the literal prompt, coloured by which memory it came from
  state_diff      - what the last node changed

Implementation note: everything is emitted as ONE line of HTML via
st.markdown(unsafe_allow_html=True). A blank line inside an HTML block would
make Streamlit's markdown parser leave HTML mode, so newlines become <br>.
"""

from __future__ import annotations

import html
import json

import streamlit as st

from src.rag.context import TIER_COLORS

# Alternating chunk tints for the ribbon (Carbon blue / purple / teal at low alpha).
CHUNK_TINTS = ["rgba(69,137,255,0.20)", "rgba(190,149,255,0.20)", "rgba(8,189,186,0.18)"]


def _md(h: str) -> None:
    st.markdown(h, unsafe_allow_html=True)


def esc(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")


def panel_title(text: str) -> None:
    _md(f'<div class="panel-title">{esc(text)}</div>')


# ------------------------------------------------------------ pipeline strip

def pipeline_strip(groups: list[tuple[str, list[tuple[str, str]]]], active_key: str, done_keys: set[str]) -> None:
    """groups = [(group label, [(step key, node label), ...]), ...]"""
    parts = ['<div class="pipe">']
    for g_label, nodes in groups:
        parts.append(f'<span class="grp">{esc(g_label)}</span>')
        for i, (key, label) in enumerate(nodes):
            cls = "active" if key == active_key else ("done" if key in done_keys else "")
            parts.append(f'<span class="node {cls}">{esc(label)}</span>')
            if i < len(nodes) - 1:
                parts.append('<span class="arrow">▶</span>')
    parts.append("</div>")
    _md("".join(parts))


# ---------------------------------------------------------- memory tier card

def tier_card(title: str, color: str, count_label: str, fill: float, items_html: str, pulse: bool = False) -> None:
    fill = max(0.0, min(1.0, fill))
    _md(
        f'<div class="tier{" pulse" if pulse else ""}" style="--c:{color}">'
        f'<div class="hd"><span>{esc(title)}</span><span class="n">{esc(count_label)}</span></div>'
        f'<div class="bar"><div style="width:{fill * 100:.0f}%"></div></div>'
        f'<div class="items">{items_html}</div></div>'
    )


def context_gauge(segments: list[dict], budget: int) -> None:
    """Stacked bar of included tokens by tier, scaled to the budget."""
    used = {k: 0 for k in TIER_COLORS}
    for s in segments:
        if s["included"]:
            used[s["tier"]] += s["tokens"]
    total = sum(used.values())
    scale = max(budget, total)
    bars = "".join(
        f'<div title="{k}: {v} tok" style="width:{v / scale * 100:.2f}%;background:{TIER_COLORS[k]}"></div>'
        for k, v in used.items() if v
    )
    legend = "".join(
        f'<span><i style="background:{TIER_COLORS[k]}"></i>{k} {v}</span>' for k, v in used.items() if v
    )
    evicted = [s for s in segments if not s["included"]]
    ev_html = ""
    if evicted:
        ev_html = '<div class="evicted">⛔ evicted: ' + ", ".join(
            f'{esc(s["label"])} ({s["tokens"]} tok)' for s in evicted) + "</div>"
    pct = total / budget * 100 if budget else 0
    _md(
        f'<div class="tier" style="--c:#f1c21b"><div class="hd"><span>Context window (prompt)</span>'
        f'<span class="n">{total}/{budget} tok · {pct:.0f}%</span></div>'
        f'<div class="gauge">{bars}</div><div class="gauge-legend">{legend}</div>{ev_html}</div>'
    )


# ------------------------------------------------------------ document ribbon

def doc_ribbon(doc: str, chunks: list, needle: tuple[int, int] | None, hit_ids: set[str] | None = None,
               fact_intact: bool = True) -> None:
    """Render the source text with chunk tints, overlap in yellow, the answer
    fact underlined, and ✂ where a chunk boundary falls inside the fact."""
    hit_ids = hit_ids or set()
    # Every position where "what covers this char" can change.
    cuts = {0, len(doc)}
    for c in chunks:
        cuts.update((c.start, c.end))
    if needle:
        cuts.update(needle)
    cuts = sorted(x for x in cuts if 0 <= x <= len(doc))
    starts = {c.start: c for c in chunks}
    ends_in_needle = {c.end for c in chunks if needle and needle[0] < c.end < needle[1]}

    out = ['<div class="ribbon">']
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a in starts:
            c = starts[a]
            out.append(f'<span class="tag{" hit" if c.chunk_id in hit_ids else ""}">{c.chunk_id}</span>')
        covering = [i for i, c in enumerate(chunks) if c.start <= a < c.end]
        seg = esc(doc[a:b])
        classes = ["ck"]
        style = ""
        if len(covering) >= 2:
            classes.append("ov")          # overlap: text stored in 2+ chunks
        elif covering:
            idx = covering[0]
            tint = CHUNK_TINTS[idx % len(CHUNK_TINTS)]
            if chunks[idx].chunk_id in hit_ids:
                tint = tint.replace("0.20", "0.45").replace("0.18", "0.42")
            style = f"background:{tint}"
        if needle and needle[0] <= a < needle[1]:
            classes.append("needle")
        out.append(f'<span class="{" ".join(classes)}" style="{style}">{seg}</span>')
        if b in ends_in_needle:
            # Red = the fact is broken everywhere; amber = overlap rescued it.
            if fact_intact:
                out.append('<span class="cut" style="background:#684600;color:#f1c21b" title="boundary inside fact, but overlap kept a whole copy">✂</span>')
            else:
                out.append('<span class="cut" title="boundary cuts the answer fact: no chunk holds it whole">✂</span>')
    out.append("</div>")
    _md("".join(out))


# -------------------------------------------------------------- prompt view

def prompt_blocks(segments: list[dict]) -> None:
    parts = []
    for s in segments:
        c = TIER_COLORS[s["tier"]]
        txt = s["text"] if len(s["text"]) < 420 else s["text"][:420] + "…"
        state = "" if s["included"] else " out"
        flag = "" if s["included"] else " ⛔ EVICTED"
        parts.append(
            f'<div class="pblock{state}" style="--c:{c}"><div class="lbl">{esc(s["tier"].upper())} · '
            f'{esc(s["label"])} · {s["tokens"]} tok{flag}</div>{esc(txt)}</div>'
        )
    _md("".join(parts))


# --------------------------------------------------------------- state diff

def _short(v) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= 70 else s[:67] + "…"


def state_diff(prev: dict, cur: dict, diff: list[dict]) -> None:
    """One line per key: + added (green), ~ changed (amber), unchanged (grey)."""
    kinds = {d["key"]: d for d in diff}
    lines = []
    for key in cur:
        d = kinds.get(key)
        if d and d["kind"] == "added":
            lines.append(f'<div class="add">+ {esc(key)}: {esc(_short(cur[key]))}</div>')
        elif d and d["kind"] == "changed":
            note = f' <span class="same">({esc(d["note"])})</span>' if d.get("note") else ""
            lines.append(f'<div class="chg">~ {esc(key)}: {esc(_short(cur[key]))}{note}</div>')
        else:
            lines.append(f'<div class="same">&nbsp; {esc(key)}: {esc(_short(cur[key]))}</div>')
    for d in diff:
        if d["kind"] == "removed":
            lines.append(f'<div class="rem">− {esc(d["key"])}</div>')
    _md('<div class="diff">' + "".join(lines) + "</div>")


# -------------------------------------------------------------- health tiles

def tiles(items: list[tuple[str, str, str, str]]) -> None:
    """items = [(label, value, subtext, color), ...]"""
    _md('<div class="tiles">' + "".join(
        f'<div class="tile" style="--c:{c}"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
        f'<div class="s">{esc(s)}</div></div>' for k, v, s, c in items) + "</div>")
