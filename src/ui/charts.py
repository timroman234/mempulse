"""
Plotly charts: the geometric view of memory.

WHAT THIS TEACHES
-----------------
  embedding_heatmap  - vectors are just rows of numbers. Similar chunks show
                       similar colour patterns.
  pca_map            - chunks as points in 2-D. The query ★ lands near the
                       chunks it will retrieve. Lines connect it to its top-k.
  similarity_bars    - the actual ranking with the top-k cut-off line. The
                       gap between the winner and the runner-up (margin) says
                       how *confident* retrieval is.
  sensitivity_map    - how the answer's survival depends on chunk_size ×
                       overlap. It shows a "safe zone", not a single magic value.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.rag.diagnostics import SENS_OVERLAPS, SENS_SIZES

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#262626",
    font=dict(family="IBM Plex Sans, sans-serif", size=11, color="#c6c6c6"),
    title_font=dict(size=13),
    margin=dict(l=10, r=10, t=30, b=10),
)


def embedding_heatmap(vectors: np.ndarray, ids: list[str], dims: int = 64) -> go.Figure:
    v = vectors[:, :dims]
    fig = go.Figure(go.Heatmap(
        z=v, y=ids, colorscale="RdBu", zmid=0, showscale=False,
        hovertemplate="%{y} · dim %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT, height=260, title=f"Vector store · first {dims}/{vectors.shape[1]} dims",
                      xaxis_title="dimension", yaxis=dict(autorange="reversed"))
    return fig


def pca_map(points: np.ndarray, ids: list[str], sims: np.ndarray | None, qpt: np.ndarray | None,
            top_ids: list[str], answer_id: str | None) -> go.Figure:
    fig = go.Figure()
    # Lines from the query to each top-k chunk: "these are what got pulled into memory".
    if qpt is not None:
        for tid in top_ids:
            i = ids.index(tid)
            fig.add_trace(go.Scatter(x=[qpt[0], points[i, 0]], y=[qpt[1], points[i, 1]], mode="lines",
                                     line=dict(color="#4589ff", width=1.5, dash="dot"), hoverinfo="skip", showlegend=False))
    color = sims if sims is not None else np.zeros(len(ids))
    sizes = [16 if i in top_ids else 10 for i in ids]
    lines = [3 if i == answer_id else 0 for i in ids]
    fig.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1], mode="markers+text", text=ids, textposition="top center",
        marker=dict(size=sizes, color=color, colorscale="Blues", cmin=float(np.min(color)) if sims is not None else 0,
                    cmax=float(np.max(color)) if sims is not None else 1,
                    line=dict(width=lines, color="#ffb000")),
        hovertemplate="%{text}<br>sim %{marker.color:.3f}<extra></extra>", showlegend=False,
    ))
    if qpt is not None:
        fig.add_trace(go.Scatter(x=[qpt[0]], y=[qpt[1]], mode="markers+text", text=["query"],
                                 textposition="bottom center",
                                 marker=dict(symbol="star", size=20, color="#f1c21b"), showlegend=False))
    fig.update_layout(**_LAYOUT, height=300, title="Vector space (PCA 2-D) · ○ gold = answer",
                      xaxis=dict(showticklabels=False, zeroline=False), yaxis=dict(showticklabels=False, zeroline=False))
    return fig


def vector_space_3d(points: np.ndarray, ids: list[str], sims: np.ndarray | None, qpt: np.ndarray | None,
                    top_ids: list[str], answer_id: str | None) -> go.Figure:
    """Chunks and query as ARROWS from the origin in a 3-D PCA projection.

    Why arrows? Cosine similarity only cares about DIRECTION, not length. Two
    arrows pointing the same way = similar meaning, no matter how long. So we
    scale every arrow to unit length: all tips sit on a sphere, and the only
    thing left to see is the angle θ between arrows (cos θ = similarity).

    Honesty note: this squeezes 384 dimensions into 3, so on-screen angles are
    approximate. The hover shows the TRUE 384-D cosine and angle.
    """
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n else v

    fig = go.Figure()
    arrows = []  # (id, tip xyz, colour, width, hover text)
    for i, cid in enumerate(ids):
        if qpt is None:
            # Ingest steps: no question yet, so nothing is "retrieved" or "left
            # behind". Every chunk is simply a stored vector.
            col, w = "#a6c8ff", 3
        elif cid == answer_id:
            col, w = "#ffb000", 6        # gold = the chunk holding the answer
        elif cid in top_ids:
            col, w = "#42be65", 5        # green = retrieved into memory
        else:
            col, w = "#6f6f6f", 2        # grey = left behind
        hover = cid
        if sims is not None:
            theta = np.degrees(np.arccos(np.clip(sims[i], -1, 1)))
            hover += f"<br>cos = {sims[i]:.3f} · θ = {theta:.1f}° (true 384-D)"
        arrows.append((cid, unit(points[i]), col, w, hover))
    if qpt is not None:
        arrows.append(("query", unit(qpt), "#4589ff", 8, "query embedding"))

    for cid, tip, col, w, hover in arrows:
        # Shaft: a line from the origin to the tip.
        fig.add_trace(go.Scatter3d(x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]], mode="lines",
                                   line=dict(color=col, width=w), hoverinfo="skip", showlegend=False))
        # Head: a small cone at the tip pointing outward, plus a label.
        fig.add_trace(go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[tip[0]], v=[tip[1]], w=[tip[2]],
                              anchor="tip", sizemode="absolute", sizeref=0.18, showscale=False,
                              colorscale=[[0, col], [1, col]], hoverinfo="skip"))
        # Before any question is asked, label every chunk; afterwards only the important ones.
        big = qpt is None or cid == "query" or col != "#6f6f6f"
        fig.add_trace(go.Scatter3d(x=[tip[0] * 1.08], y=[tip[1] * 1.08], z=[tip[2] * 1.08], mode="text",
                                   text=[cid if big else ""], hovertext=[hover], hoverinfo="text",
                                   textfont=dict(color=col, size=12 if big else 9), showlegend=False))

    # Legend entries (dummy traces) so the colours are self-explanatory.
    legend = ([("stored chunk (no question asked yet)", "#a6c8ff")] if qpt is None else
              [("query", "#4589ff"), ("retrieved (top-k)", "#42be65"), ("answer chunk", "#ffb000"), ("not retrieved", "#6f6f6f")])
    for name, col in legend:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line=dict(color=col, width=6), name=name))

    axis = dict(showticklabels=False, title="", backgroundcolor="#262626", gridcolor="#393939",
                zerolinecolor="#6f6f6f", range=[-1.15, 1.15], showspikes=False)
    fig.update_layout(**_LAYOUT, height=400, title="Vector space 3-D · drag to rotate" if qpt is not None else "Stored vectors 3-D · drag to rotate",
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube",
                                 camera=dict(eye=dict(x=1.15, y=1.15, z=0.75))),
                      legend=dict(orientation="h", x=0, y=1.0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    # Extra headroom: Plotly's hover toolbar sits in the top-right corner and
    # would otherwise cover the title. Push the title down under the toolbar
    # and the 3-D scene below the legend.
    fig.update_layout(margin=dict(l=10, r=10, t=78, b=10),
                      title=dict(y=1 - 40 / 400, yanchor="top"))
    fig.update_scenes(domain=dict(y=[0.0, 0.9]))
    return fig


def similarity_bars(ids: list[str], sims: np.ndarray, top_k: int, answer_id: str | None) -> go.Figure:
    order = np.argsort(-sims)
    ys = [ids[i] for i in order]
    xs = [float(sims[i]) for i in order]
    colors = ["#ffb000" if ys[r] == answer_id else ("#42be65" if r < top_k else "#525252") for r in range(len(ys))]
    fig = go.Figure(go.Bar(x=xs, y=ys, orientation="h", marker_color=colors,
                           text=[f"{x:.3f}" for x in xs], textposition="outside",
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    # Horizontal separator after rank k: everything below never enters memory.
    fig.add_hline(y=top_k - 0.5, line=dict(color="#fa4d56", dash="dash"),
                  annotation_text=f"top-{top_k} cut-off", annotation_position="bottom right",
                  annotation_font_color="#fa4d56", annotation_font_size=10)
    height = max(268, 22 * len(ys) + 108)
    fig.update_layout(**_LAYOUT, height=height,
                      title="Cosine similarity · gold = answer",
                      yaxis=dict(autorange="reversed"), xaxis=dict(range=[min(0, min(xs)) - 0.05, 1.3]))
    # Same headroom as the 3-D chart: keep the title below Plotly's hover
    # toolbar (title top sits 40 px down, like the 3-D chart beside it).
    fig.update_layout(margin=dict(l=10, r=10, t=78, b=10),
                      title=dict(y=1 - 40 / height, yanchor="top"))
    return fig


def sensitivity_map(grid: np.ndarray, cur_size: int, cur_overlap: int) -> go.Figure:
    labels = [[f"{v:.0%}" for v in row] for row in grid]
    fig = go.Figure(go.Heatmap(
        z=grid, x=[str(s) for s in SENS_SIZES], y=[str(o) for o in SENS_OVERLAPS], zmin=0, zmax=1,
        colorscale=[[0, "#750e13"], [0.5, "#8e6a00"], [0.99, "#198038"], [1, "#24a148"]],
        text=labels, texttemplate="%{text}", showscale=False,
        hovertemplate="size %{x} · overlap %{y}<br>answer intact in context: %{z:.0%}<extra></extra>",
    ))
    # "You are here" marker, snapped to the nearest grid cell.
    xi = int(np.argmin([abs(s - cur_size) for s in SENS_SIZES]))
    yi = int(np.argmin([abs(o - cur_overlap) for o in SENS_OVERLAPS]))
    fig.add_trace(go.Scatter(x=[str(SENS_SIZES[xi])], y=[str(SENS_OVERLAPS[yi])], mode="markers",
                             marker=dict(symbol="square-open", size=34, color="#ffffff", line=dict(width=3)),
                             hoverinfo="skip", showlegend=False))
    fig.update_layout(**_LAYOUT, height=260, title="Sensitivity · % of answer fact reaching the LLM",
                      xaxis_title="chunk_size (chars)", yaxis_title="overlap")
    return fig
