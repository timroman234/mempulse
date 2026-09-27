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


def similarity_bars(ids: list[str], sims: np.ndarray, top_k: int, answer_id: str | None) -> go.Figure:
    order = np.argsort(-sims)
    ys = [ids[i] for i in order]
    xs = [float(sims[i]) for i in order]
    colors = ["#ffb000" if ys[r] == answer_id else ("#0f62fe" if r < top_k else "#525252") for r in range(len(ys))]
    fig = go.Figure(go.Bar(x=xs, y=ys, orientation="h", marker_color=colors,
                           text=[f"{x:.3f}" for x in xs], textposition="outside",
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    # Horizontal separator after rank k: everything below never enters memory.
    fig.add_hline(y=top_k - 0.5, line=dict(color="#fa4d56", dash="dash"),
                  annotation_text=f"top-{top_k} cut-off", annotation_position="bottom right",
                  annotation_font_color="#fa4d56", annotation_font_size=10)
    fig.update_layout(**_LAYOUT, height=max(220, 22 * len(ys) + 60),
                      title="Cosine similarity · gold = answer",
                      yaxis=dict(autorange="reversed"), xaxis=dict(range=[min(0, min(xs)) - 0.05, 1.05]))
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
