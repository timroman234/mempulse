"""
Diagnostics: turning "memory state" into a verdict you can act on.

WHAT THIS TEACHES
-----------------
Knowing the answer sentence (the "needle") lets us trace exactly where a RAG
pipeline loses information. There are only four places it can go wrong:

  1. CHUNKING broke it  - no single chunk contains the whole fact. A boundary
                          cut through it and there wasn't enough overlap.
  2. RETRIEVAL missed it- a chunk holds the fact, but the query vector was
                          closer to other chunks, so it ranked below top-k.
  3. CONTEXT evicted it - it was retrieved, but the context budget ran out
                          before it could be packed into the prompt.
  4. OK                 - the whole fact reached the LLM.

The single score used for the heatmap is "intact coverage": the largest
fraction of the needle found inside ONE chunk that made it into the prompt.
1.0 = the LLM sees the complete fact; 0.5 = it sees half a dosage sentence.
"""

from __future__ import annotations

import numpy as np

from src.rag.context import build_segments, pack_context
from src.rag.index import VectorIndex

# Short labels for the What-if table.
VERDICT_SHORT = {
    "ok": "✅ intact in context",
    "split": "✂️ split by chunker",
    "missed": "🎯 missed by retrieval",
    "evicted": "🚫 evicted from context",
    "partial": "⚠️ only partly in context",
    "n/a": "·",
}

VERDICT_STYLE = {
    "ok": ("✅", "Answer fact reached the LLM intact"),
    "split": ("✂️", "Chunking broke the fact across chunks"),
    "missed": ("🎯", "Retrieval ranked the right chunk below top-k"),
    "evicted": ("🚫", "Retrieved, but evicted by the context budget"),
    "partial": ("⚠️", "Only part of the fact reached the LLM"),
    "n/a": ("·", "No ground-truth fact for this question"),
}


def _fraction_inside(span: tuple[int, int], start: int, end: int) -> float:
    """What fraction of the needle span lies inside [start, end)?"""
    a, b = span
    inter = max(0, min(b, end) - max(a, start))
    return inter / (b - a)


def evaluate(
    index: VectorIndex,
    query: str,
    span: tuple[int, int] | None,
    top_k: int,
    budget: int,
    ltm: dict,
    history: list[dict] | None = None,
) -> dict:
    """Run retrieval + context packing and classify what happened to the needle."""
    sims = index.scores(query)
    order = list(np.argsort(-sims))
    retrieved = index.search(query, top_k)
    segs = pack_context(build_segments(ltm, retrieved, history or [], query), budget)
    in_context = {s["chunk_id"] for s in segs if s["tier"] == "chunk" and s["included"]}
    used = sum(s["tokens"] for s in segs if s["included"])

    top1 = float(sims[order[0]]) if order else 0.0
    top2 = float(sims[order[1]]) if len(order) > 1 else 0.0
    result = {
        "top1": top1,
        "margin": top1 - top2,
        "tokens": used,
        "n_chunks": len(index.chunks),
        "retrieved_ids": [r["chunk_id"] for r in retrieved],
        "in_context_ids": sorted(in_context),
        "verdict": "n/a", "coverage": None, "answer_rank": None, "answer_chunk": None, "snr": None,
    }
    if span is None:
        return result

    # Per chunk: how much of the needle it holds (1.0 = the whole fact).
    frac = [_fraction_inside(span, c.start, c.end) for c in index.chunks]
    best_i = int(np.argmax(frac))
    rank_of = {i: r + 1 for r, i in enumerate(order)}
    whole = [i for i, f in enumerate(frac) if f >= 0.999]

    # The "answer chunk" is the best-ranked chunk holding the whole fact, or
    # failing that, the chunk holding the biggest piece of it.
    ans_i = min(whole, key=lambda i: rank_of[i]) if whole else best_i
    coverage = max([frac[i] for i, c in enumerate(index.chunks) if c.chunk_id in in_context] or [0.0])

    # Signal-to-noise: how much of the chunk text in the prompt is the fact itself.
    ctx_chars = sum(len(c.text) for c in index.chunks if c.chunk_id in in_context)
    needle_len = span[1] - span[0]
    snr = min(1.0, needle_len * coverage / ctx_chars) if ctx_chars else 0.0

    retrieved_ids = set(result["retrieved_ids"])
    ans_id = index.chunks[ans_i].chunk_id
    if coverage >= 0.999:
        verdict = "ok"
    elif not whole:
        verdict = "split"
    elif ans_id in retrieved_ids and ans_id not in in_context:
        verdict = "evicted"
    elif coverage > 0:
        verdict = "partial"
    else:
        verdict = "missed"

    result.update(
        verdict=verdict, coverage=coverage, answer_rank=rank_of[ans_i],
        answer_chunk=ans_id, snr=snr,
    )
    return result


def find_splitting_size(doc: str, span: tuple[int, int], around: int, strategy: str) -> int:
    """Find a chunk size near `around` where, with ZERO overlap, a boundary
    lands inside the needle. Used by the 'No overlap' what-if so the demo
    reliably shows the failure (and says honestly how it was chosen)."""
    from src.rag.chunker import chunk_document

    for delta in range(0, 400, 5):
        for size in (around + delta, around - delta):
            if size < 120:
                continue
            chunks = chunk_document(doc, size, 0, strategy)
            if not any(_fraction_inside(span, c.start, c.end) >= 0.999 for c in chunks):
                return size
    return around


def what_if(doc: str, query: str, span, cfg: dict, ltm: dict, history: list[dict] | None = None) -> list[dict]:
    """Same question, several memory configurations, side by side."""
    presets = [
        ("Current", dict(cfg)),
        ("Tiny", {**cfg, "chunk_size": 80, "chunk_overlap": 0}),
    ]
    if span is not None:
        size = find_splitting_size(doc, span, cfg["chunk_size"], cfg["strategy"])
        presets.append(("No overlap", {**cfg, "chunk_size": size, "chunk_overlap": 0}))
    presets += [
        ("Giant", {**cfg, "chunk_size": 2000, "chunk_overlap": 200}),
        ("Hashing", {**cfg, "embedder": "hashing"}),
    ]
    rows = []
    for name, c in presets:
        idx = VectorIndex.build(doc, c["chunk_size"], c["chunk_overlap"], c["strategy"], c["embedder"])
        r = evaluate(idx, query, span, c["top_k"], c["budget"], ltm, history)
        rows.append({"name": name, "cfg": c, **r})
    return rows


SENS_SIZES = [100, 150, 200, 300, 400, 600, 1000, 2000]
SENS_OVERLAPS = [0, 50, 100]


def sensitivity_grid(doc: str, query: str, span, cfg: dict, ltm: dict, history: list[dict] | None = None) -> np.ndarray:
    """Intact-coverage score for every (chunk_size, overlap) combination.

    Rows = overlaps, columns = sizes. Embeddings are cached per chunk text, so
    this is ~40 cheap re-chunks rather than 40 full embedding runs.
    """
    grid = np.zeros((len(SENS_OVERLAPS), len(SENS_SIZES)))
    for r, ov in enumerate(SENS_OVERLAPS):
        for c, size in enumerate(SENS_SIZES):
            idx = VectorIndex.build(doc, size, ov, cfg["strategy"], cfg["embedder"])
            res = evaluate(idx, query, span, cfg["top_k"], cfg["budget"], ltm, history)
            grid[r, c] = res["coverage"] or 0.0
    return grid
