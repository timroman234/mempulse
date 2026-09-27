"""
Trace: the timeline the UI scrubs through.

WHAT THIS TEACHES
-----------------
Debugging memory means asking "what did memory look like at step N, and what
changed since step N-1?". So we record a flat list of Steps:

    ingest: load → chunk → embed → index        (once per document/config)
    turn 1: recall → retrieve → assemble → reason → write
    turn 2: ...

Each Step holds a full snapshot of the state at that moment. The UI never
recomputes when you press Prev/Next. It just indexes into this list, which
is exactly how "time-travel debugging" works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.index import VectorIndex

INGEST_NODES = ["load", "chunk", "embed", "index"]
INGEST_LABELS = {"load": "Load", "chunk": "Chunk", "embed": "Embed", "index": "Index"}
INGEST_EXPLAIN = {
    "load": "The raw document enters memory as one long string. Nothing is searchable yet.",
    "chunk": "The splitter cuts the document into chunks. Boundaries decide which facts stay whole. Yellow = overlap, gold underline = answer facts.",
    "embed": "Each chunk becomes a 384-number vector. Similar meaning → similar direction.",
    "index": "Vectors are stored in the index: the agent's searchable knowledge memory. Ready for questions.",
}


@dataclass
class Step:
    phase: str                 # "ingest" or "turn"
    node: str                  # node key, e.g. "chunk" or "retrieve"
    turn: int                  # 0 for ingest
    state: dict = field(default_factory=dict)


def build_ingest_steps(doc: str, index: VectorIndex) -> list[Step]:
    """Snapshots for the ingestion pipeline. The 'state' here is the ingestion
    memory: what exists after each stage."""
    n, d = index.vectors.shape if index.vectors.size else (0, 0)
    doc_info = {"chars": len(doc), "est_tokens": len(doc) // 4, "preview": doc[:120] + "…"}
    chunk_list = [{"chunk_id": c.chunk_id, "char_range": [c.start, c.end], "len": len(c.text),
                   "overlap_prev": c.overlap_prev} for c in index.chunks]
    return [
        Step("ingest", "load", 0, {"active_node": "load", "document": doc_info}),
        Step("ingest", "chunk", 0, {"active_node": "chunk", "document": doc_info, "chunks": chunk_list}),
        Step("ingest", "embed", 0, {"active_node": "embed", "document": doc_info, "chunks": chunk_list,
                                     "vectors": {"shape": [n, d], "embedder": index.embedder_name,
                                                 "C00_preview": [round(float(x), 4) for x in index.vectors[0][:8]] if n else []}}),
        Step("ingest", "index", 0, {"active_node": "index", "document": doc_info, "chunks": chunk_list,
                                     "vectors": {"shape": [n, d], "embedder": index.embedder_name},
                                     "index": {"type": "exact cosine (brute force)", "size": n, "ready": True}}),
    ]


def diff_states(prev: dict, cur: dict) -> list[dict]:
    """Top-level key diff between two snapshots.

    kind: "added" (key new), "changed" (value differs), "removed".
    For list values we also report how many items were appended, since
    'memory grew by 1 fact' is more useful than 'list changed'.
    """
    out = []
    for key in cur:
        if key not in prev:
            out.append({"key": key, "kind": "added"})
        elif cur[key] != prev[key]:
            note = ""
            a, b = prev[key], cur[key]
            if isinstance(a, list) and isinstance(b, list):
                note = f"{len(a)} → {len(b)} items"
            elif isinstance(a, dict) and isinstance(b, dict):
                sub = [k for k in b if a.get(k) != b.get(k)]
                note = "changed: " + ", ".join(sub) if sub else ""
            out.append({"key": key, "kind": "changed", "note": note})
    for key in prev:
        if key not in cur:
            out.append({"key": key, "kind": "removed"})
    return out
