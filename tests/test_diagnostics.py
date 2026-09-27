"""Needle tracking: the verdicts must reflect where memory actually failed."""

from src.data.sample import CLINIC_GUIDE, INITIAL_LTM, SCRIPTED_TURNS, needle_span
from src.rag.context import build_segments, pack_context
from src.rag.diagnostics import evaluate, find_splitting_size
from src.rag.index import VectorIndex

Q1 = SCRIPTED_TURNS[0]["prompt"]
SPAN1 = needle_span(CLINIC_GUIDE, SCRIPTED_TURNS[0]["needle"])


def test_needles_exist_in_document():
    for t in SCRIPTED_TURNS:
        if t["needle"]:
            assert needle_span(CLINIC_GUIDE, t["needle"]) is not None


def test_split_detected_when_boundary_cuts_fact_without_overlap():
    size = find_splitting_size(CLINIC_GUIDE, SPAN1, 400, "recursive")
    idx = VectorIndex.build(CLINIC_GUIDE, size, 0, "recursive", "hashing")
    r = evaluate(idx, Q1, SPAN1, top_k=3, budget=2000, ltm=INITIAL_LTM)
    assert r["verdict"] == "split"
    assert r["coverage"] < 1.0


def test_whole_document_chunk_is_ok_with_big_budget():
    idx = VectorIndex.build(CLINIC_GUIDE, 5000, 0, "recursive", "hashing")
    r = evaluate(idx, Q1, SPAN1, top_k=1, budget=5000, ltm=INITIAL_LTM)
    assert r["verdict"] == "ok" and r["coverage"] == 1.0


def test_eviction_detected_when_budget_too_small():
    idx = VectorIndex.build(CLINIC_GUIDE, 5000, 0, "recursive", "hashing")
    r = evaluate(idx, Q1, SPAN1, top_k=1, budget=200, ltm=INITIAL_LTM)
    assert r["verdict"] == "evicted"


def test_pack_context_keeps_required_and_respects_budget():
    retrieved = [{"chunk_id": f"C{i}", "similarity": 0.5, "snippet": "x" * 400} for i in range(3)]
    segs = pack_context(build_segments(INITIAL_LTM, retrieved, [], "hi"), budget=150)
    assert all(s["included"] for s in segs if s["required"])
    optional_used = sum(s["tokens"] for s in segs if s["included"] and not s["required"])
    required_used = sum(s["tokens"] for s in segs if s["required"])
    assert optional_used + required_used <= 150 or optional_used == 0
    # Higher-ranked chunks survive before lower-ranked ones.
    kept = [s["chunk_id"] for s in segs if s["tier"] == "chunk" and s["included"]]
    assert kept == sorted(kept)
