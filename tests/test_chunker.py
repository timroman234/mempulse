"""Chunk boundaries must map exactly back onto the source text."""

import pytest

from src.data.sample import CLINIC_GUIDE
from src.rag.chunker import chunk_document


@pytest.mark.parametrize("strategy", ["recursive", "heading"])
@pytest.mark.parametrize("size,overlap", [(100, 0), (400, 60), (800, 150), (2000, 200)])
def test_char_ranges_reconstruct_source(strategy, size, overlap):
    chunks = chunk_document(CLINIC_GUIDE, size, overlap, strategy)
    assert chunks, "document should produce at least one chunk"
    for c in chunks:
        # The stored text must be exactly the slice its range points to.
        assert CLINIC_GUIDE[c.start:c.end] == c.text
    # Chunks appear in document order.
    assert [c.start for c in chunks] == sorted(c.start for c in chunks)


def test_recursive_chunks_respect_size():
    for c in chunk_document(CLINIC_GUIDE, 300, 50):
        assert len(c.text) <= 300


def test_overlap_reported_and_zero_when_disabled():
    with_ov = chunk_document(CLINIC_GUIDE, 300, 80)
    no_ov = chunk_document(CLINIC_GUIDE, 300, 0)
    assert any(c.overlap_prev > 0 for c in with_ov)
    assert all(c.overlap_prev == 0 for c in no_ov)
    for prev, cur in zip(with_ov, with_ov[1:]):
        assert cur.overlap_prev == max(0, prev.end - cur.start)


def test_changing_size_changes_chunk_count():
    assert len(chunk_document(CLINIC_GUIDE, 150, 0)) > len(chunk_document(CLINIC_GUIDE, 1000, 0))
