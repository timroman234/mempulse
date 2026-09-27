"""Vector math: cosine bounds and top-k ordering (PRD §9.1 item 4).

Uses the hashing embedder so tests run offline and instantly.
"""

import numpy as np

from src.data.sample import CLINIC_GUIDE
from src.rag.embeddings import HashingEmbedder, cosine, cosine_matrix, pca_2d
from src.rag.index import VectorIndex


def test_cosine_bounds_and_identity():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.normal(size=384), rng.normal(size=384)
        assert -1.0 <= cosine(a, b) <= 1.0
    v = rng.normal(size=16)
    assert abs(cosine(v, v) - 1.0) < 1e-9
    assert abs(cosine(v, -v) + 1.0) < 1e-9
    assert cosine(np.zeros(4), v[:4]) == 0.0


def test_cosine_matrix_matches_scalar():
    rng = np.random.default_rng(1)
    m, q = rng.normal(size=(5, 8)), rng.normal(size=8)
    np.testing.assert_allclose(cosine_matrix(q, m), [cosine(q, r) for r in m], atol=1e-6)


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed_query("ibuprofen dose"), e.embed_query("ibuprofen dose")
    np.testing.assert_array_equal(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6


def test_search_sorted_descending_and_in_range():
    idx = VectorIndex.build(CLINIC_GUIDE, 300, 50, "recursive", "hashing")
    hits = idx.search("antibiotic for sinusitis", 5)
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in sims)


def test_pca_shapes():
    m = np.random.default_rng(2).normal(size=(6, 20))
    pts, q = pca_2d(m, m[:1])
    assert pts.shape == (6, 2) and q.shape == (1, 2)


def test_pca_3d_shapes_and_padding():
    m = np.random.default_rng(3).normal(size=(6, 20))
    pts, q = pca_2d(m, m[:1], n=3)
    assert pts.shape == (6, 3) and q.shape == (1, 3)
    # Only 2 chunks → at most 1 real component; the rest is zero-padded.
    pts2, q2 = pca_2d(m[:2], m[:1], n=3)
    assert pts2.shape == (2, 3) and q2.shape == (1, 3)
