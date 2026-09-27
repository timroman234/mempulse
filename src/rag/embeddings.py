"""
Embeddings: how text becomes a point in space, and how "relevance" is measured.

WHAT THIS TEACHES
-----------------
An embedding model maps text → a vector of numbers. Texts with similar
*meaning* should land close together. Retrieval is then just geometry: embed
the question, find the chunk vectors pointing in the most similar direction.

We ship two embedders so you can *see* the difference quality makes:

  FastEmbedder   - BAAI/bge-small-en-v1.5 (384 dims), a real neural model run
                   locally with ONNX via `fastembed`. It knows "painkiller" ≈
                   "analgesic".
  HashingEmbedder- a bag-of-words "hashing trick" (also 384 dims). Each word is
                   hashed into a bucket. It only matches *exact words*, so a
                   paraphrased question can miss the right chunk entirely.
                   This is a stand-in for any weak / mismatched embedding.

Similarity metric (PRD §5.2):
    cos(q, c) = (q · c) / (‖q‖ ‖c‖)      always in [-1, 1]
Direction matters, length doesn't. That's why long and short chunks can be
compared fairly.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

EMBEDDER_LABELS = {
    "bge-small": "bge-small-en-v1.5 (semantic, 384D)",
    "hashing": "Hashing bag-of-words (lexical, 384D)",
}

# Words carrying no topic signal. Removing them keeps the hashing embedder
# from matching on "the", "can", "I"...
_STOPWORDS = set(
    """a an the and or of to in on for with is are be it its this that as at by
    from i i'm im me my you your we our us can do does should what which how
    when who whom will would could may might not no if so than then there
    their they them any all every about into also just only per""".split()
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, written out so the math is visible."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0  # an all-zero vector has no direction: treat as unrelated
    return float(np.dot(a, b) / denom)


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query against every row of `matrix` at once."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = 1.0
    return np.clip(matrix @ query / norms, -1.0, 1.0)


class HashingEmbedder:
    """The 'bad' embedder: exact-word overlap only, no notion of meaning."""

    name = "hashing"
    dim = 384

    def _tokens(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [w for w in words if w not in _STOPWORDS and len(w) > 1]

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            # md5, not Python's hash(): hash() is randomized per process, which
            # would make vectors change between app reloads.
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            sign = 1.0 if (h >> 20) & 1 else -1.0  # signed hashing reduces collisions' bias
            v[h % self.dim] += sign
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


class FastEmbedder:
    """The 'good' embedder: a real semantic model, cached per text.

    bge models are *asymmetric*: queries get an instruction prefix
    ("Represent this sentence for searching relevant passages: ") while
    passages do not. fastembed's `query_embed` adds it for us. Mixing these
    up is a classic silent RAG bug.
    """

    name = "bge-small"
    dim = 384
    model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        from fastembed import TextEmbedding  # imported lazily: first load downloads ~70 MB

        self._model = TextEmbedding(self.model_name)
        # Cache so re-chunking the same text (What-if Lab, sensitivity grid)
        # doesn't recompute vectors. Keyed by exact chunk text.
        self._doc_cache: dict[str, np.ndarray] = {}
        self._query_cache: dict[str, np.ndarray] = {}

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # Sort by length and use small batches: a transformer pads every text in
        # a batch to the longest one, so mixing a 2,000-char chunk with 80-char
        # chunks in one big batch wastes a huge amount of compute on CPU.
        missing = sorted((t for t in dict.fromkeys(texts) if t not in self._doc_cache), key=len)
        if missing:
            for t, v in zip(missing, self._model.passage_embed(missing, batch_size=8)):
                self._doc_cache[t] = np.asarray(v, dtype=np.float32)
        if not texts:
            return np.zeros((0, self.dim))
        return np.vstack([self._doc_cache[t] for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._query_cache:
            self._query_cache[text] = np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)
        return self._query_cache[text]


@lru_cache(maxsize=None)
def get_embedder(name: str):
    """One shared instance per embedder type (the model loads only once)."""
    return FastEmbedder() if name == "bge-small" else HashingEmbedder()


def pca_2d(matrix: np.ndarray, extra: np.ndarray | None = None, n: int = 2) -> tuple[np.ndarray, np.ndarray | None]:
    """Project 384-D vectors down to 2-D (or n-D, e.g. 3) so humans can look at them.

    PCA finds the two directions along which the chunk vectors vary most. We fit
    on the chunks only, then project the query with the *same* transform, so the
    query star lands in the same coordinate system. 2-D is a lossy view: nearby
    on screen usually means similar, but trust the cosine bars for the truth.
    """
    if len(matrix) < 2:
        pts = np.zeros((len(matrix), n))
        return pts, (np.zeros((len(extra), n)) if extra is not None else None)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # SVD: rows of vt are the principal directions, sorted by variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:n].T
    pts = centered @ comps
    if pts.shape[1] < n:  # fewer chunks than requested dims: pad with zeros
        pad = n - pts.shape[1]
        pts = np.pad(pts, ((0, 0), (0, pad)))
        extra_pts = np.pad((extra - mean) @ comps, ((0, 0), (0, pad))) if extra is not None else None
        return pts, extra_pts
    extra_pts = (extra - mean) @ comps if extra is not None else None
    return pts, extra_pts
