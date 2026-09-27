"""
Vector index: the "long-term knowledge" memory tier.

WHAT THIS TEACHES
-----------------
A vector store is just two parallel arrays: the chunk texts and an N × D
matrix of their embeddings. "Search" = cosine similarity of the query vector
against every row, then take the top-k. Production stores (FAISS, pgvector,
Chroma...) add approximate-nearest-neighbour tricks for speed, but the idea
is exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.rag.chunker import Chunk, chunk_document
from src.rag.embeddings import cosine_matrix, get_embedder


@dataclass
class VectorIndex:
    chunks: list[Chunk]
    vectors: np.ndarray      # shape (N, D), row i ↔ chunks[i]
    embedder_name: str

    @classmethod
    def build(cls, doc: str, chunk_size: int, chunk_overlap: int, strategy: str, embedder_name: str) -> "VectorIndex":
        chunks = chunk_document(doc, chunk_size, chunk_overlap, strategy)
        vectors = get_embedder(embedder_name).embed_documents([c.text for c in chunks])
        return cls(chunks, vectors, embedder_name)

    def query_vector(self, query: str) -> np.ndarray:
        return get_embedder(self.embedder_name).embed_query(query)

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query to EVERY chunk (not just top-k).
        Seeing the losers is how you debug a retrieval miss."""
        return cosine_matrix(self.query_vector(query), self.vectors)

    def search(self, query: str, k: int) -> list[dict]:
        """Top-k chunks as PRD `RetrievedChunk` dicts, highest similarity first."""
        sims = self.scores(query)
        order = np.argsort(-sims)[:k]
        return [
            {
                "chunk_id": self.chunks[i].chunk_id,
                "similarity": round(float(sims[i]), 4),
                "snippet": self.chunks[i].text,
                "char_range": [self.chunks[i].start, self.chunks[i].end],
                # First 8 dims only: enough to show "it's just numbers" in the JSON.
                "vector_preview": [round(float(x), 4) for x in self.vectors[i][:8]],
            }
            for i in order
        ]
