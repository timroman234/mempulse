"""
Chunker: turns one long document into the pieces that become "memory".

WHAT THIS TEACHES
-----------------
An LLM never sees your document. It only ever sees the few *chunks* that
retrieval hands it. So the chunk boundaries you choose here decide what the
agent can possibly know later. Three knobs matter:

  chunk_size    - max characters per chunk. Too small → facts lose their
                  context ("1,200 mg" with no drug name). Too large → each
                  vector is an average of many topics, similarity scores get
                  blurry, and the chunks eat your context-window budget.
  chunk_overlap - characters repeated between neighbouring chunks. Overlap is
                  insurance: if a boundary lands in the middle of a key
                  sentence, the next chunk still holds the whole sentence.
  strategy      - "recursive" cuts by size (trying paragraph → line → word
                  breaks first); "heading" cuts at markdown headings so each
                  chunk is one coherent topic.

Every chunk keeps its exact character range (start, end) in the source. That
lets the UI draw boundaries on the original text and lets diagnostics check
whether a ground-truth fact survived chunking intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    chunk_id: str      # stable, human-readable id, e.g. "C03"
    text: str          # the chunk contents exactly as stored in the vector DB
    start: int         # inclusive char offset in the source document
    end: int           # exclusive char offset in the source document
    overlap_prev: int  # chars shared with the previous chunk (0 = no overlap)

    def to_dict(self) -> dict:
        return asdict(self)


def _recursive(doc: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, str]]:
    """Fixed-size chunking with LangChain's RecursiveCharacterTextSplitter.

    `add_start_index=True` makes LangChain record where each chunk begins, so
    we don't have to guess. It tries separators in order ("\\n\\n", "\\n",
    " ", "") and only falls back to smaller ones when a piece is still too big.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    docs = splitter.create_documents([doc])
    return [(d.metadata["start_index"], d.page_content) for d in docs]


def _by_heading(doc: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, str]]:
    """Semantic-ish chunking: one chunk per markdown section.

    We split at lines starting with '#'. A section longer than `chunk_size`
    is sub-split with the recursive splitter, because one giant section would
    bring back the "blurry vector" problem. Offsets are kept exact by adding
    the section's own start offset.
    """
    # Positions where a heading line starts (plus doc start and end).
    starts = [m.start() for m in re.finditer(r"^#{1,6} ", doc, flags=re.MULTILINE)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [len(doc)]

    pieces: list[tuple[int, str]] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        section = doc[s:e]
        if not section.strip():
            continue
        if len(section) <= chunk_size:
            # Strip trailing whitespace but keep the offsets honest.
            text = section.rstrip()
            pieces.append((s, text))
        else:
            for sub_start, sub_text in _recursive(section, chunk_size, chunk_overlap):
                pieces.append((s + sub_start, sub_text))
    return pieces


def chunk_document(
    doc: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    strategy: str = "recursive",
) -> list[Chunk]:
    """Split `doc` into Chunks with exact character ranges.

    The overlap can't be as large as the chunk itself (the splitter would
    loop forever), so we clamp it to half the chunk size.
    """
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))
    raw = (_by_heading if strategy == "heading" else _recursive)(doc, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    prev_end = 0
    for i, (start, text) in enumerate(raw):
        end = start + len(text)
        # Overlap = how far this chunk reaches back into the previous one.
        overlap = max(0, prev_end - start) if i > 0 else 0
        chunks.append(Chunk(f"C{i:02d}", text, start, end, overlap))
        prev_end = end
    return chunks
