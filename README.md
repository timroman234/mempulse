# MemPulse: Visual Agent Memory & State Debugger

A single-screen Streamlit app that shows **what an AI agent / RAG pipeline has in memory at every step**:
from chunking and embedding a document, through each conversation turn of a 5-node LangGraph agent.
It also shows **why that matters**: the same question succeeds or fails depending on chunking, embedding,
and context budget.

## Run it

```bash
uv sync
uv run streamlit run app.py
```

The first load takes about 20 s, because it downloads and loads the `bge-small-en-v1.5` embedding model (~70 MB, once).
Claude is optional. Put `ANTHROPIC_API_KEY=...` in a `.env` file (gitignored), export it, or paste a key in the sidebar. Without one, the LLM node uses a
transparent offline "extractive" answerer, and the memory writer uses regex rules.

## For students

A step-by-step classroom walkthrough in plain language is in **[docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md)**.

## The screen

| Area | What it shows |
|---|---|
| **Pipeline strip** | Ingest (Load → Chunk → Embed → Index), then each turn (Recall LTM → Retrieve → Assemble → LLM Reason → Write Memory). ◀ Prev / Next ▶ time-travel through recorded snapshots. |
| **Memory stack** (left) | The four memory tiers (knowledge/vector store, long-term profile, short-term history, working AgentState) plus a **context-window gauge** that shows evicted items in red. |
| **Stage lens** (centre) | A visual for the active node: the document ribbon with chunk boundaries, yellow overlap and the underlined answer fact (✂ where a boundary cuts it); the embedding heatmap; PCA vector space with the query ★; the cosine ranking with the top-k cut-off; the literal prompt coloured by memory tier; the answer; LTM writes. |
| **State diff** (right) | AgentState changes vs the previous step (+ added, ~ changed). |
| **Why memory state matters** (bottom) | A health check (verdict, fact intact %, answer-chunk rank, similarity margin, signal/noise, tokens), the **What-if Lab** (the same question under Tiny / No-overlap / Giant chunks and the Hashing embedder, with one-click Apply that replays the whole conversation), and an optional **sensitivity sweep** heatmap of chunk_size × overlap. |

## The scripted story (3 turns)

1. *"What's the most painkiller I can take in a day?"* is a paraphrase. The semantic embedder finds
   "analgesic … 24 hours"; the hashing embedder misses it.
2. *"By the way, I'm allergic to penicillin."* The Memory Writer commits it to long-term memory (green pulse).
3. *"Which antibiotic should I take for a sinus infection?"* A safe answer needs **both** the LTM allergy **and**
   the penicillin-alternative chunk. Break chunking or embedding and watch it fail.

## Layout

```
app.py                 Streamlit entry: layout, navigation, trace in session_state
src/graph/             AgentState, the LangGraph pipeline, trace/diff helpers
src/rag/               chunker, embeddings (fastembed + hashing), vector index,
                       context packing (budget/eviction), diagnostics (verdicts, what-if, sweep)
src/llm/claude.py      optional Claude calls (answer + memory extraction) with offline fallbacks
src/data/sample.py     clinic guide, patient profile, scripted turns + ground-truth "needles"
src/ui/                Carbon Gray 100 CSS, HTML widgets, Plotly charts
tests/                 unit tests + a headless AppTest smoke test
```

## Tests

```bash
uv run pytest
```
