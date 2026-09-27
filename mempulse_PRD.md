# Product Requirement Document (PRD)
## Project Name: MemPulse (Visual Agent Memory & State Debugger)

---

## 1. Executive Summary & Vision

### 1.1 Executive Summary
**MemPulse** is an interactive, visual runtime inspector and memory state debugger designed for graph-based AI agent frameworks (specifically LangGraph). While existing telemetry tools (e.g., LangSmith, Phoenix) focus on latency and LLM trace logs, MemPulse focuses on **demystifying agent memory state transitions**. It provides a real-time, side-by-side visualization of short-term state payload mutations, document chunking parameter boundaries, vector embedding matrices, and long-term memory reads/writes across node graph steps.

### 1.2 Vision Statement
To convert black-box AI agent execution into a transparent, visual state machine that developers, prompt engineers, and AI architects can debug, optimize, and demonstrate with ease.

---

## 2. Core Objectives & Target Audience

### 2.1 Objectives
* **Visual State Transparency:** Render real-time `AgentState` dictionary mutations across every node execution in a LangGraph workflow.
* **Interactive Chunking Engine:** Allow developers to adjust chunk sizes and overlap dynamically, visualizing character boundary shifts and overlap markers in real-time.
* **Vector & Memory Inspection:** Display high-dimensional vector representations ($1536\text{D}$ / $3072\text{D}$) alongside vector similarity scoring ($\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$).
* **Production-Grade Aesthetics:** Adopt IBM Carbon Design System (Gray 100 theme) for high-contrast, enterprise-ready visual presentation.

### 2.2 Target Personas
1. **AI / LangGraph Engineer:** Needs to debug state mutations, memory overwrites, and poor retrieval accuracy during node hops.
2. **Enterprise Solution Architect:** Needs a polished, high-level visual demonstration tool for stakeholders to prove how memory, safety controls, and RAG function in production environments.
3. **AI Researcher / Educator:** Requires an interactive sandbox to teach document chunking, vector spaces, and graph-based agent orchestration.

---

## 3. Technology Stack & Package Management (`uv`)

### 3.1 Technology Stack
* **Language:** Python 3.11+
* **Frontend / Dashboard:** Streamlit (v1.35.0+)
* **Agent Framework:** LangGraph / LangChain Core
* **Chunking Engine:** `langchain-text-splitters`
* **Math & Data Processing:** NumPy, Pandas
* **Styling:** Custom CSS based on IBM Carbon Design System (Gray 100 Theme)

### 3.2 Package Management with `uv`
This project strictly utilizes `uv` as the ultra-fast Python package installer and resolver.

#### Initialization & Environment Setup
```bash
# Initialize uv project
uv init mempulse
cd mempulse

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install streamlit langchain-text-splitters numpy pandas
```

#### `pyproject.toml`
```toml
[project]
name = "mempulse"
version = "0.1.0"
description = "Visual Agent Memory & State Debugger for LangGraph Workflows"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "streamlit>=1.35.0",
    "langchain-text-splitters>=0.2.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "black>=24.0.0",
    "ruff>=0.4.0"
]
```

---

## 4. System Architecture & File Layout

### 4.1 Architecture Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION                              │
│                [ User Prompt ] ──► [ Parameter Controls ]               │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       MEMPULSE AGENT GRAPH ENGINE                       │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ 1. Intent Store  │───►│ 2. RAG Retriever  │───►│ 3. LLM Reasoning │   │
│  │      Node        │    │      Node        │    │      Node        │   │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘   │
│           │                       │                       │             │
│           ▼                       ▼                       ▼             │
│   Reads Profile/LTM      Chunks Doc & Embeds     Synthesizes Prompt     │
│   State Mutation         Vector Similarity       + Context + Profile    │
│                                                           │             │
│  ┌────────────────────────────────────────────────────────┘             │
│  │                                                                      │
│  ▼                                                                      │
│  ┌──────────────────┐                                                   │
│  │ 4. Memory Writer │ ──► Commits New Entity / Interaction to LTM      │
│  │      Node        │                                                   │
│  └──────────────────┘                                                   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT VISUAL LAYER                          │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ Tab 1: State Json│    │ Tab 2: Chunking  │    │ Tab 3: Vector    │   │
│  │   Mutations      │    │    Boundaries    │    │   Space Math     │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Recommended Directory Structure

```
mempulse/
├── .venv/
├── pyproject.toml
├── README.md
├── .gitignore
├── app.py                     # Main Streamlit Application entrypoint
├── src/
│   ├── __init__.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py           # TypedDict AgentState definitions
│   │   └── pipeline.py        # LangGraph Node implementation & runner
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py         # Text chunking logic & boundary markers
│   │   └── embeddings.py      # Vector generation & similarity math
│   └── ui/
│       ├── __init__.py
│       ├── styles.py          # Carbon Dark CSS injection strings
│       └── components.py      # Reusable UI widgets (cards, badges, matrices)
└── tests/
    ├── test_chunker.py
    └── test_state.py
```

---

## 5. Data Schemas & State Specifications

### 5.1 `AgentState` TypedDict
The central state payload passed between graph nodes must adhere strictly to the following dictionary contract:

```python
from typing import TypedDict, List, Dict, Any, Optional

class RetrievedChunk(TypedDict):
    chunk_id: str
    similarity: float
    snippet: str
    char_range: List[int]
    vector_preview: List[float]

class LongTermMemory(TypedDict):
    user_id: str
    medical_history: List[str]
    preferences: List[str]

class AgentState(TypedDict):
    thread_id: str
    active_node: str
    user_prompt: str
    long_term_memory: LongTermMemory
    retrieved_chunks: List[RetrievedChunk]
    generated_response: str
    execution_step: int
```

### 5.2 Embedding Vector Mathematical Specification
Cosine Similarity calculation between Query Vector $\mathbf{q}$ and Chunk Vector $\mathbf{c}_i$:

$$\text{Similarity}(\mathbf{q}, \mathbf{c}_i) = \frac{\mathbf{q} \cdot \mathbf{c}_i}{\|\mathbf{q}\|_2 \|\mathbf{c}_i\|_2} = \frac{\sum_{j=1}^{d} q_j c_{i,j}}{\sqrt{\sum_{j=1}^{d} q_j^2} \sqrt{\sum_{j=1}^{d} c_{i,j}^2}}$$

Where $d \in \{1536, 3072\}$ represents embedding dimensionality.

---

## 6. Functional Requirements & Feature Matrix

| ID | Feature Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **FR-01** | Step-by-Step Execution Control | Interactive "Next Step", "Prev Step", and "Reset" controls to navigate through graph nodes sequentially. | P0 |
| **FR-02** | Live State JSON Inspector | Expandable JSON viewer rendering exact `AgentState` mutations as nodes execute. | P0 |
| **FR-03** | Dynamic Chunking Visualizer | Adjust `chunk_size` and `chunk_overlap` sliders; view source text split into discrete chunks with highlighted overlap text. | P0 |
| **FR-04** | Vector Similarity Matrix | Table and bar chart displaying top-$k$ ranked chunks with cosine similarity metrics and float previews. | P0 |
| **FR-05** | IBM Carbon UI Theme | Full dark mode styling (`#161616` background, `#262626` cards, `#0f62fe` active accents, monospace code blocks). | P1 |
| **FR-06** | Semantic vs Fixed Splitting Switch | Toggle between fixed-size recursive character chunking and heading-based semantic chunking. | P2 |
| **FR-07** | Custom Document Ingestion | Allow users to upload or paste custom markdown/text documents for live chunking and vector inspection. | P2 |

---

## 7. UI/UX Specifications (IBM Carbon Dark Guidelines)

### 7.1 Color Palette Tokens
* **Background Layer (`$background`):** `#161616` (Carbon Gray 100)
* **Container / Card (`$layer-01`):** `#262626` (Carbon Gray 90)
* **Hover / Active Container (`$layer-02`):** `#353535` (Carbon Gray 80)
* **Primary Interactive Accent (`$interactive`):** `#0f62fe` (Carbon Blue 60)
* **Active Highlight (`$highlight`):** `#4589ff` (Carbon Blue 40)
* **Text Primary (`$text-01`):** `#f4f4f4` (Carbon Gray 10)
* **Text Secondary (`$text-02`):** `#c6c6c6` (Carbon Gray 30)
* **Overlap Highlight:** `#684600` background with `#f1c21b` text (Carbon Yellow 30)

### 7.2 Typography Guidelines
* **Body Font:** `IBM Plex Sans`, `-apple-system`, `sans-serif`
* **Code / Json / Vector Displays:** `IBM Plex Mono`, `Courier New`, `monospace`

---

## 8. Development Roadmap & Phased Execution

### Phase 1: MVP Core (Current Baseline)
- [x] Streamlit single-page application prototype (`app.py`).
- [x] Custom Carbon CSS design system integration.
- [x] LangGraph 4-step node pipeline visualization.
- [x] Interactive `RecursiveCharacterTextSplitter` chunking visualization.
- [x] Synthetic vector generator and cosine similarity scoring matrix.

### Phase 2: Modular Architecture & Real Embeddings
- [ ] Refactor `app.py` into modular package layout (`src/graph`, `src/rag`, `src/ui`).
- [ ] Integrate optional OpenAI / HuggingFace API key input for real vector embeddings (`text-embedding-3-small`).
- [ ] Add custom text document paste/upload functionality.

### Phase 3: Advanced RAG & Memory Persistence
- [ ] Implement two-stage re-ranking visualizer (Bi-Encoder Retrieval + Cross-Encoder Re-ranker).
- [ ] Integrate persistent SQLite / DuckDB backend to store long-term memory across app reloads.
- [ ] Export state execution trace as downloadable JSON file.

---

## 9. Verification & Acceptance Criteria

### 9.1 Verification Checklist
1. **Dependency Verification:** Application launches cleanly via `uv run streamlit run app.py` without package conflicts.
2. **State Accuracy:** Navigating from Node 1 to Node 4 sequentially updates `state_snapshot['active_node']` and appends long-term memory items without state corruption.
3. **Chunk Boundary Correctness:** Changing `chunk_size` or `chunk_overlap` recalculates the total chunk count immediately and updates overlap highlights without overflowing container boundaries.
4. **Vector Math Precision:** Cosine similarity scores always fall strictly in the range $[-1.0, 1.0]$ and top-$k$ sorting displays the highest similarity score at the top.
5. **Responsiveness:** Layout scales seamlessly across viewport widths from $1280\text{px}$ to $2560\text{px}$.

---

## 10. Prompt Instructions for Claude Code / VS Code Execution

When instructing Claude Code in VS Code to construct or refactor this codebase, run the following command sequence:

```bash
# 1. Setup Environment
uv venv
source .venv/bin/activate
uv pip install streamlit langchain-text-splitters numpy pandas pytest

# 2. Instruct Claude Code
# "Implement the directory structure and code modules outlined in PRD.md under section 4.2."
```