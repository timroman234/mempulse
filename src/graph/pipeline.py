"""
The agent: a 5-node LangGraph workflow, run one conversation turn at a time.

WHAT THIS TEACHES
-----------------
    recall_ltm → retrieve → assemble_context → reason → write_memory

  recall_ltm        loads the long-term profile (in production, a DB read).
  retrieve          embeds the question and pulls the top-k chunks.
  assemble_context  packs everything into a token budget. This is where
                    memory gets EVICTED if it doesn't fit.
  reason            the LLM call (Claude, or the offline stand-in).
  write_memory      extracts durable facts and commits them to LTM, and
                    appends this turn to short-term conversation history.

`graph.stream(..., stream_mode="values")` yields the FULL state after every
node. We deep-copy each one into a snapshot list. Those snapshots are what the
UI scrubs through. Without deep copies, later nodes would mutate the lists in
earlier snapshots and the time-travel view would lie.

Carrying memory between turns: we pass the previous turn's final LTM and
history into the next turn's input state by hand. LangGraph's checkpointers
(e.g. MemorySaver) automate this. Here we do it explicitly so you can watch it.
"""

from __future__ import annotations

import copy

from langgraph.graph import END, START, StateGraph

from src.graph.state import AgentState
from src.llm import claude
from src.rag.context import SYSTEM_PROMPT, build_segments, pack_context, render_prompt
from src.rag.embeddings import get_embedder
from src.rag.index import VectorIndex

NODES = ["recall_ltm", "retrieve", "assemble_context", "reason", "write_memory"]
NODE_LABELS = {
    "recall_ltm": "Recall LTM",
    "retrieve": "Retrieve",
    "assemble_context": "Assemble",
    "reason": "LLM Reason",
    "write_memory": "Write Memory",
}
# Plain-English one-liners shown under the stage lens.
NODE_EXPLAIN = {
    "recall_ltm": "Loads the patient's long-term profile into working memory before anything else happens.",
    "retrieve": "Embeds the question and ranks every chunk by cosine similarity. Only the top-k enter working memory.",
    "assemble_context": "Packs system prompt, profile, chunks and history into the token budget. Whatever doesn't fit is evicted.",
    "reason": "The LLM sees ONLY the assembled prompt. That prompt is its entire memory for this turn.",
    "write_memory": "Extracts durable facts from the message and commits them to long-term memory, then appends the turn to short-term history.",
}


def build_graph(index: VectorIndex, cfg: dict, client, model: str):
    """Compile the graph. Nodes close over the index/config (they are not state)."""

    def recall_ltm(state: AgentState) -> dict:
        # Real systems would query a DB by user_id here. Our LTM is carried in
        # the input state, so this node's visible job is to mark the load.
        return {"active_node": "recall_ltm", "execution_step": state["execution_step"] + 1,
                "long_term_memory": copy.deepcopy(state["long_term_memory"])}

    def retrieve(state: AgentState) -> dict:
        chunks = index.search(state["user_prompt"], cfg["top_k"])
        return {"active_node": "retrieve", "execution_step": state["execution_step"] + 1,
                "retrieved_chunks": chunks}

    def assemble_context(state: AgentState) -> dict:
        segs = build_segments(state["long_term_memory"], state["retrieved_chunks"],
                              state.get("conversation_history", []), state["user_prompt"])
        packed = pack_context(segs, cfg["budget"])
        return {"active_node": "assemble_context", "execution_step": state["execution_step"] + 1,
                "context_window": packed, "prompt_text": render_prompt(packed)}

    def reason(state: AgentState) -> dict:
        in_ctx = [{"chunk_id": s["chunk_id"], "text": s["text"]}
                  for s in state["context_window"] if s["tier"] == "chunk" and s["included"]]
        text, meta = claude.answer(
            client, model, SYSTEM_PROMPT, state["prompt_text"],
            fallback_ctx=dict(query=state["user_prompt"], chunks=in_ctx,
                              ltm=state["long_term_memory"], embedder=get_embedder(cfg["embedder"])),
        )
        return {"active_node": "reason", "execution_step": state["execution_step"] + 1,
                "generated_response": text, "llm_meta": meta}

    def write_memory(state: AgentState) -> dict:
        facts, method = claude.extract_memory(client, model, state["user_prompt"])
        ltm = copy.deepcopy(state["long_term_memory"])
        written = []
        for f in facts:
            bucket = ltm.setdefault(f["category"], [])
            if f["fact"] not in bucket:  # don't store duplicates
                bucket.append(f["fact"])
                written.append({**f, "method": method})
        history = list(state.get("conversation_history", [])) + [
            {"role": "user", "content": state["user_prompt"]},
            {"role": "assistant", "content": state["generated_response"]},
        ]
        return {"active_node": "write_memory", "execution_step": state["execution_step"] + 1,
                "long_term_memory": ltm, "memory_writes": written, "conversation_history": history}

    g = StateGraph(AgentState)
    for name, fn in [("recall_ltm", recall_ltm), ("retrieve", retrieve),
                     ("assemble_context", assemble_context), ("reason", reason),
                     ("write_memory", write_memory)]:
        g.add_node(name, fn)
    g.add_edge(START, "recall_ltm")
    g.add_edge("recall_ltm", "retrieve")
    g.add_edge("retrieve", "assemble_context")
    g.add_edge("assemble_context", "reason")
    g.add_edge("reason", "write_memory")
    g.add_edge("write_memory", END)
    return g.compile()


def initial_state(ltm: dict, thread_id: str = "demo-thread") -> AgentState:
    """The state before turn 1: nothing but the stored long-term profile."""
    return {
        "thread_id": thread_id, "active_node": "(start)", "user_prompt": "",
        "long_term_memory": copy.deepcopy(ltm), "retrieved_chunks": [],
        "generated_response": "", "execution_step": 0, "turn": 0,
        "conversation_history": [], "context_window": [], "prompt_text": "",
        "llm_meta": {}, "memory_writes": [],
    }


def run_turn(index: VectorIndex, cfg: dict, prev: AgentState, prompt: str, client=None, model: str = "claude-opus-5") -> list[AgentState]:
    """Run one turn and return a deep-copied state snapshot per node."""
    graph = build_graph(index, cfg, client, model)
    # Short-term + long-term memory carry over; per-turn fields reset.
    start = {
        **initial_state(prev["long_term_memory"], prev["thread_id"]),
        "conversation_history": copy.deepcopy(prev["conversation_history"]),
        "execution_step": prev["execution_step"],
        "turn": prev["turn"] + 1,
        "user_prompt": prompt,
    }
    snapshots = []
    for i, state in enumerate(graph.stream(start, stream_mode="values")):
        if i == 0:
            continue  # the first emission is just the input state
        snapshots.append(copy.deepcopy(state))
    return snapshots
