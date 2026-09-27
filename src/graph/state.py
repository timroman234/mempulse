"""
AgentState: the single dictionary that flows through the LangGraph graph.

WHAT THIS TEACHES
-----------------
In LangGraph, "working memory" is literally this dict. Each node receives the
current state, returns ONLY the keys it wants to change, and LangGraph merges
those into the next state. MemPulse snapshots the state after every node, so
you can scrub back and forth and see exactly which node wrote which key.

Memory tiers mapped onto the state:
  working memory  → the whole AgentState for the current turn
  short-term      → conversation_history (carried turn to turn)
  long-term       → long_term_memory (profile facts that persist)
  knowledge       → the vector index (lives outside the state; the state only
                    holds the retrieved_chunks pulled from it)

Schema follows PRD §5.1, extended with the fields needed to visualise the
context window and memory writes.
"""

from __future__ import annotations

from typing import TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str
    similarity: float
    snippet: str
    char_range: list[int]
    vector_preview: list[float]


class LongTermMemory(TypedDict):
    user_id: str
    medical_history: list[str]
    preferences: list[str]


class AgentState(TypedDict, total=False):
    # --- PRD §5.1 core fields ---
    thread_id: str
    active_node: str
    user_prompt: str
    long_term_memory: LongTermMemory
    retrieved_chunks: list[RetrievedChunk]
    generated_response: str
    execution_step: int
    # --- MemPulse extensions ---
    turn: int                          # 1-based conversation turn
    conversation_history: list[dict]   # short-term memory: [{role, content}]
    context_window: list[dict]         # packed prompt segments (see rag/context.py)
    prompt_text: str                   # the literal prompt sent to the LLM
    llm_meta: dict                     # model name, token usage, errors
    memory_writes: list[dict]          # facts committed to LTM this turn
