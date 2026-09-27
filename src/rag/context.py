"""
Context window assembly: the final and most overlooked memory layer.

WHAT THIS TEACHES
-----------------
Everything the agent "remembers" has to be squeezed into one prompt: the
system instructions, the long-term profile, the retrieved chunks, recent
conversation, and the new question. That prompt has a budget. When things
don't fit, something gets EVICTED, silently, in most frameworks.

MemPulse makes eviction visible. Segments are packed by priority:
    1. system prompt + user prompt   (required, never dropped)
    2. long-term memory profile      (small and high value)
    3. retrieved chunks, best rank first
    4. conversation history, newest turn first
Anything that doesn't fit is marked `included=False` and shown in red.

Token counts are an ESTIMATE (~4 characters per token for English). That's
good enough to show proportions. The LLM node reports real usage when Claude
is enabled.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful clinic assistant. Answer ONLY from the provided clinic "
    "guide excerpts and the patient profile. Always respect the patient's "
    "allergies. If the excerpts don't contain the answer, say so."
)

# Colours per memory tier, shared by the gauge, the prompt view and charts.
TIER_COLORS = {
    "system": "#8d8d8d",
    "ltm": "#be95ff",      # purple: long-term memory
    "chunk": "#4589ff",    # blue: retrieved knowledge
    "history": "#08bdba",  # teal: short-term memory
    "prompt": "#f1c21b",   # yellow: the new question
}


def est_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def format_ltm(ltm: dict) -> str:
    history = "; ".join(ltm.get("medical_history", [])) or "none recorded"
    prefs = "; ".join(ltm.get("preferences", [])) or "none recorded"
    return f"Patient {ltm.get('user_id', '?')}. Medical history: {history}. Preferences: {prefs}."


def build_segments(
    ltm: dict,
    retrieved: list[dict],
    history: list[dict],
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """Every candidate piece of the prompt, tagged with its memory tier.

    `priority` decides packing order (lower = packed first). Chunks keep their
    retrieval rank in the priority so rank 1 survives before rank 3.
    """
    segs = [
        {"tier": "system", "label": "System prompt", "text": system_prompt, "required": True, "priority": 0},
        {"tier": "prompt", "label": "User prompt", "text": user_prompt, "required": True, "priority": 0},
        {"tier": "ltm", "label": "Long-term profile", "text": format_ltm(ltm), "required": False, "priority": 1},
    ]
    for rank, c in enumerate(retrieved, start=1):
        segs.append({
            "tier": "chunk", "label": f"{c['chunk_id']} (rank {rank}, sim {c['similarity']:.2f})",
            "text": c["snippet"], "required": False, "priority": 10 + rank, "chunk_id": c["chunk_id"],
        })
    # Newest history first: if the budget is tight, old turns go first.
    for age, msg in enumerate(reversed(history)):
        segs.append({
            "tier": "history", "label": f"{msg['role']}: turn -{age // 2 + 1}",
            "text": msg["content"], "required": False, "priority": 100 + age,
        })
    for s in segs:
        s["tokens"] = est_tokens(s["text"])
    return segs


def pack_context(segments: list[dict], budget: int) -> list[dict]:
    """Greedy packing by priority. Returns segments in display order with
    `included` set. Required segments are always included, even over budget.
    """
    used = 0
    for s in sorted(segments, key=lambda s: s["priority"]):
        if s["required"] or used + s["tokens"] <= budget:
            s["included"] = True
            used += s["tokens"]
        else:
            s["included"] = False
    order = {"system": 0, "ltm": 1, "chunk": 2, "history": 3, "prompt": 4}
    # Display in the order the prompt is actually written.
    return sorted(segments, key=lambda s: (order[s["tier"]], s["priority"] if s["tier"] != "history" else -s["priority"]))


def render_prompt(segments: list[dict]) -> str:
    """Turn included segments into the literal text sent to the LLM."""
    parts = []
    ltm = [s for s in segments if s["included"] and s["tier"] == "ltm"]
    chunks = [s for s in segments if s["included"] and s["tier"] == "chunk"]
    hist = [s for s in segments if s["included"] and s["tier"] == "history"]
    prompt = [s for s in segments if s["tier"] == "prompt"]
    if ltm:
        parts.append("<patient_profile>\n" + ltm[0]["text"] + "\n</patient_profile>")
    if chunks:
        body = "\n\n".join(f"[{s['chunk_id']}] {s['text']}" for s in chunks)
        parts.append("<clinic_guide_excerpts>\n" + body + "\n</clinic_guide_excerpts>")
    if hist:
        parts.append("<recent_conversation>\n" + "\n".join(s["label"].split(":")[0] + ": " + s["text"] for s in hist) + "\n</recent_conversation>")
    parts.append("<question>\n" + prompt[0]["text"] + "\n</question>")
    return "\n\n".join(parts)
