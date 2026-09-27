"""
Optional Claude integration for the two "thinking" nodes of the graph.

WHAT THIS TEACHES
-----------------
The LLM is stateless: it knows only what is in the prompt we assemble. That's
why MemPulse shows the exact prompt text. It *is* the model's memory for this
turn.

Two jobs:
  answer()          - the LLM Reasoning node. It sends the assembled context.
  extract_memory()  - the Memory Writer node. It pulls durable facts
                      ("allergic to penicillin") out of the user's message so
                      they can be written to long-term memory.

With no API key, both fall back to transparent offline logic, so the app
always runs and the memory visuals still work.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
from dotenv import load_dotenv

# Pick up ANTHROPIC_API_KEY from a local .env file (gitignored). Real
# environment variables win over .env values.
load_dotenv()

MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]


def get_client(api_key: str | None):
    """Return an Anthropic client, or None when no key is available.

    MEMPULSE_OFFLINE=1 forces offline mode (used by the test suite so tests
    never spend API credits, even when a key is present in .env).
    """
    if os.environ.get("MEMPULSE_OFFLINE") == "1":
        return None
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=key)


def _call(client, model: str, system: str, user: str, max_tokens: int = 2000) -> tuple[str, dict]:
    """One Messages API call. Returns (text, usage-info)."""
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if model != "claude-haiku-4-5":
        # Short factual answers don't need deep reasoning: low effort keeps it fast.
        kwargs["output_config"] = {"effort": "low"}
    if model == "claude-opus-5":
        # Opus 5 may decline some requests via safety classifiers (stop_reason
        # "refusal"). Server-side fallbacks retry on a suitable model automatically.
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        resp = client.messages.create(**kwargs)

    if resp.stop_reason == "refusal":
        return "(Claude declined to answer this request.)", {"model": resp.model, "stop_reason": "refusal"}
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    usage = {
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "stop_reason": resp.stop_reason,
    }
    return text, usage


# ------------------------------------------------------------------ answer()

def answer(client, model: str, system: str, prompt_text: str, fallback_ctx: dict) -> tuple[str, dict]:
    """LLM Reasoning node. Use Claude if available, otherwise answer extractively."""
    if client is not None:
        try:
            return _call(client, model, system, prompt_text)
        except Exception as e:  # show the error in the UI instead of crashing the demo
            text, meta = _offline_answer(**fallback_ctx)
            meta["error"] = f"{type(e).__name__}: {e}"
            return text, meta
    return _offline_answer(**fallback_ctx)


def _offline_answer(query: str, chunks: list[dict], ltm: dict, embedder) -> tuple[str, dict]:
    """A deterministic stand-in for an LLM: pick the sentences from the
    in-context chunks that are most similar to the question.

    It's deliberately literal: if the right chunk isn't in context, the
    answer is visibly wrong. That's the point: garbage memory in, garbage out.
    """
    sentences = []
    for c in chunks:
        for s in re.split(r"(?<=[.!?])\s+", c["text"]):
            if len(s.split()) >= 5:
                sentences.append((c["chunk_id"], s.strip()))
    if not sentences:
        return "I couldn't find anything relevant in the retrieved context.", {"model": "offline-extractive"}

    q = embedder.embed_query(query)
    vecs = embedder.embed_documents([s for _, s in sentences])
    sims = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
    best = [sentences[i] for i in np.argsort(-sims)[:2]]
    body = " ".join(f"{s} [{cid}]" for cid, s in best)

    # Safety cross-check against long-term memory, the way a real agent should.
    allergies = [h for h in ltm.get("medical_history", []) if "allerg" in h.lower() and "seasonal" not in h.lower()]
    note = ""
    if allergies:
        note = f"\n\nMemory check: your record lists **{'; '.join(allergies)}**."
        if any("penicillin" in a.lower() for a in allergies) and re.search(r"amoxicillin|ampicillin", body, re.I):
            if re.search(r"penicillin allergy", body, re.I):
                # The right chunk is in memory: steer to the allergy-safe option.
                note += " ✅ Retrieved context includes the penicillin-allergy alternative. Use that option, not amoxicillin."
            else:
                # The allergy is known but the safe chunk never reached memory.
                note += " ⚠️ The retrieved text recommends a penicillin-class drug, which is unsafe for you, and no alternative was retrieved."
    return body + note, {"model": "offline-extractive"}


# --------------------------------------------------------- extract_memory()

_EXTRACT_SYSTEM = (
    "Extract durable personal facts about the patient from their message that "
    "should be saved to their long-term medical record: allergies, conditions, "
    "current medications, or stated preferences. Reply with ONLY a JSON array "
    'of objects like {"category": "medical_history" | "preferences", "fact": "..."}. '
    "Reply [] if there is nothing worth saving."
)

_RULES = [
    (r"\ballergic to ([a-z][a-z \-]+)", "medical_history", "Allergy: {}"),
    (r"\bi (?:am taking|'m taking|currently take|am on|'m on)([a-z][a-z0-9 \-]+)", "medical_history", "Current medication: {}"),
    (r"\bi (?:have|was diagnosed with) ([a-z][a-z \-]+)", "medical_history", "Condition: {}"),
    (r"\bi prefer ([a-z][a-z \-]+)", "preferences", "Prefers {}"),
]


def extract_memory(client, model: str, message: str) -> tuple[list[dict], str]:
    """Memory Writer node. Returns (facts, method)."""
    if client is not None:
        try:
            text, _ = _call(client, model, _EXTRACT_SYSTEM, message, max_tokens=1000)
            m = re.search(r"\[.*\]", text, re.S)
            facts = json.loads(m.group(0)) if m else []
            facts = [f for f in facts if f.get("category") in ("medical_history", "preferences") and f.get("fact")]
            return facts, f"Claude ({model})"
        except Exception:
            pass  # fall through to the rules so memory still gets written
    facts = []
    low = message.lower()
    for pattern, cat, tmpl in _RULES:
        for m in re.finditer(pattern, low):
            value = re.split(r"\b(and|but|so|because)\b|[.,;!?]", m.group(1))[0].strip()
            if value:
                facts.append({"category": cat, "fact": tmpl.format(value)})
    return facts, "offline regex rules"
