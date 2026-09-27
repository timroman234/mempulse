"""Graph snapshots: node order, LTM writes, and no cross-snapshot corruption
(PRD §9.1 item 2). Runs fully offline (no API key, hashing embedder)."""

from src.data.sample import CLINIC_GUIDE, INITIAL_LTM, SCRIPTED_TURNS
from src.graph.pipeline import NODES, initial_state, run_turn
from src.llm import claude
from src.rag.index import VectorIndex

CFG = dict(chunk_size=400, chunk_overlap=60, strategy="recursive", embedder="hashing", top_k=3, budget=500)


def _run_all():
    idx = VectorIndex.build(CLINIC_GUIDE, 400, 60, "recursive", "hashing")
    state, all_snaps = initial_state(INITIAL_LTM), []
    for t in SCRIPTED_TURNS:
        snaps = run_turn(idx, CFG, state, t["prompt"], client=None)
        all_snaps.append(snaps)
        state = snaps[-1]
    return all_snaps


def test_each_turn_visits_nodes_in_order_and_steps_increase():
    turns = _run_all()
    steps = []
    for snaps in turns:
        assert [s["active_node"] for s in snaps] == NODES
        steps += [s["execution_step"] for s in snaps]
    assert steps == list(range(1, len(steps) + 1))


def test_ltm_write_persists_and_earlier_snapshots_untouched():
    turns = _run_all()
    allergy = "Allergy: penicillin"
    # Turn 1 never saw the allergy, and must still not see it after later turns ran.
    assert all(allergy not in s["long_term_memory"]["medical_history"] for s in turns[0])
    # Turn 2's writer commits it; turn 3 recalls it.
    assert allergy in turns[1][-1]["long_term_memory"]["medical_history"]
    assert turns[1][-1]["memory_writes"][0]["fact"] == allergy
    assert allergy in turns[2][0]["long_term_memory"]["medical_history"]
    # Short-term history grows by 2 messages per turn.
    assert [len(t[-1]["conversation_history"]) for t in turns] == [2, 4, 6]


def test_offline_memory_rules_ignore_questions():
    facts, _ = claude.extract_memory(None, "", "Which antibiotic should I take for a sinus infection?")
    assert facts == []
