"""Smoke test: drive the real Streamlit app headlessly through all scripted turns.

Slow-ish (loads the bge-small model), but catches UI wiring errors that unit
tests can't see.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _button(at, prefix):
    return next(b for b in at.button if b.label.startswith(prefix))


def test_app_runs_through_all_turns_without_errors():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception
    # Step through ingest, then run all three scripted turns, then scrub back.
    for label in ["Next", "Next", "Next", "⏭ Run next turn", "⏭ Run next turn", "⏭ Run next turn"] + ["◀ Prev"] * 10:
        _button(at, label).click().run()
        assert not at.exception, at.exception
    assert len(at.session_state["prompts"]) == 3
    # Turn 2 wrote the allergy to long-term memory; turn 3 carries it.
    final = at.session_state["final_state"]
    assert "Allergy: penicillin" in final["long_term_memory"]["medical_history"]
