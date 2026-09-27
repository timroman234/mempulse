"""
Sample scenario: a small clinic medication guide + a patient profile.

WHAT THIS TEACHES
-----------------
A RAG demo is only convincing if we know the *right answer* ahead of time.
So every scripted question below carries a "needle": the exact sentence in
the document that contains the answer. MemPulse tracks that needle through
chunking, embedding and retrieval, and can then say precisely *where* memory
failed (e.g. "the chunker sliced the dosage sentence in half").

The document is written on purpose so that:
  * Turn 1 is phrased with DIFFERENT words than the document uses
    ("painkiller ... in a day" vs "analgesic ... 24 hours"). A keyword/hashing
    embedder struggles; a semantic embedder (bge-small) copes.
  * Turn 3 has a dangerous distractor: the first-line sinusitis antibiotic is
    amoxicillin (a penicillin). The patient told us in Turn 2 they are
    penicillin-allergic. Only long-term memory + the right chunk together give
    a safe answer. That is the "why memory state matters" story.
"""

# The source document. Markdown so the heading-based splitter has structure.
CLINIC_GUIDE = """# Riverside Family Clinic: Patient Medication Guide

This guide summarizes common medication guidance from Riverside Family Clinic. It does not replace advice from your own clinician. Bring this guide to every appointment and ask the front desk for a printed copy if you need one.

## Analgesics and Fever Reducers

Acetaminophen is usually the first choice for mild aches and fever because it is gentle on the stomach. Take it with water, and never combine two products that both contain acetaminophen. The maximum daily analgesic limit for adults is 3,000 mg of acetaminophen in 24 hours; for ibuprofen, do not exceed 1,200 mg in 24 hours unless a clinician supervises a higher dose. Ibuprofen and naproxen are anti-inflammatory drugs. They should be taken with food, and patients with kidney disease, stomach ulcers or who take blood thinners should ask before using them. Children need weight-based dosing, so always check the label or call us.

## Antibiotics

Antibiotics treat bacterial infections only; they do nothing for colds or flu, which are viral. Finish the full course even if you feel better after a few days, and never share leftover antibiotics. For acute bacterial sinusitis, first-line treatment is amoxicillin 500 mg three times a day for 5 to 7 days. Most sinus infections are viral and improve within ten days without antibiotics. For patients with a penicillin allergy, doxycycline 100 mg twice a day is the preferred alternative for bacterial sinusitis; do not prescribe amoxicillin or any other penicillin. Common side effects of antibiotics include an upset stomach and diarrhea. Taking a probiotic or eating yogurt may help. Call the clinic if you develop a rash, swelling or trouble breathing.

## Allergies and Your Record

Tell us about every drug allergy, even mild ones. Your allergy list is stored in your patient record and is checked by our pharmacist before every new prescription. A true penicillin allergy usually means avoiding amoxicillin and ampicillin as well.

Seasonal allergies can be treated with over-the-counter antihistamines such as loratadine or cetirizine. Some antihistamines cause drowsiness, so take them at night the first time you use them.

## Appointments and Refills

Same-day appointments open at 8 a.m. every day of the week. You can book online, through the patient app, or by calling the front desk. Please arrive ten minutes early to take care of paperwork.

Prescription refills take two business days. Request refills through the patient app so that your clinician can take a look at your record before approving them. Controlled medications need an in-person visit every three months.
"""

# The long-term memory (LTM) the agent starts with. In production this would
# live in a database keyed by user_id and be loaded at the start of a turn.
INITIAL_LTM = {
    "user_id": "patient-0042",
    "medical_history": [
        "Mild asthma (uses an albuterol inhaler)",
        "Seasonal allergies",
    ],
    "preferences": [
        "Prefers short, plain-language answers",
    ],
}

# Scripted conversation. `needle` is the ground-truth sentence (must appear
# verbatim in CLINIC_GUIDE) that a correct answer depends on. Turn 2 has no
# needle: it is about WRITING memory, not retrieving it.
SCRIPTED_TURNS = [
    {
        "prompt": "What's the most painkiller I can take in a day?",
        "needle": (
            "The maximum daily analgesic limit for adults is 3,000 mg of "
            "acetaminophen in 24 hours; for ibuprofen, do not exceed 1,200 mg "
            "in 24 hours unless a clinician supervises a higher dose."
        ),
        "lesson": "Paraphrased question. Semantic embeddings bridge 'painkiller/day' ↔ 'analgesic/24 hours'.",
    },
    {
        "prompt": "By the way, I'm allergic to penicillin.",
        "needle": None,
        "lesson": "No retrieval needed. The Memory Writer commits a new fact to long-term memory.",
    },
    {
        "prompt": "Which antibiotic should I take for a sinus infection?",
        "needle": (
            "For patients with a penicillin allergy, doxycycline 100 mg twice a "
            "day is the preferred alternative for bacterial sinusitis; do not "
            "prescribe amoxicillin or any other penicillin."
        ),
        "lesson": "The safe answer needs LTM (the allergy) AND the right chunk. Without either, the agent would suggest amoxicillin.",
    },
]


def needle_span(doc: str, needle: str | None) -> tuple[int, int] | None:
    """Return the (start, end) character span of the needle inside `doc`.

    Returns None when there is no needle, or when the user swapped in a custom
    document that does not contain it (the health check then says 'n/a').
    """
    if not needle:
        return None
    start = doc.find(needle)
    if start < 0:
        return None
    return (start, start + len(needle))
