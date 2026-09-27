# MemPulse Student Guide

### Seeing what an AI "remembers", one step at a time

**Time needed:** about 45 minutes
**You don't need:** any coding. You only click buttons and read the screen.

---

## Before you start: the big idea

When you ask an AI assistant a question about a document, it does **not** read the whole document.
Here is what happens instead:

1. The document is cut into small pieces called **chunks**.
2. Each chunk is turned into a list of numbers called an **embedding** (think of it as a "meaning fingerprint").
3. Your question gets a fingerprint too. The computer finds the chunks whose fingerprints are **most similar**.
4. Only those few chunks, plus a few notes about you, are pasted into a prompt for the AI.
5. The AI answers using **only what's in that prompt**.

So the AI's "memory" is whatever made it into that prompt. If the right information gets lost at
any step, the AI can't answer correctly, no matter how smart it is.

**MemPulse lets you watch this happen step by step.**

### Words you'll see

| Word | Simple meaning |
|---|---|
| **Chunk** | A small piece of the document (a few sentences). |
| **Chunk size** | How many characters go in each chunk. |
| **Overlap** | Text shared between two neighbouring chunks, like a safety margin. |
| **Embedding / vector** | A list of numbers that captures what a piece of text *means*. |
| **Similarity** | How close two fingerprints are, from 0 (unrelated) to 1 (identical meaning). |
| **Top-k** | How many of the best-matching chunks get used (here, 3). |
| **Context window** | The prompt the AI actually reads. It has a size limit (the **budget**). |
| **Evicted** | Kicked out of the prompt because there wasn't room. |
| **Long-term memory** | Facts saved about you that last between conversations (like an allergy). |
| **Short-term memory** | The recent back-and-forth of this conversation. |

---

## Part 0: Open the app

1. Your teacher will start the app, or you run `uv run streamlit run app.py` in the project folder.
2. Open the link it prints (usually **http://localhost:8501**).
3. **Wait about 20 seconds** the first time. Clicks during loading are ignored.

### Find your way around the screen

- **Top row of boxes (the pipeline):** every step the system goes through. The **glowing blue box** is where you are now.
- **Buttons:** **◀ Prev** and **Next ▶** move one step. **⏭ Run next turn** runs a whole question. **↺ Reset** starts over.
- **Blue bar under the buttons:** explains the current step in one sentence. **Always read it.**
- **Left column (Memory stack):** how full each type of memory is right now.
- **Middle (Stage lens):** a picture of what this step is doing.
- **Right column (State diff):** what just changed. Green `+` = new, yellow `~` = changed.
- **Bottom (Why memory state matters):** a report card for the current question and "what if" experiments.
- **Left sidebar:** settings you'll change later (chunk size, overlap, embedder...).

---

## Part 1: Watch the document become memory

You start on **Step 1 · Load**.

**Step 1: Load**
- Look at the middle panel: it's the whole clinic guide as plain text.
- 👀 **Look for:** one sentence is **underlined in gold**. That's the answer to Question 1 (the maximum daily painkiller dose). We'll follow this sentence the whole way.
- 👀 In the left column, *Knowledge · vector store* says **raw text**: nothing is searchable yet.

Click **Next ▶**.

**Step 2: Chunk**
- The document is now coloured in bands. Each band is one chunk, labelled `C00`, `C01`, `C02`...
- 👀 **Look for yellow highlighted text.** That's **overlap**: text stored in *two* chunks.
- 👀 **Look at the tile "Answer fact".** It should say **intact**, meaning the gold sentence fits completely inside one chunk.
- 👀 If you ever see a **✂** inside the gold sentence, a chunk boundary cut through it.

> ✏️ **Write down:** How many chunks are there? ____ What's the average chunk length? ____

Click **Next ▶**.

**Step 3: Embed**
- The top picture is a grid of colours: each row is one chunk, each column is one of its numbers.
- The bottom picture puts every chunk on a map. **Chunks about similar topics sit closer together.**
- 👀 In the left column, the vector store now shows **11 × 384**: 11 chunks, 384 numbers each.

Click **Next ▶**.

**Step 4: Index**
- The chunks and their numbers are stored and ready to search. Ingestion is done!

---

## Part 2: Ask the three questions

Click **Next ▶** again. MemPulse starts **Turn 1** automatically.

### Turn 1: "What's the most painkiller I can take in a day?"

Step through each box with **Next ▶** and look for these things:

**Recall LTM**
- The AI loads what it already knows about the patient (asthma, seasonal allergies, likes short answers).

**Retrieve** ⭐ *the most important step*
- **Left chart:** the map again, now with a **yellow star ★** for your question. Dotted lines connect it to the 3 chunks it picked.
- 👀 One chunk has a **gold ring**: that's the chunk holding the answer.
- **Right chart:** every chunk ranked by similarity. Blue bars made it in; grey bars didn't. The **red dashed line** is the top-3 cut-off.
- 👀 **Is the gold bar above the red line?** If yes, the answer got retrieved.

> 🤔 **Notice:** your question says "painkiller" and "day", but the document says "analgesic" and "24 hours".
> The embedding still matched them because it understands **meaning, not just exact words**.

**Assemble**
- This is **the exact prompt** the AI will read, colour-coded by where each piece came from:
  grey = instructions, purple = long-term memory, blue = document chunks, teal = conversation, yellow = your question.
- 👀 In the left column, find the **Context window** bar. How full is it? Anything marked **⛔ evicted** didn't fit.

**LLM Reason**
- The AI's answer. Below it: which chunks were in its prompt.
- 👀 Open **"Exact prompt sent to the model"** to see *everything* the AI saw. That was its entire memory for this answer.

**Write Memory**
- The AI checks whether your message contained something worth remembering long-term. For this question, nothing. Only the short-term history grew.

### Turn 2: "By the way, I'm allergic to penicillin."

Click **⏭ Run next turn**. It jumps to the **Write Memory** step.

- 👀 **Left column, Long-term memory:** a new fact appears in **green** with a **＋**. The AI just learned something that will last.
- 👀 Short-term memory now holds 4 messages.

> 🤔 **Bonus puzzle (if Claude is turned on):** press **◀ Prev** to go to Turn 2's **LLM Reason** step.
> The AI may say the allergy "isn't in your record yet". Look at the long-term memory card at that step. **Why is the AI saying that?**
> *(Hint: which comes first in the pipeline, Reason or Write Memory?)*

### Turn 3: "Which antibiotic should I take for a sinus infection?"

Click **⏭ Run next turn**, then **◀ Prev** once to see **LLM Reason**.

- 👀 **The trap:** the guide says the normal first choice is **amoxicillin**, which is a penicillin! This patient is allergic.
- 👀 A safe answer needs **two memories working together**:
  1. **Long-term memory:** knows about the penicillin allergy (from Turn 2).
  2. **Retrieved chunk:** contains the safe alternative (**doxycycline**).
- Check the answer: does it recommend doxycycline and warn against amoxicillin?

> ✏️ **Write down:** Which chunks were in the prompt for Turn 3? ________

---

## Part 3: The report card

Scroll to the bottom: **Why memory state matters**. It grades the current question.

| Tile | What it means | Good sign |
|---|---|---|
| **Verdict** | Did the answer sentence reach the AI? | ✅ OK |
| **Answer fact intact** | How much of the answer sentence the AI could see in one piece | 100% |
| **Answer chunk rank** | Where the answer chunk ranked out of all chunks | #3 or better (top-k = 3) |
| **Top-1 sim · margin** | Best similarity score, and how far ahead of 2nd place | a bigger margin = more confident |
| **Signal / noise** | How much of the retrieved text is actually the answer | higher = less clutter |
| **Context used** | How much of the prompt budget is used | under the budget |

### The four ways memory can fail

| Verdict | What went wrong, in plain words |
|---|---|
| ✂️ **split by chunker** | The answer sentence was cut in half, so no single chunk has the whole thing. |
| 🎯 **missed by retrieval** | A chunk had the answer, but it didn't rank in the top 3. |
| 🚫 **evicted from context** | The answer was found but thrown out because the prompt was full. |
| ⚠️ **only partly in context** | The AI saw just a piece of the answer. |

---

## Part 4: Experiments (break it on purpose!)

This is where you learn **why memory state matters**. Stay on a Turn 3 step. The **What-if Lab** table
at the bottom shows the *same question* with different settings. Read the **Verdict** column first.

For each experiment, click its **Apply ▸** button. The whole conversation is **replayed** with the new setting.
Then step through Turn 3 again and fill in the table.

| Experiment | Button | What to look at | Your notes |
|---|---|---|---|
| **A. Bad embedder** | Apply ▸ Hashing | Retrieve step: where is the gold bar now? What does the answer talk about? | |
| **B. Tiny chunks** | Apply ▸ Tiny | Chunk step: how many chunks now? Is the gold sentence split (✂)? | |
| **C. No overlap** | Apply ▸ No overlap | Chunk step: find the ✂ inside the gold sentence. Is there any yellow? | |
| **D. Giant chunks** | Apply ▸ Giant | Assemble step: which blocks are crossed out as ⛔ EVICTED? | |

To go back to normal: set the sidebar to **chunk_size 400**, **chunk_overlap 60**, **bge-small**, or refresh the page.

### What you should discover

- **A. Bad embedder:** the "hashing" embedder only matches **exact words**. "Antibiotic for a sinus infection" doesn't match the right chunk's wording, so the AI gets chunks about refills and antihistamines and **can't give safe advice**.
- **B. Tiny chunks:** facts get chopped up. The AI may see only part of the key sentence, for example a dose without the words saying who it is for.
- **C. No overlap:** one unlucky cut through the key sentence and it's broken everywhere. **Overlap is insurance.**
- **D. Giant chunks:** each chunk is so big that it doesn't fit in the prompt budget, so it gets **evicted**. Also, a giant chunk's fingerprint is a blurry average of many topics.

### Bonus: find the sweet spot

1. At the bottom right, click **▦ Run sensitivity sweep**. Wait 20–40 seconds.
2. You'll see a grid: **chunk size** across, **overlap** down. **Green** = the answer survives, **red** = it doesn't.
3. The white square □ shows your current settings.

> 🤔 **Look for:** Is there one perfect setting, or a *zone* of good settings? What happens at the far left (tiny) and far right (giant)?

### Bonus: your own settings

Use the sidebar sliders yourself:
- Lower the **Context budget** to 300. What gets evicted first: old conversation, or document chunks?
- Change **top-k** to 1. Does Turn 3 still work?
- Switch **Strategy** to **heading** (one chunk per document section). Does the verdict change?

---

## Part 5: Reflection questions

Answer these in your own words:

1. The AI in Turn 3 is the same "smart" model in every experiment. So why does it give a good answer sometimes and a bad one other times?
2. Why do you think real companies use **overlap** even though it stores some text twice?
3. What's the difference between **long-term memory** and **retrieved chunks**? Why did Turn 3 need both?
4. If an AI gave a wrong answer in real life, which screen in MemPulse would you check first, and why?
5. In one sentence: **why is it valuable to be able to see an AI's memory?**

---

## Quick troubleshooting

| Problem | Fix |
|---|---|
| Buttons don't respond | The app is still loading (first run takes ~20 s). Wait, then click again. |
| Header says "Claude off · offline answers" | No API key found. The app still works; answers are copied from the best-matching sentences instead of written by Claude. |
| I'm lost | Click **↺ Reset** and start from Part 1. |
| Everything looks broken after experiments | Refresh the browser page to return to default settings. |
