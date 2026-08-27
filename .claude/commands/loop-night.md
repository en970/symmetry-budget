---
description: Start the overnight autonomous loop over the open phase
---

Start a `/loop` that runs `/tick` every 20 minutes.

Twenty minutes because a cell takes roughly that long on a Kaggle GPU; polling faster spends
tokens to learn nothing, and the concurrency limit means there is rarely more than one slot
to fill per turn.

The loop must stop — not continue, not work around — when `tools/tick.py` exits 3. That is
the designed handoff back to a human, for exactly the two cases in `/tick`.

Before starting, confirm: a phase is open, Kaggle quota remains, and the last dispatched cell
did not fail. Report those three, then start.
