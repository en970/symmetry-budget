---
description: Start the overnight autonomous loop over the open phase
---

Start a `/loop` that runs `/tick` every 30 minutes.

Thirty minutes: a cell takes roughly that long on a Kaggle GPU, and it matches the standing
rule that long-running work is committed every half hour, so each turn produces at most one
checkpoint commit. Polling faster spends tokens to learn nothing.

The loop must stop — not continue, not work around — when `tools/tick.py` exits 3. That is
the designed handoff back to a human, for exactly the two cases in `/tick`.

Before starting, confirm: a phase is open, Kaggle quota remains, and the last dispatched cell
did not fail. Report those three, then start.
