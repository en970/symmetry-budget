---
description: Show grid progress and what the loop should do next
---

Run `python3 tools/status.py`.

Then state, in at most three lines, what the next action is:

- Cells still pending in the open phase → how many, and dispatch the next batch.
- Phase complete → run the `result-auditor` subagent on it before writing any prose.
- Phase inconclusive (>20% failures) → report that; do not analyse the survivors.

Do not report or speculate about metrics while a phase is open. PROTOCOL.md §5.
