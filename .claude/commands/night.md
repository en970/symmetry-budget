---
description: Dispatch pending cells and hand off to the overnight loop
---

1. `python3 tools/status.py` — confirm which phase is open.
2. Dispatch pending cells for that phase only, using the `cell-runner` subagent, respecting
   the Kaggle concurrency limit (default 2 simultaneous kernels).
3. Record what was dispatched in `results/DISPATCH.log` with timestamps.
4. Report the count dispatched and the expected completion window.

Never dispatch a later phase while an earlier one is open — the phases are ordered in
PROTOCOL.md and out-of-order execution breaks the stopping rule.
