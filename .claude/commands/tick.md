---
description: One turn of the autonomous loop — collect, dispatch, close a phase if done
---

Run `python3 tools/tick.py --phase 1 --limit 2`.

Interpret its exit code, do not improvise around it:

- **0** — normal turn. Report in one line what was collected and dispatched. Nothing else.
- **3** — the loop has stopped on purpose. Two possible reasons, and they are handled
  differently:
  - *Phase closed.* Do NOT write any prose about the numbers. Launch the `result-auditor`
    subagent first (`/audit`), then relay its verdict.
  - *Failure rate above 20%.* The phase is inconclusive under PROTOCOL §5. Report the failure
    reasons found in `results/*.json`, and do not dispatch anything further.

Never edit a result file to make it pass validation, never retry a cell with different
settings, and never extend the grid because a result looked promising. If something needs a
judgement call, stop and say what it is.
