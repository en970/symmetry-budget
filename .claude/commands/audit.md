---
description: Adversarially audit a completed phase before it reaches the report
---

Launch the `result-auditor` subagent on the most recently completed phase (or the phase given
in $ARGUMENTS).

Give it: the phase number, the path to `results/`, and `PROTOCOL.md`.

Relay its verdict verbatim — supported / falsified / inconclusive per hypothesis, plus every
claim it rejected. Do not soften its language, and do not write the report section until the
audit has run.
