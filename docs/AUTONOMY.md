# How this repository is run

The grid is 120 independent cells. Nothing about running them requires judgement, and
everything about interpreting them does. The machinery below is arranged around that split:
execution is automated aggressively, interpretation is gated deliberately.

## The layers

| layer | file | what it does | why it exists |
|---|---|---|---|
| Pre-registration | `PROTOCOL.md` | fixes hypotheses, grid, metrics, stopping rule | an agent that can choose its success criterion after seeing results will always succeed |
| Grid expansion | `tools/grid.py` | protocol → 120 explicit cells with stable ids | the cell list is derived, never hand-maintained, so it cannot drift from the protocol |
| Execution | `.claude/agents/cell-runner.md` | runs exactly one cell, records failures honestly | narrow scope; a runner that may "improve things" corrupts the comparison |
| Validation | `tools/validate_result.py` + `PostToolUse` hook | rejects malformed or implausible results at write time | the harness enforces this, not the model's good intentions |
| Progress | `tools/status.py` | counts only, no metrics while a phase is open | prevents the loop from steering toward whichever axis looked promising |
| Audit | `.claude/agents/result-auditor.md` | tries to refute each finding | a separate context with no stake in the earlier work |
| Report | `tools/report.py` | regenerates from `results/` | no hand-edited numbers, ever |

## The tools, and when each is worth it

**`/loop`** — re-runs a prompt on an interval, or lets the model pace itself. Right for
*"dispatch pending cells, collect what finished, stop when the phase closes"*: a genuinely
repeating task with a termination condition. Wrong as a way to wait for something — polling a
job every 60s burns tokens to learn nothing. Give it a real interval and a real exit.

**Workflow (fan-out)** — a script that spawns many subagents deterministically. This project's
natural fit: 48 cells, no dependencies, one agent each, results collected at a barrier. Worth
it when the work is genuinely parallel and each unit is self-contained. Not worth it for
sequential work with one bottleneck — the orchestration then costs more than it saves.

**Cron** — scheduled runs. Right for *"at 03:00, dispatch whatever is pending"* when Kaggle
quota resets on a clock. Wrong for anything needing a decision, because nobody is there to
make it. Pair every cron with a report you actually read in the morning.

**Hooks** — shell commands the harness runs on tool events, outside the model's control. The
single highest-value piece of automation here, because it is the only layer the model cannot
talk its way past. Use for validation and gates. Do not use for anything slow: it runs on
every matching tool call.

**Subagents** — a fresh context with its own instructions. Two uses that matter: parallelism
(`cell-runner`) and **independence** (`result-auditor`). The second is the important one. An
auditor that did not write the code has no sunk cost in the result being real.

**`continuum`** — carries long work across session boundaries and usage limits. Worth loading
when a phase takes longer than one session, which Phase 1 will.

## The failure mode this is built against

An autonomous loop optimises for finishing. Left alone it will retry a failing cell with a
smaller model, drop the cells that crashed, notice that one N looks promising and add points
there, and write a report about a real-looking effect. Every one of those steps is locally
reasonable and collectively fatal.

The guard rails are therefore structural, not advisory:

- the success criteria were frozen before the first run (`PROTOCOL.md`)
- results are validated by a hook, outside the model's control
- progress reporting withholds metrics until a phase closes
- findings are audited by an agent whose instructions are to refute them
- failures are recorded as data, and >20% failures makes a phase inconclusive rather than
  interesting

## Running it

```bash
make grid       # expand protocol into experiments/grid.json
make status     # counts only
make submit     # dispatch pending cells for the open phase
make collect    # pull finished runs
make audit      # adversarial review of a closed phase
make report     # regenerate reports/ from results/
```

In a Claude Code session: `/grid` to see state and next action, `/night` to dispatch, `/audit`
when a phase closes.
