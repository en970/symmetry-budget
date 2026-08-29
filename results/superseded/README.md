# Superseded results

Not failures. These cells ran to completion against a specification that was later
found to be wrong, and are kept because PROTOCOL §5 does not permit dropping a cell
silently. They are in a subdirectory so the `results/*.json` glob used by status.py,
audit.py, report.py and validate_result.py does not count them.

## 2026-08-28 — Phase 3 M1, superseded by amendment A2

Run with `FLOPS_MATCHED` at hidden=1396, which reaches 789,281,648 FLOPs/forward — 74%
of M3's 1,069,142,976, not the match A1 committed Phase 3 to. Re-run at hidden=1625
(1,069,094,000, 99.995%). See PROTOCOL.md amendment A2.

| cell | N | seed | flops/fwd | params |
|---|---|---|---|---|
| `2aa30a7690` | 100,000 | 0 | 789,281,648 | 7,814,810 |
| `3ba777f949` | 3,000 | 1 | 789,281,648 | 7,814,810 |
| `6cb3115442` | 300,000 | 1 | 789,281,648 | 7,814,810 |
| `7402d32298` | 1,000 | 1 | 789,281,648 | 7,814,810 |
| `83c460d28a` | 10,000 | 1 | 789,281,648 | 7,814,810 |
| `83f15d4500` | 30,000 | 1 | 789,281,648 | 7,814,810 |
| `86f0eebfe0` | 1,000 | 0 | 789,281,648 | 7,814,810 |
| `aabd1920a4` | 300,000 | 0 | 789,281,648 | 7,814,810 |
| `b20cba7c32` | 3,000 | 0 | 789,281,648 | 7,814,810 |
| `b918007d24` | 10,000 | 0 | 789,281,648 | 7,814,810 |
| `c0b36ef067` | 100,000 | 1 | 789,281,648 | 7,814,810 |
| `c698c338e4` | 30,000 | 0 | 789,281,648 | 7,814,810 |

## 2026-08-28 — the re-run that never happened

The twelve cells above were re-dispatched at 19:43 after A2, and `collect.py` reported
them collected at 20:14. They were not re-run: the returned files are byte-identical to
the superseded copies, `train_seconds` included, which no genuine re-execution produces.
Kaggle serves the last completed output for a slug and re-pushing does not clear it, and
`collect.py` had no way to tell which run an output came from.

`audit-phase3-superseded.json` is the H3 verdict computed from those stale cells
("falsified — advantage is stable at every tested size"). It is kept as the record of
what the under-provisioned baseline produced, and is not a result.

Fixed by stamping a per-dispatch `run_token` into every result and rejecting outputs
that do not carry the expected one.
