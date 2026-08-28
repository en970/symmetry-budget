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
