#!/usr/bin/env python3
"""Grid progress — deliberately without metrics.

PROTOCOL.md §5 forbids analysing a phase before it is complete. So this reports
counts only. Peeking at partial effect sizes is how a grid quietly turns into a
search for the axis that looked promising, and the loop calls this many times a
night. The numbers unlock in `make report`, once the phase closes.
"""
import json, pathlib, sys

grid = json.loads(pathlib.Path("experiments/grid.json").read_text())
done, failed = {}, {}
for p in pathlib.Path("results").glob("*.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    (failed if d.get("status") == "failed" else done)[d.get("id")] = d

print(f"{'phase':>5}  {'done':>5} {'failed':>7} {'pending':>8} {'total':>6}   state")
any_open = False
for ph in sorted({c["phase"] for c in grid}):
    cells = [c for c in grid if c["phase"] == ph]
    ids = {c["id"] for c in cells}
    d, f = len(ids & done.keys()), len(ids & failed.keys())
    pend = len(ids) - d - f
    if pend:
        state, any_open = "running — results locked", True
    elif f / max(len(ids), 1) > 0.20:
        state = "INCONCLUSIVE (>20% failed, protocol §5)"
    else:
        state = "complete — ready to analyse"
    print(f"{ph:>5}  {d:>5} {f:>7} {pend:>8} {len(ids):>6}   {state}")

if any_open:
    print("\nMetrics are withheld until a phase closes (PROTOCOL.md §5).")
sys.exit(0)
