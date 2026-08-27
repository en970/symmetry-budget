#!/usr/bin/env python3
"""Expand the pre-registered protocol into an explicit cell list.

The grid is generated, never hand-written: PROTOCOL.md fixes the axes, and this
script is the only thing allowed to turn them into runnable cells. If the two
disagree, the protocol wins and this file is the bug.
"""
import json, hashlib, itertools, pathlib, sys

N_VALUES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
SEEDS = [0, 1]

PHASES = {
    1: dict(models=["M0", "M1", "M2", "M3"], breaks=["none"],                 budget="params"),
    2: dict(models=["M1", "M3"],             breaks=["acceptance", "axis"],   budget="params"),
    3: dict(models=["M1", "M3"],             breaks=["none"],                 budget="flops"),
}


def cell_id(c):
    """Stable short id: same spec always maps to the same filename."""
    key = f"{c['phase']}|{c['model']}|{c['n']}|{c['seed']}|{c['break']}|{c['budget']}"
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def build(phase):
    spec = PHASES[phase]
    cells = []
    for model, n, seed, brk in itertools.product(spec["models"], N_VALUES, SEEDS, spec["breaks"]):
        c = dict(phase=phase, model=model, n=n, seed=seed,
                 **{"break": brk}, budget=spec["budget"])
        c["id"] = cell_id(c)
        cells.append(c)
    return cells


def main():
    phases = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    cells = [c for p in phases for c in build(p)]
    out = pathlib.Path("experiments/grid.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cells, indent=2) + "\n")
    for p in phases:
        print(f"  phase {p}: {sum(1 for c in cells if c['phase'] == p):3d} cells")
    print(f"total {len(cells)} cells -> {out}")


if __name__ == "__main__":
    main()
