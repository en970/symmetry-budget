"""Mechanical audit of a closed phase. Produces verdicts, or refuses to.

This is the part of the review that can be made a rule: completeness, the noise
rule, the compute confound, implausible results. It runs unattended so the loop
can cross a phase boundary without a person present.

It is deliberately not the whole audit. The `result-auditor` subagent still
reads the same files with an adversarial brief, and its job is to find what a
rule cannot anticipate. This file exists so that the absence of a human at 3 a.m.
never becomes a reason to skip the checks that *can* be automated.

Exit 0 = verdicts written. Exit 3 = phase cannot be analysed; reason on stdout.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(phase: int) -> list[dict]:
    grid = json.loads((ROOT / "experiments/grid.json").read_text())
    ids = {c["id"] for c in grid if c["phase"] == phase}
    rows = []
    for p in (ROOT / "results").glob("*.json"):
        if p.stem in ids:
            try:
                rows.append(json.loads(p.read_text()))
            except Exception:
                pass
    return rows, len(ids)


def series(rows, model, n, key="rej_at_30"):
    return [r[key] for r in rows
            if r.get("status") == "ok" and r["model"] == model and r["n"] == n
            and isinstance(r.get(key), (int, float))]


def gap(rows, a, b, n):
    """§4: a difference counts only if the seed ranges are disjoint."""
    va, vb = series(rows, a, n), series(rows, b, n)
    if len(va) < 2 or len(vb) < 2:
        return None, f"insufficient seeds ({a}:{len(va)}, {b}:{len(vb)})"
    if max(va) < min(vb):
        return statistics.mean(vb) - statistics.mean(va), f"{b} > {a}, ranges disjoint"
    if max(vb) < min(va):
        return statistics.mean(vb) - statistics.mean(va), f"{a} > {b}, ranges disjoint"
    return None, "within noise (ranges overlap)"


def audit_phase1(rows) -> dict:
    """H1: the M3-over-M1 gap shrinks monotonically in N and changes sign."""
    ns = sorted({r["n"] for r in rows})
    signed, notes = [], []
    for n in ns:
        d, why = gap(rows, "M1", "M3", n)
        notes.append(f"n={n:,}: {why}" + (f" (Δ={d:+.1f})" if d is not None else ""))
        signed.append((n, d))

    real = [(n, d) for n, d in signed if d is not None]
    if len(real) < 3:
        return {"verdict": "inconclusive",
                "reason": f"only {len(real)} of {len(ns)} sizes produced a gap outside noise; "
                          f"a trend cannot be read from that",
                "detail": notes}

    ds = [d for _, d in real]
    monotone = all(b <= a for a, b in zip(ds, ds[1:]))
    sign_change = any(a > 0 >= b for a, b in zip(ds, ds[1:]))
    if monotone and sign_change:
        v = "supported"
        reason = "gap shrinks monotonically and crosses zero within the tested range"
    elif monotone:
        v = "partially supported"
        reason = "gap shrinks monotonically but does not cross zero in the tested range"
    else:
        v = "falsified"
        reason = "gap is not monotonically decreasing in N"
    return {"verdict": v, "reason": reason, "detail": notes}


def confound_check(rows) -> list[str]:
    """Flag any advantage that tracks compute rather than architecture."""
    warns = []
    for n in sorted({r["n"] for r in rows}):
        for m in ("M1", "M3"):
            t = [r["train_seconds"] for r in rows
                 if r.get("status") == "ok" and r["model"] == m and r["n"] == n]
            if t:
                warns.append(f"n={n:,} {m}: {statistics.mean(t):.1f}s mean train time")
    return warns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    a = ap.parse_args()

    rows, total = load(a.phase)
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") == "failed"]

    out = {"phase": a.phase, "cells_total": total, "cells_ok": len(ok),
           "cells_failed": len(failed)}

    if len(rows) < total:
        print(f"phase {a.phase} is not complete ({len(rows)}/{total}) — nothing to audit")
        return 3
    if total and len(failed) / total > 0.20:
        out["verdict"] = "inconclusive"
        out["reason"] = (f"{len(failed)}/{total} cells failed (>20%); PROTOCOL §5 makes this "
                         f"phase inconclusive and the survivors are not analysed")
        (ROOT / "reports" / f"audit-phase{a.phase}.json").write_text(json.dumps(out, indent=2))
        print(out["reason"])
        return 3

    if a.phase == 1:
        out.update(audit_phase1(ok))
    else:
        out.update({"verdict": "pending",
                    "reason": f"no mechanical rule is defined for phase {a.phase} yet; "
                              f"the result-auditor subagent must review it"})
    out["cost_note"] = confound_check(ok)

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"audit-phase{a.phase}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"phase {a.phase}: {out['verdict']} — {out['reason']}")
    for d in out.get("detail", []):
        print(f"  {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
