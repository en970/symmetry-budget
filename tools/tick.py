"""One turn of the autonomous loop: collect, dispatch, and cross phase boundaries.

Runs unattended, including across phase boundaries. Every decision is mechanical;
the two that are not — an inconclusive phase, and the final interpretation — stop
the loop instead of being guessed.

Exit codes:
    0  ran normally; there is more to do
    3  stopped and wants a human — reason on stdout
"""
from __future__ import annotations

import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sh(*cmd: str) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r.returncode, (r.stdout + r.stderr).strip()


def counts(phase: int) -> tuple[int, int, int]:
    grid = json.loads((ROOT / "experiments/grid.json").read_text())
    ids = {c["id"] for c in grid if c["phase"] == phase}
    done = failed = 0
    for p in (ROOT / "results").glob("*.json"):
        if p.stem not in ids:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        done += 1
        failed += d.get("status") == "failed"
    return done, failed, len(ids)


def phases() -> list[int]:
    grid = json.loads((ROOT / "experiments/grid.json").read_text())
    return sorted({c["phase"] for c in grid})


def open_phase() -> int | None:
    """The first phase with cells still outstanding. Phases run in order (§5)."""
    for ph in phases():
        done, _, total = counts(ph)
        if done < total:
            return ph
    return None


def audited(phase: int) -> bool:
    return (ROOT / "reports" / f"audit-phase{phase}.json").exists()


def unaudited_phase() -> int | None:
    """The first phase whose cells are all in but which has never been audited.

    This has to be its own check rather than a `done >= total` test on the open
    phase. A single turn's collect() can land the last cells of one phase while
    the next is already dispatched, and open_phase() then names the successor —
    so the phase that just finished is never the one under examination, and it
    crosses the boundary unaudited. Phase 1 did exactly that on 2026-08-28.
    """
    for ph in phases():
        done, _, total = counts(ph)
        if done >= total and not audited(ph):
            return ph
    return None


def checkpoint(msg: str, *, push: bool) -> None:
    sh("git", "add", "results", "reports")
    if sh("git", "diff", "--cached", "--quiet")[0] == 0:
        print("checkpoint: no change")
        return
    code, out = sh("git", "commit", "-m",
                   f"{msg}\n\nCheckpoint written by tools/tick.py.\n\n"
                   f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>")
    if code == 0:
        if push:
            sh("git", "push", "origin", "main")
        print(f"checkpoint committed{' and pushed' if push else ''}")
    else:
        print(f"checkpoint failed: {out.splitlines()[0] if out else '?'}")


def close_phase(phase: int, *, push: bool) -> int:
    """Audit a finished phase and decide whether the loop may continue."""
    print(f"\n— phase {phase} complete, auditing —")
    code, out = sh(sys.executable, "tools/audit.py", "--phase", str(phase))
    print(out)
    sh(sys.executable, "tools/report.py")

    if code == 3:
        checkpoint(f"Phase {phase} inconclusive", push=push)
        print(f"\nSTOP: phase {phase} cannot be analysed. A human should read the failure "
              f"reasons in results/ before anything else runs.")
        return 3

    checkpoint(f"Phase {phase} closed and audited", push=push)
    nxt = open_phase()
    if nxt is None:
        print("\nAll phases complete. STOP: the loop does not write the final "
              "interpretation. Run /audit for the adversarial review, then the report.")
        return 3
    print(f"\nphase {phase} closed; continuing with phase {nxt}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, help="pin to one phase (default: follow the open one)")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    print("— collect —")
    print(sh(sys.executable, "tools/collect.py")[1])

    # Before anything else: a phase that finished in an earlier turn and was never
    # audited is closed now. Auditing is not optional and not retrospective — no
    # later phase may be dispatched on top of an unexamined one.
    stale = unaudited_phase()
    if stale is not None and a.phase in (None, stale):
        return close_phase(stale, push=not a.no_push)

    phase = a.phase or open_phase()
    if phase is None:
        return close_phase(phases()[-1], push=not a.no_push)

    done, failed, total = counts(phase)
    print(f"\nphase {phase}: {done}/{total} returned, {failed} failed")

    # Checked before dispatching more: spending quota on a phase §5 already makes
    # inconclusive helps nobody.
    if total and failed / total > 0.20:
        checkpoint(f"Phase {phase} inconclusive ({failed}/{total} failed)", push=not a.no_push)
        print(f"\nSTOP: {failed}/{total} cells failed (>20%). PROTOCOL §5 makes this phase "
              f"inconclusive. Not dispatching further cells.")
        return 3

    if done >= total:
        return close_phase(phase, push=not a.no_push)

    print("\n— dispatch —")
    print(sh(sys.executable, "tools/submit.py", "--phase", str(phase),
             "--limit", str(a.limit), "--epochs", str(a.epochs))[1])
    checkpoint(f"Phase {phase} progress: {done}/{total} cells ({failed} failed)",
               push=not a.no_push)
    return 0


if __name__ == "__main__":
    sys.exit(main())
