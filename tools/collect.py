"""Pull finished Kaggle runs into results/ and validate them on arrival.

Validation happens here as well as in the write hook, because results that come
back from Kaggle are not written through the model's tools and would otherwise
bypass the gate entirely.
"""
from __future__ import annotations

import argparse, json, pathlib, subprocess, sys, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.kaggle_common import kernel_status  # noqa: E402
from tools.validate_result import validate     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PENDING = ROOT / "results" / "PENDING.json"


def fetch_outputs(ref: str, cells: list[dict]) -> list[pathlib.Path]:
    """Pull every result file a grouped kernel produced. A kernel that died
    part-way still delivers the cells that finished before it."""
    got = []
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["kaggle", "kernels", "output", ref, "-p", td],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return got
        for cell in cells:
            src = pathlib.Path(td) / f"{cell['id']}.json"
            if src.exists():
                dest = ROOT / "results" / f"{cell['id']}.json"
                dest.write_text(src.read_text())
                got.append(dest)
    return got


def _token_of(cell_id: str) -> str | None:
    """The run token stamped into a freshly fetched result, if any."""
    try:
        return json.loads((ROOT / "results" / f"{cell_id}.json").read_text()).get("run_token")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-failed", action="store_true",
                    help="leave errored kernels in PENDING for inspection")
    a = ap.parse_args()

    pending = json.loads(PENDING.read_text()) if PENDING.exists() else {}
    if not pending:
        print("nothing pending")
        return

    still, collected, rejected = {}, 0, 0
    for ref, entry in pending.items():
        cells = entry.get("cells", [entry])          # tolerate the old single-cell format
        state = kernel_status(ref)
        if state in ("running", "queued", "unknown"):
            still[ref] = entry
            print(f"  {state:<8} {ref}  {entry.get('label', '')}")
            continue

        got = {p.stem for p in fetch_outputs(ref, cells)}

        # Kaggle serves the last completed output for a slug, and re-pushing does
        # not clear it. Without an identity check the collector reads a previous
        # version's results back as though they were this dispatch's — which is
        # exactly what happened to the twelve Phase 3 M1 cells on 2026-08-28,
        # silently undoing amendment A2. Cells dispatched before tokens existed
        # have no expectation recorded and are collected as before.
        want = entry.get("run_token")
        if want:
            stale = [c["id"] for c in cells if c["id"] in got and _token_of(c["id"]) != want]
            if stale:
                for cid in stale:
                    (ROOT / "results" / f"{cid}.json").unlink(missing_ok=True)
                still[ref] = entry
                print(f"  STALE    {ref}  {entry.get('label', '')}\n"
                      f"      {len(stale)} output(s) carry an older run token; expected "
                      f"{want}. Kernel state was '{state}'. Left pending, not collected.")
                continue

        for cell in cells:
            dest = ROOT / "results" / f"{cell['id']}.json"
            if cell["id"] not in got:
                # A cell the kernel never produced is data too, not a silent gap.
                dest.write_text(json.dumps({
                    **{k: cell[k] for k in ("id", "phase", "model", "n", "seed", "break", "budget")},
                    "status": "failed",
                    "failure_reason": f"kernel {state}; no output for this cell (check Kaggle logs)",
                }, indent=2) + "\n")
                print(f"  NO OUTPUT {cell['id']}  {cell['model']} n={cell['n']}")
                continue
            errs = validate(str(dest))
            if errs:
                rejected += 1
                print(f"  REJECTED  {cell['id']}")
                for e in errs:
                    print(f"      - {e}")
                dest.unlink()
            else:
                collected += 1
        print(f"  {state:<8} {ref}  {entry.get('label', '')}")

    PENDING.write_text(json.dumps(still, indent=2) + "\n")
    print(f"\n{collected} collected, {rejected} rejected, {len(still)} still pending")


if __name__ == "__main__":
    main()
