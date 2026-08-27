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


def fetch_output(ref: str, dest: pathlib.Path) -> pathlib.Path | None:
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["kaggle", "kernels", "output", ref, "-p", td],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None
        src = pathlib.Path(td) / "result.json"
        if not src.exists():
            return None
        dest.write_text(src.read_text())
        return dest


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
    for ref, cell in pending.items():
        state = kernel_status(ref)
        if state in ("running", "queued", "unknown"):
            still[ref] = cell
            print(f"  {state:<8} {ref}")
            continue

        dest = ROOT / "results" / f"{cell['id']}.json"
        got = fetch_output(ref, dest)
        if got is None:
            # The kernel finished without producing a result: that is itself data.
            dest.write_text(json.dumps({
                **{k: cell[k] for k in ("id", "phase", "model", "n", "seed", "break", "budget")},
                "status": "failed",
                "failure_reason": f"kernel {state} with no result.json (check Kaggle logs)",
            }, indent=2) + "\n")
            print(f"  NO OUTPUT {ref} -> recorded as failed")
            if a.keep_failed:
                still[ref] = cell
            continue

        errs = validate(str(dest))
        if errs:
            rejected += 1
            print(f"  REJECTED  {ref}")
            for e in errs:
                print(f"      - {e}")
            dest.unlink()
            still[ref] = cell
        else:
            collected += 1
            print(f"  collected {ref}")

    PENDING.write_text(json.dumps(still, indent=2) + "\n")
    print(f"\n{collected} collected, {rejected} rejected, {len(still)} still pending")


if __name__ == "__main__":
    main()
