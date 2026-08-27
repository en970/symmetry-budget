"""Dispatch pending grid cells to Kaggle, one kernel per cell.

The repository source is vendored into each kernel rather than fetched at run
time: a training run must reproduce exactly the code that was current when it
was dispatched. Pulling from GitHub inside the kernel would silently mix code
versions across a grid that takes days to finish, and the resulting comparison
would not be between models but between commits.
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.kaggle_common import push_kernel, kernel_status, username  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PENDING = ROOT / "results" / "PENDING.json"
DATA_KERNEL = "sb-prepare-data"

HEADER = '''"""Auto-generated cell runner. Do not edit — regenerate with tools/submit.py."""
import os, sys, subprocess, pathlib, json

# Kaggle allocates a P100 (sm_60); the image's PyTorch 2.10 supports sm_70 and
# above, so every GPU cell dies on cudaErrorNoKernelImageForDevice. The GPU type
# cannot be chosen through the API (--accelerator is accepted and ignored, even
# for nonsense values), so the fix is to install a build that still supports
# sm_60. It must happen before torch is imported: reloading it in-process fails
# because the C extensions cannot re-register their namespaces.
if os.environ.get("SB_TORCH_READY") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    os.environ["SB_TORCH_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

SOURCE = {source!r}
for rel, text in SOURCE.items():
    p = pathlib.Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)

sys.path.insert(0, ".")
import torch
from src.train import run

CELL = {cell!r}
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| cell:", CELL, flush=True)

# Search rather than hard-code: Kaggle mounts a kernel's output under
# /kaggle/input/notebooks/<account>/<slug>/, not /kaggle/input/<slug>/. Searching
# keeps the account name out of the repository and survives Kaggle changing the
# layout again.
found = sorted(pathlib.Path("/kaggle/input").rglob("toptagging.npz"))
if not found:
    have = [str(p) for p in pathlib.Path("/kaggle/input").rglob("*")][:25]
    raise SystemExit(f"prepared dataset not found; /kaggle/input contains {{have}}")
data = str(found[0])
print("data:", data, flush=True)

try:
    result = run(CELL, data, epochs={epochs}, device=device)
except Exception as exc:
    # A failure is a recorded outcome, not a silent gap in the grid.
    result = {{**{{k: CELL[k] for k in ("id","phase","model","n","seed","break","budget")}},
              "status": "failed", "failure_reason": f"{{type(exc).__name__}}: {{exc}}"[:500]}}
    print("FAILED:", result["failure_reason"], flush=True)

pathlib.Path("/kaggle/working/result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
'''


def bundle() -> dict[str, str]:
    """Every source file the kernel needs, as {relative path: text}."""
    return {
        str(p.relative_to(ROOT)): p.read_text()
        for p in sorted(ROOT.glob("src/**/*.py"))
    }


def load_pending() -> dict:
    return json.loads(PENDING.read_text()) if PENDING.exists() else {}


def save_pending(d: dict) -> None:
    PENDING.parent.mkdir(exist_ok=True)
    PENDING.write_text(json.dumps(d, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--limit", type=int, default=2, help="max simultaneous kernels")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--cell", help="dispatch one specific cell id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # Dispatching before the data kernel has finished produces a grid of cells
    # that all fail on a missing mount — which is exactly how the first two cells
    # of this project were wasted. Check once, here, rather than per cell.
    data_ref = f"{username()}/{DATA_KERNEL}"
    state = kernel_status(data_ref)
    if state != "complete":
        print(f"data kernel {data_ref} is '{state}', not 'complete' — refusing to dispatch. "
              f"Cells would mount an empty input and fail.")
        return

    grid = json.loads((ROOT / "experiments/grid.json").read_text())
    done = {p.stem for p in (ROOT / "results").glob("*.json") if p.stem != "PENDING"}
    pending = load_pending()

    live = [ref for ref, _ in pending.items() if kernel_status(ref) in ("running", "queued")]
    slots = max(0, a.limit - len(live))
    print(f"{len(live)} kernel(s) live, {slots} slot(s) free")

    if a.cell:
        queue = [c for c in grid if c["id"] == a.cell]
    else:
        submitted_ids = {v["id"] for v in pending.values()}
        queue = [c for c in grid
                 if c["phase"] == a.phase and c["id"] not in done and c["id"] not in submitted_ids]

    if not queue:
        print("nothing to dispatch")
        return
    if slots == 0:
        print(f"{len(queue)} cell(s) waiting; no free slot")
        return

    source = bundle()
    for cell in queue[:slots]:
        slug = f"sb-{cell['id']}"
        script = HEADER.format(source=source, cell=cell, epochs=a.epochs,
                               data_kernel=DATA_KERNEL)
        if a.dry_run:
            print(f"  [dry-run] {slug}  {cell['model']} n={cell['n']} seed={cell['seed']}")
            continue
        ref = push_kernel(slug, script, kernels=[f"{username()}/{DATA_KERNEL}"],
                          gpu=True, internet=False)
        pending[ref] = cell
        print(f"  dispatched {ref}  {cell['model']} n={cell['n']} seed={cell['seed']}")

    if not a.dry_run:
        save_pending(pending)
        print(f"{len(queue) - min(slots, len(queue))} cell(s) still waiting")


if __name__ == "__main__":
    main()
