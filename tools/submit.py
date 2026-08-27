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

# Verify the install actually took. A silent pip failure previously left the cell
# running on an incompatible build and failing much later with a CUDA error that
# said nothing about its real cause.
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    print(f"torch {{torch.__version__}} | device {{torch.cuda.get_device_name(0)}} "
          f"sm_{{cap[0]}}{{cap[1]}} | supports {{archs}}", flush=True)
    if f"sm_{{cap[0]}}{{cap[1]}}" not in archs:
        raise SystemExit(f"PyTorch {{torch.__version__}} does not support sm_{{cap[0]}}{{cap[1]}}; "
                         f"the compatibility install did not take")

CELLS = {cells!r}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {{device}} | {{len(CELLS)}} cell(s) in this kernel", flush=True)

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

# One kernel per group of cells: the compatibility install costs ~163 s and the
# smallest cells train in under a second, so per-cell kernels spend nearly all
# their time on setup. Each cell still writes its own result file, so a kernel
# that dies part-way keeps the cells that already finished.
for i, CELL in enumerate(CELLS, 1):
    print(f"--- cell {{i}}/{{len(CELLS)}}: {{CELL['model']}} n={{CELL['n']}} seed={{CELL['seed']}}",
          flush=True)
    try:
        result = run(CELL, data, epochs={epochs}, device=device)
    except Exception as exc:
        result = {{**{{k: CELL[k] for k in ("id","phase","model","n","seed","break","budget")}},
                  "status": "failed", "failure_reason": f"{{type(exc).__name__}}: {{exc}}"[:500]}}
        print("FAILED:", result["failure_reason"], flush=True)
    pathlib.Path(f"/kaggle/working/{{CELL['id']}}.json").write_text(json.dumps(result, indent=2))
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
        submitted_ids = {c["id"] for v in pending.values() for c in v.get("cells", [])}
        queue = [c for c in grid
                 if c["phase"] == a.phase and c["id"] not in done and c["id"] not in submitted_ids]

    if not queue:
        print("nothing to dispatch")
        return
    if slots == 0:
        print(f"{len(queue)} cell(s) waiting; no free slot")
        return

    # Group by (model, seed, break): one kernel covers all six training-set sizes
    # for that combination. The compatibility install costs ~163 s while the
    # smallest cells train in under a second, so per-cell kernels spend nearly all
    # their time on setup. Grouping by model alone would put M3's largest cells in
    # one kernel and risk the 12-hour limit.
    groups: dict[tuple, list[dict]] = {}
    for c in queue:
        groups.setdefault((c["model"], c["seed"], c["break"]), []).append(c)
    for g in groups.values():
        g.sort(key=lambda c: c["n"])      # cheap cells first, so a truncated kernel still delivers

    source = bundle()
    for (model, seed, brk), cells in list(groups.items())[:slots]:
        slug = f"sb-p{a.phase}-{model.lower()}-s{seed}-{brk}"
        script = HEADER.format(source=source, cells=cells, epochs=a.epochs,
                               data_kernel=DATA_KERNEL)
        label = f"{model} seed={seed} break={brk} ({len(cells)} cells)"
        if a.dry_run:
            print(f"  [dry-run] {slug}  {label}")
            continue
        # Internet must stay on: the cell installs a PyTorch build that still
        # supports the P100 Kaggle hands out. With it off, pip fails silently
        # ("from versions: none") and the cell runs on the incompatible build.
        ref = push_kernel(slug, script, kernels=[f"{username()}/{DATA_KERNEL}"],
                          gpu=True, internet=True)
        pending[ref] = {"cells": cells, "label": label}
        print(f"  dispatched {ref}  {label}")

    if not a.dry_run:
        save_pending(pending)
        print(f"{max(0, len(groups) - slots)} group(s) still waiting")


if __name__ == "__main__":
    main()
