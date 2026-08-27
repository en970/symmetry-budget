"""Shared Kaggle plumbing.

The account name is read at run time and never written into the repository.
Kernel metadata is generated into a scratch directory for each push, so no file
tracked by git contains the account identity.
"""
from __future__ import annotations

import json, pathlib, subprocess, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_SLUG = "symmetry-budget-toptagging"


def username() -> str:
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "username:" in line:
            return line.split("username:")[1].strip()
    raise RuntimeError("Kaggle username not configured — run `kaggle config set -n username -v <name>`")


def push_kernel(slug: str, script: str, *, datasets: list[str] | None = None,
                kernels: list[str] | None = None,
                gpu: bool = True, internet: bool = False) -> str:
    """Write a kernel + its metadata to a temp dir and push it. Returns the ref."""
    user = username()
    ref = f"{user}/{slug}"
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / f"{slug}.py").write_text(script)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": ref,
            "title": slug.replace("-", " ")[:50],
            "code_file": f"{slug}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": gpu,
            "enable_internet": internet,
            "dataset_sources": datasets or [],
            "competition_sources": [],
            "kernel_sources": kernels or [],
        }, indent=2))
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"kernel push failed: {r.stderr.strip() or r.stdout.strip()}")
    return ref


def kernel_status(ref: str) -> str:
    r = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True)
    text = (r.stdout + r.stderr).lower()
    for state in ("complete", "error", "cancelacknowledged", "running", "queued"):
        if state in text:
            return state
    return "unknown"
