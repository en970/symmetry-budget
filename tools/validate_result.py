#!/usr/bin/env python3
"""Gate every result file against the pre-registered contract.

Wired to a PostToolUse hook, so a malformed or protocol-violating result is
rejected the moment it is written rather than at report time. An autonomous loop
that can silently write partial results is a loop that will eventually report
them as findings.

Exit 0 = accepted. Exit 2 = rejected, with the reason on stderr.
"""
import json, sys, pathlib

REQUIRED = {
    "id": str, "phase": int, "model": str, "n": int, "seed": int,
    "break": str, "budget": str,
    "rej_at_30": (int, float),      # primary metric
    "auc": (int, float),
    "accuracy": (int, float),
    "train_seconds": (int, float),
    "n_params": int,
    "flops_per_forward": int,
    "status": str,                  # ok | failed
}
MODELS = {"M0", "M1", "M2", "M3"}
BREAKS = {"none", "acceptance", "axis"}
BUDGETS = {"params", "flops"}


def validate(path):
    errs = []
    try:
        d = json.loads(pathlib.Path(path).read_text())
    except Exception as e:
        return [f"unreadable JSON: {e}"]

    if d.get("status") == "failed":
        # A failure is a legitimate, reportable outcome — but it must say why.
        if not d.get("failure_reason"):
            errs.append("status=failed requires a non-empty failure_reason")
        return errs

    for key, typ in REQUIRED.items():
        if key not in d:
            errs.append(f"missing field: {key}")
        elif not isinstance(d[key], typ):
            errs.append(f"{key}: expected {typ}, got {type(d[key]).__name__}")

    if d.get("model") not in MODELS:  errs.append(f"unknown model: {d.get('model')!r}")
    if d.get("break") not in BREAKS:  errs.append(f"unknown break: {d.get('break')!r}")
    if d.get("budget") not in BUDGETS: errs.append(f"unknown budget: {d.get('budget')!r}")

    for key in ("auc", "accuracy"):
        v = d.get(key)
        if isinstance(v, (int, float)) and not 0.0 <= v <= 1.0:
            errs.append(f"{key}={v} outside [0,1]")

    # A result that beats the published state of the art on this benchmark is far
    # more likely to be a leak than a discovery. Flag it instead of celebrating it.
    if isinstance(d.get("auc"), (int, float)) and d["auc"] > 0.995:
        errs.append(f"auc={d['auc']} implausibly high — check for train/test leakage")

    if d.get("id") and pathlib.Path(path).stem != d["id"]:
        errs.append(f"filename {pathlib.Path(path).stem} does not match id {d['id']}")

    return errs


def main():
    paths = sys.argv[1:] or [str(p) for p in pathlib.Path("results").glob("*.json")]
    bad = 0
    for p in paths:
        errs = validate(p)
        if errs:
            bad += 1
            print(f"REJECTED {p}", file=sys.stderr)
            for e in errs:
                print(f"   - {e}", file=sys.stderr)
    if bad:
        print(f"\n{bad} file(s) rejected", file=sys.stderr)
        sys.exit(2)
    print(f"{len(paths)} result file(s) accepted")


if __name__ == "__main__":
    main()
