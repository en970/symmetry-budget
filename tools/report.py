"""Regenerate reports/ from results/. Numbers are never hand-edited.

Implements PROTOCOL §4's noise rule directly: a difference between two models at
a given N counts only if their seed-to-seed ranges do not overlap. Everything
narrower is printed as "within noise" and is not allowed to become a trend. The
rule lives in code because prose written by a model that has just seen an
encouraging number will find a way to call it suggestive.
"""
from __future__ import annotations

import json, pathlib, random, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS, REPORTS = ROOT / "results", ROOT / "reports"


def load() -> list[dict]:
    out = []
    for p in RESULTS.glob("*.json"):
        if p.stem == "PENDING":
            continue
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def phase_state(rows: list[dict], grid: list[dict], phase: int) -> tuple[str, int, int, int]:
    ids = {c["id"] for c in grid if c["phase"] == phase}
    got = {r["id"] for r in rows if r["id"] in ids}
    failed = {r["id"] for r in rows if r["id"] in ids and r.get("status") == "failed"}
    if len(got) < len(ids):
        return "open", len(got), len(failed), len(ids)
    if len(failed) / max(len(ids), 1) > 0.20:
        return "inconclusive", len(got), len(failed), len(ids)
    return "closed", len(got), len(failed), len(ids)


def by_cell(rows, phase, model, n, key="rej_at_30"):
    vals = [r[key] for r in rows
            if r.get("status") == "ok" and r["phase"] == phase
            and r["model"] == model and r["n"] == n
            and isinstance(r.get(key), (int, float))]
    return vals


def compare(rows, phase, a, b, n) -> tuple[str, str]:
    """Return (verdict, detail) for model `a` vs `b` at size `n` under §4."""
    va, vb = by_cell(rows, phase, a, n), by_cell(rows, phase, b, n)
    if len(va) < 2 or len(vb) < 2:
        return "insufficient", f"{a}:{len(va)} {b}:{len(vb)} seeds"
    lo_a, hi_a = min(va), max(va)
    lo_b, hi_b = min(vb), max(vb)
    ma, mb = statistics.mean(va), statistics.mean(vb)
    if hi_a < lo_b:
        return f"{b} > {a}", f"{ma:.1f} vs {mb:.1f} (ranges disjoint)"
    if hi_b < lo_a:
        return f"{a} > {b}", f"{ma:.1f} vs {mb:.1f} (ranges disjoint)"
    return "within noise", f"{ma:.1f} vs {mb:.1f} (ranges overlap — not a gap)"


def infer_background_count(rows) -> int | None:
    """Recover the test set's background count B from the metric itself.

    rej_at_30 is 1/ε_B, and ε_B is a count ratio, so every value is B/k for an
    integer k of surviving background jets. Recovering B lets the interval below
    use the metric's own counting noise instead of pretending two seeds are a
    sample. Inferred rather than hard-coded so it cannot drift away from whatever
    kaggle/prepare_data.py actually produced.
    """
    vals = [r["rej_at_30"] for r in rows
            if r.get("status") == "ok" and isinstance(r.get("rej_at_30"), (int, float))
            and r["rej_at_30"] > 0]
    if not vals:
        return None
    for B in range(1000, 100_001):
        if all(abs(B / v - round(B / v)) < 1e-6 and round(B / v) >= 1 for v in vals):
            return B
    return None


def symmetry_budget(rows, phase, n, B, brk=None, trials=20000):
    """PROTOCOL §4's derived quantity, with an interval rather than a point.

    (relative gain in 1/ε_B over M1) / (relative cost increase over M1).

    §4 names three costs — wall-clock training seconds, parameter count and
    measured FLOPs/forward — and never says which one is the denominator. They do
    not agree: the three differ by roughly a factor of three and can disagree in
    sign about whether equivariance is worth its price. Choosing one after seeing
    the results is exactly the freedom this protocol exists to remove, so all
    three are reported and none is privileged. See amendment A3.

    The interval is a bootstrap over the metric's counting noise (surviving
    background jets, Poisson) crossed with a resample over seeds. Two seeds
    cannot carry a confidence interval by themselves; the counting term is what
    makes one honest, and it is still a lower bound on the true uncertainty
    because it omits everything that varies with initialisation beyond the two
    draws observed.
    """
    def cells(model):
        return [r for r in rows if r.get("status") == "ok" and r["phase"] == phase
                and r["model"] == model and r["n"] == n
                and (brk is None or r.get("break") == brk)]

    m1, m3 = cells("M1"), cells("M3")
    if len(m1) < 2 or len(m3) < 2 or not B:
        return None

    costs = {"train_seconds": "wall-clock", "flops_per_forward": "FLOPs/forward",
             "n_params": "parameters"}
    rng = random.Random(0)
    out, skipped = {}, {}
    for key, label in costs.items():
        if any(not isinstance(r.get(key), (int, float)) or r[key] <= 0 for r in m1 + m3):
            skipped[label] = "cost not recorded"
            continue
        c1 = statistics.mean(r[key] for r in m1)
        c3 = statistics.mean(r[key] for r in m3)
        rel_cost = c3 / c1 - 1.0
        if rel_cost <= 0:
            # Not a rounding case: under budget=params M3 is deliberately the
            # *smaller* model, so this denominator is negative and the ratio would
            # report a cheaper-and-better model as a negative budget. Recorded
            # rather than dropped — a measure that cannot be computed is itself a
            # finding about §4's definition.
            skipped[label] = f"denominator {rel_cost:+.1%}, not a cost increase"
            continue
        point = (statistics.mean(r["rej_at_30"] for r in m3)
                 / statistics.mean(r["rej_at_30"] for r in m1) - 1.0) / rel_cost
        draws = []
        for _ in range(trials):
            def resample(cells):
                tot = 0.0
                for _ in range(len(cells)):
                    k_obs = max(round(B / rng.choice(cells)["rej_at_30"]), 1)
                    # Poisson(k) via its normal limit is wrong in this tail: at the
                    # large-N end k drops to ~30 surviving jets, so it is sampled
                    # exactly.
                    k = poisson(rng, k_obs)
                    tot += B / max(k, 1)
                return tot / len(cells)
            draws.append((resample(m3) / resample(m1) - 1.0) / rel_cost)
        draws.sort()
        lo = draws[int(0.025 * len(draws))]
        hi = draws[int(0.975 * len(draws)) - 1]
        out[label] = (point, lo, hi, rel_cost)
    return {"budgets": out, "skipped": skipped} if out or skipped else None


def poisson(rng, lam: float) -> int:
    """Knuth's method. lam here is a jet count in the tens to low thousands."""
    if lam > 500:                      # normal approximation is safe this far out
        return max(0, round(rng.gauss(lam, lam ** 0.5)))
    L, k, p = 2.718281828459045 ** -lam, 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def main() -> None:
    grid = json.loads((ROOT / "experiments/grid.json").read_text())
    rows = load()
    B = infer_background_count(rows)
    REPORTS.mkdir(exist_ok=True)
    lines = ["# Results", "",
             "Generated by `tools/report.py`. Do not edit — regenerate instead.", ""]

    for phase in sorted({c["phase"] for c in grid}):
        state, got, failed, total = phase_state(rows, grid, phase)
        lines += [f"## Phase {phase}", "",
                  f"- cells: {got}/{total} returned, {failed} failed",
                  f"- state: **{state}**", ""]

        if state == "open":
            lines += ["Metrics withheld until the phase closes (PROTOCOL §5).", ""]
            continue
        if state == "inconclusive":
            lines += ["More than 20% of cells failed. Reported as inconclusive; the "
                      "surviving cells are not analysed (PROTOCOL §5).", ""]
            continue

        if phase == 1:
            lines += ["Equivariant (M3) vs augmented baseline (M1), by training-set size.",
                      "", "| N | verdict | detail |", "|---|---|---|"]
            for n in sorted({c["n"] for c in grid if c["phase"] == phase}):
                v, d = compare(rows, phase, "M1", "M3", n)
                lines.append(f"| {n:,} | {v} | {d} |")
            lines.append("")

        # §4's derived quantity. Emitted for every closed phase that has both
        # models, because "is equivariance worth its cost" is the question the
        # protocol actually asks, and a rejection table alone does not answer it.
        budget_lines, skipped_notes = [], {}
        # Phase 2 runs two different breaks at the same (model, N); pooling them
        # would average two distinct interventions into one ratio, so each break
        # gets its own rows.
        breaks = sorted({c["break"] for c in grid if c["phase"] == phase})
        for brk in breaks:
            for n in sorted({c["n"] for c in grid if c["phase"] == phase}):
                sb = symmetry_budget(rows, phase, n, B, brk=brk)
                if not sb:
                    continue
                skipped_notes.update(sb["skipped"])
                tag = f" | {brk}" if len(breaks) > 1 else ""
                for label, (point, lo, hi, rel_cost) in sb["budgets"].items():
                    budget_lines.append(
                        f"| {n:,}{tag} | {label} | {rel_cost:+.0%} | {point:.3f} "
                        f"| [{lo:.3f}, {hi:.3f}] |")
        if budget_lines:
            lines += [
                "**Symmetry budget** (PROTOCOL §4): relative gain in 1/ε_B over M1, divided "
                "by relative cost increase over M1. A value of 1.0 means the gain and the "
                "extra cost are proportional; below 1.0 the equivariant model buys less "
                "than it spends.", "",
                "§4 names three costs and does not say which is the denominator, so all "
                "three are shown. Intervals are 95% bootstrap over the metric's background "
                "counting noise crossed with a seed resample — a lower bound on the true "
                "uncertainty, not a full one (amendment A3).", "",
                ("| N | break | cost measure | rel. cost | budget | 95% interval |"
                 if len(breaks) > 1 else
                 "| N | cost measure | rel. cost | budget | 95% interval |"),
                ("|---|---|---|---|---|---|" if len(breaks) > 1 else "|---|---|---|---|---|"),
                *budget_lines, ""]
            if skipped_notes:
                lines += ["Cost measures §4 names that could not be computed here: "
                          + "; ".join(f"**{k}** ({v})" for k, v in sorted(skipped_notes.items()))
                          + ".", ""]

    (REPORTS / "results.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORTS / 'results.md'}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
