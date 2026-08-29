# Pre-registration — Symmetry Budget

**Status:** frozen before any training run. Amendments are appended, never edited in place.
**Frozen on:** 2026-08-27

This document fixes the hypotheses, the grid, the metrics and the stopping rule *before*
results exist. It exists because the experiments are executed by an autonomous agent loop:
if the success criteria were written after seeing the numbers, the loop would be an
elaborate way of confirming a prior.

---

## 1. Question

Embedding a symmetry into a model architecture (equivariance) is usually presented as a
free lunch: the model cannot waste capacity learning what the physics already guarantees.
That argument is asymptotically about *small data*. It says little about the regime most
practitioners are in, and nothing about data whose symmetry is only approximate.

> **When does an equivariant architecture actually pay for itself, and when is it a tax?**

The comparison is deliberately three-way, because the interesting rival is not the naive
baseline but data augmentation:

| | symmetry knowledge enters via |
|---|---|
| **Architecture** | equivariant layers (hard constraint) |
| **Data** | augmentation with group elements (soft constraint) |
| **Neither** | unconstrained baseline |

A result that equivariance beats an unconstrained baseline is uninteresting and expected.
The claim under test is whether it beats *augmentation at equal compute*.

## 2. Hypotheses

Stated so they can fail.

- **H1 (vanishing gain).** The accuracy gap between the equivariant model and the augmented
  baseline decreases monotonically in training-set size N, and crosses zero at some finite
  N*. *Falsified if* the gap is flat or widening across the tested range.
- **H2 (brittleness under broken symmetry).** When the data's symmetry is broken (detector
  acceptance cuts, a preferred axis), the equivariant model loses more than the augmented
  baseline. *Falsified if* it loses less or equally.
- **H3 (compute is the confound).** At matched FLOP budget rather than matched parameter
  count, the equivariant advantage shrinks by more than half at every N. *Falsified if* the
  advantage is stable under the budget change.

H3 is the one most likely to embarrass the field, and therefore the one that must be run
with the least freedom in analysis.

## 3. Grid

**Task.** Top-quark jet tagging (binary). Public benchmark, Lorentz symmetry is the natural
group, and the discrimination is genuinely hard so ceiling effects do not mask differences.

**Models.** Four, matched to within 10% on parameter count in Phase 1.

| id | model | symmetry |
|---|---|---|
| `M0` | Deep Sets | permutation only |
| `M1` | Deep Sets + Lorentz augmentation | permutation, soft Lorentz |
| `M2` | ParticleNet-lite (dynamic kNN) | permutation, local geometry |
| `M3` | LorentzNet-style | permutation + Lorentz equivariant |

The M1 vs M3 contrast is the experiment. M0 and M2 are anchors.

**Axes.**

- `N` ∈ {1k, 3k, 10k, 30k, 100k, 300k} — log-spaced, six points
- `seed` ∈ {0, 1} — Phase 1; a third seed is added only where the M1/M3 gap is within noise
- `break` ∈ {none, acceptance, axis} — Phase 2 only
- `budget` ∈ {params, flops} — Phase 3 only

**Phases and size.** Phases are run in order and reported even if a later phase is never run.

| phase | cells | tests |
|---|---|---|
| 1 | 4 models × 6 N × 2 seeds = **48** | H1 |
| 2 | 2 models × 6 N × 2 breaks × 2 seeds = **48** | H2 |
| 3 | 2 models × 6 N × 2 seeds = **24** | H3 |

## 4. Metrics

Fixed now, computed identically in every cell.

- **Primary:** background rejection at 30% signal efficiency (1/ε_B @ ε_S = 0.3). HEP
  convention; more sensitive than accuracy in the tail that matters.
- **Secondary:** AUC, accuracy.
- **Cost:** wall-clock training seconds, parameter count, measured FLOPs/forward.
- **Derived — the "symmetry budget":** (relative gain in 1/ε_B over M1) ÷ (relative cost
  increase over M1). Reported with its confidence interval, never as a bare point estimate.

**What counts as a real difference.** A gap is called a gap only if the seed-to-seed spread
of the two models does not overlap at that N. Anything narrower is reported as "within
noise" and is not narrated as a trend. This rule is what stops an autonomous agent from
building a story out of jitter.

## 5. Stopping rule

- Phase 1 is analysed only when all 48 cells have finished or failed. No peeking at a
  partial grid to decide the next axis.
- Failed cells are reported with their failure mode, never silently dropped. A phase with
  more than 20% failures is reported as inconclusive, not as a result.
- The grid is not extended because a result "almost" reached significance.
- Any change to this protocol after the first run is appended below as an amendment with
  its date and reason.

## 6. Publication commitment

The report is written and published whatever the outcome, including the outcome that all
three hypotheses are falsified and equivariance is simply better everywhere. The repository
states results in the README table in the form "H1: supported / falsified / inconclusive"
before any prose discussion.

Negative and null results are the expected product here, not a disappointment. The prior
work in `structure-first-ml` returned "no, but complementary", "the null model was the
finding", and "lead-lag refuted" — this protocol is written to make that kind of honesty
the default rather than an act of will.

---

## Amendments

### A1 — 2026-08-28: how Phase 3 matches compute

**Recorded before any Phase 3 cell has run, and before Phase 1 has been analysed.**

§3 specifies `budget ∈ {params, flops}` but does not say what "matched FLOPs" means, and the
implementation did not match anything at all: `build()` returned identical models regardless
of budget, so Phase 3 would have been a verbatim repeat of Phase 1 and H3 would never have
been tested. That is a defect, not a change of intent, and it is fixed here.

Measured at P=100 constituents: M1 costs 10.1M FLOPs per forward pass, M3 costs 1.069B — a
factor of **106** at matched parameter count. So "equal parameters" hands the equivariant
model two orders of magnitude more compute, which is precisely what H3 suspects.

**Decision.** Phase 3 matches *forward-pass* FLOPs, and matches them by **enlarging M1**
rather than shrinking M3. Reaching M3's cost requires M1 at hidden=1396 — 7.8M parameters
against M3's 98k, an 80-fold difference in capacity.

Rejected alternatives, and why:

- *Shrinking M3 to M1's budget* would cripple the model under test and manufacture the
  conclusion H3 predicts. A hypothesis must not be handed its own evidence.
- *Matching total training compute instead* is defensible and arguably closer to the
  practitioner's question, but it confounds architecture with optimisation schedule: more
  steps is not the same intervention as more capacity.

**Known limitation, stated in advance.** A 7.8M-parameter Deep Sets model will almost
certainly overfit at n=1,000. That is not a flaw in the design — it is a real property of the
comparison, and it makes H3 falsifiable in an informative direction: if the enlarged baseline
fails at small N *because* it overfits, then equivariance is buying sample efficiency rather
than merely spending compute, and H3 is falsified for a reason worth reporting.

### A2 — 2026-08-28: the matched-FLOPs baseline was never actually matched

**Recorded while Phase 3 is open. No Phase 3 outcome metric was consulted in reaching this
decision** — only `flops_per_forward` and `n_params`, which are cost fields fixed by the
architecture before a single gradient step. Phase 3's `rej_at_30`, `auc` and `accuracy`
remain unread, as §5 requires.

A1 committed Phase 3 to matching *forward-pass FLOPs* and stated that reaching M3's cost
required M1 at `hidden=1396`. That number was asserted, not measured, and it is wrong.
Measured with `src.train.measure_flops` at P=100, the harness that produced every recorded
`flops_per_forward` in this repository:

| | FLOPs / forward | fraction of M3 |
|---|---|---|
| M3 (reference) | 1,069,142,976 | — |
| M1 at `hidden=1396` (A1) | 789,281,648 | **74%** |
| M1 at `hidden=1625` (A2) | 1,069,094,000 | 99.995% |

Both A1 figures reproduce exactly from the Phase 1 and Phase 3 result files, so this is a
defect in the constant, not in the measurement.

**Why this is not survivable as a stated limitation.** Phase 3 exists to hold cost equal and
ask what the equivariant advantage is worth once it is. Under-provisioning the baseline by 26%
in that one comparison inflates the matched-FLOPs gap and biases H3 toward *falsified* — the
verdict that flatters equivariance. That is the same failure A1 refused when it declined to
shrink M3: "A hypothesis must not be handed its own evidence." A1 refused it in one direction
and then committed it in the other.

**Decision.** `FLOPS_MATCHED` becomes `hidden=1625, latent=1625`. The twelve Phase 3 M1 cells
already run at `hidden=1396` (6 N × 2 seeds) are **superseded, not deleted**: they are moved to
`results/superseded/` with this amendment's date, and re-run at the corrected width. Per §5,
they are retained and reported rather than silently dropped.

Phase 3's M3 cells are unaffected — M3 is the fixed reference and does not appear in
`FLOPS_MATCHED`. Phases 1 and 2 are unaffected: the constant is read only when
`budget == "flops"`, which is Phase 3 only.

**Updated known limitation.** A1 anticipated a baseline of 7.8M parameters against M3's 98k,
an 80-fold capacity gap, and predicted it would overfit at n=1,000. At `hidden=1625` the
baseline is 10,585,252 parameters — a 108-fold gap. A1's stated limitation therefore applies
more strongly, not less, and in the same informative direction: if the enlarged baseline fails
at small N *because* it overfits, equivariance is buying sample efficiency rather than merely
spending compute, and H3 is falsified for a reason worth reporting.

**Rule for any future change to this constant.** It must be solved against
`src.train.measure_flops`, never estimated from a scaling argument. The comparison this number
defines is the one the whole phase rests on.

### A3 — 2026-08-29: the symmetry budget is implemented; two analysis defects are recorded, not fixed

**Written after Phases 1 and 2 closed and were audited.** That timing is why this amendment
distinguishes carefully between adding a quantity §4 already required and changing a rule that
decides a hypothesis. It does the first and refuses the second.

**1. The headline metric existed only on paper.** §4 names the "symmetry budget" — relative
gain in 1/ε_B over M1 divided by relative cost increase over M1, "reported with its confidence
interval, never as a bare point estimate" — and nothing in this repository computed it. It is
now in `tools/report.py` and appears for every closed phase.

**2. §4 does not say which cost is the denominator.** It names three (wall-clock training
seconds, parameter count, measured FLOPs/forward). They disagree by roughly a factor of three.
Picking one now, with the results visible, is exactly the freedom this document exists to
remove, so the report shows all three and privileges none. Under `budget=params` the
parameter-count denominator is *negative* — M3 is deliberately the smaller model — so the
ratio is undefined there; the report says so per measure rather than omitting the row, since a
cost measure that cannot be computed is itself a finding about §4's definition.

**3. The interval is a bootstrap, and it is a lower bound.** Two seeds cannot carry a
confidence interval. 1/ε_B is a count ratio, so each cell's value is B/k for an integer k of
surviving background jets; B = 20,100 is inferred from the metric itself rather than
hard-coded. The interval resamples k as Poisson and resamples across seeds. It omits every
source of variation not visible in two draws and must not be read as a full uncertainty.

**4. Two defects in the verdict code are recorded here and deliberately left in place.**
Both were found by the adversarial audits of Phases 1 and 2. Changing analysis code that
decides a hypothesis, after seeing what it decided, is the researcher degree of freedom this
protocol was written to prevent — so the frozen rule stands and its consequence is stated
instead:

- `audit_phase1()`'s monotonicity test is non-strict (`b <= a`). §2 names a *flat* gap as
  falsifying, but a perfectly flat series would return "partially supported". Phase 1's gap is
  not flat, so this did not affect the recorded H1 verdict.
- `audit_phase2()` divides by the mean of the unbroken baseline, discarding that baseline's own
  two-seed spread instead of propagating it. Propagating it reduces the comparisons that clear
  §4's noise rule from 6 of 12 to 4 of 12, all of them from the `axis` break alone; `acceptance`
  clears none. The frozen rule reports H2 as **supported**; the same data under propagated
  uncertainty supports **inconclusive**, which is what the Phase 2 adversarial audit returned.
  Both are on the record and the report states them together.

Additionally, `audit_phase2()`'s 0.75 / 0.25 decision thresholds appear nowhere in §2–§5. They
were written into `tools/audit.py` before any Phase 2 data existed, so they are not reverse-fit,
but they were never appended here as §5 requires for a rule that decides a verdict. They are
appended now, as a record of what the code does — not as a ratification of it.
