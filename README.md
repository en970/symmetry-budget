# symmetry-budget

**When does an equivariant architecture pay for itself, and when is it a tax?**

Embedding a symmetry into a network is usually argued for asymptotically: the model cannot
waste capacity learning what physics already guarantees. That argument is about the
small-data limit. It says little about the regime most practitioners work in, and nothing
about measurements whose symmetry is only approximate — which is every real detector.

This repository measures the trade-off instead of arguing it. The comparison that matters is
not equivariance against an unconstrained baseline (it wins, uninterestingly) but
**equivariance against data augmentation at equal compute**.

| | how symmetry knowledge enters |
|---|---|
| Architecture | equivariant layers — hard constraint |
| Data | augmentation with group elements — soft constraint |
| Neither | unconstrained baseline |

## Results

Hypotheses are pre-registered in [`PROTOCOL.md`](PROTOCOL.md), frozen before the first
training run. Verdicts appear here as phases complete.

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | The equivariant-vs-augmented gap vanishes with training-set size | *not yet run* |
| H2 | Equivariant models are more brittle when the symmetry is broken | *not yet run* |
| H3 | The advantage shrinks at matched FLOPs rather than matched parameters | *not yet run* |

The report is published whatever the outcome, including the outcome that every hypothesis is
falsified.

## Design

Task: top-quark jet tagging. Four models — Deep Sets, Deep Sets + Lorentz augmentation,
ParticleNet-lite, LorentzNet-style — across six training-set sizes, two seeds, three phases.
120 cells total. Primary metric is background rejection at 30% signal efficiency; cost is
measured, not assumed.

Full grid, metrics and stopping rule: [`PROTOCOL.md`](PROTOCOL.md).

## How this is run

Every cell of the grid is independent, which makes the experiment a natural fit for an
autonomous loop: cells are dispatched to Kaggle GPUs, results are collected as JSON, and the
report regenerates from those files. Nothing in the analysis path is hand-edited — see
[`docs/AUTONOMY.md`](docs/AUTONOMY.md) for the machinery and its guard rails.

```bash
make grid       # expand PROTOCOL into experiments/grid.yaml
make submit     # dispatch pending cells to Kaggle
make collect    # pull finished runs into results/
make report     # regenerate reports/ from results/
```

## Related

- [`structure-first-ml`](https://github.com/en970/structure-first-ml) — the methodological
  parent: identify the mathematical object first, attach a learner second.
- [`particlenet-jet-tagging`](https://github.com/en970/particlenet-jet-tagging) — earlier
  honest comparison of GNN vs Deep Sets on this task.

## Licence

MIT.
